//! The even-odd Pauli-Villars multigrid of `src/` (`pv_eo`) on a configuration, with test
//! vectors from a `mdwf_evenodd_coherence` checkpoint.
//!
//! ```text
//! cargo run --release --example mdwf_pv_eo_mg -- CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [left] [alpha:A] [pvalpha] [gcr:K] [boot:R] [coarse64] [cgne] [exact] [bicgstab] [seeds:A-B]
//! ```
//!
//! N test vectors (N/2 Ritz vectors y of Ã, default 32) on BLOCK⁴ aggregates (default 4),
//! improved by R rounds of the multigrid bootstrap first with `boot:R`.
//! `left` solves with Ã_L instead of Ã, `alpha:A` puts Neff's α into D (`pvalpha`: into
//! D_PV as well). `cgne` solves the normal equations of the left Schur complement with the
//! multigrid instead (CGNE with α), `exact` takes the second coarse factor on its exact
//! adjoint, `bicgstab` uses flexible BiCGStab outside instead of GCR. Printed per right-hand side (seeds 91..91 by default): GCR(K) steps
//! (default K = 2), applications of the operator (outer and inside the cycle), coarse
//! applications, |D x − b|²/|b|² with the plain D, time, and the memory in use.

use ruttice::fermion::{EoForm, FermionOp, Mdwf, MdwfParams, PvEvenOdd, Variant};
use ruttice::io::read_bridge_text;
use ruttice::kernel::Dirac;
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_basis, pv_eo_bootstrap, Aggregation, BootstrapParams, PvEoCycle, PvEoParams,
};
use ruttice::solver::{BiCgStab, Gcr, SolveError, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Field, Half, Lattice, LinearFn, Parity, Spinor, Vector, C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_pv_eo_mg CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [left] [alpha:A] [pvalpha] [gcr:K] [boot:R] [coarse64] [cgne] [exact] [bicgstab] [seeds:A-B]";

type H = Field<f64, Spinor, Half>;

/// The resident memory and its peak so far (Linux: VmRSS and VmHWM).
fn memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
            .map_or(f64::NAN, |kb| kb / 1048576.0)
    };
    format!(
        "memory {:.2} GB, peak {:.2} GB",
        field("VmRSS:"),
        field("VmHWM:")
    )
}

/// The first `count` vectors of a `mdwf_evenodd_coherence` checkpoint, read site by site.
fn load(path: &PathBuf, like: &H, count: usize) -> Result<H, Box<dyn Error>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<u64> {
        f.read_exact(&mut word)?;
        Ok(u64::from_le_bytes(word))
    };
    let _ = next(&mut f)?;
    let m = next(&mut f)? as usize;
    let len = next(&mut f)? as usize;
    let c = count.min(m);
    let mut x = like.zeros_with(c);
    let sites = x.num_sites();
    let reals = like.site_len() / like.num_vectors();
    if sites * m * reals != len {
        return Err("checkpoint for another shape".into());
    }
    for _ in 0..m {
        let _ = next(&mut f)?;
    }
    let mut buf = vec![0u8; 8 * m * reals];
    for site in 0..sites {
        f.read_exact(&mut buf)?;
        for (k, v) in x.site_mut(0, site).iter_mut().enumerate() {
            *v = f64::from_le_bytes(buf[8 * k..8 * k + 8].try_into().expect("8 bytes"));
        }
    }
    Ok(x)
}

/// The cycle with single- or double-precision coarse solves.
enum Cycle<'a> {
    Single(PvEoCycle<'a, Dirac, f32>),
    Double(PvEoCycle<'a, Dirac, f64>),
}

impl Cycle<'_> {
    fn top(&self) -> f64 {
        match self {
            Cycle::Single(c) => c.top(),
            Cycle::Double(c) => c.top(),
        }
    }

    fn coarse_applications(&self) -> usize {
        match self {
            Cycle::Single(c) => c.coarse_applications(),
            Cycle::Double(c) => c.coarse_applications(),
        }
    }
}

impl LinearFn<H> for Cycle<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        match self {
            Cycle::Single(c) => c.mult(out, inp),
            Cycle::Double(c) => c.mult(out, inp),
        }
    }
}

/// Counts the applications of the operator it wraps.
struct Counted<'a, A> {
    a: &'a mut A,
    n: &'a Cell<usize>,
}

impl<A: LinearFn<H>> LinearFn<H> for Counted<'_, A> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.mult(out, inp);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err(USAGE.into());
    }
    let is_flag = |a: &str| {
        ["left", "pvalpha", "coarse64", "cgne", "exact", "bicgstab"].contains(&a)
            || (a.contains(':') && !a.contains('/'))
    };
    let positional: Vec<&String> = args.iter().filter(|a| !is_flag(a)).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |prefix: &str| args.iter().find_map(|a| a.strip_prefix(prefix));
    let path = PathBuf::from(positional[0]);
    let l: usize = positional[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = positional[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = positional[3].parse().map_err(|_| USAGE)?;
    let checkpoint = PathBuf::from(positional[4]);
    let number = |i: usize, d: usize| -> Result<usize, Box<dyn Error>> {
        Ok(positional
            .get(i)
            .filter(|s| s.parse::<usize>().is_ok())
            .map_or(Ok(d), |s| s.parse())?)
    };
    let (ls, n, block) = (number(5, 8)?, number(6, 32)?, number(7, 4)?);
    let left = flag("left");
    let alpha: f64 = value("alpha:").map_or(Ok(1.0), str::parse)?;
    let gcr_k: usize = value("gcr:").map_or(Ok(2), str::parse)?;
    let (seed_lo, seed_hi): (u64, u64) = value("seeds:")
        .and_then(|s| s.split_once('-'))
        .map_or(Ok((91, 91)), |(a, b)| -> Result<_, Box<dyn Error>> {
            Ok((a.parse()?, b.parse()?))
        })?;

    let lat = Lattice::single_node([l; ND]);
    let u = read_bridge_text(&path, &lat)?;
    let params = MdwfParams {
        ls,
        m5,
        mass,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let pv_params = MdwfParams {
        mass: 1.0,
        ..params
    };
    let make = |p: MdwfParams, a: f64| -> Result<Mdwf<f64>, Box<dyn Error>> {
        let mut op = Mdwf::<f64>::new(&lat, p);
        op.set_gauge(&u)?;
        op.set_alpha(a);
        Ok(op)
    };
    let (op, pv) = (make(params, 1.0)?, make(pv_params, 1.0)?);
    let op_a = make(params, alpha)?;
    let pv_a = make(pv_params, if flag("pvalpha") { alpha } else { 1.0 })?;
    // `cgne`: the normal equations of the left Schur complement, N = L^(α)† L^(α), which is
    // the operator with D in place of D_PV; the second coarse factor is then P†D†P.
    let cgne = flag("cgne");
    let (pv, pv_a) = if cgne { (&op, &op_a) } else { (&pv, &pv_a) };
    let cycle_params = PvEoParams {
        exact_adjoint: cgne || flag("exact"),
        ..PvEoParams::default()
    };
    let form = if left || cgne {
        EoForm::Left
    } else {
        EoForm::Right
    };
    let mut like = H::with_shape(&lat, 24, ls, 1, 1);
    like.set_parity(Parity::Even);

    let mut y = load(&checkpoint, &like, n / 2)?;
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; {form:?}, α = {alpha} ({}); N = {n} on {block}^4; {}",
        if flag("pvalpha") { "D and D_PV" } else { "D only" },
        memory()
    );
    let start = Instant::now();
    let agg = Aggregation::new(&lat, [block; ND]);
    // `boot:R`: R rounds of the multigrid bootstrap on the vectors first.
    let rounds: usize = value("boot:").map_or(Ok(0), str::parse)?;
    let basis = if rounds > 0 {
        pv_eo_bootstrap(
            &op,
            pv,
            &agg,
            &mut y,
            BootstrapParams {
                rounds,
                ..BootstrapParams::default()
            },
        )?
    } else {
        pv_eo_basis(&op, pv, &agg, &y)?
    };
    drop(y);
    // `coarse64`: the coarse solves in double precision (single by default).
    let outer_ops = (&op_a, pv_a, form);
    let mut cycle = if flag("coarse64") {
        Cycle::Double(PvEoCycle::with_coarse_storage(
            &op,
            pv,
            outer_ops,
            &basis,
            cycle_params,
        )?)
    } else {
        Cycle::Single(PvEoCycle::new(&op, pv, outer_ops, &basis, cycle_params)?)
    };
    println!(
        "  basis and cycle: {:.0} s, |λ_max| ≈ {:.3}; {}",
        start.elapsed().as_secs_f64(),
        cycle.top() / 1.1,
        memory()
    );

    let mut a = PvEvenOdd::new(&op_a, pv_a, form, &like);
    let steps = PvEoParams::default().chebyshev_steps;
    for seed in seed_lo..=seed_hi {
        let mut b = Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
        b.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, seed).as_slice());
        let mut rhs = like.zeros_like();
        a.right_hand_side(&mut rhs, &b);
        let (outer, coarse_before) = (Cell::new(0), cycle.coarse_applications());
        let mut sol = like.zeros_like();
        let start = Instant::now();
        let cycles = Cell::new(0);
        let stop = Stop::simple(1e-18, 2000);
        let mut counted_a = Counted {
            a: &mut a,
            n: &outer,
        };
        let mut counted_cycle = Counted {
            a: &mut cycle,
            n: &cycles,
        };
        // `bicgstab`: flexible BiCGStab outside (two cycles per iteration) instead of GCR.
        let (result, solver) = if flag("bicgstab") {
            (
                BiCgStab::new(stop).solve(&mut counted_a, Some(&mut counted_cycle), &mut sol, &rhs),
                "BiCGStab".to_string(),
            )
        } else {
            (
                Gcr::new(stop, gcr_k).solve(
                    &mut counted_a,
                    Some(&mut counted_cycle),
                    &mut sol,
                    &rhs,
                ),
                format!("GCR({gcr_k})"),
            )
        };
        let iterations = match &result {
            Ok(i) => format!("{}", i.iterations),
            Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
            Err(e) => format!("failed: {e}"),
        };
        // Inside a cycle: two smoother applications of steps − 1 each, two residuals.
        let inner = cycles.get() * (2 * (steps - 1) + 2);
        let mut x = b.zeros_like();
        a.solution(&mut x, &sol, &b);
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut rr, mut bb) = ([0.0], [0.0]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        println!(
            "  seed {seed}: {solver} {iterations} iterations, {} cycles: {} Ã = {} sweeps, {} coarse, |Dx−b|² {:.0e}, {:.0} s; {}",
            cycles.get(),
            outer.get() + inner,
            2 * (outer.get() + inner),
            cycle.coarse_applications() - coarse_before,
            rr[0] / bb[0],
            start.elapsed().as_secs_f64(),
            memory()
        );
    }
    Ok(())
}
