//! The field of values of Ã and of the operator GCR iterates with, Ã M (M the multigrid cycle,
//! right preconditioning): [λ_min, λ_max] of the Hermitian parts (Ã + Ã†)/2 and (Ã M + M† Ã†)/2
//! by Lanczos. λ_min < 0 means that the field of values reaches into the left half plane, where a
//! restarted minimal-residual method can stagnate.
//!
//! For (Ã M)† = M† Ã† M must be a fixed linear map and M† its adjoint: both coarse factors are
//! solved on their exact adjoints (BiCGStab) to |r|²/|b|² < `--coarse` (1e-20), and M† is the cycle
//! with D and D_PV exchanged on the same P and with the same top of the smoother's interval. The
//! program first checks ⟨z, Ã M w⟩ = ⟨M† Ã† z, w⟩. On the device: `preconditioned_fov_device`.
//!
//! ```text
//! cargo run --release --example preconditioned_fov -- L FORMAT FILE M5 MASS SMEAR [--block B]
//!     [--vectors N] [--steps S] [--ratio R] [--coarse T] [--tol T] [--t T]
//! ```
//!
//! Ls 8, b 1.5, c 0.5; SMEAR 1 for stout 0.1 × 3. Setup as the solver's: 16 vectors filtered in
//! 2 rounds, the bootstrap 2 × 4. Defaults: blocks of 4⁴, 16 test vectors, Chebyshev 8 steps on
//! [hi/30, hi], Lanczos to 1e-4.

use ruttice::eigen::{lowest, LanczosParams};
use ruttice::fermion::{EoForm, FermionOp, Mdwf, MdwfParams, PvEvenOdd};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseKrylov, LowModeParams,
    PvEoCycle, PvEoParams,
};
use ruttice::{
    extract_parity, random_spinor, Boundary, Field, Half, Lattice, LinearFn, Parity, Spinor,
    Vector, C64, ND,
};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

type H = Field<f64, Spinor, Half>;

/// s (B + B†)/2 for B = Ã M (or Ã alone without the cycles), B† = M† Ã†.
struct HermitianPart<'a, 'c> {
    a: PvEvenOdd<'a, f64>,
    a_dag: PvEvenOdd<'a, f64>,
    cycles: Option<(&'c mut PvEoCycle<'a>, &'c mut PvEoCycle<'a>)>,
    sign: f64,
    applications: usize,
}

impl HermitianPart<'_, '_> {
    /// B x and B† x.
    fn both(&mut self, x: &H) -> (H, H) {
        self.applications += 1;
        let (mut bx, mut bdx) = (x.zeros_like(), x.zeros_like());
        match self.cycles.as_mut() {
            Some((m, m_dag)) => {
                let mut t = x.zeros_like();
                m.mult(&mut t, x);
                self.a.mult(&mut bx, &t);
                let mut s = x.zeros_like();
                self.a_dag.mult(&mut s, x);
                m_dag.mult(&mut bdx, &s);
            }
            None => {
                self.a.mult(&mut bx, x);
                self.a_dag.mult(&mut bdx, x);
            }
        }
        (bx, bdx)
    }
}

impl LinearFn<H> for HermitianPart<'_, '_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        let (bx, bdx) = self.both(inp);
        out.copy_from(&bx);
        out.axpy(&[C64::new(1.0, 0.0)], &bdx);
        out.scale(&[0.5 * self.sign]);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        return Err("L FORMAT FILE M5 MASS SMEAR [--block B] [--vectors N] [--steps S] [--ratio R] [--coarse T] [--tol T]".into());
    }
    let value = |name: &str, default: f64| -> Result<f64, Box<dyn Error>> {
        match args.iter().position(|a| a == name) {
            Some(i) => Ok(args.get(i + 1).ok_or("a value")?.parse()?),
            None => Ok(default),
        }
    };
    let l: usize = args[0].parse()?;
    let (m5, mass): (f64, f64) = (args[3].parse()?, args[4].parse()?);
    let smear = args[5] == "1";
    let block = value("--block", 4.0)? as usize;
    let vectors = value("--vectors", 16.0)? as usize;
    let coarse = value("--coarse", 1e-20)?;
    let tol = value("--tol", 1e-4)?;
    let start = Instant::now();
    // `--t T`: an L³×T lattice.
    let nt = value("--t", l as f64)? as usize;
    let lat = Lattice::single_node([l, l, l, nt]);
    let mut u = ruttice::app::read_gauge_named(&args[1], Path::new(&args[2]), &lat)?;
    if smear {
        let s = GaugeSmearing {
            smear: Smear::Stout(Stout { rho: uniform(0.1) }),
            steps: 3,
        };
        u = s.levels(&u).pop().expect("three steps");
    }
    let params = |m: f64| MdwfParams {
        ls: 8,
        m5,
        mass: m,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut op = Mdwf::<f64>::new(&lat, params(mass));
    op.set_gauge(&u)?;
    let mut pv = Mdwf::<f64>::new(&lat, params(1.0));
    pv.set_gauge(&u)?;
    let cycle_params = PvEoParams {
        chebyshev_steps: value("--steps", 8.0)? as usize,
        chebyshev_ratio: value("--ratio", 30.0)?,
        ..PvEoParams::default()
    };
    let (y, _) = pv_eo_low_modes(&op, &pv, &lat, LowModeParams::default());
    let mut y = y.select(&(0..vectors).collect::<Vec<_>>());
    let agg = Aggregation::new(&lat, [block; ND]);
    let basis = pv_eo_bootstrap(
        &op,
        &pv,
        &agg,
        &mut y,
        BootstrapParams {
            cycle: cycle_params,
            ..BootstrapParams::default()
        },
    )?;
    println!(
        "# {l}^3x{nt} {}; M5 {m5}, m {mass}, stout 0.1x3: {smear}; blocks {block}^4, {vectors} vectors, Chebyshev {} on [hi/{}, hi]; coarse solves exact adjoints to {coarse:e}; setup {:.0} s",
        args[2],
        cycle_params.chebyshev_steps,
        cycle_params.chebyshev_ratio,
        start.elapsed().as_secs_f64()
    );
    let exact = PvEoParams {
        pv_solver: CoarseKrylov::BiCgStab,
        coarse_tolerance_sqr: coarse,
        pv_tolerance_sqr: coarse,
        coarse_max_iterations: 20_000,
        pv_max_iterations: 20_000,
        ..cycle_params
    };
    let mut m = PvEoCycle::new(&op, &pv, (&op, &pv, EoForm::Right), &basis, exact)?;
    // M† with the polynomial of M: the same top of the smoother's interval, not that of a
    // power iteration of its own (which made ⟨z, Ã M w⟩ and ⟨M† Ã† z, w⟩ differ by 3e-6).
    let parts = PvEoCycle::new(&pv, &op, (&pv, &op, EoForm::Right), &basis, exact)?
        .into_parts()
        .with_top(m.top());
    let mut m_dag =
        PvEoCycle::from_parts(&pv, &op, (&pv, &op, EoForm::Right), &basis, exact, parts);
    let like = y.select(&[0]);
    let index = op.even_odd_index().expect("even-odd").clone();
    let random_even = |seed: u64| {
        let mut v = like.zeros_like();
        extract_parity(&mut v, &random_spinor(&lat, 8, seed), &index, Parity::Even);
        v
    };
    // ⟨z, Ã M w⟩ = ⟨M† Ã† z, w⟩.
    {
        let (z, w) = (random_even(61), random_even(62));
        let mut h = HermitianPart {
            a: PvEvenOdd::new(&op, &pv, EoForm::Right, &like),
            a_dag: PvEvenOdd::new(&pv, &op, EoForm::Right, &like),
            cycles: Some((&mut m, &mut m_dag)),
            sign: 1.0,
            applications: 0,
        };
        let (bw, _) = h.both(&w);
        let (_, bdz) = h.both(&z);
        let (mut l, mut r) = ([C64::ZERO], [C64::ZERO]);
        z.dot(&bw, &mut l);
        bdz.dot(&w, &mut r);
        let rel = ((l[0] - r[0]).norm_sqr() / l[0].norm_sqr()).sqrt();
        println!(
            "# adjoint check: <z, A M w> = {:.10e} {:+.10e}i, <M^dag A^dag z, w> = {:.10e} {:+.10e}i, relative difference {rel:.1e}",
            l[0].re, l[0].im, r[0].re, r[0].im
        );
        if rel > 1e-6 {
            println!("# WARNING: M^dag is not the adjoint of M to 1e-6; the numbers below are not those of a Hermitian part");
        }
    }
    let lanczos = LanczosParams {
        wanted: 1,
        keep: 8,
        max_dim: 40,
        tolerance: tol,
        max_restarts: 2000,
        chebyshev: None,
    };
    let x0 = random_even(63);
    for preconditioned in [false, true] {
        let mut ends = Vec::new();
        let t = Instant::now();
        for sign in [1.0, -1.0] {
            let mut h = HermitianPart {
                a: PvEvenOdd::new(&op, &pv, EoForm::Right, &like),
                a_dag: PvEvenOdd::new(&pv, &op, EoForm::Right, &like),
                cycles: preconditioned.then_some((&mut m, &mut m_dag)),
                sign,
                applications: 0,
            };
            let e = lowest(&mut h, &x0, lanczos);
            ends.push((sign * e.values[0], e.converged, e.restarts));
        }
        println!(
            "{}: Re of the field of values in [{:+.6e}, {:+.6e}]  (converged {} {}, {} + {} restarts, {:.0} s){}",
            if preconditioned { "A~ M " } else { "A~   " },
            ends[0].0,
            ends[1].0,
            ends[0].1,
            ends[1].1,
            ends[0].2,
            ends[1].2,
            t.elapsed().as_secs_f64(),
            if ends[0].0 < 0.0 { "  lambda_min < 0" } else { "  in the right half plane" }
        );
    }
    Ok(())
}
