//! Möbius domain-wall solves on the even sites, with or without the even-odd
//! Pauli-Villars multigrid of `src/` (`pv_eo`), from a parameter file in the style of
//! Bridge++ (`ruttice::app::params`, YAML or TOML).
//!
//! ```text
//! cargo run --release --example mdwf_pv_eo_mg -- examples/mdwf_pv_eo_mg.yaml [seeds:A-B]
//! ```
//!
//! What will run is printed first (operator, solver, preconditioner, α, and the blocks of
//! the file that are not used). Per right-hand side (random, seeds A..B, 91 by default):
//! iterations, preconditioner applications, applications of the operator (outer and inside
//! the preconditioner), coarse applications, |D x − b|²/|b|² with the plain D, time, nominal
//! GFlop/s (Bridge++'s count for the domain-wall operator, two per application of the
//! even-odd operator, and 72 dim² per coarse site for a coarse application), and the memory
//! now and at its peak. A `Setup.checkpoint` in the MultiGrid block takes the low modes from
//! a `mdwf_evenodd_coherence` checkpoint instead of the filter.

use ruttice::app::memory_line;
use ruttice::app::params::{Params, PrecisionType, PreconditionerType, SolverType};
use ruttice::fermion::{FermionOp, Mdwf, PvEvenOdd, Variant};
use ruttice::io::read_bridge_text;
use ruttice::kernel::{Dirac, GammaRepre};
use ruttice::measure::{
    channel, meson_correlator, pion_correlator, point_source, CHANNELS, NUM_SOURCES,
};
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_low_modes, Aggregation, PvEoBasis, PvEoCycle,
};
use ruttice::solver::{
    largest_magnitude, BiCgStab, Cg, Chebyshev, Fgmres, Gcr, SolveError, SolveInfo, Solver, Stop,
};
use ruttice::{
    extract_parity, random_spinor, Field, Half, Lattice, LinearFn, Parity, Spinor, Vector, C64,
};
use std::cell::Cell;
use std::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: mdwf_pv_eo_mg PARAMS.yaml|PARAMS.toml [seeds:A-B]";

type H = Field<f64, Spinor, Half>;

/// The first `count` vectors of a `mdwf_evenodd_coherence` checkpoint, read site by site.
fn load(path: &Path, like: &H, count: usize) -> Result<H, Box<dyn Error>> {
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

/// The preconditioner of the outer solver.
enum Preconditioner<'a> {
    None,
    Chebyshev(Box<Chebyshev<PvEvenOdd<'a, f64, Dirac>, H>>),
    Single(Box<PvEoCycle<'a, Dirac, f32>>),
    Double(Box<PvEoCycle<'a, Dirac, f64>>),
}

impl Preconditioner<'_> {
    fn coarse_applications(&self) -> usize {
        match self {
            Preconditioner::Single(c) => c.coarse_applications(),
            Preconditioner::Double(c) => c.coarse_applications(),
            _ => 0,
        }
    }

    /// Applications of the operator inside one application of the preconditioner.
    fn inner_applications(&self, steps: usize) -> usize {
        match self {
            Preconditioner::None => 0,
            Preconditioner::Chebyshev(_) => steps - 1,
            // Two smoother applications of steps − 1 each, two residuals.
            _ => 2 * (steps - 1) + 2,
        }
    }

    fn as_fn(&mut self) -> Option<&mut dyn LinearFn<H>> {
        match self {
            Preconditioner::None => None,
            Preconditioner::Chebyshev(c) => Some(c.as_mut()),
            Preconditioner::Single(c) => Some(c.as_mut()),
            Preconditioner::Double(c) => Some(c.as_mut()),
        }
    }
}

/// Counts the applications of the map it wraps.
struct Counted<'a> {
    a: &'a mut dyn LinearFn<H>,
    n: &'a Cell<usize>,
}

impl LinearFn<H> for Counted<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.mult(out, inp);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = PathBuf::from(args.first().ok_or(USAGE)?);
    let (seed_lo, seed_hi): (u64, u64) = args
        .iter()
        .find_map(|a| a.strip_prefix("seeds:")?.split_once('-'))
        .map_or(Ok((91, 91)), |(a, b)| -> Result<_, Box<dyn Error>> {
            Ok((a.parse()?, b.parse()?))
        })?;
    let params = Params::load(&file)?;
    println!("{params}");
    let relative = |p: &PathBuf| file.parent().map_or(p.clone(), |d| d.join(p));

    let lat = Lattice::single_node(params.main.lattice_size);
    let u = read_bridge_text(&relative(&params.gauge.config_filename_input), &lat)?;
    let fo = &params.fopr;
    let make = |mass: f64, alpha: f64| -> Result<Mdwf<f64>, Box<dyn Error>> {
        let mut op = Mdwf::<f64>::new(&lat, params.mdwf_params(mass));
        op.set_gauge(&u)?;
        op.set_alpha(alpha);
        Ok(op)
    };
    let op = make(fo.quark_mass, 1.0)?;
    let pv_plain = make(fo.quark_mass_pauli_villars, 1.0)?;
    let op_a = make(fo.quark_mass, fo.neff_alpha)?;
    let pv_alpha = make(fo.quark_mass_pauli_villars, fo.neff_alpha_pauli_villars)?;
    // The normal equations are the operator with D in place of D_PV.
    let (form, normal) = params.form();
    let (pv, pv_a) = if normal {
        (&op, &op_a)
    } else {
        (&pv_plain, &pv_alpha)
    };
    let ls = fo.extent_of_5th_dimension;
    let mut like = H::with_shape(&lat, 24, ls, 1, 1);
    like.set_parity(Parity::Even);
    let index = op
        .even_odd_index()
        .ok_or("no even-odd decomposition")?
        .clone();

    // The multigrid basis lives as long as the cycle that borrows it.
    let start = Instant::now();
    let s = &params.solver;
    let mg_settings = params.multigrid();
    let basis: Option<PvEoBasis> = match (&mg_settings, &params.multigrid) {
        (Some((_, low, boot)), Some(mg)) => {
            let keep = mg.level1.number_of_test_vectors / 2;
            let mut y = match &mg.setup.checkpoint {
                Some(path) => load(&relative(path), &like, keep)?,
                None => {
                    let (x, theta) = pv_eo_low_modes(&op, pv, &lat, *low);
                    println!(
                        "  filter: Ritz values of Ã†Ã {:.3e} .. {:.3e} ({:.0} s)",
                        theta[0],
                        theta[theta.len() - 1],
                        start.elapsed().as_secs_f64()
                    );
                    x.select(&(0..keep).collect::<Vec<_>>())
                }
            };
            let agg = Aggregation::new(&lat, mg.level1.block_size);
            Some(if boot.rounds > 0 {
                pv_eo_bootstrap(&op, pv, &agg, &mut y, *boot)?
            } else {
                pv_eo_basis(&op, pv, &agg, &y)?
            })
        }
        _ => None,
    };
    let mut coarse_flops = 0.0;
    let (mut pre, steps) = match (s.preconditioner_type, &mg_settings, &basis) {
        (PreconditionerType::Multigrid, Some((cycle_params, _, _)), Some(basis)) => {
            let mg = params.multigrid.as_ref().expect("multigrid() found it");
            let dim = basis.coarse_dim() as f64;
            let coarse_sites = basis.aggregation().coarse_lattice().num_sites() as f64;
            coarse_flops = 72.0 * dim * dim * coarse_sites;
            let outer = (&op_a, pv_a, form);
            let cycle = match mg.coarse_precision {
                PrecisionType::Single => Preconditioner::Single(Box::new(PvEoCycle::new(
                    &op,
                    pv,
                    outer,
                    basis,
                    *cycle_params,
                )?)),
                PrecisionType::Double => Preconditioner::Double(Box::new(
                    PvEoCycle::with_coarse_storage(&op, pv, outer, basis, *cycle_params)?,
                )),
            };
            (cycle, cycle_params.chebyshev_steps)
        }
        (PreconditionerType::Chebyshev, _, _) => {
            let c = s.chebyshev_settings();
            let mut start_vector = like.zeros_like();
            extract_parity(
                &mut start_vector,
                &random_spinor(&lat, ls, 5),
                &index,
                Parity::Even,
            );
            let hi = 1.1
                * largest_magnitude(
                    &mut PvEvenOdd::new(&op_a, pv_a, form, &like),
                    &mut start_vector,
                    50,
                );
            let cheb = Chebyshev::new(
                PvEvenOdd::new(&op_a, pv_a, form, &like),
                c.number_of_steps,
                hi / c.interval_ratio,
                hi,
                &like,
            );
            (Preconditioner::Chebyshev(Box::new(cheb)), c.number_of_steps)
        }
        _ => (Preconditioner::None, 1),
    };
    println!(
        "  setup: {:.0} s; {}",
        start.elapsed().as_secs_f64(),
        memory_line()
    );

    let mut a = PvEvenOdd::new(&op_a, pv_a, form, &like);
    let fine_flops = 2.0 * (op.flops_per_site() * lat.num_sites()) as f64;
    let stop: Stop = s.stop();
    let k = s.number_of_kept_directions;
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let mut solve = |b: &Field<f64, Spinor>, label: &str| -> Field<f64, Spinor> {
        let nv = b.num_vectors();
        let mut rhs = like.zeros_with(nv);
        a.right_hand_side(&mut rhs, b);
        let (outer, applied) = (Cell::new(0), Cell::new(0));
        let coarse_before = pre.coarse_applications();
        let inner_per = pre.inner_applications(steps);
        let mut sol = rhs.zeros_like();
        let start = Instant::now();
        let result: Result<SolveInfo, SolveError> = {
            let mut counted_a = Counted {
                a: &mut a,
                n: &outer,
            };
            let mut counted_pre = pre.as_fn().map(|m| Counted { a: m, n: &applied });
            let m = counted_pre.as_mut().map(|c| c as &mut dyn LinearFn<H>);
            match s.solver_type {
                SolverType::Gcr => Gcr::new(stop, k).solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::BiCgStab => {
                    BiCgStab::new(stop).solve(&mut counted_a, m, &mut sol, &rhs)
                }
                SolverType::Fgmres => Fgmres::new(Stop {
                    max_iterations: k,
                    max_restarts: stop.max_iterations / k.max(1),
                    ..stop
                })
                .solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::Cg => Cg::new(stop).solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::Minres => unreachable!("refused when the file is read"),
            }
        };
        let seconds = start.elapsed().as_secs_f64();
        let iterations = match &result {
            Ok(i) => format!("{}", i.iterations),
            Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
            Err(e) => format!("failed: {e}"),
        };
        // Applications of the operator, per right-hand side.
        let applications = (outer.get() + applied.get() * inner_per) / nv;
        let coarse = (pre.coarse_applications() - coarse_before) / nv;
        let gflops = (applications as f64 * fine_flops + coarse as f64 * coarse_flops) * nv as f64
            / seconds
            * 1e-9;
        let mut x = b.zeros_like();
        a.solution(&mut x, &sol, b);
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, &x, Variant::D);
        r.aypx(&vec![C64::new(-1.0, 0.0); nv], b);
        let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        let worst = rr.iter().zip(&bb).fold(0.0_f64, |m, (r, b)| m.max(r / b));
        println!(
            "  {label}: {} {iterations} iterations; per right-hand side {} preconditioner applications, {applications} applications of the operator = {} Wilson sweeps, {coarse} coarse; worst |Dx−b|² {worst:.0e}; {seconds:.1} s, {gflops:.1} GFlop/s; {}",
            s.solver_type,
            applied.get() / nv,
            2 * applications,
            memory_line()
        );
        x
    };

    let Some(src) = &params.source else {
        for seed in seed_lo..=seed_hi {
            let mut b = Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
            b.as_mut_slice()
                .copy_from_slice(random_spinor(&lat, ls, seed).as_slice());
            solve(&b, &format!("seed {seed}"));
        }
        return Ok(());
    };

    // The propagator from the point source, `number_of_rhs_per_solve` columns at a time:
    // the four-dimensional quark field and the midpoint field of every column.
    let eta = point_source::<f64>(&lat, src.source_position);
    let (mut prop, mut mid) = (eta.zeros_like(), eta.zeros_like());
    let columns: Vec<usize> = (0..NUM_SOURCES).collect();
    for chunk in columns.chunks(src.number_of_rhs_per_solve) {
        let eta_c = eta.select(chunk);
        let mut b = Field::<f64, Spinor>::with_shape(&lat, 24, ls, chunk.len(), 1);
        op.source_5d(&mut op.work(), &mut b, &eta_c);
        let psi = solve(&b, &format!("columns {chunk:?}"));
        let (mut q, mut m) = (eta_c.zeros_like(), eta_c.zeros_like());
        op.propagator_4d(&mut q, &psi);
        op.midpoint_4d(&mut m, &psi);
        prop.assign(chunk, &q);
        mid.assign(chunk, &m);
    }
    // Bridge++'s layout of the output: one block per channel, t from the source.
    let t0 = src.source_position[3];
    let nt = params.main.lattice_size[3];
    let shift = |c: &[C64]| -> Vec<C64> { (0..nt).map(|t| c[(t + t0) % nt]).collect() };
    for name in CHANNELS {
        let (sink, source) = channel::<Dirac>(name).expect("a channel of the list");
        let c = shift(&meson_correlator(
            &prop,
            &prop,
            &sink,
            &source,
            &Dirac::GAMMA5,
        ));
        println!("  {name}:");
        for (t, z) in c.iter().enumerate() {
            println!("    {t:3}  {:+.15e}  {:+.15e}", z.re, z.im);
        }
    }
    let (pp, j5q) = (pion_correlator(&prop), pion_correlator(&mid));
    println!("  residual mass m_res(t) = C_J5q(t) / C_PS(t):");
    for t in 0..nt {
        let at = (t + t0) % nt;
        println!(
            "    {t:3}  {:.6e}   (C_J5q {:.6e}, C_PS {:.6e})",
            j5q[at] / pp[at],
            j5q[at],
            pp[at]
        );
    }
    Ok(())
}
