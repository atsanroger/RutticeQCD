//! How exact the factorized coarse correction is on the vectors its prolongation is made from
//! (MDWF_MULTIGRID_NOTE, Proposition 1 and the remarks after it).
//!
//! With the coarse factors solved to |r|²/|b|² < 1e-24 in double precision on their exact
//! adjoints (BiCGStab), and P as the solver stores it (half precision), the correction e(r) of
//! the right form should return y for r = Ã_R y, y one of the vectors the basis was built
//! from, to the rounding of P. The program prints ‖e − y‖/‖y‖ for every such y, and for
//! comparison for random vectors, for the left form (α = 1, y' = C_e⁻¹ y, r = Ã_L y'), and for
//! the dual correction (D and D_PV exchanged, the cycle used for Ã_R†) with r = Ã_R† y.
//!
//! ```text
//! cargo run --release --example correction_exactness -- L FORMAT FILE M5 MASS SMEAR [--t T]
//!     [--block B] [--vectors N] [--steps S] [--ratio R] [--gcr-only]
//! ```
//!
//! Ls 8, b 1.5, c 0.5; SMEAR 1 for stout 0.1 × 3. Setup as the solver's (16 vectors filtered in
//! 2 rounds, the bootstrap 2 × 4). Defaults: blocks of 2⁴, all 16 vectors; the smoother of the
//! GCR(2) solves S Chebyshev steps on [hi/R, hi], 8 and 10 (setup S; setup B: 12 and 30).

use ruttice::fermion::{Dagger, EoForm, EvenOdd, FermionOp, Mdwf, MdwfParams, PvEvenOdd};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::kernel::Dirac;
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseKrylov, LowModeParams,
    PvEoCycle, PvEoParams,
};
use ruttice::solver::{Gcr, Solver, Stop};
use ruttice::{
    extract_parity, random_spinor, Aos, Boundary, Field, Half, Lattice, LinearFn, Parity, Spinor,
    Vector, C64, ND,
};
use std::error::Error;
use std::path::Path;

type H = Field<f64, Spinor, Half>;

/// ‖a − b‖ / ‖b‖.
fn relative_difference(a: &H, b: &H) -> f64 {
    let mut d = a.zeros_like();
    d.copy_from(a);
    d.axpy(&[C64::new(-1.0, 0.0)], b);
    let (mut nd, mut nb) = ([0.0], [0.0]);
    d.norm_sqr(&mut nd);
    b.norm_sqr(&mut nb);
    (nd[0] / nb[0]).sqrt()
}

/// The smallest, median and largest of `v`.
fn summary(v: &mut [f64]) -> String {
    v.sort_by(f64::total_cmp);
    format!(
        "min {:.2e}  median {:.2e}  max {:.2e}  ({} vectors)",
        v[0],
        v[v.len() / 2],
        v[v.len() - 1],
        v.len()
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        return Err(
            "L FORMAT FILE M5 MASS SMEAR [--t T] [--block B] [--vectors N] [--steps S] [--ratio R]"
                .into(),
        );
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
    let nt = value("--t", l as f64)? as usize;
    let block = value("--block", 2.0)? as usize;
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

    // The setup of the solver.
    let (y, _) = pv_eo_low_modes(&op, &pv, &lat, LowModeParams::default());
    let vectors = value("--vectors", y.num_vectors() as f64)? as usize;
    let mut y = y.select(&(0..vectors).collect::<Vec<_>>());
    let agg = Aggregation::new(&lat, [block; ND]);
    let basis = pv_eo_bootstrap(&op, &pv, &agg, &mut y, BootstrapParams::default())?;
    println!(
        "# {l}^3x{nt} {}; M5 {m5}, m {mass}, stout 0.1x3: {smear}; blocks {block}^4, {vectors} vectors (P in half precision); coarse solves BiCGStab on the exact adjoints to 1e-24 in double precision",
        args[2]
    );

    // Exact coarse solves.
    let exact = PvEoParams {
        d_solver: CoarseKrylov::BiCgStab,
        pv_solver: CoarseKrylov::BiCgStab,
        coarse_tolerance_sqr: 1e-24,
        pv_tolerance_sqr: 1e-24,
        coarse_max_iterations: 50_000,
        pv_max_iterations: 50_000,
        ..PvEoParams::default()
    };
    let like = y.select(&[0]);
    let index = op.even_odd_index().expect("even-odd").clone();
    let random_even = |seed: u64| {
        let mut v = like.zeros_like();
        extract_parity(&mut v, &random_spinor(&lat, 8, seed), &index, Parity::Even);
        v
    };

    // e(r) of a cycle for r = a y, against y.
    let check = |cycle: &mut PvEoCycle<'_, Dirac, f64, Aos, f64>,
                 a: &mut PvEvenOdd<'_, f64>,
                 targets: &[H]|
     -> Vec<f64> {
        targets
            .iter()
            .map(|t| {
                let (mut r, mut e) = (t.zeros_like(), t.zeros_like());
                a.mult(&mut r, t);
                cycle.coarse_correction(&mut e, &r);
                relative_difference(&e, t)
            })
            .collect()
    };
    let ys: Vec<H> = (0..vectors).map(|j| y.select(&[j])).collect();
    let randoms: Vec<H> = (0..4).map(|k| random_even(100 + k)).collect();
    let mut a_right = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);
    let mut a_dag = PvEvenOdd::new(&pv, &op, EoForm::Right, &like);

    // --gcr-only: the GCR(2) solves below without the exactness checks.
    if !args.iter().any(|a| a == "--gcr-only") {
        // Right form: Proposition 1.
        let mut right = PvEoCycle::<Dirac, f64>::with_coarse_storage(
            &op,
            &pv,
            (&op, &pv, EoForm::Right),
            &basis,
            exact,
        )?;
        println!(
            "right form, r = A_R y:         {}",
            summary(&mut check(&mut right, &mut a_right, &ys))
        );
        println!(
            "right form, r = A_R z, z random: {}",
            summary(&mut check(&mut right, &mut a_right, &randoms))
        );

        // Left form, alpha = 1: y' = C_e^-1 y.
        let mut left = PvEoCycle::<Dirac, f64>::with_coarse_storage(
            &op,
            &pv,
            (&op, &pv, EoForm::Left),
            &basis,
            exact,
        )?;
        let mut a_left = PvEvenOdd::new(&op, &pv, EoForm::Left, &like);
        let mut work = op.even_odd_work();
        let primed: Vec<H> = ys
            .iter()
            .map(|yj| {
                let mut out = yj.zeros_like();
                op.diag_inv(&mut work, &mut out, yj, Parity::Even, Dagger::No);
                out
            })
            .collect();
        println!(
            "left form, r = A_L C_e^-1 y:   {}",
            summary(&mut check(&mut left, &mut a_left, &primed))
        );
        println!(
            "left form, r = A_L z, z random: {}",
            summary(&mut check(&mut left, &mut a_left, &randoms))
        );

        // The dual correction (D and D_PV exchanged on the same P), as the cycle for A_R^dag.
        let mut dual = PvEoCycle::<Dirac, f64>::with_coarse_storage(
            &pv,
            &op,
            (&pv, &op, EoForm::Right),
            &basis,
            exact,
        )?;
        println!(
            "dual, r = A_R^dag y:           {}",
            summary(&mut check(&mut dual, &mut a_dag, &ys))
        );
        println!(
            "dual, r = A_R^dag z, z random: {}",
            summary(&mut check(&mut dual, &mut a_dag, &randoms))
        );
    }

    // What the inexactness costs in a solve: GCR(2) on A_R with M and on A_R^dag with the dual
    // cycle, the coarse solves as the solver stops them (1e-2; BiCGStab for both factors in both
    // directions, for a like comparison), to |r|^2/|b|^2 < 1e-18 from the same random b.
    let solver = PvEoParams {
        pv_solver: CoarseKrylov::BiCgStab,
        chebyshev_steps: value("--steps", 8.0)? as usize,
        chebyshev_ratio: value("--ratio", 10.0)?,
        ..PvEoParams::default()
    };
    let b = random_even(200);
    for (name, outer, a) in [
        ("A_R x = b, cycle M", (&op, &pv), &mut a_right),
        ("A_R^dag x = b, dual cycle", (&pv, &op), &mut a_dag),
    ] {
        let mut cycle = PvEoCycle::<Dirac, f64>::with_coarse_storage(
            outer.0,
            outer.1,
            (outer.0, outer.1, EoForm::Right),
            &basis,
            solver,
        )?;
        let mut x = b.zeros_like();
        let info = Gcr::new(Stop::simple(1e-18, 1000), 2).solve(a, Some(&mut cycle), &mut x, &b);
        match info {
            Ok(info) => println!(
                "GCR(2) {name}: {} iterations, {} coarse applications",
                info.iterations,
                cycle.coarse_applications()
            ),
            Err(e) => println!("GCR(2) {name}: {e}"),
        }
    }
    Ok(())
}
