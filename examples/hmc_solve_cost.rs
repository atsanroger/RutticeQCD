//! What the solves of HMC cost with and without the multigrid of domain-wall fermions: x =
//! (D†D)⁻¹ b by CG, and by the two solves through Ã and Ã† with the cycles M and M†
//! (`hmc::dwf_multigrid_normal_solve`), on a configuration and then on links moved as a step of
//! the molecular dynamics moves them (U → exp(ε iP) U, P Gaussian), where the multigrid keeps its
//! test vectors and only makes P and the coarse operators again.
//!
//! ```text
//! cargo run --release --example hmc_solve_cost -- L FORMAT FILE M5 MASS SMEAR [--vectors N]
//!     [--block B] [--steps S] [--ratio R] [--inner T] [--eps E] [--moves K]
//! ```
//!
//! On the L⁴ configuration FILE, Ls 8, b 1.5, c 0.5, SMEAR 1 for stout 0.1 × 3; the solves to
//! |r|²/|b|² < 1e-24. Defaults: 16 test vectors, blocks of 4⁴, Chebyshev 8 steps on [hi/30, hi],
//! each solve on the even sites to 1e-10, ε 0.05 (a step of a trajectory of length 1 in 20),
//! three moves.

use ruttice::fermion::{Bound, FermionOp, Mdwf, MdwfParams, Variant};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::hmc::{dwf_multigrid_normal_solve, DwfMultigridSetup, LinkField, Mseries};
use ruttice::solver::preconditioner::multigrid::{MdwfMultigridSettings, PvEoParams};
use ruttice::solver::{Cg, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Gauge, Lattice, Normal, Spinor, Vector, ND};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        return Err("L FORMAT FILE M5 MASS SMEAR [--vectors N] [--block B] [--steps S] [--ratio R] [--inner T] [--eps E] [--moves K]".into());
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
    let vectors = value("--vectors", 16.0)? as usize;
    let block = value("--block", 4.0)? as usize;
    let cycle = PvEoParams {
        chebyshev_steps: value("--steps", 8.0)? as usize,
        chebyshev_ratio: value("--ratio", 30.0)?,
        ..PvEoParams::default()
    };
    let inner = value("--inner", 1e-10)?;
    let eps = value("--eps", 0.05)?;
    let moves = value("--moves", 3.0)? as usize;
    let lat = Lattice::single_node([l; ND]);
    let mut u = ruttice::app::read_gauge_named(&args[1], Path::new(&args[2]), &lat)?;
    let smearing = smear.then_some(GaugeSmearing {
        smear: Smear::Stout(Stout { rho: uniform(0.1) }),
        steps: 3,
    });
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
    let (mut op, mut pv) = (
        Mdwf::<f64>::new(&lat, params(mass)),
        Mdwf::<f64>::new(&lat, params(1.0)),
    );
    let set_links = |u: &Field<f64, Gauge>, op: &mut Mdwf<f64>, pv: &mut Mdwf<f64>| {
        let v = match &smearing {
            Some(s) => s.levels(u).pop().expect("three steps"),
            None => u.duplicate(),
        };
        op.set_gauge(&v).expect("the links");
        pv.set_gauge(&v).expect("the links");
    };
    set_links(&u, &mut op, &mut pv);
    let mut b = Field::<f64, Spinor>::with_shape(&lat, 24, 8, 1, 1);
    b.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, 8, 77).as_slice());
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 100_000,
        max_restarts: 20,
    };
    println!(
        "# {l}^4 {}; M5 {m5}, m {mass}, Ls 8, b 1.5 c 0.5, stout 0.1x3: {smear}; {vectors} vectors, blocks {block}^4, Chebyshev {} on [hi/{}, hi], inner 1e-{:.0}; eps {eps}",
        args[2],
        cycle.chebyshev_steps,
        cycle.chebyshev_ratio,
        -inner.log10()
    );
    let mut setup = DwfMultigridSetup::new(
        MdwfMultigridSettings::new([block; ND], vectors, cycle),
        Stop::simple(inner, 4000),
    );
    let mut rng = Mseries::new(5);
    println!("# move  CG: iterations seconds  multigrid: setup rebuild solve seconds, solves on the even sites, GCR iterations");
    for k in 0..=moves {
        if k > 0 {
            let p = <Field<f64, Gauge> as LinkField>::draw_momenta(&lat, &mut rng);
            u.exp_update(eps, &p, 12);
            set_links(&u, &mut op, &mut pv);
        }
        let t = Instant::now();
        let mut x = b.zeros_like();
        let mut d = Bound::new(&op, Variant::D);
        let cg = Cg::new(stop).solve(&mut Normal::new(&mut d, &b), None, &mut x, &b)?;
        let cg_seconds = t.elapsed().as_secs_f64();
        let before = (setup.seconds, setup.inner_solves);
        let mut y = b.zeros_like();
        let info =
            dwf_multigrid_normal_solve(&op, &pv, &mut setup, (k == 0, k > 0), &mut y, &b, stop)?;
        // The two solutions agree to the tolerance.
        let mut diff = y.zeros_like();
        diff.copy_from(&y);
        diff.axpy(&[ruttice::C64::new(-1.0, 0.0)], &x);
        let (mut dd, mut xx) = ([0.0], [0.0]);
        diff.norm_sqr(&mut dd);
        x.norm_sqr(&mut xx);
        println!(
            "{k:4}  {:6} {:8.2}  {:8.2} {:8.2} {:8.2}  {:3} {:5}  |x_mg - x_cg|/|x| {:.1e}",
            cg.iterations,
            cg_seconds,
            setup.seconds.setup - before.0.setup,
            setup.seconds.rebuild - before.0.rebuild,
            setup.seconds.solve - before.0.solve,
            setup.inner_solves - before.1,
            info.iterations,
            (dd[0] / xx[0]).sqrt()
        );
        let _ = FermionOp::flops_per_site(&op);
    }
    Ok(())
}
