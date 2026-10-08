//! Does the coarse correction of the domain-wall multigrid start to help when the test
//! vectors carry the low modes of A? The reference setting takes the eigenvectors of A
//! with the smallest |λ| (IRArnoldi); here the cheap stand-in is inverse iteration: solve
//! A x = v for random v to a loose tolerance.
//!
//! ```text
//! cargo run --release --example mdwf_testvectors -- CONFIG L M5 MASS [LS] [BLOCK] [NVEC]
//! ```
//!
//! For each source of test vectors and each coarse operator: the residual the cycle
//! leaves on a random and on a smooth vector, |v - A M v| (with no coarse correction, two
//! GMRES(4) smoothings leave 0.079 and 0.246 on the 12^4 test), and BiCGStab and
//! FGMRES(20) preconditioned by the cycle, to |r|^2/|b|^2 < 1e-16 of A x = P b, with
//! |D x - b|^2/|b|^2.

use ruttice::fermion::{FermionOp, Mdwf, MdwfAt, MdwfParams, PvPreconditioned, SiteInverse, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, CycleParts, Multigrid, SetupHooks,
};
use ruttice::solver::{BiCgStab, Fgmres, SolveError, Solve, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_testvectors CONFIG L M5 MASS [LS] [BLOCK] [NVEC]";

type V = Field<f64, Spinor>;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=7).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args.get(4).map_or(Ok(8), |s| s.parse()).map_err(|_| USAGE)?;
    let block: usize = args.get(5).map_or(Ok(3), |s| s.parse()).map_err(|_| USAGE)?;
    let nvec: usize = args.get(6).map_or(Ok(8), |s| s.parse()).map_err(|_| USAGE)?;

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
    let mut op = Mdwf::<f64>::new(&lat, params);
    op.set_gauge(&u)?;
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4, {nvec} test vectors");

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let normalise = |v: &mut V| {
        let mut n = vec![0.0; v.num_vectors()];
        v.norm_sqr(&mut n);
        let s: Vec<f64> = n.iter().map(|x| 1.0 / x.sqrt()).collect();
        v.scale(&s);
    };
    let b = random(1, 91);
    let mut rhs = b.zeros_like();
    PvPreconditioned::new(&op, &b).precondition(&mut rhs, &b);
    let mut rough = random(1, 301);
    normalise(&mut rough);
    let mut smooth = rough.zeros_like();
    {
        let v = random(1, 302);
        let mut a = PvPreconditioned::new(&op, &v);
        let _ = BiCgStab::new(Stop::simple(1e-12, 2000)).solve(&mut a, None, &mut smooth, &v);
        normalise(&mut smooth);
    }
    let physical = |x: &V| {
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        left[0] / size[0]
    };

    // Sources of test vectors: random, or inverse iteration to a tolerance.
    let start = random(nvec, 1);
    for inverse_tolerance in [None, Some(1e-2), Some(1e-4), Some(1e-6)] {
        let mut initial = start.zeros_like();
        initial.copy_from(&start);
        let label = match inverse_tolerance {
            None => "random".to_string(),
            Some(tol) => {
                let t = Instant::now();
                let mut a = PvPreconditioned::new(&op, &start);
                let mut x = start.zeros_like();
                let info = BiCgStab::new(Stop::simple(tol, 2000)).solve(&mut a, None, &mut x, &start);
                let iterations = match info {
                    Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
                    Err(e) => return Err(e.to_string().into()),
                };
                normalise(&mut x);
                initial = x;
                format!(
                    "inverse iteration to {tol:.0e} ({iterations} BiCGStab iterations, {:.1} s)",
                    t.elapsed().as_secs_f64()
                )
            }
        };
        println!("test vectors: {label}");
        for galerkin in [false, true] {
            let like = b.zeros_like();
            let gmres4 = || -> Box<dyn LinearFn<V> + '_> {
                Box::new(Solve::new(
                    Fgmres::new(Stop::simple(1e-6, 4)),
                    PvPreconditioned::new(&op, &like),
                ))
            };
            let parts = CycleParts {
                operator: Box::new(PvPreconditioned::new(&op, &like)),
                pre_smoother: Some(gmres4()),
                post_smoother: gmres4(),
                coarse_solver: Box::new(Fgmres::new(Stop {
                    tolerance_sqr: 1e-4,
                    max_iterations: 5,
                    max_restarts: 1,
                })),
            };
            let mut hooks = SetupHooks {
                precondition_coarse_diagonal: true,
                ..SetupHooks::default()
            };
            if galerkin {
                hooks.galerkin_of = Some(Box::new(PvPreconditioned::new(&op, &like)));
            } else {
                hooks.after_smoothing = Some(Box::new(MdwfAt::new(&op, Variant::D, 1.0)));
                hooks.coarse_right = Some(Box::new(SiteInverse {
                    op: &op,
                    mass: 1.0,
                    dagger: false,
                }));
            }
            let agg = Aggregation::new(&lat, [block; ND]);
            let mut vectors = initial.zeros_like();
            vectors.copy_from(&initial);
            let mut mg = Multigrid::with_parts(&op, &agg, vectors, 0, parts, hooks)?;
            let name = if galerkin {
                "Galerkin P†AP, no D_PV"
            } else {
                "P†D C_PV^-1 P, D_PV (reference)"
            };

            let mut a = PvPreconditioned::new(&op, &like);
            let mut leaves = |v: &V| {
                let (mut mv, mut amv) = (v.zeros_like(), v.zeros_like());
                mg.mult(&mut mv, v);
                a.mult(&mut amv, &mv);
                amv.aypx(&[C64::new(-1.0, 0.0)], v);
                let mut left = [0.0];
                amv.norm_sqr(&mut left);
                left[0].sqrt()
            };
            let (on_rough, on_smooth) = (leaves(&rough), leaves(&smooth));
            let stop = Stop {
                tolerance_sqr: 1e-16,
                max_iterations: 1000,
                max_restarts: 1,
            };
            let mut outer = |solver: &mut dyn Solver<V>| {
                let mut x = b.zeros_like();
                let mut a = PvPreconditioned::new(&op, &b);
                let it = match solver.solve(&mut a, Some(&mut mg), &mut x, &rhs) {
                    Ok(i) => format!("{}", i.iterations),
                    Err(SolveError::NotConverged(i)) => format!("{} (not converged)", i.iterations),
                    Err(e) => format!("failed: {e}"),
                };
                format!("{it}, |Dx-b|^2/|b|^2 {:.1e}", physical(&x))
            };
            let bicgstab = outer(&mut BiCgStab::new(stop));
            let fgmres = outer(&mut Fgmres::new(Stop {
                max_iterations: 20,
                max_restarts: 50,
                ..stop
            }));
            println!("  {name:<32} |v-AMv| random {on_rough:.3} smooth {on_smooth:.3}");
            println!("      BiCGStab {bicgstab};  FGMRES(20) {fgmres}");
        }
    }
    Ok(())
}
