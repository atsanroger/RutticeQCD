//! Chebyshev-filtered test vectors for the domain-wall multigrid (DESIGN.md §7.6.5 (b)),
//! judged on genuine low modes of A.
//!
//! ```text
//! cargo run --release --example mdwf_lowmodes -- CONFIG L M5 MASS [LS] [BLOCK] [NVEC] [DEGREE]
//! ```
//!
//! H = A†A. A low-mode error t is a random vector filtered by T_n(ℓ(H)) with a high degree
//! (2 DEGREE), normalised; |A t| / |t| shows how low it is (a random vector gives about
//! 1.5, the solution of A x = v for a random v about 0.3).
//!
//! For test vectors from inverse iteration (A x = v to 1e-6) and from the Chebyshev
//! filter (degree DEGREE, lo = hi/1000, hi = 1.1 λ_max(H)): |P P† t| / |t| and what the
//! exact Galerkin coarse correction P (P†A P)^-1 P† A leaves of t; then BiCGStab and
//! FGMRES(20) with the multigrid cycle (GMRES(4) smoother, probed Galerkin coarse
//! operator, diagonal-block preconditioned) to |r|^2/|b|^2 < 1e-16 of A y = X† b.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, Basis, CycleParts, Multigrid, SetupHooks,
};
use ruttice::solver::{
    largest_magnitude, BiCgStab, ChebyshevFilter, Fgmres, Solve, SolveError, Solver, Stop,
};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, Normal, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_lowmodes CONFIG L M5 MASS [LS] [BLOCK] [NVEC] [DEGREE]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

/// Normalise every vector of the field.
fn normalise(v: &mut V) {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    let s: Vec<f64> = n.iter().map(|x| 1.0 / x.sqrt()).collect();
    v.scale(&s);
}

/// P† A P through the fine lattice.
struct Exact<'a, 'b> {
    a: PvPreconditioned<'a, f64>,
    p: &'b Basis<f64>,
    fine: V,
    a_fine: V,
}

impl LinearFn<C> for Exact<'_, '_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.p.prolong(&mut self.fine, inp);
        self.a.mult(&mut self.a_fine, &self.fine);
        self.p.restrict(out, &self.a_fine);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=8).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args
        .get(4)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let block: usize = args
        .get(5)
        .map_or(Ok(3), |s| s.parse())
        .map_err(|_| USAGE)?;
    let nvec: usize = args
        .get(6)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let degree: usize = args
        .get(7)
        .map_or(Ok(100), |s| s.parse())
        .map_err(|_| USAGE)?;

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
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4, {nvec} test vectors, filter degree {degree}"
    );

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let like = random(1, 1);
    let mut a = PvPreconditioned::new(&op, &like);
    let ratio = |a: &mut PvPreconditioned<'_, f64>, v: &V| {
        let mut w = v.zeros_like();
        a.mult(&mut w, v);
        norm(&w) / norm(v)
    };

    let start = Instant::now();
    let hi = 1.1
        * largest_magnitude(
            &mut Normal::new(PvPreconditioned::new(&op, &like), &like),
            &mut random(1, 5),
            50,
        );
    let lo = hi / 1000.0;
    println!(
        "lambda_max(A†A) ~ {:.3}; filter on [{lo:.4}, {hi:.3}] ({:.1} s)",
        hi / 1.1,
        start.elapsed().as_secs_f64()
    );
    let filter = |x: &V, degree: usize| {
        let mut f = ChebyshevFilter::new(
            Normal::new(PvPreconditioned::new(&op, x), x),
            degree,
            lo,
            hi,
            x,
        );
        let mut y = x.zeros_like();
        f.mult(&mut y, x);
        normalise(&mut y);
        y
    };

    let start = Instant::now();
    let t = filter(&random(1, 302), 2 * degree);
    println!(
        "low-mode error t (filter degree {}): |A t|/|t| {:.4}  ({:.1} s);  random vector {:.4}",
        2 * degree,
        ratio(&mut a, &t),
        start.elapsed().as_secs_f64(),
        ratio(&mut a, &random(1, 77))
    );

    let b = random(1, 91);
    let mut rhs = b.zeros_like();
    a.right_hand_side(&mut rhs, &b);
    let physical = |x: &V| {
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        norm(&r).powi(2) / norm(&b).powi(2)
    };

    let agg = Aggregation::new(&lat, [block; ND]);
    for source in ["inverse iteration to 1e-6", "Chebyshev filter"] {
        let start = Instant::now();
        let mut vectors = if source == "Chebyshev filter" {
            filter(&random(nvec, 11), degree)
        } else {
            let v = random(nvec, 11);
            let mut x = v.zeros_like();
            let mut a = PvPreconditioned::new(&op, &v);
            let _ = BiCgStab::new(Stop::simple(1e-6, 4000)).solve(&mut a, None, &mut x, &v);
            x
        };
        normalise(&mut vectors);
        let made = start.elapsed().as_secs_f64();
        let mut g5 = vectors.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
        let p = Basis::from_vectors(&agg, &vectors, &g5)?;

        // How much of t the coarse space holds, and what the exact correction leaves.
        let mut c = p.coarse_field(1);
        p.restrict(&mut c, &t);
        let mut inside = like.zeros_like();
        p.prolong(&mut inside, &c);
        let held = norm(&inside);
        let mut a_t = t.zeros_like();
        a.mult(&mut a_t, &t);
        let mut rc = p.coarse_field(1);
        p.restrict(&mut rc, &a_t);
        let mut y = rc.zeros_like();
        let mut exact = Exact {
            a: PvPreconditioned::new(&op, &like),
            p: &p,
            fine: like.zeros_like(),
            a_fine: like.zeros_like(),
        };
        let _ = Fgmres::new(Stop {
            tolerance_sqr: 1e-20,
            max_iterations: 100,
            max_restarts: 50,
        })
        .solve(&mut exact, None, &mut y, &rc);
        let mut left = like.zeros_like();
        p.prolong(&mut left, &y);
        left.aypx(&[C64::new(-1.0, 0.0)], &t);
        println!(
            "{source} ({made:.0} s): |PP†t|/|t| {held:.3}  exact coarse correction leaves {:.3} of t (floor {:.3})",
            norm(&left),
            (1.0 - held * held).sqrt()
        );

        // The cycle.
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
        let hooks = SetupHooks {
            precondition_coarse_diagonal: true,
            galerkin_of: Some(Box::new(PvPreconditioned::new(&op, &like))),
            ..SetupHooks::default()
        };
        let mut mg = Multigrid::with_parts(&op, &agg, vectors, 0, parts, hooks)?;
        let stop = Stop {
            tolerance_sqr: 1e-16,
            max_iterations: 1500,
            max_restarts: 1,
        };
        let mut line = String::from("    cycle:");
        for flexible in [false, true] {
            let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
            let mut a = PvPreconditioned::new(&op, &b);
            let result = if flexible {
                Fgmres::new(Stop {
                    max_iterations: 20,
                    max_restarts: 100,
                    ..stop
                })
                .solve(&mut a, Some(&mut mg), &mut y, &rhs)
            } else {
                BiCgStab::new(stop).solve(&mut a, Some(&mut mg), &mut y, &rhs)
            };
            a.solution(&mut x, &y);
            let it = match result {
                Ok(i) => format!("{}", i.iterations),
                Err(SolveError::NotConverged(i)) => format!("{} (not converged)", i.iterations),
                Err(e) => format!("failed: {e}"),
            };
            line += &format!(
                "  {} {it}, |Dx-b|^2/|b|^2 {:.1e};",
                if flexible { "FGMRES(20)" } else { "BiCGStab" },
                physical(&x)
            );
        }
        println!("{line}");
    }
    Ok(())
}
