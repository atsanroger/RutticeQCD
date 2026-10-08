//! The solvers of the Möbius domain-wall operator side by side, on one configuration and
//! one right-hand side, compared at the same true residual |D x - b| / |b|.
//!
//! ```text
//! cargo run --release --example mdwf_solvers -- CONFIG L M5 MASS [LS] [BLOCK] [ONLY]
//! ```
//!
//! CONFIG is a plain-text configuration on an L^4 lattice. Möbius b = 1.5, c = 0.5,
//! boundary conditions (1, 1, 1, -1), LS = 8 and aggregates of BLOCK^4 (3) unless given.
//! ONLY, if given, runs only the solvers whose name contains it.
//!
//! Every solver stops at |r|^2/|b|^2 of the system it solves, which is not D x = b for
//! most of them; so each is run to a ladder of tolerances, 1e-10 down to 1e-22, from a
//! zero start each time. The solvers are deterministic, so the rungs are points of one
//! convergence history. Each line: the tolerance, iterations, applications of the Wilson
//! kernel by the outer operator (not those inside a preconditioner: compare seconds
//! there), seconds, the solver's own
//! final residual and |D x - b|^2/|b|^2. A ladder stops at the first rung that does not
//! converge.
//!
//! - BiCGStab and FGMRES(20) solve A y = X† b, x = C^-1 y, with A = (D_PV C_PV^-1)† D C^-1
//!   and X = D_PV C_PV^-1, without a
//!   preconditioner, with the frozen polynomial of A alone (two passes: what a cycle
//!   does without its coarse correction), and with the multigrid cycle smoothing with
//!   GMRES(4) or with the polynomial.
//! - CGNE: CG on the normal equations of the even-odd Schur complement of D.

use ruttice::fermion::{solve_even_odd, FermionOp, Mdwf, MdwfParams, PvPreconditioned, Variant};
use ruttice::gauge::plaquette;
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    mdwf_multigrid, MdwfMultigridParams, MdwfSmoother,
};
use ruttice::solver::{
    solve_cgne, BiCgStab, Cg, Fgmres, Polynomial, SolveError, SolveInfo, Solver, Stop,
};
use ruttice::{
    random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Stencil, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_solvers CONFIG L M5 MASS [LS] [BLOCK] [ONLY]";

/// Tolerances on |r|^2/|b|^2 of the system each solver works on.
const LADDER: [f64; 7] = [1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22];

type V = Field<f64, Spinor>;

/// One solve of a ladder: the stopping rule and a zero `x` in; the result and the sweeps of
/// the Wilson kernel out.
type Run<'a> = dyn FnMut(Stop, &mut V) -> (Result<SolveInfo, SolveError>, usize) + 'a;

/// Counts the applications of the operator it wraps.
struct Counted<'a, A> {
    a: &'a mut A,
    count: usize,
}

impl<A: LinearFn<V>> LinearFn<V> for Counted<'_, A> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.count += 1;
        self.a.mult(out, inp);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=7).contains(&args.len()) {
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
    let only = args.get(6).cloned().unwrap_or_default();
    let wanted = |name: &str| name.contains(only.as_str());

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
        "{l}^4, plaquette {:.6}, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4",
        plaquette(&u, &Stencil::nearest(&lat, [Boundary::Periodic; ND]))
    );

    let mut b = V::with_shape(&lat, 24, ls, 1, 1);
    b.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, ls, 91).as_slice());
    let physical = |x: &V| {
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        left[0] / size[0]
    };
    // One ladder: `run` solves to the given stopping rule into a zero `x` and returns the
    // result and the Wilson sweeps of the outer operator.
    let ladder = |name: &str, run: &mut Run<'_>| {
        println!("{name}");
        println!("    tol    iter   sweeps        s    own res  |Dx-b|^2/|b|^2");
        for tol in LADDER {
            let stop = Stop {
                tolerance_sqr: tol,
                max_iterations: 2000,
                max_restarts: 1,
            };
            let mut x = b.zeros_like();
            let start = Instant::now();
            let (result, sweeps) = run(stop, &mut x);
            let seconds = start.elapsed().as_secs_f64();
            let (info, note) = match result {
                Ok(info) => (Some(info), ""),
                Err(SolveError::NotConverged(info)) => (Some(info), "  not converged"),
                Err(e) => {
                    println!("  {tol:.0e}  failed: {e}");
                    return;
                }
            };
            let info = info.unwrap();
            println!(
                "  {tol:.0e} {:6} {sweeps:8} {seconds:8.2}  {:9.1e}  {:9.1e}{note}",
                info.iterations,
                info.residual_sqr[0],
                physical(&x)
            );
            if !note.is_empty() {
                return;
            }
        }
    };

    let mut rhs = b.zeros_like();
    PvPreconditioned::new(&op, &b).right_hand_side(&mut rhs, &b);
    let sample = {
        let mut s = b.zeros_like();
        s.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, 7).as_slice());
        s
    };
    // A applies D and D_PV†: two Wilson sweeps. Solves A y = X† b, then x = C^-1 y.
    let on_a = |solver: &mut dyn Solver<V>, m: Option<&mut dyn LinearFn<V>>, x: &mut V| {
        let mut a = PvPreconditioned::new(&op, &b);
        let mut y = b.zeros_like();
        let (result, count) = {
            let mut counted = Counted {
                a: &mut a,
                count: 0,
            };
            let result = solver.solve(&mut counted, m, &mut y, &rhs);
            (result, counted.count)
        };
        a.solution(x, &y);
        (result, 2 * count)
    };

    if wanted("BiCGStab, A") {
        ladder("BiCGStab, A", &mut |stop, x| {
            on_a(&mut BiCgStab::new(stop), None, x)
        });
    }
    for degree in [4, 8] {
        let name = format!("BiCGStab, polynomial {degree} x 2");
        if !wanted(&name) {
            continue;
        }
        let mut p = Polynomial::fit(PvPreconditioned::new(&op, &b), &sample, degree, 2)
            .map_err(|_| "the powers of A are dependent")?;
        ladder(&name, &mut |stop, x| {
            on_a(&mut BiCgStab::new(stop), Some(&mut p), x)
        });
    }
    let smoothers = [
        (
            "polynomial 4",
            MdwfSmoother::Polynomial {
                degree: 4,
                passes: 1,
            },
        ),
        (
            "polynomial 8",
            MdwfSmoother::Polynomial {
                degree: 8,
                passes: 1,
            },
        ),
        ("GMRES(4)", MdwfSmoother::Gmres(Stop::simple(1e-6, 4))),
    ];
    for (smoother_name, smoother) in smoothers {
        let bicgstab = format!("BiCGStab, cycle with {smoother_name}");
        let fgmres = format!("FGMRES(20), cycle with {smoother_name}");
        if !wanted(&bicgstab) && !wanted(&fgmres) {
            continue;
        }
        let choice = MdwfMultigridParams {
            smoother,
            ..MdwfMultigridParams::note([block; ND])
        };
        let start = Instant::now();
        let mut mg = mdwf_multigrid(&op, &lat, &choice)?;
        println!(
            "setup, smoother {smoother_name}: {:.2} s",
            start.elapsed().as_secs_f64()
        );
        if wanted(&bicgstab) {
            ladder(&bicgstab, &mut |stop, x| {
                on_a(&mut BiCgStab::new(stop), Some(&mut mg), x)
            });
        }
        if wanted(&fgmres) {
            ladder(&fgmres, &mut |stop, x| {
                let restarted = Stop {
                    max_iterations: 20,
                    max_restarts: 200,
                    ..stop
                };
                on_a(&mut Fgmres::new(restarted), Some(&mut mg), x)
            });
        }
    }

    let name = "CGNE, even-odd";
    if wanted(name) {
        // An iteration applies the Schur complement and its adjoint, each two half-lattice
        // hops: two Wilson sweeps.
        ladder(name, &mut |stop, x| {
            let result =
                solve_even_odd(&op, op.even_odd_index().unwrap(), x, &b, |schur, x_e, r| {
                    solve_cgne(&mut Cg::new(stop), schur, x_e, r)
                });
            let sweeps = match &result {
                Ok(info) | Err(SolveError::NotConverged(info)) => 2 * info.iterations,
                Err(_) => 0,
            };
            (result, sweeps)
        });
    }
    Ok(())
}
