//! BiCGStab with the domain-wall multigrid cycle for several smoothers and right-hand
//! sides: is a fixed (Chebyshev) smoother steadier than GMRES(m)?
//!
//! ```text
//! cargo run --release --example mdwf_smoother_compare -- CONFIG L M5 MASS [LS] [BLOCK] [alone]
//! ```
//!
//! The cycle is `mdwf_multigrid` with the settings of the working note (8 random test
//! vectors, reference coarse operator) and the smoother replaced. For each smoother:
//! BiCGStab to |r|^2/|b|^2 < 1e-16 of A y = X† b for right-hand sides 91..95, and
//! FGMRES(20) for the first; iterations, and the applications of A per outer iteration
//! (smoother cost: GMRES(m) m, Chebyshev k-1, per smoothing; two smoothings and a
//! residual per cycle; BiCGStab applies the cycle twice per iteration). Then the Chebyshev
//! polynomial alone as the preconditioner, without the cycle, for comparison: does the
//! cycle add anything to its smoother? `alone` runs only this part.

use ruttice::fermion::{FermionOp, Mdwf, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    mdwf_multigrid, MdwfMultigridParams, MdwfSmoother,
};
use ruttice::solver::{largest_magnitude, BiCgStab, Chebyshev, Fgmres, SolveError, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_smoother_compare CONFIG L M5 MASS [LS] [BLOCK] [alone]";

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
    let ls: usize = args
        .get(4)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let block: usize = args
        .get(5)
        .map_or(Ok(3), |s| s.parse())
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
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4");

    let source = |seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, 1, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, seed).as_slice());
        v
    };
    let physical = |x: &V, b: &V| {
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        left[0] / size[0]
    };
    let stop = Stop {
        tolerance_sqr: 1e-16,
        max_iterations: 1500,
        max_restarts: 1,
    };

    let smoothers = [
        ("GMRES(4)", MdwfSmoother::Gmres(Stop::simple(1e-6, 4)), 4),
        (
            "Chebyshev 5, ratio 10",
            MdwfSmoother::Chebyshev {
                steps: 5,
                ratio: 10.0,
            },
            4,
        ),
        (
            "Chebyshev 5, ratio 30",
            MdwfSmoother::Chebyshev {
                steps: 5,
                ratio: 30.0,
            },
            4,
        ),
        (
            "Chebyshev 9, ratio 30",
            MdwfSmoother::Chebyshev {
                steps: 9,
                ratio: 30.0,
            },
            8,
        ),
        (
            "Chebyshev 9, ratio 100",
            MdwfSmoother::Chebyshev {
                steps: 9,
                ratio: 100.0,
            },
            8,
        ),
    ];
    println!("smoother                 A per cycle  BiCGStab, seeds 91..95              FGMRES(20), seed 91");
    let alone = args.get(6).map(String::as_str) == Some("alone");
    for (name, smoother, per_smoothing) in smoothers.into_iter().filter(|_| !alone) {
        let start = Instant::now();
        let choice = MdwfMultigridParams {
            smoother,
            ..MdwfMultigridParams::note([block; ND])
        };
        let mut mg = mdwf_multigrid(&op, &lat, &choice)?;
        let per_cycle = 2 * per_smoothing + 1;
        let mut counts = Vec::new();
        let mut worst: f64 = 0.0;
        for seed in 91..=95 {
            let b = source(seed);
            let mut a = PvPreconditioned::new(&op, &b);
            let mut rhs = b.zeros_like();
            a.right_hand_side(&mut rhs, &b);
            let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
            let result = BiCgStab::new(stop).solve(&mut a, Some(&mut mg), &mut y, &rhs);
            a.solution(&mut x, &y);
            counts.push(match result {
                Ok(i) => format!("{}", i.iterations),
                Err(SolveError::NotConverged(i)) => format!("{}*", i.iterations),
                Err(SolveError::Breakdown { iterations, .. }) => format!("{iterations}!"),
                Err(e) => format!("({e})"),
            });
            worst = worst.max(physical(&x, &b));
        }
        let b = source(91);
        let mut a = PvPreconditioned::new(&op, &b);
        let mut rhs = b.zeros_like();
        a.right_hand_side(&mut rhs, &b);
        let mut y = b.zeros_like();
        let fgmres = match Fgmres::new(Stop {
            max_iterations: 20,
            max_restarts: 100,
            ..stop
        })
        .solve(&mut a, Some(&mut mg), &mut y, &rhs)
        {
            Ok(i) => format!("{}", i.iterations),
            Err(SolveError::NotConverged(i)) => format!("{}*", i.iterations),
            Err(e) => format!("({e})"),
        };
        println!(
            "{name:<24} {per_cycle:>5}        {:<20} worst |Dx-b|^2 {worst:.0e}   {fgmres:>5}   ({:.0} s)",
            counts.join(" "),
            start.elapsed().as_secs_f64()
        );
    }

    // The Chebyshev polynomial alone, hi = 1.1 |λ_max(A)| as in the cycle.
    let hi = 1.1
        * largest_magnitude(
            &mut PvPreconditioned::new(&op, &source(1)),
            &mut source(0x70B),
            30,
        );
    println!("Chebyshev alone (no cycle), hi = {hi:.3}; A per application = steps - 1");
    for (steps, ratio) in [
        (9, 100.0),
        (9, 300.0),
        (17, 300.0),
        (17, 1000.0),
        (33, 1000.0),
    ] {
        let start = Instant::now();
        let mut counts = Vec::new();
        let mut worst: f64 = 0.0;
        for seed in 91..=95 {
            let b = source(seed);
            let mut cheb =
                Chebyshev::new(PvPreconditioned::new(&op, &b), steps, hi / ratio, hi, &b);
            let mut a = PvPreconditioned::new(&op, &b);
            let mut rhs = b.zeros_like();
            a.right_hand_side(&mut rhs, &b);
            let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
            let result = BiCgStab::new(stop).solve(&mut a, Some(&mut cheb), &mut y, &rhs);
            a.solution(&mut x, &y);
            counts.push(match result {
                Ok(i) => format!("{}", i.iterations),
                Err(SolveError::NotConverged(i)) => format!("{}*", i.iterations),
                Err(SolveError::Breakdown { iterations, .. }) => format!("{iterations}!"),
                Err(e) => format!("({e})"),
            });
            worst = worst.max(physical(&x, &b));
        }
        println!(
            "Chebyshev {steps}, ratio {ratio:<6}  {:>5}        {:<20} worst |Dx-b|^2 {worst:.0e}   ({:.0} s)",
            steps - 1,
            counts.join(" "),
            start.elapsed().as_secs_f64()
        );
    }
    println!("* not converged, ! breakdown");
    Ok(())
}
