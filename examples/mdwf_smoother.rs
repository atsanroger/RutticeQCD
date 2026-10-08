//! How good each candidate preconditioner M of A is on its own: for a vector v, the
//! residual it leaves, |v - A M v| / |v| (above one: M makes things worse), and
//! Re<v, A M v> / |v|^2 (negative: A M is not positive in that direction).
//!
//! ```text
//! cargo run --release --example mdwf_smoother -- CONFIG L M5 MASS [LS] [BLOCK]
//! ```
//!
//! Two vectors: a random one, and a smooth one, the solution of A x = P b for a random b,
//! normalised, which is dominated by the low modes of A.

use ruttice::fermion::{FermionOp, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    mdwf_multigrid, MdwfMultigridParams, MdwfSmoother,
};
use ruttice::solver::{BiCgStab, Fgmres, Polynomial, Solve, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_smoother CONFIG L M5 MASS [LS] [BLOCK]";

type V = Field<f64, Spinor>;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args.get(4).map_or(Ok(8), |s| s.parse()).map_err(|_| USAGE)?;
    let block: usize = args.get(5).map_or(Ok(3), |s| s.parse()).map_err(|_| USAGE)?;

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

    let random = |seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, 1, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, seed).as_slice());
        v
    };
    let normalise = |v: &mut V| {
        let mut n = [0.0];
        v.norm_sqr(&mut n);
        v.scale(&[1.0 / n[0].sqrt()]);
    };
    let mut rough = random(301);
    normalise(&mut rough);
    let mut smooth = rough.zeros_like();
    {
        let b = random(302);
        let mut rhs = b.zeros_like();
        let mut a = PvPreconditioned::new(&op, &b);
        a.precondition(&mut rhs, &b);
        let _ = BiCgStab::new(Stop::simple(1e-12, 2000)).solve(&mut a, None, &mut smooth, &rhs);
        normalise(&mut smooth);
    }

    let mut a = PvPreconditioned::new(&op, &rough);
    let mut measure = |name: &str, m: &mut dyn LinearFn<V>| {
        let mut line = format!("{name:<28}");
        for (which, v) in [("random", &rough), ("smooth", &smooth)] {
            let (mut mv, mut amv) = (v.zeros_like(), v.zeros_like());
            m.mult(&mut mv, v);
            a.mult(&mut amv, &mv);
            let mut along = [C64::ZERO];
            v.dot(&amv, &mut along);
            amv.aypx(&[C64::new(-1.0, 0.0)], v);
            let mut left = [0.0];
            amv.norm_sqr(&mut left);
            line += &format!(
                "  {which}: |v-AMv| {:7.3}  Re<v,AMv> {:+7.3}",
                left[0].sqrt(),
                along[0].re
            );
        }
        println!("{line}");
    };

    measure("identity", &mut Identity);
    let sample = random(7);
    for (degree, passes) in [(2, 1), (4, 1), (4, 2), (6, 1), (8, 1)] {
        match Polynomial::fit(PvPreconditioned::new(&op, &sample), &sample, degree, passes) {
            Ok(mut p) => measure(&format!("polynomial {degree} x {passes}"), &mut p),
            Err(_) => println!("polynomial {degree} x {passes}: the powers are dependent"),
        }
    }
    for m in [4, 8] {
        let mut gmres = Solve::new(
            Fgmres::new(Stop::simple(0.0, m)),
            PvPreconditioned::new(&op, &sample),
        );
        measure(&format!("GMRES({m})"), &mut gmres);
    }
    let smoothers = [
        ("polynomial 4", MdwfSmoother::Polynomial { degree: 4, passes: 1 }),
        ("GMRES(4)", MdwfSmoother::Gmres(Stop::simple(1e-6, 4))),
    ];
    for (name, smoother) in smoothers {
        let choice = MdwfMultigridParams {
            smoother,
            ..MdwfMultigridParams::note([block; ND])
        };
        let mut mg = mdwf_multigrid(&op, &lat, &choice)?;
        measure(&format!("cycle, {name}"), &mut mg);
        let exact = MdwfMultigridParams {
            coarse: Stop {
                tolerance_sqr: 1e-24,
                max_iterations: 50,
                max_restarts: 40,
            },
            ..choice
        };
        let mut mg = mdwf_multigrid(&op, &lat, &exact)?;
        measure(&format!("cycle, {name}, exact coarse"), &mut mg);
    }
    Ok(())
}

/// M = 1: what A alone leaves.
struct Identity;

impl LinearFn<V> for Identity {
    fn mult(&mut self, out: &mut V, inp: &V) {
        out.copy_from(inp);
    }
}
