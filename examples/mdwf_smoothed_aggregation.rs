//! Why the coarse correction of the domain-wall multigrid misses a smooth error, and
//! whether smoothing the prolongation (smoothed aggregation) helps.
//!
//! ```text
//! cargo run --release --example mdwf_smoothed_aggregation -- CONFIG L M5 MASS [LS] [BLOCK] [NVEC]
//! ```
//!
//! P is built from NVEC (8) test vectors, inverse iteration A x = v to 1e-6; s is a smooth
//! error, the solution of A x = v to 1e-12, normalised.
//!
//! 1. The guess to test: the part of s that P can represent, P P† s, is cut at the block
//!    boundaries and carries high modes, so A P P† s is not small. Printed:
//!    |A P P† s| / |P P† s| against |A s| / |s|, and the same with s replaced by the
//!    component of s orthogonal to span(P).
//! 2. Smoothed aggregation: P_ω = (1 - ω A) P, restriction P_ω†, coarse operator
//!    P_ω† A P_ω applied exactly through the fine lattice and solved to 1e-20. For ω = 0
//!    (plain aggregation) and a few ω in units of 1/|λ_max(A)| (power iteration), what the
//!    correction leaves of s, |s - P_ω (P_ω† A P_ω)^-1 P_ω† A s| / |s|, and of an error
//!    inside span(P_ω).

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{BiCgStab, Fgmres, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, LinearOp, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_smoothed_aggregation CONFIG L M5 MASS [LS] [BLOCK] [NVEC]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

/// The prolongation P_ω = (1 - ω A) P and its adjoint P_ω† = P† (1 - ω* A†).
struct Smoothed<'a, 'b> {
    a: PvPreconditioned<'a, f64>,
    p: &'b Basis<f64>,
    omega: f64,
    t: V,
    u: V,
}

impl Smoothed<'_, '_> {
    fn prolong(&mut self, out: &mut V, c: &C) {
        self.p.prolong(&mut self.t, c);
        out.copy_from(&self.t);
        if self.omega != 0.0 {
            self.a.mult(&mut self.u, &self.t);
            out.axpy(&[C64::new(-self.omega, 0.0)], &self.u);
        }
    }

    fn restrict(&mut self, c: &mut C, v: &V) {
        self.t.copy_from(v);
        if self.omega != 0.0 {
            self.a.mult_dag(&mut self.u, v);
            self.t.axpy(&[C64::new(-self.omega, 0.0)], &self.u);
        }
        self.p.restrict(c, &self.t);
    }
}

/// The coarse operator P_ω† A P_ω, applied through the fine lattice.
struct Galerkin<'a, 'b> {
    s: Smoothed<'a, 'b>,
    a: PvPreconditioned<'a, f64>,
    fine: V,
    a_fine: V,
}

impl LinearFn<C> for Galerkin<'_, '_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.s.prolong(&mut self.fine, inp);
        self.a.mult(&mut self.a_fine, &self.fine);
        self.s.restrict(out, &self.a_fine);
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
    let nvec: usize = args
        .get(6)
        .map_or(Ok(8), |s| s.parse())
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
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4, {nvec} test vectors");

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let like = random(1, 1);
    let inverse = |v: &V, tol: f64| {
        let mut x = v.zeros_like();
        let mut a = PvPreconditioned::new(&op, v);
        let _ = BiCgStab::new(Stop::simple(tol, 4000)).solve(&mut a, None, &mut x, v);
        x
    };
    let mut a = PvPreconditioned::new(&op, &like);
    let mut smooth = inverse(&random(1, 302), 1e-12);
    smooth.scale(&[1.0 / norm(&smooth)]);

    // |λ_max(A)| by power iteration.
    let lambda_max = {
        let mut v = random(1, 5);
        let mut w = v.zeros_like();
        let mut estimate = 0.0;
        for _ in 0..50 {
            v.scale(&[1.0 / norm(&v)]);
            a.mult(&mut w, &v);
            estimate = norm(&w);
            std::mem::swap(&mut v, &mut w);
        }
        estimate
    };
    println!("|lambda_max(A)| ~ {lambda_max:.3} (power iteration, 50 steps)");

    let agg = Aggregation::new(&lat, [block; ND]);
    let vectors = inverse(&random(nvec, 11), 1e-6);
    let mut g5 = vectors.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
    let p = Basis::from_vectors(&agg, &vectors, &g5)?;

    // 1. How rough is the represented part of s?
    let applied = |a: &mut PvPreconditioned<'_, f64>, v: &V| {
        let mut w = v.zeros_like();
        a.mult(&mut w, v);
        norm(&w) / norm(v)
    };
    let mut c = p.coarse_field(1);
    p.restrict(&mut c, &smooth);
    let mut inside = like.zeros_like();
    p.prolong(&mut inside, &c);
    let mut outside = like.zeros_like();
    outside.copy_from(&smooth);
    outside.axpy(&[C64::new(-1.0, 0.0)], &inside);
    println!(
        "|PP†s|/|s| {:.3};  |A v|/|v| for v = s: {:.4},  PP†s: {:.4},  s - PP†s: {:.4},  a random vector: {:.4}",
        norm(&inside),
        applied(&mut a, &smooth),
        applied(&mut a, &inside),
        applied(&mut a, &outside),
        applied(&mut a, &random(1, 77))
    );

    // 2. Smoothed aggregation.
    let mut a_s = smooth.zeros_like();
    a.mult(&mut a_s, &smooth);
    for omega_units in [0.0, 0.25, 0.5, 1.0] {
        let omega = omega_units / lambda_max;
        let smoothed = || Smoothed {
            a: PvPreconditioned::new(&op, &like),
            p: &p,
            omega,
            t: like.zeros_like(),
            u: like.zeros_like(),
        };
        let mut s = smoothed();
        let mut coarse = Galerkin {
            s: smoothed(),
            a: PvPreconditioned::new(&op, &like),
            fine: like.zeros_like(),
            a_fine: like.zeros_like(),
        };
        let leaves = |e: &V, a_e: &V, coarse: &mut Galerkin<'_, '_>, s: &mut Smoothed<'_, '_>| {
            let mut rc = p.coarse_field(1);
            s.restrict(&mut rc, a_e);
            let mut y = rc.zeros_like();
            let _ = Fgmres::new(Stop {
                tolerance_sqr: 1e-20,
                max_iterations: 100,
                max_restarts: 50,
            })
            .solve(coarse, None, &mut y, &rc);
            let mut back = e.zeros_like();
            s.prolong(&mut back, &y);
            back.aypx(&[C64::new(-1.0, 0.0)], e);
            norm(&back) / norm(e)
        };
        let mut c = p.coarse_field(1);
        for (i, x) in c.as_mut_slice().iter_mut().enumerate() {
            *x = ((i * 7 + 3) as f64 * 0.37).sin();
        }
        let mut e = like.zeros_like();
        s.prolong(&mut e, &c);
        let mut a_e = e.zeros_like();
        a.mult(&mut a_e, &e);
        let on_inside = leaves(&e, &a_e, &mut coarse, &mut s);
        let on_smooth = leaves(&smooth, &a_s, &mut coarse, &mut s);
        println!(
            "omega = {omega_units:.2} / |lambda_max|:  left of P_w c {on_inside:.3}  left of s {on_smooth:.3}"
        );
    }
    Ok(())
}
