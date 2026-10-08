//! Is the coarse correction of the domain-wall multigrid sound, and does it reach a
//! low mode? Galerkin (restriction P†) against Petrov-Galerkin (restriction Q† from the
//! left low modes of A).
//!
//! ```text
//! cargo run --release --example mdwf_coarse_check -- CONFIG L M5 MASS [LS] [BLOCK]
//! ```
//!
//! A = (D_PV C_PV^-1)† C^-1 D is not Γ5-Hermitian: for a low mode A v = σ u the residual
//! direction u is not Γ5 v, so a restriction built from the right low modes need not see
//! the residual of an error it can represent. Here
//! - P is spanned by right low modes, inverse iteration A x = v to 1e-6;
//! - Q is spanned by left low modes, inverse iteration A† y = w to 1e-6 (or Q = P);
//! - s is a smooth error, the solution of A x = v to 1e-12, normalised, and r = A s.
//!
//! Printed for 8 and 16 test vectors: |P P† s| / |s| (the error the coarse space can
//! represent), |Q Q† r| / |r| (the residual the restriction sees), and what the coarse
//! correction P (Q† A P)^-1 Q† A leaves of an error e: e = P c (zero up to the couplings
//! probing leaves out) and e = s, with the coarse operator probed and cut down to nearest
//! neighbours as the multigrid builds it, and applied exactly through the fine lattice.
//! The coarse system is solved to 1e-20. The floor for s, sqrt(1 - |PP†s|^2/|s|^2), is
//! what a correction that removed the whole represented part would leave.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp};
use ruttice::solver::{BiCgStab, Fgmres, Solver, Stop};
use ruttice::{
    random_spinor, Adjoint, Boundary, Coarse, Field, Lattice, LinearFn, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_coarse_check CONFIG L M5 MASS [LS] [BLOCK]";

type V = Field<f64, Spinor>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

/// The coarse operator Q† A P applied exactly, through the fine lattice: no coupling is
/// left out.
struct Exact<'a, 'b> {
    a: PvPreconditioned<'a, f64>,
    p: &'b Basis<f64>,
    q: &'b Basis<f64>,
    fine: V,
    a_fine: V,
}

impl LinearFn<Field<f64, Coarse>> for Exact<'_, '_> {
    fn mult(&mut self, out: &mut Field<f64, Coarse>, inp: &Field<f64, Coarse>) {
        self.p.prolong(&mut self.fine, inp);
        self.a.mult(&mut self.a_fine, &self.fine);
        self.q.restrict(out, &self.a_fine);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
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

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let like = random(1, 1);
    // x = A^-1 v, or A†^-1 v, to |r|^2/|b|^2 < tol.
    let inverse = |v: &V, adjoint: bool, tol: f64| {
        let mut x = v.zeros_like();
        let stop = Stop::simple(tol, 4000);
        let a = PvPreconditioned::new(&op, v);
        let _ = if adjoint {
            BiCgStab::new(stop).solve(&mut Adjoint(a), None, &mut x, v)
        } else {
            let mut a = a;
            BiCgStab::new(stop).solve(&mut a, None, &mut x, v)
        };
        x
    };
    let mut a = PvPreconditioned::new(&op, &like);
    let mut smooth = inverse(&random(1, 302), false, 1e-12);
    smooth.scale(&[1.0 / norm(&smooth)]);
    let mut residual = smooth.zeros_like();
    a.mult(&mut residual, &smooth);
    residual.scale(&[1.0 / norm(&residual)]);

    let agg = Aggregation::new(&lat, [block; ND]);
    let basis = |x: &V| -> Result<Basis<f64>, Box<dyn Error>> {
        let mut g5x = x.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5x, x);
        Ok(Basis::from_vectors(&agg, x, &g5x)?)
    };

    for nvec in [8, 16] {
        let right = inverse(&random(nvec, 11), false, 1e-6);
        let left = inverse(&random(nvec, 12), true, 1e-6);
        let p = basis(&right)?;
        let q = basis(&left)?;
        for (name, restriction) in [("Galerkin, Q = P", &p), ("Petrov-Galerkin, Q from A†", &q)] {
            let seen = |b: &Basis<f64>, v: &V| {
                let mut c = b.coarse_field(1);
                b.restrict(&mut c, v);
                let mut back = v.zeros_like();
                b.prolong(&mut back, &c);
                norm(&back) / norm(v)
            };
            let mut probed = CoarseOp::probe(&mut a, &p, restriction, false)?;
            let mut exact = Exact {
                a: PvPreconditioned::new(&op, &like),
                p: &p,
                q: restriction,
                fine: like.zeros_like(),
                a_fine: like.zeros_like(),
            };
            let leaves = |e: &V,
                          coarse: &mut dyn LinearFn<Field<f64, Coarse>>,
                          a: &mut PvPreconditioned<'_, f64>| {
                let mut r = e.zeros_like();
                a.mult(&mut r, e);
                let mut rc = restriction.coarse_field(1);
                restriction.restrict(&mut rc, &r);
                let mut y = rc.zeros_like();
                let _ = Fgmres::new(Stop {
                    tolerance_sqr: 1e-20,
                    max_iterations: 100,
                    max_restarts: 50,
                })
                .solve(coarse, None, &mut y, &rc);
                let mut back = e.zeros_like();
                p.prolong(&mut back, &y);
                back.aypx(&[C64::new(-1.0, 0.0)], e);
                norm(&back) / norm(e)
            };
            let mut c = p.coarse_field(1);
            for (i, x) in c.as_mut_slice().iter_mut().enumerate() {
                *x = ((i * 7 + 3) as f64 * 0.37).sin();
            }
            let mut inside = like.zeros_like();
            p.prolong(&mut inside, &c);
            let in_space = seen(&p, &smooth);
            println!(
                "{nvec:2} vectors, {name:<28} |PP†s|/|s| {in_space:.3}  |QQ†r|/|r| {:.3}  \
                 floor for s {:.3}",
                seen(restriction, &residual),
                (1.0 - in_space * in_space).sqrt()
            );
            for (which, coarse) in [
                (
                    "probed (nearest neighbours)",
                    &mut probed as &mut dyn LinearFn<Field<f64, Coarse>>,
                ),
                ("exact (P†A P every time)", &mut exact),
            ] {
                let on_inside = leaves(&inside, coarse, &mut a);
                let on_smooth = leaves(&smooth, coarse, &mut a);
                println!("      coarse operator {which:<28} left of P c {on_inside:.3}  left of s {on_smooth:.3}");
            }
        }
    }
    Ok(())
}
