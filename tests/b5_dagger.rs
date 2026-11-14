//! D† through D for Möbius fermions with constant b, c. With D = 1 + Y + W B5,
//! B5 = b − cY, the Γ5 properties of Y and W give Γ5 D† Γ5 = 1 + Y + B5 W = B5 D B5⁻¹, so
//! (D†)⁻¹ = Γ5 B5 D⁻¹ B5⁻¹ Γ5. B5 is the site-local part C = d+ + d- Y of an operator with
//! d+ = b, d- = −c (M5 = 3, b − 1, c + 1).

use ruttice::fermion::{FermionOp, Mdwf, MdwfParams, Variant};
use ruttice::gauge::random_gauge;
use ruttice::solver::{solve_cgne, Cg, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, Spinor, Vector, C64};
use std::sync::Arc;

type V = Field<f64, Spinor>;

fn random(lat: &Arc<Lattice>, ls: usize, seed: u64) -> V {
    let flat = random_spinor(lat, ls, seed);
    let mut f = V::with_shape(lat, 24, ls, 1, 1);
    f.as_mut_slice().copy_from_slice(flat.as_slice());
    f
}

fn rel(a: &V, b: &V) -> f64 {
    let mut d = a.zeros_like();
    d.copy_from(a);
    d.axpy(&[C64::new(-1.0, 0.0)], b);
    let (mut x, mut y) = ([0.0], [0.0]);
    d.norm_sqr(&mut x);
    b.norm_sqr(&mut y);
    (x[0] / y[0]).sqrt()
}

#[test]
fn test_b5_dagger_through_d() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 35);
    for mass in [0.03, 0.001] {
        let params = MdwfParams {
            ls: 6,
            m5: 1.4,
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
        op.set_gauge(&u).unwrap();
        let b5 = Mdwf::<f64>::new(&lat, params.b5_site());
        let apply = |v: Variant, x: &V| {
            let mut out = x.zeros_like();
            op.mult(&mut op.work(), &mut out, x, v);
            out
        };
        let g5 = |x: &V| {
            let mut out = x.zeros_like();
            op.mult_gamma5_on(&mut out, x);
            out
        };
        let b5_mul = |x: &V, inverse: bool| {
            let mut out = x.zeros_like();
            match inverse {
                false => b5.mult_site(&mut out, x, mass, false),
                true => b5.mult_site_inv(&mut out, x, mass, false),
            }
            out
        };
        // 1. The identity on a random vector, no solver.
        let x = random(&lat, params.ls, 41);
        let lhs = g5(&apply(Variant::Dagger, &g5(&x)));
        let rhs = b5_mul(&apply(Variant::D, &b5_mul(&x, true)), false);
        let e1 = rel(&lhs, &rhs);
        // 2. D† w = b solved by one solve with D: D y = B5⁻¹ Γ5 b, w = Γ5 B5 y.
        let b = random(&lat, params.ls, 42);
        let src = b5_mul(&g5(&b), true);
        let mut y = b.zeros_like();
        let stop = Stop {
            tolerance_sqr: 1e-28,
            max_iterations: 20000,
            max_restarts: 0,
        };
        let mut d_op = ruttice::fermion::Bound::new(&op, ruttice::fermion::Variant::D);
        let info = solve_cgne(&mut Cg::new(stop), &mut d_op, &mut y, &src).unwrap();
        let w = g5(&b5_mul(&y, false));
        let e2 = rel(&apply(Variant::Dagger, &w), &b);
        eprintln!("m = {mass}: |Γ5 D† Γ5 x − B5 D B5⁻¹ x|/|B5 D B5⁻¹ x| = {e1:.2e}; CGNE {} iterations, |D† w − b|/|b| = {e2:.2e}", info.iterations);
        assert!(e1 < 1e-13, "{e1}");
        assert!(e2 < 1e-12, "{e2}");
    }
}

/// B5 = b − cY is singular only for b = 0 with c = 0 or m = 0 (Y has the eigenvalues ±1, and 0
/// at m = 0): there the D† solves of HMC default to Γ5 M Γ5.
#[test]
fn test_b5_invertible() {
    let p = |b, c, mass| MdwfParams {
        ls: 8,
        m5: 1.8,
        mass,
        b,
        c,
        boundary: [Boundary::Periodic; 4],
    };
    assert!(p(1.5, 0.5, 0.001).b5_invertible());
    assert!(p(1.0, 0.0, 0.0).b5_invertible());
    assert!(p(0.0, 1.0, 0.1).b5_invertible());
    assert!(!p(0.0, 0.0, 0.1).b5_invertible());
    assert!(!p(0.0, 1.0, 0.0).b5_invertible());
}
