//! Checks: the Möbius domain-wall operator.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::fermion::{FermionOp, Mdwf, MdwfParams, Variant};
use ruttice::gauge::random_gauge;
use ruttice::io::read_bridge_text;
use ruttice::{
    assemble, local_part, random_spinor, Boundary, Field, Gauge, Lattice, LocalComm, Spinor,
    Vector, C64, ND,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;

const BOUNDARY: [Boundary; ND] = [
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];

const MOBIUS: MdwfParams = MdwfParams {
    ls: 6,
    m5: 1.4,
    mass: 0.03,
    b: 1.5,
    c: 0.5,
    boundary: BOUNDARY,
};

fn operator(lat: &Arc<Lattice>, params: MdwfParams, u: &Field<f64, Gauge>) -> Mdwf<f64> {
    let mut op = Mdwf::new(lat, params);
    op.set_gauge(u).unwrap();
    op
}

/// A random field with `ls` spinors per site in each of `num_vectors` vectors.
fn random(lat: &Arc<Lattice>, ls: usize, num_vectors: usize, seed: u64) -> V {
    let flat = random_spinor(lat, ls * num_vectors, seed);
    let mut f = V::with_shape(lat, 24, ls, num_vectors, 1);
    f.as_mut_slice().copy_from_slice(flat.as_slice());
    f
}

fn mult(op: &Mdwf<f64>, inp: &V, variant: Variant) -> V {
    let mut out = inp.zeros_like();
    op.mult(&mut op.work(), &mut out, inp, variant);
    out
}

fn dot(a: &V, b: &V) -> C64 {
    let mut d = vec![C64::ZERO; a.num_vectors()];
    a.dot(b, &mut d);
    d[0]
}

/// The Bridge++ test `Mult_Domainwall`: Shamir's operator on a field of ones, on the
/// Bridge++ test configuration. The reference is the norm Bridge++ expects.
#[test]
fn bridge_mult_domainwall() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = read_bridge_text(&path, &lat).unwrap();
    let params = MdwfParams {
        ls: 8,
        m5: 1.6,
        mass: 0.1,
        b: 1.0,
        c: 0.0,
        boundary: BOUNDARY,
    };
    let op = operator(&lat, params, &u);
    let mut ones = V::with_shape(&lat, 24, 8, 1, 1);
    ones.as_mut_slice().fill(1.0);
    let out = mult(&op, &ones, Variant::D);
    let mut norm = [0.0];
    out.norm_sqr(&mut norm);
    let (ours, reference) = (norm[0].sqrt(), 1.023_523_203_771_08e3);
    assert!(
        (ours - reference).abs() < 1e-12 * reference,
        "ours {ours:.15e}, Bridge++ {reference:.15e}"
    );
}

/// D† is the adjoint of D, and Γ5 D is Hermitian for Shamir's operator; with the Möbius
/// coefficients it is D† that is checked, Γ5 D is not Hermitian there.
#[test]
fn dagger_is_the_adjoint() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let u = random_gauge(&lat, 41);
    for params in [
        MOBIUS,
        MdwfParams {
            b: 1.0,
            c: 0.0,
            ..MOBIUS
        },
    ] {
        let op = operator(&lat, params, &u);
        let (x, y) = (
            random(&lat, params.ls, 1, 42),
            random(&lat, params.ls, 1, 43),
        );
        let scale = dot(&x, &x).re;
        // <x, D y> = <D† x, y>
        let (lhs, rhs) = (
            dot(&x, &mult(&op, &y, Variant::D)),
            dot(&mult(&op, &x, Variant::Dagger), &y),
        );
        assert!(
            (lhs - rhs).norm_sqr().sqrt() < 1e-13 * scale,
            "c = {}: {lhs:?} vs {rhs:?}",
            params.c
        );
        // <x, Γ5 D y> = conj <y, Γ5 D x> for c = 0
        let (lhs, rhs) = (
            dot(&x, &mult(&op, &y, Variant::Gamma5D)),
            dot(&y, &mult(&op, &x, Variant::Gamma5D)).conj(),
        );
        let hermitian = (lhs - rhs).norm_sqr().sqrt() < 1e-13 * scale;
        assert_eq!(hermitian, params.c == 0.0, "c = {}", params.c);
    }
}

/// Two vectors in one field give what each gives by itself.
#[test]
fn several_vectors_are_independent() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 51));
    let both = random(&lat, MOBIUS.ls, 2, 52);
    let out = mult(&op, &both, Variant::D);
    for j in 0..2 {
        let single = both.select(&[j]);
        let alone = mult(&op, &single, Variant::D);
        assert!(
            out.select(&[j]).as_slice() == alone.as_slice(),
            "vector {j}"
        );
    }
}

/// On several processes the operator gives the bits it gives on one.
#[test]
fn several_processes_give_the_same_bits() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 61);
    let psi = random(&whole, MOBIUS.ls, 2, 62);
    for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
        let want = mult(&operator(&whole, MOBIUS, &u), &psi, variant);
        for grid in [[1, 1, 2, 2], [2, 2, 1, 2]] {
            let parts = LocalComm::run(global, grid, |lat| {
                let op = operator(&lat, MOBIUS, &local_part(&u, &lat));
                mult(&op, &local_part(&psi, &lat), variant)
            });
            assert!(
                assemble(&whole, &parts).as_slice() == want.as_slice(),
                "{variant:?}, grid {grid:?}"
            );
        }
    }
}
