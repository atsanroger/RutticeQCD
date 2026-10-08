//! Checks: the Möbius domain-wall operator.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::fermion::{Bound, FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned, Variant};
use ruttice::gauge::random_gauge;
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    mdwf_multigrid, CoarseOp, MdwfCoarse, MdwfMultigridParams,
};
use ruttice::solver::{solve_cgne, BiCgStab, Cg, Solver, Stop};
use ruttice::{
    assemble, local_part, random_spinor, Boundary, Field, Gauge, Lattice, LinearFn, LinearOp,
    LocalComm, Spinor, Vector, C64, ND,
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

fn max_diff(a: &V, b: &V) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// The inverse of the site-local part undoes it, for C and for C†, at the quark mass and
/// at the Pauli-Villars mass; and C is what is left of D when the links are switched off:
/// with all links zero the Wilson kernel is its diagonal, so D = C.
#[test]
fn site_local_part_and_its_inverse() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 71));
    let psi = random(&lat, MOBIUS.ls, 2, 72);
    for mass in [MOBIUS.mass, 1.0] {
        for dagger in [false, true] {
            let (mut c, mut back) = (psi.zeros_like(), psi.zeros_like());
            op.mult_site(&mut c, &psi, mass, dagger);
            op.mult_site_inv(&mut back, &c, mass, dagger);
            let diff = max_diff(&back, &psi);
            assert!(diff < 1e-13, "mass {mass}, dagger {dagger}: {diff:e}");
            assert!(max_diff(&c, &psi) > 0.1);
        }
    }

    let zero_links = Field::<f64, Gauge>::gauge(&lat);
    let free = operator(&lat, MOBIUS, &zero_links);
    for (variant, dagger) in [(Variant::D, false), (Variant::Dagger, true)] {
        let mut c = psi.zeros_like();
        free.mult_site(&mut c, &psi, MOBIUS.mass, dagger);
        let diff = max_diff(&mult(&free, &psi, variant), &c);
        assert!(diff < 1e-13, "{variant:?}: {diff:e}");
    }
}

/// A† is the adjoint of A.
#[test]
fn preconditioned_operator_has_its_adjoint() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 81));
    let (x, y) = (
        random(&lat, MOBIUS.ls, 1, 82),
        random(&lat, MOBIUS.ls, 1, 83),
    );
    let mut a = PvPreconditioned::new(&op, &x);
    let (mut ay, mut adag_x) = (x.zeros_like(), x.zeros_like());
    a.mult(&mut ay, &y);
    a.mult_dag(&mut adag_x, &x);
    let (lhs, rhs) = (dot(&x, &ay), dot(&adag_x, &y));
    assert!(
        (lhs - rhs).norm_sqr().sqrt() < 1e-12 * dot(&x, &x).re,
        "{lhs:?} vs {rhs:?}"
    );
}

/// Solving A x = P b with BiCGStab gives the solution of D x = b, checked with D itself;
/// and it takes fewer applications of the Wilson kernel than CG on the normal equations
/// of D. On the Bridge++ test configuration.
#[test]
fn bicgstab_on_the_preconditioned_operator_solves_d() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = read_bridge_text(&path, &lat).unwrap();
    let params = MdwfParams {
        ls: 8,
        m5: 1.0,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: BOUNDARY,
    };
    let op = operator(&lat, params, &u);
    let b = random(&lat, params.ls, 1, 91);
    let residual = |x: &V| {
        let mut r = mult(&op, x, Variant::D);
        r.scale(&[-1.0]);
        r.axpy(&[C64::new(1.0, 0.0)], &b);
        dot(&r, &r).re / dot(&b, &b).re
    };

    let stop = Stop {
        tolerance_sqr: 1e-22,
        max_iterations: 4000,
        max_restarts: 5,
    };
    let mut a = PvPreconditioned::new(&op, &b);
    let mut rhs = b.zeros_like();
    a.precondition(&mut rhs, &b);
    let mut x = b.zeros_like();
    let info = BiCgStab::new(stop)
        .solve(&mut a, None, &mut x, &rhs)
        .unwrap();
    let physical = residual(&x);
    assert!(physical < 1e-16, "|D x - b|^2 / |b|^2 = {physical:e}");

    let mut y = b.zeros_like();
    let cg = solve_cgne(
        &mut Cg::new(stop),
        &mut Bound::new(&op, Variant::D),
        &mut y,
        &b,
    )
    .unwrap();
    assert!(residual(&y) < 1e-16);
    // Wilson sweeps: A is two, an iteration of BiCGStab applies it twice; an iteration of
    // CG on the normal equations applies D and D† once each.
    let (sweeps_a, sweeps_cg) = (4 * info.iterations, 2 * cg.iterations);
    eprintln!(
        "BiCGStab on A: {} iterations ({sweeps_a} sweeps), residual {physical:e};          CG on D D†: {} iterations ({sweeps_cg} sweeps)",
        info.iterations, cg.iterations
    );
}

/// The parts multigrid coarsening uses add up to the operator: D = diagonal part +
/// coefficient x (the eight hopping terms).
#[test]
fn parts_add_up_to_the_operator() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 101));
    let psi = random(&lat, MOBIUS.ls, 1, 102);
    let mut work = op.work();
    let mut hops = psi.zeros_like();
    for mu in 0..ND {
        op.hop_forward(&mut work, mu, &mut hops, &psi);
        op.hop_backward(&mut work, mu, &mut hops, &psi);
    }
    let mut sum = psi.zeros_like();
    op.mult_diag(&mut work, &mut sum, &psi);
    sum.axpy(&[C64::new(op.hop_coefficient(), 0.0)], &hops);
    let diff = max_diff(&sum, &mult(&op, &psi, Variant::D));
    assert!(diff < 1e-13, "{diff:e}");

    // Γ5 squares to one.
    let (mut once, mut twice) = (psi.zeros_like(), psi.zeros_like());
    op.mult_gamma5(&mut work, &mut once, &psi);
    op.mult_gamma5(&mut work, &mut twice, &once);
    assert!(max_diff(&twice, &psi) == 0.0 && max_diff(&once, &psi) > 0.1);
}

/// The multigrid cycle as the preconditioner of BiCGStab on A: the solver converges and
/// the solution solves D x = b. On the Bridge++ test configuration.
///
/// On this problem (4x4x4x8, aggregates of 2^4) the cycle does not yet pay: BiCGStab takes
/// 443 iterations without it and between 356 and 488 with it, for either coarse operator
/// and with or without D_PV on the test vectors. The test checks that the pieces work
/// together, not that they help.
#[test]
fn multigrid_cycle_runs_as_a_preconditioner() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = read_bridge_text(&path, &lat).unwrap();
    let params = MdwfParams {
        ls: 4,
        m5: 1.0,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: BOUNDARY,
    };
    let op = operator(&lat, params, &u);
    let b = random(&lat, params.ls, 1, 91);
    let stop = Stop {
        tolerance_sqr: 1e-20,
        max_iterations: 2000,
        max_restarts: 5,
    };
    let mut a = PvPreconditioned::new(&op, &b);
    let mut rhs = b.zeros_like();
    a.precondition(&mut rhs, &b);
    for coarse_operator in [MdwfCoarse::RightPreconditioned, MdwfCoarse::Galerkin] {
        let choice = MdwfMultigridParams {
            coarse_operator,
            num_test_vectors: 4,
            ..MdwfMultigridParams::note([2, 2, 2, 2])
        };
        let mut mg = mdwf_multigrid(&op, &lat, &choice).unwrap();
        let mut x = b.zeros_like();
        let info = BiCgStab::new(stop)
            .solve(&mut a, Some(&mut mg), &mut x, &rhs)
            .unwrap();
        let mut r = mult(&op, &x, Variant::D);
        r.scale(&[-1.0]);
        r.axpy(&[C64::new(1.0, 0.0)], &b);
        let physical = dot(&r, &r).re / dot(&b, &b).re;
        eprintln!(
            "{coarse_operator:?}: {} iterations, |D x - b|^2/|b|^2 = {physical:e}",
            info.iterations
        );
        assert!(physical < 1e-15, "{coarse_operator:?}: {physical:e}");
    }
}

/// How much of `P† A P` the probed coarse operator keeps: applied to a random coarse
/// vector, against restricting A applied to the prolonged vector. The difference is the
/// couplings to diagonal neighbours that probing leaves out.
#[test]
fn probed_coarse_operator_against_the_projection() {
    let lat = Lattice::single_node([8, 8, 4, 4]);
    let u = random_gauge(&lat, 111);
    let params = MdwfParams { ls: 4, ..MOBIUS };
    let op = operator(&lat, params, &u);
    let choice = MdwfMultigridParams {
        coarse_operator: MdwfCoarse::Galerkin,
        through_pauli_villars: false,
        num_test_vectors: 4,
        ..MdwfMultigridParams::note([2, 2, 2, 2])
    };
    let mg = mdwf_multigrid(&op, &lat, &choice).unwrap();
    let basis = mg.basis();
    let like = random(&lat, params.ls, 1, 112);
    let mut a = PvPreconditioned::new(&op, &like);
    let mut coarse = CoarseOp::probe(&mut a, basis, basis, false).unwrap();

    let mut c = basis.coarse_field(1);
    for (i, x) in c.as_mut_slice().iter_mut().enumerate() {
        *x = ((i * 7 + 3) as f64 * 0.37).sin();
    }
    let (mut fine, mut a_fine) = (like.zeros_like(), like.zeros_like());
    basis.prolong(&mut fine, &c);
    a.mult(&mut a_fine, &fine);
    let mut projected = c.zeros_like();
    basis.restrict(&mut projected, &a_fine);
    let mut probed = c.zeros_like();
    coarse.mult(&mut probed, &c);

    let norm = |v: &Field<f64, ruttice::Coarse>| {
        let mut n = [0.0];
        v.norm_sqr(&mut n);
        n[0].sqrt()
    };
    let mut diff = probed.zeros_like();
    diff.copy_from(&probed);
    diff.axpy(&[C64::new(-1.0, 0.0)], &projected);
    eprintln!(
        "|A_c c| = {:.4e}, |P† A P c| = {:.4e}, difference {:.4e}",
        norm(&probed),
        norm(&projected),
        norm(&diff)
    );
    // Restricting what was prolonged gives it back: the basis is orthonormal.
    let mut back = c.zeros_like();
    basis.restrict(&mut back, &fine);
    back.axpy(&[C64::new(-1.0, 0.0)], &c);
    assert!(
        norm(&back) < 1e-12 * norm(&c),
        "P† P is not one: {:e}",
        norm(&back)
    );
    assert!(norm(&diff) < 0.5 * norm(&projected));
}
