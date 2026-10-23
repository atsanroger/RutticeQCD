//! Checks: the Möbius domain-wall operator.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::fermion::{
    hop_then_of_blocks, inverse_hops_twice_of, solve_even_odd, Bound, Dagger, EoForm, EvenOdd,
    FermionOp, HopTerms, Mdwf, MdwfParams, PvEvenOdd, PvPreconditioned, SchurFactor, Then, Variant,
};
use ruttice::gauge::random_gauge;
use ruttice::io::read_bridge_text;
use ruttice::kernel::{Dirac, Gamma, GammaRepre, Unit};
use ruttice::measure::{meson_correlator, pion_correlator, point_source, NUM_SOURCES};
use ruttice::solver::preconditioner::multigrid::{
    mdwf_multigrid, pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes,
    Aggregation, Basis, BasisBuilder, BootstrapParams, CoarseKrylov, CoarseOp, LowModeParams,
    MdwfCoarse, MdwfMultigridParams, MdwfSmoother, PvEoCycle, PvEoParams, Reflection,
};
use ruttice::solver::{
    largest_magnitude, solve_cgne, BiCgStab, Cg, ChebyshevFilter, Convert, Fgmres, Gcr, Polynomial,
    Solve, Solver, Stop,
};
use ruttice::{
    assemble, extract_parity, insert_parity, local_part, random_spinor, Boundary, Field, Gauge,
    Half, Lattice, LinearFn, LinearOp, LocalComm, Normal, Parity, Spinor, Storage, Vector, C64, ND,
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

fn max_half_diff(a: &Field<f64, Spinor, Half>, b: &Field<f64, Spinor, Half>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0, |m, (x, y)| m.max((x - y).abs()))
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

/// Solving A y = X† b with BiCGStab and x = C^-1 y gives the solution of D x = b, checked with D
/// itself.
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
    a.right_hand_side(&mut rhs, &b);
    let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
    let info = BiCgStab::new(stop)
        .solve(&mut a, None, &mut y, &rhs)
        .unwrap();
    a.solution(&mut x, &y);
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
    a.right_hand_side(&mut rhs, &b);
    for coarse_operator in [MdwfCoarse::RightPreconditioned, MdwfCoarse::Galerkin] {
        let choice = MdwfMultigridParams {
            coarse_operator,
            num_test_vectors: 4,
            ..MdwfMultigridParams::note([2, 2, 2, 2])
        };
        let mut mg = mdwf_multigrid(&op, &lat, &choice).unwrap();
        let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
        let info = BiCgStab::new(stop)
            .solve(&mut a, Some(&mut mg), &mut y, &rhs)
            .unwrap();
        a.solution(&mut x, &y);
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

/// The even-odd blocks put together are D, and those of the adjoint are D†; also in
/// Neff's form D^(α).
#[test]
fn even_odd_blocks_add_up_to_the_operator() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut op = operator(&lat, MOBIUS, &random_gauge(&lat, 31));
    for alpha in [1.0, 0.5] {
        op.set_alpha(alpha);
        even_odd_blocks_of(&lat, &op);
    }
}

/// The fused hops (`hop_then`, `inverse_hops_twice`: Neff's Q inside the sweeps) are the blocks
/// one after the other.
#[test]
fn fused_hops_are_the_blocks() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut op = operator(&lat, MOBIUS, &random_gauge(&lat, 35));
    let index = op.even_odd_index().unwrap().clone();
    let x = random(&lat, MOBIUS.ls, 2, 36);
    let half = |parity: Parity| {
        let mut h = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 2, 1);
        h.set_parity(parity);
        h
    };
    let (mut even, mut odd) = (half(Parity::Even), half(Parity::Odd));
    extract_parity(&mut even, &x, &index, Parity::Even);
    extract_parity(&mut odd, &x, &index, Parity::Odd);
    for alpha in [1.0, 0.6] {
        op.set_alpha(alpha);
        let mut work = op.even_odd_work();
        for dagger in [Dagger::No, Dagger::Yes] {
            for (site_inv, subtract) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let then = || Then {
                    site_inv,
                    from: subtract.then_some(&even),
                };
                let (mut fused, mut blocks, mut scratch) =
                    (half(Parity::Even), half(Parity::Even), half(Parity::Even));
                op.hop_then(
                    &mut work,
                    &mut fused,
                    &odd,
                    Parity::Even,
                    dagger,
                    then(),
                    &mut scratch,
                );
                hop_then_of_blocks(
                    &op,
                    &mut work,
                    &mut blocks,
                    &odd,
                    Parity::Even,
                    dagger,
                    then(),
                    &mut scratch,
                );
                let diff = max_half_diff(&fused, &blocks);
                assert!(
                    diff < 1e-13,
                    "α {alpha} {dagger:?} C⁻¹ {site_inv} from {subtract}: {diff:e}"
                );
            }
            let (mut o1, mut o2, mut e) =
                (half(Parity::Odd), half(Parity::Odd), half(Parity::Even));
            let (mut fused, mut blocks) = (half(Parity::Even), half(Parity::Even));
            op.inverse_hops_twice(
                &mut work,
                (&mut o1, &mut o2, &mut e),
                &mut fused,
                &even,
                dagger,
            );
            inverse_hops_twice_of(
                &op,
                &mut work,
                (&mut o1, &mut o2, &mut e),
                &mut blocks,
                &even,
                dagger,
            );
            let diff = max_half_diff(&fused, &blocks);
            assert!(diff < 1e-13, "α {alpha} {dagger:?} twice: {diff:e}");
        }
    }
}

fn even_odd_blocks_of(lat: &Arc<Lattice>, op: &Mdwf<f64>) {
    let lat = lat.clone();
    let index = op.even_odd_index().unwrap();
    let x = random(&lat, MOBIUS.ls, 2, 32);
    let half = |parity: Parity| {
        let mut h = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 2, 1);
        h.set_parity(parity);
        h
    };
    let mut work = op.even_odd_work();
    for (dagger, variant) in [(Dagger::No, Variant::D), (Dagger::Yes, Variant::Dagger)] {
        let mut blocks = x.zeros_like();
        for to in [Parity::Even, Parity::Odd] {
            let (mut same, mut other) = (half(to), half(to.flip()));
            extract_parity(&mut same, &x, index, to);
            extract_parity(&mut other, &x, index, to.flip());
            let (mut local, mut hopped) = (half(to), half(to));
            op.diag(&mut work, &mut local, &same, to, dagger);
            op.hop(&mut work, &mut hopped, &other, to, dagger);
            local.axpy(&[C64::new(1.0, 0.0); 2], &hopped);
            insert_parity(&mut blocks, &local, index);

            // The inverse of the diagonal block undoes it.
            let mut back = half(to);
            op.diag(&mut work, &mut hopped, &same, to, dagger);
            op.diag_inv(&mut work, &mut back, &hopped, to, dagger);
            back.axpy(&[C64::new(-1.0, 0.0); 2], &same);
            let (mut left, mut size) = ([0.0; 2], [0.0; 2]);
            back.norm_sqr(&mut left);
            same.norm_sqr(&mut size);
            assert!(
                left[0] < 1e-26 * size[0],
                "{dagger:?} {to:?}: {:e}",
                left[0] / size[0]
            );
        }
        let whole = mult(op, &x, variant);
        let diff = max_diff(&blocks, &whole);
        assert!(diff < 1e-13, "α = {}, {variant:?}: {diff:e}", op.alpha());
    }
}

/// Neff's form D^(α) = D Q (arXiv:2602.14454): it is D applied after Q, its adjoint is Q D†, Γ5
/// D^(α) = Γ5 D Q, and its site-local part C Q has the inverse Q⁻¹ C⁻¹.
#[test]
fn neff_alpha_form() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let u = random_gauge(&lat, 91);
    let plain = operator(&lat, MOBIUS, &u);
    let mut op = operator(&lat, MOBIUS, &u);
    let alpha = 0.5;
    op.set_alpha(alpha);
    let (x, y) = (
        random(&lat, MOBIUS.ls, 2, 92),
        random(&lat, MOBIUS.ls, 2, 93),
    );
    let q = |inp: &V, inverse: bool| {
        let mut out = inp.zeros_like();
        op.mult_q(&mut out, inp, inverse);
        out
    };
    assert!(max_diff(&q(&q(&x, false), true), &x) < 1e-14);
    assert!(max_diff(&q(&x, false), &x) > 0.1);
    // Q is Hermitian.
    let (lhs, rhs) = (dot(&y, &q(&x, false)), dot(&q(&y, false), &x));
    assert!((lhs - rhs).norm_sqr().sqrt() < 1e-13 * dot(&x, &x).re);

    let qx = q(&x, false);
    for (variant, want) in [
        (Variant::D, mult(&plain, &qx, Variant::D)),
        (Variant::Gamma5D, mult(&plain, &qx, Variant::Gamma5D)),
        (
            Variant::Dagger,
            q(&mult(&plain, &x, Variant::Dagger), false),
        ),
    ] {
        let diff = max_diff(&mult(&op, &x, variant), &want);
        assert!(diff < 1e-13, "{variant:?}: {diff:e}");
    }
    let scale = dot(&x, &x).re;
    let (lhs, rhs) = (
        dot(&y, &mult(&op, &x, Variant::D)),
        dot(&mult(&op, &y, Variant::Dagger), &x),
    );
    assert!((lhs - rhs).norm_sqr().sqrt() < 1e-13 * scale);

    for mass in [MOBIUS.mass, 1.0] {
        for dagger in [false, true] {
            let (mut c, mut back, mut want) = (x.zeros_like(), x.zeros_like(), x.zeros_like());
            op.mult_site(&mut c, &x, mass, dagger);
            if dagger {
                plain.mult_site(&mut want, &x, mass, true);
                want = q(&want, false);
            } else {
                plain.mult_site(&mut want, &qx, mass, false);
            }
            assert!(max_diff(&c, &want) < 1e-13, "mass {mass}, dagger {dagger}");
            op.mult_site_inv(&mut back, &c, mass, dagger);
            let diff = max_diff(&back, &x);
            assert!(diff < 1e-13, "mass {mass}, dagger {dagger}: {diff:e}");
        }
    }

    let heavy = MdwfParams {
        mass: 0.4,
        ..MOBIUS
    };
    let mut op_heavy = operator(&lat, heavy, &u);
    op_heavy.set_alpha(alpha);
    let plain_heavy = operator(&lat, heavy, &u);
    let difference = |a: &Mdwf<f64>, b: &Mdwf<f64>| {
        let mut d = mult(a, &x, Variant::D);
        d.axpy(&[C64::new(-1.0, 0.0); 2], &mult(b, &x, Variant::D));
        d
    };
    let (with, without) = (difference(&op, &op_heavy), difference(&plain, &plain_heavy));
    let diff = max_diff(&with, &without);
    assert!(diff < 1e-13, "the mass term sees Q: {diff:e}");
    assert!(max_diff(&without, &x.zeros_like()) > 0.01);
}

/// CG on the normal equations of the Schur complement solves D x = b, in fewer iterations
/// than on those of D.
#[test]
fn even_odd_solve_solves_the_full_system() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 33));
    let b = random(&lat, MOBIUS.ls, 2, 34);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 4000,
        max_restarts: 2,
    };
    let mut x = b.zeros_like();
    let info = solve_even_odd(
        &op,
        op.even_odd_index().unwrap(),
        &mut x,
        &b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap();
    let mut r = mult(&op, &x, Variant::D);
    r.scale(&[-1.0; 2]);
    r.axpy(&[C64::new(1.0, 0.0); 2], &b);
    let (mut left, mut size) = ([0.0; 2], [0.0; 2]);
    r.norm_sqr(&mut left);
    b.norm_sqr(&mut size);
    let mut y = b.zeros_like();
    let whole = solve_cgne(
        &mut Cg::new(stop),
        &mut Bound::new(&op, Variant::D),
        &mut y,
        &b,
    )
    .unwrap();
    eprintln!(
        "even-odd: {} iterations, |D x - b|^2/|b|^2 = {:e}; whole lattice: {} iterations",
        info.iterations,
        left[0] / size[0],
        whole.iterations
    );
    assert!(left.iter().zip(&size).all(|(l, s)| l < &(1e-20 * s)));
    assert!(info.iterations < whole.iterations);
}

/// D† x = b through Ã†, the Pauli-Villars operator with the roles of D and D_PV exchanged
/// (`adjoint_right_hand_side`, `adjoint_solution`): GCR on Ã† y = rhs, then x from y, solves
/// D† x = b; and Ã† is the adjoint of Ã.
#[test]
fn adjoint_pv_operator_solves_d_dagger() {
    use ruttice::fermion::{adjoint_right_hand_side, adjoint_solution, PvRatio};
    type H = Field<f64, Spinor, Half>;
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 35);
    let op = operator(&lat, MOBIUS, &u);
    let pv = operator(
        &lat,
        MdwfParams {
            mass: 1.0,
            ..MOBIUS
        },
        &u,
    );
    let index = op.even_odd_index().unwrap().clone();
    let mut even = H::with_shape(&lat, 24, MOBIUS.ls, 1, 1);
    even.set_parity(Parity::Even);
    let mut odd = even.zeros_like();
    odd.set_parity(Parity::Odd);
    let b = random(&lat, MOBIUS.ls, 1, 36);
    let mut rhs = even.zeros_like();
    adjoint_right_hand_side(&op, &*index, &mut rhs, &b);
    let mut a_dag = PvRatio::new(&pv, &op, EoForm::Right, &even, &odd);
    let stop = Stop {
        tolerance_sqr: 1e-26,
        max_iterations: 4000,
        max_restarts: 0,
    };
    let mut y = even.zeros_like();
    let info = Gcr::new(stop, 8)
        .solve(&mut a_dag, None, &mut y, &rhs)
        .unwrap();
    let mut x = b.zeros_like();
    adjoint_solution(&op, &pv, &*index, &mut x, &y, &b);
    let mut r = mult(&op, &x, Variant::Dagger);
    r.aypx(&[C64::new(-1.0, 0.0)], &b);
    let (mut left, mut size) = ([0.0], [0.0]);
    r.norm_sqr(&mut left);
    b.norm_sqr(&mut size);
    eprintln!(
        "Ã†: GCR(8) {} iterations, |D† x - b|²/|b|² = {:e}",
        info.iterations,
        left[0] / size[0]
    );
    assert!(left[0] < 1e-20 * size[0], "{:e}", left[0] / size[0]);

    // <z, Ã w> = <Ã† z, w>.
    let mut a = PvRatio::new(&op, &pv, EoForm::Right, &even, &odd);
    let (mut z, mut w) = (even.zeros_like(), even.zeros_like());
    extract_parity(
        &mut z,
        &random(&lat, MOBIUS.ls, 1, 37),
        &index,
        Parity::Even,
    );
    extract_parity(
        &mut w,
        &random(&lat, MOBIUS.ls, 1, 38),
        &index,
        Parity::Even,
    );
    let (mut aw, mut adz) = (even.zeros_like(), even.zeros_like());
    a.mult(&mut aw, &w);
    a_dag.mult(&mut adz, &z);
    let (mut l, mut rr) = ([C64::ZERO], [C64::ZERO]);
    z.dot(&aw, &mut l);
    adz.dot(&w, &mut rr);
    assert!(
        (l[0] - rr[0]).norm_sqr() < 1e-24 * l[0].norm_sqr(),
        "{l:?} {rr:?}"
    );
}

/// The frozen polynomial of A: it is the minimal-residual polynomial of its sample, so on
/// the sample it leaves what GMRES of the same length leaves; it is the same linear map for
/// every input; further passes reduce the residual further; and BiCGStab preconditioned by
/// a cycle that smooths with it solves D x = b.
#[test]
fn frozen_polynomial_smoother() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 17);
    let params = MdwfParams {
        ls: 4,
        m5: 1.0,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: BOUNDARY,
    };
    let op = operator(&lat, params, &u);
    let sample = random(&lat, params.ls, 1, 5);
    let mut a = PvPreconditioned::new(&op, &sample);
    let mut left = |x: &V, b: &V| {
        let mut r = b.zeros_like();
        a.mult(&mut r, x);
        r.scale(&[-1.0]);
        r.axpy(&[C64::new(1.0, 0.0)], b);
        (dot(&r, &r).re / dot(b, b).re).sqrt()
    };

    let degree = 4;
    let mut once =
        Polynomial::fit(PvPreconditioned::new(&op, &sample), &sample, degree, 1).unwrap();
    let mut gmres = Solve::new(
        Fgmres::new(Stop::simple(0.0, degree)),
        PvPreconditioned::new(&op, &sample),
    );
    let (mut x, mut y) = (sample.zeros_like(), sample.zeros_like());
    once.mult(&mut x, &sample);
    gmres.mult(&mut y, &sample);
    let (polynomial, krylov) = (left(&x, &sample), left(&y, &sample));
    eprintln!("on its sample: polynomial {polynomial:.6}, GMRES({degree}) {krylov:.6}");
    assert!((polynomial - krylov).abs() < 1e-8 * krylov);

    // Linear: p(A)(2 v + i w) = 2 p(A) v + i p(A) w.
    let (v, w) = (random(&lat, params.ls, 1, 6), random(&lat, params.ls, 1, 7));
    let mut combined = v.zeros_like();
    combined.axpy(&[C64::new(2.0, 0.0)], &v);
    combined.axpy(&[C64::new(0.0, 1.0)], &w);
    let (mut pv, mut pw, mut pc) = (v.zeros_like(), v.zeros_like(), v.zeros_like());
    once.mult(&mut pv, &v);
    once.mult(&mut pw, &w);
    once.mult(&mut pc, &combined);
    pc.axpy(&[C64::new(-2.0, 0.0)], &pv);
    pc.axpy(&[C64::new(0.0, -1.0)], &pw);
    assert!(dot(&pc, &pc).re < 1e-24 * dot(&pv, &pv).re);

    // On another vector it cannot beat GMRES, and a second pass leaves less.
    gmres.mult(&mut y, &v);
    let mut twice =
        Polynomial::fit(PvPreconditioned::new(&op, &sample), &sample, degree, 2).unwrap();
    twice.mult(&mut x, &v);
    let (one, two, krylov) = (left(&pv, &v), left(&x, &v), left(&y, &v));
    eprintln!(
        "on another vector: one pass {one:.4}, two passes {two:.4}, GMRES({degree}) {krylov:.4}"
    );
    assert!(krylov <= one * (1.0 + 1e-10) && two < one && one < 1.0);

    let b = random(&lat, params.ls, 1, 91);
    let mut rhs = b.zeros_like();
    let mut a = PvPreconditioned::new(&op, &b);
    a.right_hand_side(&mut rhs, &b);
    let choice = MdwfMultigridParams {
        num_test_vectors: 4,
        smoother: MdwfSmoother::Polynomial { degree, passes: 1 },
        ..MdwfMultigridParams::note([2, 2, 2, 2])
    };
    let mut mg = mdwf_multigrid(&op, &lat, &choice).unwrap();
    let stop = Stop {
        tolerance_sqr: 1e-20,
        max_iterations: 2000,
        max_restarts: 5,
    };
    let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
    let info = BiCgStab::new(stop)
        .solve(&mut a, Some(&mut mg), &mut y, &rhs)
        .unwrap();
    a.solution(&mut x, &y);
    let mut r = mult(&op, &x, Variant::D);
    r.scale(&[-1.0]);
    r.axpy(&[C64::new(1.0, 0.0)], &b);
    let physical = dot(&r, &r).re / dot(&b, &b).re;
    eprintln!(
        "BiCGStab with the cycle: {} iterations, |D x - b|^2/|b|^2 = {physical:e}",
        info.iterations
    );
    assert!(physical < 1e-15, "{physical:e}");
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

/// BiCGStab preconditioned by a cycle that smooths with Chebyshev iteration solves D x = b;
/// the cycle is then a fixed linear map (apart from the coarse solve).
#[test]
fn chebyshev_smoother_in_the_cycle() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 17);
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
    let mut a = PvPreconditioned::new(&op, &b);
    let mut rhs = b.zeros_like();
    a.right_hand_side(&mut rhs, &b);
    let choice = MdwfMultigridParams {
        num_test_vectors: 4,
        smoother: MdwfSmoother::Chebyshev {
            steps: 4,
            ratio: 10.0,
        },
        ..MdwfMultigridParams::note([2, 2, 2, 2])
    };
    let mut mg = mdwf_multigrid(&op, &lat, &choice).unwrap();
    let stop = Stop {
        tolerance_sqr: 1e-20,
        max_iterations: 2000,
        max_restarts: 5,
    };
    let (mut y, mut x) = (b.zeros_like(), b.zeros_like());
    let info = BiCgStab::new(stop)
        .solve(&mut a, Some(&mut mg), &mut y, &rhs)
        .unwrap();
    a.solution(&mut x, &y);
    let mut r = mult(&op, &x, Variant::D);
    r.scale(&[-1.0]);
    r.axpy(&[C64::new(1.0, 0.0)], &b);
    let physical = dot(&r, &r).re / dot(&b, &b).re;
    eprintln!(
        "BiCGStab with the Chebyshev cycle: {} iterations, |D x - b|^2/|b|^2 = {physical:e}",
        info.iterations
    );
    assert!(physical < 1e-15, "{physical:e}");
}

/// The basis of four random test vectors on 2⁴ blocks, folded with `fold` if given.
fn folded_and_full(op: &Mdwf<f64>, lat: &Arc<Lattice>, fold: Option<Gamma>) -> Basis<f64> {
    let agg = Aggregation::new(lat, [2, 2, 2, 2]);
    let n = 4;
    let mut builder = BasisBuilder::<f64>::new(&agg, n);
    if let Some(g) = fold {
        builder = builder.folded(Reflection(g));
    }
    for i in 0..n {
        let x = random(lat, MOBIUS.ls, 1, 120 + i as u64);
        let mut g5x = x.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5x, &x);
        builder.add(i, &x, &g5x);
    }
    builder.finish().unwrap()
}

/// A basis folded on the fifth dimension (Γ5 = γ5 R, only s < Ls/2 kept) restricts and
/// prolongs as the full one does, in half the memory.
#[test]
fn folded_basis_acts_as_the_full_one() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 119));
    let full = folded_and_full(&op, &lat, None);
    let folded = folded_and_full(&op, &lat, Some(Dirac::GAMMA5));
    assert!(folded.is_folded() && !full.is_folded());

    let f = random(&lat, MOBIUS.ls, 2, 130);
    let (mut a, mut b) = (full.coarse_field(2), full.coarse_field(2));
    full.restrict(&mut a, &f);
    folded.restrict(&mut b, &f);
    let scale = a.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(diff < 1e-12 * scale, "restriction: {diff:e}");

    let (mut pa, mut pb) = (f.zeros_like(), f.zeros_like());
    full.prolong(&mut pa, &a);
    folded.prolong(&mut pb, &a);
    let diff = max_diff(&pa, &pb);
    assert!(diff < 1e-12, "prolongation: {diff:e}");
    // P† P = 1 for the folded basis.
    let mut back = a.zeros_like();
    folded.restrict(&mut back, &pb);
    let diff = back
        .as_slice()
        .iter()
        .zip(a.as_slice())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(diff < 1e-12 * scale, "P† P: {diff:e}");
}

/// With a wrong reflection (here -γ5) the builder refuses the test vectors.
#[test]
#[should_panic(expected = "Γ5 is not R ⊗ g")]
fn folding_with_the_wrong_reflection_is_refused() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 119));
    let minus = Gamma {
        col: Dirac::GAMMA5.col,
        val: [Unit::MinusOne; 4],
    };
    folded_and_full(&op, &lat, Some(minus));
}

/// The even-odd Pauli-Villars operator, right and left forms, with Neff's α in the quark
/// operator alone or in both: its adjoint is its adjoint, and BiCGStab on it with its
/// right-hand side and solution solves D x = b (checked with the plain D).
#[test]
fn even_odd_pv_operator_solves_d_in_every_form() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 141);
    let plain = operator(&lat, MOBIUS, &u);
    let index = plain.even_odd_index().unwrap();
    let b = random(&lat, MOBIUS.ls, 2, 142);
    let mut even = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 2, 1);
    even.set_parity(Parity::Even);
    for (form, alpha, alpha_pv) in [
        (EoForm::Right, 1.0, 1.0),
        (EoForm::Left, 1.0, 1.0),
        (EoForm::Left, 0.6, 0.6),
        (EoForm::Left, 0.6, 1.0),
        (EoForm::Right, 0.6, 0.6),
    ] {
        let mut op = operator(&lat, MOBIUS, &u);
        op.set_alpha(alpha);
        let mut pv = operator(
            &lat,
            MdwfParams {
                mass: 1.0,
                ..MOBIUS
            },
            &u,
        );
        pv.set_alpha(alpha_pv);
        let mut a = PvEvenOdd::new(&op, &pv, form, &even);

        let mut x = even.zeros_like();
        let mut y = even.zeros_like();
        extract_parity(
            &mut x,
            &random(&lat, MOBIUS.ls, 2, 143),
            index,
            Parity::Even,
        );
        extract_parity(
            &mut y,
            &random(&lat, MOBIUS.ls, 2, 144),
            index,
            Parity::Even,
        );
        let (mut ay, mut adag_x) = (even.zeros_like(), even.zeros_like());
        a.mult(&mut ay, &y);
        a.mult_dag(&mut adag_x, &x);
        let (mut lhs, mut rhs) = ([C64::ZERO; 2], [C64::ZERO; 2]);
        x.dot(&ay, &mut lhs);
        adag_x.dot(&y, &mut rhs);
        let mut scale = [0.0; 2];
        x.norm_sqr(&mut scale);
        assert!(
            (lhs[0] - rhs[0]).norm_sqr().sqrt() < 1e-12 * scale[0],
            "{form:?} α {alpha}/{alpha_pv}: {:?} vs {:?}",
            lhs[0],
            rhs[0]
        );

        let mut rhs = even.zeros_like();
        a.right_hand_side(&mut rhs, &b);
        let mut sol = even.zeros_like();
        let info = BiCgStab::new(Stop::simple(1e-26, 4000))
            .solve(&mut a, None, &mut sol, &rhs)
            .unwrap();
        let mut x = b.zeros_like();
        a.solution(&mut x, &sol, &b);
        let mut r = mult(&plain, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0); 2], &b);
        let (mut left, mut size) = ([0.0; 2], [0.0; 2]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        eprintln!(
            "{form:?}, α {alpha} (PV {alpha_pv}): {} iterations, |D x - b|^2/|b|^2 = {:e}",
            info.iterations,
            left[0] / size[0]
        );
        assert!(left.iter().zip(&size).all(|(l, s)| l < &(1e-16 * s)));
    }
}

/// The Schur factors of `PvEvenOdd` do C⁻¹ and the difference with the input in the sweeps
/// of the hops. Ã and Ã† are what the even-odd blocks give one pass at a time, in both forms.
#[test]
fn even_odd_pv_operator_is_made_of_the_blocks() {
    type H = Field<f64, Spinor, Half>;
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 151);
    let op = operator(&lat, MOBIUS, &u);
    let pv = operator(
        &lat,
        MdwfParams {
            mass: 1.0,
            ..MOBIUS
        },
        &u,
    );
    let index = op.even_odd_index().unwrap();
    let mut even = H::with_shape(&lat, 24, MOBIUS.ls, 2, 1);
    even.set_parity(Parity::Even);
    let mut x = even.zeros_like();
    extract_parity(
        &mut x,
        &random(&lat, MOBIUS.ls, 2, 152),
        index,
        Parity::Even,
    );
    // 1 − D_eo C_o⁻¹ D_oe C_e⁻¹ (right form) or 1 − C_e⁻¹ D_eo C_o⁻¹ D_oe (left form), or
    // the adjoint, which takes the steps in the opposite order.
    let factor = |m: &Mdwf<f64>, form: EoForm, dagger: Dagger, inp: &H| {
        let mut work = m.even_odd_work();
        let (e, o) = (Parity::Even, Parity::Odd);
        let mut o1 = inp.zeros_like();
        o1.set_parity(o);
        let mut o2 = o1.zeros_like();
        let (mut t, mut out) = (inp.zeros_like(), inp.zeros_like());
        if (form == EoForm::Right) == (dagger == Dagger::No) {
            m.diag_inv(&mut work, &mut t, inp, e, dagger);
            m.hop(&mut work, &mut o1, &t, o, dagger);
            m.diag_inv(&mut work, &mut o2, &o1, o, dagger);
            m.hop(&mut work, &mut out, &o2, e, dagger);
        } else {
            m.hop(&mut work, &mut o1, inp, o, dagger);
            m.diag_inv(&mut work, &mut o2, &o1, o, dagger);
            m.hop(&mut work, &mut t, &o2, e, dagger);
            m.diag_inv(&mut work, &mut out, &t, e, dagger);
        }
        out.aypx(&[C64::new(-1.0, 0.0); 2], inp);
        out
    };
    let diff = |a: &H, b: &H| {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    };
    for form in [EoForm::Right, EoForm::Left] {
        let mut a = PvEvenOdd::new(&op, &pv, form, &even);
        let (mut ax, mut adag_x) = (even.zeros_like(), even.zeros_like());
        a.mult(&mut ax, &x);
        a.mult_dag(&mut adag_x, &x);
        let want = factor(&pv, form, Dagger::Yes, &factor(&op, form, Dagger::No, &x));
        let want_dag = factor(&op, form, Dagger::Yes, &factor(&pv, form, Dagger::No, &x));
        let (d, d_dag) = (diff(&ax, &want), diff(&adag_x, &want_dag));
        assert!(diff(&ax, &x) > 0.1 && diff(&adag_x, &x) > 0.1);
        assert!(d < 1e-13 && d_dag < 1e-13, "{form:?}: {d:e}, {d_dag:e}");

        // One factor alone (`SchurFactor`, through `schur_factor` and the fused `hop_then`).
        let mut odd = even.zeros_like();
        odd.set_parity(Parity::Odd);
        let mut k = SchurFactor::new(&op, form, &even, &odd);
        let (mut kx, mut kdag_x) = (even.zeros_like(), even.zeros_like());
        k.mult(&mut kx, &x);
        k.mult_dag(&mut kdag_x, &x);
        let (d, d_dag) = (
            diff(&kx, &factor(&op, form, Dagger::No, &x)),
            diff(&kdag_x, &factor(&op, form, Dagger::Yes, &x)),
        );
        assert!(
            d < 1e-13 && d_dag < 1e-13,
            "SchurFactor {form:?}: {d:e}, {d_dag:e}"
        );
    }

    // Γ5 = γ5 R on the even sites is that of the whole lattice there.
    let whole = random(&lat, MOBIUS.ls, 2, 153);
    let g_whole = {
        let mut g = whole.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g, &whole);
        let mut g_e = even.zeros_like();
        extract_parity(&mut g_e, &g, index, Parity::Even);
        g_e
    };
    let mut w_e = even.zeros_like();
    extract_parity(&mut w_e, &whole, index, Parity::Even);
    let mut g_half = even.zeros_like();
    op.mult_gamma5_on(&mut g_half, &w_e);
    assert!(diff(&g_half, &g_whole) < 1e-15, "Γ5 on the even sites");
}

/// P† ι and π P on the sites of one parity give what restricting the field put onto the
/// whole lattice and taking the sites of the prolongation give, folded or not.
#[test]
fn restriction_and_prolongation_on_one_parity() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 119));
    let index = op.even_odd_index().unwrap();
    for fold in [None, Some(Dirac::GAMMA5)] {
        let basis = folded_and_full(&op, &lat, fold);
        for parity in [Parity::Even, Parity::Odd] {
            let full = random(&lat, MOBIUS.ls, 2, 150);
            let mut half = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 2, 1);
            half.set_parity(parity);
            extract_parity(&mut half, &full, index, parity);
            let mut injected = full.zeros_like();
            insert_parity(&mut injected, &half, index);

            let (mut a, mut b) = (basis.coarse_field(2), basis.coarse_field(2));
            basis.restrict(&mut a, &injected);
            basis.restrict_half(&mut b, &half, index);
            let diff = a
                .as_slice()
                .iter()
                .zip(b.as_slice())
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(diff < 1e-13, "{fold:?} {parity:?} restriction: {diff:e}");

            let mut prolonged = full.zeros_like();
            basis.prolong(&mut prolonged, &a);
            let mut want = half.zeros_like();
            extract_parity(&mut want, &prolonged, index, parity);
            let mut got = half.zeros_like();
            basis.prolong_half(&mut got, &a, index);
            let diff = want
                .as_slice()
                .iter()
                .zip(got.as_slice())
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(diff < 1e-13, "{fold:?} {parity:?} prolongation: {diff:e}");
        }
    }
}

/// The even-odd Pauli-Villars multigrid: test vectors from a Chebyshev filter on Ã†Ã,
/// the folded basis, the cycle; GCR(2) with the cycle solves D x = b in the right form, in
/// the left form, and in the left form with Neff's α in D, each in fewer steps than
/// BiCGStab without a preconditioner needs.
#[test]
fn even_odd_pv_multigrid_solves_d() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 161);
    let pv_params = MdwfParams {
        mass: 1.0,
        ..MOBIUS
    };
    let (op, pv) = (operator(&lat, MOBIUS, &u), operator(&lat, pv_params, &u));
    let index = op.even_odd_index().unwrap();
    let even_random = |n: usize, seed: u64| {
        let mut h = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, n, 1);
        h.set_parity(Parity::Even);
        extract_parity(
            &mut h,
            &random(&lat, MOBIUS.ls, n, seed),
            index,
            Parity::Even,
        );
        h
    };
    let like = even_random(1, 1);

    // Four approximate low modes of Ã, orthonormalised.
    let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);
    let hi = 1.1 * largest_magnitude(&mut a, &mut even_random(1, 162), 50);
    let mut filter = ChebyshevFilter::new(
        Normal::new(PvEvenOdd::new(&op, &pv, EoForm::Right, &like), &like),
        40,
        hi * hi / 100.0,
        hi * hi,
        &like,
    );
    let start = even_random(4, 163);
    let mut y = start.zeros_like();
    filter.mult(&mut y, &start);
    for i in 0..4 {
        let mut v = y.select(&[i]);
        for j in 0..i {
            let q = y.select(&[j]);
            let mut d = [C64::ZERO];
            q.dot(&v, &mut d);
            v.axpy(&[-d[0]], &q);
        }
        let mut n = [0.0];
        v.norm_sqr(&mut n);
        v.scale(&[1.0 / n[0].sqrt()]);
        y.assign(&[i], &v);
    }
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    let basis = pv_eo_basis(&op, &pv, &agg, &y).unwrap();
    assert!(basis.is_folded());

    let b = random(&lat, MOBIUS.ls, 1, 164);
    let stop = Stop::simple(1e-24, 500);
    let mut op_alpha = operator(&lat, MOBIUS, &u);
    op_alpha.set_alpha(0.6);
    for (op_a, form) in [
        (&op, EoForm::Right),
        (&op, EoForm::Left),
        (&op_alpha, EoForm::Left),
    ] {
        let mut a = PvEvenOdd::new(op_a, &pv, form, &like);
        let mut rhs = like.zeros_like();
        a.right_hand_side(&mut rhs, &b);
        let mut plain = like.zeros_like();
        let unpreconditioned = BiCgStab::new(stop)
            .solve(&mut a, None, &mut plain, &rhs)
            .unwrap();
        let mut cycle =
            PvEoCycle::new(&op, &pv, (op_a, &pv, form), &basis, PvEoParams::default()).unwrap();
        let mut sol = like.zeros_like();
        let info = Gcr::new(stop, 2)
            .solve(&mut a, Some(&mut cycle), &mut sol, &rhs)
            .unwrap();
        let mut x = b.zeros_like();
        a.solution(&mut x, &sol, &b);
        let mut r = mult(&op, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        eprintln!(
            "{form:?}, α {}: GCR(2) with the cycle {} steps ({} coarse applications), BiCGStab alone {}; |D x - b|^2/|b|^2 = {:e}",
            op_a.alpha(),
            info.iterations,
            cycle.coarse_applications(),
            unpreconditioned.iterations,
            left[0] / size[0]
        );
        assert!(left[0] < 1e-16 * size[0]);
        assert!(info.iterations < unpreconditioned.iterations);

        // A level below the coarse lattice (super-blocks of 2⁴ coarse sites, the 1 × 1 × 1 × 2 of
        // them solved exactly) under the solves of the quark factor: the same solution in as many
        // outer steps.
        let mut two =
            PvEoCycle::new(&op, &pv, (op_a, &pv, form), &basis, PvEoParams::default()).unwrap();
        let vectors = pv_eo_coarse_vectors(&op, &pv, &basis, &y);
        let setup = two.add_levels(&vectors, &[2]).unwrap();
        let mut sol = like.zeros_like();
        let two_info = Gcr::new(stop, 2)
            .solve(&mut a, Some(&mut two), &mut sol, &rhs)
            .unwrap();
        a.solution(&mut x, &sol, &b);
        let mut r = mult(&op, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        r.norm_sqr(&mut left);
        eprintln!(
            "  with a level below: {} steps ({} coarse applications, {setup} to set it up); |D x - b|^2/|b|^2 = {:e}",
            two_info.iterations,
            two.coarse_applications(),
            left[0] / size[0]
        );
        assert!(left[0] < 1e-16 * size[0]);
        assert!(two_info.iterations <= info.iterations + 1);

        // The same level under the solves of the second (Pauli-Villars) factor as well.
        let mut both =
            PvEoCycle::new(&op, &pv, (op_a, &pv, form), &basis, PvEoParams::default()).unwrap();
        both.add_levels(&vectors, &[2]).unwrap();
        let setup = both.add_pv_levels(&vectors, &[2]).unwrap();
        let mut sol = like.zeros_like();
        let both_info = Gcr::new(stop, 2)
            .solve(&mut a, Some(&mut both), &mut sol, &rhs)
            .unwrap();
        a.solution(&mut x, &sol, &b);
        let mut r = mult(&op, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        r.norm_sqr(&mut left);
        eprintln!(
            "  with the level below both factors: {} steps ({} coarse applications, {setup} more to set it up); |D x - b|^2/|b|^2 = {:e}",
            both_info.iterations,
            both.coarse_applications(),
            left[0] / size[0]
        );
        assert!(left[0] < 1e-16 * size[0]);
        assert!(both_info.iterations <= info.iterations + 1);
    }
}

/// The even-odd Pauli-Villars multigrid split over processes (threads standing in for them): the
/// coarse operator and its exact adjoint agree with the ones on one process, and GCR(2) with the
/// cycle, with and without a level below the coarse lattice, gives the same solution in as many
/// steps.
#[test]
fn even_odd_pv_multigrid_on_several_processes() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 181);
    let pv_params = MdwfParams {
        mass: 1.0,
        ..MOBIUS
    };
    let start = random(&whole, MOBIUS.ls, 4, 182);
    let b = random(&whole, MOBIUS.ls, 1, 183);

    // On every process: D_c v and D_c† v for a coarse v, and the solutions without and
    // with the level below.
    let run = |lat: Arc<Lattice>, coarse_v: Option<&Field<f64, ruttice::Coarse>>| {
        let u = local_part(&u, &lat);
        let (op, pv) = (operator(&lat, MOBIUS, &u), operator(&lat, pv_params, &u));
        let index = op.even_odd_index().unwrap().clone();
        let mut y = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 4, 1);
        y.set_parity(Parity::Even);
        extract_parity(&mut y, &local_part(&start, &lat), &index, Parity::Even);
        let like = y.select(&[0]);
        let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
        let basis = pv_eo_basis(&op, &pv, &agg, &y).unwrap();

        let mut coarse = basis.coarse_op(&op, &mut op.work());
        let v = match coarse_v {
            Some(v) => local_part(v, agg.coarse_lattice()),
            None => {
                let mut v = basis.coarse_field(2);
                for (i, x) in v.as_mut_slice().iter_mut().enumerate() {
                    *x = ((i * 7919) % 1000) as f64 / 500.0 - 1.0;
                }
                v
            }
        };
        let (mut dv, mut ddv) = (v.zeros_like(), v.zeros_like());
        coarse.mult(&mut dv, &v);
        coarse.mult_adjoint(&mut ddv, &v);
        // The exact adjoint as an operator of its own is the same map, on every process.
        let mut adjoint = coarse.adjoint();
        let mut by_adjoint = v.zeros_like();
        adjoint.mult(&mut by_adjoint, &v);
        assert!(
            by_adjoint
                .as_slice()
                .iter()
                .zip(ddv.as_slice())
                .all(|(a, b)| (a - b).abs() <= 1e-13 * (1.0 + b.abs())),
            "CoarseOp::adjoint is not mult_adjoint"
        );

        let b = local_part(&b, &lat);
        let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);
        let mut rhs = like.zeros_like();
        a.right_hand_side(&mut rhs, &b);
        let mut solutions = Vec::new();
        for levels in [false, true] {
            let mut cycle = PvEoCycle::new(
                &op,
                &pv,
                (&op, &pv, EoForm::Right),
                &basis,
                PvEoParams::default(),
            )
            .unwrap();
            if levels {
                let vectors = pv_eo_coarse_vectors(&op, &pv, &basis, &y);
                cycle.add_levels(&vectors, &[2]).unwrap();
            }
            let mut sol = like.zeros_like();
            let info = Gcr::new(Stop::simple(1e-22, 200), 2)
                .solve(&mut a, Some(&mut cycle), &mut sol, &rhs)
                .unwrap();
            let mut x = b.zeros_like();
            a.solution(&mut x, &sol, &b);
            solutions.push((info.iterations, x));
        }
        (v, dv, ddv, solutions)
    };

    let (v, dv, ddv, one) = run(whole.clone(), None);
    let coarse_whole = v.lattice().clone();
    for grid in [[1, 1, 1, 2], [2, 1, 1, 2]] {
        let parts = LocalComm::run(global, grid, |lat| run(lat, Some(&v)));
        let rel = |want: &V, got: V| {
            let mut d = got;
            d.aypx(&vec![C64::new(-1.0, 0.0); want.num_vectors()], want);
            let (mut e, mut n) = (vec![0.0; want.num_vectors()], vec![0.0; want.num_vectors()]);
            d.norm_sqr(&mut e);
            want.norm_sqr(&mut n);
            e.iter()
                .zip(&n)
                .fold(0.0_f64, |m, (e, n)| m.max((e / n).sqrt()))
        };
        let coarse_diff = |want: &Field<f64, ruttice::Coarse>,
                           got: &[Field<f64, ruttice::Coarse>]| {
            let got = assemble(&coarse_whole, got);
            want.as_slice()
                .iter()
                .zip(got.as_slice())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
        };
        let (mut dvs, mut ddvs) = (Vec::new(), Vec::new());
        let (mut xs, mut split_steps) = ([Vec::new(), Vec::new()], [0; 2]);
        for (_, dv, ddv, solutions) in parts {
            dvs.push(dv);
            ddvs.push(ddv);
            for (k, (steps, x)) in solutions.into_iter().enumerate() {
                split_steps[k] = steps;
                xs[k].push(x);
            }
        }
        let d_diff = coarse_diff(&dv, &dvs);
        let dd_diff = coarse_diff(&ddv, &ddvs);
        eprintln!("grid {grid:?}: |D_c v| differs by {d_diff:e}, |D_c† v| by {dd_diff:e}");
        assert!(d_diff < 1e-12 && dd_diff < 1e-12, "grid {grid:?}");
        for (k, (iterations, x)) in one.iter().enumerate() {
            let split_iterations = split_steps[k];
            let diff = rel(x, assemble(&whole, &xs[k]));
            eprintln!(
                "  {}: {iterations} steps on one process, {split_iterations} split; solutions differ by {diff:e}",
                if k == 0 { "two levels" } else { "with a level below" }
            );
            assert!(split_iterations.abs_diff(*iterations) <= 1);
            assert!(diff < 1e-8, "grid {grid:?}");
        }
    }
}

/// The setup of the even-odd Pauli-Villars multigrid from scratch with everything above the
/// coarse lattice stored as `T`: filtered low modes, one bootstrap round, then GCR(2) in double
/// precision with the cycle solves D x = b. Its steps.
fn setup_from_scratch<T: Storage>() -> usize {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 171);
    let pv_params = MdwfParams {
        mass: 1.0,
        ..MOBIUS
    };
    let (op, pv) = (operator(&lat, MOBIUS, &u), operator(&lat, pv_params, &u));
    let stored = |params: MdwfParams| {
        let mut op = Mdwf::<T>::new(&lat, params);
        op.set_gauge(&u).unwrap();
        op
    };
    let (op_t, pv_t) = (stored(MOBIUS), stored(pv_params));
    let (mut y, theta) = pv_eo_low_modes(
        &op_t,
        &pv_t,
        &lat,
        LowModeParams {
            vectors: 6,
            rounds: 2,
            chunk: 3,
            seed: 172,
        },
    );
    assert!(theta.windows(2).all(|w| w[0] <= w[1]), "{theta:?}");
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    let basis = pv_eo_bootstrap(
        &op_t,
        &pv_t,
        &agg,
        &mut y,
        BootstrapParams {
            rounds: 1,
            ..BootstrapParams::default()
        },
    )
    .unwrap();
    let mut like = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 1, 1);
    like.set_parity(Parity::Even);
    let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);
    let cycle = PvEoCycle::new(
        &op_t,
        &pv_t,
        (&op_t, &pv_t, EoForm::Right),
        &basis,
        PvEoParams::default(),
    )
    .unwrap();
    let mut like_t = Field::<T, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 1, 1);
    like_t.set_parity(Parity::Even);
    let mut cycle = Convert::new(cycle, &like_t);
    let b = random(&lat, MOBIUS.ls, 1, 173);
    let mut rhs = like.zeros_like();
    a.right_hand_side(&mut rhs, &b);
    let mut sol = like.zeros_like();
    let info = Gcr::new(Stop::simple(1e-24, 200), 2)
        .solve(&mut a, Some(&mut cycle), &mut sol, &rhs)
        .unwrap();
    let mut x = b.zeros_like();
    a.solution(&mut x, &sol, &b);
    let mut r = mult(&op, &x, Variant::D);
    r.aypx(&[C64::new(-1.0, 0.0)], &b);
    let (mut left, mut size) = ([0.0], [0.0]);
    r.norm_sqr(&mut left);
    b.norm_sqr(&mut size);
    eprintln!(
        "Ritz values of Ã†Ã {:.3e} .. {:.3e}; GCR(2) {} steps; |D x - b|^2/|b|^2 = {:e}",
        theta[0],
        theta[theta.len() - 1],
        info.iterations,
        left[0] / size[0]
    );
    assert!(left[0] < 1e-16 * size[0]);
    info.iterations
}

/// In single precision above the coarse lattice the multigrid is as good a preconditioner.
#[test]
fn even_odd_pv_multigrid_setup_from_scratch() {
    let (double, single) = (setup_from_scratch::<f64>(), setup_from_scratch::<f32>());
    assert!(single <= double + 1, "{single} steps against {double}");
}

/// The exact adjoint of a coarse operator: <x, D_c y> = <D_c† x, y>; and it differs from
/// the Γ5-adjoint for Möbius (c ≠ 0), where D† = Γ5 D Γ5 holds only approximately.
#[test]
fn coarse_operator_exact_adjoint() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = operator(&lat, MOBIUS, &random_gauge(&lat, 119));
    let basis = folded_and_full(&op, &lat, Some(Dirac::GAMMA5));
    let mut coarse = CoarseOp::build(&op, &mut op.work(), &basis);
    let fill = |seed: usize| {
        let mut c = basis.coarse_field(1);
        for (i, v) in c.as_mut_slice().iter_mut().enumerate() {
            *v = ((i * 7 + seed) as f64 * 0.37).sin();
        }
        c
    };
    let (x, y) = (fill(3), fill(11));
    let (mut dy, mut dag_x, mut g5_x) = (y.zeros_like(), x.zeros_like(), x.zeros_like());
    coarse.mult(&mut dy, &y);
    coarse.mult_adjoint(&mut dag_x, &x);
    coarse.mult_dag(&mut g5_x, &x);
    let dot = |a: &Field<f64, ruttice::Coarse>, b: &Field<f64, ruttice::Coarse>| {
        let mut d = [C64::ZERO];
        a.dot(b, &mut d);
        d[0]
    };
    let (lhs, rhs) = (dot(&x, &dy), dot(&dag_x, &y));
    let scale = dot(&x, &x).re.sqrt() * dot(&dy, &dy).re.sqrt();
    assert!(
        (lhs - rhs).norm_sqr().sqrt() < 1e-12 * scale,
        "{lhs:?} vs {rhs:?}"
    );
    let mut diff = dag_x.zeros_like();
    diff.copy_from(&dag_x);
    diff.axpy(&[C64::new(-1.0, 0.0)], &g5_x);
    eprintln!(
        "|D_c† x − Γ5 D_c Γ5 x| / |D_c† x| = {:.2e}",
        (dot(&diff, &diff).re / dot(&dag_x, &dag_x).re).sqrt()
    );
}

/// CGNE with Neff's α and the multigrid: the operator is the normal operator N = L^(α)† L^(α) of
/// the left-preconditioned Schur complement (`PvEvenOdd` with the quark operator in place of the
/// Pauli-Villars one), whose inverse the cycle approximates with the second coarse factor P†D†P
/// solved on the exact adjoint.
#[test]
fn cgne_with_alpha_and_the_multigrid() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 181);
    let op = operator(&lat, MOBIUS, &u);
    let (y, _) = pv_eo_low_modes(
        &op,
        &op,
        &lat,
        LowModeParams {
            vectors: 6,
            rounds: 2,
            chunk: 3,
            seed: 182,
        },
    );
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    let basis = pv_eo_basis(&op, &op, &agg, &y).unwrap();
    let mut like = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 1, 1);
    like.set_parity(Parity::Even);
    let b = random(&lat, MOBIUS.ls, 1, 183);
    for alpha in [1.0, 0.6] {
        let mut op_a = operator(&lat, MOBIUS, &u);
        op_a.set_alpha(alpha);
        let mut n = PvEvenOdd::new(&op_a, &op_a, EoForm::Left, &like);
        let mut rhs = like.zeros_like();
        n.right_hand_side(&mut rhs, &b);
        let stop = Stop::simple(1e-24, 2000);
        let mut plain = like.zeros_like();
        let cg = Cg::new(stop).solve(&mut n, None, &mut plain, &rhs).unwrap();
        let params = PvEoParams {
            pv_solver: CoarseKrylov::BiCgStab,
            ..PvEoParams::default()
        };
        let mut cycle =
            PvEoCycle::new(&op, &op, (&op_a, &op_a, EoForm::Left), &basis, params).unwrap();
        let mut sol = like.zeros_like();
        let info = Gcr::new(stop, 2)
            .solve(&mut n, Some(&mut cycle), &mut sol, &rhs)
            .unwrap();
        let mut x = b.zeros_like();
        n.solution(&mut x, &sol, &b);
        let mut r = mult(&op, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        eprintln!(
            "α {alpha}: GCR(2) with the cycle {} steps ({} coarse), CG alone {}; |D x - b|^2/|b|^2 = {:e}",
            info.iterations,
            cycle.coarse_applications(),
            cg.iterations,
            left[0] / size[0]
        );
        assert!(left[0] < 1e-14 * size[0]);
        assert!(info.iterations < cg.iterations);
    }
}

/// The four-dimensional propagator S η = P- ψ_0 + P+ ψ_(Ls-1) of D ψ = source_5d(η) is
/// γ5-Hermitian, <χ, S η> = <γ5 S γ5 χ, η>, for Möbius and for Shamir (it is the inverse of
/// an overlap-type operator, up to a contact term), which the wrong pairing of walls and
/// chiralities is not; and it and the midpoint field do not depend on Neff's α.
#[test]
fn four_dimensional_propagator_is_gamma5_hermitian_and_alpha_free() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let u = random_gauge(&lat, 191);
    let four = |seed: u64| {
        let mut f = V::with_shape(&lat, 24, 1, 1, 1);
        f.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, 1, seed).as_slice());
        f
    };
    let gamma5 = |f: &V| {
        let mut g = f.zeros_like();
        for site in 0..lat.num_sites() {
            let p = ruttice::kernel::Spinor::<f64>::from_reals(f.site(0, site));
            Dirac::GAMMA5.apply(&p).write_reals(g.site_mut(0, site));
        }
        g
    };
    let mut even = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 1, 1);
    even.set_parity(Parity::Even);
    for (params, alpha) in [
        (MOBIUS, 1.0),
        (MOBIUS, 0.6),
        (
            MdwfParams {
                b: 1.0,
                c: 0.0,
                ..MOBIUS
            },
            1.0,
        ),
    ] {
        let plain = operator(&lat, params, &u);
        let mut op = operator(&lat, params, &u);
        op.set_alpha(alpha);
        let pv = operator(
            &lat,
            MdwfParams {
                mass: 1.0,
                ..params
            },
            &u,
        );
        let form = if alpha == 1.0 {
            EoForm::Right
        } else {
            EoForm::Left
        };
        // ψ for the source η, with the operator carrying α; returns (S η, q_mid).
        let solve = |eta: &V| {
            let mut b = V::with_shape(&lat, 24, params.ls, 1, 1);
            plain.source_5d(&mut plain.work(), &mut b, eta);
            let mut a = PvEvenOdd::new(&op, &pv, form, &even);
            let mut rhs = even.zeros_like();
            a.right_hand_side(&mut rhs, &b);
            let mut y = even.zeros_like();
            BiCgStab::new(Stop::simple(1e-28, 4000))
                .solve(&mut a, None, &mut y, &rhs)
                .unwrap();
            let mut psi = b.zeros_like();
            a.solution(&mut psi, &y, &b);
            let (mut s, mut mid) = (eta.zeros_like(), eta.zeros_like());
            plain.propagator_4d(&mut s, &psi);
            plain.midpoint_4d(&mut mid, &psi);
            (s, mid)
        };
        let (eta, chi) = (four(192), four(193));
        let (s_eta, mid_eta) = solve(&eta);
        let (s_g5chi, _) = solve(&gamma5(&chi));
        let (lhs, rhs) = (dot(&chi, &s_eta), dot(&gamma5(&s_g5chi), &eta));
        let scale = dot(&s_eta, &s_eta).re.sqrt() * dot(&chi, &chi).re.sqrt();
        eprintln!(
            "b = {}, c = {}, α = {alpha}: |<χ, S η> − <γ5 S γ5 χ, η>| / |S η||χ| = {:.2e}",
            params.b,
            params.c,
            (lhs - rhs).norm_sqr().sqrt() / scale
        );
        assert!(
            (lhs - rhs).norm_sqr().sqrt() < 1e-10 * scale,
            "{lhs:?} vs {rhs:?}"
        );
        if alpha != 1.0 {
            // The same as with the plain operator.
            let pv_plain = operator(
                &lat,
                MdwfParams {
                    mass: 1.0,
                    ..params
                },
                &u,
            );
            let mut b = V::with_shape(&lat, 24, params.ls, 1, 1);
            plain.source_5d(&mut plain.work(), &mut b, &eta);
            let mut a = PvEvenOdd::new(&plain, &pv_plain, EoForm::Right, &even);
            let mut rhs = even.zeros_like();
            a.right_hand_side(&mut rhs, &b);
            let mut y = even.zeros_like();
            BiCgStab::new(Stop::simple(1e-28, 4000))
                .solve(&mut a, None, &mut y, &rhs)
                .unwrap();
            let mut psi = b.zeros_like();
            a.solution(&mut psi, &y, &b);
            let (mut s, mut mid) = (eta.zeros_like(), eta.zeros_like());
            plain.propagator_4d(&mut s, &psi);
            plain.midpoint_4d(&mut mid, &psi);
            assert!(max_diff(&s, &s_eta) < 1e-10, "S depends on α");
            assert!(max_diff(&mid, &mid_eta) < 1e-10, "q_mid depends on α");
        }
    }
}

/// `expected_result` of the Bridge++ test `Spectrum_Domainwall_Hadron2ptFunction`
/// (tests/ParameterFiles/Spectrum/test_Spectrum_Domainwall_Hadron2ptFunction.yaml): the real part
/// of the pseudoscalar correlator at t = 0 for a point source at the origin, Shamir's operator (b =
/// 1, c = 0), M0 = 1.6, m = 0.1, Ls = 8, antiperiodic in time.
const BRIDGE_DW_PION_T0: f64 = 0.434204939377926;

#[test]
fn bridge_spectrum_domainwall_pion_and_residual_mass() {
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
    let eta = point_source::<f64>(&lat, [0, 0, 0, 0]);
    let mut even = Field::<f64, Spinor, Half>::with_shape(&lat, 24, params.ls, NUM_SOURCES, 1);
    even.set_parity(Parity::Even);
    // The four-dimensional propagator and the midpoint field, with α in the quark operator.
    let propagate = |params: MdwfParams, alpha: f64| {
        let plain = operator(&lat, params, &u);
        let mut op = operator(&lat, params, &u);
        op.set_alpha(alpha);
        let pv = operator(
            &lat,
            MdwfParams {
                mass: 1.0,
                ..params
            },
            &u,
        );
        let form = if alpha == 1.0 {
            EoForm::Right
        } else {
            EoForm::Left
        };
        let mut b = V::with_shape(&lat, 24, params.ls, NUM_SOURCES, 1);
        plain.source_5d(&mut plain.work(), &mut b, &eta);
        let mut a = PvEvenOdd::new(&op, &pv, form, &even);
        let mut rhs = even.zeros_like();
        a.right_hand_side(&mut rhs, &b);
        let mut y = even.zeros_like();
        BiCgStab::new(Stop::simple(1e-28, 4000))
            .solve(&mut a, None, &mut y, &rhs)
            .unwrap();
        let mut psi = b.zeros_like();
        a.solution(&mut psi, &y, &b);
        let mut r = mult(&plain, &psi, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0); NUM_SOURCES], &b);
        let (mut rr, mut bb) = ([0.0; NUM_SOURCES], [0.0; NUM_SOURCES]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        let worst = rr.iter().zip(&bb).fold(0.0_f64, |m, (r, b)| m.max(r / b));
        let (mut s, mut mid) = (eta.zeros_like(), eta.zeros_like());
        plain.propagator_4d(&mut s, &psi);
        plain.midpoint_4d(&mut mid, &psi);
        (s, mid, worst)
    };
    let (s, mid, worst) = propagate(params, 1.0);
    let pion = pion_correlator(&s);
    let j5q = pion_correlator(&mid);
    let general = meson_correlator(&s, &s, &Dirac::GAMMA5, &Dirac::GAMMA5, &Dirac::GAMMA5);
    eprintln!("worst |D ψ - b|^2/|b|^2 over the 12 columns: {worst:e}");
    for (t, (c, j)) in pion.iter().zip(&j5q).enumerate() {
        eprintln!(
            "  t = {t}  C_PS(t) = {c:.12e}  C_J5q(t) = {j:.6e}  m_res(t) = {:.6e}",
            j / c
        );
    }
    eprintln!(
        "C_PS(0) = {:.13}, Bridge++ = {BRIDGE_DW_PION_T0:.13}",
        pion[0]
    );
    assert!(worst < 1e-24);
    let relative = (pion[0] - BRIDGE_DW_PION_T0).abs() / BRIDGE_DW_PION_T0;
    assert!(relative < 1e-10, "relative difference {relative:e}");
    assert!((general[0].re - pion[0]).abs() < 1e-11 && general[0].im.abs() < 1e-11);

    // Möbius with Neff's α (left form) against Möbius without: the same correlators.
    let mobius = MdwfParams {
        b: 1.5,
        c: 0.5,
        ..params
    };
    let (s1, mid1, _) = propagate(mobius, 1.0);
    let (s2, mid2, _) = propagate(mobius, 0.6);
    let (p1, p2) = (pion_correlator(&s1), pion_correlator(&s2));
    let (j1, j2) = (pion_correlator(&mid1), pion_correlator(&mid2));
    for t in 0..p1.len() {
        assert!((p1[t] - p2[t]).abs() < 1e-10 * p1[t], "C_PS({t})");
        assert!((j1[t] - j2[t]).abs() < 1e-8 * j1[t] + 1e-20, "C_J5q({t})");
    }
    eprintln!(
        "Möbius b = 1.5, c = 0.5: C_PS(0) = {:.13}, Bridge++ = {BRIDGE_MOBIUS_PION_T0:.13}; m_res(4) = {:.4e}, the same with α = 0.6",
        p1[0],
        j1[4] / p1[4]
    );
    let relative = (p1[0] - BRIDGE_MOBIUS_PION_T0).abs() / BRIDGE_MOBIUS_PION_T0;
    assert!(relative < 1e-10, "Möbius: relative difference {relative:e}");
}

/// The same with Möbius coefficients b = 1.5, c = 0.5 (the 5D source then carries
/// D₋ = 1 - c W): C_PS(0) from Bridge++ 2.0.0 `sample_Spectrum` (Fopr_Domainwall, CGNR to
/// 1e-28, no smearing) on conf_04040408.txt.
const BRIDGE_MOBIUS_PION_T0: f64 = 0.4350524283957;

/// The pion correlator printed by a domain-wall run (`app::mdwf::run_on`), from its output.
fn printed_pion(text: &str) -> Vec<f64> {
    text.lines()
        .skip_while(|l| l.trim() != "PS:")
        .skip(1)
        .take_while(|l| l.starts_with("    "))
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect()
}

/// A domain-wall parameter file run on several processes (threads standing in for them)
/// prints the pion correlator of the run on one process: without a preconditioner, with the
/// Chebyshev polynomial of Ã, and with the multigrid. The three give the same correlator, and so
/// does the configuration read from Bridge++'s binary format.
#[test]
fn a_domain_wall_run_on_several_processes() {
    let dir = std::env::temp_dir().join(format!("ruttice_mdwf_ranks_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let whole = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&whole, 191);
    ruttice::io::write_bridge_text(&dir.join("conf.txt"), &u).unwrap();
    ruttice::io::GaugeFile::write(&ruttice::io::Binary, &dir.join("conf.bin"), &u).unwrap();
    let base = "Main:\n  lattice_size : [4, 4, 4, 8]\nGauge:\n  config_filename_input : conf.txt\nFopr:\n  fermion_type : Mobius\n  quark_mass : 0.1\n  domain_wall_height : 1.4\n  extent_of_5th_dimension : 4\nSolver:\n  solver_type : BiCGStab\n  convergence_criterion_squared : 1.0e-24\n  maximum_number_of_iteration : 4000\nSource:\n  source_position : [0, 0, 0, 0]\n  number_of_rhs_per_solve : 12\n";
    let multigrid = base.replace("solver_type : BiCGStab", "solver_type : GCR\n  preconditioner_type : MG")
        + "MultiGrid:\n  Setup:\n    filter_number_of_vectors : 4\n    filter_number_of_rounds : 1\n    bootstrap_number_of_rounds : 0\n  MultiGrid_Level1:\n    block_size : [2, 2, 2, 2]\n    number_of_test_vectors : 8\n";
    let chebyshev = base.replace(
        "solver_type : BiCGStab",
        "solver_type : BiCGStab\n  preconditioner_type : Chebyshev",
    );
    let binary = base.replace(
        "config_filename_input : conf.txt",
        "gauge_config_type_input : Binary\n  config_filename_input : conf.bin",
    );
    let mut plain: Option<Vec<f64>> = None;
    for (name, text) in [
        ("plain", base.to_string()),
        ("chebyshev", chebyshev),
        ("multigrid", multigrid),
        ("binary", binary),
    ] {
        let file = dir.join(format!("{name}.yaml"));
        std::fs::write(&file, &text).unwrap();
        let params = ruttice::app::params::Params::load(&file).unwrap();
        let mut one = Vec::new();
        ruttice::app::mdwf::run(&params, &file, (91, 91), &mut one).unwrap();
        let want = printed_pion(&String::from_utf8(one).unwrap());
        assert_eq!(want.len(), 8, "{name}");
        for (a, b) in want
            .iter()
            .zip(plain.get_or_insert_with(|| want.clone()).iter())
        {
            assert!((a - b).abs() < 1e-8 * b.abs(), "{name}: {a} vs plain {b}");
        }
        for grid in [[1, 1, 1, 2], [2, 1, 1, 2]] {
            let printed = LocalComm::run([4, 4, 4, 8], grid, |lat| {
                let mut out = Vec::new();
                let mut p = params.clone();
                p.main.grid_size = grid;
                ruttice::app::mdwf::run_on(&p, &file, (91, 91), &mut out, lat.comm.clone())
                    .unwrap();
                String::from_utf8(out).unwrap()
            });
            let got = printed_pion(&printed[0]);
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a - b).abs() < 1e-9 * b.abs(),
                    "{name}, grid {grid:?}: {a} vs {b}"
                );
            }
            eprintln!(
                "{name}, grid {grid:?}: C_PS(0) {} (one process {})",
                got[0], want[0]
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
