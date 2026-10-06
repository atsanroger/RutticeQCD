//! Stage 2 checks (DESIGN.md §9.1): field strength and the clover operator.

use ruttice_core::{
    dot, extract_parity, fill, insert_parity, norm_sqr, Boundary, Field, Gauge, Half, Lattice,
    LinearFn, Parity, Spinor, Stencil, Vector, C64, ND,
};
use ruttice_fermion::{
    solve_even_odd, Bound, Clover, CloverParams, Dagger, EvenOdd, FermionOp, HopTerms, Schur,
    Variant, Wilson, WilsonParams,
};
use ruttice_gauge::{field_strength, link, set_unit, PLANES};
use ruttice_io::read_bridge_text;
use ruttice_kernel::{Dirac, GammaRepre, Spinor as SiteSpinor};
use ruttice_solver::{solve_cgne, Cg, Stop};
use ruttice_validation::{
    bridge_test_file, gauge_transform, gauge_transform_spinor, max_abs_diff, random_gauge,
    random_spinor, random_su3, TestRng, BRIDGE_TESTS_ENV,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;

const PERIODIC: [Boundary; ND] = [Boundary::Periodic; ND];
/// Periodic in space, antiperiodic in time: what the Bridge++ clover tests use.
const TIME_ANTIPERIODIC: [Boundary; ND] = [
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];
const MIXED: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];

fn clover(
    lat: &Arc<Lattice>,
    csw: f64,
    boundary: [Boundary; ND],
    u: &Field<f64, Gauge>,
) -> Clover<f64> {
    let mut op = Clover::new(
        lat,
        CloverParams {
            kappa: 0.12,
            csw,
            boundary,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

fn mult(op: &Clover<f64>, inp: &V, variant: Variant) -> V {
    let mut out = inp.zeros_like();
    op.mult(&mut op.work(), &mut out, inp, variant);
    out
}

// ---- Field strength ----

#[test]
fn field_strength_is_antihermitian_and_vanishes_for_unit_links() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let st = Stencil::nearest(&lat, PERIODIC);
    let mut unit = Field::<f64, Gauge>::gauge(&lat);
    set_unit(&mut unit);
    assert!(field_strength(&unit, &st)
        .as_slice()
        .iter()
        .all(|x| *x == 0.0));

    let f = field_strength(&random_gauge(&lat, 201), &st);
    assert_eq!(f.outer(), PLANES.len());
    assert!(norm_sqr(&f) > 1.0);
    for plane in 0..PLANES.len() {
        for site in 0..lat.num_sites() {
            let m = link(&f, plane, site);
            let sum = m.add(&m.adj());
            for a in 0..3 {
                for b in 0..3 {
                    assert!(sum.0[a][b].norm_sqr() < 1e-28);
                }
            }
        }
    }
}

/// F(x) -> G(x) F(x) G(x)^† under a gauge transformation.
#[test]
fn field_strength_is_gauge_covariant() {
    let lat = Lattice::single_node([4, 4, 6, 4]);
    let st = Stencil::nearest(&lat, PERIODIC);
    let u = random_gauge(&lat, 202);
    let mut rng = TestRng::new(203);
    let g: Vec<_> = (0..lat.num_sites()).map(|_| random_su3(&mut rng)).collect();
    let (f, fg) = (
        field_strength(&u, &st),
        field_strength(&gauge_transform(&u, &g, &st), &st),
    );
    for plane in 0..PLANES.len() {
        for site in 0..lat.num_sites() {
            let want = g[site].mul(&link(&f, plane, site)).mul(&g[site].adj());
            let diff = want.sub(&link(&fg, plane, site));
            for a in 0..3 {
                for b in 0..3 {
                    assert!(diff.0[a][b].norm_sqr() < 1e-26);
                }
            }
        }
    }
}

// ---- Clover operator: checks that need no reference ----

/// Without the clover term, or without a field strength, the operator is Wilson's.
#[test]
fn reduces_to_wilson_when_the_clover_term_vanishes() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let psi = random_spinor(&lat, 1, 210);
    let wilson = |u: &Field<f64, Gauge>| {
        let mut op = Wilson::<f64>::new(
            &lat,
            WilsonParams {
                kappa: 0.12,
                boundary: MIXED,
            },
        );
        op.set_gauge(u).unwrap();
        let mut out = psi.zeros_like();
        op.mult(&mut op.work(), &mut out, &psi, Variant::D);
        out
    };

    let u = random_gauge(&lat, 211);
    assert!(
        max_abs_diff(
            &mult(&clover(&lat, 0.0, MIXED, &u), &psi, Variant::D),
            &wilson(&u)
        ) < 1e-15
    );

    let mut unit = Field::<f64, Gauge>::gauge(&lat);
    set_unit(&mut unit);
    assert!(
        max_abs_diff(
            &mult(&clover(&lat, 1.7, MIXED, &unit), &psi, Variant::D),
            &wilson(&unit)
        ) < 1e-15
    );

    let with_term = mult(&clover(&lat, 1.0, MIXED, &u), &psi, Variant::D);
    assert!(
        max_abs_diff(&with_term, &wilson(&u)) > 1e-3,
        "the clover term should do something"
    );
}

#[test]
fn gauge_covariance() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 220);
    let psi = random_spinor(&lat, 1, 221);
    let mut rng = TestRng::new(222);
    let g: Vec<_> = (0..lat.num_sites()).map(|_| random_su3(&mut rng)).collect();
    let ug = gauge_transform(&u, &g, &Stencil::nearest(&lat, PERIODIC));

    let lhs = mult(
        &clover(&lat, 1.3, MIXED, &ug),
        &gauge_transform_spinor(&psi, &g),
        Variant::D,
    );
    let rhs = gauge_transform_spinor(&mult(&clover(&lat, 1.3, MIXED, &u), &psi, Variant::D), &g);
    let diff = max_abs_diff(&lhs, &rhs);
    assert!(diff < 1e-13, "max difference {diff}");
}

#[test]
fn dagger_is_the_adjoint() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = clover(&lat, 1.0, MIXED, &random_gauge(&lat, 230));
    let (psi, chi) = (random_spinor(&lat, 1, 231), random_spinor(&lat, 1, 232));
    let (a_re, a_im) = dot(&chi, &mult(&op, &psi, Variant::D));
    let (b_re, b_im) = dot(&mult(&op, &chi, Variant::Dagger), &psi);
    assert!(a_re.abs() > 1.0);
    assert!(
        (a_re - b_re).abs() < 1e-11 && (a_im - b_im).abs() < 1e-11,
        "{a_re} {b_re} {a_im} {b_im}"
    );
}

/// The fused sweep equals the diagonal block plus the eight hopping terms, and the
/// even-odd blocks reassemble to the same thing.
#[test]
fn parts_and_blocks_reproduce_the_full_operator() {
    let lat = Lattice::single_node([4, 6, 4, 4]);
    let op = clover(&lat, 1.0, MIXED, &random_gauge(&lat, 240));
    let psi = random_spinor(&lat, 2, 241);
    let fused = mult(&op, &psi, Variant::D);
    let one = [C64::new(1.0, 0.0); 2];

    let mut work = op.work();
    let mut hops = psi.zeros_like();
    for mu in 0..ND {
        op.hop_forward(&mut work, mu, &mut hops, &psi);
        op.hop_backward(&mut work, mu, &mut hops, &psi);
    }
    let mut parts = psi.zeros_like();
    op.mult_diag(&mut work, &mut parts, &psi);
    parts.axpy(&[C64::new(op.hop_coefficient(), 0.0); 2], &hops);
    let diff = max_abs_diff(&fused, &parts);
    assert!(diff < 1e-14, "parts: max difference {diff}");

    let idx = op.even_odd_index().unwrap();
    let mut psi_e = H::spinor_half(&lat, 2, Parity::Even);
    let mut psi_o = H::spinor_half(&lat, 2, Parity::Odd);
    extract_parity(&mut psi_e, &psi, idx, Parity::Even);
    extract_parity(&mut psi_o, &psi, idx, Parity::Odd);
    let mut w = op.even_odd_work();
    let mut blocks = psi.zeros_like();
    for (to, same, other) in [
        (Parity::Even, &psi_e, &psi_o),
        (Parity::Odd, &psi_o, &psi_e),
    ] {
        let (mut out, mut hop) = (same.zeros_like(), same.zeros_like());
        op.diag(&mut w, &mut out, same, to, Dagger::No);
        op.hop(&mut w, &mut hop, other, to, Dagger::No);
        out.axpy(&one, &hop);
        insert_parity(&mut blocks, &out, idx);
    }
    let diff = max_abs_diff(&fused, &blocks);
    assert!(diff < 1e-15, "blocks: max difference {diff}");
}

#[test]
fn diagonal_block_times_its_inverse_is_the_identity() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = clover(&lat, 1.5, PERIODIC, &random_gauge(&lat, 250));
    let idx = op.even_odd_index().unwrap();
    let psi = random_spinor(&lat, 1, 251);
    let mut w = op.even_odd_work();
    for parity in [Parity::Even, Parity::Odd] {
        let mut half = H::spinor_half(&lat, 1, parity);
        extract_parity(&mut half, &psi, idx, parity);
        let (mut t, mut back) = (half.zeros_like(), half.zeros_like());
        op.diag(&mut w, &mut t, &half, parity, Dagger::No);
        op.diag_inv(&mut w, &mut back, &t, parity, Dagger::No);
        assert!(
            max_abs_diff(&t, &half) > 1e-3,
            "the block should not be the identity"
        );
        let diff = max_abs_diff(&back, &half);
        assert!(diff < 1e-13, "max difference {diff}");
    }
}

#[test]
fn even_odd_solve_solves_the_full_system() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = clover(&lat, 1.0, TIME_ANTIPERIODIC, &random_gauge(&lat, 260));
    let b = random_spinor(&lat, 1, 261);
    let stop = Stop {
        tolerance_sqr: 1e-26,
        max_iterations: 1000,
        max_restarts: 2,
    };
    let mut x = b.zeros_like();
    solve_even_odd(
        &op,
        op.even_odd_index().unwrap(),
        &mut x,
        &b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap();

    let mut r = b.zeros_like();
    Bound::new(&op, Variant::D).mult(&mut r, &x);
    r.scale(&[-1.0]);
    r.axpy(&[C64::new(1.0, 0.0)], &b);
    let residual = norm_sqr(&r) / norm_sqr(&b);
    assert!(residual < 1e-24, "residual {residual:e}");
}

/// The stored block (two packed 6 × 6 matrices in the γ5 eigenbasis) agrees with the
/// defining formula applied directly, T ψ = ψ - κ c_SW Σ_{mu<nu} F_mu_nu γ_mu γ_nu ψ,
/// for a clover coefficient other than one.
#[test]
fn diagonal_block_matches_the_direct_formula() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let u = random_gauge(&lat, 270);
    let (kappa, csw) = (0.12, 1.7);
    let op = clover(&lat, csw, PERIODIC, &u);
    let psi = random_spinor(&lat, 1, 271);
    let mut got = psi.zeros_like();
    op.mult_diag(&mut op.work(), &mut got, &psi);

    let f = field_strength(&u, &Stencil::nearest(&lat, PERIODIC));
    let mut want = psi.zeros_like();
    for site in 0..lat.num_sites() {
        let p = SiteSpinor::from_reals(psi.site(0, site));
        let mut acc = p;
        for (plane, (mu, nu)) in PLANES.into_iter().enumerate() {
            let gg = Dirac::GAMMA[mu].mul(&Dirac::GAMMA[nu]);
            let term = gg.apply(&p).color_mul(&link(&f, plane, site));
            acc = acc.sub_scaled(kappa * csw, &term);
        }
        acc.write_reals(want.site_mut(0, site));
    }
    assert!(
        max_abs_diff(&want, &psi) > 1e-2,
        "the term should be visible"
    );
    let diff = max_abs_diff(&got, &want);
    assert!(diff < 1e-14, "max difference {diff}");
}

// ---- Comparison with Bridge++ (skipped unless the test inputs are provided) ----

/// `expected_result` of the Bridge++ test `Mult_Clover`: the 2-norm of D applied once to a
/// spinor whose reals are all 1; κ = 0.12, c_SW = 1, periodic in space, antiperiodic in time.
const BRIDGE_MULT_CLOVER_NORM: f64 = 1.23817935905285e+02;

/// `expected_result` of the Bridge++ test `Mult_Clover_eo`: the same for the even-odd
/// preconditioned operator 1 - T_ee^{-1} M_eo T_oo^{-1} M_oe on the even sites.
const BRIDGE_MULT_CLOVER_EO_NORM: f64 = 8.10390521901559e+01;

fn bridge_clover() -> Option<(Arc<Lattice>, Clover<f64>)> {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return None;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = clover(
        &lat,
        1.0,
        TIME_ANTIPERIODIC,
        &read_bridge_text(&path, &lat).unwrap(),
    );
    Some((lat, op))
}

#[test]
fn bridge_mult_clover() {
    let Some((lat, op)) = bridge_clover() else {
        return;
    };
    let mut b = V::spinor(&lat, 1);
    fill(&mut b, 1.0);
    let norm = norm_sqr(&mult(&op, &b, Variant::D)).sqrt();
    eprintln!("|D b| = {norm:.14e}, Bridge++ = {BRIDGE_MULT_CLOVER_NORM:.14e}");
    let relative = (norm - BRIDGE_MULT_CLOVER_NORM).abs() / BRIDGE_MULT_CLOVER_NORM;
    assert!(relative < 1e-13, "relative difference {relative:e}");
}

#[test]
fn bridge_mult_clover_eo() {
    let Some((lat, op)) = bridge_clover() else {
        return;
    };
    let mut b = H::spinor_half(&lat, 1, Parity::Even);
    fill(&mut b, 1.0);
    let odd = H::spinor_half(&lat, 1, Parity::Odd);
    // Bridge++ normalises the preconditioned operator by the inverse diagonal block:
    // 1 - T_ee^{-1} M_eo T_oo^{-1} M_oe = T_ee^{-1} (T_ee - M_eo T_oo^{-1} M_oe).
    let (mut s_b, mut y) = (b.zeros_like(), b.zeros_like());
    Schur::new(&op, Parity::Even, &b, &odd).mult(&mut s_b, &b);
    op.diag_inv(
        &mut op.even_odd_work(),
        &mut y,
        &s_b,
        Parity::Even,
        Dagger::No,
    );
    let norm = norm_sqr(&y).sqrt();
    eprintln!("|T^-1 S b| = {norm:.14e}, Bridge++ = {BRIDGE_MULT_CLOVER_EO_NORM:.14e}");
    let relative = (norm - BRIDGE_MULT_CLOVER_EO_NORM).abs() / BRIDGE_MULT_CLOVER_EO_NORM;
    assert!(relative < 1e-13, "relative difference {relative:e}");
}
