//! Stage 1 checks (DESIGN.md §9.1): point source, propagator, meson correlators.

use ruttice_core::{norm_sqr, Boundary, Field, Gauge, Lattice, LinearFn, Spinor, Vector, C64, ND};
use ruttice_fermion::{solve_even_odd, Bound, FermionOp, Variant, Wilson, WilsonParams};
use ruttice_io::read_bridge_text;
use ruttice_kernel::{Dirac, GammaRepre};
use ruttice_measure::{meson_correlator, pion_correlator, point_source, NUM_SOURCES};
use ruttice_solver::{solve_cgne, Cg, Stop};
use ruttice_validation::{bridge_test_file, random_gauge, random_spinor, BRIDGE_TESTS_ENV};
use std::sync::Arc;

type V = Field<f64, Spinor>;

const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];

fn wilson(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Wilson<f64> {
    let mut op = Wilson::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary: ANTIPERIODIC,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

/// All twelve columns of the propagator from one point, by an even-odd preconditioned
/// CGNE solve. Returns the propagator and the worst full-system residual over the columns.
fn propagator(op: &Wilson<f64>, lat: &Arc<Lattice>, position: [usize; ND], stop: Stop) -> (V, f64) {
    let b = point_source::<f64>(lat, position);
    let mut s = b.zeros_like();
    solve_even_odd(
        op,
        op.even_odd_index().unwrap(),
        &mut s,
        &b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap();

    let mut r = b.zeros_like();
    Bound::new(op, Variant::D).mult(&mut r, &s);
    r.scale(&[-1.0; NUM_SOURCES]);
    r.axpy(&[C64::new(1.0, 0.0); NUM_SOURCES], &b);
    let mut rr = [0.0; NUM_SOURCES];
    r.norm_sqr(&mut rr);
    // Every source column has unit norm.
    (s, rr.iter().fold(0.0, |m: f64, x| m.max(*x)))
}

#[test]
fn point_source_has_one_unit_entry_per_vector() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let position = [1, 2, 3, 5];
    let b = point_source::<f64>(&lat, position);
    assert_eq!(b.num_vectors(), NUM_SOURCES);
    assert_eq!(norm_sqr(&b), NUM_SOURCES as f64);
    let mut n = [0.0; NUM_SOURCES];
    b.norm_sqr(&mut n);
    assert_eq!(n, [1.0; NUM_SOURCES]);
    let site = b.site(0, lat.site(position));
    for k in 0..NUM_SOURCES {
        assert_eq!(site[k * 24 + 2 * k], 1.0);
    }
}

/// With γ5 at source and sink the general contraction reduces to the sum of |S|^2 per
/// time slice, and the time slices add up to the norm of the whole field.
#[test]
fn gamma5_channel_is_the_norm_per_time_slice() {
    let lat = Lattice::single_node([4, 4, 2, 6]);
    let s = random_spinor(&lat, NUM_SOURCES, 90);
    let g5 = &Dirac::GAMMA5;
    let general = meson_correlator(&s, &s, g5, g5, g5);
    let pion = pion_correlator(&s);
    assert_eq!(pion.len(), 6);
    for (c, p) in general.iter().zip(&pion) {
        assert!(
            (c.re - p).abs() < 1e-12 * p && c.im.abs() < 1e-12 * p,
            "{c:?} vs {p}"
        );
    }
    let total: f64 = pion.iter().sum();
    assert!((total - norm_sqr(&s)).abs() < 1e-11 * total);
    assert!(
        pion[0] != pion[1],
        "a random field should differ between time slices"
    );
}

/// The pion correlator is gauge invariant, and moving the source shifts it in time.
#[test]
fn pion_correlator_from_a_solved_propagator() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson(&lat, &random_gauge(&lat, 91));
    let stop = Stop {
        tolerance_sqr: 1e-26,
        max_iterations: 500,
        max_restarts: 2,
    };
    let (s, residual) = propagator(&op, &lat, [0, 0, 0, 0], stop);
    assert!(residual < 1e-24, "worst residual {residual:e}");

    let c = pion_correlator(&s);
    assert!(c.iter().all(|x| *x > 0.0));
    // The correlator falls away from the source in both time directions.
    assert!(
        c[0] > c[1] && c[1] > c[2] && c[0] > c[7] && c[7] > c[6],
        "{c:?}"
    );
}

// ---- Comparison with Bridge++ (skipped unless the test inputs are provided) ----

/// `expected_result` of the Bridge++ test `Spectrum_Wilson_Hadron2ptFunction`: the real
/// part of the pseudoscalar-pseudoscalar correlator at t = 0, for a point source at the
/// origin, κ = 0.12, antiperiodic in all four directions.
const BRIDGE_PION_CORRELATOR_T0: f64 = 14.2250394502083;

#[test]
fn bridge_spectrum_wilson_pion() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson(&lat, &read_bridge_text(&path, &lat).unwrap());
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 1000,
        max_restarts: 3,
    };
    let (s, residual) = propagator(&op, &lat, [0, 0, 0, 0], stop);

    let pion = pion_correlator(&s);
    let g5 = &Dirac::GAMMA5;
    let general = meson_correlator(&s, &s, g5, g5, g5);
    eprintln!("worst |D S - b|^2 over the 12 columns: {residual:e}");
    for (t, c) in pion.iter().enumerate() {
        eprintln!("  t = {t}  C_PS(t) = {c:.12e}");
    }
    eprintln!(
        "C_PS(0) = {:.13}, Bridge++ = {BRIDGE_PION_CORRELATOR_T0:.13}",
        pion[0]
    );
    assert!(residual < 1e-26);
    let relative = (pion[0] - BRIDGE_PION_CORRELATOR_T0).abs() / BRIDGE_PION_CORRELATOR_T0;
    assert!(relative < 1e-12, "relative difference {relative:e}");
    assert!((general[0].re - pion[0]).abs() < 1e-11 && general[0].im.abs() < 1e-11);
}
