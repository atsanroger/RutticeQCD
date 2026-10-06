//! Comparisons against output of the Bridge++ test program that we produced ourselves
//!: every meson channel and time slice, a second quark flavour, an
//! exponential source, and a clover coefficient other than one. The numbers are in
//! `tests/data/`, with their provenance in the file headers; the gauge configuration is
//! still read from `RUTTICE_BRIDGE_TESTS`.

use ruttice_core::{
    fill, norm_sqr, Boundary, Field, Gauge, Lattice, Spinor, Storage, Vector, C64, ND,
};
use ruttice_fermion::{
    solve_even_odd, Clover, CloverParams, DiagTerm, FermionOp, Variant, Wilson, WilsonOp,
    WilsonParams,
};
use ruttice_io::read_bridge_text;
use ruttice_kernel::{Dirac, GammaRepre};
use ruttice_measure::{channel, exponential_source, meson_correlator, point_source};
use ruttice_solver::{solve_cgne, Cg, Stop};
use ruttice_validation::{bridge_test_file, BRIDGE_TESTS_ENV};
use std::sync::Arc;

type V = Field<f64, Spinor>;

const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];
const TIME_ANTIPERIODIC: [Boundary; ND] = [
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];

struct Row {
    flavours: String,
    channel: String,
    t: usize,
    value: C64,
}

fn parse(table: &str) -> Vec<Row> {
    table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            Row {
                flavours: f[0].to_string(),
                channel: f[1].to_string(),
                t: f[2].parse().unwrap(),
                value: C64::new(f[3].parse().unwrap(), f[4].parse().unwrap()),
            }
        })
        .collect()
}

fn gauge(lat: &Arc<Lattice>) -> Option<Field<f64, Gauge>> {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return None;
    };
    Some(read_bridge_text(&path, lat).unwrap())
}

/// Solve D S = b for every vector of `b` by even-odd preconditioned CGNE.
fn propagator<T: Storage, D: DiagTerm<T>>(
    op: &WilsonOp<T, Dirac, D>,
    b: &Field<T, Spinor>,
) -> Field<T, Spinor> {
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 2000,
        max_restarts: 3,
    };
    let mut s = b.zeros_like();
    solve_even_odd(
        op,
        op.even_odd_index().unwrap(),
        &mut s,
        b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap();
    s
}

/// Compare every row of a table with `meson_correlator`. `props` maps a flavour digit to
/// a propagator. Returns the number of rows compared.
fn compare(table: &str, props: &[&V]) -> usize {
    let rows = parse(table);
    let scale = rows
        .iter()
        .fold(0.0f64, |m, r| m.max(r.value.norm_sqr().sqrt()));
    // The tables carry 13 significant digits.
    let tolerance = 2e-12 * scale;
    let mut worst = 0.0f64;
    let mut cache: Option<(String, String, Vec<C64>)> = None;
    for row in &rows {
        if !cache
            .as_ref()
            .is_some_and(|(f, c, _)| *f == row.flavours && *c == row.channel)
        {
            let flavour = |i: usize| props[row.flavours.as_bytes()[i] as usize - b'1' as usize];
            let (sink, source) = channel::<Dirac>(&row.channel).expect("known channel");
            let c = meson_correlator(flavour(0), flavour(1), &sink, &source, &Dirac::GAMMA5);
            cache = Some((row.flavours.clone(), row.channel.clone(), c));
        }
        let ours = cache.as_ref().unwrap().2[row.t];
        let diff = (ours - row.value).norm_sqr().sqrt();
        worst = worst.max(diff);
        assert!(
            diff <= tolerance,
            "flavours {} channel {} t = {}: ours {:?}, reference {:?}",
            row.flavours,
            row.channel,
            row.t,
            ours,
            row.value
        );
    }
    eprintln!(
        "{} values compared, largest difference {worst:.2e} (tolerance {tolerance:.2e})",
        rows.len()
    );
    rows.len()
}

/// All thirteen meson channels on all time slices for Wilson fermions with a point source.
/// Bridge++ fixed the gauge to Landau gauge first; these correlators do not depend on it.
#[test]
fn wilson_meson_channels() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let Some(u) = gauge(&lat) else { return };
    let mut op = Wilson::<f64>::new(
        &lat,
        WilsonParams {
            kappa: 0.12,
            boundary: ANTIPERIODIC,
        },
    );
    op.set_gauge(&u).unwrap();
    let s = propagator(&op, &point_source::<f64>(&lat, [0; ND]));
    let n = compare(include_str!("data/bridge_spectrum_wilson.txt"), &[&s]);
    assert_eq!(n, 13 * 8);
}

/// Clover fermions, two flavours, exponential source, no gauge fixing on either side:
/// all channels and time slices for the pairs (1,1), (2,2) and (1,2).
#[test]
fn clover_two_flavours_exponential_source() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let Some(u) = gauge(&lat) else { return };
    // Each flavour has its own hopping parameter and its own source width.
    let solve = |kappa: f64, slope: f64| {
        let mut op = Clover::<f64>::new(
            &lat,
            CloverParams {
                kappa,
                csw: 1.0,
                boundary: TIME_ANTIPERIODIC,
            },
        );
        op.set_gauge(&u).unwrap();
        propagator(&op, &exponential_source::<f64>(&lat, [0; ND], slope, 2.0))
    };
    let (s1, s2) = (solve(0.12, 0.25), solve(0.115, 0.5));
    let n = compare(
        include_str!("data/bridge_spectrum_clover_nogfix.txt"),
        &[&s1, &s2],
    );
    assert_eq!(n, 3 * 13 * 8);
}

/// The Bridge++ test `Mult_Clover` rerun with clover_coefficient changed from 1.0 to 1.7.
const BRIDGE_MULT_CLOVER_CSW_1_7_NORM: f64 = 1.24728260729018e+02;

#[test]
fn clover_mult_with_another_coefficient() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let Some(u) = gauge(&lat) else { return };
    let mut op = Clover::<f64>::new(
        &lat,
        CloverParams {
            kappa: 0.12,
            csw: 1.7,
            boundary: TIME_ANTIPERIODIC,
        },
    );
    op.set_gauge(&u).unwrap();
    let mut b = V::spinor(&lat, 1);
    fill(&mut b, 1.0);
    let mut y = b.zeros_like();
    op.mult(&mut op.work(), &mut y, &b, Variant::D);
    let norm = norm_sqr(&y).sqrt();
    eprintln!("|D b| = {norm:.14e}, Bridge++ = {BRIDGE_MULT_CLOVER_CSW_1_7_NORM:.14e}");
    let relative = (norm - BRIDGE_MULT_CLOVER_CSW_1_7_NORM).abs() / BRIDGE_MULT_CLOVER_CSW_1_7_NORM;
    assert!(relative < 1e-13, "relative difference {relative:e}");
}
