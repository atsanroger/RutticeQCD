//! Comparisons against output of the Bridge++ test program that we produced ourselves
//!: every meson channel and time slice, a second quark flavour, an
//! exponential source, Coulomb gauge fixing, and a clover coefficient other than one. The numbers are in
//! `tests/data/`, with their provenance in the file headers; the gauge configuration is
//! still read from `RUTTICE_BRIDGE_TESTS`.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::fermion::{
    solve_even_odd, Clover, CloverParams, DiagTerm, FermionOp, Variant, Wilson, WilsonOp,
    WilsonParams,
};
use ruttice::gauge::{fix_coulomb, plaquette, CoulombParams};
use ruttice::io::read_bridge_text;
use ruttice::kernel::{Dirac, GammaRepre};
use ruttice::measure::{
    channel, exponential_source, meson_correlator, meson_correlator_with_momentum, point_source,
    proton_correlator,
};
use ruttice::solver::{solve_cgne, Cg, Stop};
use ruttice::{
    fill, norm_sqr, Boundary, Field, Gauge, Lattice, Spinor, Stencil, Storage, Vector, C64, ND,
};
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
    let (s1, s2) = clover_two_flavours(&lat, &u);
    let n = compare(
        include_str!("data/bridge_spectrum_clover_nogfix.txt"),
        &[&s1, &s2],
    );
    assert_eq!(n, 3 * 13 * 8);
}

/// Meson correlators with momentum, Bridge++'s `meson_momentum_all`: PS, V1, V2, V3 at ten
/// momenta for the pairs (1,1), (2,2) and (1,2) of `clover_two_flavours_exponential_source`.
#[test]
fn clover_mesons_with_momentum() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let Some(u) = gauge(&lat) else { return };
    let (s1, s2) = clover_two_flavours(&lat, &u);
    let props = [&s1, &s2];
    let table = include_str!("data/bridge_momentum_clover.txt");
    let rows: Vec<Vec<&str>> = table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().collect())
        .collect();
    let scale = rows.iter().fold(0.0f64, |m, f| {
        m.max(
            C64::new(f[4].parse().unwrap(), f[5].parse().unwrap())
                .norm_sqr()
                .sqrt(),
        )
    });
    // The table carries 13 significant digits.
    let tolerance = 2e-12 * scale;
    let mut worst = 0.0f64;
    let mut cache: Option<(String, Vec<C64>)> = None;
    for f in &rows {
        let key = format!("{} {} {}", f[0], f[1], f[2]);
        if !cache.as_ref().is_some_and(|(k, _)| *k == key) {
            let flavour = |i: usize| props[f[0].as_bytes()[i] as usize - b'1' as usize];
            let n: [i64; 3] = std::array::from_fn(|mu| i64::from(f[2].as_bytes()[mu] - b'0'));
            let (sink, source) = channel::<Dirac>(f[1]).expect("known channel");
            let c = meson_correlator_with_momentum(
                flavour(0),
                flavour(1),
                &sink,
                &source,
                &Dirac::GAMMA5,
                n,
                [0; ND],
            );
            cache = Some((key.clone(), c));
        }
        let t: usize = f[3].parse().unwrap();
        let ours = cache.as_ref().unwrap().1[t];
        let reference = C64::new(f[4].parse().unwrap(), f[5].parse().unwrap());
        let diff = (ours - reference).norm_sqr().sqrt();
        worst = worst.max(diff);
        assert!(
            diff <= tolerance,
            "{key} t = {t}: ours {ours:?}, reference {reference:?}"
        );
    }
    eprintln!(
        "{} values compared, largest difference {worst:.2e} (tolerance {tolerance:.2e})",
        rows.len()
    );
    assert_eq!(rows.len(), 3 * 4 * 10 * 8);
}

/// Proton correlators, Bridge++'s `proton_test`: Γ = 1, γ4 and their mean (UPPER), with
/// u = flavour 1 and d = flavour 2, and with both from flavour 1.
#[test]
fn clover_proton() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let Some(u) = gauge(&lat) else { return };
    let (s1, s2) = clover_two_flavours(&lat, &u);
    let props = [&s1, &s2];
    let identity = Dirac::GAMMA[3].mul(&Dirac::GAMMA[3]);
    let table = include_str!("data/bridge_proton_clover.txt");
    let rows: Vec<Vec<&str>> = table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().collect())
        .collect();
    let scale = rows.iter().fold(0.0f64, |m, f| {
        m.max(
            C64::new(f[3].parse().unwrap(), f[4].parse().unwrap())
                .norm_sqr()
                .sqrt(),
        )
    });
    // The table carries 13 significant digits.
    let tolerance = 2e-12 * scale;
    let mut worst = 0.0f64;
    for flavours in ["12", "11"] {
        let flavour = |i: usize| props[flavours.as_bytes()[i] as usize - b'1' as usize];
        let proton =
            |g| proton_correlator::<Dirac, f64>(flavour(0), flavour(1), g, [0; 3], [0; ND]);
        let (unity, gamma4) = (proton(&identity), proton(&Dirac::GAMMA[3]));
        for f in rows.iter().filter(|f| f[0] == flavours) {
            let t: usize = f[2].parse().unwrap();
            let ours = match f[1] {
                "UNITY" => unity[t],
                "GAMMA0" => gamma4[t],
                "UPPER" => (unity[t] + gamma4[t]).scale(0.5),
                other => panic!("unknown projection {other}"),
            };
            let reference = C64::new(f[3].parse().unwrap(), f[4].parse().unwrap());
            let diff = (ours - reference).norm_sqr().sqrt();
            worst = worst.max(diff);
            assert!(
                diff <= tolerance,
                "{flavours} {} t = {t}: ours {ours:?}, reference {reference:?}",
                f[1]
            );
        }
    }
    eprintln!(
        "{} values compared, largest difference {worst:.2e} (tolerance {tolerance:.2e})",
        rows.len()
    );
    assert_eq!(rows.len(), 2 * 3 * 8);
}

/// Two clover propagators from exponential sources, for the flavours of the Bridge++ test
/// `Spectrum_Clover_Hadron2ptFunction`.
fn clover_two_flavours(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> (V, V) {
    // Each flavour has its own hopping parameter and its own source width.
    let solve = |kappa: f64, slope: f64| {
        let mut op = Clover::<f64>::new(
            lat,
            CloverParams {
                kappa,
                csw: 1.0,
                boundary: TIME_ANTIPERIODIC,
            },
        );
        op.set_gauge(u).unwrap();
        propagator(&op, &exponential_source::<f64>(lat, [0; ND], slope, 2.0))
    };
    (solve(0.12, 0.25), solve(0.115, 0.5))
}

/// The unmodified Bridge++ test `Spectrum_Clover_Hadron2ptFunction`: Coulomb gauge first,
/// then the same as `clover_two_flavours_exponential_source`. Its `expected_result` is the
/// first row of the table.
#[test]
fn clover_two_flavours_coulomb_gauge() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let Some(mut u) = gauge(&lat) else { return };
    let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
    let before = plaquette(&u, &st);
    let report = fix_coulomb(&mut u, &CoulombParams::default()).unwrap();
    // Bridge++ printed "converged at iter = 120" and a plaquette change of 4.9e-15.
    assert_eq!(report.iterations, 120);
    assert!((plaquette(&u, &st) - before).abs() < 1e-13);
    let (s1, s2) = clover_two_flavours(&lat, &u);
    let n = compare(
        include_str!("data/bridge_spectrum_clover_coulomb.txt"),
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
