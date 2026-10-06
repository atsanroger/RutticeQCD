//! Checks: configuration I/O and the plaquette.

mod common;

use common::{bridge_test_file, gauge_transform, BRIDGE_TESTS_ENV};
use ruttice::gauge::{link, plaquette, set_unit};
use ruttice::gauge::{random_gauge, random_su3};
use ruttice::io::read_bridge_text;
use ruttice::kernel::ColorMatrix;
use ruttice::TestRng;
use ruttice::{Boundary, Field, Gauge, Lattice, Stencil, ND};

const PERIODIC: [Boundary; ND] = [Boundary::Periodic; ND];

fn su3_defect(m: &ColorMatrix<f64>) -> f64 {
    let p = m.mul(&m.adj());
    let id = ColorMatrix::<f64>::identity();
    let mut d = 0.0f64;
    for a in 0..3 {
        for b in 0..3 {
            d = d.max((p.0[a][b] - id.0[a][b]).norm_sqr().sqrt());
        }
    }
    d
}

// ---- Physics checks: no reference values needed ----

#[test]
fn unit_gauge_has_plaquette_one() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let st = Stencil::nearest(&lat, PERIODIC);
    let mut u = Field::<f64, Gauge>::gauge(&lat);
    set_unit(&mut u);
    assert_eq!(plaquette(&u, &st), 1.0);
}

#[test]
fn plaquette_is_gauge_invariant() {
    let lat = Lattice::single_node([4, 6, 4, 8]);
    let st = Stencil::nearest(&lat, PERIODIC);
    let u = random_gauge(&lat, 11);
    let mut rng = TestRng::new(29);
    let g: Vec<_> = (0..lat.num_sites()).map(|_| random_su3(&mut rng)).collect();
    let v = gauge_transform(&u, &g, &st);

    let (p, q) = (plaquette(&u, &st), plaquette(&v, &st));
    assert!(
        p.abs() < 0.1,
        "a random configuration should have a plaquette near zero, got {p}"
    );
    assert!((p - q).abs() < 1e-14, "{p} vs {q}");
    assert_ne!(u.as_slice(), v.as_slice());
}

#[test]
fn plaquette_does_not_depend_on_thread_count() {
    let lat = Lattice::single_node([8, 8, 8, 8]);
    let st = Stencil::nearest(&lat, PERIODIC);
    let u = random_gauge(&lat, 5);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| plaquette(&u, &st))
    };
    let p1 = run(1);
    assert_eq!(run(3), p1);
    assert_eq!(run(8), p1);
}

// ---- Comparison with Bridge++ (skipped unless the test inputs are provided) ----

/// `expected_result` of the Bridge++ test `Gauge_lex` (tests/test_Gauge_lex.yaml).
const BRIDGE_GAUGE_LEX_PLAQUETTE: f64 = 0.625129946974942;

#[test]
fn bridge_gauge_lex_plaquette() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = read_bridge_text(&path, &lat).unwrap();

    let mut worst = 0.0f64;
    for mu in 0..ND {
        for site in 0..lat.num_sites() {
            worst = worst.max(su3_defect(&link(&u, mu, site)));
        }
    }
    assert!(worst < 1e-12, "links are not unitary: defect {worst}");

    let p = plaquette(&u, &Stencil::nearest(&lat, PERIODIC));
    eprintln!("plaquette = {p:.15}, Bridge++ = {BRIDGE_GAUGE_LEX_PLAQUETTE:.15}");
    assert!((p - BRIDGE_GAUGE_LEX_PLAQUETTE).abs() < 1e-13, "{p}");
}
