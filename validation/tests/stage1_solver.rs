//! Stage 1 checks (DESIGN.md §9.1): solvers on the Wilson operator.

use ruttice_core::{
    Adjoint, Boundary, ConvertFrom, Field, Gauge, Lattice, LinearFn, Product, Spinor, Vector, ND,
};
use ruttice_fermion::{Bound, FermionOp, Variant, Wilson, WilsonParams};
use ruttice_io::read_bridge_text;
use ruttice_solver::{BiCgStab, Cg, Convert, Refine, Solve, SolveError, Solver, Stop};
use ruttice_validation::{
    bridge_test_file, max_abs_diff, random_gauge, random_spinor, BRIDGE_TESTS_ENV,
};
use std::sync::Arc;

type V64 = Field<f64, Spinor>;
type V32 = Field<f32, Spinor>;

const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];
const PARAMS: WilsonParams = WilsonParams {
    kappa: 0.12,
    boundary: ANTIPERIODIC,
};

fn wilson<T: ruttice_core::Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Wilson<T> {
    let mut op = Wilson::new(lat, PARAMS);
    op.set_gauge(u).unwrap();
    op
}

/// D†D as a composition of two bindings of the same operator.
fn ddag_d<'a, T: ruttice_core::Storage>(
    op: &'a Wilson<T>,
    like: &Field<T, Spinor>,
) -> impl LinearFn<Field<T, Spinor>> + 'a {
    Product::new(
        Adjoint(Bound::new(op, Variant::D)),
        Bound::new(op, Variant::D),
        like,
    )
}

/// `|b - A x|^2 / |b|^2` per vector, computed here rather than taken from the solver.
fn residual_sqr(a: &mut dyn LinearFn<V64>, x: &V64, b: &V64) -> Vec<f64> {
    let nv = b.num_vectors();
    let mut r = b.zeros_like();
    a.mult(&mut r, x);
    r.scale(&vec![-1.0; nv]);
    r.axpy(&vec![ruttice_core::C64::new(1.0, 0.0); nv], b);
    let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    rr.iter()
        .zip(&bb)
        .map(|(r, b)| if *b > 0.0 { r / b } else { *r })
        .collect()
}

#[test]
fn cg_solves_the_normal_equations() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 1));
    let b = random_spinor(&lat, 1, 2);
    let mut a = ddag_d(&op, &b);

    let mut x = b.zeros_like();
    let info = Cg::new(Stop::simple(1e-24, 500))
        .solve(&mut a, None, &mut x, &b)
        .unwrap();
    assert!(
        info.iterations > 5,
        "suspiciously few iterations: {}",
        info.iterations
    );
    let r = residual_sqr(&mut a, &x, &b);
    assert!(
        r[0] <= 1.1e-24,
        "true residual {:e} after {} iterations",
        r[0],
        info.iterations
    );
}

/// Three right-hand sides solved together, one of them zero, give the same answers as
/// three separate solves.
#[test]
fn cg_with_several_vectors_matches_separate_solves() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 3));
    let singles = [
        random_spinor(&lat, 1, 10),
        V64::spinor(&lat, 1),
        random_spinor(&lat, 1, 11),
    ];

    let mut b = V64::spinor(&lat, 3);
    for site in 0..lat.num_sites() {
        for (j, s) in singles.iter().enumerate() {
            b.site_mut(0, site)[j * 24..(j + 1) * 24].copy_from_slice(s.site(0, site));
        }
    }
    let stop = Stop::simple(1e-26, 500);
    let mut x = b.zeros_like();
    let info = Cg::new(stop)
        .solve(&mut ddag_d(&op, &b), None, &mut x, &b)
        .unwrap();
    assert_eq!(info.residual_sqr[1], 0.0);

    for (j, s) in singles.iter().enumerate() {
        let mut xs = s.zeros_like();
        Cg::new(stop)
            .solve(&mut ddag_d(&op, s), None, &mut xs, s)
            .unwrap();
        let mut column = s.zeros_like();
        for site in 0..lat.num_sites() {
            column
                .site_mut(0, site)
                .copy_from_slice(&x.site(0, site)[j * 24..(j + 1) * 24]);
        }
        let diff = max_abs_diff(&column, &xs);
        assert!(diff < 1e-11, "vector {j}: max difference {diff}");
    }
}

#[test]
fn bicgstab_solves_d() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 4));
    let b = random_spinor(&lat, 2, 5);
    let mut a = Bound::new(&op, Variant::D);

    let mut x = b.zeros_like();
    let info = BiCgStab::new(Stop::simple(1e-24, 500))
        .solve(&mut a, None, &mut x, &b)
        .unwrap();
    assert!(info.iterations > 3);
    for r in residual_sqr(&mut a, &x, &b) {
        assert!(r <= 1.1e-24, "true residual {r:e}");
    }
}

/// Double-precision accuracy from single-precision solves: defect correction outside,
/// a converted f32 CG inside.
#[test]
fn mixed_precision_reaches_double_accuracy() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 6);
    let (op64, op32) = (wilson::<f64>(&lat, &u), wilson::<f32>(&lat, &u));
    let b = random_spinor(&lat, 1, 7);
    let like32 = V32::spinor(&lat, 1);

    // Single precision alone cannot get there.
    let mut b32 = like32.zeros_like();
    b32.convert_from(&b);
    let mut x32 = like32.zeros_like();
    let alone = Cg::new(Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 300,
        max_restarts: 1,
    })
    .solve(&mut ddag_d(&op32, &like32), None, &mut x32, &b32);
    assert!(
        matches!(alone, Err(SolveError::NotConverged(_))),
        "{alone:?}"
    );

    let mut inner = Cg::<V32>::new(Stop::simple(1e-10, 300));
    let mut approx = Convert::new(Solve::new(&mut inner, ddag_d(&op32, &like32)), &like32);
    let mut a = ddag_d(&op64, &b);
    let mut x = b.zeros_like();
    let info = Refine::new(Stop::simple(1e-24, 20))
        .solve(&mut a, Some(&mut approx), &mut x, &b)
        .unwrap();

    assert!(
        info.iterations >= 2 && info.iterations <= 8,
        "{} outer iterations",
        info.iterations
    );
    let r = residual_sqr(&mut a, &x, &b);
    assert!(r[0] <= 1.1e-24, "true residual {:e}", r[0]);
}

/// The ownership pattern of DESIGN.md §7.3 in use: an outer and an inner solver share
/// operators through short-lived bindings, and the gauge field is replaced between solves.
#[test]
fn nested_solves_across_gauge_updates() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let b = random_spinor(&lat, 1, 20);
    let like32 = V32::spinor(&lat, 1);
    let mut op64 = Wilson::<f64>::new(&lat, PARAMS);
    let mut op32 = Wilson::<f32>::new(&lat, PARAMS);
    // Long-lived solver state, created once.
    let mut inner = Cg::<V32>::new(Stop::simple(1e-10, 300));
    let mut outer = Refine::<V64>::new(Stop::simple(1e-22, 20));

    let mut previous: Option<V64> = None;
    for step in 0..3 {
        let u = random_gauge(&lat, 100 + step);
        // Needs exclusive access: compiles only because last step's bindings are gone.
        op64.set_gauge(&u).unwrap();
        op32.set_gauge(&u).unwrap();

        let mut approx = Convert::new(Solve::new(&mut inner, ddag_d(&op32, &like32)), &like32);
        let mut a = ddag_d(&op64, &b);
        let mut x = b.zeros_like();
        outer.solve(&mut a, Some(&mut approx), &mut x, &b).unwrap();
        assert!(residual_sqr(&mut a, &x, &b)[0] <= 1.1e-22);

        if let Some(p) = &previous {
            assert!(
                max_abs_diff(p, &x) > 1e-3,
                "a new gauge field must give a new solution"
            );
        }
        previous = Some(x);
    }
}

// ---- Comparison with Bridge++ (skipped unless the test inputs are provided) ----

/// `expected_result` of the Bridge++ test `Solver_Wilson_CGNE`: round(Nconv / 10), where
/// Nconv is one plus the number of CG steps on D D† for a point source at the origin
/// (spin 0, colour 0), starting from x = b, restarting every 100 steps, until
/// |r|^2 / |b|^2 < 1e-28.
const BRIDGE_CGNE_NCONV_OVER_10: f64 = 11.0;

#[test]
fn bridge_solver_wilson_cgne_iteration_count() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson::<f64>(&lat, &read_bridge_text(&path, &lat).unwrap());

    let mut b = V64::spinor(&lat, 1);
    b.site_mut(0, 0)[0] = 1.0;

    // CGNE: solve D D† y = b, then x = D† y.
    let mut d_ddag = Product::new(
        Bound::new(&op, Variant::D),
        Adjoint(Bound::new(&op, Variant::D)),
        &b,
    );
    let mut y = b.zeros_like();
    y.copy_from(&b);
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 100,
        max_restarts: 40,
    };
    let info = Cg::new(stop).solve(&mut d_ddag, None, &mut y, &b).unwrap();
    let mut x = b.zeros_like();
    Bound::new(&op, Variant::Dagger).mult(&mut x, &y);

    let nconv = info.iterations + 1;
    let r = residual_sqr(&mut Bound::new(&op, Variant::D), &x, &b);
    eprintln!(
        "Nconv = {nconv} ({} restarts), |D x - b|^2/|b|^2 = {:e}; Bridge++ round(Nconv/10) = {}",
        info.restarts, r[0], BRIDGE_CGNE_NCONV_OVER_10
    );
    assert!(r[0] < 1e-26);
    assert_eq!((nconv as f64 / 10.0).round(), BRIDGE_CGNE_NCONV_OVER_10);
}
