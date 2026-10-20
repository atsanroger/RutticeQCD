//! Checks: solvers on the Wilson operator.

mod common;

use common::{bridge_test_file, max_abs_diff, BRIDGE_TESTS_ENV};
use ruttice::fermion::{Bound, FermionOp, Variant, Wilson, WilsonParams};
use ruttice::gauge::random_gauge;
use ruttice::io::read_bridge_text;
use ruttice::random_spinor;
use ruttice::solver::{
    reliable_cg, BiCgStab, Cg, Convert, Refine, Solve, SolveError, Solver, Stop,
};
use ruttice::{
    Adjoint, Boundary, ConvertFrom, Field, Gauge, Lattice, LinearFn, Product, Spinor, Vector, ND,
};
use std::sync::Arc;

type V64 = Field<f64, Spinor>;
type V32 = Field<f32, Spinor>;

const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];
const PARAMS: WilsonParams = WilsonParams {
    kappa: 0.12,
    boundary: ANTIPERIODIC,
};

fn wilson<T: ruttice::Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Wilson<T> {
    let mut op = Wilson::new(lat, PARAMS);
    op.set_gauge(u).unwrap();
    op
}

/// D†D as a composition of two bindings of the same operator.
fn ddag_d<'a, T: ruttice::Storage>(
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
    r.axpy(&vec![ruttice::C64::new(1.0, 0.0); nv], b);
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

/// A linear map that counts the vectors it is applied to.
struct Counted<'a, A> {
    a: A,
    vectors: &'a std::cell::Cell<usize>,
}

impl<A: LinearFn<V64>> LinearFn<V64> for Counted<'_, A> {
    fn mult(&mut self, out: &mut V64, inp: &V64) {
        self.vectors.set(self.vectors.get() + inp.num_vectors());
        self.a.mult(out, inp);
    }
}

/// Vectors that have converged leave the iteration: the solver goes on with a smaller set, the
/// operator is applied to fewer vectors, and the vectors that go on get the bits they get when they
/// are solved by themselves.
#[test]
fn converged_vectors_leave_the_iteration() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 31));
    let hard = random_spinor(&lat, 2, 32);
    let stop = Stop::simple(1e-24, 500);
    let (fresh, solved) = ([0, 2], [1, 3]);
    let mut four = hard.zeros_with(4);
    four.assign(&fresh, &hard);
    four.assign(&solved, &hard);

    for name in ["cg", "bicgstab"] {
        let count = std::cell::Cell::new(0);
        let run = |x: &mut V64, b: &V64| {
            count.set(0);
            let info = if name == "cg" {
                let mut a = Counted {
                    a: ddag_d(&op, b),
                    vectors: &count,
                };
                Cg::new(stop).solve(&mut a, None, x, b).unwrap()
            } else {
                let mut a = Counted {
                    a: Bound::new(&op, Variant::D),
                    vectors: &count,
                };
                BiCgStab::new(stop).solve(&mut a, None, x, b).unwrap()
            };
            (info, count.get())
        };

        let mut alone = hard.zeros_like();
        let (info_alone, vectors_alone) = run(&mut alone, &hard);

        let mut x = four.zeros_like();
        x.assign(&solved, &alone);
        let (info, vectors) = run(&mut x, &four);

        assert_eq!(info.iterations, info_alone.iterations, "{name}");
        assert!(
            x.select(&fresh).as_slice() == alone.as_slice(),
            "{name}: the vectors that went on differ from the ones solved alone"
        );
        assert!(
            x.select(&solved).as_slice() == alone.as_slice(),
            "{name}: the vectors that had converged were changed"
        );
        assert!(info.residual_sqr.iter().all(|r| *r <= 1e-24), "{name}");
        // All four vectors only for the first residual and for the last one.
        assert_eq!(vectors, vectors_alone + 2 * 2, "{name}");
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

/// A point source makes BiCGStab break down on its second step: hopping out and straight
/// back is projected to zero, so the residual has no component along the source. The
/// solver has to recover by starting a new cycle, not report failure.
#[test]
fn bicgstab_recovers_from_the_point_source_breakdown() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 8));
    let mut b = V64::spinor(&lat, 1);
    b.site_mut(0, lat.site([1, 2, 3, 4]))[10] = 1.0;
    let mut a = Bound::new(&op, Variant::D);

    let mut x = b.zeros_like();
    let info = BiCgStab::new(Stop::simple(1e-24, 500))
        .solve(&mut a, None, &mut x, &b)
        .unwrap();
    assert!(
        info.restarts >= 1,
        "expected a breakdown and a new cycle, got {info:?}"
    );
    assert!(residual_sqr(&mut a, &x, &b)[0] <= 1.1e-24);
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

/// CG in single precision with reliable updates in double precision: the true residual of
/// double precision, in about the iterations of CG in double precision (no restarts from zero),
/// on several vectors, one of them zero.
#[test]
fn reliable_updates_reach_double_accuracy_in_cg_iterations() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 6);
    let (op64, op32) = (wilson::<f64>(&lat, &u), wilson::<f32>(&lat, &u));
    let mut b = random_spinor(&lat, 3, 8);
    let zero = b.select(&[1]).zeros_like();
    b.assign(&[1], &zero);
    let like32 = V32::spinor(&lat, 1);
    let tol = 1e-24;
    let mut a = ddag_d(&op64, &b);

    let mut plain = b.zeros_like();
    let cg = Cg::new(Stop::simple(tol, 2000))
        .solve(&mut a, None, &mut plain, &b)
        .unwrap();
    let mut x = b.zeros_like();
    let mut low = ddag_d(&op32, &like32);
    let info = reliable_cg(
        &mut a,
        &mut low,
        &like32,
        &mut x,
        &b,
        Stop::simple(tol, 2000),
        1e-2,
    )
    .unwrap();
    eprintln!(
        "CG in double precision {} iterations; in single with reliable updates {} iterations, {} updates",
        cg.iterations, info.iterations, info.restarts
    );
    for r in residual_sqr(&mut a, &x, &b) {
        assert!(r <= 1.1 * tol, "true residual {r:e}");
    }
    assert!(
        info.iterations * 10 <= cg.iterations * 13 + 100,
        "{} iterations against {} of CG in double precision",
        info.iterations,
        cg.iterations
    );
    assert!(max_abs_diff(&x, &plain) < 1e-9);
    assert!(x.select(&[1]).as_slice().iter().all(|v| *v == 0.0));
}

/// The ownership pattern in use: an outer and an inner solver share
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

/// `expected_result` of the Bridge++ test `Solver_Wilson_CGNE`: round(Nconv / 10), where Nconv is
/// one plus the number of CG steps on D D† for a point source at the origin (spin 0, colour 0),
/// starting from x = b, restarting every 100 steps, until |r|^2 / |b|^2 < 1e-28.
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

/// `out = λ inp`: an operator with one eigenvalue, on which a polynomial of the operator is
/// a number that can be computed independently.
struct Scalar(f64);

impl LinearFn<V64> for Scalar {
    fn mult(&mut self, out: &mut V64, inp: &V64) {
        out.copy_from(inp);
        out.scale(&[self.0]);
    }
}

/// T_n(x) for real x, from cos or cosh.
fn chebyshev_t(n: usize, x: f64) -> f64 {
    let n = n as f64;
    if x.abs() <= 1.0 {
        (n * x.acos()).cos()
    } else {
        x.signum().powi(n as i32) * (n * x.abs().acosh()).cosh()
    }
}

/// How far `out` is from `factor * v`, relative to the largest entry of `factor * v`.
fn off_by(out: &V64, v: &V64, factor: f64) -> f64 {
    let mut d = v.zeros_like();
    d.copy_from(v);
    d.scale(&[factor]);
    let size = d.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    max_abs_diff(out, &d) / size
}

/// The Chebyshev filter is T_n(ℓ(λ)) on an eigenvalue λ: bounded by one on [lo, hi],
/// growing below lo.
#[test]
fn chebyshev_filter_is_the_chebyshev_polynomial() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let v: V64 = random_spinor(&lat, 1, 3);
    let (lo, hi) = (0.1, 4.0);
    for degree in [1, 2, 7, 30] {
        for lambda in [0.0, 0.05, 0.1, 1.3, 4.0] {
            let mut f = ruttice::solver::ChebyshevFilter::new(Scalar(lambda), degree, lo, hi, &v);
            let mut out = v.zeros_like();
            f.mult(&mut out, &v);
            let expected = chebyshev_t(degree, (2.0 * lambda - (hi + lo)) / (hi - lo));
            let error = off_by(&out, &v, expected);
            assert!(error < 1e-12, "degree {degree}, λ {lambda}: {error:e}");
            if (lo..=hi).contains(&lambda) {
                assert!(expected.abs() <= 1.0 + 1e-12);
            }
        }
    }
}

/// Far below `lo` and at a high degree T_n passes the largest double: the filter scales the
/// vector down as it goes and gives a finite positive multiple of it, the direction T_n gives.
#[test]
fn chebyshev_filter_does_not_overflow() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let v: V64 = random_spinor(&lat, 1, 3);
    // ℓ(0) = −5/3: T_1000 is about e^1099.
    let mut f = ruttice::solver::ChebyshevFilter::new(Scalar(0.0), 1000, 1.0, 4.0, &v);
    let mut out = v.zeros_like();
    f.mult(&mut out, &v);
    assert!(out.as_slice().iter().all(|x| x.is_finite()));
    let (mut d, mut n) = ([ruttice::C64::ZERO], [0.0]);
    v.dot(&out, &mut d);
    out.norm_sqr(&mut n);
    let mut vv = [0.0];
    v.norm_sqr(&mut vv);
    // T_1000 is even and positive at ℓ < −1: out = c v with c > 0.
    let cos = d[0].re / (n[0] * vv[0]).sqrt();
    assert!(d[0].re > 0.0 && (cos - 1.0).abs() < 1e-12, "{cos}");
}

/// Chebyshev iteration from zero leaves the residual polynomial
/// T_k((θ - λ)/δ) / T_k(θ/δ) on an eigenvalue λ (Saad, Iterative Methods, eq. 12.8): one at
/// λ = 0, small on [lo, hi], and k steps apply A k - 1 times.
#[test]
fn chebyshev_iteration_leaves_its_residual_polynomial() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let b: V64 = random_spinor(&lat, 1, 4);
    let (lo, hi) = (0.2, 3.0);
    let (theta, delta) = ((hi + lo) / 2.0, (hi - lo) / 2.0);
    for steps in [1, 2, 5, 12] {
        for lambda in [0.2, 0.9, 3.0] {
            let mut cheb = ruttice::solver::Chebyshev::new(Scalar(lambda), steps, lo, hi, &b);
            let mut x = b.zeros_like();
            cheb.mult(&mut x, &b);
            // r = b - λ x
            let mut r = b.zeros_like();
            r.copy_from(&b);
            r.axpy(&[ruttice::C64::new(-lambda, 0.0)], &x);
            let expected =
                chebyshev_t(steps, (theta - lambda) / delta) / chebyshev_t(steps, theta / delta);
            let error = off_by(&r, &b, expected);
            assert!(error < 1e-10, "steps {steps}, λ {lambda}: {error:e}");
            assert!(expected.abs() < 1.0);
        }
    }
}

/// On the Wilson D†D: the filtered vector has a far smaller Rayleigh quotient than the
/// random one it came from; Chebyshev iteration is linear and reduces the residual; the
/// power iteration finds the top of the spectrum.
#[test]
fn chebyshev_on_the_wilson_normal_operator() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 21);
    let op = wilson::<f64>(&lat, &u);
    let v: V64 = random_spinor(&lat, 1, 22);
    let rayleigh = |h: &mut dyn LinearFn<V64>, x: &V64| {
        let mut hx = x.zeros_like();
        h.mult(&mut hx, x);
        let (mut d, mut n) = ([ruttice::C64::ZERO], [0.0]);
        x.dot(&hx, &mut d);
        x.norm_sqr(&mut n);
        d[0].re / n[0]
    };
    let mut h = ddag_d(&op, &v);
    let mut start: V64 = random_spinor(&lat, 1, 23);
    let top = ruttice::solver::largest_magnitude(&mut h, &mut start, 100);
    // D = 1 - κ H with |H| ≤ 8, so the spectrum of D†D lies in [0, (1 + 8κ)^2].
    assert!(
        top > 0.5 && top <= (1.0 + 8.0 * PARAMS.kappa).powi(2) * (1.0 + 1e-9),
        "{top}"
    );

    let hi = 1.1 * top;
    // A random gauge field at κ = 0.12 leaves little of the spectrum near zero; the
    // threshold sits at a quarter of the top so that there is something below it to keep.
    let mut filter = ruttice::solver::ChebyshevFilter::new(ddag_d(&op, &v), 40, hi / 4.0, hi, &v);
    let mut filtered = v.zeros_like();
    filter.mult(&mut filtered, &v);
    let (before, after) = (rayleigh(&mut h, &v), rayleigh(&mut h, &filtered));
    assert!(after < before / 5.0, "{before} -> {after}");

    let mut cheb = ruttice::solver::Chebyshev::new(ddag_d(&op, &v), 10, hi / 50.0, hi, &v);
    let w: V64 = random_spinor(&lat, 1, 24);
    let mut combined = v.zeros_like();
    combined.copy_from(&v);
    combined.axpy(&[ruttice::C64::new(0.0, 2.0)], &w);
    let (mut pv, mut pw, mut pc) = (v.zeros_like(), v.zeros_like(), v.zeros_like());
    cheb.mult(&mut pv, &v);
    cheb.mult(&mut pw, &w);
    cheb.mult(&mut pc, &combined);
    pc.axpy(&[ruttice::C64::new(-1.0, 0.0)], &pv);
    pc.axpy(&[ruttice::C64::new(0.0, -2.0)], &pw);
    let (mut left, mut size) = ([0.0], [0.0]);
    pc.norm_sqr(&mut left);
    pv.norm_sqr(&mut size);
    assert!(
        left[0] < 1e-24 * size[0],
        "not linear: {:e}",
        left[0] / size[0]
    );
    let after = residual_sqr(&mut h, &pv, &v)[0];
    assert!(after < 0.5, "|b - H p(H) b|^2/|b|^2 = {after}");
}

/// A preconditioner that starts returning NaN after a few steps is reported as a breakdown, not as
/// a solve that ran out of iterations with a NaN solution.
#[test]
fn bicgstab_reports_a_non_finite_preconditioner() {
    struct Poisoned(usize);
    impl LinearFn<V64> for Poisoned {
        fn mult(&mut self, out: &mut V64, inp: &V64) {
            self.0 += 1;
            out.copy_from(inp);
            if self.0 >= 4 && self.0.is_multiple_of(2) {
                out.as_mut_slice()[0] = f64::NAN;
            }
        }
    }
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let u = random_gauge(&lat, 7);
    let op = wilson::<f64>(&lat, &u);
    let b: V64 = random_spinor(&lat, 1, 8);
    let mut x = b.zeros_like();
    let result = BiCgStab::new(Stop {
        tolerance_sqr: 1e-20,
        max_iterations: 50,
        max_restarts: 3,
    })
    .solve(
        &mut Bound::new(&op, Variant::D),
        Some(&mut Poisoned(0)),
        &mut x,
        &b,
    );
    assert!(
        matches!(result, Err(SolveError::Breakdown { .. })),
        "{result:?}"
    );
}
