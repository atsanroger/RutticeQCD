//! Checks: flexible GMRES on the Wilson and clover operators.

use ruttice_core::{
    Boundary, ConvertFrom, Field, Gauge, Lattice, LinearFn, Normal, Spinor, Vector, C64, ND,
};
use ruttice_fermion::{Bound, Clover, CloverParams, FermionOp, Variant, Wilson, WilsonParams};
use ruttice_solver::{BiCgStab, Cg, Convert, Fgmres, Solve, SolveError, Solver, Stop};
use ruttice_validation::{max_abs_diff, random_gauge, random_spinor};
use std::sync::Arc;

type V64 = Field<f64, Spinor>;
type V32 = Field<f32, Spinor>;

const TIME_ANTIPERIODIC: [Boundary; ND] = [
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];

fn wilson<T: ruttice_core::Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Wilson<T> {
    let mut op = Wilson::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary: TIME_ANTIPERIODIC,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

fn residual_sqr(a: &mut dyn LinearFn<V64>, x: &V64, b: &V64) -> Vec<f64> {
    let nv = b.num_vectors();
    let mut r = b.zeros_like();
    a.mult(&mut r, x);
    r.scale(&vec![-1.0; nv]);
    r.axpy(&vec![C64::new(1.0, 0.0); nv], b);
    let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    rr.iter()
        .zip(&bb)
        .map(|(r, b)| if *b > 0.0 { r / b } else { *r })
        .collect()
}

/// Three right-hand sides at once, one of them zero, with a Krylov space too small to
/// converge without restarting.
#[test]
fn solves_d_with_restarts() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 301));
    let mut b = random_spinor(&lat, 3, 302);
    for site in 0..lat.num_sites() {
        b.site_mut(0, site)[24..48].fill(0.0);
    }
    let mut a = Bound::new(&op, Variant::D);

    let mut x = b.zeros_like();
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 12,
        max_restarts: 40,
    };
    let info = Fgmres::new(stop).solve(&mut a, None, &mut x, &b).unwrap();
    assert!(
        info.restarts >= 1,
        "the Krylov space was meant to be too small: {info:?}"
    );
    let r = residual_sqr(&mut a, &x, &b);
    assert!(r[0] <= 1.1e-24 && r[2] <= 1.1e-24, "{r:?}");
    assert_eq!(r[1], 0.0);
    for site in 0..lat.num_sites() {
        assert!(x.site(0, site)[24..48].iter().all(|v| *v == 0.0));
    }

    // Too few restarts is reported, with the solution so far.
    let mut y = b.zeros_like();
    let short = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 4,
        max_restarts: 1,
    };
    let outcome = Fgmres::new(short).solve(&mut a, None, &mut y, &b);
    assert!(
        matches!(outcome, Err(SolveError::NotConverged(ref i)) if i.iterations == 8),
        "{outcome:?}"
    );
}

/// A point source, which makes BiCGStab break down, is no problem for GMRES; and on a
/// Hermitian positive system it agrees with conjugate gradient.
#[test]
fn point_source_and_agreement_with_cg() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 303));
    let mut b = V64::spinor(&lat, 1);
    b.site_mut(0, lat.site([1, 0, 3, 2]))[5] = 1.0;

    let mut d = Bound::new(&op, Variant::D);
    let mut x = b.zeros_like();
    Fgmres::new(Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 20,
    })
    .solve(&mut d, None, &mut x, &b)
    .unwrap();
    assert!(residual_sqr(&mut d, &x, &b)[0] <= 1.1e-24);

    let stop = Stop {
        tolerance_sqr: 1e-26,
        max_iterations: 60,
        max_restarts: 20,
    };
    let mut ddag_d = Normal::new(Bound::new(&op, Variant::D), &b);
    let (mut by_gmres, mut by_cg) = (b.zeros_like(), b.zeros_like());
    Fgmres::new(stop)
        .solve(&mut ddag_d, None, &mut by_gmres, &b)
        .unwrap();
    Cg::new(Stop::simple(1e-26, 500))
        .solve(&mut ddag_d, None, &mut by_cg, &b)
        .unwrap();
    let diff = max_abs_diff(&by_gmres, &by_cg);
    assert!(diff < 1e-11, "max difference {diff}");
}

/// The flexible part: the preconditioner is an inner iterative solve in single precision,
/// stopped early, so it is a different linear map every time it is applied. The outer
/// iteration still converges to double precision, in fewer steps than without it.
#[test]
fn an_inner_solver_as_preconditioner() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 304);
    let mut op64 = Clover::<f64>::new(
        &lat,
        CloverParams {
            kappa: 0.12,
            csw: 1.0,
            boundary: TIME_ANTIPERIODIC,
        },
    );
    let mut op32 = Clover::<f32>::new(
        &lat,
        CloverParams {
            kappa: 0.12,
            csw: 1.0,
            boundary: TIME_ANTIPERIODIC,
        },
    );
    op64.set_gauge(&u).unwrap();
    op32.set_gauge(&u).unwrap();
    let b = random_spinor(&lat, 2, 305);
    let like32 = V32::spinor(&lat, 2);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 20,
    };

    let mut a = Bound::new(&op64, Variant::D);
    let mut plain = b.zeros_like();
    let without = Fgmres::new(stop)
        .solve(&mut a, None, &mut plain, &b)
        .unwrap();

    let mut inner = BiCgStab::<V32>::new(Stop::simple(1e-4, 4));
    let mut precond = Convert::new(
        Solve::new(&mut inner, Bound::new(&op32, Variant::D)),
        &like32,
    );
    let mut x = b.zeros_like();
    let with = Fgmres::new(stop)
        .solve(&mut a, Some(&mut precond), &mut x, &b)
        .unwrap();

    for r in residual_sqr(&mut a, &x, &b) {
        assert!(r <= 1.1e-24, "true residual {r:e}");
    }
    eprintln!(
        "outer iterations: {} with the inner solver, {} without",
        with.iterations, without.iterations
    );
    assert!(with.iterations < without.iterations);
    assert!(max_abs_diff(&x, &plain) < 1e-10);

    // The conversion is used both ways on every application.
    let mut check = like32.zeros_like();
    check.convert_from(&x);
    assert!(check.as_slice().iter().any(|v| *v != 0.0));
}
