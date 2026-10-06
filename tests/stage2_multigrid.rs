//! Checks: the multigrid preconditioner. The coarse-grid components (aggregation,
//! restriction and prolongation, Galerkin coarse operator) are identities that hold for any
//! set of test vectors, so random vectors are used; then the SAP smoother and the
//! two-level cycle inside FGMRES.

mod common;

use common::max_abs_diff;
use ruttice::fermion::{
    Bound, Clover, CloverParams, FermionOp, HopTerms, Variant, Wilson, WilsonParams,
};
use ruttice::gauge::random_gauge;
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, Basis, CoarseOp, Multigrid, MultigridError, MultigridParams, Sap,
};
use ruttice::solver::{Convert, Fgmres, Solver, Stop};
use ruttice::{random_spinor, TestRng};
use ruttice::{
    Boundary, Coarse, ConvertFrom, Field, Gauge, Lattice, LinearFn, LinearOp, Spinor, Storage,
    Vector, C64, ND,
};
use std::sync::Arc;

type V64 = Field<f64, Spinor>;
type C64Field = Field<f64, Coarse>;

const MIXED: [Boundary; ND] = [
    Boundary::Periodic,
    Boundary::Antiperiodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];

fn lattice() -> Arc<Lattice> {
    Lattice::single_node([4, 4, 4, 8])
}

fn wilson<T: Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Wilson<T> {
    let mut op = Wilson::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary: MIXED,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

fn clover<T: Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Clover<T> {
    let mut op = Clover::new(
        lat,
        CloverParams {
            kappa: 0.12,
            csw: 1.3,
            boundary: MIXED,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

fn random_coarse(basis: &Basis<f64>, num_vectors: usize, seed: u64) -> C64Field {
    let mut rng = TestRng::new(seed);
    let mut c = basis.coarse_field(num_vectors);
    for v in c.as_mut_slice() {
        *v = 2.0 * rng.uniform() - 1.0;
    }
    c
}

fn max_abs_diff_coarse(a: &C64Field, b: &C64Field) -> f64 {
    max_abs_diff(a, b)
}

#[test]
fn restriction_undoes_prolongation() {
    let lat = lattice();
    for block in [[2, 2, 2, 2], [2, 2, 2, 4], [4, 4, 4, 8], [1, 2, 4, 2]] {
        let agg = Aggregation::new(&lat, block);
        let basis = Basis::<f64>::from_vectors(&agg, &random_spinor(&lat, 5, 401)).unwrap();
        assert_eq!(basis.coarse_dim(), 10);
        let c = random_coarse(&basis, 2, 402);
        let mut fine = V64::spinor(&lat, 2);
        let mut back = basis.coarse_field(2);
        basis.prolong(&mut fine, &c);
        basis.restrict(&mut back, &fine);
        let diff = max_abs_diff_coarse(&back, &c);
        assert!(diff < 1e-13, "block {block:?}: R P - 1 = {diff:e}");
    }
}

/// P R is the orthogonal projector onto the range of P: it is idempotent, and R is the
/// adjoint of P.
#[test]
fn prolongation_after_restriction_is_an_orthogonal_projector() {
    let lat = lattice();
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = Basis::<f64>::from_vectors(&agg, &random_spinor(&lat, 4, 411)).unwrap();

    let v = random_spinor(&lat, 1, 412);
    let mut c = basis.coarse_field(1);
    let mut pv = v.zeros_like();
    basis.restrict(&mut c, &v);
    basis.prolong(&mut pv, &c);
    let mut c2 = basis.coarse_field(1);
    let mut ppv = v.zeros_like();
    basis.restrict(&mut c2, &pv);
    basis.prolong(&mut ppv, &c2);
    assert!(max_abs_diff(&ppv, &pv) < 1e-13);

    // <P c, v> = <c, R v>
    let w = random_coarse(&basis, 1, 413);
    let mut pw = v.zeros_like();
    basis.prolong(&mut pw, &w);
    let (mut lhs, mut rhs) = ([C64::ZERO], [C64::ZERO]);
    pw.dot(&v, &mut lhs);
    w.dot(&c, &mut rhs);
    assert!((lhs[0] - rhs[0]).norm_sqr().sqrt() < 1e-11 * lhs[0].norm_sqr().sqrt());
}

/// With one block covering the whole lattice per chirality, the chiral halves of every test
/// vector are in the range of P.
#[test]
fn test_vectors_are_reproduced() {
    let lat = lattice();
    let vectors = random_spinor(&lat, 3, 421);
    for block in [[2, 2, 2, 2], [4, 4, 2, 4]] {
        let agg = Aggregation::new(&lat, block);
        let basis = Basis::<f64>::from_vectors(&agg, &vectors).unwrap();
        let mut c = basis.coarse_field(3);
        let mut back = vectors.zeros_like();
        basis.restrict(&mut c, &vectors);
        basis.prolong(&mut back, &c);
        let diff = max_abs_diff(&back, &vectors);
        assert!(diff < 1e-13, "block {block:?}: {diff:e}");
    }
}

#[test]
fn prolongation_commutes_with_gamma5() {
    let lat = lattice();
    let op = wilson::<f64>(&lat, &random_gauge(&lat, 431));
    let mut work = op.work();
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    let basis = Basis::<f64>::from_vectors(&agg, &random_spinor(&lat, 4, 432)).unwrap();
    let n = basis.num_test_vectors();

    let c = random_coarse(&basis, 1, 433);
    let mut gamma5_c = basis.coarse_field(1);
    for site in 0..c.num_sites() {
        for (k, (o, i)) in gamma5_c
            .site_mut(0, site)
            .iter_mut()
            .zip(c.site(0, site))
            .enumerate()
        {
            *o = if k < 2 * n { *i } else { -*i };
        }
    }
    let mut p_c = V64::spinor(&lat, 1);
    let mut lhs = V64::spinor(&lat, 1);
    let mut rhs = V64::spinor(&lat, 1);
    basis.prolong(&mut p_c, &c);
    op.mult_gamma5(&mut work, &mut lhs, &p_c);
    basis.prolong(&mut rhs, &gamma5_c);
    assert!(max_abs_diff(&lhs, &rhs) < 1e-13);
}

#[test]
fn dependent_test_vectors_are_reported() {
    let lat = lattice();
    let mut vectors = random_spinor(&lat, 3, 441);
    for site in 0..lat.num_sites() {
        let s = vectors.site_mut(0, site);
        for k in 0..24 {
            s[48 + k] = 0.5 * s[k] - s[24 + k];
        }
    }
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    match Basis::<f64>::from_vectors(&agg, &vectors) {
        Err(MultigridError::DependentTestVectors { vector: 2, .. }) => {}
        other => panic!("expected a dependence error, got {:?}", other.err()),
    }
}

/// `D_c c = R D P c` for a random coarse vector.
fn check_galerkin<F: HopTerms<V64>>(op: &F, block: [usize; ND], seed: u64) -> f64 {
    let lat = lattice();
    let agg = Aggregation::new(&lat, block);
    let basis = Basis::<f64>::from_vectors(&agg, &random_spinor(&lat, 4, seed)).unwrap();
    let mut work = op.work();
    let mut coarse_op = CoarseOp::build(op, &mut work, &basis);
    assert_eq!(coarse_op.dim(), 8);

    let c = random_coarse(&basis, 2, seed + 1);
    let mut fine = V64::spinor(&lat, 2);
    let mut d_fine = V64::spinor(&lat, 2);
    let mut want = basis.coarse_field(2);
    basis.prolong(&mut fine, &c);
    op.mult(&mut work, &mut d_fine, &fine, Variant::D);
    basis.restrict(&mut want, &d_fine);

    let mut got = basis.coarse_field(2);
    coarse_op.mult(&mut got, &c);
    max_abs_diff_coarse(&got, &want)
}

#[test]
fn coarse_operator_is_the_galerkin_product() {
    let lat = lattice();
    let u = random_gauge(&lat, 451);
    let w = wilson::<f64>(&lat, &u);
    let c = clover::<f64>(&lat, &u);
    // The last two have a single block along some directions, where forward and backward
    // hops reach the same coarse site.
    for block in [[2, 2, 2, 2], [2, 2, 2, 4], [4, 2, 4, 8], [4, 4, 4, 8]] {
        let diff = check_galerkin(&w, block, 452);
        assert!(diff < 1e-13, "Wilson, block {block:?}: {diff:e}");
        let diff = check_galerkin(&c, block, 453);
        assert!(diff < 1e-13, "clover, block {block:?}: {diff:e}");
    }
}

/// `<a, D_c b> = <D_c† a, b>`, and `D_c†` agrees with `R D† P`.
#[test]
fn coarse_adjoint() {
    let lat = lattice();
    let op = clover::<f64>(&lat, &random_gauge(&lat, 461));
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = Basis::<f64>::from_vectors(&agg, &random_spinor(&lat, 4, 462)).unwrap();
    let mut work = op.work();
    let mut coarse_op = CoarseOp::build(&op, &mut work, &basis);

    let a = random_coarse(&basis, 1, 463);
    let b = random_coarse(&basis, 1, 464);
    let mut d_b = basis.coarse_field(1);
    let mut dag_a = basis.coarse_field(1);
    coarse_op.mult(&mut d_b, &b);
    coarse_op.mult_dag(&mut dag_a, &a);
    let (mut lhs, mut rhs) = ([C64::ZERO], [C64::ZERO]);
    a.dot(&d_b, &mut lhs);
    dag_a.dot(&b, &mut rhs);
    assert!((lhs[0] - rhs[0]).norm_sqr().sqrt() < 1e-11 * lhs[0].norm_sqr().sqrt());

    let mut fine = V64::spinor(&lat, 1);
    let mut d_fine = V64::spinor(&lat, 1);
    let mut want = basis.coarse_field(1);
    basis.prolong(&mut fine, &a);
    Bound::new(&op, Variant::D).mult_dag(&mut d_fine, &fine);
    basis.restrict(&mut want, &d_fine);
    assert!(max_abs_diff_coarse(&dag_a, &want) < 1e-13);
}

/// The single-precision hierarchy is the double-precision one rounded.
#[test]
fn single_precision_agrees_with_double() {
    let lat = lattice();
    let u = random_gauge(&lat, 471);
    let vectors = random_spinor(&lat, 4, 472);
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);

    let op64 = clover::<f64>(&lat, &u);
    let basis64 = Basis::<f64>::from_vectors(&agg, &vectors).unwrap();
    let mut coarse64 = CoarseOp::build(&op64, &mut op64.work(), &basis64);
    let op32 = clover::<f32>(&lat, &u);
    let basis32 = Basis::<f32>::from_vectors(&agg, &vectors).unwrap();
    let mut coarse32 = CoarseOp::build(&op32, &mut op32.work(), &basis32);

    let c64 = random_coarse(&basis64, 1, 473);
    let mut c32 = basis32.coarse_field(1);
    c32.convert_from(&c64);
    let mut out64 = basis64.coarse_field(1);
    let mut out32 = basis32.coarse_field(1);
    coarse64.mult(&mut out64, &c64);
    coarse32.mult(&mut out32, &c32);
    let mut widened = basis64.coarse_field(1);
    widened.convert_from(&out32);
    let diff = max_abs_diff_coarse(&widened, &out64);
    assert!(diff < 2e-5, "{diff:e}");
}

// ---- The smoother and the two-level preconditioner ----

fn clover_at<T: Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>, kappa: f64) -> Clover<T> {
    let mut op = Clover::new(
        lat,
        CloverParams {
            kappa,
            csw: 1.0,
            boundary: MIXED,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

/// |b - D x|^2 / |b|^2 for every vector.
fn residual_sqr<F: FermionOp<V64>>(op: &F, x: &V64, b: &V64) -> Vec<f64> {
    let nv = b.num_vectors();
    let mut r = b.zeros_like();
    Bound::new(op, Variant::D).mult(&mut r, x);
    r.aypx(&vec![C64::new(-1.0, 0.0); nv], b);
    let (mut rn, mut bn) = (vec![0.0; nv], vec![0.0; nv]);
    r.norm_sqr(&mut rn);
    b.norm_sqr(&mut bn);
    rn.iter().zip(&bn).map(|(r, b)| r / b).collect()
}

/// SAP on its own is a convergent iteration for a heavy enough quark: more cycles, smaller
/// residual.
#[test]
fn sap_reduces_the_residual() {
    let lat = lattice();
    let op = clover_at::<f64>(&lat, &random_gauge(&lat, 501), 0.12);
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    let b = random_spinor(&lat, 1, 502);
    let mut previous = 1.0;
    for cycles in [1, 2, 4, 8] {
        let mut x = b.zeros_like();
        Sap::new(&op, &agg, cycles, 6).mult(&mut x, &b);
        let r = residual_sqr(&op, &x, &b)[0];
        assert!(
            r < 0.1 * previous,
            "{cycles} cycles: {r:e} after {previous:e}"
        );
        previous = r;
    }
    assert!(previous < 1e-8, "{previous:e}");
}

/// Vectors of a multi-vector are smoothed independently.
#[test]
fn sap_treats_vectors_independently() {
    let lat = lattice();
    let op = clover_at::<f64>(&lat, &random_gauge(&lat, 511), 0.12);
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let b = random_spinor(&lat, 3, 512);
    let mut sap = Sap::new(&op, &agg, 2, 4);
    let mut together = b.zeros_like();
    sap.mult(&mut together, &b);
    for v in 0..3 {
        let mut one = V64::spinor(&lat, 1);
        let mut x = V64::spinor(&lat, 1);
        for site in 0..lat.num_sites() {
            let len = one.site_len();
            one.site_mut(0, site)
                .copy_from_slice(&b.site(0, site)[v * len..(v + 1) * len]);
        }
        sap.mult(&mut x, &one);
        for site in 0..lat.num_sites() {
            let len = x.site_len();
            let t = &together.site(0, site)[v * len..(v + 1) * len];
            assert_eq!(x.site(0, site), t, "vector {v}, site {site}");
        }
    }
}

/// The test vectors handed out are orthonormal on the aggregates: restricting them gives
/// the coarse vectors they were built from.
#[test]
fn test_vectors_restrict_to_unit_coarse_vectors() {
    let lat = lattice();
    let op = clover_at::<f64>(&lat, &random_gauge(&lat, 521), 0.12);
    let mut params = MultigridParams::bridge([2, 2, 2, 2]);
    params.num_test_vectors = 6;
    params.setup_iterations = 1;
    let mg = Multigrid::new(&op, &lat, &params).unwrap();
    let n = 6;
    let mut c = mg.basis().coarse_field(n);
    mg.basis().restrict(&mut c, &mg.test_vectors());
    let dim = mg.basis().coarse_dim();
    for site in 0..c.num_sites() {
        for (i, v) in c.site(0, site).chunks_exact(2 * dim).enumerate() {
            for k in 0..dim {
                let want = if k == i || k == n + i { 1.0 } else { 0.0 };
                assert!((v[2 * k] - want).abs() < 1e-12 && v[2 * k + 1].abs() < 1e-12);
            }
        }
    }
}

/// FGMRES on D with the two-level preconditioner reaches the tolerance in far fewer
/// iterations than without, in double and in single precision inside a double outer
/// solve.
#[test]
fn multigrid_preconditioned_fgmres_converges() {
    let lat = lattice();
    let u = random_gauge(&lat, 531);
    let op = clover_at::<f64>(&lat, &u, 0.2);
    let op32 = clover_at::<f32>(&lat, &u, 0.2);
    let b = random_spinor(&lat, 1, 532);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 30,
    };
    let solve = |m: Option<&mut dyn LinearFn<V64>>| {
        let mut x = b.zeros_like();
        let info = Fgmres::new(stop)
            .solve(&mut Bound::new(&op, Variant::D), m, &mut x, &b)
            .unwrap();
        assert!(residual_sqr(&op, &x, &b)[0] < 1e-24);
        info.iterations
    };
    let plain = solve(None);
    let params = MultigridParams::bridge([2, 2, 2, 2]);
    let mut mg = Multigrid::new(&op, &lat, &params).unwrap();
    let with_mg = solve(Some(&mut mg));
    let mg32 = Multigrid::new(&op32, &lat, &params).unwrap();
    let mut mixed = Convert::new(mg32, &Field::<f32, Spinor>::spinor(&lat, 1));
    let with_mg32 = solve(Some(&mut mixed));
    eprintln!("FGMRES iterations: {plain} plain, {with_mg} with multigrid, {with_mg32} with single-precision multigrid");
    assert!(with_mg * 5 < plain && with_mg32 * 5 < plain);
}

/// The single-precision preconditioner really works in single precision: every component
/// it returns is a single-precision number.
#[test]
fn single_precision_multigrid_returns_single_precision_values() {
    let lat = lattice();
    let op32 = clover_at::<f32>(&lat, &random_gauge(&lat, 541), 0.12);
    let mut params = MultigridParams::bridge([2, 2, 2, 2]);
    params.num_test_vectors = 8;
    params.setup_iterations = 1;
    let mg32 = Multigrid::new(&op32, &lat, &params).unwrap();
    let mut m = Convert::new(mg32, &Field::<f32, Spinor>::spinor(&lat, 1));
    let b = random_spinor(&lat, 1, 542);
    let mut x = b.zeros_like();
    m.mult(&mut x, &b);
    assert!(x.as_slice().iter().all(|v| (*v as f32) as f64 == *v));
    assert!(x.as_slice().iter().any(|v| *v != 0.0));
}

/// The Bridge++ test configuration with the operator of the Bridge++ multigrid test
/// (`test_alt_Multigrid.yaml`: clover, κ = 0.13, c_SW = 1.1, periodic) and its multigrid
/// parameters on 2^4 blocks. No reference iteration count: the Bridge++ multigrid test
/// could not be built here (DESIGN.md §14). Skipped without the configuration.
#[test]
fn multigrid_on_the_bridge_configuration() {
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let lat = lattice();
    let u = ruttice::io::read_bridge_text(&path, &lat).unwrap();
    let params = CloverParams {
        kappa: 0.13,
        csw: 1.1,
        boundary: [Boundary::Periodic; ND],
    };
    let mut op = Clover::<f64>::new(&lat, params);
    op.set_gauge(&u).unwrap();
    let mut op32 = Clover::<f32>::new(&lat, params);
    op32.set_gauge(&u).unwrap();
    let b = random_spinor(&lat, 1, 551);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 30,
    };
    let solve = |m: Option<&mut dyn LinearFn<V64>>| {
        let mut x = b.zeros_like();
        let info = Fgmres::new(stop)
            .solve(&mut Bound::new(&op, Variant::D), m, &mut x, &b)
            .unwrap();
        assert!(residual_sqr(&op, &x, &b)[0] < 1e-24);
        info.iterations
    };
    let plain = solve(None);
    let mg32 = Multigrid::new(&op32, &lat, &MultigridParams::bridge([2, 2, 2, 2])).unwrap();
    let with_mg = solve(Some(&mut Convert::new(
        mg32,
        &Field::<f32, Spinor>::spinor(&lat, 1),
    )));
    eprintln!("FGMRES iterations: {plain} plain, {with_mg} with single-precision multigrid");
    assert!(with_mg * 10 < plain);
}
