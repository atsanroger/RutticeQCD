//! Checks: the multigrid preconditioner. The coarse-grid components (aggregation,
//! restriction and prolongation, Galerkin coarse operator) are identities that hold for any
//! set of test vectors, so random vectors are used; then the SAP smoother and the
//! two-level cycle inside FGMRES.

mod common;

use common::max_abs_diff;
use ruttice::fermion::{
    solve_even_odd, Bound, Clover, CloverParams, FermionOp, HopTerms, OnParity, Variant, Wilson,
    WilsonParams,
};
use ruttice::gauge::random_gauge;
use ruttice::kernel::{Dirac, GammaRepre, Spinor as SiteSpinor, SPINOR_REALS};
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, Basis, CoarseOp, CycleParts, Multigrid, MultigridError, MultigridParams, Sap,
    SetupHooks,
};
use ruttice::solver::{BiCgStab, Convert, Fgmres, Solve, Solver, Stop};
use ruttice::{assemble, local_part, random_spinor, LocalComm, TestRng};
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

/// Γ5 in the Dirac representation, applied to every spinor of `v`: the operators of these
/// tests use it.
fn gamma5(v: &V64) -> V64 {
    let mut out = v.zeros_like();
    for site in 0..v.num_sites() {
        let src = v.site(0, site).as_chunks::<SPINOR_REALS>().0;
        let dst = out.site_mut(0, site).as_chunks_mut::<SPINOR_REALS>().0;
        for (o, s) in dst.iter_mut().zip(src) {
            Dirac::GAMMA5
                .apply(&SiteSpinor::from_reals(s))
                .write_reals(o);
        }
    }
    out
}

fn basis64(agg: &Arc<Aggregation>, v: &V64) -> Result<Basis<f64>, MultigridError> {
    Basis::from_vectors(agg, v, &gamma5(v))
}

fn max_abs_diff_coarse(a: &C64Field, b: &C64Field) -> f64 {
    max_abs_diff(a, b)
}

#[test]
fn restriction_undoes_prolongation() {
    let lat = lattice();
    for block in [[2, 2, 2, 2], [2, 2, 2, 4], [4, 4, 4, 8], [1, 2, 4, 2]] {
        let agg = Aggregation::new(&lat, block);
        let basis = basis64(&agg, &random_spinor(&lat, 5, 401)).unwrap();
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
    let basis = basis64(&agg, &random_spinor(&lat, 4, 411)).unwrap();

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
        let basis = basis64(&agg, &vectors).unwrap();
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
    let basis = basis64(&agg, &random_spinor(&lat, 4, 432)).unwrap();
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
    match basis64(&agg, &vectors) {
        Err(MultigridError::DependentTestVectors { vector: 2, .. }) => {}
        other => panic!("expected a dependence error, got {:?}", other.err()),
    }
}

/// `D_c c = R D P c` for a random coarse vector.
fn check_galerkin<F: HopTerms<V64>>(op: &F, block: [usize; ND], seed: u64) -> f64 {
    let lat = lattice();
    let agg = Aggregation::new(&lat, block);
    let basis = basis64(&agg, &random_spinor(&lat, 4, seed)).unwrap();
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
    let basis = basis64(&agg, &random_spinor(&lat, 4, 462)).unwrap();
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

/// A site-local factor for `CoarseOp::build_with`: the site-diagonal part of an operator.
struct SiteDiagonal<'a, F: HopTerms<V64>> {
    op: &'a F,
    work: F::Work,
}

impl<F: HopTerms<V64>> LinearFn<V64> for SiteDiagonal<'_, F> {
    fn mult(&mut self, out: &mut V64, inp: &V64) {
        out.set_zero();
        self.op.mult_diag(&mut self.work, out, inp);
    }
}

/// With a site-local right factor R the coarse operator is P† D R P.
#[test]
fn coarse_operator_with_a_right_factor() {
    let lat = lattice();
    let u = random_gauge(&lat, 481);
    let op = wilson::<f64>(&lat, &u);
    let factor = clover::<f64>(&lat, &u);
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = basis64(&agg, &random_spinor(&lat, 4, 482)).unwrap();
    let mut work = op.work();
    let mut right = SiteDiagonal {
        op: &factor,
        work: factor.work(),
    };
    let mut coarse = CoarseOp::build_with(&op, &mut work, &basis, Some(&mut right), false).unwrap();

    let c = random_coarse(&basis, 2, 483);
    let (mut p_c, mut r_p_c, mut d_r_p_c) = (
        V64::spinor(&lat, 2),
        V64::spinor(&lat, 2),
        V64::spinor(&lat, 2),
    );
    basis.prolong(&mut p_c, &c);
    right.mult(&mut r_p_c, &p_c);
    op.mult(&mut work, &mut d_r_p_c, &r_p_c, Variant::D);
    let mut want = basis.coarse_field(2);
    basis.restrict(&mut want, &d_r_p_c);
    let mut got = basis.coarse_field(2);
    coarse.mult(&mut got, &c);
    assert!(max_abs_diff_coarse(&got, &want) < 1e-12);
}

/// The Schur complement of a coarse operator on its even sites, as a solver takes it.
struct CoarseSchur<'a>(&'a CoarseOp<f64>, C64Field);

impl LinearFn<C64Field> for CoarseSchur<'_> {
    fn mult(&mut self, out: &mut C64Field, inp: &C64Field) {
        self.0.schur(out, inp, &mut self.1);
    }
}

/// The even-odd form of the coarse operator: solving the Schur complement on the even coarse sites
/// and filling in the odd ones solves D_c c = g, and the Schur complement stays on the even sites.
#[test]
fn coarse_even_odd_solves_the_coarse_system() {
    let lat = lattice();
    let op = clover::<f64>(&lat, &random_gauge(&lat, 471));
    for block in [[2, 2, 2, 2], [2, 2, 2, 4]] {
        let agg = Aggregation::new(&lat, block);
        let basis = basis64(&agg, &random_spinor(&lat, 4, 472)).unwrap();
        let mut a = CoarseOp::build(&op, &mut op.work(), &basis);
        assert!(a.prepare_even_odd().unwrap() && a.has_even_odd());
        let g = random_coarse(&basis, 2, 473);
        let (mut source, mut odd) = (basis.coarse_field(2), basis.coarse_field(2));
        a.schur_source(&mut source, &g, &mut odd);

        // S of a field on the even sites is on the even sites: adding anything on the odd
        // sites of the input changes nothing, and S x has no odd part to give A_oo⁻¹.
        let mut s_source = basis.coarse_field(2);
        a.schur(&mut s_source, &source, &mut odd);
        let mut back = basis.coarse_field(2);
        a.schur_source(&mut back, &s_source, &mut odd);
        assert!(max_abs_diff_coarse(&back, &s_source) < 1e-13, "{block:?}");

        let stop = Stop::simple(1e-26, 2000);
        let mut c = basis.coarse_field(2);
        let schur = BiCgStab::new(stop)
            .solve(
                &mut CoarseSchur(&a, basis.coarse_field(2)),
                None,
                &mut c,
                &source,
            )
            .unwrap();
        a.schur_solution(&mut c, &g, &mut odd);
        let mut d_c = basis.coarse_field(2);
        a.mult(&mut d_c, &c);
        let diff = max_abs_diff_coarse(&d_c, &g);
        assert!(diff < 1e-10, "{block:?}: {diff:e}");

        let mut direct = basis.coarse_field(2);
        let whole = BiCgStab::new(stop)
            .solve(&mut a, None, &mut direct, &g)
            .unwrap();
        assert!(max_abs_diff_coarse(&direct, &c) < 1e-9, "{block:?}");
        eprintln!(
            "{block:?}: {} iterations on the even sites, {} on all",
            schur.iterations, whole.iterations
        );
    }
    // A coarse lattice with an odd extent has no even-odd form.
    let odd_lat = Lattice::single_node([6, 4, 4, 4]);
    let op = wilson::<f64>(&odd_lat, &random_gauge(&odd_lat, 474));
    let agg = Aggregation::new(&odd_lat, [2, 2, 2, 2]);
    let basis = basis64(&agg, &random_spinor(&odd_lat, 4, 475)).unwrap();
    let mut a = CoarseOp::build(&op, &mut op.work(), &basis);
    assert!(!a.prepare_even_odd().unwrap() && !a.has_even_odd());
}

/// Preconditioned by its diagonal blocks the coarse operator is A^{-1} D_c, and its
/// adjoint is still the adjoint.
#[test]
fn diagonally_preconditioned_coarse_operator() {
    let lat = lattice();
    let op = clover::<f64>(&lat, &random_gauge(&lat, 491));
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    let basis = basis64(&agg, &random_spinor(&lat, 4, 492)).unwrap();
    let mut work = op.work();
    let mut plain = CoarseOp::build(&op, &mut work, &basis);
    let mut prec = CoarseOp::build_with(&op, &mut work, &basis, None, true).unwrap();
    assert!(prec.is_diagonal_preconditioned() && !plain.is_diagonal_preconditioned());

    let c = random_coarse(&basis, 1, 493);
    let mut d_c = basis.coarse_field(1);
    let mut want = basis.coarse_field(1);
    plain.mult(&mut d_c, &c);
    prec.apply_diagonal_inverse(&mut want, &d_c);
    let mut got = basis.coarse_field(1);
    prec.mult(&mut got, &c);
    assert!(max_abs_diff_coarse(&got, &want) < 1e-12);

    let a = random_coarse(&basis, 1, 494);
    let (mut m_c, mut m_dag_a) = (basis.coarse_field(1), basis.coarse_field(1));
    prec.mult(&mut m_c, &c);
    prec.mult_dag(&mut m_dag_a, &a);
    let (mut lhs, mut rhs) = ([C64::ZERO], [C64::ZERO]);
    a.dot(&m_c, &mut lhs);
    m_dag_a.dot(&c, &mut rhs);
    assert!((lhs[0] - rhs[0]).norm_sqr().sqrt() < 1e-11 * lhs[0].norm_sqr().sqrt());
}

/// The single-precision hierarchy is the double-precision one rounded.
#[test]
fn single_precision_agrees_with_double() {
    let lat = lattice();
    let u = random_gauge(&lat, 471);
    let vectors = random_spinor(&lat, 4, 472);
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);

    let op64 = clover::<f64>(&lat, &u);
    let basis64 = basis64(&agg, &vectors).unwrap();
    let mut coarse64 = CoarseOp::build(&op64, &mut op64.work(), &basis64);
    let op32 = clover::<f32>(&lat, &u);
    let basis32 = Basis::<f32>::from_vectors(&agg, &vectors, &gamma5(&vectors)).unwrap();
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
    // The coarse system solved on the even coarse sites: the same cycle to the accuracy of
    // the coarse solves, so about the same number of outer iterations.
    let even_odd = MultigridParams {
        coarse_even_odd: true,
        ..params
    };
    let mut mg_eo = Multigrid::new(&op, &lat, &even_odd).unwrap();
    let with_eo = solve(Some(&mut mg_eo));
    eprintln!("FGMRES iterations: {plain} plain, {with_mg} with multigrid, {with_mg32} with single-precision multigrid, {with_eo} with the coarse solves on the even coarse sites");
    assert!(with_mg * 5 < plain && with_mg32 * 5 < plain);
    assert!(with_eo <= with_mg + 2, "{with_eo} against {with_mg}");
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
/// (`test_alt_Multigrid.yaml`: clover, κ = 0.13, c_SW = 1.1, periodic) and its multigrid parameters
/// on 2^4 blocks.
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

/// The members of the cycle can be exchanged: the domain-wall choices of the CUDA version (GMRES(m)
/// before and after the coarse correction, GMRES(m) on the coarse lattice, coarse operator
/// preconditioned by its diagonal blocks), here on a clover operator, still make a good
/// preconditioner.
#[test]
fn exchangeable_members_and_outer_solvers() {
    let lat = lattice();
    let op = clover_at::<f64>(&lat, &random_gauge(&lat, 561), 0.2);
    let b = random_spinor(&lat, 1, 562);
    let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
    // GMRES(m) is FGMRES without a preconditioner, m iterations, no restarts.
    let smoother = || {
        Box::new(Solve::new(
            Fgmres::<V64>::new(Stop::simple(1e-3, 4)),
            Bound::new(&op, Variant::D),
        ))
    };
    let parts = CycleParts {
        operator: Box::new(Bound::new(&op, Variant::D)),
        pre_smoother: Some(smoother()),
        post_smoother: smoother(),
        coarse_solver: Box::new(Fgmres::<C64Field>::new(Stop::simple(1e-2, 5))),
    };
    let hooks = SetupHooks {
        precondition_coarse_diagonal: true,
        ..SetupHooks::default()
    };
    let mut mg =
        Multigrid::with_parts(&op, &agg, random_spinor(&lat, 8, 563), 2, parts, hooks).unwrap();
    assert!(mg.coarse_operator().is_diagonal_preconditioned());

    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 30,
    };
    let plain = {
        let mut x = b.zeros_like();
        Fgmres::new(stop)
            .solve(&mut Bound::new(&op, Variant::D), None, &mut x, &b)
            .unwrap()
            .iterations
    };
    let mut outer: [(&str, Box<dyn Solver<V64>>); 2] = [
        ("FGMRES", Box::new(Fgmres::new(stop))),
        ("BiCGStab", Box::new(BiCgStab::new(stop))),
    ];
    for (name, solver) in outer.iter_mut() {
        let mut x = b.zeros_like();
        let info = solver
            .solve(&mut Bound::new(&op, Variant::D), Some(&mut mg), &mut x, &b)
            .unwrap();
        assert!(residual_sqr(&op, &x, &b)[0] < 1e-24);
        eprintln!(
            "{name} with the domain-wall style cycle: {} iterations (FGMRES alone {plain})",
            info.iterations
        );
        assert!(info.iterations * 5 < plain, "{name}");
    }
}

/// The clover multigrid split over processes (threads standing in for them) is the same
/// cycle as on one process: with a fixed number of coarse iterations and SAP steps, one
/// application and a whole FGMRES solve agree to rounding, in as many iterations.
#[test]
fn multigrid_on_several_processes() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 571);
    let initial = random_spinor(&whole, 8, 572);
    let b = random_spinor(&whole, 2, 573);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 30,
    };
    let run = |lat: Arc<Lattice>| {
        let op = clover_at::<f64>(&lat, &local_part(&u, &lat), 0.2);
        let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
        let parts = CycleParts {
            operator: Box::new(Bound::new(&op, Variant::D)),
            pre_smoother: None,
            post_smoother: Box::new(Sap::new(&op, &agg, 4, 6)),
            // Always 20 iterations: the cycle is then the same map on any split.
            coarse_solver: Box::new(BiCgStab::<C64Field>::new(Stop::simple(1e-40, 20))),
        };
        let mut mg = Multigrid::with_parts(
            &op,
            &agg,
            local_part(&initial, &lat),
            2,
            parts,
            SetupHooks::default(),
        )
        .unwrap();
        let b = local_part(&b, &lat);
        let mut once = b.zeros_like();
        mg.mult(&mut once, &b);
        let mut x = b.zeros_like();
        let info = Fgmres::new(stop)
            .solve(&mut Bound::new(&op, Variant::D), Some(&mut mg), &mut x, &b)
            .unwrap();
        (once, x, info.iterations)
    };
    let (once, x, iterations) = run(whole.clone());
    for grid in [[1, 1, 1, 2], [2, 1, 2, 1], [2, 2, 1, 2]] {
        let parts = LocalComm::run(global, grid, run);
        let (mut ones, mut xs) = (Vec::new(), Vec::new());
        let mut split_iterations = Vec::new();
        for (o, x, i) in parts {
            ones.push(o);
            xs.push(x);
            split_iterations.push(i);
        }
        let (d_once, d_x) = (
            max_abs_diff(&assemble(&whole, &ones), &once),
            max_abs_diff(&assemble(&whole, &xs), &x),
        );
        eprintln!(
            "grid {grid:?}: one cycle differs by {d_once:e}, the solution by {d_x:e}; {:?} iterations against {iterations}",
            split_iterations
        );
        assert!(d_once < 1e-10 && d_x < 1e-10, "grid {grid:?}");
        assert!(split_iterations.iter().all(|i| i.abs_diff(iterations) <= 1));
    }
}

/// The multigrid of the whole lattice under a solver on the even sites (`OnParity`): the even block
/// of an approximate inverse of D approximates the inverse of the Schur complement, so FGMRES on
/// the Schur complement needs about as many iterations as on D, and far fewer than without it.
#[test]
fn multigrid_under_an_even_odd_solver() {
    let lat = lattice();
    let op = clover_at::<f64>(&lat, &random_gauge(&lat, 581), 0.2);
    let b = random_spinor(&lat, 1, 582);
    let index = op.even_odd_index().unwrap().clone();
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 30,
    };
    let mut mg = Multigrid::new(&op, &lat, &MultigridParams::bridge([2, 2, 2, 2])).unwrap();
    let whole = {
        let mut x = b.zeros_like();
        Fgmres::new(stop)
            .solve(&mut Bound::new(&op, Variant::D), Some(&mut mg), &mut x, &b)
            .unwrap()
            .iterations
    };
    let even_odd = |m: Option<&mut dyn LinearFn<V64>>| {
        let mut x = b.zeros_like();
        let info = solve_even_odd(&op, &index, &mut x, &b, |schur, x_e, rhs| {
            let mut on_even = m.map(|m| OnParity::new(m, &index, &b));
            let m = on_even
                .as_mut()
                .map(|m| m as &mut dyn LinearFn<Field<f64, Spinor, ruttice::Half>>);
            Fgmres::new(stop).solve(schur, m, x_e, rhs)
        })
        .unwrap();
        assert!(residual_sqr(&op, &x, &b)[0] < 1e-22);
        info.iterations
    };
    let plain = even_odd(None);
    let with_mg = even_odd(Some(&mut mg));
    eprintln!("FGMRES iterations: on D with the multigrid {whole}; on the even sites {plain} alone, {with_mg} with it");
    assert!(with_mg <= whole + 2 && with_mg * 3 < plain);
}

/// A level below the coarse lattice (super-blocks of 2^4 coarse sites, solved exactly)
/// under the coarse solves of the clover multigrid, from the setup's test vectors on the
/// coarse lattice: FGMRES still converges, in about as many iterations.
#[test]
fn multigrid_with_a_level_below_the_coarse_lattice() {
    let lat = lattice();
    let op = clover_at::<f64>(&lat, &random_gauge(&lat, 591), 0.2);
    let b = random_spinor(&lat, 1, 592);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 30,
        max_restarts: 30,
    };
    let solve = |params: MultigridParams| {
        let mut mg = Multigrid::new(&op, &lat, &params).unwrap();
        let mut x = b.zeros_like();
        let info = Fgmres::new(stop)
            .solve(&mut Bound::new(&op, Variant::D), Some(&mut mg), &mut x, &b)
            .unwrap();
        assert!(residual_sqr(&op, &x, &b)[0] < 1e-24);
        info.iterations
    };
    let two = solve(MultigridParams::bridge([2, 2, 2, 2]));
    let three = solve(MultigridParams {
        levels: [2, 0, 0],
        ..MultigridParams::bridge([2, 2, 2, 2])
    });
    let even = solve(MultigridParams {
        levels: [2, 0, 0],
        coarse_even_odd: true,
        ..MultigridParams::bridge([2, 2, 2, 2])
    });
    eprintln!(
        "FGMRES iterations: {two} with two levels, {three} with a level below the coarse lattice, {even} with that on the even coarse sites"
    );
    assert!(three <= two + 2);
    assert!(even <= two + 2);
}
