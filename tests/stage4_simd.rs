//! Checks: fields in blocks of W sites (`Aosoa<W>`, the SIMD layout) against the same
//! fields in the usual layout.

mod common;

use common::max_abs_diff;
use ruttice::fermion::{
    solve_even_odd, Bound, Clover, CloverParams, CloverSimd, Dagger, DiagTerm, EoForm, EvenOdd,
    FermionOp, Mdwf, MdwfParams, PvEvenOdd, Variant, Wilson, WilsonOp, WilsonParams, WilsonSimd,
};
use ruttice::gauge::random_gauge;
use ruttice::kernel::{Dirac, LaneOf};
use ruttice::solver::preconditioner::multigrid::{pv_eo_basis, Aggregation, PvEoCycle, PvEoParams};
use ruttice::solver::{solve_cgne, BiCgStab, Cg, Gcr, Solver, Stop};
use ruttice::{
    assemble, extract_parity, insert_parity, local_part, random_spinor, Aosoa, Boundary,
    ConvertFrom, Field, Gauge, Half, Lattice, LinearFn, LocalComm, Parity, Spinor, Storage, Vector,
    C64, ND,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;
type B<const W: usize> = Field<f64, Spinor, ruttice::Lexical, Aosoa<W>>;

fn lattice() -> Arc<Lattice> {
    Lattice::single_node([4, 4, 4, 8])
}

fn to_blocks<const W: usize>(a: &V) -> B<W> {
    let mut b = B::<W>::with_shape(a.lattice(), 24, 1, a.num_vectors(), 1);
    b.convert_from(a);
    b
}

fn to_sites<const W: usize>(b: &B<W>) -> V {
    let mut a = V::spinor(b.lattice(), b.num_vectors());
    a.convert_from(b);
    a
}

/// Converting to blocks and back gives the same field, and the blocks hold every real of
/// a block once per lane.
#[test]
fn conversion_is_a_rearrangement() {
    let lat = lattice();
    let a = random_spinor(&lat, 2, 901);
    fn check<const W: usize>(a: &V) {
        let b = to_blocks::<W>(a);
        assert_eq!(b.block_len(), 2 * 24 * W);
        assert_eq!(b.num_blocks() * W, a.num_sites());
        assert!(to_sites(&b).as_slice() == a.as_slice(), "W = {W}");
        let blocks = ruttice::Blocks::new(a.lattice().local, W).unwrap();
        let (block, lane, vector, c) = (7 % b.num_blocks(), W - 1, 1, 5);
        let at = lane + W * (c + 24 * (vector + 2 * block));
        assert_eq!(
            b.as_slice()[at],
            a.site(0, blocks.site(block, lane))[24 * vector + c]
        );
    }
    check::<1>(&a);
    check::<2>(&a);
    check::<4>(&a);
    check::<8>(&a);
    check::<16>(&a);
}

/// The vector operations the solvers use give the same results in both layouts: updates
/// bit for bit, sums up to the order of the additions.
#[test]
fn vector_operations_agree_with_the_usual_layout() {
    let lat = lattice();
    let (x, y) = (random_spinor(&lat, 3, 911), random_spinor(&lat, 3, 912));
    let coefficients = [C64::new(0.3, -1.1), C64::new(-0.7, 0.2), C64::new(1.5, 0.9)];
    fn check<const W: usize>(x: &V, y: &V, k: &[C64; 3]) {
        let (bx, by) = (to_blocks::<W>(x), to_blocks::<W>(y));

        let (mut a, mut b) = (y.zeros_like(), by.zeros_like());
        a.copy_from(y);
        b.copy_from(&by);
        a.axpy(k, x);
        b.axpy(k, &bx);
        assert!(to_sites(&b).as_slice() == a.as_slice(), "axpy, W = {W}");
        a.aypx(k, x);
        b.aypx(k, &bx);
        assert!(to_sites(&b).as_slice() == a.as_slice(), "aypx, W = {W}");
        a.scale(&[0.5, -2.0, 3.0]);
        b.scale(&[0.5, -2.0, 3.0]);
        assert!(to_sites(&b).as_slice() == a.as_slice(), "scale, W = {W}");

        let (mut dot_a, mut dot_b) = ([C64::ZERO; 3], [C64::ZERO; 3]);
        let (mut norm_a, mut norm_b) = ([0.0; 3], [0.0; 3]);
        a.dot(x, &mut dot_a);
        b.dot(&bx, &mut dot_b);
        a.norm_sqr(&mut norm_a);
        b.norm_sqr(&mut norm_b);
        for j in 0..3 {
            assert!((dot_a[j] - dot_b[j]).norm_sqr().sqrt() < 1e-12 * norm_a[j].sqrt());
            assert!((norm_a[j] - norm_b[j]).abs() < 1e-13 * norm_a[j]);
        }

        b.set_zero();
        assert!(b.as_slice().iter().all(|v| *v == 0.0));
        assert!(b.same_shape(&bx) && b.num_vectors() == 3);
    }
    check::<1>(&x, &y, &coefficients);
    check::<4>(&x, &y, &coefficients);
    check::<8>(&x, &y, &coefficients);
}

/// Single precision blocks from a double precision field, and back.
#[test]
fn conversion_can_change_the_precision() {
    let lat = lattice();
    let a = random_spinor(&lat, 1, 921);
    let mut single =
        Field::<f32, Spinor, ruttice::Lexical, Aosoa<8>>::with_shape(&lat, 24, 1, 1, 1);
    single.convert_from(&a);
    let mut back = V::spinor(&lat, 1);
    back.convert_from(&single);
    let diff = max_abs_diff(&back, &a);
    assert!(diff > 0.0 && diff < 1e-7, "{diff:e}");
}

const MIXED: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];

fn params() -> WilsonParams {
    WilsonParams {
        kappa: 0.12,
        boundary: MIXED,
    }
}

fn simd<T: Storage, const W: usize>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> WilsonSimd<T, W>
where
    T::Compute: LaneOf<W>,
{
    let mut op = WilsonSimd::<T, W>::new(lat, params()).unwrap();
    op.set_gauge(u).unwrap();
    op
}

/// The operator on blocks gives every site the bits the operator on sites gives it: D, D†
/// and γ5 D, two spinors per site, every kind of boundary, every width. With W = 1 a block
/// is one site; with W = 16 every direction is cut and every hop out of a sub-lattice
/// rotates lanes.
#[test]
fn wilson_on_blocks_is_identical() {
    let lat = lattice();
    let u = random_gauge(&lat, 931);
    let psi = random_spinor(&lat, 2, 932);
    let mut reference = Wilson::<f64>::new(&lat, params());
    reference.set_gauge(&u).unwrap();
    fn check<const W: usize>(reference: &Wilson<f64>, u: &Field<f64, Gauge>, psi: &V)
    where
        f64: LaneOf<W>,
    {
        let op = simd::<f64, W>(psi.lattice(), u);
        let inp = to_blocks::<W>(psi);
        for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
            let mut want = psi.zeros_like();
            reference.mult(&mut reference.work(), &mut want, psi, variant);
            let mut out = inp.zeros_like();
            op.mult(&mut op.work(), &mut out, &inp, variant);
            assert!(
                to_sites(&out).as_slice() == want.as_slice(),
                "W = {W}, {variant:?}: max difference {:e}",
                max_abs_diff(&to_sites(&out), &want)
            );
        }
    }
    check::<1>(&reference, &u, &psi);
    check::<2>(&reference, &u, &psi);
    check::<4>(&reference, &u, &psi);
    check::<8>(&reference, &u, &psi);
}

#[test]
fn wilson_on_blocks_is_identical_in_single_precision() {
    let lat = lattice();
    let u = random_gauge(&lat, 941);
    let mut psi = Field::<f32, Spinor>::spinor(&lat, 1);
    psi.convert_from(&random_spinor(&lat, 1, 942));
    let mut reference = Wilson::<f32>::new(&lat, params());
    reference.set_gauge(&u).unwrap();
    let mut want = psi.zeros_like();
    reference.mult(&mut reference.work(), &mut want, &psi, Variant::Dagger);
    fn check<const W: usize>(
        u: &Field<f64, Gauge>,
        psi: &Field<f32, Spinor>,
        want: &Field<f32, Spinor>,
    ) where
        f32: LaneOf<W>,
    {
        let lat = psi.lattice();
        let op = simd::<f32, W>(lat, u);
        let mut inp =
            Field::<f32, Spinor, ruttice::Lexical, Aosoa<W>>::with_shape(lat, 24, 1, 1, 1);
        inp.convert_from(psi);
        let mut out = inp.zeros_like();
        op.mult(&mut op.work(), &mut out, &inp, Variant::Dagger);
        let mut back = psi.zeros_like();
        back.convert_from(&out);
        assert!(back.as_slice() == want.as_slice(), "W = {W}");
    }
    check::<4>(&u, &psi, &want);
    check::<8>(&u, &psi, &want);
    check::<16>(&u, &psi, &want);
}

/// The solvers know only vectors and operators, so they run on blocks as they are.
#[test]
fn solvers_run_on_blocks() {
    let lat = lattice();
    let u = random_gauge(&lat, 951);
    let b = random_spinor(&lat, 2, 952);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 1000,
        max_restarts: 3,
    };
    let op = simd::<f64, 4>(&lat, &u);
    let rhs = to_blocks::<4>(&b);

    let mut x = rhs.zeros_like();
    solve_cgne(
        &mut Cg::new(stop),
        &mut Bound::new(&op, Variant::D),
        &mut x,
        &rhs,
    )
    .unwrap();
    let mut y = rhs.zeros_like();
    BiCgStab::new(stop)
        .solve(&mut Bound::new(&op, Variant::D), None, &mut y, &rhs)
        .unwrap();

    // The residual, with the operator on sites.
    let mut reference = Wilson::<f64>::new(&lat, params());
    reference.set_gauge(&u).unwrap();
    for (name, solution) in [("CG", to_sites(&x)), ("BiCGStab", to_sites(&y))] {
        let mut r = b.zeros_like();
        reference.mult(&mut reference.work(), &mut r, &solution, Variant::D);
        r.scale(&[-1.0; 2]);
        r.axpy(&[C64::new(1.0, 0.0); 2], &b);
        let (mut rr, mut bb) = ([0.0; 2], [0.0; 2]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        assert!(
            rr[0] / bb[0] < 2e-24 && rr[1] / bb[1] < 2e-24,
            "{name}: {rr:?}"
        );
    }
}

/// A lattice that cannot be cut into W equal sub-lattices is refused when the operator is
/// built.
#[test]
fn lattices_that_cannot_be_cut_are_refused() {
    let odd = Lattice::single_node([3, 3, 3, 4]);
    assert!(WilsonSimd::<f64, 4>::new(&odd, params()).is_some());
    assert!(WilsonSimd::<f64, 8>::new(&odd, params()).is_none());
}

fn clover_params() -> CloverParams {
    CloverParams {
        kappa: 0.12,
        csw: 1.3,
        boundary: MIXED,
    }
}

/// The clover operator on blocks gives every site the bits the operator on sites gives it.
/// Its hopping term works on blocks, its clover term on one lane at a time.
#[test]
fn clover_on_blocks_is_identical() {
    let lat = lattice();
    let u = random_gauge(&lat, 961);
    let psi = random_spinor(&lat, 2, 962);
    let mut reference = Clover::<f64>::new(&lat, clover_params());
    reference.set_gauge(&u).unwrap();
    fn check<const W: usize>(reference: &Clover<f64>, u: &Field<f64, Gauge>, psi: &V)
    where
        f64: LaneOf<W>,
    {
        let mut op = CloverSimd::<f64, W>::new(psi.lattice(), clover_params()).unwrap();
        op.set_gauge(u).unwrap();
        assert_eq!(op.flops_per_site(), reference.flops_per_site());
        let inp = to_blocks::<W>(psi);
        for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
            let mut want = psi.zeros_like();
            reference.mult(&mut reference.work(), &mut want, psi, variant);
            let mut out = inp.zeros_like();
            op.mult(&mut op.work(), &mut out, &inp, variant);
            assert!(
                to_sites(&out).as_slice() == want.as_slice(),
                "W = {W}, {variant:?}: max difference {:e}",
                max_abs_diff(&to_sites(&out), &want)
            );
        }
    }
    check::<1>(&reference, &u, &psi);
    check::<4>(&reference, &u, &psi);
    check::<8>(&reference, &u, &psi);
}

type HalfSites = Field<f64, Spinor, Half>;
type HalfBlocks<const W: usize> = Field<f64, Spinor, Half, Aosoa<W>>;

/// The even-odd blocks of an operator on fields in blocks against the same blocks on
/// sites: hops onto both parities, the diagonal term and its inverse, for M and M†. A half
/// field in blocks holds the blocks of one parity, so the two are compared after putting
/// the results back into whole fields.
fn even_odd_blocks_agree<D: DiagTerm<f64>, const W: usize>(
    sites: &WilsonOp<f64, Dirac, D>,
    blocks: &WilsonOp<f64, Dirac, D, Aosoa<W>>,
    psi: &V,
) where
    f64: LaneOf<W>,
{
    let lat = psi.lattice();
    let copies = psi.num_vectors();
    let (site_index, block_index) = (
        sites.even_odd_index().unwrap(),
        blocks.even_odd_index().unwrap(),
    );
    let psi_blocks = to_blocks::<W>(psi);
    let half_sites = || HalfSites::with_shape(lat, 24, 1, copies, 1);
    let half_blocks = || HalfBlocks::<W>::with_shape(lat, 24, 1, copies, 1);
    let mut work = sites.even_odd_work();
    let mut block_work = blocks.even_odd_work();

    for to in [Parity::Even, Parity::Odd] {
        for dagger in [Dagger::No, Dagger::Yes] {
            // Three results per case: the hop onto `to`, and the diagonal term and its
            // inverse on `to`.
            let (mut inp_s, mut same_s) = (half_sites(), half_sites());
            let (mut inp_b, mut same_b) = (half_blocks(), half_blocks());
            extract_parity(&mut inp_s, psi, site_index, to.flip());
            extract_parity(&mut same_s, psi, site_index, to);
            extract_parity(&mut inp_b, &psi_blocks, block_index, to.flip());
            extract_parity(&mut same_b, &psi_blocks, block_index, to);
            assert_eq!(inp_b.num_sites(), inp_s.num_sites());

            for case in 0..3 {
                let (mut out_s, mut out_b) = (half_sites(), half_blocks());
                out_s.set_parity(to);
                out_b.set_parity(to);
                match case {
                    0 => {
                        sites.hop(&mut work, &mut out_s, &inp_s, to, dagger);
                        blocks.hop(&mut block_work, &mut out_b, &inp_b, to, dagger);
                    }
                    1 => {
                        sites.diag(&mut work, &mut out_s, &same_s, to, dagger);
                        blocks.diag(&mut block_work, &mut out_b, &same_b, to, dagger);
                    }
                    _ => {
                        sites.diag_inv(&mut work, &mut out_s, &same_s, to, dagger);
                        blocks.diag_inv(&mut block_work, &mut out_b, &same_b, to, dagger);
                    }
                }
                let (mut whole_s, mut whole_b) = (psi.zeros_like(), psi_blocks.zeros_like());
                insert_parity(&mut whole_s, &out_s, site_index);
                insert_parity(&mut whole_b, &out_b, block_index);
                let mut norm = vec![0.0; copies];
                whole_s.norm_sqr(&mut norm);
                assert!(norm[0] > 0.0);
                assert!(
                    to_sites(&whole_b).as_slice() == whole_s.as_slice(),
                    "W = {W}, onto {to:?}, {dagger:?}, case {case}: max difference {:e}",
                    max_abs_diff(&to_sites(&whole_b), &whole_s)
                );
            }
        }
    }
}

#[test]
fn even_odd_blocks_of_wilson_are_identical() {
    let lat = lattice();
    let u = random_gauge(&lat, 971);
    let psi = random_spinor(&lat, 2, 972);
    let mut reference = Wilson::<f64>::new(&lat, params());
    reference.set_gauge(&u).unwrap();
    even_odd_blocks_agree(&reference, &simd::<f64, 1>(&lat, &u), &psi);
    even_odd_blocks_agree(&reference, &simd::<f64, 2>(&lat, &u), &psi);
    even_odd_blocks_agree(&reference, &simd::<f64, 4>(&lat, &u), &psi);
    even_odd_blocks_agree(&reference, &simd::<f64, 8>(&lat, &u), &psi);
}

#[test]
fn even_odd_blocks_of_clover_are_identical() {
    let lat = lattice();
    let u = random_gauge(&lat, 981);
    let psi = random_spinor(&lat, 2, 982);
    let mut reference = Clover::<f64>::new(&lat, clover_params());
    reference.set_gauge(&u).unwrap();
    fn check<const W: usize>(reference: &Clover<f64>, u: &Field<f64, Gauge>, psi: &V)
    where
        f64: LaneOf<W>,
    {
        let mut op = CloverSimd::<f64, W>::new(psi.lattice(), clover_params()).unwrap();
        op.set_gauge(u).unwrap();
        even_odd_blocks_agree(reference, &op, psi);
    }
    check::<4>(&reference, &u, &psi);
    check::<8>(&reference, &u, &psi);
}

/// The even-odd solve is the same generic code for both layouts; on blocks it gives a
/// solution of the whole system, checked with the operator on sites.
#[test]
fn even_odd_solve_runs_on_blocks() {
    let lat = lattice();
    let u = random_gauge(&lat, 991);
    let b = random_spinor(&lat, 2, 992);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 1000,
        max_restarts: 3,
    };
    let rhs = to_blocks::<4>(&b);
    let residual = |reference: &dyn Fn(&mut V, &V), solution: &V| {
        let mut r = b.zeros_like();
        reference(&mut r, solution);
        r.scale(&[-1.0; 2]);
        r.axpy(&[C64::new(1.0, 0.0); 2], &b);
        let (mut rr, mut bb) = ([0.0; 2], [0.0; 2]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        (rr[0] / bb[0]).max(rr[1] / bb[1])
    };

    let op = simd::<f64, 4>(&lat, &u);
    let mut x = rhs.zeros_like();
    let index = op.even_odd_index().unwrap();
    let whole = solve_cgne(
        &mut Cg::new(stop),
        &mut Bound::new(&op, Variant::D),
        &mut x,
        &rhs,
    )
    .unwrap();
    let mut y = rhs.zeros_like();
    let halved = solve_even_odd(&op, index, &mut y, &rhs, |schur, x_e, rhs| {
        solve_cgne(&mut Cg::new(stop), schur, x_e, rhs)
    })
    .unwrap();
    assert!(
        halved.iterations < whole.iterations,
        "{} iterations with even-odd, {} without",
        halved.iterations,
        whole.iterations
    );
    let mut reference = Wilson::<f64>::new(&lat, params());
    reference.set_gauge(&u).unwrap();
    let wilson = |out: &mut V, inp: &V| reference.mult(&mut reference.work(), out, inp, Variant::D);
    let r = residual(&wilson, &to_sites(&y));
    assert!(r < 1e-22, "Wilson: {r:e}");

    let mut op = CloverSimd::<f64, 4>::new(&lat, clover_params()).unwrap();
    op.set_gauge(&u).unwrap();
    let mut z = rhs.zeros_like();
    let index = op.even_odd_index().unwrap();
    solve_even_odd(&op, index, &mut z, &rhs, |schur, x_e, rhs| {
        BiCgStab::new(stop).solve(schur, None, x_e, rhs)
    })
    .unwrap();
    let mut reference = Clover::<f64>::new(&lat, clover_params());
    reference.set_gauge(&u).unwrap();
    let clover = |out: &mut V, inp: &V| reference.mult(&mut reference.work(), out, inp, Variant::D);
    let r = residual(&clover, &to_sites(&z));
    assert!(r < 1e-22, "clover: {r:e}");
}

/// Blocks have one parity only if the lattice of blocks has even extents; otherwise the
/// operator offers no even-odd numbering.
#[test]
fn even_odd_needs_a_lattice_of_blocks_with_even_extents() {
    let lat = Lattice::single_node([4, 4, 6, 8]);
    // Four lanes halve t and z: the blocks form a 4 x 4 x 3 x 4 lattice.
    let op = WilsonSimd::<f64, 4>::new(&lat, params()).unwrap();
    assert!(op.even_odd_index().is_none());
    let op = WilsonSimd::<f64, 2>::new(&lat, params()).unwrap();
    assert!(op.even_odd_index().is_some());
}

/// The fused vector operations give the bits of the compositions they stand for: the
/// vectors bit for bit, and the sums too, because both add the sites in the same chunks.
/// Checked on a lattice large enough for the parallel path, in both layouts.
#[test]
fn fused_vector_operations_are_their_compositions() {
    fn check<X: Vector>(x: &X, y: &X, z: &X, same: impl Fn(&X, &X) -> bool, layout: &str) {
        let a = [C64::new(0.3, -1.1), C64::new(-0.7, 0.2)];
        let b = [C64::new(1.5, 0.9), C64::new(0.0, -0.4)];
        let copy = |v: &X| {
            let mut c = v.zeros_like();
            c.copy_from(v);
            c
        };

        // axpy_norm_sqr
        let (mut fused, mut plain) = (copy(z), copy(z));
        let (mut n1, mut n2) = ([0.0; 2], [0.0; 2]);
        fused.axpy_norm_sqr(&a, x, &mut n1);
        plain.axpy(&a, x);
        plain.norm_sqr(&mut n2);
        assert!(same(&fused, &plain) && n1 == n2, "{layout}: axpy_norm_sqr");

        // xpay
        let (mut fused, mut plain) = (copy(z), copy(z));
        fused.xpay(x, &a, y);
        plain.copy_from(x);
        plain.axpy(&a, y);
        assert!(same(&fused, &plain), "{layout}: xpay");

        // xpay_norm_sqr_dot
        let (mut fused, mut plain) = (copy(z), copy(z));
        let (mut d1, mut d2) = ([C64::ZERO; 2], [C64::ZERO; 2]);
        fused.xpay_norm_sqr_dot(x, &a, y, &mut n1, z, &mut d1);
        plain.copy_from(x);
        plain.axpy(&a, y);
        plain.norm_sqr(&mut n2);
        z.dot(&plain, &mut d2);
        assert!(same(&fused, &plain), "{layout}: xpay_norm_sqr_dot");
        assert!(n1 == n2 && d1 == d2, "{layout}: sums of xpay_norm_sqr_dot");

        // axpy2
        let (mut fused, mut plain) = (copy(z), copy(z));
        fused.axpy2(&a, x, &b, y);
        plain.axpy(&a, x);
        plain.axpy(&b, y);
        assert!(same(&fused, &plain), "{layout}: axpy2");

        // axpy_aypx
        let (mut fused, mut plain) = (copy(z), copy(z));
        fused.axpy_aypx(&a, x, &b, y);
        plain.axpy(&a, x);
        plain.aypx(&b, y);
        assert!(same(&fused, &plain), "{layout}: axpy_aypx");

        // dot_norms
        let (mut m1, mut m2) = ([0.0; 2], [0.0; 2]);
        let (mut k1, mut k2) = ([0.0; 2], [0.0; 2]);
        x.dot_norms(y, &mut d1, &mut m1, &mut k1);
        x.dot(y, &mut d2);
        x.norm_sqr(&mut m2);
        y.norm_sqr(&mut k2);
        assert!(d1 == d2 && m1 == m2 && k1 == k2, "{layout}: dot_norms");
    }
    let lat = Lattice::single_node([8, 8, 8, 16]);
    let (x, y, z) = (
        random_spinor(&lat, 2, 1101),
        random_spinor(&lat, 2, 1102),
        random_spinor(&lat, 2, 1103),
    );
    check(&x, &y, &z, |a, b| a.as_slice() == b.as_slice(), "sites");
    check(
        &to_blocks::<4>(&x),
        &to_blocks::<4>(&y),
        &to_blocks::<4>(&z),
        |a, b| a.as_slice() == b.as_slice(),
        "blocks",
    );
    let single = |v: &V| {
        let mut s = Field::<f32, Spinor>::spinor(&lat, 2);
        s.convert_from(v);
        s
    };
    check(
        &single(&x),
        &single(&y),
        &single(&z),
        |a, b| a.as_slice() == b.as_slice(),
        "single precision",
    );
}

// ---------------------------------------------------------------------------------------
// Several processes (threads of the test, `LocalComm`).

const GLOBAL: [usize; ND] = [4, 4, 8, 8];

/// Splits that leave lattices of blocks with even extents for W = 4, and some that do not.
const GRIDS: [[usize; ND]; 5] = [
    [1, 1, 2, 2],
    [2, 2, 1, 1],
    [1, 2, 2, 2],
    [1, 1, 1, 4],
    [1, 1, 1, 1],
];

/// The operator on blocks on every process of every grid against the operator on sites on
/// one process: D, D† and γ5 D and, where the blocks have a parity, the hops onto both
/// parities for M and M†. `sites` and `blocks` build the two operators on a lattice.
fn blocks_on_processes_agree<D: DiagTerm<f64>, const W: usize>(
    sites: impl Fn(&Arc<Lattice>) -> WilsonOp<f64, Dirac, D>,
    blocks: impl Fn(&Arc<Lattice>) -> Option<WilsonOp<f64, Dirac, D, Aosoa<W>>> + Sync,
    u: &Field<f64, Gauge>,
    psi: &V,
) where
    f64: LaneOf<W>,
{
    const VARIANTS: [Variant; 3] = [Variant::D, Variant::Dagger, Variant::Gamma5D];
    let hops = || {
        [Parity::Even, Parity::Odd]
            .into_iter()
            .flat_map(|to| [Dagger::No, Dagger::Yes].map(|dagger| (to, dagger)))
    };
    let whole = psi.lattice();
    let copies = psi.num_vectors();

    let mut reference = sites(whole);
    reference.set_gauge(u).unwrap();
    let mut want: Vec<V> = VARIANTS
        .iter()
        .map(|variant| {
            let mut out = psi.zeros_like();
            reference.mult(&mut reference.work(), &mut out, psi, *variant);
            out
        })
        .collect();
    let index = reference.even_odd_index().unwrap();
    for (to, dagger) in hops() {
        let mut inp = HalfSites::with_shape(whole, 24, 1, copies, 1);
        let mut out = inp.zeros_like();
        extract_parity(&mut inp, psi, index, to.flip());
        out.set_parity(to);
        reference.hop(&mut reference.even_odd_work(), &mut out, &inp, to, dagger);
        let mut full = psi.zeros_like();
        insert_parity(&mut full, &out, index);
        want.push(full);
    }

    for grid in GRIDS {
        // Result i of every process, in rank order.
        let mut results: Vec<Vec<V>> = Vec::new();
        for part in LocalComm::run(GLOBAL, grid, |lat| {
            let mut op = blocks(&lat).unwrap();
            op.set_gauge(&local_part(u, &lat)).unwrap();
            let inp = to_blocks::<W>(&local_part(psi, &lat));
            let mut work = op.work();
            let mut got: Vec<V> = VARIANTS
                .iter()
                .map(|variant| {
                    let mut out = inp.zeros_like();
                    // Twice with the same work object: the exchange buffers are reused.
                    op.mult(&mut work, &mut out, &inp, *variant);
                    op.mult(&mut work, &mut out, &inp, *variant);
                    to_sites(&out)
                })
                .collect();
            let Some(index) = op.even_odd_index() else {
                return got;
            };
            let mut work = op.even_odd_work();
            for (to, dagger) in hops() {
                let mut half = HalfBlocks::<W>::with_shape(&lat, 24, 1, copies, 1);
                let mut out = half.zeros_like();
                extract_parity(&mut half, &inp, index, to.flip());
                out.set_parity(to);
                op.hop(&mut work, &mut out, &half, to, dagger);
                let mut full = inp.zeros_like();
                insert_parity(&mut full, &out, index);
                got.push(to_sites(&full));
            }
            got
        }) {
            for (i, field) in part.into_iter().enumerate() {
                if results.len() <= i {
                    results.push(Vec::new());
                }
                results[i].push(field);
            }
        }
        assert!(results.len() == 3 || results.len() == want.len());
        for (i, (parts, want)) in results.iter().zip(&want).enumerate() {
            let got = assemble(whole, parts);
            assert!(
                got.as_slice() == want.as_slice(),
                "W = {W}, grid {grid:?}, result {i}: max difference {:e}",
                max_abs_diff(&got, want)
            );
        }
    }
}

#[test]
fn wilson_on_blocks_is_identical_on_several_processes() {
    let whole = Lattice::single_node(GLOBAL);
    let u = random_gauge(&whole, 1001);
    let psi = random_spinor(&whole, 2, 1002);
    let sites = |lat: &Arc<Lattice>| Wilson::<f64>::new(lat, params());
    blocks_on_processes_agree(
        sites,
        |lat| WilsonSimd::<f64, 1>::new(lat, params()),
        &u,
        &psi,
    );
    blocks_on_processes_agree(
        sites,
        |lat| WilsonSimd::<f64, 2>::new(lat, params()),
        &u,
        &psi,
    );
    blocks_on_processes_agree(
        sites,
        |lat| WilsonSimd::<f64, 4>::new(lat, params()),
        &u,
        &psi,
    );
    blocks_on_processes_agree(
        sites,
        |lat| WilsonSimd::<f64, 8>::new(lat, params()),
        &u,
        &psi,
    );
}

#[test]
fn clover_on_blocks_is_identical_on_several_processes() {
    let whole = Lattice::single_node(GLOBAL);
    let u = random_gauge(&whole, 1011);
    let psi = random_spinor(&whole, 2, 1012);
    let sites = |lat: &Arc<Lattice>| Clover::<f64>::new(lat, clover_params());
    let blocks = |lat: &Arc<Lattice>| CloverSimd::<f64, 4>::new(lat, clover_params());
    blocks_on_processes_agree(sites, blocks, &u, &psi);
}

/// The even-odd solve on blocks on several processes solves the system of the whole
/// lattice.
#[test]
fn even_odd_solve_runs_on_blocks_on_several_processes() {
    let whole = Lattice::single_node(GLOBAL);
    let u = random_gauge(&whole, 1021);
    let b = random_spinor(&whole, 1, 1022);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 1000,
        max_restarts: 3,
    };
    let parts = LocalComm::run(GLOBAL, [1, 1, 2, 2], |lat| {
        let mut op = CloverSimd::<f64, 4>::new(&lat, clover_params()).unwrap();
        op.set_gauge(&local_part(&u, &lat)).unwrap();
        let rhs = to_blocks::<4>(&local_part(&b, &lat));
        let mut x = rhs.zeros_like();
        let index = op.even_odd_index().unwrap();
        solve_even_odd(&op, index, &mut x, &rhs, |schur, x_e, rhs| {
            BiCgStab::new(stop).solve(schur, None, x_e, rhs)
        })
        .unwrap();
        to_sites(&x)
    });
    let x = assemble(&whole, &parts);
    let mut reference = Clover::<f64>::new(&whole, clover_params());
    reference.set_gauge(&u).unwrap();
    let mut r = b.zeros_like();
    reference.mult(&mut reference.work(), &mut r, &x, Variant::D);
    r.scale(&[-1.0]);
    r.axpy(&[C64::new(1.0, 0.0)], &b);
    let (mut rr, mut bb) = ([0.0], [0.0]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    assert!(rr[0] / bb[0] < 1e-22, "{:e}", rr[0] / bb[0]);
}

/// A random field with `ls` spinors per site in each of `nv` vectors.
fn random_5d(lat: &Arc<Lattice>, ls: usize, nv: usize, seed: u64) -> V {
    let flat = random_spinor(lat, ls * nv, seed);
    let mut v = V::with_shape(lat, 24, ls, nv, 1);
    v.as_mut_slice().copy_from_slice(flat.as_slice());
    v
}

fn to_blocks_5d<const W: usize>(a: &V) -> B<W> {
    let mut b = B::<W>::with_shape(a.lattice(), 24, a.ls(), a.num_vectors(), 1);
    b.convert_from(a);
    b
}

fn to_sites_5d<const W: usize>(b: &B<W>) -> V {
    let mut a = V::with_shape(b.lattice(), 24, b.ls(), b.num_vectors(), 1);
    a.convert_from(b);
    a
}

const MOBIUS: MdwfParams = MdwfParams {
    ls: 6,
    m5: 1.4,
    mass: 0.03,
    b: 1.5,
    c: 0.5,
    boundary: [
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Antiperiodic,
    ],
};

/// Möbius domain-wall fermions on fields in blocks of W sites, against the usual layout:
/// D, D† and Γ5 D, with and without Neff's α; the even-odd Pauli-Villars operator Ã in both
/// forms; and the solution of D x = b through it. The fifth dimension is the same code on
/// numbers of W lanes, so the results agree to rounding.
#[test]
fn mdwf_on_blocks_is_identical() {
    let lat = lattice();
    let u = random_gauge(&lat, 991);
    let psi = random_5d(&lat, MOBIUS.ls, 2, 992);
    let b = random_5d(&lat, MOBIUS.ls, 1, 993);
    let pv_params = MdwfParams {
        mass: 1.0,
        ..MOBIUS
    };
    fn check<const W: usize>(
        lat: &Arc<Lattice>,
        u: &Field<f64, Gauge>,
        psi: &V,
        b: &V,
        pv_params: MdwfParams,
    ) where
        f64: LaneOf<W>,
    {
        let tol = 1e-12;
        let sites = |p: MdwfParams| {
            let mut op = Mdwf::<f64>::new(lat, p);
            op.set_gauge(u).unwrap();
            op
        };
        let blocks = |p: MdwfParams| {
            let mut op = Mdwf::<f64, Dirac, Aosoa<W>>::in_blocks(lat, p).unwrap();
            op.set_gauge(u).unwrap();
            op
        };
        let (mut op_s, mut op_b) = (sites(MOBIUS), blocks(MOBIUS));
        let (pv_s, pv_b) = (sites(pv_params), blocks(pv_params));
        let psi_b = to_blocks_5d::<W>(psi);
        for alpha in [1.0, 0.6] {
            op_s.set_alpha(alpha);
            op_b.set_alpha(alpha);
            for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
                let (mut out_s, mut out_b) = (psi.zeros_like(), psi_b.zeros_like());
                op_s.mult(&mut op_s.work(), &mut out_s, psi, variant);
                op_b.mult(&mut op_b.work(), &mut out_b, &psi_b, variant);
                let d = max_abs_diff(&to_sites_5d(&out_b), &out_s);
                assert!(d < tol, "W = {W}, α = {alpha}, {variant:?}: {d:e}");
            }
            // Ã in both forms on one vector of the even sites.
            let (index_s, index_b) = (
                op_s.even_odd_index().unwrap().clone(),
                op_b.even_odd_index().unwrap().clone(),
            );
            let mut even_s = Field::<f64, Spinor, Half>::with_shape(lat, 24, MOBIUS.ls, 2, 1);
            let mut even_b =
                Field::<f64, Spinor, Half, Aosoa<W>>::with_shape(lat, 24, MOBIUS.ls, 2, 1);
            extract_parity(&mut even_s, psi, &index_s, Parity::Even);
            extract_parity(&mut even_b, &psi_b, &index_b, Parity::Even);
            for form in [EoForm::Right, EoForm::Left] {
                let mut a_s = PvEvenOdd::new(&op_s, &pv_s, form, &even_s);
                let mut a_b = PvEvenOdd::new(&op_b, &pv_b, form, &even_b);
                let (mut out_s, mut out_b) = (even_s.zeros_like(), even_b.zeros_like());
                a_s.mult(&mut out_s, &even_s);
                a_b.mult(&mut out_b, &even_b);
                let (mut whole_s, mut whole_b) = (psi.zeros_like(), psi_b.zeros_like());
                insert_parity(&mut whole_s, &out_s, &index_s);
                insert_parity(&mut whole_b, &out_b, &index_b);
                let d = max_abs_diff(&to_sites_5d(&whole_b), &whole_s);
                assert!(d < tol, "W = {W}, α = {alpha}, Ã {form:?}: {d:e}");
            }
        }
        // D x = b through Ã with GCR(4): the same solution in the same number of steps.
        let mut like_b = Field::<f64, Spinor, Half, Aosoa<W>>::with_shape(lat, 24, MOBIUS.ls, 1, 1);
        like_b.set_parity(Parity::Even);
        let mut like_s = Field::<f64, Spinor, Half>::with_shape(lat, 24, MOBIUS.ls, 1, 1);
        like_s.set_parity(Parity::Even);
        let b_b = to_blocks_5d::<W>(b);
        let stop = Stop::simple(1e-20, 400);
        let mut a_s = PvEvenOdd::new(&op_s, &pv_s, EoForm::Right, &like_s);
        let mut a_b = PvEvenOdd::new(&op_b, &pv_b, EoForm::Right, &like_b);
        let (mut rhs_s, mut rhs_b) = (like_s.zeros_like(), like_b.zeros_like());
        a_s.right_hand_side(&mut rhs_s, b);
        a_b.right_hand_side(&mut rhs_b, &b_b);
        let (mut y_s, mut y_b) = (like_s.zeros_like(), like_b.zeros_like());
        let info_s = Gcr::new(stop, 4)
            .solve(&mut a_s, None, &mut y_s, &rhs_s)
            .unwrap();
        let info_b = Gcr::new(stop, 4)
            .solve(&mut a_b, None, &mut y_b, &rhs_b)
            .unwrap();
        let (mut x_s, mut x_b) = (b.zeros_like(), b_b.zeros_like());
        a_s.solution(&mut x_s, &y_s, b);
        a_b.solution(&mut x_b, &y_b, &b_b);
        let d = max_abs_diff(&to_sites_5d(&x_b), &x_s);
        eprintln!(
            "W = {W}: GCR(4) {} steps on sites, {} on blocks; solutions differ by {d:e}",
            info_s.iterations, info_b.iterations
        );
        assert!(info_s.iterations.abs_diff(info_b.iterations) <= 1);
        assert!(d < 1e-8, "W = {W}: {d:e}");
    }
    check::<2>(&lat, &u, &psi, &b, pv_params);
    check::<4>(&lat, &u, &psi, &b, pv_params);
    check::<8>(&lat, &u, &psi, &b, pv_params);
}

/// The even-odd Pauli-Villars multigrid with the operator and the smoother in blocks of W
/// sites (the coarse correction in the usual layout): GCR(2) with it takes as many steps as
/// with the cycle in the usual layout, and gives the same solution.
#[test]
fn mdwf_multigrid_on_blocks_agrees() {
    let lat = lattice();
    let u = random_gauge(&lat, 995);
    let pv_params = MdwfParams {
        mass: 1.0,
        ..MOBIUS
    };
    let make = |p: MdwfParams| {
        let mut op = Mdwf::<f64>::new(&lat, p);
        op.set_gauge(&u).unwrap();
        op
    };
    let (op, pv) = (make(MOBIUS), make(pv_params));
    let index = op.even_odd_index().unwrap().clone();
    let mut y = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 4, 1);
    extract_parity(
        &mut y,
        &random_5d(&lat, MOBIUS.ls, 4, 996),
        &index,
        Parity::Even,
    );
    let basis = pv_eo_basis(&op, &pv, &Aggregation::new(&lat, [2, 2, 2, 2]), &y).unwrap();
    let b = random_5d(&lat, MOBIUS.ls, 1, 997);
    let stop = Stop::simple(1e-20, 200);

    let mut like = Field::<f64, Spinor, Half>::with_shape(&lat, 24, MOBIUS.ls, 1, 1);
    like.set_parity(Parity::Even);
    let mut cycle = PvEoCycle::new(
        &op,
        &pv,
        (&op, &pv, EoForm::Right),
        &basis,
        PvEoParams::default(),
    )
    .unwrap();
    let mut a = cycle.operator();
    let mut rhs = like.zeros_like();
    a.right_hand_side(&mut rhs, &b);
    let mut sol = like.zeros_like();
    let info = Gcr::new(stop, 2)
        .solve(&mut a, Some(&mut cycle), &mut sol, &rhs)
        .unwrap();
    let mut x = b.zeros_like();
    a.solution(&mut x, &sol, &b);

    fn in_blocks<const W: usize>(
        lat: &Arc<Lattice>,
        u: &Field<f64, Gauge>,
        (op, pv, pv_params): (&Mdwf<f64>, &Mdwf<f64>, MdwfParams),
        basis: &ruttice::solver::preconditioner::multigrid::PvEoBasis,
        b: &V,
        stop: Stop,
    ) -> (usize, V)
    where
        f64: LaneOf<W>,
    {
        let make = |p: MdwfParams| {
            let mut op = Mdwf::<f64, Dirac, Aosoa<W>>::in_blocks(lat, p).unwrap();
            op.set_gauge(u).unwrap();
            op
        };
        let (op_b, pv_b) = (make(MOBIUS), make(pv_params));
        let mut cycle = PvEoCycle::<Dirac, f32, Aosoa<W>>::in_blocks(
            op,
            pv,
            (op, pv, EoForm::Right),
            (&op_b, &pv_b),
            basis,
            PvEoParams::default(),
        )
        .unwrap();
        let mut a = cycle.operator();
        let mut like = Field::<f64, Spinor, Half, Aosoa<W>>::with_shape(lat, 24, MOBIUS.ls, 1, 1);
        like.set_parity(Parity::Even);
        let b_b = to_blocks_5d::<W>(b);
        let mut rhs = like.zeros_like();
        a.right_hand_side(&mut rhs, &b_b);
        let mut sol = like.zeros_like();
        let info = Gcr::new(stop, 2)
            .solve(&mut a, Some(&mut cycle), &mut sol, &rhs)
            .unwrap();
        let mut x = b_b.zeros_like();
        a.solution(&mut x, &sol, &b_b);
        (info.iterations, to_sites_5d(&x))
    }
    for (w, (steps, x_b)) in [
        (
            2,
            in_blocks::<2>(&lat, &u, (&op, &pv, pv_params), &basis, &b, stop),
        ),
        (
            4,
            in_blocks::<4>(&lat, &u, (&op, &pv, pv_params), &basis, &b, stop),
        ),
        (
            8,
            in_blocks::<8>(&lat, &u, (&op, &pv, pv_params), &basis, &b, stop),
        ),
    ] {
        let d = max_abs_diff(&x_b, &x);
        eprintln!(
            "W = {w}: {steps} steps in blocks, {} in the usual layout; solutions differ by {d:e}",
            info.iterations
        );
        assert!(steps.abs_diff(info.iterations) <= 1);
        assert!(d < 1e-8, "W = {w}: {d:e}");
    }
}
