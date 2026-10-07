//! Checks: fields in blocks of W sites (`Aosoa<W>`, the SIMD layout) against the same
//! fields in the usual layout.

mod common;

use common::max_abs_diff;
use ruttice::fermion::{
    solve_even_odd, Bound, Clover, CloverParams, CloverSimd, Dagger, DiagTerm, EvenOdd, FermionOp,
    Variant, Wilson, WilsonOp, WilsonParams, WilsonSimd,
};
use ruttice::gauge::random_gauge;
use ruttice::kernel::{Dirac, LaneOf};
use ruttice::solver::{solve_cgne, BiCgStab, Cg, Solver, Stop};
use ruttice::{
    extract_parity, insert_parity, random_spinor, Aosoa, Boundary, ConvertFrom, Field, Gauge, Half,
    Lattice, Parity, Spinor, Storage, Vector, C64, ND,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;
type B<const W: usize> = Field<f64, Spinor, ruttice::Full, Aosoa<W>>;

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
    let mut single = Field::<f32, Spinor, ruttice::Full, Aosoa<8>>::with_shape(&lat, 24, 1, 1, 1);
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
            op.mult(&mut (), &mut out, &inp, variant);
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
        let mut inp = Field::<f32, Spinor, ruttice::Full, Aosoa<W>>::with_shape(lat, 24, 1, 1, 1);
        inp.convert_from(psi);
        let mut out = inp.zeros_like();
        op.mult(&mut (), &mut out, &inp, Variant::Dagger);
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
            op.mult(&mut (), &mut out, &inp, variant);
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
                        blocks.hop(&mut (), &mut out_b, &inp_b, to, dagger);
                    }
                    1 => {
                        sites.diag(&mut work, &mut out_s, &same_s, to, dagger);
                        blocks.diag(&mut (), &mut out_b, &same_b, to, dagger);
                    }
                    _ => {
                        sites.diag_inv(&mut work, &mut out_s, &same_s, to, dagger);
                        blocks.diag_inv(&mut (), &mut out_b, &same_b, to, dagger);
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
