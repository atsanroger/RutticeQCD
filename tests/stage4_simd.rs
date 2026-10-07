//! Checks: fields in blocks of W sites (`Aosoa<W>`, the SIMD layout) against the same
//! fields in the usual layout.

mod common;

use common::max_abs_diff;
use ruttice::{random_spinor, Aosoa, ConvertFrom, Field, Lattice, Spinor, Vector, C64};
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
