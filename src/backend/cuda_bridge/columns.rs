//! Sets of vectors on the device, one after the other in a buffer: the linear algebra of
//! `Vector` for the fields that are such sets (`DeviceSpinor`, `DeviceCoarse`). A vector is
//! complex numbers in blocks of 32 real parts and 32 imaginary parts.

use super::*;
use crate::C64;

/// Where vector `v` of `vectors` begins, and its reals.
fn part<T>(buffer: &DeviceBuffer<T>, vectors: usize, v: usize) -> (*mut T, usize) {
    assert!(v < vectors);
    let one = buffer.len / vectors;
    // SAFETY: vector `v` is inside the buffer.
    (unsafe { buffer.ptr.add(v * one) }, one)
}

/// One vector of a set into one of another: (buffer, its vectors, the vector).
fn copy_vector<T: DeviceReal>(
    dst: (&DeviceBuffer<T>, usize, usize),
    src: (&DeviceBuffer<T>, usize, usize),
) {
    let ((to, n), (from, m)) = (part(dst.0, dst.1, dst.2), part(src.0, src.1, src.2));
    assert_eq!(n, m, "vectors of another size");
    // SAFETY: two vectors of `n` reals inside their buffers, which are not the same one.
    must(check(unsafe {
        cudaMemcpy(
            to.cast(),
            from.cast_const().cast(),
            n * std::mem::size_of::<T>(),
            DEVICE_TO_DEVICE,
        )
    }));
}

/// `out_k = src_(which[k])`.
pub(super) fn select<T: DeviceReal>(
    out: &DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    src_vectors: usize,
    which: &[usize],
) {
    for (k, v) in which.iter().enumerate() {
        copy_vector((out, which.len(), k), (src, src_vectors, *v));
    }
}

/// `dst_(which[k]) = from_k`.
pub(super) fn assign<T: DeviceReal>(
    dst: &DeviceBuffer<T>,
    dst_vectors: usize,
    which: &[usize],
    from: &DeviceBuffer<T>,
) {
    for (k, v) in which.iter().enumerate() {
        copy_vector((dst, dst_vectors, *v), (from, which.len(), k));
    }
}

pub(super) fn scale<T: DeviceReal>(x: &DeviceBuffer<T>, vectors: usize, a: &[f64]) {
    assert!(a.len() >= vectors);
    // SAFETY: `vectors` vectors of `len / vectors` reals and a coefficient for each.
    unsafe { T::scal(x.ptr, a.as_ptr(), (x.len / vectors) as i32, vectors as i32) };
}

/// `y_v += a_v x_v`.
pub(super) fn axpy<T: DeviceReal>(
    y: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    vectors: usize,
    a: &[C64],
) {
    assert!(a.len() >= vectors && x.len == y.len);
    // SAFETY: two buffers of `vectors` vectors of the same size, not the same one (the caller
    // borrows `y` mutably); a complex number is its real and its imaginary part.
    unsafe {
        T::caxpy(
            y.ptr,
            x.ptr,
            a.as_ptr().cast(),
            (y.len / vectors) as i32,
            vectors as i32,
        )
    };
}

/// `y_v = a_v y_v + x_v`.
pub(super) fn aypx<T: DeviceReal>(
    y: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    vectors: usize,
    a: &[C64],
) {
    assert!(a.len() >= vectors && x.len == y.len);
    // SAFETY: as in `axpy`.
    unsafe {
        T::caypx(
            y.ptr,
            x.ptr,
            a.as_ptr().cast(),
            (y.len / vectors) as i32,
            vectors as i32,
        )
    };
}

/// `out_v = <x_v|y_v>`, all of them fetched from the device together.
pub(super) fn dot<T: DeviceReal>(
    x: &DeviceBuffer<T>,
    y: &DeviceBuffer<T>,
    vectors: usize,
    out: &mut [C64],
) {
    assert!(out.len() >= vectors && x.len == y.len);
    let (mut re, mut im) = (vec![0.0; vectors], vec![0.0; vectors]);
    // SAFETY: two buffers of `vectors` vectors of the same size, and places for that many
    // results.
    unsafe {
        T::dotc(
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            x.ptr,
            y.ptr,
            (x.len / vectors) as i32,
            vectors as i32,
        );
    }
    finish();
    for (out, (re, im)) in out.iter_mut().zip(re.into_iter().zip(im)) {
        *out = C64::new(re, im);
    }
}

pub(super) fn norm_sqr<T: DeviceReal>(x: &DeviceBuffer<T>, vectors: usize, out: &mut [f64]) {
    assert!(out.len() >= vectors);
    // SAFETY: a buffer of `vectors` vectors, and places for that many results.
    unsafe {
        T::norm2(
            out.as_mut_ptr(),
            x.ptr,
            (x.len / vectors) as i32,
            vectors as i32,
        );
    }
    finish();
}

/// The most vectors `rotate` takes.
pub(super) const ROTATE_VECTORS: usize = 32;

/// `x_j ← Σ_i x_i c[i n + j]` for the n vectors of the set, in place.
pub(super) fn rotate<T: DeviceReal>(x: &DeviceBuffer<T>, vectors: usize, c: &[C64]) {
    assert!(vectors <= ROTATE_VECTORS && c.len() == vectors * vectors);
    let reals: Vec<f64> = c.iter().flat_map(|z| [z.re, z.im]).collect();
    let c = must(DeviceBuffer::<f64>::from_host(&reals));
    // SAFETY: `vectors` vectors of `len / vectors` reals, and the coefficients of that many.
    unsafe { T::rotate(x.ptr, c.ptr, (x.len / vectors) as i32, vectors as i32) };
}
