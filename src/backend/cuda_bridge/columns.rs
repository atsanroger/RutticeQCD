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

/// Counts a pass over `reals` reals of `T`: `flops` and `moved` (reals read and written) for
/// each of them (`meter`). The reals are those of the lattice: the padding that fills the last
/// block of 32 sites of a vector is not lattice work, and the CPU does not have it. Every
/// function here takes `lattice`, the reals of the lattice in one vector, for that.
fn count<T>(reals: usize, flops: f64, moved: f64) {
    crate::meter::vector(reals, std::mem::size_of::<T>(), flops, moved);
}

/// One vector of a set into one of another: (buffer, its vectors, the vector).
fn copy_vector<T: DeviceReal>(
    dst: (&DeviceBuffer<T>, usize, usize),
    src: (&DeviceBuffer<T>, usize, usize),
    lattice: usize,
) {
    let ((to, n), (from, m)) = (part(dst.0, dst.1, dst.2), part(src.0, src.1, src.2));
    assert_eq!(n, m, "vectors of another size");
    // SAFETY: two vectors of `n` reals inside their buffers, which are not the same one.
    count::<T>(lattice, 0.0, 2.0);
    must(check(unsafe {
        cudaMemcpy(
            to.cast(),
            from.cast_const().cast(),
            n * std::mem::size_of::<T>(),
            DEVICE_TO_DEVICE,
        )
    }));
}

/// `dst = src`, all the vectors.
pub(super) fn copy<T: DeviceReal>(
    dst: &mut DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    vectors: usize,
    lattice: usize,
) {
    count::<T>(lattice * vectors, 0.0, 2.0);
    must(dst.copy_from(src));
}

/// `x = 0`, all the vectors.
pub(super) fn zero<T: DeviceReal>(x: &mut DeviceBuffer<T>, vectors: usize, lattice: usize) {
    count::<T>(lattice * vectors, 0.0, 1.0);
    must(x.set_zero());
}

/// `out_k = src_(which[k])`.
pub(super) fn select<T: DeviceReal>(
    out: &DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    src_vectors: usize,
    which: &[usize],
    lattice: usize,
) {
    for (k, v) in which.iter().enumerate() {
        copy_vector((out, which.len(), k), (src, src_vectors, *v), lattice);
    }
}

/// `dst_(which[k]) = from_k`.
pub(super) fn assign<T: DeviceReal>(
    dst: &DeviceBuffer<T>,
    dst_vectors: usize,
    which: &[usize],
    from: &DeviceBuffer<T>,
    lattice: usize,
) {
    for (k, v) in which.iter().enumerate() {
        copy_vector((dst, dst_vectors, *v), (from, which.len(), k), lattice);
    }
}

pub(super) fn scale<T: DeviceReal>(x: &DeviceBuffer<T>, vectors: usize, a: &[f64], lattice: usize) {
    assert!(a.len() >= vectors);
    count::<T>(lattice * vectors, 1.0, 2.0);
    // SAFETY: `vectors` vectors of `len / vectors` reals and a coefficient for each.
    unsafe { T::scal(x.ptr, a.as_ptr(), (x.len / vectors) as i32, vectors as i32) };
}

/// `y_v += a_v x_v`.
pub(super) fn axpy<T: DeviceReal>(
    y: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    vectors: usize,
    a: &[C64],
    lattice: usize,
) {
    assert!(a.len() >= vectors && x.len == y.len);
    count::<T>(lattice * vectors, 4.0, 3.0);
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
    lattice: usize,
) {
    assert!(a.len() >= vectors && x.len == y.len);
    count::<T>(lattice * vectors, 4.0, 3.0);
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

/// Reals of one vector of a set; the vectors of `x` and `y` of the same size.
fn reals<T>(x: &DeviceBuffer<T>, y: &DeviceBuffer<T>, vectors: usize) -> i32 {
    assert!(vectors > 0 && x.len == y.len);
    (x.len / vectors) as i32
}

/// The complex numbers `a` as (re, im).
fn reals_of(a: &[C64], vectors: usize) -> *const f64 {
    assert!(a.len() >= vectors);
    a.as_ptr().cast()
}

/// `out_v = <x_v|y_v>`, all of them fetched from the device together.
pub(super) fn dot<T: DeviceReal>(
    x: &DeviceBuffer<T>,
    y: &DeviceBuffer<T>,
    vectors: usize,
    out: &mut [C64],
    lattice: usize,
) {
    assert!(out.len() >= vectors);
    let n = reals(x, y, vectors);
    let (mut re, mut im) = (vec![0.0; vectors], vec![0.0; vectors]);
    count::<T>(lattice * vectors, 4.0, 2.0);
    // SAFETY: two buffers of `vectors` vectors of `n` reals, and places for that many results.
    must(check(unsafe {
        T::dotc(
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            x.ptr,
            y.ptr,
            n,
            vectors as i32,
        )
    }));
    for (out, (re, im)) in out.iter_mut().zip(re.into_iter().zip(im)) {
        *out = C64::new(re, im);
    }
}

pub(super) fn norm_sqr<T: DeviceReal>(
    x: &DeviceBuffer<T>,
    vectors: usize,
    out: &mut [f64],
    lattice: usize,
) {
    assert!(out.len() >= vectors);
    let n = reals(x, x, vectors);
    count::<T>(lattice * vectors, 2.0, 1.0);
    // SAFETY: a buffer of `vectors` vectors, and places for that many results.
    must(check(unsafe {
        T::norm2(out.as_mut_ptr(), x.ptr, n, vectors as i32)
    }));
}

/// `y_v += a_v x_v`, then `out_v = |y_v|²`, in one pass.
pub(super) fn axpy_norm_sqr<T: DeviceReal>(
    y: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    vectors: usize,
    a: &[C64],
    out: &mut [f64],
    lattice: usize,
) {
    assert!(out.len() >= vectors);
    let n = reals(x, y, vectors);
    count::<T>(lattice * vectors, 6.0, 3.0);
    // SAFETY: two buffers of `vectors` vectors of `n` reals, not the same one (the caller
    // borrows `y` mutably), a coefficient and a place for the result of each.
    must(check(unsafe {
        T::axpy_norm2(
            y.ptr,
            x.ptr,
            reals_of(a, vectors),
            out.as_mut_ptr(),
            n,
            vectors as i32,
        )
    }));
}

/// `s_v = x_v + a_v y_v`, then `norm_v = |s_v|²` and `dot_v = <z_v|s_v>`, in one pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn xpay_norm_sqr_dot<T: DeviceReal>(
    s: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    a: &[C64],
    y: &DeviceBuffer<T>,
    z: &DeviceBuffer<T>,
    vectors: usize,
    norm: &mut [f64],
    dot: &mut [C64],
    lattice: usize,
) {
    assert!(norm.len() >= vectors && dot.len() >= vectors);
    let n = reals(s, x, vectors);
    assert!(y.len == s.len && z.len == s.len);
    let (mut re, mut im) = (vec![0.0; vectors], vec![0.0; vectors]);
    count::<T>(lattice * vectors, 10.0, 4.0);
    // SAFETY: four buffers of `vectors` vectors of `n` reals, `s` none of the others (the
    // caller borrows it mutably), and places for the results.
    must(check(unsafe {
        T::xpay_norm2_dot(
            s.ptr,
            x.ptr,
            reals_of(a, vectors),
            y.ptr,
            z.ptr,
            norm.as_mut_ptr(),
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            n,
            vectors as i32,
        )
    }));
    for (out, (re, im)) in dot.iter_mut().zip(re.into_iter().zip(im)) {
        *out = C64::new(re, im);
    }
}

/// `dot_v = <s_v|x_v>`, `norm_v = |s_v|²` and `norm_x_v = |x_v|²`, in one pass.
pub(super) fn dot_norms<T: DeviceReal>(
    s: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    vectors: usize,
    dot: &mut [C64],
    norm: &mut [f64],
    norm_x: &mut [f64],
    lattice: usize,
) {
    assert!(dot.len() >= vectors && norm.len() >= vectors && norm_x.len() >= vectors);
    let n = reals(s, x, vectors);
    let (mut re, mut im) = (vec![0.0; vectors], vec![0.0; vectors]);
    count::<T>(lattice * vectors, 8.0, 2.0);
    // SAFETY: two buffers of `vectors` vectors of `n` reals, and places for the results.
    must(check(unsafe {
        T::dot_norms(
            s.ptr,
            x.ptr,
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            norm.as_mut_ptr(),
            norm_x.as_mut_ptr(),
            n,
            vectors as i32,
        )
    }));
    for (out, (re, im)) in dot.iter_mut().zip(re.into_iter().zip(im)) {
        *out = C64::new(re, im);
    }
}

/// `<x_i|y_j>` for the `nx` vectors of `x` and the `ny` of `y` (of the same size), row-major.
pub(super) fn gram<T: DeviceReal>(
    x: &DeviceBuffer<T>,
    nx: usize,
    y: &DeviceBuffer<T>,
    ny: usize,
    lattice: usize,
) -> Vec<C64> {
    assert!(nx > 0 && ny > 0 && x.len / nx == y.len / ny);
    let n = (x.len / nx) as i32;
    let (mut re, mut im) = (vec![0.0; nx * ny], vec![0.0; nx * ny]);
    // Every pair of vectors: 4 flops a real; each vector read once.
    count::<T>(lattice * nx * ny, 4.0, 0.0);
    count::<T>(lattice * (nx + ny), 0.0, 1.0);
    // SAFETY: two sets of `nx` and `ny` vectors of `n` reals, and places for nx ny results.
    must(check(unsafe {
        T::gram(
            re.as_mut_ptr(),
            im.as_mut_ptr(),
            x.ptr,
            y.ptr,
            n,
            nx as i32,
            ny as i32,
        )
    }));
    re.into_iter()
        .zip(im)
        .map(|(re, im)| C64::new(re, im))
        .collect()
}

/// `s_v = c_v s_v + a_v x_v + b_v y_v`, then `norm_v = |s_v|²`, in one pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn scale_axpy2_norm_sqr<T: DeviceReal>(
    s: &DeviceBuffer<T>,
    c: &[f64],
    a: &[C64],
    x: &DeviceBuffer<T>,
    b: &[C64],
    y: &DeviceBuffer<T>,
    vectors: usize,
    norm: &mut [f64],
    lattice: usize,
) {
    assert!(c.len() >= vectors && norm.len() >= vectors);
    let n = reals(s, x, vectors);
    assert!(y.len == s.len);
    count::<T>(lattice * vectors, 11.0, 4.0);
    // SAFETY: three buffers of `vectors` vectors of `n` reals, `s` neither of the others (the
    // caller borrows it mutably), coefficients and a place for the result of each vector.
    must(check(unsafe {
        T::scale_axpy2_norm2(
            s.ptr,
            c.as_ptr(),
            reals_of(a, vectors),
            x.ptr,
            reals_of(b, vectors),
            y.ptr,
            norm.as_mut_ptr(),
            n,
            vectors as i32,
        )
    }));
}

/// `s_v = x_v + a_v y_v`.
pub(super) fn xpay<T: DeviceReal>(
    s: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    a: &[C64],
    y: &DeviceBuffer<T>,
    vectors: usize,
    lattice: usize,
) {
    let n = reals(s, x, vectors);
    assert!(y.len == s.len);
    count::<T>(lattice * vectors, 4.0, 3.0);
    // SAFETY: three buffers of `vectors` vectors of `n` reals, `s` neither of the others, and
    // a coefficient for each vector.
    unsafe { T::xpay(s.ptr, x.ptr, reals_of(a, vectors), y.ptr, n, vectors as i32) };
}

/// `s_v += a_v x_v + b_v y_v`.
pub(super) fn axpy2<T: DeviceReal>(
    s: &DeviceBuffer<T>,
    a: &[C64],
    x: &DeviceBuffer<T>,
    b: &[C64],
    y: &DeviceBuffer<T>,
    vectors: usize,
    lattice: usize,
) {
    let n = reals(s, x, vectors);
    assert!(y.len == s.len);
    count::<T>(lattice * vectors, 8.0, 4.0);
    // SAFETY: as in `xpay`.
    unsafe {
        T::axpy2(
            s.ptr,
            reals_of(a, vectors),
            x.ptr,
            reals_of(b, vectors),
            y.ptr,
            n,
            vectors as i32,
        )
    };
}

/// `s_v = y_v + b_v (s_v + a_v x_v)`.
pub(super) fn axpy_aypx<T: DeviceReal>(
    s: &DeviceBuffer<T>,
    a: &[C64],
    x: &DeviceBuffer<T>,
    b: &[C64],
    y: &DeviceBuffer<T>,
    vectors: usize,
    lattice: usize,
) {
    let n = reals(s, x, vectors);
    assert!(y.len == s.len);
    count::<T>(lattice * vectors, 8.0, 4.0);
    // SAFETY: as in `xpay`.
    unsafe {
        T::axpy_aypx(
            s.ptr,
            reals_of(a, vectors),
            x.ptr,
            reals_of(b, vectors),
            y.ptr,
            n,
            vectors as i32,
        )
    };
}

/// `out_v += d_v`, `r_v −= ad_v`, `d_v = c_v d_v + a_v r_v`.
#[allow(clippy::too_many_arguments)]
pub(super) fn chebyshev_step<T: DeviceReal>(
    out: &DeviceBuffer<T>,
    d: &DeviceBuffer<T>,
    r: &DeviceBuffer<T>,
    ad: &DeviceBuffer<T>,
    c: &[f64],
    a: &[C64],
    vectors: usize,
    lattice: usize,
) {
    let n = reals(out, d, vectors);
    assert!(r.len == out.len && ad.len == out.len && c.len() >= vectors);
    count::<T>(lattice * vectors, 7.0, 7.0);
    // SAFETY: four distinct buffers of `vectors` vectors of `n` reals, and coefficients for
    // each vector.
    unsafe {
        T::chebyshev_step(
            out.ptr,
            d.ptr,
            r.ptr,
            ad.ptr,
            c.as_ptr(),
            reals_of(a, vectors),
            n,
            vectors as i32,
        )
    };
}

/// The most vectors `rotate` takes.
pub(super) const ROTATE_VECTORS: usize = 32;

/// `out_j = Σ_i x_i c[i n + j]` for the n vectors of `x` into the n of `out`, any number of
/// them: for a rotation of more vectors than `rotate` takes, with one more set and no others.
pub(super) fn combine<T: DeviceReal>(
    out: &DeviceBuffer<T>,
    x: &DeviceBuffer<T>,
    vectors: usize,
    c: &[C64],
    lattice: usize,
) {
    assert!(c.len() == vectors * vectors && out.len == x.len);
    let reals: Vec<f64> = c.iter().flat_map(|z| [z.re, z.im]).collect();
    let c = must(DeviceBuffer::<f64>::from_host(&reals));
    let (n, v) = ((x.len / vectors) as i32, vectors as i32);
    count::<T>(lattice * vectors, 4.0 * vectors as f64, 2.0);
    // SAFETY: two sets of `vectors` vectors of `n` reals, not the same buffer, and the
    // coefficients of that many.
    unsafe { T::combine(out.ptr, x.ptr, c.ptr, n, v, v) };
}

/// `x_j ← Σ_i x_i c[i n + j]` for the n vectors of the set, in place.
pub(super) fn rotate<T: DeviceReal>(
    x: &DeviceBuffer<T>,
    vectors: usize,
    c: &[C64],
    lattice: usize,
) {
    assert!(vectors <= ROTATE_VECTORS && c.len() == vectors * vectors);
    let reals: Vec<f64> = c.iter().flat_map(|z| [z.re, z.im]).collect();
    let c = must(DeviceBuffer::<f64>::from_host(&reals));
    count::<T>(lattice * vectors, 4.0 * vectors as f64, 2.0);
    // SAFETY: `vectors` vectors of `len / vectors` reals, and the coefficients of that many.
    unsafe { T::rotate(x.ptr, c.ptr, (x.len / vectors) as i32, vectors as i32) };
}
