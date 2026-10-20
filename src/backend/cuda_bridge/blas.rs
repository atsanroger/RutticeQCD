//! The reductions and fused operations of `cuda_bridge/src/linalg/blas.cu` on sets of vectors:
//! one pass over the fields each, the sums finished on the device in a fixed order, and only
//! the results brought to the host (one wait for the stream). The reductions return the code
//! of the CUDA runtime.

use super::*;

/// The operations of `blas.cu` in one precision. Pointers: `vectors` vectors of `n` reals one
/// after the other on the device; coefficients (re, im) and results on the host, one for each
/// vector.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub trait DeviceBlas: Sized {
    unsafe fn norm2(out: *mut f64, x: *const Self, n: i32, vectors: i32) -> i32;
    unsafe fn dotc(
        re: *mut f64,
        im: *mut f64,
        x: *const Self,
        y: *const Self,
        n: i32,
        vectors: i32,
    ) -> i32;
    /// `y += a x`, then `|y|²`.
    unsafe fn axpy_norm2(
        y: *mut Self,
        x: *const Self,
        a: *const f64,
        norm: *mut f64,
        n: i32,
        vectors: i32,
    ) -> i32;
    /// `s = x + a y`, then `|s|²` and `<z|s>`.
    unsafe fn xpay_norm2_dot(
        s: *mut Self,
        x: *const Self,
        a: *const f64,
        y: *const Self,
        z: *const Self,
        norm: *mut f64,
        re: *mut f64,
        im: *mut f64,
        n: i32,
        vectors: i32,
    ) -> i32;
    /// `<s|x>`, `|s|²` and `|x|²`.
    unsafe fn dot_norms(
        s: *const Self,
        x: *const Self,
        re: *mut f64,
        im: *mut f64,
        norm_s: *mut f64,
        norm_x: *mut f64,
        n: i32,
        vectors: i32,
    ) -> i32;
    /// The coarse unit vectors of the components `first ..`, `vectors` of them, into a zero
    /// coarse field (`multigrid/fold.cuh`).
    unsafe fn coarse_units(v: *mut Self, n_coarse: i32, dim: i32, first: i32, vectors: i32);
    /// Column `first + v` of every coarse site's matrix (`f32` with `target_single`, else `f64`)
    /// from vector `v` of the coarse field `c`.
    unsafe fn coarse_columns(
        target: *mut c_void,
        target_single: i32,
        c: *const Self,
        n_coarse: i32,
        dim: i32,
        first: i32,
        vectors: i32,
    );
    /// `<x_i|y_j>` for the `nx` vectors of `x` and the `ny` of `y`, at `i ny + j`.
    unsafe fn gram(
        re: *mut f64,
        im: *mut f64,
        x: *const Self,
        y: *const Self,
        n: i32,
        nx: i32,
        ny: i32,
    ) -> i32;
    /// `s = c s + a x + b y` (`c` real), then `|s|²`.
    unsafe fn scale_axpy2_norm2(
        s: *mut Self,
        c: *const f64,
        a: *const f64,
        x: *const Self,
        b: *const f64,
        y: *const Self,
        norm: *mut f64,
        n: i32,
        vectors: i32,
    ) -> i32;
    /// `out_j = Σ_i x_i c[i n_out + j]`, `x` and `out` sets of `n_in` and `n_out` vectors; `c`
    /// on the device.
    unsafe fn combine(out: *mut Self, x: *const Self, c: *const f64, n: i32, n_in: i32, n_out: i32);
    /// `s = x + a y`.
    unsafe fn xpay(
        s: *mut Self,
        x: *const Self,
        a: *const f64,
        y: *const Self,
        n: i32,
        vectors: i32,
    );
    /// `s += a x + b y`.
    unsafe fn axpy2(
        s: *mut Self,
        a: *const f64,
        x: *const Self,
        b: *const f64,
        y: *const Self,
        n: i32,
        vectors: i32,
    );
    /// `s = y + b (s + a x)`.
    unsafe fn axpy_aypx(
        s: *mut Self,
        a: *const f64,
        x: *const Self,
        b: *const f64,
        y: *const Self,
        n: i32,
        vectors: i32,
    );
    /// `out += d`, `r −= ad`, `d = c d + a r` (c real): a step of Chebyshev iteration.
    #[allow(clippy::too_many_arguments)]
    unsafe fn chebyshev_step(
        out: *mut Self,
        d: *mut Self,
        r: *mut Self,
        ad: *const Self,
        c: *const f64,
        a: *const f64,
        n: i32,
        vectors: i32,
    );
}

/// `DeviceBlas` for one type, on the functions of that precision.
macro_rules! device_blas {
    ($t:ty: $norm2:ident, $dotc:ident, $axpy_norm2:ident, $xpay_norm2_dot:ident, $dot_norms:ident,
     $gram:ident, $scale_axpy2_norm2:ident, $xpay:ident, $axpy2:ident, $axpy_aypx:ident,
     $chebyshev_step:ident, $combine:ident, $units:ident, $columns:ident) => {
        extern "C" {
            fn $chebyshev_step(
                out: *mut $t,
                d: *mut $t,
                r: *mut $t,
                ad: *const $t,
                c: *const f64,
                a: *const f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $units(
                v: *mut $t,
                n_coarse: i32,
                dim: i32,
                first: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $columns(
                target: *mut c_void,
                target_single: i32,
                c: *const $t,
                n_coarse: i32,
                dim: i32,
                first: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $gram(
                re: *mut f64,
                im: *mut f64,
                x: *const $t,
                y: *const $t,
                n: i32,
                nx: i32,
                ny: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $scale_axpy2_norm2(
                s: *mut $t,
                c: *const f64,
                a: *const f64,
                x: *const $t,
                b: *const f64,
                y: *const $t,
                norm: *mut f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $norm2(
                out: *mut f64,
                x: *const $t,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $dotc(
                re: *mut f64,
                im: *mut f64,
                x: *const $t,
                y: *const $t,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $axpy_norm2(
                y: *mut $t,
                x: *const $t,
                a: *const f64,
                norm: *mut f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $xpay_norm2_dot(
                s: *mut $t,
                x: *const $t,
                a: *const f64,
                y: *const $t,
                z: *const $t,
                norm: *mut f64,
                re: *mut f64,
                im: *mut f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $dot_norms(
                s: *const $t,
                x: *const $t,
                re: *mut f64,
                im: *mut f64,
                norm_s: *mut f64,
                norm_x: *mut f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $combine(
                out: *mut $t,
                x: *const $t,
                c: *const f64,
                n: i32,
                n_in: i32,
                n_out: i32,
                stream: *mut c_void,
            );
            fn $xpay(
                s: *mut $t,
                x: *const $t,
                a: *const f64,
                y: *const $t,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $axpy2(
                s: *mut $t,
                a: *const f64,
                x: *const $t,
                b: *const f64,
                y: *const $t,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $axpy_aypx(
                s: *mut $t,
                a: *const f64,
                x: *const $t,
                b: *const f64,
                y: *const $t,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
        }

        impl DeviceBlas for $t {
            unsafe fn norm2(out: *mut f64, x: *const Self, n: i32, vectors: i32) -> i32 {
                $norm2(out, x, n, vectors, STREAM)
            }
            unsafe fn dotc(
                re: *mut f64,
                im: *mut f64,
                x: *const Self,
                y: *const Self,
                n: i32,
                vectors: i32,
            ) -> i32 {
                $dotc(re, im, x, y, n, vectors, STREAM)
            }
            unsafe fn axpy_norm2(
                y: *mut Self,
                x: *const Self,
                a: *const f64,
                norm: *mut f64,
                n: i32,
                vectors: i32,
            ) -> i32 {
                $axpy_norm2(y, x, a, norm, n, vectors, STREAM)
            }
            unsafe fn xpay_norm2_dot(
                s: *mut Self,
                x: *const Self,
                a: *const f64,
                y: *const Self,
                z: *const Self,
                norm: *mut f64,
                re: *mut f64,
                im: *mut f64,
                n: i32,
                vectors: i32,
            ) -> i32 {
                $xpay_norm2_dot(s, x, a, y, z, norm, re, im, n, vectors, STREAM)
            }
            unsafe fn dot_norms(
                s: *const Self,
                x: *const Self,
                re: *mut f64,
                im: *mut f64,
                norm_s: *mut f64,
                norm_x: *mut f64,
                n: i32,
                vectors: i32,
            ) -> i32 {
                $dot_norms(s, x, re, im, norm_s, norm_x, n, vectors, STREAM)
            }
            unsafe fn coarse_units(
                v: *mut Self,
                n_coarse: i32,
                dim: i32,
                first: i32,
                vectors: i32,
            ) {
                $units(v, n_coarse, dim, first, vectors, STREAM)
            }
            unsafe fn coarse_columns(
                target: *mut c_void,
                target_single: i32,
                c: *const Self,
                n_coarse: i32,
                dim: i32,
                first: i32,
                vectors: i32,
            ) {
                $columns(
                    target,
                    target_single,
                    c,
                    n_coarse,
                    dim,
                    first,
                    vectors,
                    STREAM,
                )
            }
            unsafe fn gram(
                re: *mut f64,
                im: *mut f64,
                x: *const Self,
                y: *const Self,
                n: i32,
                nx: i32,
                ny: i32,
            ) -> i32 {
                $gram(re, im, x, y, n, nx, ny, STREAM)
            }
            unsafe fn scale_axpy2_norm2(
                s: *mut Self,
                c: *const f64,
                a: *const f64,
                x: *const Self,
                b: *const f64,
                y: *const Self,
                norm: *mut f64,
                n: i32,
                vectors: i32,
            ) -> i32 {
                $scale_axpy2_norm2(s, c, a, x, b, y, norm, n, vectors, STREAM)
            }
            unsafe fn combine(
                out: *mut Self,
                x: *const Self,
                c: *const f64,
                n: i32,
                n_in: i32,
                n_out: i32,
            ) {
                $combine(out, x, c, n, n_in, n_out, STREAM)
            }
            unsafe fn xpay(
                s: *mut Self,
                x: *const Self,
                a: *const f64,
                y: *const Self,
                n: i32,
                vectors: i32,
            ) {
                $xpay(s, x, a, y, n, vectors, STREAM)
            }
            unsafe fn axpy2(
                s: *mut Self,
                a: *const f64,
                x: *const Self,
                b: *const f64,
                y: *const Self,
                n: i32,
                vectors: i32,
            ) {
                $axpy2(s, a, x, b, y, n, vectors, STREAM)
            }
            unsafe fn axpy_aypx(
                s: *mut Self,
                a: *const f64,
                x: *const Self,
                b: *const f64,
                y: *const Self,
                n: i32,
                vectors: i32,
            ) {
                $axpy_aypx(s, a, x, b, y, n, vectors, STREAM)
            }
            unsafe fn chebyshev_step(
                out: *mut Self,
                d: *mut Self,
                r: *mut Self,
                ad: *const Self,
                c: *const f64,
                a: *const f64,
                n: i32,
                vectors: i32,
            ) {
                $chebyshev_step(out, d, r, ad, c, a, n, vectors, STREAM)
            }
        }
    };
}

device_blas!(f32: bridge_cuda_blas_norm2_f32, bridge_cuda_blas_dotc_f32,
    bridge_cuda_blas_axpy_norm2_f32, bridge_cuda_blas_xpay_norm2_dot_f32,
    bridge_cuda_blas_dot_norms_f32, bridge_cuda_blas_gram_f32, bridge_cuda_blas_scale_axpy2_norm2_f32,
    bridge_cuda_blas_xpay_f32, bridge_cuda_blas_axpy2_f32,
    bridge_cuda_blas_axpy_aypx_f32, bridge_cuda_blas_chebyshev_step_f32, bridge_cuda_blas_combine_f32,
    bridge_cuda_coarse_units_f32, bridge_cuda_coarse_columns_f32);
device_blas!(f64: bridge_cuda_blas_norm2_f64, bridge_cuda_blas_dotc_f64,
    bridge_cuda_blas_axpy_norm2_f64, bridge_cuda_blas_xpay_norm2_dot_f64,
    bridge_cuda_blas_dot_norms_f64, bridge_cuda_blas_gram_f64, bridge_cuda_blas_scale_axpy2_norm2_f64,
    bridge_cuda_blas_xpay_f64, bridge_cuda_blas_axpy2_f64,
    bridge_cuda_blas_axpy_aypx_f64, bridge_cuda_blas_chebyshev_step_f64, bridge_cuda_blas_combine_f64,
    bridge_cuda_coarse_units_f64, bridge_cuda_coarse_columns_f64);
