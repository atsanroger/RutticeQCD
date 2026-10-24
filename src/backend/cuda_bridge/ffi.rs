//! The functions of the kernels (`cuda_bridge/include/bridge_cuda.h`) and of the CUDA runtime,
//! the structures they take, and `DeviceReal`, which gives the kernels of one precision to the
//! generic code of the other modules.

use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Geometry {
    pub(super) dims: [i32; 4],
    pub(super) ls: i32,
    pub(super) bc: [i32; 4],
    pub(super) vol: i32,
    pub(super) vol5d: i32,
    pub(super) vol_pad: i32,
    pub(super) gamma_rep: i32,
    /// Rows of a link the operators read (`DeviceGauge`).
    pub(super) link_rows: i32,
}

#[repr(C)]
pub struct MdwfKernelParams {
    pub(super) m5: f64,
    pub(super) mass: f64,
    pub(super) alpha: f64,
    pub(super) b5: f64,
    pub(super) c5: f64,
}

/// Largest fifth dimension: the matrix of the fifth dimension must fit in the shared memory
/// of a block of threads (`fifth_gemm.cuh`).
pub(super) const MAX_LS: usize = 32;

pub(super) const HOST_TO_DEVICE: i32 = 1;
pub(super) const DEVICE_TO_HOST: i32 = 2;
pub(super) const DEVICE_TO_DEVICE: i32 = 3;

/// The default stream.
pub(super) const STREAM: *mut c_void = std::ptr::null_mut();

/// `cudaDevAttrMemoryPoolsSupported` and `cudaMemPoolAttrReleaseThreshold` of the runtime.
pub(super) const MEMORY_POOLS_SUPPORTED: i32 = 115;
pub(super) const POOL_RELEASE_THRESHOLD: i32 = 4;

extern "C" {
    pub(super) fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    pub(super) fn cudaFree(ptr: *mut c_void) -> i32;
    pub(super) fn cudaMallocAsync(ptr: *mut *mut c_void, bytes: usize, stream: *mut c_void) -> i32;
    pub(super) fn cudaFreeAsync(ptr: *mut c_void, stream: *mut c_void) -> i32;
    pub(super) fn cudaGetDevice(device: *mut i32) -> i32;
    pub(super) fn cudaSetDevice(device: i32) -> i32;
    pub(super) fn cudaGetDeviceCount(count: *mut i32) -> i32;
    #[cfg(feature = "nccl")]
    pub(super) fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    #[cfg(feature = "nccl")]
    pub(super) fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    #[cfg(feature = "nccl")]
    pub(super) fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    #[cfg(feature = "nccl")]
    pub(super) fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    pub(super) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    pub(super) fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
    pub(super) fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> i32;
    pub(super) fn cudaMemPoolSetAttribute(
        pool: *mut c_void,
        attribute: i32,
        value: *mut c_void,
    ) -> i32;
    pub(super) fn cudaMemPoolTrimTo(pool: *mut c_void, keep: usize) -> i32;
    pub(super) fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    pub(super) fn cudaMemset(ptr: *mut c_void, value: i32, bytes: usize) -> i32;
    pub(super) fn cudaDeviceSynchronize() -> i32;
    pub(super) fn cudaGetLastError() -> i32;
    pub(super) fn bridge_cuda_cast_f64_to_f32(
        dst: *mut f32,
        src: *const f64,
        n: i32,
        stream: *mut c_void,
    );
    pub(super) fn bridge_cuda_cast_f32_to_f64(
        dst: *mut f64,
        src: *const f32,
        n: i32,
        stream: *mut c_void,
    );
    pub(super) fn bridge_cuda_blocks_inverse(
        out: *mut f64,
        a: *const f64,
        scratch: *mut f64,
        fail: *mut u32,
        n_sites: i32,
        dim: i32,
        stream: *mut c_void,
    );
    pub(super) fn bridge_cuda_blocks_abs_inverse_gamma5(
        out: *mut f64,
        a: *const f64,
        scratch: *mut f64,
        n_sites: i32,
        dim: i32,
        stream: *mut c_void,
    );
    pub(super) fn bridge_cuda_fold_orthonormalise(
        p: *mut c_void,
        half: i32,
        w: *mut f64,
        sites: *const u32,
        fail: *mut u32,
        first_site: i32,
        count: i32,
        block_volume: i32,
        n: i32,
        kept: i32,
        scale: f64,
        stream: *mut c_void,
    );
    pub(super) fn bridge_cuda_fold_orthonormalise_cholesky(
        p: *mut c_void,
        half: i32,
        w: *mut f64,
        sites: *const u32,
        fail: *mut u32,
        first_site: i32,
        count: i32,
        block_volume: i32,
        n: i32,
        kept: i32,
        scale: f64,
        stream: *mut c_void,
    );
}

/// A real of the device, `f32` or `f64`: the kernels in that precision. The methods are the
/// functions of `cuda_bridge/include/bridge_cuda.h`; the pointers are device memory of the
/// sizes those read and write.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub trait DeviceReal:
    super::blas::DeviceBlas
    + super::halo::SplitReal
    + Copy
    + Default
    + PartialEq
    + std::fmt::Debug
    + 'static
{
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;

    unsafe fn clover_term(
        out: *mut Self,
        inp: *const Self,
        clover: *const Self,
        geometry: *const Geometry,
        parity: i32,
        vectors: i32,
    );
    unsafe fn mdwf_dslash(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        params: *const MdwfKernelParams,
        vectors: i32,
    );
    unsafe fn hop(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        from: *const Self,
        clover: *const Self,
        geometry: *const Geometry,
        kappa: f64,
        mass: f64,
        b5: f64,
        c5: f64,
        dagger: i32,
        target_parity: i32,
        vectors: i32,
    );
    unsafe fn mdwf_fifth(
        out: *mut Self,
        x: *const Self,
        a1: f64,
        y1: f64,
        z: *const Self,
        a2: f64,
        y2: f64,
        sites: i32,
        ls: i32,
        mass: f64,
        dagger: i32,
    );
    /// Whether it was done (see `bridge_cuda_hop_gemm`).
    unsafe fn hop_gemm(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        from: *const Self,
        matrix: *const Self,
        geometry: *const Geometry,
        mass: f64,
        b5: f64,
        c5: f64,
        dagger: i32,
        target_parity: i32,
        vectors: i32,
    ) -> bool;
    unsafe fn fifth_gemm(
        out: *mut Self,
        inp: *const Self,
        from: *const Self,
        matrix: *const Self,
        sites: i32,
        ls: i32,
    );
    unsafe fn gamma5(out: *mut Self, inp: *const Self, sites: i32, ls: i32, reflect: i32);
    unsafe fn parity_copy(
        half: *mut Self,
        full: *mut Self,
        geometry: *const Geometry,
        parity: i32,
        to_half: i32,
        vectors: i32,
    );
    unsafe fn walls(
        out: *mut Self,
        inp: *const Self,
        sites: i32,
        ls: i32,
        s_minus: i32,
        s_plus: i32,
        to_4d: i32,
        vectors: i32,
    );
    unsafe fn fold_restrict(
        coarse: *mut Self,
        fine: *const Self,
        p: *const c_void,
        sites: *const u32,
        n_coarse: i32,
        block_volume: i32,
        n: i32,
        dims: *const i32,
        ls: i32,
        parity: i32,
        gamma: *const i32,
        leg: i32,
        block: *const i32,
        vectors: i32,
    );
    unsafe fn fold_prolong(
        fine: *mut Self,
        coarse: *const Self,
        p: *const c_void,
        coarse_of: *const u32,
        n_coarse: i32,
        n: i32,
        dims: *const i32,
        ls: i32,
        parity: i32,
        gamma: *const i32,
        vectors: i32,
    );
    unsafe fn coarse_pass(
        out: *mut Self,
        base: *const Self,
        diag_of: *const Self,
        hops_of: *const Self,
        diag: *const Self,
        links: *const Self,
        inverse: *const Self,
        neighbours: *const u32,
        odd: *const u32,
        n_coarse: i32,
        dim: i32,
        parity: i32,
        minus: i32,
        gamma5: i32,
        vectors: i32,
    );
    unsafe fn coarse_gamma5(v: *mut Self, n_coarse: i32, dim: i32, vectors: i32);
    unsafe fn hop_split(
        cross: *mut Self,
        inside: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        kappa: f64,
        mass: f64,
        b5: f64,
        c5: f64,
        leg: i32,
        block: *const i32,
        vectors: i32,
    );
    unsafe fn fold_fill(
        p: *mut c_void,
        fine: *const Self,
        n: i32,
        first: i32,
        dims: *const i32,
        ls: i32,
        gamma: *const i32,
        vectors: i32,
    );
    unsafe fn rotate(x: *mut Self, c: *const f64, n: i32, vectors: i32);
    /// `a`: a real for each vector; in `caxpy` and `caypx` its real and its imaginary part.
    unsafe fn scal(x: *mut Self, a: *const f64, n: i32, vectors: i32);
    unsafe fn caxpy(y: *mut Self, x: *const Self, a: *const f64, n: i32, vectors: i32);
    unsafe fn caypx(y: *mut Self, x: *const Self, a: *const f64, n: i32, vectors: i32);
}

/// `DeviceReal` for one type, on the functions of that precision.
macro_rules! device_real {
    ($t:ty: $clover_term:ident, $mdwf:ident, $hop:ident,
     $fifth:ident, $gemm:ident, $hop_gemm:ident, $gamma5:ident, $parity_copy:ident, $scal:ident,
     $caxpy:ident, $caypx:ident, $walls:ident, $restrict:ident, $prolong:ident, $pass:ident,
     $coarse_gamma5:ident, $hop_split:ident, $fill:ident, $rotate:ident) => {
        extern "C" {
            fn $clover_term(
                out: *mut $t,
                inp: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                parity: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $mdwf(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                params: *const MdwfKernelParams,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $hop(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $fifth(
                out: *mut $t,
                x: *const $t,
                a1: f64,
                y1: f64,
                z: *const $t,
                a2: f64,
                y2: f64,
                sites: i32,
                ls: i32,
                mass: f64,
                dagger: i32,
                stream: *mut c_void,
            );
            fn $hop_gemm(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                matrix: *const $t,
                geometry: *const Geometry,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
                vectors: i32,
                stream: *mut c_void,
            ) -> i32;
            fn $gemm(
                out: *mut $t,
                inp: *const $t,
                from: *const $t,
                matrix: *const $t,
                sites: i32,
                ls: i32,
                stream: *mut c_void,
            );
            fn $gamma5(
                out: *mut $t,
                inp: *const $t,
                sites: i32,
                ls: i32,
                reflect: i32,
                stream: *mut c_void,
            );
            fn $parity_copy(
                half: *mut $t,
                full: *mut $t,
                geometry: *const Geometry,
                parity: i32,
                to_half: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $walls(
                out: *mut $t,
                inp: *const $t,
                sites: i32,
                ls: i32,
                s_minus: i32,
                s_plus: i32,
                to_4d: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $restrict(
                coarse: *mut $t,
                fine: *const $t,
                p: *const c_void,
                sites: *const u32,
                n_coarse: i32,
                block_volume: i32,
                n: i32,
                dims: *const i32,
                ls: i32,
                parity: i32,
                gamma: *const i32,
                leg: i32,
                block: *const i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $prolong(
                fine: *mut $t,
                coarse: *const $t,
                p: *const c_void,
                coarse_of: *const u32,
                n_coarse: i32,
                n: i32,
                dims: *const i32,
                ls: i32,
                parity: i32,
                gamma: *const i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $pass(
                out: *mut $t,
                base: *const $t,
                diag_of: *const $t,
                hops_of: *const $t,
                diag: *const $t,
                links: *const $t,
                inverse: *const $t,
                neighbours: *const u32,
                odd: *const u32,
                n_coarse: i32,
                dim: i32,
                parity: i32,
                minus: i32,
                gamma5: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $coarse_gamma5(
                v: *mut $t,
                n_coarse: i32,
                dim: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $hop_split(
                cross: *mut $t,
                inside: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                leg: i32,
                block: *const i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $fill(
                p: *mut c_void,
                fine: *const $t,
                n: i32,
                first: i32,
                dims: *const i32,
                ls: i32,
                gamma: *const i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $rotate(x: *mut $t, c: *const f64, n: i32, vectors: i32, stream: *mut c_void);
            fn $scal(x: *mut $t, a: *const f64, n: i32, vectors: i32, stream: *mut c_void);
            fn $caxpy(
                y: *mut $t,
                x: *const $t,
                a: *const f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $caypx(
                y: *mut $t,
                x: *const $t,
                a: *const f64,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
        }

        impl DeviceReal for $t {
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            unsafe fn clover_term(
                out: *mut Self,
                inp: *const Self,
                clover: *const Self,
                geometry: *const Geometry,
                parity: i32,
                vectors: i32,
            ) {
                $clover_term(out, inp, clover, geometry, parity, vectors, STREAM)
            }
            unsafe fn mdwf_dslash(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                geometry: *const Geometry,
                params: *const MdwfKernelParams,
                vectors: i32,
            ) {
                $mdwf(out, inp, gauge, geometry, params, vectors, STREAM)
            }
            unsafe fn hop(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                from: *const Self,
                clover: *const Self,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
                vectors: i32,
            ) {
                $hop(
                    out,
                    inp,
                    gauge,
                    from,
                    clover,
                    geometry,
                    kappa,
                    mass,
                    b5,
                    c5,
                    dagger,
                    target_parity,
                    vectors,
                    STREAM,
                )
            }
            unsafe fn mdwf_fifth(
                out: *mut Self,
                x: *const Self,
                a1: f64,
                y1: f64,
                z: *const Self,
                a2: f64,
                y2: f64,
                sites: i32,
                ls: i32,
                mass: f64,
                dagger: i32,
            ) {
                $fifth(out, x, a1, y1, z, a2, y2, sites, ls, mass, dagger, STREAM)
            }
            unsafe fn hop_gemm(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                from: *const Self,
                matrix: *const Self,
                geometry: *const Geometry,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
                vectors: i32,
            ) -> bool {
                $hop_gemm(
                    out,
                    inp,
                    gauge,
                    from,
                    matrix,
                    geometry,
                    mass,
                    b5,
                    c5,
                    dagger,
                    target_parity,
                    vectors,
                    STREAM,
                ) != 0
            }
            unsafe fn fifth_gemm(
                out: *mut Self,
                inp: *const Self,
                from: *const Self,
                matrix: *const Self,
                sites: i32,
                ls: i32,
            ) {
                $gemm(out, inp, from, matrix, sites, ls, STREAM)
            }
            unsafe fn gamma5(out: *mut Self, inp: *const Self, sites: i32, ls: i32, reflect: i32) {
                $gamma5(out, inp, sites, ls, reflect, STREAM)
            }
            unsafe fn parity_copy(
                half: *mut Self,
                full: *mut Self,
                geometry: *const Geometry,
                parity: i32,
                to_half: i32,
                vectors: i32,
            ) {
                $parity_copy(half, full, geometry, parity, to_half, vectors, STREAM)
            }
            unsafe fn walls(
                out: *mut Self,
                inp: *const Self,
                sites: i32,
                ls: i32,
                s_minus: i32,
                s_plus: i32,
                to_4d: i32,
                vectors: i32,
            ) {
                $walls(out, inp, sites, ls, s_minus, s_plus, to_4d, vectors, STREAM)
            }
            unsafe fn fold_restrict(
                coarse: *mut Self,
                fine: *const Self,
                p: *const c_void,
                sites: *const u32,
                n_coarse: i32,
                block_volume: i32,
                n: i32,
                dims: *const i32,
                ls: i32,
                parity: i32,
                gamma: *const i32,
                leg: i32,
                block: *const i32,
                vectors: i32,
            ) {
                $restrict(
                    coarse,
                    fine,
                    p,
                    sites,
                    n_coarse,
                    block_volume,
                    n,
                    dims,
                    ls,
                    parity,
                    gamma,
                    leg,
                    block,
                    vectors,
                    STREAM,
                )
            }
            unsafe fn fold_prolong(
                fine: *mut Self,
                coarse: *const Self,
                p: *const c_void,
                coarse_of: *const u32,
                n_coarse: i32,
                n: i32,
                dims: *const i32,
                ls: i32,
                parity: i32,
                gamma: *const i32,
                vectors: i32,
            ) {
                $prolong(
                    fine, coarse, p, coarse_of, n_coarse, n, dims, ls, parity, gamma, vectors,
                    STREAM,
                )
            }
            unsafe fn coarse_pass(
                out: *mut Self,
                base: *const Self,
                diag_of: *const Self,
                hops_of: *const Self,
                diag: *const Self,
                links: *const Self,
                inverse: *const Self,
                neighbours: *const u32,
                odd: *const u32,
                n_coarse: i32,
                dim: i32,
                parity: i32,
                minus: i32,
                gamma5: i32,
                vectors: i32,
            ) {
                $pass(
                    out, base, diag_of, hops_of, diag, links, inverse, neighbours, odd, n_coarse,
                    dim, parity, minus, gamma5, vectors, STREAM,
                )
            }
            unsafe fn coarse_gamma5(v: *mut Self, n_coarse: i32, dim: i32, vectors: i32) {
                $coarse_gamma5(v, n_coarse, dim, vectors, STREAM)
            }
            unsafe fn hop_split(
                cross: *mut Self,
                inside: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                leg: i32,
                block: *const i32,
                vectors: i32,
            ) {
                $hop_split(
                    cross, inside, inp, gauge, geometry, kappa, mass, b5, c5, leg, block, vectors,
                    STREAM,
                )
            }
            unsafe fn fold_fill(
                p: *mut c_void,
                fine: *const Self,
                n: i32,
                first: i32,
                dims: *const i32,
                ls: i32,
                gamma: *const i32,
                vectors: i32,
            ) {
                $fill(p, fine, n, first, dims, ls, gamma, vectors, STREAM)
            }
            unsafe fn rotate(x: *mut Self, c: *const f64, n: i32, vectors: i32) {
                $rotate(x, c, n, vectors, STREAM)
            }
            unsafe fn scal(x: *mut Self, a: *const f64, n: i32, vectors: i32) {
                $scal(x, a, n, vectors, STREAM)
            }
            unsafe fn caxpy(y: *mut Self, x: *const Self, a: *const f64, n: i32, vectors: i32) {
                $caxpy(y, x, a, n, vectors, STREAM)
            }
            unsafe fn caypx(y: *mut Self, x: *const Self, a: *const f64, n: i32, vectors: i32) {
                $caypx(y, x, a, n, vectors, STREAM)
            }
        }
    };
}

device_real!(f32: bridge_cuda_clover_term_f32, bridge_cuda_mdwf_dslash_1pass_f32, bridge_cuda_hop_f32,
    bridge_cuda_mdwf_fifth_f32, bridge_cuda_fifth_gemm_f32, bridge_cuda_hop_gemm_f32, bridge_cuda_gamma5_f32,
    bridge_cuda_parity_copy_f32, bridge_cuda_blas_scal_f32, bridge_cuda_blas_caxpy_f32,
    bridge_cuda_blas_caypx_f32, 
    bridge_cuda_walls_f32, bridge_cuda_fold_restrict_f32, bridge_cuda_fold_prolong_f32,
    bridge_cuda_coarse_pass_f32, bridge_cuda_coarse_gamma5_f32, bridge_cuda_hop_split_f32,
    bridge_cuda_fold_fill_f32, bridge_cuda_blas_rotate_f32);
device_real!(f64: bridge_cuda_clover_term_f64, bridge_cuda_mdwf_dslash_1pass_f64, bridge_cuda_hop_f64,
    bridge_cuda_mdwf_fifth_f64, bridge_cuda_fifth_gemm_f64, bridge_cuda_hop_gemm_f64, bridge_cuda_gamma5_f64,
    bridge_cuda_parity_copy_f64, bridge_cuda_blas_scal_f64, bridge_cuda_blas_caxpy_f64,
    bridge_cuda_blas_caypx_f64, 
    bridge_cuda_walls_f64, bridge_cuda_fold_restrict_f64, bridge_cuda_fold_prolong_f64,
    bridge_cuda_coarse_pass_f64, bridge_cuda_coarse_gamma5_f64, bridge_cuda_hop_split_f64,
    bridge_cuda_fold_fill_f64, bridge_cuda_blas_rotate_f64);
