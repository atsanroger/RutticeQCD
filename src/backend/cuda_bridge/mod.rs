//! The CUDA kernels of `cuda_bridge/` (extracted from Bridge++): fields on the device and the
//! operators on them. One process, the Dirac representation, single or double precision
//! (`DeviceReal`).
//!
//! A device field is the AoSoA layout of the kernels, 32 sites to a block: component `c` of
//! site `x` is at `((x / 32) * n + c) * 32 + x % 32` for `n` components per site, the number of
//! sites padded to a multiple of 32. The components of a spinor and of a link are in the order
//! of the fields here (spin outermost, then colour, then real and imaginary part; a link row by
//! row, the four directions of a site one after the other).
//!
//! `DeviceSpinor` is a `Vector` and the operators are `FermionOp` and `EvenOdd` on it, so the
//! solvers run on them as they are. Making a field or an operator returns a `CudaError` if the
//! device refuses (no memory); the operations of `Vector`, `FermionOp` and `EvenOdd` have no
//! error to return and panic if the CUDA runtime fails; a kernel that fails is noticed at the
//! next reduction or transfer, not where it was launched.

mod field;
mod mdwf;
mod wilson;

pub use field::{DeviceParities, DeviceSpinor};
pub use mdwf::CudaMdwf;
pub use wilson::{CudaClover, CudaWilson};

use crate::fermion::GaugeError;
use crate::{Boundary, Field, Gauge, Lattice, Parity, ND};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

/// Sites per block of the device layout.
const W: usize = 32;
const SPINOR: usize = 24;
/// Reals of one row of the four links of a site.
const LINK_ROW: usize = 24;

#[repr(C)]
pub struct Geometry {
    dims: [i32; 4],
    ls: i32,
    bc: [i32; 4],
    vol: i32,
    vol5d: i32,
    vol_pad: i32,
    gamma_rep: i32,
    /// Rows of a link the operators read (`DeviceGauge`).
    link_rows: i32,
}

#[repr(C)]
pub struct WilsonKernelParams {
    kappa: f64,
}

#[repr(C)]
pub struct CloverKernelParams {
    kappa: f64,
    csw: f64,
}

#[repr(C)]
pub struct MdwfKernelParams {
    m5: f64,
    mass: f64,
    alpha: f64,
    b5: f64,
    c5: f64,
}

/// Largest fifth dimension: the matrix of the fifth dimension must fit in the shared memory
/// of a block of threads (`fifth_gemm.cuh`).
const MAX_LS: usize = 32;

const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;
const DEVICE_TO_DEVICE: i32 = 3;

/// The default stream.
const STREAM: *mut c_void = std::ptr::null_mut();

extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, bytes: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn bridge_cuda_cast_f64_to_f32(dst: *mut f32, src: *const f64, n: i32, stream: *mut c_void);
    fn bridge_cuda_cast_f32_to_f64(dst: *mut f64, src: *const f32, n: i32, stream: *mut c_void);
}

/// A real of the device, `f32` or `f64`: the kernels in that precision. The methods are the
/// functions of `cuda_bridge/include/bridge_cuda.h`; the pointers are device memory of the
/// sizes those read and write.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub trait DeviceReal: Copy + Default + PartialEq + std::fmt::Debug + 'static {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;

    unsafe fn wilson_dslash(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        params: *const WilsonKernelParams,
        dagger: i32,
    );
    unsafe fn clover_term(
        out: *mut Self,
        inp: *const Self,
        clover: *const Self,
        geometry: *const Geometry,
        parity: i32,
    );
    unsafe fn clover_dslash(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        clover: *const Self,
        geometry: *const Geometry,
        params: *const CloverKernelParams,
        dagger: i32,
    );
    unsafe fn mdwf_dslash(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        params: *const MdwfKernelParams,
    );
    unsafe fn hop(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        from: *const Self,
        geometry: *const Geometry,
        kappa: f64,
        mass: f64,
        b5: f64,
        c5: f64,
        dagger: i32,
        target_parity: i32,
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
    );
    unsafe fn scal(x: *mut Self, a: f64, n: i32);
    unsafe fn caxpy(y: *mut Self, x: *const Self, ar: f64, ai: f64, n: i32);
    unsafe fn caypx(y: *mut Self, x: *const Self, ar: f64, ai: f64, n: i32);
    unsafe fn norm2(x: *const Self, n: i32) -> f64;
    unsafe fn dotc(re: *mut f64, im: *mut f64, x: *const Self, y: *const Self, n: i32);
}

/// `DeviceReal` for one type, on the functions of that precision.
macro_rules! device_real {
    ($t:ty: $wilson:ident, $clover_term:ident, $clover:ident, $mdwf:ident, $hop:ident,
     $fifth:ident, $gemm:ident, $hop_gemm:ident, $gamma5:ident, $parity_copy:ident, $scal:ident,
     $caxpy:ident, $caypx:ident, $norm2:ident, $dotc:ident) => {
        extern "C" {
            fn $wilson(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                params: *const WilsonKernelParams,
                dagger: i32,
                stream: *mut c_void,
            );
            fn $clover_term(
                out: *mut $t,
                inp: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                parity: i32,
                stream: *mut c_void,
            );
            fn $clover(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                params: *const CloverKernelParams,
                dagger: i32,
                stream: *mut c_void,
            );
            fn $mdwf(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                params: *const MdwfKernelParams,
                stream: *mut c_void,
            );
            fn $hop(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
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
                stream: *mut c_void,
            );
            fn $scal(x: *mut $t, a: $t, n: i32, stream: *mut c_void);
            fn $caxpy(y: *mut $t, x: *const $t, ar: $t, ai: $t, n: i32, stream: *mut c_void);
            fn $caypx(y: *mut $t, x: *const $t, ar: $t, ai: $t, n: i32, stream: *mut c_void);
            fn $norm2(x: *const $t, n: i32, stream: *mut c_void) -> f64;
            fn $dotc(
                re: *mut f64,
                im: *mut f64,
                x: *const $t,
                y: *const $t,
                n: i32,
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
            unsafe fn wilson_dslash(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                geometry: *const Geometry,
                params: *const WilsonKernelParams,
                dagger: i32,
            ) {
                $wilson(out, inp, gauge, geometry, params, dagger, STREAM)
            }
            unsafe fn clover_term(
                out: *mut Self,
                inp: *const Self,
                clover: *const Self,
                geometry: *const Geometry,
                parity: i32,
            ) {
                $clover_term(out, inp, clover, geometry, parity, STREAM)
            }
            unsafe fn clover_dslash(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                clover: *const Self,
                geometry: *const Geometry,
                params: *const CloverKernelParams,
                dagger: i32,
            ) {
                $clover(out, inp, gauge, clover, geometry, params, dagger, STREAM)
            }
            unsafe fn mdwf_dslash(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                geometry: *const Geometry,
                params: *const MdwfKernelParams,
            ) {
                $mdwf(out, inp, gauge, geometry, params, STREAM)
            }
            unsafe fn hop(
                out: *mut Self,
                inp: *const Self,
                gauge: *const Self,
                from: *const Self,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
            ) {
                $hop(
                    out,
                    inp,
                    gauge,
                    from,
                    geometry,
                    kappa,
                    mass,
                    b5,
                    c5,
                    dagger,
                    target_parity,
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
            ) {
                $parity_copy(half, full, geometry, parity, to_half, STREAM)
            }
            unsafe fn scal(x: *mut Self, a: f64, n: i32) {
                $scal(x, a as $t, n, STREAM)
            }
            unsafe fn caxpy(y: *mut Self, x: *const Self, ar: f64, ai: f64, n: i32) {
                $caxpy(y, x, ar as $t, ai as $t, n, STREAM)
            }
            unsafe fn caypx(y: *mut Self, x: *const Self, ar: f64, ai: f64, n: i32) {
                $caypx(y, x, ar as $t, ai as $t, n, STREAM)
            }
            unsafe fn norm2(x: *const Self, n: i32) -> f64 {
                $norm2(x, n, STREAM)
            }
            unsafe fn dotc(re: *mut f64, im: *mut f64, x: *const Self, y: *const Self, n: i32) {
                $dotc(re, im, x, y, n, STREAM)
            }
        }
    };
}

device_real!(f32: bridge_cuda_wilson_dslash_f32, bridge_cuda_clover_term_f32,
    bridge_cuda_clover_dslash_f32, bridge_cuda_mdwf_dslash_1pass_f32, bridge_cuda_hop_f32,
    bridge_cuda_mdwf_fifth_f32, bridge_cuda_fifth_gemm_f32, bridge_cuda_hop_gemm_f32, bridge_cuda_gamma5_f32,
    bridge_cuda_parity_copy_f32, bridge_cuda_blas_scal_f32, bridge_cuda_blas_caxpy_f32,
    bridge_cuda_blas_caypx_f32, bridge_cuda_blas_norm2_f32, bridge_cuda_blas_dotc_f32);
device_real!(f64: bridge_cuda_wilson_dslash_f64, bridge_cuda_clover_term_f64,
    bridge_cuda_clover_dslash_f64, bridge_cuda_mdwf_dslash_1pass_f64, bridge_cuda_hop_f64,
    bridge_cuda_mdwf_fifth_f64, bridge_cuda_fifth_gemm_f64, bridge_cuda_hop_gemm_f64, bridge_cuda_gamma5_f64,
    bridge_cuda_parity_copy_f64, bridge_cuda_blas_scal_f64, bridge_cuda_blas_caxpy_f64,
    bridge_cuda_blas_caypx_f64, bridge_cuda_blas_norm2_f64, bridge_cuda_blas_dotc_f64);

/// A call of the CUDA runtime failed with this code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub i32);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the CUDA runtime returned error {}", self.0)
    }
}

impl std::error::Error for CudaError {}

fn check(code: i32) -> Result<(), CudaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError(code))
    }
}

/// For the operations that have no error to return (see the top of the file).
fn must<T>(result: Result<T, CudaError>) -> T {
    result.unwrap_or_else(|e| panic!("{e}"))
}

/// Wait for the kernels launched so far; an error of any of them shows here. The kernels are
/// not waited for one by one: what needs their result on the host waits (a reduction, a
/// transfer), and so does freeing memory they may still use.
fn finish() {
    // SAFETY: no arguments.
    must(check(unsafe { cudaDeviceSynchronize() }));
    // A kernel that could not be launched is not an error of the device: it is asked for.
    // SAFETY: no arguments.
    must(check(unsafe { cudaGetLastError() }));
}

/// Reals on the device.
struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    /// The memory is the device's: not to be sent between threads behind our back.
    _not_send: PhantomData<*mut T>,
}

impl<T: DeviceReal> DeviceBuffer<T> {
    fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    fn zeros(len: usize) -> Result<Self, CudaError> {
        let mut buffer = Self::uninitialised(len)?;
        buffer.set_zero()?;
        Ok(buffer)
    }

    fn uninitialised(len: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid place for the pointer; the size is that of `len` reals.
        check(unsafe { cudaMalloc(&mut ptr, len * std::mem::size_of::<T>()) })?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
            _not_send: PhantomData,
        })
    }

    /// The reals `host`, in the precision of the buffer, on the device.
    fn from_host(host: &[f64]) -> Result<Self, CudaError> {
        let host: Vec<T> = host.iter().map(|x| T::from_f64(*x)).collect();
        let buffer = Self::uninitialised(host.len())?;
        // SAFETY: both ranges hold `len` reals: the device one was allocated with that size.
        check(unsafe {
            cudaMemcpy(
                buffer.ptr.cast(),
                host.as_ptr().cast(),
                buffer.bytes(),
                HOST_TO_DEVICE,
            )
        })?;
        Ok(buffer)
    }

    fn set_zero(&mut self) -> Result<(), CudaError> {
        // SAFETY: the range is the buffer's; all bits zero is the real 0.
        check(unsafe { cudaMemset(self.ptr.cast(), 0, self.bytes()) })
    }

    fn copy_from(&mut self, other: &Self) -> Result<(), CudaError> {
        assert_eq!(other.len, self.len);
        // SAFETY: two device ranges of `len` reals, which do not overlap (`self` is borrowed
        // mutably).
        check(unsafe {
            cudaMemcpy(
                self.ptr.cast(),
                other.ptr.cast_const().cast(),
                self.bytes(),
                DEVICE_TO_DEVICE,
            )
        })
    }

    fn download(&self) -> Result<Vec<f64>, CudaError> {
        let mut host = vec![T::default(); self.len];
        // SAFETY: as in `from_host`, the other way.
        check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr.cast_const().cast(),
                self.bytes(),
                DEVICE_TO_HOST,
            )
        })?;
        Ok(host.into_iter().map(T::to_f64).collect())
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `cudaMalloc` and is freed once, after the kernels that
        // may still read or write it.
        unsafe {
            cudaDeviceSynchronize();
            cudaFree(self.ptr.cast());
        }
    }
}

/// Where component `c` of site `x` is, with `n` components per site.
#[inline]
fn at(c: usize, x: usize, n: usize) -> usize {
    ((x / W) * n + c) * W + x % W
}

fn padded(sites: usize) -> usize {
    sites.div_ceil(W) * W
}

/// `reals` per site of every site of the lattice, site after site, in the device layout.
fn in_blocks(host: &[f64], reals: usize) -> Vec<f64> {
    let sites = host.len() / reals;
    let mut device = vec![0.0; padded(sites) * reals];
    for (x, site) in host.chunks_exact(reals).enumerate() {
        for (c, v) in site.iter().enumerate() {
            device[at(c, x, reals)] = *v;
        }
    }
    device
}

/// The geometry the kernels take for fields of `ls` spinors per site on `lat`.
fn geometry(lat: &Lattice, boundary: [Boundary; ND], ls: usize) -> Geometry {
    assert_eq!(lat.comm.size(), 1, "the CUDA backend runs on one process");
    let sites = lat.num_sites();
    let bc = boundary.map(|b| match b {
        Boundary::Periodic => 1,
        Boundary::Antiperiodic => -1,
        Boundary::Dirichlet => 0,
        Boundary::Phase(..) => {
            panic!("the CUDA kernels take periodic, antiperiodic and Dirichlet boundaries")
        }
    });
    Geometry {
        dims: lat.local.map(|l| l as i32),
        ls: ls as i32,
        bc,
        vol: sites as i32,
        vol5d: (sites * ls) as i32,
        vol_pad: padded(sites) as i32,
        gamma_rep: 0,
        link_rows: 3,
    }
}

/// The links of an operator on the device: zero until `set`. Links in SU(3) are kept as their
/// first two rows (the kernels rebuild the third), others whole.
struct DeviceGauge<T> {
    lat: Arc<Lattice>,
    links: DeviceBuffer<T>,
    rows: usize,
}

impl<T: DeviceReal> DeviceGauge<T> {
    fn new(lat: &Arc<Lattice>) -> Result<Self, CudaError> {
        Ok(Self {
            lat: lat.clone(),
            links: DeviceBuffer::zeros(padded(lat.num_sites()) * 3 * LINK_ROW)?,
            rows: 3,
        })
    }

    /// The links `u`, and the geometry told how they are kept.
    fn set(&mut self, u: &Field<f64, Gauge>, geometry: &mut Geometry) -> Result<(), GaugeError> {
        if u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let sites = self.lat.num_sites();
        let rows = if crate::fermion::links_in_su3(u) {
            2
        } else {
            3
        };
        // The rows of a link one after the other, the links of a site one after the other.
        let (reals, kept) = (rows * LINK_ROW, 6 * rows);
        let mut host = vec![0.0; padded(sites) * reals];
        for mu in 0..ND {
            for x in 0..sites {
                for (c, v) in u.site(mu, x)[..kept].iter().enumerate() {
                    host[at(kept * mu + c, x, reals)] = *v;
                }
            }
        }
        self.links = must(DeviceBuffer::from_host(&host));
        self.rows = rows;
        geometry.link_rows = rows as i32;
        Ok(())
    }
}

/// The number the kernels have for the sites a field is on: the parity, or -1 for the whole
/// lattice.
fn parity_number(parity: Option<Parity>) -> i32 {
    parity.map_or(-1, |p| p.index() as i32)
}

/// The fifth dimension in the hops of an operator: the mass and (b, c) of `b − c Y`. Those of a
/// four-dimensional operator: no Y.
#[derive(Clone, Copy)]
struct FifthInHops {
    mass: f64,
    b: f64,
    c: f64,
}

impl FifthInHops {
    const NONE: Self = Self {
        mass: 0.0,
        b: 1.0,
        c: 0.0,
    };
}

/// `out = H (b − c Y) inp`, or with `H†`, H the hops of the Wilson kernel at `kappa`: on the
/// whole lattice (`to` none), or `out` on the sites of parity `to` and `inp` on the others.
/// With `from`, a field on the sites of `out`, `out = from −` that.
#[allow(clippy::too_many_arguments)]
fn hops<T: DeviceReal>(
    (gauge, geometry): (&DeviceGauge<T>, &Geometry),
    out: &mut DeviceSpinor<T>,
    inp: &DeviceSpinor<T>,
    from: Option<&DeviceSpinor<T>>,
    kappa: f64,
    fifth: FifthInHops,
    dagger: bool,
    to: Option<Parity>,
) {
    let ls = geometry.ls as usize;
    assert!(inp.is(out, ls, to.map(Parity::flip)) && out.ls == ls);
    assert_eq!(out.data.len, inp.data.len);
    out.parity = to;
    let from = match from {
        Some(f) => {
            assert!(f.is(out, ls, to) && f.data.len == out.data.len);
            f.data.ptr.cast_const()
        }
        None => std::ptr::null(),
    };
    // SAFETY: the spinor buffers hold the sites of the lattice or of one parity, as the
    // kernel is told; the links are those of the whole lattice; `out` is not `inp`; a null
    // `from` is not read.
    unsafe {
        T::hop(
            out.data.ptr,
            inp.data.ptr,
            gauge.links.ptr,
            from,
            geometry,
            kappa,
            fifth.mass,
            fifth.b,
            fifth.c,
            i32::from(dagger),
            parity_number(to),
        );
    }
}

/// `dst = src`, from one precision to another.
fn convert<T: DeviceReal, U: DeviceReal>(dst: &mut DeviceBuffer<T>, src: &DeviceBuffer<U>) {
    use std::any::TypeId;
    assert_eq!(dst.len, src.len);
    let n = dst.len as i32;
    let (t, u) = (TypeId::of::<T>(), TypeId::of::<U>());
    // SAFETY: two device buffers of `len` reals, of the types the checks say.
    unsafe {
        if t == TypeId::of::<f32>() && u == TypeId::of::<f64>() {
            bridge_cuda_cast_f64_to_f32(dst.ptr.cast(), src.ptr.cast_const().cast(), n, STREAM);
        } else if t == TypeId::of::<f64>() && u == TypeId::of::<f32>() {
            bridge_cuda_cast_f32_to_f64(dst.ptr.cast(), src.ptr.cast_const().cast(), n, STREAM);
        } else {
            assert_eq!(t, u);
            must(check(cudaMemcpy(
                dst.ptr.cast(),
                src.ptr.cast_const().cast(),
                dst.bytes(),
                DEVICE_TO_DEVICE,
            )));
        }
    }
}
