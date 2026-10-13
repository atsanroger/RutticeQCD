//! The CUDA kernels of `cuda_bridge/` (extracted from Bridge++): fields on the device and the
//! operators on them. One process, one thread using the device, double precision, the Dirac
//! representation.
//!
//! A device field is the AoSoA layout of the kernels, 32 sites to a block: component `c` of
//! site `x` is at `((x / 32) * n + c) * 32 + x % 32` for `n` components per site, the number of
//! sites padded to a multiple of 32. The components of a spinor and of a link are in the order
//! of the fields here (spin outermost, then colour, then real and imaginary part; a link row by
//! row, the four directions of a site one after the other).
//!
//! `DeviceSpinor` is a `Vector` and the operators are `FermionOp<DeviceSpinor>`, so the solvers
//! run on them as they are. Making a field or an operator returns a `CudaError` if the device
//! refuses (no memory); the operations of `Vector` and `FermionOp` have no error to return and
//! panic if the CUDA runtime fails.

mod field;
mod mdwf;
mod wilson;

pub use field::DeviceSpinor;
pub use mdwf::CudaMdwf;
pub use wilson::{CudaClover, CudaWilson};

use crate::fermion::GaugeError;
use crate::{Boundary, Field, Gauge, Lattice, ND};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

/// Sites per block of the device layout.
const W: usize = 32;
const SPINOR: usize = 24;
const LINKS: usize = 72;

#[repr(C)]
struct Geometry {
    dims: [i32; 4],
    ls: i32,
    bc: [i32; 4],
    vol: i32,
    vol5d: i32,
    vol_pad: i32,
    gamma_rep: i32,
}

#[repr(C)]
struct WilsonKernelParams {
    kappa: f64,
}

#[repr(C)]
struct CloverKernelParams {
    kappa: f64,
    csw: f64,
}

#[repr(C)]
struct MdwfKernelParams {
    m5: f64,
    mass: f64,
    alpha: f64,
    b5: f64,
    c5: f64,
}

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

    fn bridge_cuda_wilson_dslash_f64(
        out: *mut f64,
        inp: *const f64,
        gauge: *const f64,
        geometry: *const Geometry,
        params: *const WilsonKernelParams,
        dagger: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_clover_term_f64(
        out: *mut f64,
        inp: *const f64,
        clover: *const f64,
        geometry: *const Geometry,
        stream: *mut c_void,
    );
    fn bridge_cuda_clover_dslash_f64(
        out: *mut f64,
        inp: *const f64,
        gauge: *const f64,
        clover: *const f64,
        geometry: *const Geometry,
        params: *const CloverKernelParams,
        dagger: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_mdwf_dslash_1pass_f64(
        out: *mut f64,
        inp: *const f64,
        gauge: *const f64,
        geometry: *const Geometry,
        params: *const MdwfKernelParams,
        stream: *mut c_void,
    );
    fn bridge_cuda_mdwf_hop_f64(
        out: *mut f64,
        inp: *const f64,
        gauge: *const f64,
        geometry: *const Geometry,
        mass: f64,
        b5: f64,
        c5: f64,
        dagger: i32,
        target_parity: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_mdwf_fifth_f64(
        out: *mut f64,
        x: *const f64,
        a1: f64,
        y1: f64,
        z: *const f64,
        a2: f64,
        y2: f64,
        sites: i32,
        ls: i32,
        mass: f64,
        dagger: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_mdwf_site_inv_f64(
        out: *mut f64,
        inp: *const f64,
        sites: i32,
        ls: i32,
        params: *const MdwfKernelParams,
        dagger: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_gamma5_f64(
        out: *mut f64,
        inp: *const f64,
        sites: i32,
        ls: i32,
        reflect: i32,
        stream: *mut c_void,
    );

    fn bridge_cuda_blas_scal_f64(x: *mut f64, a: f64, n: i32, stream: *mut c_void);
    fn bridge_cuda_blas_caxpy_f64(
        y: *mut f64,
        x: *const f64,
        ar: f64,
        ai: f64,
        n: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_blas_caypx_f64(
        y: *mut f64,
        x: *const f64,
        ar: f64,
        ai: f64,
        n: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_blas_norm2_f64(x: *const f64, n: i32, stream: *mut c_void) -> f64;
    fn bridge_cuda_blas_dotc_f64(
        re: *mut f64,
        im: *mut f64,
        x: *const f64,
        y: *const f64,
        n: i32,
        stream: *mut c_void,
    );
}

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

/// Wait for the kernels launched so far; an error of any of them shows here.
fn finish() {
    // SAFETY: no arguments.
    must(check(unsafe { cudaDeviceSynchronize() }));
}

/// Reals on the device.
struct DeviceBuffer {
    ptr: *mut f64,
    len: usize,
    /// The memory is the device's: not to be sent between threads behind our back.
    _not_send: PhantomData<*mut f64>,
}

impl DeviceBuffer {
    fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<f64>()
    }

    fn zeros(len: usize) -> Result<Self, CudaError> {
        let mut buffer = Self::uninitialised(len)?;
        buffer.set_zero()?;
        Ok(buffer)
    }

    fn uninitialised(len: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid place for the pointer; the size is that of `len` reals.
        check(unsafe { cudaMalloc(&mut ptr, len * std::mem::size_of::<f64>()) })?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
            _not_send: PhantomData,
        })
    }

    /// The reals `host` on the device.
    fn from_host(host: &[f64]) -> Result<Self, CudaError> {
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
        let mut host = vec![0.0; self.len];
        // SAFETY: as in `from_host`, the other way.
        check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr.cast_const().cast(),
                self.bytes(),
                DEVICE_TO_HOST,
            )
        })?;
        Ok(host)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `cudaMalloc` and is freed once.
        unsafe {
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
    }
}

/// The links of an operator on the device: zero until `set`.
struct DeviceGauge {
    lat: Arc<Lattice>,
    links: DeviceBuffer,
}

impl DeviceGauge {
    fn new(lat: &Arc<Lattice>) -> Result<Self, CudaError> {
        let unit = Field::<f64, Gauge>::gauge(lat);
        Ok(Self {
            lat: lat.clone(),
            links: Self::upload(&unit)?,
        })
    }

    fn upload(u: &Field<f64, Gauge>) -> Result<DeviceBuffer, CudaError> {
        let sites = u.lattice().num_sites();
        let mut host = vec![0.0; padded(sites) * LINKS];
        for mu in 0..ND {
            for x in 0..sites {
                for (c, v) in u.site(mu, x).iter().enumerate() {
                    host[at(18 * mu + c, x, LINKS)] = *v;
                }
            }
        }
        DeviceBuffer::from_host(&host)
    }

    fn set(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        self.links = must(Self::upload(u));
        Ok(())
    }
}
