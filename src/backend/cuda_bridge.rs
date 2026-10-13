//! The CUDA kernels of `cuda_bridge/` (extracted from Bridge++): fields on the device and the
//! operators on them.
//!
//! A device field is the AoSoA layout of the kernels, 32 sites to a block: component `c` of
//! site `x` is at `((x / 32) * n + c) * 32 + x % 32` for `n` components per site, the number of
//! sites padded to a multiple of 32. The components of a spinor and of a link are in the order
//! of the fields here (spin outermost, then colour, then real and imaginary part; a link row by
//! row, the four directions of a site one after the other).

use crate::fermion::{MdwfParams, WilsonParams};
use crate::{Boundary, Field, Gauge, Lattice, Spinor, ND};
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
struct MdwfKernelParams {
    m5: f64,
    mass: f64,
    alpha: f64,
    b5: f64,
    c5: f64,
}

const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn bridge_cuda_wilson_dslash_f64(
        out: *mut f64,
        inp: *const f64,
        gauge: *const f64,
        geometry: *const Geometry,
        params: *const WilsonKernelParams,
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

/// Reals on the device.
struct DeviceBuffer {
    ptr: *mut f64,
    len: usize,
    /// The memory is the device's: not to be sent between threads behind our back.
    _not_send: PhantomData<*mut f64>,
}

impl DeviceBuffer {
    fn zeros(len: usize) -> Result<Self, CudaError> {
        let buffer = Self::uninitialised(len)?;
        buffer.upload(&vec![0.0; len])?;
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

    fn upload(&self, host: &[f64]) -> Result<(), CudaError> {
        assert_eq!(host.len(), self.len);
        // SAFETY: both ranges hold `len` reals: the device one was allocated with that size.
        check(unsafe {
            cudaMemcpy(
                self.ptr.cast(),
                host.as_ptr().cast(),
                self.len * std::mem::size_of::<f64>(),
                HOST_TO_DEVICE,
            )
        })
    }

    fn download(&self) -> Result<Vec<f64>, CudaError> {
        let mut host = vec![0.0; self.len];
        // SAFETY: as in `upload`, the other way.
        check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr.cast_const().cast(),
                self.len * std::mem::size_of::<f64>(),
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

/// Where component `c` of the spinor at `s` of site `x` is, with `ls` spinors per site: the
/// `ls` spinors of a block of sites are one after the other.
#[inline]
fn at_fifth(c: usize, s: usize, x: usize, ls: usize) -> usize {
    (((x / W) * ls + s) * SPINOR + c) * W + x % W
}

/// A spinor field (one vector, `ls` spinors per site) on the device.
pub struct DeviceSpinor {
    data: DeviceBuffer,
    lat: Arc<Lattice>,
    ls: usize,
}

impl DeviceSpinor {
    /// A field of four-dimensional spinors.
    pub fn zeros(lat: &Arc<Lattice>) -> Result<Self, CudaError> {
        Self::zeros_fifth(lat, 1)
    }

    /// A field with `ls` spinors per site.
    pub fn zeros_fifth(lat: &Arc<Lattice>, ls: usize) -> Result<Self, CudaError> {
        Ok(Self {
            data: DeviceBuffer::zeros(padded(lat.num_sites()) * ls * SPINOR)?,
            lat: lat.clone(),
            ls,
        })
    }

    /// The field `psi` (one vector, the usual layout) on the device.
    pub fn from_host(psi: &Field<f64, Spinor>) -> Result<Self, CudaError> {
        assert_eq!(psi.num_vectors(), 1);
        let (lat, ls) = (psi.lattice(), psi.ls());
        let mut host = vec![0.0; padded(lat.num_sites()) * ls * SPINOR];
        for x in 0..lat.num_sites() {
            for (s, spinor) in psi.site(0, x).as_chunks::<SPINOR>().0.iter().enumerate() {
                for (c, v) in spinor.iter().enumerate() {
                    host[at_fifth(c, s, x, ls)] = *v;
                }
            }
        }
        let data = DeviceBuffer::uninitialised(host.len())?;
        data.upload(&host)?;
        Ok(Self {
            data,
            lat: lat.clone(),
            ls,
        })
    }

    /// The field back on the host.
    pub fn to_host(&self) -> Result<Field<f64, Spinor>, CudaError> {
        let host = self.data.download()?;
        let ls = self.ls;
        let mut psi = Field::<f64, Spinor>::with_shape(&self.lat, SPINOR, ls, 1, 1);
        for x in 0..self.lat.num_sites() {
            for (s, spinor) in psi
                .site_mut(0, x)
                .as_chunks_mut::<SPINOR>()
                .0
                .iter_mut()
                .enumerate()
            {
                for (c, v) in spinor.iter_mut().enumerate() {
                    *v = host[at_fifth(c, s, x, ls)];
                }
            }
        }
        Ok(psi)
    }
}

/// The links `u` on the device and the geometry the kernels take, for `ls` spinors per site.
fn device_gauge(
    u: &Field<f64, Gauge>,
    boundary: [Boundary; ND],
    ls: usize,
) -> Result<(DeviceBuffer, Geometry), CudaError> {
    let lat = u.lattice();
    assert_eq!(lat.comm.size(), 1, "the CUDA backend runs on one process");
    let sites = lat.num_sites();
    let mut host = vec![0.0; padded(sites) * LINKS];
    for mu in 0..ND {
        for x in 0..sites {
            for (c, v) in u.site(mu, x).iter().enumerate() {
                host[at(18 * mu + c, x, LINKS)] = *v;
            }
        }
    }
    let gauge = DeviceBuffer::uninitialised(host.len())?;
    gauge.upload(&host)?;
    let bc = boundary.map(|b| match b {
        Boundary::Periodic => 1,
        Boundary::Antiperiodic => -1,
        Boundary::Dirichlet => 0,
        Boundary::Phase(..) => {
            panic!("the CUDA kernels take periodic, antiperiodic and Dirichlet boundaries")
        }
    });
    let geometry = Geometry {
        dims: lat.local.map(|l| l as i32),
        ls: ls as i32,
        bc,
        vol: sites as i32,
        vol5d: (sites * ls) as i32,
        vol_pad: padded(sites) as i32,
        gamma_rep: 0,
    };
    Ok((gauge, geometry))
}

/// The Wilson operator on the device (one process, the Dirac representation).
pub struct CudaWilson {
    params: WilsonParams,
    geometry: Geometry,
    gauge: DeviceBuffer,
}

impl CudaWilson {
    /// The operator on the links `u`.
    pub fn new(u: &Field<f64, Gauge>, params: WilsonParams) -> Result<Self, CudaError> {
        let (gauge, geometry) = device_gauge(u, params.boundary, 1)?;
        Ok(Self {
            params,
            geometry,
            gauge,
        })
    }

    /// `out = D inp`.
    pub fn mult(&self, out: &mut DeviceSpinor, inp: &DeviceSpinor) -> Result<(), CudaError> {
        assert!(out.ls == 1 && inp.ls == 1 && Arc::ptr_eq(&out.lat, &inp.lat));
        let params = WilsonKernelParams {
            kappa: self.params.kappa,
        };
        // SAFETY: the three buffers are device memory of the sizes the kernel reads and writes
        // for this geometry (`padded(sites)` blocks of spinors and of links), `out` is not
        // `inp` (one is borrowed mutably), and the structures outlive the call.
        unsafe {
            bridge_cuda_wilson_dslash_f64(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.ptr,
                &self.geometry,
                &params,
                std::ptr::null_mut(),
            );
            check(cudaDeviceSynchronize())
        }
    }
}

/// The Möbius domain-wall operator on the device (one process, the Dirac representation, the
/// plain operator: no α of Neff).
pub struct CudaMdwf {
    params: MdwfParams,
    geometry: Geometry,
    gauge: DeviceBuffer,
}

impl CudaMdwf {
    /// The operator on the links `u`.
    pub fn new(u: &Field<f64, Gauge>, params: MdwfParams) -> Result<Self, CudaError> {
        let (gauge, geometry) = device_gauge(u, params.boundary, params.ls)?;
        Ok(Self {
            params,
            geometry,
            gauge,
        })
    }

    /// `out = D inp`.
    pub fn mult(&self, out: &mut DeviceSpinor, inp: &DeviceSpinor) -> Result<(), CudaError> {
        let ls = self.params.ls;
        assert!(out.ls == ls && inp.ls == ls && Arc::ptr_eq(&out.lat, &inp.lat));
        let params = MdwfKernelParams {
            m5: self.params.m5,
            mass: self.params.mass,
            alpha: 1.0,
            b5: self.params.b,
            c5: self.params.c,
        };
        // SAFETY: as in `CudaWilson::mult`, the spinor buffers holding `ls` spinors per site.
        unsafe {
            bridge_cuda_mdwf_dslash_1pass_f64(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.ptr,
                &self.geometry,
                &params,
                std::ptr::null_mut(),
            );
            check(cudaDeviceSynchronize())
        }
    }
}
