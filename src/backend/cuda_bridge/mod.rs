//! The CUDA kernels of `cuda_bridge/` (extracted from Bridge++): fields on the device and the
//! operators on them. The Dirac representation, single or double precision (`DeviceReal`). One
//! process, or for the domain-wall operator several, each with its own device (`halo`).
//!
//! A device field is the AoSoA layout of the kernels, 32 sites to a block: component `c` of
//! site `x` is at `((x / 32) * n + c) * 32 + x % 32` for `n` components per site, the number of
//! sites padded to a multiple of 32. The components of a spinor and of a link are in the order
//! of the fields here (spin outermost, then colour, then real and imaginary part; a link row by
//! row, the four directions of a site one after the other).
//!
//! `DeviceSpinor` is a `Vector` (of one or several vectors) and the operators are `FermionOp`
//! and `EvenOdd` on it, so the solvers run on them as they are; a kernel takes all the vectors
//! of a field in one launch. Making a field or an operator returns a `CudaError` if the
//! device refuses (no memory); the operations of `Vector`, `FermionOp` and `EvenOdd` have no
//! error to return and panic if the CUDA runtime fails; a kernel that fails is noticed at the
//! next reduction or transfer, not where it was launched.

mod blas;
mod coarse;
mod columns;
mod correction;
mod ffi;
mod field;
mod gauge;
mod halo;
pub mod hmc_mixed;
pub mod hmc_multigrid;
mod levels;
mod mdwf;
pub mod mdwf_backend;
pub mod multigrid;
mod wilson;

pub use coarse::{DeviceBasis, DeviceCoarse, DeviceCoarseOp};
pub use correction::{DeviceCorrection, DeviceParts};
pub use ffi::DeviceReal;
pub use ffi::{CloverKernelParams, Geometry, MdwfKernelParams, WilsonKernelParams};
pub(crate) use field::Stashed;
pub use field::{DeviceParities, DeviceSpinor};
pub use gauge::DeviceLinks;
pub use halo::SplitReal;
pub use levels::LevelReal;
pub use levels::{DeviceAbsMultilevel, DeviceMultilevel};
pub use mdwf::CudaMdwf;
pub use wilson::{CudaClover, CudaWilson};

use crate::fermion::GaugeError;
use crate::{Boundary, Field, Gauge, Lattice, Parity, Stencil, ND};
use ffi::*;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Sites per block of the device layout.
const W: usize = 32;
const SPINOR: usize = 24;
/// Reals of one row of the four links of a site.
const LINK_ROW: usize = 24;

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

static IN_USE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// The stream-ordered memory pool of the device the fields are allocated from, made to keep
/// what is freed for the next allocation (by default it gives it back at every
/// synchronisation); null where the device has no pools, and then every field is a
/// `cudaMalloc` of its own and freeing one waits for the device.
///
/// The solvers and cycles make and drop work fields all the time (a few for every application
/// of a preconditioner): a `cudaMalloc` and a `cudaDeviceSynchronize` with `cudaFree` for each
/// cost about 0.65 ms for a coarse field of 24³×48 on an RTX 3080 under WSL, against about
/// 17 µs from the pool, and every free stopped the queue of kernels.
fn pool() -> *mut c_void {
    static POOL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let pool = *POOL.get_or_init(|| {
        let (mut device, mut supported) = (0, 0);
        let mut pool: *mut c_void = std::ptr::null_mut();
        // SAFETY: places for the results.
        let usable = unsafe {
            cudaGetDevice(&mut device) == 0
                && cudaDeviceGetAttribute(&mut supported, MEMORY_POOLS_SUPPORTED, device) == 0
                && supported != 0
                && cudaDeviceGetDefaultMemPool(&mut pool, device) == 0
        };
        if !usable {
            // SAFETY: no arguments; the failed query is not an error of the kernels.
            unsafe { cudaGetLastError() };
            return 0;
        }
        let mut keep = u64::MAX;
        let value: *mut c_void = (&mut keep as *mut u64).cast();
        // SAFETY: the attribute is a u64.
        if unsafe { cudaMemPoolSetAttribute(pool, POOL_RELEASE_THRESHOLD, value) } != 0 {
            // SAFETY: as above.
            unsafe { cudaGetLastError() };
        }
        pool as usize
    });
    pool as *mut c_void
}

/// The device of this process: `RUTTICE_DEVICE` if set, else its rank among the processes of
/// its node (as the MPI launchers and Slurm tell it) modulo the devices it sees, so that the
/// processes of a node take one device each. Chosen once for each thread, before its first
/// allocation (the device is the runtime's state of a thread).
pub(crate) fn device_of_this_process() -> i32 {
    let mut count = 0;
    // SAFETY: a place for the number.
    if unsafe { cudaGetDeviceCount(&mut count) } != 0 || count < 1 {
        // SAFETY: no arguments; the failed query is not an error of the kernels.
        unsafe { cudaGetLastError() };
        return 0;
    }
    let var = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<i32>().ok());
    if let Some(d) = var("RUTTICE_DEVICE") {
        return d;
    }
    let local = [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
    ]
    .iter()
    .find_map(|name| var(name))
    .unwrap_or(0);
    local % count
}

fn select_device() {
    thread_local! {
        static SELECTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if !SELECTED.with(|s| s.replace(true)) {
        // SAFETY: a device number the runtime checks.
        must(check(unsafe { cudaSetDevice(device_of_this_process()) }));
    }
}

/// `bytes` bytes on the device, counted for `device_memory`: from the pool (`pool`), in the
/// order of the default stream. If the device refuses, what the pool keeps unused is given
/// back and the allocation tried once more.
fn device_alloc(bytes: usize) -> Result<*mut c_void, CudaError> {
    select_device();
    let mut ptr: *mut c_void = std::ptr::null_mut();
    if bytes == 0 {
        return Ok(ptr);
    }
    let pool = pool();
    let code = if pool.is_null() {
        // SAFETY: `ptr` is a valid place for the pointer.
        unsafe { cudaMalloc(&mut ptr, bytes) }
    } else {
        // SAFETY: `ptr` is a valid place for the pointer; the pool is the device's.
        unsafe {
            match cudaMallocAsync(&mut ptr, bytes, STREAM) {
                0 => 0,
                _ => {
                    // The refusal is not an error of the kernels: not for `finish` to find.
                    cudaGetLastError();
                    cudaDeviceSynchronize();
                    cudaMemPoolTrimTo(pool, 0);
                    cudaMallocAsync(&mut ptr, bytes, STREAM)
                }
            }
        }
    };
    check(code)?;
    PEAK.fetch_max(
        IN_USE.fetch_add(bytes, Ordering::Relaxed) + bytes,
        Ordering::Relaxed,
    );
    Ok(ptr)
}

/// Frees what `device_alloc` gave for `bytes` bytes, after the kernels that may still use it:
/// in the order of the default stream, which all of them are launched on (without a pool, by
/// waiting for the device).
///
/// # Safety
/// `ptr` came from `device_alloc(bytes)` and is not used afterwards.
unsafe fn device_free(ptr: *mut c_void, bytes: usize) {
    if bytes == 0 {
        return;
    }
    if pool().is_null() {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    } else {
        cudaFreeAsync(ptr, STREAM);
    }
    IN_USE.fetch_sub(bytes, Ordering::Relaxed);
}

/// FNV-1a of the bytes of `data`: what a buffer holds, to compare runs bit by bit.
fn checksum_of<T: Copy>(data: &[T]) -> u64 {
    // SAFETY: the bytes of a slice of plain numbers.
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
    };
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// With `RUTTICE_CHECKSUM` set, the checksum of `what` on the standard error: for the stages
/// inside the library (the application prints its own with its report).
pub(crate) fn trace_checksum(what: &str, sum: impl FnOnce() -> Result<u64, CudaError>) {
    if std::env::var_os("RUTTICE_CHECKSUM").is_some() {
        match sum() {
            Ok(s) => eprintln!("  checksum of {what}: {s:016x}"),
            Err(e) => eprintln!("  checksum of {what}: {e}"),
        }
    }
}

/// The bytes this program holds on the device, and the most it has held.
pub(crate) fn device_memory() -> (usize, usize) {
    (IN_USE.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
}

/// The bytes of the device not held by this program now: its total memory less `device_memory`
/// (what the memory pool keeps unused counts as free, as the pool gives it out again).
pub(crate) fn device_room() -> usize {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: places for the two numbers.
    if unsafe { cudaMemGetInfo(&mut free, &mut total) } != 0 {
        // SAFETY: no arguments; the failed query is not an error of the kernels.
        unsafe { cudaGetLastError() };
        return 0;
    }
    total.saturating_sub(IN_USE.load(Ordering::Relaxed))
}

/// Reals on the device.
pub struct DeviceBuffer<T> {
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
        if len > 0 {
            buffer.set_zero()?;
        }
        Ok(buffer)
    }

    fn uninitialised(len: usize) -> Result<Self, CudaError> {
        let ptr = device_alloc(len * std::mem::size_of::<T>())?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
            _not_send: PhantomData,
        })
    }

    /// The reals `host`, in the precision of the buffer, on the device.
    fn from_host(host: &[f64]) -> Result<Self, CudaError> {
        let host: Vec<T> = host.iter().map(|x| T::from_f64(*x)).collect();
        Self::upload(&host)
    }

    /// The reals `host` as they are, on the device.
    fn upload(host: &[T]) -> Result<Self, CudaError> {
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
        Ok(self.download_raw()?.into_iter().map(T::to_f64).collect())
    }

    /// A checksum of the reals (`checksum_of`).
    fn checksum(&self) -> Result<u64, CudaError> {
        Ok(checksum_of(&self.download_raw()?))
    }

    /// The reals as they are, on the host.
    fn download_raw(&self) -> Result<Vec<T>, CudaError> {
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
        Ok(host)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `device_alloc` with this size and is freed once.
        unsafe { device_free(self.ptr.cast(), self.len * std::mem::size_of::<T>()) }
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

/// What runs on the device on one process only so far (the domain-wall operator exchanges the
/// faces of its hops, `halo`; nothing else does yet).
fn one_process(lat: &Lattice, what: &str) {
    assert_eq!(
        lat.comm.size(),
        1,
        "{what} on the device runs on one process so far"
    );
}

/// The geometry the kernels take for fields of `ls` spinors per site on the local lattice of
/// `lat` (on several processes the boundaries of the directions cut between them are those of
/// `halo::Halos`).
fn geometry(lat: &Lattice, boundary: [Boundary; ND], ls: usize) -> Geometry {
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

    /// The links `u`, already on the device (kept whole, in the precision `T`).
    fn set_on_device(
        &mut self,
        u: &DeviceLinks,
        geometry: &mut Geometry,
    ) -> Result<(), GaugeError> {
        if u.lat.local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let mut links = must(DeviceBuffer::<T>::uninitialised(u.data.len));
        convert(&mut links, &u.data);
        self.links = links;
        self.rows = 3;
        geometry.link_rows = 3;
        Ok(())
    }
}

/// Bridge++'s count for the eight legs of the hops of one spinor.
const HOP_FLOPS: f64 = 1320.0;

/// Counts work on the fine lattice (`meter`): `sites` sites with `fibres` spinors each (the
/// fifth dimension times the vectors), `flops` and `spinors` read or written for every fibre,
/// and `site_reals` read once a site (links, clover blocks), shared by its fibres. As QUDA counts
/// its kernels: every value a site reads from memory, the caches left out.
fn count_fine<T>(sites: usize, fibres: usize, flops: f64, spinors: f64, site_reals: f64) {
    let n = (sites * fibres) as f64;
    let size = std::mem::size_of::<T>();
    let bytes = (n * spinors * 24.0 + sites as f64 * site_reals) * size as f64;
    crate::meter::add(crate::meter::Work::Fine, size, n * flops, bytes);
}

/// The reals of the eight links the hops of a site read.
fn hop_link_reals(geometry: &Geometry) -> f64 {
    8.0 * 6.0 * geometry.link_rows as f64
}

/// Counts the hops of `inp` (`hops`, the parts of the several processes): eight neighbours and
/// the result a spinor, one more with `from`.
fn count_hops<T: DeviceReal>(
    geometry: &Geometry,
    inp: &DeviceSpinor<T>,
    from: bool,
    extra_flops: f64,
) {
    let (flops, spinors) = if from {
        (HOP_FLOPS + 48.0, 10.0)
    } else {
        (HOP_FLOPS, 9.0)
    };
    let fibres = inp.ls * inp.vectors;
    count_fine::<T>(
        inp.sites(),
        fibres,
        flops + extra_flops,
        spinors,
        hop_link_reals(geometry),
    );
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
/// With `from`, a field on the sites of `out`, `out = from −` that. Every vector of the fields.
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
    count_hops(geometry, inp, from.is_some(), 0.0);
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
            inp.vectors as i32,
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
