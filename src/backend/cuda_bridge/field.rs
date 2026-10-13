//! Spinor fields on the device.

use super::*;
use crate::fermion::SplitParity;
use crate::{EvenOddIndex, Half, Parity, Spinor, Vector, C64};

/// Where component `c` of the spinor at `s` of site `x` is, with `ls` spinors per site: the
/// `ls` spinors of a block of sites are one after the other.
#[inline]
fn at_fifth(c: usize, s: usize, x: usize, ls: usize) -> usize {
    (((x / W) * ls + s) * SPINOR + c) * W + x % W
}

/// Where the site `full` of the whole lattice is in the field of its parity.
fn in_half(full: usize, nx: usize) -> usize {
    (full % nx) / 2 + (nx / 2) * (full / nx)
}

/// A spinor field (one vector, `ls` spinors per site) on the device: on the whole lattice,
/// or on the sites of one parity, site (x, y, z, t) of it at x/2 + (nx/2)(y + ny (z + nz t)).
pub struct DeviceSpinor {
    pub(super) data: DeviceBuffer,
    pub(super) lat: Arc<Lattice>,
    pub(super) ls: usize,
    pub(super) parity: Option<Parity>,
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
            parity: None,
        })
    }

    /// A field with `ls` spinors per site on the sites of `parity`.
    pub fn zeros_half(lat: &Arc<Lattice>, ls: usize, parity: Parity) -> Result<Self, CudaError> {
        assert!(lat.local[0].is_multiple_of(2), "an odd extent in x");
        Ok(Self {
            data: DeviceBuffer::zeros(padded(lat.num_sites() / 2) * ls * SPINOR)?,
            lat: lat.clone(),
            ls,
            parity: Some(parity),
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
        Ok(Self {
            data: DeviceBuffer::from_host(&host)?,
            lat: lat.clone(),
            ls,
            parity: None,
        })
    }

    /// The field back on the host.
    pub fn to_host(&self) -> Result<Field<f64, Spinor>, CudaError> {
        assert!(self.parity.is_none(), "a field on one parity");
        let host = self.data.download()?;
        let ls = self.ls;
        let mut psi = Field::<f64, Spinor>::with_shape(&self.lat, SPINOR, ls, 1, 1);
        for x in 0..self.lat.num_sites() {
            let site = psi.site_mut(0, x).as_chunks_mut::<SPINOR>().0;
            for (s, spinor) in site.iter_mut().enumerate() {
                for (c, v) in spinor.iter_mut().enumerate() {
                    *v = host[at_fifth(c, s, x, ls)];
                }
            }
        }
        Ok(psi)
    }

    /// The field `psi` on the sites of one parity (one vector), numbered by `index`, on the
    /// device.
    pub fn from_host_half(
        psi: &Field<f64, Spinor, Half>,
        index: &EvenOddIndex,
    ) -> Result<Self, CudaError> {
        assert_eq!(psi.num_vectors(), 1);
        let (lat, ls, parity) = (psi.lattice(), psi.ls(), psi.parity());
        assert!(lat.local[0].is_multiple_of(2), "an odd extent in x");
        let mut host = vec![0.0; padded(lat.num_sites() / 2) * ls * SPINOR];
        for (h, full) in index.full_sites(parity).iter().enumerate() {
            let x = in_half(*full as usize, lat.local[0]);
            for (s, spinor) in psi.site(0, h).as_chunks::<SPINOR>().0.iter().enumerate() {
                for (c, v) in spinor.iter().enumerate() {
                    host[at_fifth(c, s, x, ls)] = *v;
                }
            }
        }
        Ok(Self {
            data: DeviceBuffer::from_host(&host)?,
            lat: lat.clone(),
            ls,
            parity: Some(parity),
        })
    }

    /// A field on one parity back on the host, numbered by `index`.
    pub fn to_host_half(
        &self,
        index: &EvenOddIndex,
    ) -> Result<Field<f64, Spinor, Half>, CudaError> {
        let parity = self.parity.expect("a field on one parity");
        let host = self.data.download()?;
        let ls = self.ls;
        let mut psi = Field::<f64, Spinor, Half>::with_shape(&self.lat, SPINOR, ls, 1, 1);
        psi.set_parity(parity);
        for (h, full) in index.full_sites(parity).iter().enumerate() {
            let x = in_half(*full as usize, self.lat.local[0]);
            let site = psi.site_mut(0, h).as_chunks_mut::<SPINOR>().0;
            for (s, spinor) in site.iter_mut().enumerate() {
                for (c, v) in spinor.iter_mut().enumerate() {
                    *v = host[at_fifth(c, s, x, ls)];
                }
            }
        }
        Ok(psi)
    }

    /// Spinors per site.
    pub fn ls(&self) -> usize {
        self.ls
    }

    /// The parity of the sites of the field; `None` on the whole lattice.
    pub fn parity(&self) -> Option<Parity> {
        self.parity
    }

    /// Sites of the field (not counting the padding).
    pub(super) fn sites(&self) -> usize {
        self.lat.num_sites() / if self.parity.is_some() { 2 } else { 1 }
    }

    /// Reals of the buffer, as the kernels count them.
    pub(super) fn len(&self) -> i32 {
        self.data.len as i32
    }

    /// Whether the field is on the lattice of `like`, with `ls` spinors per site, on the
    /// whole lattice (`None`) or on `parity`.
    pub(super) fn is(&self, like: &Self, ls: usize, parity: Option<Parity>) -> bool {
        Arc::ptr_eq(&self.lat, &like.lat) && self.ls == ls && self.parity == parity
    }
}

/// One vector per field: `select`, `assign` and `zeros_with` take that one only. The padding
/// of a field is zero and stays zero under these operations.
impl Vector for DeviceSpinor {
    fn num_vectors(&self) -> usize {
        1
    }

    fn zeros_like(&self) -> Self {
        Self {
            data: must(DeviceBuffer::zeros(self.data.len)),
            lat: self.lat.clone(),
            ls: self.ls,
            parity: self.parity,
        }
    }

    fn zeros_with(&self, num_vectors: usize) -> Self {
        assert_eq!(num_vectors, 1, "a device field holds one vector");
        self.zeros_like()
    }

    fn select(&self, which: &[usize]) -> Self {
        assert_eq!(which, [0], "a device field holds one vector");
        let mut out = self.zeros_like();
        out.copy_from(self);
        out
    }

    fn assign(&mut self, which: &[usize], from: &Self) {
        assert_eq!(which, [0], "a device field holds one vector");
        self.copy_from(from);
    }

    fn same_shape(&self, x: &Self) -> bool {
        Arc::ptr_eq(&self.lat, &x.lat) && self.ls == x.ls && self.data.len == x.data.len
    }

    fn copy_from(&mut self, x: &Self) {
        assert!(self.same_shape(x));
        must(self.data.copy_from(&x.data));
        self.parity = x.parity;
    }

    fn set_zero(&mut self) {
        must(self.data.set_zero());
    }

    fn scale(&mut self, a: &[f64]) {
        // SAFETY: the buffer holds `len` reals.
        unsafe { bridge_cuda_blas_scal_f64(self.data.ptr, a[0], self.len(), STREAM) };
        finish();
    }

    fn axpy(&mut self, a: &[C64], x: &Self) {
        assert!(self.same_shape(x));
        // SAFETY: two buffers of `len` reals, not the same one (`self` is borrowed mutably).
        unsafe {
            bridge_cuda_blas_caxpy_f64(
                self.data.ptr,
                x.data.ptr,
                a[0].re,
                a[0].im,
                self.len(),
                STREAM,
            );
        }
        finish();
    }

    fn aypx(&mut self, a: &[C64], x: &Self) {
        assert!(self.same_shape(x));
        // SAFETY: as in `axpy`.
        unsafe {
            bridge_cuda_blas_caypx_f64(
                self.data.ptr,
                x.data.ptr,
                a[0].re,
                a[0].im,
                self.len(),
                STREAM,
            );
        }
        finish();
    }

    fn dot(&self, x: &Self, out: &mut [C64]) {
        assert!(self.same_shape(x));
        let (mut re, mut im) = (0.0, 0.0);
        // SAFETY: two buffers of `len` reals and two places for the result.
        unsafe {
            bridge_cuda_blas_dotc_f64(
                &mut re,
                &mut im,
                self.data.ptr,
                x.data.ptr,
                self.len(),
                STREAM,
            );
        }
        finish();
        out[0] = C64::new(re, im);
    }

    fn norm_sqr(&self, out: &mut [f64]) {
        // SAFETY: the buffer holds `len` reals.
        out[0] = unsafe { bridge_cuda_blas_norm2_f64(self.data.ptr, self.len(), STREAM) };
        finish();
    }
}

/// How a field on the device splits into its two parities (for `solve_even_odd`).
pub struct DeviceParities;

impl DeviceParities {
    /// Copies the part of `full` on the parity of `half` out of it (`to_half`), or into it.
    fn copy(half: &DeviceSpinor, full: &DeviceSpinor, to_half: bool) {
        let parity = half.parity.expect("a field on one parity");
        assert!(full.is(half, half.ls, None), "not the same lattice and Ls");
        let geometry = geometry(&full.lat, [Boundary::Periodic; ND], full.ls);
        // SAFETY: the buffers hold the sites of one parity and of the whole lattice with the
        // same `ls`; the kernel writes one of them and reads the other.
        unsafe {
            bridge_cuda_parity_copy_f64(
                half.data.ptr,
                full.data.ptr,
                &geometry,
                parity.index() as i32,
                i32::from(to_half),
                STREAM,
            );
        }
        finish();
    }
}

impl SplitParity<DeviceSpinor, DeviceSpinor> for DeviceParities {
    fn half_like(&self, like: &DeviceSpinor, parity: Parity) -> DeviceSpinor {
        must(DeviceSpinor::zeros_half(&like.lat, like.ls, parity))
    }

    fn extract(&self, half: &mut DeviceSpinor, full: &DeviceSpinor, parity: Parity) {
        half.parity = Some(parity);
        Self::copy(half, full, true);
    }

    fn insert(&self, full: &mut DeviceSpinor, half: &DeviceSpinor) {
        Self::copy(half, full, false);
    }
}
