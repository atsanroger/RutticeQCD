//! Spinor fields on the device.

use super::*;
use crate::fermion::SplitParity;
use crate::{ConvertFrom, EvenOddIndex, Half, Parity, Sites, Spinor, Vector, C64};

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

/// Spinor fields of reals `T` on the device, `vectors` of them one after the other, each with
/// `ls` spinors per site: on the whole lattice, or on the sites of one parity, site
/// (x, y, z, t) of it at x/2 + (nx/2)(y + ny (z + nz t)).
pub struct DeviceSpinor<T = f64> {
    pub(super) data: DeviceBuffer<T>,
    pub(super) lat: Arc<Lattice>,
    pub(super) ls: usize,
    pub(super) parity: Option<Parity>,
    pub(super) vectors: usize,
}

impl<T: DeviceReal> DeviceSpinor<T> {
    /// Zero fields on the whole lattice (`parity` none) or on one parity.
    fn new(
        lat: &Arc<Lattice>,
        ls: usize,
        parity: Option<Parity>,
        vectors: usize,
    ) -> Result<Self, CudaError> {
        assert!(ls > 0);
        let sites = match parity {
            None => lat.num_sites(),
            Some(_) => {
                assert!(lat.local[0].is_multiple_of(2), "an odd extent in x");
                lat.num_sites() / 2
            }
        };
        Ok(Self {
            data: DeviceBuffer::zeros(vectors * padded(sites) * ls * SPINOR)?,
            lat: lat.clone(),
            ls,
            parity,
            vectors,
        })
    }

    /// A field of four-dimensional spinors.
    pub fn zeros(lat: &Arc<Lattice>) -> Result<Self, CudaError> {
        Self::new(lat, 1, None, 1)
    }

    /// A field with `ls` spinors per site.
    pub fn zeros_fifth(lat: &Arc<Lattice>, ls: usize) -> Result<Self, CudaError> {
        Self::new(lat, ls, None, 1)
    }

    /// A field with `ls` spinors per site on the sites of `parity`.
    pub fn zeros_half(lat: &Arc<Lattice>, ls: usize, parity: Parity) -> Result<Self, CudaError> {
        Self::new(lat, ls, Some(parity), 1)
    }

    /// The shape of fields with `ls` spinors per site on the sites of `parity`: a set of no
    /// vectors, which takes no memory on the device. For `zeros_with`, and as the `like` of
    /// work fields that fit themselves to what they are given.
    pub fn shape_half(lat: &Arc<Lattice>, ls: usize, parity: Parity) -> Result<Self, CudaError> {
        Self::new(lat, ls, Some(parity), 0)
    }

    /// The fields of `psi` on the device, the host site `sites[x]` at `place(x)` there.
    fn upload<S: Sites>(
        psi: &Field<f64, Spinor, S>,
        parity: Option<Parity>,
        place: impl Fn(usize) -> usize,
    ) -> Result<Self, CudaError> {
        let (lat, ls, vectors) = (psi.lattice(), psi.ls(), psi.num_vectors());
        let mut out = Self::new(lat, ls, parity, vectors)?;
        let one = out.reals();
        let mut host = vec![0.0; out.data.len];
        for h in 0..psi.num_sites() {
            let x = place(h);
            // A site of the host field: its vectors one after the other, the `ls` spinors of
            // each together.
            for (k, spinor) in psi.site(0, h).as_chunks::<SPINOR>().0.iter().enumerate() {
                let (v, s) = (k / ls, k % ls);
                for (c, value) in spinor.iter().enumerate() {
                    host[v * one + at_fifth(c, s, x, ls)] = *value;
                }
            }
        }
        out.data = DeviceBuffer::from_host(&host)?;
        Ok(out)
    }

    /// The fields back on the host, as `upload` took them.
    fn download<S: Sites>(
        &self,
        psi: &mut Field<f64, Spinor, S>,
        place: impl Fn(usize) -> usize,
    ) -> Result<(), CudaError> {
        let host = self.data.download()?;
        let (ls, one) = (self.ls, self.reals());
        for h in 0..psi.num_sites() {
            let x = place(h);
            let site = psi.site_mut(0, h).as_chunks_mut::<SPINOR>().0;
            for (k, spinor) in site.iter_mut().enumerate() {
                let (v, s) = (k / ls, k % ls);
                for (c, value) in spinor.iter_mut().enumerate() {
                    *value = host[v * one + at_fifth(c, s, x, ls)];
                }
            }
        }
        Ok(())
    }

    /// The fields of `psi` (the usual layout) on the device.
    pub fn from_host(psi: &Field<f64, Spinor>) -> Result<Self, CudaError> {
        Self::upload(psi, None, |x| x)
    }

    /// The fields back on the host.
    pub fn to_host(&self) -> Result<Field<f64, Spinor>, CudaError> {
        assert!(self.parity.is_none(), "a field on one parity");
        let mut psi = Field::<f64, Spinor>::with_shape(&self.lat, SPINOR, self.ls, self.vectors, 1);
        self.download(&mut psi, |x| x)?;
        Ok(psi)
    }

    /// The fields of `psi` on the sites of one parity, numbered by `index`, on the device.
    pub fn from_host_half(
        psi: &Field<f64, Spinor, Half>,
        index: &EvenOddIndex,
    ) -> Result<Self, CudaError> {
        let (parity, nx) = (psi.parity(), psi.lattice().local[0]);
        let full = index.full_sites(parity);
        Self::upload(psi, Some(parity), |h| in_half(full[h] as usize, nx))
    }

    /// Fields on one parity back on the host, numbered by `index`.
    pub fn to_host_half(
        &self,
        index: &EvenOddIndex,
    ) -> Result<Field<f64, Spinor, Half>, CudaError> {
        let parity = self.parity.expect("a field on one parity");
        let mut psi =
            Field::<f64, Spinor, Half>::with_shape(&self.lat, SPINOR, self.ls, self.vectors, 1);
        psi.set_parity(parity);
        let (full, nx) = (index.full_sites(parity), self.lat.local[0]);
        self.download(&mut psi, |h| in_half(full[h] as usize, nx))?;
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

    /// Reals of one vector on the device.
    fn reals(&self) -> usize {
        self.data.len / self.vectors
    }

    /// Whether the field is on the lattice of `like`, with `ls` spinors per site, on the
    /// whole lattice (`None`) or on `parity`.
    pub(super) fn is(&self, like: &Self, ls: usize, parity: Option<Parity>) -> bool {
        Arc::ptr_eq(&self.lat, &like.lat) && self.ls == ls && self.parity == parity
    }
}

impl<T: DeviceReal> DeviceSpinor<T> {
    /// A checksum of the reals of the fields, padding included, to compare runs bit by bit.
    pub fn checksum(&self) -> Result<u64, CudaError> {
        self.data.checksum()
    }

    /// The fields to the host, to make room on the device.
    pub fn stash(self) -> Result<Stashed<T>, CudaError> {
        Ok(Stashed {
            host: self.data.download_raw()?,
            lat: self.lat.clone(),
            ls: self.ls,
            parity: self.parity,
            vectors: self.vectors,
        })
    }
}

/// Fields kept on the host as they are on the device, while the device needs the room
/// (`DeviceSpinor::stash`): back all at once or one vector at a time.
pub struct Stashed<T> {
    host: Vec<T>,
    lat: Arc<Lattice>,
    ls: usize,
    parity: Option<Parity>,
    vectors: usize,
}

impl<T: DeviceReal> Stashed<T> {
    pub fn num_vectors(&self) -> usize {
        self.vectors
    }

    /// Reals of one vector.
    fn one(&self) -> usize {
        self.host.len() / self.vectors
    }

    /// The vectors `range` on the device.
    fn part(&self, range: std::ops::Range<usize>) -> Result<DeviceSpinor<T>, CudaError> {
        let one = self.one();
        Ok(DeviceSpinor {
            data: DeviceBuffer::upload(&self.host[range.start * one..range.end * one])?,
            lat: self.lat.clone(),
            ls: self.ls,
            parity: self.parity,
            vectors: range.len(),
        })
    }

    /// All the fields on the device again.
    pub fn restore(&self) -> Result<DeviceSpinor<T>, CudaError> {
        self.part(0..self.vectors)
    }

    /// Vector `j` on the device.
    pub fn vector(&self, j: usize) -> Result<DeviceSpinor<T>, CudaError> {
        self.part(j..j + 1)
    }

    /// Vector `j` = the one vector of `v`.
    pub fn set_vector(&mut self, j: usize, v: &DeviceSpinor<T>) -> Result<(), CudaError> {
        let one = self.one();
        assert!(v.vectors == 1 && v.data.len == one && v.parity == self.parity);
        self.host[j * one..(j + 1) * one].copy_from_slice(&v.data.download_raw()?);
        Ok(())
    }
}

/// The padding of a field is zero and stays zero under these operations.
impl<T: DeviceReal> Vector for DeviceSpinor<T> {
    fn num_vectors(&self) -> usize {
        self.vectors
    }

    fn zeros_like(&self) -> Self {
        self.zeros_with(self.vectors)
    }

    fn zeros_with(&self, num_vectors: usize) -> Self {
        must(Self::new(&self.lat, self.ls, self.parity, num_vectors))
    }

    fn select(&self, which: &[usize]) -> Self {
        let out = self.zeros_with(which.len());
        columns::select(&out.data, &self.data, self.vectors, which);
        out
    }

    fn assign(&mut self, which: &[usize], from: &Self) {
        assert!(from.vectors == which.len() && self.reals() == from.reals());
        columns::assign(&self.data, self.vectors, which, &from.data);
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
        columns::scale(&self.data, self.vectors, a);
    }

    fn axpy(&mut self, a: &[C64], x: &Self) {
        assert!(self.same_shape(x));
        columns::axpy(&self.data, &x.data, self.vectors, a);
    }

    fn aypx(&mut self, a: &[C64], x: &Self) {
        assert!(self.same_shape(x));
        columns::aypx(&self.data, &x.data, self.vectors, a);
    }

    fn dot(&self, x: &Self, out: &mut [C64]) {
        assert!(self.same_shape(x));
        columns::dot(&self.data, &x.data, self.vectors, out);
    }

    fn norm_sqr(&self, out: &mut [f64]) {
        columns::norm_sqr(&self.data, self.vectors, out);
    }

    fn axpy_norm_sqr(&mut self, a: &[C64], x: &Self, out: &mut [f64]) {
        assert!(self.same_shape(x));
        columns::axpy_norm_sqr(&self.data, &x.data, self.vectors, a, out);
    }

    fn xpay(&mut self, x: &Self, a: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        columns::xpay(&self.data, &x.data, a, &y.data, self.vectors);
        self.parity = x.parity;
    }

    fn xpay_norm_sqr_dot(
        &mut self,
        x: &Self,
        a: &[C64],
        y: &Self,
        norm: &mut [f64],
        z: &Self,
        dot: &mut [C64],
    ) {
        assert!(self.same_shape(x) && self.same_shape(y) && self.same_shape(z));
        columns::xpay_norm_sqr_dot(
            &self.data,
            &x.data,
            a,
            &y.data,
            &z.data,
            self.vectors,
            norm,
            dot,
        );
        self.parity = x.parity;
    }

    fn axpy2(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        columns::axpy2(&self.data, a, &x.data, b, &y.data, self.vectors);
    }

    fn scale_axpy2_norm_sqr(
        &mut self,
        c: &[f64],
        a: &[C64],
        x: &Self,
        b: &[C64],
        y: &Self,
        out: &mut [f64],
    ) {
        assert!(self.same_shape(x) && self.same_shape(y));
        let v = self.vectors;
        columns::scale_axpy2_norm_sqr(&self.data, c, a, &x.data, b, &y.data, v, out);
    }

    fn axpy_aypx(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        columns::axpy_aypx(&self.data, a, &x.data, b, &y.data, self.vectors);
    }

    fn dot_norms(&self, x: &Self, dot: &mut [C64], norm: &mut [f64], norm_x: &mut [f64]) {
        assert!(self.same_shape(x));
        columns::dot_norms(&self.data, &x.data, self.vectors, dot, norm, norm_x);
    }
}

impl<T: DeviceReal> crate::Subspace for DeviceSpinor<T> {
    /// In place; for more vectors than the kernel takes, into a second set (one more set on
    /// the device while it runs).
    fn rotate(&mut self, c: &[C64]) {
        let n = self.vectors;
        assert_eq!(c.len(), n * n);
        if n <= columns::ROTATE_VECTORS {
            return columns::rotate(&self.data, n, c);
        }
        let out = self.zeros_like();
        columns::combine(&out.data, &self.data, n, c);
        *self = out;
    }
}

/// From one precision to another, on the device.
impl<T: DeviceReal, U: DeviceReal> ConvertFrom<DeviceSpinor<U>> for DeviceSpinor<T> {
    fn convert_from(&mut self, src: &DeviceSpinor<U>) {
        assert!(Arc::ptr_eq(&self.lat, &src.lat) && self.ls == src.ls);
        convert(&mut self.data, &src.data);
        self.parity = src.parity;
    }
}

/// How a field on the device splits into its two parities (for `solve_even_odd`).
pub struct DeviceParities;

impl DeviceParities {
    /// Copies the part of `full` on the parity of `half` out of it (`to_half`), or into it.
    fn copy<T: DeviceReal>(half: &DeviceSpinor<T>, full: &DeviceSpinor<T>, to_half: bool) {
        let parity = half.parity.expect("a field on one parity");
        assert!(full.is(half, half.ls, None), "not the same lattice and Ls");
        assert_eq!(half.vectors, full.vectors, "another number of vectors");
        let geometry = geometry(&full.lat, [Boundary::Periodic; ND], full.ls);
        // SAFETY: the fields hold the sites of one parity and of the whole lattice with the same
        // `ls` and as many vectors; the kernel writes one of them and reads the other.
        unsafe {
            T::parity_copy(
                half.data.ptr,
                full.data.ptr,
                &geometry,
                parity.index() as i32,
                i32::from(to_half),
                full.vectors as i32,
            );
        }
    }
}

impl<T: DeviceReal> SplitParity<DeviceSpinor<T>, DeviceSpinor<T>> for DeviceParities {
    fn half_like(&self, like: &DeviceSpinor<T>, parity: Parity) -> DeviceSpinor<T> {
        must(DeviceSpinor::new(
            &like.lat,
            like.ls,
            Some(parity),
            like.vectors,
        ))
    }

    fn extract(&self, half: &mut DeviceSpinor<T>, full: &DeviceSpinor<T>, parity: Parity) {
        half.parity = Some(parity);
        Self::copy(half, full, true);
    }

    fn insert(&self, full: &mut DeviceSpinor<T>, half: &DeviceSpinor<T>) {
        Self::copy(half, full, false);
    }
}
