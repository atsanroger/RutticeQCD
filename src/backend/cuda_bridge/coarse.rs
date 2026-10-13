//! The coarse lattice of the multigrid of domain-wall fermions on the device: its fields, and
//! the prolongation P (`multigrid/fold.cuh`).

use super::*;
use crate::kernel::{Number, Real, Unit};
use crate::solver::preconditioner::multigrid::Basis;
use crate::{Coarse, ConvertFrom, Storage, Vector, C64};

/// Whole numbers on the device.
struct DeviceTable {
    ptr: *mut u32,
}

impl DeviceTable {
    fn from_host(host: &[u32]) -> Result<Self, CudaError> {
        let bytes = std::mem::size_of_val(host);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid place for the pointer; then two ranges of `bytes` bytes.
        unsafe {
            check(cudaMalloc(&mut ptr, bytes))?;
            check(cudaMemcpy(ptr, host.as_ptr().cast(), bytes, HOST_TO_DEVICE))?;
        }
        Ok(Self { ptr: ptr.cast() })
    }
}

impl Drop for DeviceTable {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `cudaMalloc` and is freed once, after the kernels that
        // may still read it.
        unsafe {
            cudaDeviceSynchronize();
            cudaFree(self.ptr.cast());
        }
    }
}

/// Fields on a coarse lattice on the device: `dim` complex numbers per site, `vectors` of
/// them one after the other. Complex number q = x dim + k of a vector is at `at(0, q, 2)`
/// (real part) and `at(1, q, 2)`: the layout the linear algebra takes.
pub struct DeviceCoarse<T = f64> {
    pub(super) data: DeviceBuffer<T>,
    pub(super) sites: usize,
    pub(super) dim: usize,
    pub(super) vectors: usize,
}

impl<T: DeviceReal> DeviceCoarse<T> {
    /// Reals of one vector.
    fn reals_of(sites: usize, dim: usize) -> usize {
        2 * padded(sites * dim)
    }

    pub fn zeros(sites: usize, dim: usize, vectors: usize) -> Result<Self, CudaError> {
        assert!(sites > 0 && dim > 0 && vectors > 0);
        Ok(Self {
            data: DeviceBuffer::zeros(vectors * Self::reals_of(sites, dim))?,
            sites,
            dim,
            vectors,
        })
    }

    pub fn from_host<S: Storage>(f: &Field<S, Coarse>) -> Result<Self, CudaError> {
        let (sites, dim, vectors) = (f.num_sites(), f.internal_dof() / 2, f.num_vectors());
        let one = Self::reals_of(sites, dim);
        let mut host = vec![0.0; vectors * one];
        for x in 0..sites {
            for (v, part) in f.site(0, x).chunks_exact(2 * dim).enumerate() {
                for k in 0..dim {
                    let q = x * dim + k;
                    host[v * one + at(0, q, 2)] = part[2 * k].load().to_f64();
                    host[v * one + at(1, q, 2)] = part[2 * k + 1].load().to_f64();
                }
            }
        }
        Ok(Self {
            data: DeviceBuffer::from_host(&host)?,
            sites,
            dim,
            vectors,
        })
    }

    /// The field on the host, in the shape of `like` but for the number of vectors.
    pub fn to_host<S: Storage>(
        &self,
        like: &Field<S, Coarse>,
    ) -> Result<Field<S, Coarse>, CudaError> {
        let (sites, dim) = (self.sites, self.dim);
        assert!(like.num_sites() == sites && like.internal_dof() == 2 * dim);
        let mut f = like.zeros_with(self.vectors);
        let one = Self::reals_of(sites, dim);
        let host = self.data.download()?;
        for x in 0..sites {
            for (v, part) in f.site_mut(0, x).chunks_exact_mut(2 * dim).enumerate() {
                for k in 0..dim {
                    let q = x * dim + k;
                    part[2 * k] = S::store(Number::from_f64(host[v * one + at(0, q, 2)]));
                    part[2 * k + 1] = S::store(Number::from_f64(host[v * one + at(1, q, 2)]));
                }
            }
        }
        Ok(f)
    }
}

/// The padding of a field is zero and stays zero under these operations.
impl<T: DeviceReal> Vector for DeviceCoarse<T> {
    fn num_vectors(&self) -> usize {
        self.vectors
    }

    fn zeros_like(&self) -> Self {
        self.zeros_with(self.vectors)
    }

    fn zeros_with(&self, num_vectors: usize) -> Self {
        must(Self::zeros(self.sites, self.dim, num_vectors))
    }

    fn select(&self, which: &[usize]) -> Self {
        let out = self.zeros_with(which.len());
        columns::select(&out.data, &self.data, self.vectors, which);
        out
    }

    fn assign(&mut self, which: &[usize], from: &Self) {
        assert!(from.vectors == which.len() && (self.sites, self.dim) == (from.sites, from.dim));
        columns::assign(&self.data, self.vectors, which, &from.data);
    }

    fn same_shape(&self, x: &Self) -> bool {
        (self.sites, self.dim, self.vectors) == (x.sites, x.dim, x.vectors)
    }

    fn copy_from(&mut self, x: &Self) {
        assert!(self.same_shape(x));
        must(self.data.copy_from(&x.data));
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
}

/// From one precision to another, on the device.
impl<T: DeviceReal, U: DeviceReal> ConvertFrom<DeviceCoarse<U>> for DeviceCoarse<T> {
    fn convert_from(&mut self, src: &DeviceCoarse<U>) {
        assert!((self.sites, self.dim, self.vectors) == (src.sites, src.dim, src.vectors));
        convert(&mut self.data, &src.data);
    }
}

/// The prolongation P of the multigrid of domain-wall fermions on the device, folded on the
/// fifth dimension and in single precision as the one of the CPU it is made from
/// (`PvEoBasis`); between fields in either precision.
pub struct DeviceBasis {
    p: DeviceBuffer<f32>,
    /// The fine sites of every coarse site, and the coarse site of every fine one.
    fine_sites: DeviceTable,
    coarse_of: DeviceTable,
    lat: Arc<Lattice>,
    ls: usize,
    /// Test vectors: a coarse site has twice as many complex numbers.
    n: usize,
    coarse_sites: usize,
    block_volume: usize,
    dims: [i32; ND],
    /// γ5 of the fold: the column and the sign of the entry of every row.
    gamma: [i32; 8],
}

impl DeviceBasis {
    pub fn new<T: Storage>(basis: &Basis<T, f32>) -> Result<Self, CudaError> {
        let agg = basis.aggregation();
        let lat = agg.fine_lattice().clone();
        let coarse_sites = agg.coarse_lattice().num_sites();
        let g = basis
            .reflection()
            .expect("a basis folded on the fifth dimension");
        let mut gamma = [0; 8];
        for a in 0..4 {
            gamma[a] = g.col[a] as i32;
            gamma[4 + a] = match g.val[a] {
                Unit::One => 1,
                Unit::MinusOne => -1,
                _ => panic!("a γ5 with real entries"),
            };
        }
        let fine_sites: Vec<u32> = (0..coarse_sites)
            .flat_map(|c| agg.fine_sites(c).iter().copied())
            .collect();
        let coarse_of: Vec<u32> = (0..lat.num_sites())
            .map(|x| agg.coarse_site(x) as u32)
            .collect();
        Ok(Self {
            p: DeviceBuffer::upload(basis.stored())?,
            fine_sites: DeviceTable::from_host(&fine_sites)?,
            coarse_of: DeviceTable::from_host(&coarse_of)?,
            ls: basis.fine_reals() / SPINOR,
            n: basis.num_test_vectors(),
            coarse_sites,
            block_volume: agg.block_volume(),
            dims: lat.local.map(|l| l as i32),
            lat,
            gamma,
        })
    }

    /// A zero field on the coarse lattice.
    pub fn coarse_field<T: DeviceReal>(&self, vectors: usize) -> DeviceCoarse<T> {
        must(DeviceCoarse::zeros(self.coarse_sites, 2 * self.n, vectors))
    }

    fn check<T: DeviceReal>(&self, coarse: &DeviceCoarse<T>, fine: &DeviceSpinor<T>) {
        assert!(Arc::ptr_eq(&self.lat, &fine.lat) && fine.ls == self.ls);
        assert!((coarse.sites, coarse.dim) == (self.coarse_sites, 2 * self.n));
        assert_eq!(coarse.vectors, fine.vectors, "another number of vectors");
    }

    /// `coarse = P† fine`, or `P† ι fine` for a field on the sites of one parity (ι puts it
    /// onto the whole lattice, zero on the other sites).
    pub fn restrict<T: DeviceReal>(&self, coarse: &mut DeviceCoarse<T>, fine: &DeviceSpinor<T>) {
        self.check(coarse, fine);
        // SAFETY: the fields are those of this basis (checked), P and the tables those of its
        // lattices; the small arrays outlive the call, which copies them.
        unsafe {
            T::fold_restrict(
                coarse.data.ptr,
                fine.data.ptr,
                self.p.ptr,
                self.fine_sites.ptr,
                self.coarse_sites as i32,
                self.block_volume as i32,
                self.n as i32,
                self.dims.as_ptr(),
                self.ls as i32,
                parity_number(fine.parity),
                self.gamma.as_ptr(),
                fine.vectors as i32,
            );
        }
    }

    /// `fine = P coarse`, or `π P coarse` on the sites of the parity `fine` is on.
    pub fn prolong<T: DeviceReal>(&self, fine: &mut DeviceSpinor<T>, coarse: &DeviceCoarse<T>) {
        self.check(coarse, fine);
        // SAFETY: as in `restrict`.
        unsafe {
            T::fold_prolong(
                fine.data.ptr,
                coarse.data.ptr,
                self.p.ptr,
                self.coarse_of.ptr,
                self.coarse_sites as i32,
                self.n as i32,
                self.dims.as_ptr(),
                self.ls as i32,
                parity_number(fine.parity),
                self.gamma.as_ptr(),
                fine.vectors as i32,
            );
        }
    }
}
