//! The coarse lattice of the multigrid of domain-wall fermions on the device: its fields, the
//! prolongation P and the coarse operators (`multigrid/fold.cuh`).

use super::halo::Faces;
use super::*;
use crate::fermion::{Dagger, EvenOdd, FermionOp, SplitParity, Variant};
use crate::kernel::{Dirac, Gamma, GammaRepre, Number, Real, Unit};
use crate::solver::preconditioner::multigrid::{
    Aggregation, Basis, CoarseOp, CoarseOperator, MultigridError, Transfer,
};
use crate::Parity;
use crate::{Coarse, ConvertFrom, Storage, Vector, C64, F16};
use std::cell::RefCell;
use std::rc::Rc;

/// Whole numbers on the device.
pub(super) struct DeviceTable {
    pub(super) ptr: *mut u32,
    bytes: usize,
}

impl DeviceTable {
    pub(super) fn from_host(host: &[u32]) -> Result<Self, CudaError> {
        let bytes = std::mem::size_of_val(host);
        let ptr = device_alloc(bytes)?;
        // SAFETY: two ranges of `bytes` bytes.
        check(unsafe { cudaMemcpy(ptr, host.as_ptr().cast(), bytes, HOST_TO_DEVICE) })?;
        Ok(Self {
            ptr: ptr.cast(),
            bytes,
        })
    }

    /// The first `len` numbers.
    pub(super) fn download(&self, len: usize) -> Result<Vec<u32>, CudaError> {
        let mut host = vec![0u32; len];
        // SAFETY: two ranges of `len` numbers; the table has at least as many.
        check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr.cast_const().cast(),
                4 * len,
                DEVICE_TO_HOST,
            )
        })?;
        Ok(host)
    }
}

impl Drop for DeviceTable {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `device_alloc` with this size and is freed once.
        unsafe { device_free(self.ptr.cast(), self.bytes) }
    }
}

/// P as it is stored on the device: reals in single or in half precision.
struct Stored {
    ptr: *mut c_void,
    bytes: usize,
    half: bool,
}

impl Stored {
    /// `reals` zeros.
    fn zeros(reals: usize, half: bool) -> Result<Self, CudaError> {
        let bytes = reals * if half { 2 } else { 4 };
        let ptr = device_alloc(bytes)?;
        // SAFETY: the range it was given. All bits zero is the real 0 in either precision.
        check(unsafe { cudaMemset(ptr, 0, bytes) })?;
        Ok(Self { ptr, bytes, half })
    }

    /// `host`, in single precision.
    fn upload(host: &[f32]) -> Result<Self, CudaError> {
        let stored = Self::zeros(host.len(), false)?;
        // SAFETY: two ranges of as many bytes.
        check(unsafe {
            cudaMemcpy(
                stored.ptr,
                host.as_ptr().cast(),
                stored.bytes,
                HOST_TO_DEVICE,
            )
        })?;
        Ok(stored)
    }

    /// `host`, the bits of numbers in half precision.
    fn upload_half(host: &[u16]) -> Result<Self, CudaError> {
        let stored = Self::zeros(host.len(), true)?;
        // SAFETY: two ranges of as many bytes.
        check(unsafe {
            cudaMemcpy(
                stored.ptr,
                host.as_ptr().cast(),
                stored.bytes,
                HOST_TO_DEVICE,
            )
        })?;
        Ok(stored)
    }

    /// The reals, if they are in single precision.
    fn download(&self) -> Result<Vec<f32>, CudaError> {
        assert!(!self.half, "P in half precision is not read back");
        let mut host = vec![0.0f32; self.bytes / 4];
        // SAFETY: two ranges of as many bytes.
        check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr.cast_const(),
                self.bytes,
                DEVICE_TO_HOST,
            )
        })?;
        Ok(host)
    }
}

impl Drop for Stored {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `device_alloc` with this size and is freed once.
        unsafe { device_free(self.ptr, self.bytes) }
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
    /// The processes the coarse lattice is cut between, whose sums the reductions take; none on
    /// one process.
    pub(super) comm: Option<Arc<dyn crate::Comm>>,
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
            comm: None,
        })
    }

    /// The field on the coarse lattice of `lat`: its sums over the processes of `lat`.
    pub(super) fn on(mut self, lat: &Lattice) -> Self {
        self.comm = (lat.comm.size() > 1).then(|| lat.comm.clone());
        self
    }

    /// The local sums of a reduction summed over the processes (nothing on one).
    fn sum(&self, x: &mut [f64]) {
        if let Some(comm) = &self.comm {
            comm.sum(x);
        }
    }

    fn sum_complex(&self, x: &mut [C64]) {
        if let Some(comm) = &self.comm {
            let mut reals: Vec<f64> = x.iter().flat_map(|z| [z.re, z.im]).collect();
            comm.sum(&mut reals);
            for (z, r) in x.iter_mut().zip(reals.as_chunks::<2>().0) {
                *z = C64::new(r[0], r[1]);
            }
        }
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
            comm: None,
        })
    }

    /// Vectors that are 1 in component `first + v` of every site (vector v) and zero elsewhere.
    pub(super) fn units(
        sites: usize,
        dim: usize,
        first: usize,
        vectors: usize,
    ) -> Result<Self, CudaError> {
        assert!(first + vectors <= dim);
        let units = Self::zeros(sites, dim, vectors)?;
        // SAFETY: a zero field of `vectors` vectors of `sites` sites of `dim` components, and
        // components `first ..first + vectors` of a site.
        unsafe {
            T::coarse_units(
                units.data.ptr,
                sites as i32,
                dim as i32,
                first as i32,
                vectors as i32,
            );
        }
        Ok(units)
    }

    /// The field in the precision `U`.
    pub fn to_precision<U: DeviceReal>(&self) -> Result<DeviceCoarse<U>, CudaError> {
        let mut out = DeviceCoarse::<U>::zeros(self.sites, self.dim, self.vectors)?;
        out.comm = self.comm.clone();
        convert(&mut out.data, &self.data);
        Ok(out)
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
        let mut out = must(Self::zeros(self.sites, self.dim, num_vectors));
        out.comm = self.comm.clone();
        out
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
        self.sum_complex(out);
    }

    fn norm_sqr(&self, out: &mut [f64]) {
        columns::norm_sqr(&self.data, self.vectors, out);
        self.sum(out);
    }

    fn axpy_norm_sqr(&mut self, a: &[C64], x: &Self, out: &mut [f64]) {
        assert!(self.same_shape(x));
        columns::axpy_norm_sqr(&self.data, &x.data, self.vectors, a, out);
        self.sum(out);
    }

    fn xpay(&mut self, x: &Self, a: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        columns::xpay(&self.data, &x.data, a, &y.data, self.vectors);
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
        self.sum(norm);
        self.sum_complex(dot);
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
        self.sum(out);
    }

    fn axpy_aypx(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        columns::axpy_aypx(&self.data, a, &x.data, b, &y.data, self.vectors);
    }

    fn chebyshev_step(&mut self, d: &mut Self, r: &mut Self, ad: &Self, c: &[f64], a: &[C64]) {
        assert!(self.same_shape(d) && self.same_shape(r) && self.same_shape(ad));
        columns::chebyshev_step(&self.data, &d.data, &r.data, &ad.data, c, a, self.vectors);
    }

    fn dot_norms(&self, x: &Self, dot: &mut [C64], norm: &mut [f64], norm_x: &mut [f64]) {
        assert!(self.same_shape(x));
        columns::dot_norms(&self.data, &x.data, self.vectors, dot, norm, norm_x);
        self.sum_complex(dot);
        self.sum(norm);
        self.sum(norm_x);
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
/// fifth dimension as the one of the CPU (`PvEoBasis`), and kept in single precision as that
/// one or, made here, in half precision; between fields in either precision.
pub struct DeviceBasis {
    p: Stored,
    /// The fine sites of every coarse site, and the coarse site of every fine one.
    fine_sites: DeviceTable,
    coarse_of: DeviceTable,
    lat: Arc<Lattice>,
    ls: usize,
    /// Test vectors: a coarse site has twice as many complex numbers.
    n: usize,
    pub(super) coarse_sites: usize,
    block_volume: usize,
    /// Sites of an aggregate in every direction.
    pub(super) block: [i32; ND],
    /// The coarse lattice.
    pub(super) coarse: Arc<Lattice>,
    dims: [i32; ND],
    /// γ5 of the fold: the column and the sign of the entry of every row; then whether P is
    /// in half precision.
    gamma: [i32; 9],
}

/// Every vector of `v` scaled to |v_j|² = the number of sites, as `BasisBuilder::add` does on
/// the CPU: P does not change, and its parts stay far from the smallest numbers of half
/// precision.
fn to_unit_sites<T: DeviceReal>(v: &mut DeviceSpinor<T>) {
    let mut norms = vec![0.0; v.vectors];
    v.norm_sqr(&mut norms);
    let sites = v.lat.num_global_sites() as f64;
    let scales: Vec<f64> = norms
        .iter()
        .map(|n| if *n > 0.0 { (sites / n).sqrt() } else { 1.0 })
        .collect();
    v.scale(&scales);
}

impl DeviceBasis {
    /// The P of the CPU's domain-wall multigrid, kept as it is there.
    pub fn from_pv_eo<T: Storage>(
        basis: &crate::solver::preconditioner::multigrid::PvEoBasis<T>,
    ) -> Result<Self, CudaError> {
        use crate::solver::preconditioner::multigrid::PvEoBasis;
        match basis {
            PvEoBasis::Half(b) => Self::new(b),
            PvEoBasis::Single(b) => Self::new(b),
        }
    }

    /// The P of the CPU, `basis`, kept as it is there: in half (`F16`) or single precision.
    pub fn new<T: Storage, S: Storage>(basis: &Basis<T, S>) -> Result<Self, CudaError> {
        let g = basis
            .reflection()
            .expect("a basis folded on the fifth dimension");
        let data = basis.stored();
        let stored = if std::any::TypeId::of::<S>() == std::any::TypeId::of::<F16>() {
            // SAFETY: `F16` is a `u16` (`repr(transparent)`), the bits of a binary16 number.
            Stored::upload_half(unsafe {
                std::slice::from_raw_parts(data.as_ptr().cast::<u16>(), data.len())
            })?
        } else {
            assert_eq!(
                std::any::TypeId::of::<S>(),
                std::any::TypeId::of::<f32>(),
                "P is kept in half or single precision"
            );
            // SAFETY: `S` is `f32`.
            Stored::upload(unsafe {
                std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len())
            })?
        };
        Self::with(
            basis.aggregation(),
            basis.fine_reals() / SPINOR,
            basis.num_test_vectors(),
            g,
            stored,
        )
    }

    /// A P to be made on the device, for `n` test vectors with `ls` spinors per site on the
    /// aggregates `agg`: zero until every test vector was given (`add`) and `finish` called.
    /// `half`: kept in half precision, half the memory of what is the largest thing of a
    /// multigrid. The coarse operators are built from P as it is kept.
    pub fn empty(
        agg: &Arc<Aggregation>,
        ls: usize,
        n: usize,
        half: bool,
    ) -> Result<Self, CudaError> {
        assert!(
            ls.is_multiple_of(2),
            "folding needs an even fifth dimension"
        );
        let reals = agg.fine_lattice().num_sites() * n * 2 * 12 * ls;
        Self::with(agg, ls, n, Dirac::GAMMA5, Stored::zeros(reals, half)?)
    }

    fn with(
        agg: &Arc<Aggregation>,
        ls: usize,
        n: usize,
        g: Gamma,
        p: Stored,
    ) -> Result<Self, CudaError> {
        let lat = agg.fine_lattice().clone();
        let coarse_sites = agg.coarse_lattice().num_sites();
        let mut gamma = [0; 9];
        gamma[8] = i32::from(p.half);
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
            p,
            fine_sites: DeviceTable::from_host(&fine_sites)?,
            coarse_of: DeviceTable::from_host(&coarse_of)?,
            ls,
            n,
            coarse_sites,
            block_volume: agg.block_volume(),
            block: agg.block().map(|b| b as i32),
            coarse: agg.coarse_lattice().clone(),
            dims: lat.local.map(|l| l as i32),
            lat,
            gamma,
        })
    }

    /// The vectors of `v`, a field on the whole lattice, as the test vectors `first`,
    /// `first + 1`, ...: their chirality parts (1 ± Γ5) v / 2, not yet orthonormal.
    pub fn add<T: DeviceReal>(&mut self, first: usize, v: &DeviceSpinor<T>) {
        assert!(Arc::ptr_eq(&self.lat, &v.lat) && v.ls == self.ls && v.parity.is_none());
        assert!(first + v.vectors <= self.n, "more test vectors than P has");
        // SAFETY: a field on the whole lattice of this basis, and its P; the small arrays are
        // copied by the call.
        unsafe {
            T::fold_fill(
                self.p.ptr,
                v.data.ptr,
                self.n as i32,
                first as i32,
                self.dims.as_ptr(),
                self.ls as i32,
                self.gamma.as_ptr(),
                v.vectors as i32,
            );
        }
    }

    /// The chirality parts made orthonormal over every aggregate: P. As `BasisBuilder::finish`
    /// on the CPU (modified Gram-Schmidt in double precision on the parts as they are kept), by
    /// CholeskyQR2 where the vectors are far enough from dependent (the same Q, one sweep over
    /// them a pass instead of one for each pair), else by Gram-Schmidt.
    pub fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (n, kept) = (self.n, 12 * self.ls);
        let dim = self.block_volume * kept / 2;
        if n > dim {
            return Err(
                format!("{n} test vectors do not fit into aggregates of dimension {dim}").into(),
            );
        }
        // The vectors of an aggregate and chirality in double precision: as many coarse sites
        // at a time as an eighth of the room on the device holds, between a quarter of a GB and
        // 2 GB (a quarter of a GB was 34 blocks a launch on 24⁴ with 20 test vectors, too few
        // for CholeskyQR2 on the 56 SMs of an A30: P 2.9 s, against 1.8 s with 2 GB).
        let per_site = 2 * n * dim * 2;
        let budget = (super::device_room() / 8).clamp(1 << 28, 1 << 31) / 8;
        let count = (budget / per_site).clamp(1, self.coarse_sites);
        let scratch = DeviceBuffer::<f64>::uninitialised(count * per_site)?;
        let fail = DeviceTable::from_host(&vec![0; 2 * count])?;
        for first in (0..self.coarse_sites).step_by(count) {
            let sites = count.min(self.coarse_sites - first);
            // SAFETY: P and the sites of this basis, scratch and places for the results of
            // `sites` coarse sites.
            let run = |kernel: unsafe extern "C" fn(
                *mut c_void,
                i32,
                *mut f64,
                *const u32,
                *mut u32,
                i32,
                i32,
                i32,
                i32,
                i32,
                f64,
                *mut c_void,
            )| unsafe {
                kernel(
                    self.p.ptr,
                    i32::from(self.p.half),
                    scratch.ptr,
                    self.fine_sites.ptr,
                    fail.ptr,
                    first as i32,
                    sites as i32,
                    self.block_volume as i32,
                    n as i32,
                    kept as i32,
                    std::f64::consts::FRAC_1_SQRT_2,
                    STREAM,
                );
                finish();
            };
            let mut gram_schmidt = n > 48;
            if !gram_schmidt {
                run(bridge_cuda_fold_orthonormalise_cholesky);
                // Where a block fell back, it left P alone: all of them again by Gram-Schmidt.
                gram_schmidt = fail.download(2 * sites)?.contains(&u32::MAX);
            }
            if gram_schmidt {
                run(bridge_cuda_fold_orthonormalise);
            }
            if let Some((k, vector)) = fail
                .download(2 * sites)?
                .into_iter()
                .enumerate()
                .find(|(_, f)| *f != 0)
            {
                return Err(Box::new(MultigridError::DependentTestVectors {
                    coarse_site: first + k / 2,
                    vector: vector as usize - 1,
                }));
            }
        }
        Ok(())
    }

    /// The P of the even-odd Pauli-Villars multigrid from the vectors `y` on the even sites,
    /// as `pv_eo_basis` makes it on the CPU: for every y, s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y)
    /// and t̂ = (π D s, −(D_PV†)_oo⁻¹ (D_PV†)_oe π D s) (see the top of
    /// `multigrid::mdwf_even_odd`); `op` and `pv` the plain operators.
    pub fn pv_eo<T: DeviceReal>(
        op: &CudaMdwf<T>,
        pv: &CudaMdwf<T>,
        agg: &Arc<Aggregation>,
        y: &DeviceSpinor<T>,
        half: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(y.parity, Some(Parity::Even));
        let (n, ls, lat) = (y.vectors, y.ls, &y.lat);
        let mut basis = Self::empty(agg, ls, 2 * n, half)?;
        let (even, odd) = (Parity::Even, Parity::Odd);
        let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
        let mut full_work = FermionOp::work(op);
        // Two vectors at a time: the fields on the whole lattice are large.
        let all: Vec<usize> = (0..n).collect();
        for chunk in all.chunks(2) {
            let nv = chunk.len();
            let yj = y.select(chunk);
            let mut x_e = yj.zeros_like();
            let mut t = DeviceSpinor::zeros_half(lat, ls, odd)?.zeros_with(nv);
            let mut x_o = t.zeros_like();
            let mut s = DeviceSpinor::zeros_fifth(lat, ls)?.zeros_with(nv);
            let minus = vec![-1.0; nv];
            op.diag_inv(&mut work, &mut x_e, &yj, even, Dagger::No);
            op.hop(&mut work, &mut t, &x_e, odd, Dagger::No);
            op.diag_inv(&mut work, &mut x_o, &t, odd, Dagger::No);
            x_o.scale(&minus);
            DeviceParities.insert(&mut s, &x_e);
            DeviceParities.insert(&mut s, &x_o);
            to_unit_sites(&mut s);
            basis.add(chunk[0], &s);
            let mut ds = s.zeros_like();
            op.mult(&mut full_work, &mut ds, &s, Variant::D);
            // a = π D s
            DeviceParities.extract(&mut x_e, &ds, even);
            pv.hop(&mut work_pv, &mut t, &x_e, odd, Dagger::Yes);
            pv.diag_inv(&mut work_pv, &mut x_o, &t, odd, Dagger::Yes);
            x_o.scale(&minus);
            DeviceParities.insert(&mut s, &x_e);
            DeviceParities.insert(&mut s, &x_o);
            to_unit_sites(&mut s);
            basis.add(n + chunk[0], &s);
        }
        basis.finish()?;
        trace_checksum("P made on the device", || basis.checksum());
        Ok(basis)
    }

    /// The test vectors of the levels below the coarse lattice, as `pv_eo_coarse_vectors` of
    /// the CPU: P† of the 2 n vectors on the whole lattice that `pv_eo` made P of, from the
    /// same `op`, `pv` and `y`, as they are (not scaled), in the order of the columns of P.
    pub fn pv_eo_coarse_vectors<T: DeviceReal>(
        &self,
        op: &CudaMdwf<T>,
        pv: &CudaMdwf<T>,
        y: &DeviceSpinor<T>,
    ) -> Result<DeviceCoarse<T>, CudaError> {
        assert_eq!(y.parity, Some(Parity::Even));
        let (n, ls, lat) = (y.vectors, y.ls, &y.lat);
        let mut out =
            DeviceCoarse::<T>::zeros(self.coarse_sites, self.coarse_dim(), 2 * n)?.on(&self.coarse);
        let (even, odd) = (Parity::Even, Parity::Odd);
        let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
        let mut full_work = FermionOp::work(op);
        let all: Vec<usize> = (0..n).collect();
        for chunk in all.chunks(2) {
            let nv = chunk.len();
            let yj = y.select(chunk);
            let mut x_e = yj.zeros_like();
            let mut t = DeviceSpinor::zeros_half(lat, ls, odd)?.zeros_with(nv);
            let mut x_o = t.zeros_like();
            let mut s = DeviceSpinor::zeros_fifth(lat, ls)?.zeros_with(nv);
            let minus = vec![-1.0; nv];
            op.diag_inv(&mut work, &mut x_e, &yj, even, Dagger::No);
            op.hop(&mut work, &mut t, &x_e, odd, Dagger::No);
            op.diag_inv(&mut work, &mut x_o, &t, odd, Dagger::No);
            x_o.scale(&minus);
            DeviceParities.insert(&mut s, &x_e);
            DeviceParities.insert(&mut s, &x_o);
            let mut c = self.coarse_field::<T>(nv);
            self.restrict(&mut c, &s);
            out.assign(chunk, &c);
            let mut ds = s.zeros_like();
            op.mult(&mut full_work, &mut ds, &s, Variant::D);
            DeviceParities.extract(&mut x_e, &ds, even);
            pv.hop(&mut work_pv, &mut t, &x_e, odd, Dagger::Yes);
            pv.diag_inv(&mut work_pv, &mut x_o, &t, odd, Dagger::Yes);
            x_o.scale(&minus);
            DeviceParities.insert(&mut s, &x_e);
            DeviceParities.insert(&mut s, &x_o);
            self.restrict(&mut c, &s);
            out.assign(&chunk.iter().map(|j| n + j).collect::<Vec<_>>(), &c);
        }
        Ok(out)
    }

    /// A checksum of P as it is kept, to compare runs bit by bit.
    pub fn checksum(&self) -> Result<u64, CudaError> {
        let mut host = vec![0u8; self.p.bytes];
        // SAFETY: two ranges of as many bytes.
        check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.p.ptr.cast_const(),
                self.p.bytes,
                DEVICE_TO_HOST,
            )
        })?;
        Ok(checksum_of(&host))
    }

    /// P as it is kept (`Basis::stored`), from the device; not in half precision.
    pub fn stored(&self) -> Result<Vec<f32>, CudaError> {
        self.p.download()
    }

    /// Complex numbers of a coarse site.
    pub fn coarse_dim(&self) -> usize {
        2 * self.n
    }

    /// A zero field on the whole fine lattice.
    pub fn fine_field<T: DeviceReal>(&self, vectors: usize) -> Result<DeviceSpinor<T>, CudaError> {
        Ok(DeviceSpinor::zeros_fifth(&self.lat, self.ls)?.zeros_with(vectors))
    }

    /// A zero field on the coarse lattice.
    pub fn coarse_field<T: DeviceReal>(&self, vectors: usize) -> DeviceCoarse<T> {
        must(DeviceCoarse::zeros(self.coarse_sites, 2 * self.n, vectors)).on(&self.coarse)
    }

    fn check<T: DeviceReal>(&self, coarse: &DeviceCoarse<T>, fine: &DeviceSpinor<T>) {
        assert!(Arc::ptr_eq(&self.lat, &fine.lat) && fine.ls == self.ls);
        assert!((coarse.sites, coarse.dim) == (self.coarse_sites, 2 * self.n));
        assert_eq!(coarse.vectors, fine.vectors, "another number of vectors");
    }

    /// `coarse = P† fine`, or `P† ι fine` for a field on the sites of one parity (ι puts it
    /// onto the whole lattice, zero on the other sites).
    pub fn restrict<T: DeviceReal>(&self, coarse: &mut DeviceCoarse<T>, fine: &DeviceSpinor<T>) {
        self.restrict_leg(coarse, fine, None);
    }

    /// `restrict` of a field that is zero but where leg `leg` of the hops comes from another
    /// aggregate (`CudaMdwf::coarse_operator`): only those sites are read.
    pub(super) fn restrict_leg<T: DeviceReal>(
        &self,
        coarse: &mut DeviceCoarse<T>,
        fine: &DeviceSpinor<T>,
        leg: Option<usize>,
    ) {
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
                leg.map_or(-1, |l| l as i32),
                self.block.as_ptr(),
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

/// A coarse operator `P† D P` on the device, from its blocks and links (`CoarseOp`): with the
/// inverses of the blocks, which are made on the device, and so the even-odd form where the
/// coarse lattice has it.
pub struct DeviceCoarseOp<T = f64> {
    diag: DeviceBuffer<T>,
    links: DeviceBuffer<T>,
    inverse: DeviceBuffer<T>,
    /// The blocks in double precision, for what is made of them.
    diag_f64: DeviceBuffer<f64>,
    even_odd: bool,
    /// The site every hop of every site comes from, and whether each site is odd.
    neighbours: DeviceTable,
    odd: DeviceTable,
    /// The same on the host, for what is made of this operator (`adjoint`) without reading
    /// them back from the device.
    tables: (Vec<u32>, Vec<u32>),
    sites: usize,
    dim: usize,
    /// On several processes, the sites of the neighbouring processes the hops read.
    ghosts: Option<Rc<CoarseGhosts>>,
}

/// The precision and the number of vectors of the fields whose exchanges `CoarseGhosts` keeps.
type KeptKey = (std::any::TypeId, usize);

/// The ghosts of a coarse lattice cut between processes: for the hops of its sites on a face,
/// the sites of the neighbouring process across it, numbered from `sites` on in the tables of
/// the operator (`coarse_pass_kernel` reads them from a field of their own). A face's sites
/// are numbered by their three other coordinates (as the faces of the fine lattice), the same
/// on both sides: what arrives on face f is placed from `first[f]` on in that order.
pub(super) struct CoarseGhosts {
    lat: Arc<Lattice>,
    /// The local sites sent through each face, in the order of the face; none where the
    /// direction is not cut. The same on the host.
    send: Vec<Option<(DeviceTable, usize)>>,
    send_host: Vec<Vec<u32>>,
    first: [usize; 2 * ND],
    count: [usize; 2 * ND],
    pub(super) ghosts: usize,
    /// The global coordinates of every ghost.
    pub(super) ghost_coords: Vec<[usize; ND]>,
    /// The buffers of the exchanges and the ghosts, for each precision and number of vectors.
    kept: RefCell<Vec<(KeptKey, Box<dyn std::any::Any>)>>,
}

impl CoarseGhosts {
    /// The site every hop of every local site comes from (`[direction][site]`, directions as
    /// `comm::FACES`: forward 0..3, backward 0..3), ghosts from `sites` on, and the ghosts; no
    /// ghosts on one process.
    pub(super) fn tables(lat: &Arc<Lattice>) -> Result<(Vec<u32>, Option<Self>), CudaError> {
        let sites = lat.num_sites();
        let local = lat.local;
        let cut: [bool; ND] = core::array::from_fn(|mu| lat.grid[mu] > 1);
        let index = |c: [usize; ND]| c[0] + local[0] * (c[1] + local[1] * (c[2] + local[2] * c[3]));
        let face = |c: [usize; ND], mu: usize| {
            let d: Vec<usize> = (0..ND).filter(|&n| n != mu).collect();
            c[d[0]] + local[d[0]] * (c[d[1]] + local[d[1]] * c[d[2]])
        };
        let count: [usize; 2 * ND] = core::array::from_fn(|f| {
            if cut[f % ND] {
                sites / local[f % ND]
            } else {
                0
            }
        });
        let mut first = [0; 2 * ND];
        for f in 1..2 * ND {
            first[f] = first[f - 1] + count[f - 1];
        }
        let mut neighbours = vec![0u32; 2 * ND * sites];
        let mut send: Vec<Vec<u32>> = count.iter().map(|&n| vec![0; n]).collect();
        let mut ghost_coords = vec![[0; ND]; count.iter().sum()];
        for x in 0..sites {
            let c = lat.coords(x);
            for f in 0..2 * ND {
                let (mu, forward) = (f % ND, f < ND);
                let edge = if forward {
                    c[mu] + 1 == local[mu]
                } else {
                    c[mu] == 0
                };
                let y = if cut[mu] && edge {
                    let mut g = lat.global_coords(x);
                    g[mu] = if forward {
                        (g[mu] + 1) % lat.global[mu]
                    } else {
                        (g[mu] + lat.global[mu] - 1) % lat.global[mu]
                    };
                    ghost_coords[first[f] + face(c, mu)] = g;
                    sites + first[f] + face(c, mu)
                } else {
                    let mut n = c;
                    n[mu] = if forward {
                        (c[mu] + 1) % local[mu]
                    } else {
                        (c[mu] + local[mu] - 1) % local[mu]
                    };
                    index(n)
                };
                neighbours[f * sites + x] = y as u32;
                // The sites on a face go to the process across it.
                if cut[mu] && edge {
                    send[f][face(c, mu)] = x as u32;
                }
            }
        }
        if lat.comm.size() == 1 {
            return Ok((neighbours, None));
        }
        let send_host = send.clone();
        let send = send
            .iter()
            .map(|s| {
                (!s.is_empty())
                    .then(|| DeviceTable::from_host(s).map(|t| (t, s.len())))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;
        Ok((
            neighbours,
            Some(Self {
                lat: lat.clone(),
                send,
                send_host,
                first,
                count,
                ghosts: count.iter().sum(),
                ghost_coords,
                kept: RefCell::new(Vec::new()),
            }),
        ))
    }

    /// `per_site` reals of every local site (site after site), for the ghosts: what the
    /// neighbouring processes have at the sites the ghosts stand for, ghost after ghost.
    pub(super) fn of_host(&self, local: &[f64], per_site: usize) -> Vec<f64> {
        let neighbour = super::halo::neighbours(&self.lat);
        let spec = crate::comm::HaloSpec {
            neighbour,
            len: self.count.map(|n| n * per_site),
        };
        let mut ex = self.lat.comm.halo_f64(&spec);
        for (f, sites) in self.send_host.iter().enumerate() {
            let buf = ex.send_buffer(f);
            for (i, &x) in sites.iter().enumerate() {
                let x = x as usize;
                buf[i * per_site..(i + 1) * per_site]
                    .copy_from_slice(&local[x * per_site..(x + 1) * per_site]);
            }
        }
        ex.start();
        ex.wait();
        let mut ghosts = vec![0.0; self.ghosts * per_site];
        for f in 0..2 * ND {
            let n = self.count[f] * per_site;
            let at = self.first[f] * per_site;
            ghosts[at..at + n].copy_from_slice(ex.recv_buffer(f));
        }
        ghosts
    }

    /// The ghosts of `field`: its faces gathered, exchanged with the neighbouring processes,
    /// and placed, a coarse field of `ghosts` sites. The buffers are kept for each number of
    /// vectors and precision (a coarse solve exchanges hundreds of times).
    fn of<T: DeviceReal>(&self, field: &DeviceCoarse<T>) -> std::cell::RefMut<'_, DeviceBuffer<T>> {
        let (dim, vectors) = (field.dim, field.vectors);
        let reals = |n: usize| vectors * DeviceCoarse::<T>::reals_of(n.max(1), dim);
        let key = (std::any::TypeId::of::<T>(), vectors);
        let mut kept = self.kept.borrow_mut();
        let at = match kept.iter().position(|(k, _)| *k == key) {
            Some(at) => at,
            None => {
                let len = self.count.map(|n| if n > 0 { reals(n) } else { 0 });
                let made: (Faces<T>, DeviceBuffer<T>) = (
                    Faces::new(&self.lat, len),
                    must(DeviceBuffer::zeros(reals(self.ghosts))),
                );
                kept.push((key, Box::new(made)));
                kept.len() - 1
            }
        };
        let (faces, ghost) = kept[at]
            .1
            .downcast_mut::<(Faces<T>, DeviceBuffer<T>)>()
            .expect("kept for this precision");
        for (f, s) in self.send.iter().enumerate() {
            if let (Some((table, n)), Some(buf)) = (s, &faces.send[f]) {
                // SAFETY: the buffer is a coarse field of `n` sites of this dim and vectors,
                // the table `n` sites of `field`.
                unsafe {
                    T::coarse_gather(
                        buf.ptr,
                        field.data.ptr,
                        table.ptr,
                        *n as i32,
                        field.sites as i32,
                        dim as i32,
                        vectors as i32,
                    );
                }
            }
        }
        faces.exchange();
        for (f, buf) in faces.recv.iter().enumerate() {
            if let Some(buf) = buf {
                // SAFETY: what arrived is a coarse field of `count[f]` sites; the ghosts one of
                // `ghosts` sites, which places them from `first[f]` on.
                unsafe {
                    T::coarse_place(
                        ghost.ptr,
                        buf.ptr,
                        self.first[f] as i32,
                        self.count[f] as i32,
                        self.ghosts as i32,
                        dim as i32,
                        vectors as i32,
                    );
                }
            }
        }
        std::cell::RefMut::map(kept, |k| {
            &mut k[at]
                .1
                .downcast_mut::<(Faces<T>, DeviceBuffer<T>)>()
                .expect("kept for this precision")
                .1
        })
    }
}

/// The parities of the sites a pass acts on.
const ALL: i32 = -1;
const EVEN: i32 = 0;
const ODD: i32 = 1;

/// `src` in the precision `T`.
fn in_precision<T: DeviceReal>(src: &DeviceBuffer<f64>) -> Result<DeviceBuffer<T>, CudaError> {
    let mut dst = DeviceBuffer::<T>::uninitialised(src.len)?;
    convert(&mut dst, src);
    Ok(dst)
}

impl<T: DeviceReal> DeviceCoarseOp<T> {
    /// The operator `a` on the device. `even_odd`: with the even-odd form (`schur` and the
    /// like) if the coarse lattice has one: no odd extent.
    pub fn new<S: Storage>(
        a: &CoarseOp<S>,
        even_odd: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (sites, dim) = (a.lattice().num_sites(), a.dim());
        assert!(
            dim <= 1024,
            "a coarse site has at most 1024 complex numbers on the device"
        );
        let reals = |m: &[S]| -> Vec<f64> { m.iter().map(|x| x.load().to_f64()).collect() };
        let (diag, links) = a.blocks();
        debug_assert!({
            let table = crate::solver::preconditioner::multigrid::coarse_neighbours(a.lattice());
            (0..sites).all(|x| table[x] == a.neighbours(x))
        });
        Self::on_lattice(
            a.lattice(),
            DeviceBuffer::<f64>::from_host(&reals(diag))?,
            in_precision(&DeviceBuffer::<f64>::from_host(&reals(links))?)?,
            dim,
            even_odd,
        )
    }

    /// The operator of the blocks (double precision) and links (in the precision `T`), on the
    /// device in the layout of `coarse_pass_kernel`, of the coarse lattice `lat`, with its
    /// tables: the site every hop comes from and the parity of every site. `even_odd` as in
    /// `new`.
    pub(super) fn on_lattice(
        lat: &Arc<Lattice>,
        diag_f64: DeviceBuffer<f64>,
        links: DeviceBuffer<T>,
        dim: usize,
        even_odd: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let sites = lat.num_sites();
        let (neighbours, ghosts) = if lat.comm.size() == 1 {
            let mut neighbours = vec![0u32; 8 * sites];
            let table = crate::solver::preconditioner::multigrid::coarse_neighbours(lat);
            for (x, hops) in table.iter().enumerate() {
                for (d, y) in hops.iter().enumerate() {
                    neighbours[d * sites + x] = *y as u32;
                }
            }
            (neighbours, None)
        } else {
            CoarseGhosts::tables(lat)?
        };
        let odd: Vec<u32> = (0..sites)
            .map(|x| (lat.global_coords(x).iter().sum::<usize>() % 2) as u32)
            .collect();
        let mut op = Self::from_device_links(
            diag_f64,
            links,
            &neighbours,
            &odd,
            dim,
            even_odd && lat.global.iter().all(|l| l % 2 == 0),
        )?;
        op.ghosts = ghosts.map(Rc::new);
        Ok(op)
    }

    /// The operator of the blocks `diag_f64` and the links `links_f64` on the device, in the
    /// layout of `coarse_pass_kernel` (`multigrid/fold.cuh`), with the site every hop comes
    /// from (`[direction][site]`) and the parity of every site (1 odd), which is used if
    /// `even_odd`.
    pub(super) fn from_parts(
        diag_f64: DeviceBuffer<f64>,
        links_f64: &DeviceBuffer<f64>,
        neighbours: &[u32],
        odd: &[u32],
        dim: usize,
        even_odd: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert!(links_f64.len == 8 * diag_f64.len);
        Self::from_device_links(
            diag_f64,
            in_precision(links_f64)?,
            neighbours,
            odd,
            dim,
            even_odd,
        )
    }

    /// `from_parts` with the links in the precision `T`.
    pub(super) fn from_device_links(
        diag_f64: DeviceBuffer<f64>,
        links: DeviceBuffer<T>,
        neighbours: &[u32],
        odd: &[u32],
        dim: usize,
        even_odd: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let sites = odd.len();
        assert!(diag_f64.len == 2 * sites * dim * dim && links.len == 8 * diag_f64.len);
        assert_eq!(neighbours.len(), 8 * sites);
        // The inverses of the blocks, by Gauss-Jordan elimination in double precision.
        let inverse = DeviceBuffer::<f64>::uninitialised(diag_f64.len)?;
        let scratch = DeviceBuffer::<f64>::uninitialised(2 * diag_f64.len)?;
        let fail = DeviceTable::from_host(&vec![0; sites])?;
        // SAFETY: the blocks of `sites` sites, as many places for the inverses and for the
        // failures, and the scratch the kernel asks for.
        unsafe {
            bridge_cuda_blocks_inverse(
                inverse.ptr,
                diag_f64.ptr,
                scratch.ptr,
                fail.ptr,
                sites as i32,
                dim as i32,
                STREAM,
            );
        }
        finish();
        if let Some(coarse_site) = fail.download(sites)?.into_iter().position(|f| f != 0) {
            return Err(Box::new(MultigridError::SingularCoarseDiagonal {
                coarse_site,
            }));
        }
        Ok(Self {
            diag: in_precision(&diag_f64)?,
            links,
            inverse: in_precision(&inverse)?,
            diag_f64,
            even_odd,
            neighbours: DeviceTable::from_host(neighbours)?,
            odd: DeviceTable::from_host(odd)?,
            tables: (neighbours.to_vec(), odd.to_vec()),
            sites,
            dim,
            ghosts: None,
        })
    }

    pub(super) fn links(&self) -> &DeviceBuffer<T> {
        &self.links
    }

    /// The site every hop of every site comes from (`[direction][site]`), and whether each site
    /// is odd: the tables the device has, as they were made on the host.
    pub(super) fn tables(&self) -> (&[u32], &[u32]) {
        (&self.tables.0, &self.tables.1)
    }

    /// Complex numbers per site.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The ghosts of the lattice, if it is cut between processes.
    pub(super) fn ghosts(&self) -> Option<&Rc<CoarseGhosts>> {
        self.ghosts.as_ref()
    }

    /// The operator with the ghosts of its lattice (`CoarseGhosts::tables`, whose table it was
    /// made with).
    pub(super) fn with_ghosts(mut self, ghosts: Option<Rc<CoarseGhosts>>) -> Self {
        self.ghosts = ghosts;
        self
    }

    /// Sites of the lattice.
    pub fn sites(&self) -> usize {
        self.sites
    }

    /// The blocks in double precision.
    pub(super) fn diag_f64(&self) -> &DeviceBuffer<f64> {
        &self.diag_f64
    }

    /// Whether the even-odd form is there (`schur` and the like).
    pub fn has_even_odd(&self) -> bool {
        self.even_odd
    }

    /// The inverses of the blocks, for `apply_blocks`.
    pub fn block_inverses(&self) -> &DeviceBuffer<T> {
        &self.inverse
    }

    /// Checksums of the blocks with the links, and of the inverses of the blocks.
    pub fn checksums(&self) -> Result<(u64, u64), CudaError> {
        let parts = [self.diag.checksum()?, self.links.checksum()?];
        Ok((checksum_of(&parts), self.inverse.checksum()?))
    }

    /// |H_xx|⁻¹ = V |Λ|⁻¹ V† of H = A Γ5, the form of the operator MINRES works on, for
    /// `apply_blocks` (`BlockMap::abs_inverse_of_a_gamma5` on the CPU): by the Jacobi method
    /// in double precision.
    pub fn abs_inverse_of_a_gamma5(&self) -> Result<DeviceBuffer<T>, CudaError> {
        let out = DeviceBuffer::<f64>::uninitialised(self.diag_f64.len)?;
        let scratch = DeviceBuffer::<f64>::uninitialised(2 * self.diag_f64.len)?;
        // SAFETY: the blocks of the sites, as many places for the results, and the scratch the
        // kernel asks for.
        unsafe {
            bridge_cuda_blocks_abs_inverse_gamma5(
                out.ptr,
                self.diag_f64.ptr,
                scratch.ptr,
                self.sites as i32,
                self.dim as i32,
                STREAM,
            );
        }
        finish();
        in_precision(&out)
    }

    /// Matrices on the coarse sites, `[site][row][column]`, for `apply_blocks`.
    pub fn blocks(&self, blocks: &[Vec<C64>]) -> Result<DeviceBuffer<T>, CudaError> {
        assert!(
            blocks.len() == self.sites && blocks.iter().all(|b| b.len() == self.dim * self.dim)
        );
        let reals: Vec<f64> = blocks.iter().flatten().flat_map(|z| [z.re, z.im]).collect();
        DeviceBuffer::from_host(&reals)
    }

    /// One pass (`coarse_pass_kernel`): on the sites of `parity`,
    /// `out = [M] (base + A d ± Σ B h)`, the other sites zero.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        out: &mut DeviceCoarse<T>,
        parity: i32,
        base: Option<&DeviceCoarse<T>>,
        diag_of: Option<&DeviceCoarse<T>>,
        hops_of: Option<(&DeviceCoarse<T>, bool)>,
        matrices: Option<&DeviceBuffer<T>>,
        gamma5: bool,
    ) {
        assert!((out.sites, out.dim) == (self.sites, self.dim));
        // The kernel reads the matrices two columns at a time.
        assert!(
            self.dim.is_multiple_of(2),
            "coarse_pass needs an even dimension"
        );
        let field = |f: Option<&DeviceCoarse<T>>| match f {
            Some(f) => {
                assert!(f.same_shape(out));
                f.data.ptr.cast_const()
            }
            None => std::ptr::null(),
        };
        // On several processes the hops read the ghosts of the field they act on.
        if let (Some(ghosts), Some((h, minus))) = (&self.ghosts, hops_of) {
            let ghost = ghosts.of(h);
            let sizes = [
                self.sites as i32,
                self.dim as i32,
                parity,
                i32::from(minus),
                i32::from(gamma5),
                out.vectors as i32,
            ];
            // SAFETY: as below; the ghosts a coarse field of `ghosts` sites with the vectors of
            // `h`, which the tables index from `sites` on.
            unsafe {
                T::coarse_pass_ghost(
                    out.data.ptr,
                    field(base),
                    field(diag_of),
                    h.data.ptr.cast_const(),
                    self.diag.ptr,
                    self.links.ptr,
                    matrices.map_or(std::ptr::null(), |m| m.ptr.cast_const()),
                    self.neighbours.ptr,
                    self.odd.ptr,
                    sizes,
                    ghost.ptr.cast_const(),
                    ghosts.ghosts as i32,
                );
            }
            return;
        }
        // SAFETY: fields of the shape of `out`, which is none of them (it is borrowed
        // mutably), the matrices and tables of this lattice; null pointers are not read.
        unsafe {
            T::coarse_pass(
                out.data.ptr,
                field(base),
                field(diag_of),
                field(hops_of.map(|(h, _)| h)),
                self.diag.ptr,
                self.links.ptr,
                matrices.map_or(std::ptr::null(), |m| m.ptr.cast_const()),
                self.neighbours.ptr,
                self.odd.ptr,
                self.sites as i32,
                self.dim as i32,
                parity,
                i32::from(hops_of.is_some_and(|(_, minus)| minus)),
                i32::from(gamma5),
                out.vectors as i32,
            );
        }
    }

    /// `out = D_c inp`, or `Γ5 D_c Γ5 inp`, its adjoint.
    pub fn apply(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>, gamma5: bool) {
        self.pass(out, ALL, None, Some(inp), Some((inp, false)), None, gamma5);
    }

    /// `out = M inp` site by site, for matrices of `blocks`.
    pub fn apply_blocks(
        &self,
        out: &mut DeviceCoarse<T>,
        inp: &DeviceCoarse<T>,
        blocks: &DeviceBuffer<T>,
    ) {
        self.pass(out, ALL, Some(inp), None, None, Some(blocks), false);
    }

    /// `v = Γ5 v`: −1 on the second half of the components of a site.
    pub fn gamma5(&self, v: &mut DeviceCoarse<T>) {
        assert!((v.sites, v.dim) == (self.sites, self.dim));
        // SAFETY: a field of this lattice.
        unsafe {
            T::coarse_gamma5(
                v.data.ptr,
                self.sites as i32,
                self.dim as i32,
                v.vectors as i32,
            )
        };
    }

    /// `out = S inp`, the Schur complement on the even sites (`CoarseOp::schur`); `odd` is
    /// scratch.
    pub fn schur(
        &self,
        out: &mut DeviceCoarse<T>,
        inp: &DeviceCoarse<T>,
        odd: &mut DeviceCoarse<T>,
    ) {
        crate::fit(odd, inp);
        self.pass(
            odd,
            ODD,
            None,
            None,
            Some((inp, false)),
            Some(&self.inverse),
            false,
        );
        self.pass(out, EVEN, None, Some(inp), Some((odd, true)), None, false);
    }

    /// `out = g_e − B_eo A_oo⁻¹ g_o`, the right-hand side of the Schur system for `D_c c = g`.
    pub fn schur_source(
        &self,
        out: &mut DeviceCoarse<T>,
        g: &DeviceCoarse<T>,
        odd: &mut DeviceCoarse<T>,
    ) {
        crate::fit(odd, g);
        self.pass(odd, ODD, Some(g), None, None, Some(&self.inverse), false);
        self.pass(out, EVEN, Some(g), None, Some((odd, true)), None, false);
    }

    /// `c` holds c_e and gets `c_o = A_oo⁻¹ (g_o − B_oe c_e)` added.
    pub fn schur_solution(
        &self,
        c: &mut DeviceCoarse<T>,
        g: &DeviceCoarse<T>,
        odd: &mut DeviceCoarse<T>,
    ) {
        crate::fit(odd, g);
        self.pass(
            odd,
            ODD,
            Some(g),
            None,
            Some((c, true)),
            Some(&self.inverse),
            false,
        );
        c.axpy(&vec![C64::new(1.0, 0.0); c.vectors], odd);
    }

    /// `out = A_ee⁻¹ inp` on the even sites, zero on the odd ones.
    pub fn even_block_inverse(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        self.pass(out, EVEN, Some(inp), None, None, Some(&self.inverse), false);
    }
}

impl<T: DeviceReal> CoarseOperator<DeviceCoarse<T>> for DeviceCoarseOp<T> {
    fn has_even_odd(&self) -> bool {
        DeviceCoarseOp::has_even_odd(self)
    }
    fn mult(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        self.apply(out, inp, false);
    }
    fn mult_hermitian(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        self.apply(out, inp, true);
        DeviceCoarseOp::gamma5(self, out);
    }
    fn gamma5(&self, v: &mut DeviceCoarse<T>) {
        DeviceCoarseOp::gamma5(self, v);
    }
    fn schur(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>, odd: &mut DeviceCoarse<T>) {
        DeviceCoarseOp::schur(self, out, inp, odd);
    }
    fn schur_source(
        &self,
        out: &mut DeviceCoarse<T>,
        g: &DeviceCoarse<T>,
        odd: &mut DeviceCoarse<T>,
    ) {
        DeviceCoarseOp::schur_source(self, out, g, odd);
    }
    fn schur_solution(
        &self,
        c: &mut DeviceCoarse<T>,
        g: &DeviceCoarse<T>,
        odd: &mut DeviceCoarse<T>,
    ) {
        DeviceCoarseOp::schur_solution(self, c, g, odd);
    }
    fn even_part(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        self.pass(out, EVEN, Some(inp), None, None, None, false);
    }
    fn even_block_inverse(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        DeviceCoarseOp::even_block_inverse(self, out, inp);
    }
}

impl<T: DeviceReal> Transfer<DeviceSpinor<T>, DeviceCoarse<T>> for DeviceBasis {
    fn coarse_field(&self, vectors: usize) -> DeviceCoarse<T> {
        DeviceBasis::coarse_field(self, vectors)
    }
    fn restrict(&self, coarse: &mut DeviceCoarse<T>, fine: &DeviceSpinor<T>) {
        DeviceBasis::restrict(self, coarse, fine);
    }
    fn prolong(&self, fine: &mut DeviceSpinor<T>, coarse: &DeviceCoarse<T>) {
        DeviceBasis::prolong(self, fine, coarse);
    }
}
