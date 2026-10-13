//! The coarse lattice of the multigrid of domain-wall fermions on the device: its fields, the
//! prolongation P and the coarse operators (`multigrid/fold.cuh`).

use super::*;
use crate::fermion::{Dagger, EvenOdd, FermionOp, SplitParity, Variant};
use crate::kernel::{Dirac, Gamma, GammaRepre, Number, Real, Unit};
use crate::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp, MultigridError};
use crate::Parity;
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

    /// The first `len` numbers.
    fn download(&self, len: usize) -> Result<Vec<u32>, CudaError> {
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

    /// Vectors that are 1 in component `first + v` of every site (vector v) and zero elsewhere.
    pub(super) fn units(
        sites: usize,
        dim: usize,
        first: usize,
        vectors: usize,
    ) -> Result<Self, CudaError> {
        let one = Self::reals_of(sites, dim);
        let mut host = vec![0.0; vectors * one];
        for v in 0..vectors {
            for x in 0..sites {
                host[v * one + at(0, x * dim + first + v, 2)] = 1.0;
            }
        }
        Ok(Self {
            data: DeviceBuffer::from_host(&host)?,
            sites,
            dim,
            vectors,
        })
    }

    /// The complex numbers of the field, `[vector][site][component]`.
    pub(super) fn values(&self) -> Result<Vec<C64>, CudaError> {
        let (count, one) = (self.sites * self.dim, Self::reals_of(self.sites, self.dim));
        let host = self.data.download()?;
        Ok((0..self.vectors * count)
            .map(|k| {
                let (v, q) = (k / count, k % count);
                C64::new(host[v * one + at(0, q, 2)], host[v * one + at(1, q, 2)])
            })
            .collect())
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
    pub(super) coarse_sites: usize,
    block_volume: usize,
    /// Sites of an aggregate in every direction.
    pub(super) block: [i32; ND],
    /// The coarse lattice.
    pub(super) coarse: Arc<Lattice>,
    dims: [i32; ND],
    /// γ5 of the fold: the column and the sign of the entry of every row.
    gamma: [i32; 8],
}

impl DeviceBasis {
    /// The P of the CPU, `basis`.
    pub fn new<T: Storage>(basis: &Basis<T, f32>) -> Result<Self, CudaError> {
        let g = basis
            .reflection()
            .expect("a basis folded on the fifth dimension");
        Self::with(
            basis.aggregation(),
            basis.fine_reals() / SPINOR,
            basis.num_test_vectors(),
            g,
            DeviceBuffer::upload(basis.stored())?,
        )
    }

    /// A P to be made on the device, for `n` test vectors with `ls` spinors per site on the
    /// aggregates `agg`: zero until every test vector was given (`add`) and `finish` called.
    pub fn empty(agg: &Arc<Aggregation>, ls: usize, n: usize) -> Result<Self, CudaError> {
        assert!(
            ls.is_multiple_of(2),
            "folding needs an even fifth dimension"
        );
        let reals = agg.fine_lattice().num_sites() * n * 2 * 12 * ls;
        Self::with(agg, ls, n, Dirac::GAMMA5, DeviceBuffer::zeros(reals)?)
    }

    fn with(
        agg: &Arc<Aggregation>,
        ls: usize,
        n: usize,
        g: Gamma,
        p: DeviceBuffer<f32>,
    ) -> Result<Self, CudaError> {
        let lat = agg.fine_lattice().clone();
        let coarse_sites = agg.coarse_lattice().num_sites();
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
    /// on the CPU (modified Gram-Schmidt in double precision on the parts as they are kept).
    pub fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (n, kept) = (self.n, 12 * self.ls);
        let dim = self.block_volume * kept / 2;
        if n > dim {
            return Err(
                format!("{n} test vectors do not fit into aggregates of dimension {dim}").into(),
            );
        }
        // The vectors of an aggregate and chirality in double precision: as many coarse sites
        // at a time as keep that within half a GB.
        let per_site = 2 * n * dim * 2;
        let count = ((1 << 26) / per_site).clamp(1, self.coarse_sites);
        let scratch = DeviceBuffer::<f64>::uninitialised(count * per_site)?;
        let fail = DeviceTable::from_host(&vec![0; 2 * count])?;
        for first in (0..self.coarse_sites).step_by(count) {
            let sites = count.min(self.coarse_sites - first);
            // SAFETY: P and the sites of this basis, scratch and places for the results of
            // `sites` coarse sites.
            unsafe {
                bridge_cuda_fold_orthonormalise(
                    self.p.ptr,
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
            }
            finish();
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
    pub fn pv_eo(
        op: &CudaMdwf,
        pv: &CudaMdwf,
        agg: &Arc<Aggregation>,
        y: &DeviceSpinor,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(y.parity, Some(Parity::Even));
        let (n, ls, lat) = (y.vectors, y.ls, &y.lat);
        let mut basis = Self::empty(agg, ls, 2 * n)?;
        let (even, odd) = (Parity::Even, Parity::Odd);
        let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
        let mut full_work = FermionOp::work(op);
        // A few vectors at a time: the fields on the whole lattice are large.
        let all: Vec<usize> = (0..n).collect();
        for chunk in all.chunks(4) {
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
            basis.add(n + chunk[0], &s);
        }
        basis.finish()?;
        Ok(basis)
    }

    /// P as it is kept (`Basis::stored`), from the device.
    pub fn stored(&self) -> Result<Vec<f64>, CudaError> {
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

/// A coarse operator `P† D P` on the device (`CoarseOp`, from which it is made): its blocks
/// and links, and for the even-odd form the inverses of the blocks.
pub struct DeviceCoarseOp<T = f64> {
    diag: DeviceBuffer<T>,
    links: DeviceBuffer<T>,
    inverse: Option<DeviceBuffer<T>>,
    /// The site every hop of every site comes from, and whether each site is odd.
    neighbours: DeviceTable,
    odd: DeviceTable,
    sites: usize,
    dim: usize,
}

/// The parities of the sites a pass acts on.
const ALL: i32 = -1;
const EVEN: i32 = 0;
const ODD: i32 = 1;

impl<T: DeviceReal> DeviceCoarseOp<T> {
    /// The operator `a` on the device; with its even-odd form if `a` has it
    /// (`CoarseOp::prepare_even_odd`).
    pub fn new<S: Storage>(a: &CoarseOp<S>) -> Result<Self, CudaError> {
        let (sites, dim) = (a.lattice().num_sites(), a.dim());
        assert!(
            dim <= 1024,
            "a coarse site has at most 1024 complex numbers on the device"
        );
        let reals = |m: &[S]| -> Vec<f64> { m.iter().map(|x| x.load().to_f64()).collect() };
        let (diag, links) = a.blocks();
        let mut neighbours = vec![0u32; 8 * sites];
        for x in 0..sites {
            for (d, y) in a.neighbours(x).into_iter().enumerate() {
                neighbours[d * sites + x] = y as u32;
            }
        }
        let coords = |x| a.lattice().global_coords(x);
        let odd: Vec<u32> = (0..sites)
            .map(|x| (coords(x).iter().sum::<usize>() % 2) as u32)
            .collect();
        Ok(Self {
            diag: DeviceBuffer::from_host(&reals(diag))?,
            links: DeviceBuffer::from_host(&reals(links))?,
            inverse: match a.block_inverses() {
                Some(inverse) => Some(DeviceBuffer::from_host(&reals(inverse))?),
                None => None,
            },
            neighbours: DeviceTable::from_host(&neighbours)?,
            odd: DeviceTable::from_host(&odd)?,
            sites,
            dim,
        })
    }

    /// Whether the even-odd form is there (`schur` and the like).
    pub fn has_even_odd(&self) -> bool {
        self.inverse.is_some()
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
        let field = |f: Option<&DeviceCoarse<T>>| match f {
            Some(f) => {
                assert!(f.same_shape(out));
                f.data.ptr.cast_const()
            }
            None => std::ptr::null(),
        };
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

    fn inverse(&self) -> &DeviceBuffer<T> {
        self.inverse.as_ref().expect("no even-odd form")
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
            Some(self.inverse()),
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
        self.pass(odd, ODD, Some(g), None, None, Some(self.inverse()), false);
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
            Some(self.inverse()),
            false,
        );
        c.axpy(&vec![C64::new(1.0, 0.0); c.vectors], odd);
    }

    /// `out = A_ee⁻¹ inp` on the even sites, zero on the odd ones.
    pub fn even_block_inverse(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        self.pass(
            out,
            EVEN,
            Some(inp),
            None,
            None,
            Some(self.inverse()),
            false,
        );
    }
}
