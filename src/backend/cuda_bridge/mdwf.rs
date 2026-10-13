//! The Möbius domain-wall operator on the device.
//!
//! In the notation of `fermion::mdwf`, with H the hops of the Wilson kernel times (4 − M5),
//! the plain operator is `D = C + H B`, `D† = C† + B† H†`, with the part within a site
//! `C = d+ + d− Y` and `B = b − c Y`. With Neff's α (`set_alpha`) the operator is `D Q`.
//!
//! C, B, Q, their inverses, adjoints and products act within a site, on the fifth dimension
//! and the chirality only: each is a matrix (`FifthMatrix`), the products are made on the
//! host, and one matrix product on the device applies any of them (`gemm`; in double precision
//! on the tensor cores). So `(C Q)⁻¹` is one pass, and so is `(C Q)†⁻¹ (B Q)†` after the
//! adjoint hops.

use super::wilson::{gamma5, scratch};
use super::*;
use crate::fermion::{Dagger, EvenOdd, FermionOp, MdwfParams, Then, Variant};
use crate::kernel::NC;
use crate::solver::preconditioner::multigrid::CoarseOp;
use crate::{Parity, Vector, C64};

extern "C" {
    fn bridge_cuda_fifth_gemm_tf32_f32(
        out: *mut f32,
        inp: *const f32,
        from: *const f32,
        matrix: *const f32,
        sites: i32,
        ls: i32,
        stream: *mut c_void,
    );
}

/// An operator within a site that acts on the fifth dimension and the chirality:
/// `Σ_s' (plus_ss' P+ + minus_ss' P−)`, the two Ls × Ls matrices row after row. P± are
/// orthogonal projectors that commute with the fifth dimension, so such operators multiply,
/// invert and take adjoints matrix by matrix.
#[derive(Clone)]
struct FifthMatrix {
    ls: usize,
    plus: Vec<f64>,
    minus: Vec<f64>,
}

impl FifthMatrix {
    /// `a + y Y` at mass `mass`: `(Y ψ)_s = −P− ψ_{s+1} − P+ ψ_{s−1}`, with +m in the place of
    /// −1 across the walls. Y takes P+ from below and P− from above.
    fn linear(ls: usize, a: f64, y: f64, mass: f64) -> Self {
        let (mut plus, mut minus) = (vec![0.0; ls * ls], vec![0.0; ls * ls]);
        for s in 0..ls {
            plus[s * ls + s] = a;
            minus[s * ls + s] = a;
            let (below, above) = ((s + ls - 1) % ls, (s + 1) % ls);
            plus[s * ls + below] += y * if s == 0 { mass } else { -1.0 };
            minus[s * ls + above] += y * if s == ls - 1 { mass } else { -1.0 };
        }
        Self { ls, plus, minus }
    }

    /// Neff's Q: Q_0 = P− + α P+, Q_s = α inside, Q_(Ls−1) = P+ + α P−.
    fn q(ls: usize, alpha: f64) -> Self {
        let (mut plus, mut minus) = (vec![0.0; ls * ls], vec![0.0; ls * ls]);
        for s in 0..ls {
            plus[s * ls + s] = if s == ls - 1 { 1.0 } else { alpha };
            minus[s * ls + s] = if s == 0 { 1.0 } else { alpha };
        }
        Self { ls, plus, minus }
    }

    /// `self · other`.
    fn times(&self, other: &Self) -> Self {
        let ls = self.ls;
        let product = |a: &[f64], b: &[f64]| {
            let mut c = vec![0.0; ls * ls];
            for i in 0..ls {
                for k in 0..ls {
                    for j in 0..ls {
                        c[i * ls + j] += a[i * ls + k] * b[k * ls + j];
                    }
                }
            }
            c
        };
        Self {
            ls,
            plus: product(&self.plus, &other.plus),
            minus: product(&self.minus, &other.minus),
        }
    }

    fn adjoint(&self) -> Self {
        let ls = self.ls;
        let transpose =
            |a: &[f64]| -> Vec<f64> { (0..ls * ls).map(|k| a[(k % ls) * ls + k / ls]).collect() };
        Self {
            ls,
            plus: transpose(&self.plus),
            minus: transpose(&self.minus),
        }
    }

    /// The inverse, by Gauss-Jordan elimination with pivoting; a zero pivot is an operator
    /// that is singular within a site.
    fn inverse(&self) -> Self {
        let ls = self.ls;
        let invert = |a: &[f64]| {
            let mut a = a.to_vec();
            let mut inv = vec![0.0; ls * ls];
            for i in 0..ls {
                inv[i * ls + i] = 1.0;
            }
            for k in 0..ls {
                let pivot = (k..ls)
                    .max_by(|&i, &j| a[i * ls + k].abs().total_cmp(&a[j * ls + k].abs()))
                    .expect("at least one row");
                assert!(
                    a[pivot * ls + k] != 0.0,
                    "the part of the operator within a site is singular"
                );
                for j in 0..ls {
                    a.swap(k * ls + j, pivot * ls + j);
                    inv.swap(k * ls + j, pivot * ls + j);
                }
                let scale = 1.0 / a[k * ls + k];
                for j in 0..ls {
                    a[k * ls + j] *= scale;
                    inv[k * ls + j] *= scale;
                }
                for i in (0..ls).filter(|&i| i != k) {
                    let factor = a[i * ls + k];
                    for j in 0..ls {
                        a[i * ls + j] -= factor * a[k * ls + j];
                        inv[i * ls + j] -= factor * inv[k * ls + j];
                    }
                }
            }
            inv
        };
        Self {
            ls,
            plus: invert(&self.plus),
            minus: invert(&self.minus),
        }
    }

    /// The matrix the kernel takes (`fifth_gemm.cuh`), on the device: on the rows 2 s + h of a
    /// block, h = 0 the upper and 1 the lower spin components. γ5 exchanges the two, so
    /// P± = (1 ± γ5)/2 give (plus + minus)/2 between rows of the same h and (plus − minus)/2
    /// between the others. Rows padded to a multiple of 8, columns to a multiple of 4.
    fn on_device<T: DeviceReal>(&self) -> Result<DeviceBuffer<T>, CudaError> {
        let ls = self.ls;
        let (rows, columns) = ((2 * ls).div_ceil(8) * 8, (2 * ls).div_ceil(4) * 4);
        let mut m = vec![0.0; rows * columns];
        for i in 0..ls {
            for j in 0..ls {
                let (p, q) = (self.plus[i * ls + j], self.minus[i * ls + j]);
                for (h, k) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    m[(2 * i + h) * columns + 2 * j + k] = 0.5 * if h == k { p + q } else { p - q };
                }
            }
        }
        DeviceBuffer::from_host(&m)
    }
}

/// The matrices of an operator with Neff's α, on the device. `[_; 2]`: the operator and its
/// adjoint (`Dagger::No`, `Dagger::Yes`).
struct Matrices<T> {
    /// The part within a site, `C Q`.
    diag: [DeviceBuffer<T>; 2],
    /// Its inverse.
    inv: [DeviceBuffer<T>; 2],
    /// `B Q`, what the hops act on. Without α the hop kernel makes `B ψ` as it reads.
    pre: Option<DeviceBuffer<T>>,
    /// `(B Q)†`, what follows the adjoint hops.
    post: DeviceBuffer<T>,
    /// `(C Q)†⁻¹ (B Q)†`: the adjoint hop followed by the inverse within the site.
    inv_post: DeviceBuffer<T>,
    /// Q and Q⁻¹.
    q: [DeviceBuffer<T>; 2],
}

impl<T: DeviceReal> Matrices<T> {
    fn new(params: &MdwfParams, alpha: f64) -> Result<Self, CudaError> {
        let MdwfParams {
            ls, m5, mass, b, c, ..
        } = *params;
        let height = 4.0 - m5;
        let q = FifthMatrix::q(ls, alpha);
        let diag = FifthMatrix::linear(ls, 1.0 + b * height, 1.0 - c * height, mass).times(&q);
        let pre = FifthMatrix::linear(ls, b, -c, mass).times(&q);
        let (inv, post) = (diag.inverse(), pre.adjoint());
        Ok(Self {
            diag: [diag.on_device()?, diag.adjoint().on_device()?],
            inv: [inv.on_device()?, inv.adjoint().on_device()?],
            pre: if alpha == 1.0 {
                None
            } else {
                Some(pre.on_device()?)
            },
            inv_post: inv.adjoint().times(&post).on_device()?,
            post: post.on_device()?,
            q: [q.on_device()?, q.inverse().on_device()?],
        })
    }
}

pub struct CudaMdwf<T = f64> {
    params: MdwfParams,
    geometry: Geometry,
    gauge: DeviceGauge<T>,
    /// Neff's α (1: the plain operator).
    alpha: f64,
    matrices: Matrices<T>,
    /// The matrix products in TF32 (`set_tf32`); single precision only.
    tf32: bool,
}

impl CudaMdwf<f32> {
    /// The matrix products of the fifth dimension on the tensor cores, which in single
    /// precision is TF32: the matrix and the field rounded to ten bits of mantissa, a relative
    /// error of 1e-3. For an operator whose result is corrected afterwards (a smoother, the
    /// inner solve of a mixed-precision one), never for one whose result is kept. The hops
    /// with a product in the same pass (`EvenOdd::hop_then`) stay exact.
    pub fn set_tf32(&mut self, on: bool) {
        self.tf32 = on;
    }
}

/// Scratch fields of `CudaMdwf`, made on first use.
pub type Work<T> = [Option<DeviceSpinor<T>>; 2];

impl<T: DeviceReal> CudaMdwf<T> {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams) -> Result<Self, CudaError> {
        assert!(
            (2..=MAX_LS).contains(&params.ls),
            "the fifth dimension must have between 2 and {MAX_LS} sites"
        );
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, params.ls),
            gauge: DeviceGauge::new(lat)?,
            alpha: 1.0,
            matrices: Matrices::new(&params, 1.0)?,
            tf32: false,
        })
    }

    /// Neff's form D^(α) = D Q; 1 is the plain operator.
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(alpha > 0.0, "α must be positive");
        self.alpha = alpha;
        self.matrices = must(Matrices::new(&self.params, alpha));
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// `out = M inp`, or `from − M inp`, for a matrix of `Matrices`: one pass, on the whole
    /// lattice or on one parity, of all the vectors (which are whole blocks of the device
    /// layout one after the other, so the kernel sees more blocks).
    fn gemm(
        &self,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        matrix: &DeviceBuffer<T>,
        from: Option<&DeviceSpinor<T>>,
    ) {
        assert!(out.same_shape(inp) && inp.ls == self.params.ls);
        out.parity = inp.parity;
        let from = match from {
            Some(f) => {
                assert!(f.same_shape(inp) && f.parity == inp.parity);
                f.data.ptr.cast_const()
            }
            None => std::ptr::null(),
        };
        let sites = (inp.vectors * padded(inp.sites())) as i32;
        if self.tf32 {
            // SAFETY: as below; `tf32` is set on an operator in single precision only, so the
            // reals are `f32`.
            unsafe {
                bridge_cuda_fifth_gemm_tf32_f32(
                    out.data.ptr.cast(),
                    inp.data.ptr.cast(),
                    from.cast(),
                    matrix.ptr.cast_const().cast(),
                    sites,
                    inp.ls as i32,
                    STREAM,
                );
            }
            return;
        }
        // SAFETY: the buffers hold the same sites with `ls` spinors each (padded to blocks),
        // `out` is none of the inputs (it is borrowed mutably), the matrix is the padded one
        // of this `ls`, and a null `from` is not read.
        unsafe {
            T::fifth_gemm(
                out.data.ptr,
                inp.data.ptr,
                from,
                matrix.ptr,
                sites,
                inp.ls as i32,
            );
        }
    }

    /// `out = Q inp`, or `Q⁻¹ inp`: Q_0 = P− + α P+, Q_s = α inside, Q_(Ls−1) = P+ + α P−.
    /// The solution of D x = b is x = Q x^(α) for D^(α) x^(α) = b.
    pub fn mult_q(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, inverse: bool) {
        self.gemm(out, inp, &self.matrices.q[usize::from(inverse)], None);
    }

    /// `out = [from −] H (b − c Y) inp`, or with `H†`, for the given b and c: on the whole
    /// lattice, or `out` on the sites of parity `to` and `inp` on the others.
    fn hops(
        &self,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        from: Option<&DeviceSpinor<T>>,
        (b, c): (f64, f64),
        dagger: bool,
        to: Option<Parity>,
    ) {
        let fifth = FifthInHops {
            mass: self.params.mass,
            b,
            c,
        };
        hops(
            (&self.gauge, &self.geometry),
            out,
            inp,
            from,
            0.5,
            fifth,
            dagger,
            to,
        );
    }

    /// `out = [from −] M H (b − c Y) inp`, or with `H†`, M a matrix of `Matrices`: the hops and
    /// the product in one pass, the result of the hops in the shared memory of the kernel;
    /// through `scratch` in two if the spinors of a block do not fit there.
    #[allow(clippy::too_many_arguments)]
    fn hops_then(
        &self,
        scratch: &mut DeviceSpinor<T>,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        from: Option<&DeviceSpinor<T>>,
        (b, c): (f64, f64),
        dagger: bool,
        to: Parity,
        matrix: &DeviceBuffer<T>,
    ) {
        let ls = self.params.ls;
        assert!(inp.is(out, ls, Some(to.flip())) && out.ls == ls);
        assert_eq!(out.data.len, inp.data.len);
        out.parity = Some(to);
        let from_ptr = match from {
            Some(f) => {
                assert!(f.is(out, ls, Some(to)) && f.data.len == out.data.len);
                f.data.ptr.cast_const()
            }
            None => std::ptr::null(),
        };
        // SAFETY: the spinor buffers hold the sites of one parity each, as the kernel is told;
        // the links are those of the whole lattice; `out` is not `inp`; the matrix is the
        // padded one of this `ls`; a null `from` is not read.
        let done = unsafe {
            T::hop_gemm(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                from_ptr,
                matrix.ptr,
                &self.geometry,
                self.params.mass,
                b,
                c,
                i32::from(dagger),
                to.index() as i32,
                inp.vectors as i32,
            )
        };
        if !done {
            self.hops(scratch, inp, None, (b, c), dagger, Some(to));
            self.gemm(out, scratch, matrix, from);
        }
    }

    /// `out = [from −] H B Q inp`: the hops of the operator onto `to` (the whole lattice for
    /// `None`). Without α one pass, B made as the hops read; with α through `t`.
    fn hop_forward(
        &self,
        t: &mut Option<DeviceSpinor<T>>,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        from: Option<&DeviceSpinor<T>>,
        to: Option<Parity>,
    ) {
        match &self.matrices.pre {
            None => self.hops(out, inp, from, (self.params.b, self.params.c), false, to),
            Some(pre) => {
                let t = scratch(t, inp);
                self.gemm(t, inp, pre, None);
                self.hops(out, t, from, (1.0, 0.0), false, to);
            }
        }
    }

    /// The five-dimensional right-hand side for a four-dimensional source η (`eta`, one spinor
    /// per site): `out = D₋ b`, b = P+ η at s = 0 and P− η at s = Ls−1, D₋ = 1 − c W, as
    /// `Mdwf::source_5d`. With H the hops, `D₋ b = (1 − c (4 − M5)) b − c H b`.
    pub fn source_5d(&self, work: &mut Work<T>, out: &mut DeviceSpinor<T>, eta: &DeviceSpinor<T>) {
        let MdwfParams { ls, m5, c, .. } = self.params;
        assert!(eta.is(out, 1, None) && out.ls == ls && out.vectors == eta.vectors);
        out.parity = None;
        let [b, from] = work;
        let (b, from) = (scratch(b, out), scratch(from, out));
        b.set_zero();
        b.parity = None;
        // SAFETY: a field of one spinor per site and one of `ls`, on the same lattice with as
        // many vectors.
        unsafe {
            T::walls(
                b.data.ptr,
                eta.data.ptr,
                eta.sites() as i32,
                ls as i32,
                ls as i32 - 1,
                0,
                0,
                eta.vectors as i32,
            );
        }
        from.copy_from(b);
        from.scale(&vec![1.0 - c * (4.0 - m5); from.vectors]);
        self.hops(out, b, Some(from), (c, 0.0), false, None);
    }

    /// `out = P− ψ_minus + P+ ψ_plus`, a field of one spinor per site.
    fn walls_to_4d(
        &self,
        out: &mut DeviceSpinor<T>,
        psi: &DeviceSpinor<T>,
        (minus, plus): (usize, usize),
    ) {
        let ls = self.params.ls;
        assert!(psi.is(out, ls, None) && out.is(psi, 1, None) && out.vectors == psi.vectors);
        // SAFETY: as in `source_5d`.
        unsafe {
            T::walls(
                out.data.ptr,
                psi.data.ptr,
                psi.sites() as i32,
                ls as i32,
                minus as i32,
                plus as i32,
                1,
                psi.vectors as i32,
            );
        }
    }

    /// The four-dimensional quark field `q = P− ψ_0 + P+ ψ_(Ls−1)` of a solution ψ of the plain
    /// operator (`x = Q x^(α)` for D^(α)).
    pub fn propagator_4d(&self, out: &mut DeviceSpinor<T>, psi: &DeviceSpinor<T>) {
        self.walls_to_4d(out, psi, (0, self.params.ls - 1));
    }

    /// The field at the middle of the fifth dimension, `P− ψ_(Ls/2) + P+ ψ_(Ls/2−1)`.
    pub fn midpoint_4d(&self, out: &mut DeviceSpinor<T>, psi: &DeviceSpinor<T>) {
        let ls = self.params.ls;
        assert!(ls.is_multiple_of(2), "the midpoint needs an even Ls");
        self.walls_to_4d(out, psi, (ls / 2, ls / 2 - 1));
    }

    /// The coarse operator `P† D P` of this operator (plain: without Neff's α), as
    /// `CoarseOp::build` makes it on the CPU: for every component of a coarse site, P of the
    /// unit vectors of that component, the part of D within a site and each of the eight
    /// hops, sorted by whether the hop stays inside its aggregate, and P† of each. Several
    /// components at a time, as the vectors of the fields.
    pub fn coarse_operator(&self, basis: &DeviceBasis) -> Result<CoarseOp<f64>, CudaError> {
        assert!(
            self.alpha == 1.0,
            "the coarse operator is that of the plain operator"
        );
        let MdwfParams { mass, b, c, .. } = self.params;
        let (sites, dim) = (basis.coarse_sites, basis.coarse_dim());
        let matrix = 2 * dim * dim;
        let mut diag = vec![0.0; sites * matrix];
        let mut links = vec![0.0; 8 * sites * matrix];
        // As many components at a time as keep the three fine fields within half a GB: whole
        // groups of the vectors P is read once for, if that is several.
        let mut phi: DeviceSpinor<T> = basis.fine_field(1)?;
        let fit = ((1 << 29) / (3 * phi.data.bytes())).clamp(1, dim);
        let batch = if fit >= 8 { fit / 8 * 8 } else { fit };
        // Column `first + v` of every site's matrix from vector v of a coarse field.
        let store = |target: &mut [f64], first: usize, values: &[C64]| {
            for (v, part) in values.chunks_exact(sites * dim).enumerate() {
                for (site, rows) in part.chunks_exact(dim).enumerate() {
                    for (row, z) in rows.iter().enumerate() {
                        let k = site * matrix + 2 * (dim * row + first + v);
                        target[k] = z.re;
                        target[k + 1] = z.im;
                    }
                }
            }
        };
        for first in (0..dim).step_by(batch) {
            let nv = batch.min(dim - first);
            let unit = DeviceCoarse::<T>::units(sites, dim, first, nv)?;
            if phi.vectors != nv {
                phi = phi.zeros_with(nv);
            }
            let (mut inside, cross) = (phi.zeros_like(), phi.zeros_like());
            let mut restricted = unit.zeros_like();
            basis.prolong(&mut phi, &unit);
            self.gemm(&mut inside, &phi, &self.matrices.diag[0], None);
            for leg in 0..8 {
                // SAFETY: three fields on the whole lattice with `ls` spinors per site and as
                // many vectors, the links of the lattice; `block` is copied.
                unsafe {
                    T::hop_split(
                        cross.data.ptr,
                        inside.data.ptr,
                        phi.data.ptr,
                        self.gauge.links.ptr,
                        &self.geometry,
                        0.5,
                        mass,
                        b,
                        c,
                        leg as i32,
                        basis.block.as_ptr(),
                        nv as i32,
                    );
                }
                basis.restrict_leg(&mut restricted, &cross, Some(leg));
                let range = leg * sites * matrix..(leg + 1) * sites * matrix;
                store(&mut links[range], first, &restricted.values()?);
            }
            basis.restrict(&mut restricted, &inside);
            store(&mut diag, first, &restricted.values()?);
        }
        Ok(CoarseOp::from_parts(&basis.coarse, dim, diag, links))
    }

    /// `out = D inp` of the plain operator, without Neff's Q whatever α is, on the whole lattice
    /// in one pass and with no field besides the two.
    pub fn mult_plain(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>) {
        self.one_pass(out, inp);
    }

    /// `out = D inp` of the plain operator on the whole lattice, in one pass.
    fn one_pass(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>) {
        let ls = self.params.ls;
        assert!(inp.is(out, ls, None) && out.is(inp, ls, None) && out.same_shape(inp));
        let params = MdwfKernelParams {
            m5: self.params.m5,
            mass: self.params.mass,
            alpha: 1.0,
            b5: self.params.b,
            c5: self.params.c,
        };
        // SAFETY: as in `CudaWilson::apply`, the spinor buffers holding `ls` spinors per site.
        unsafe {
            T::mdwf_dslash(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                &self.geometry,
                &params,
                inp.vectors as i32,
            );
        }
    }
}

/// The even-odd blocks of D Q, in four dimensions (all `ls` spinors of a site have its
/// parity): the hops `H_pq B Q` and the part within a site `C Q`.
impl<T: DeviceReal> EvenOdd<DeviceSpinor<T>> for CudaMdwf<T> {
    type Work = Work<T>;

    fn even_odd_work(&self) -> Self::Work {
        [None, None]
    }

    fn hop(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        to: Parity,
        dagger: Dagger,
    ) {
        let [h, t] = work;
        match dagger {
            Dagger::No => self.hop_forward(t, out, inp, None, Some(to)),
            // ((B Q)†) H†
            Dagger::Yes => {
                let h = scratch(h, out);
                self.hops_then(h, out, inp, None, (1.0, 0.0), true, to, &self.matrices.post);
            }
        }
    }

    fn diag(
        &self,
        _work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        let m = &self.matrices.diag[usize::from(dagger == Dagger::Yes)];
        self.gemm(out, inp, m, None);
    }

    /// Exactly: the inverse of `C Q` in the fifth dimension is a matrix.
    fn diag_inv(
        &self,
        _work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        let m = &self.matrices.inv[usize::from(dagger == Dagger::Yes)];
        self.gemm(out, inp, m, None);
    }

    /// The hops, then one matrix product for all that follows them within the site: `(C Q)⁻¹`,
    /// or after the adjoint hops `(B Q)†` and `(C Q)†⁻¹` as one matrix, with the subtraction.
    fn hop_then(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        to: Parity,
        dagger: Dagger,
        then: Then<'_, DeviceSpinor<T>>,
        scratch: &mut DeviceSpinor<T>,
    ) {
        let [_, t] = work;
        scratch.parity = Some(to);
        match (dagger, then.site_inv) {
            (Dagger::No, false) => self.hop_forward(t, out, inp, then.from, Some(to)),
            (Dagger::No, true) => {
                let inv = &self.matrices.inv[0];
                match &self.matrices.pre {
                    None => {
                        let bc = (self.params.b, self.params.c);
                        self.hops_then(scratch, out, inp, then.from, bc, false, to, inv);
                    }
                    Some(pre) => {
                        let t = super::wilson::scratch(t, inp);
                        self.gemm(t, inp, pre, None);
                        self.hops_then(scratch, out, t, then.from, (1.0, 0.0), false, to, inv);
                    }
                }
            }
            (Dagger::Yes, site_inv) => {
                let m = if site_inv {
                    &self.matrices.inv_post
                } else {
                    &self.matrices.post
                };
                self.hops_then(scratch, out, inp, then.from, (1.0, 0.0), true, to, m);
            }
        }
    }
}

impl<T: DeviceReal> FermionOp<DeviceSpinor<T>> for CudaMdwf<T> {
    type Work = Work<T>;

    fn work(&self) -> Self::Work {
        [None, None]
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        self.gauge.set(u, &mut self.geometry)
    }

    fn mult(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        variant: Variant,
    ) {
        let [h, t] = work;
        let has_alpha = self.alpha != 1.0;
        // D Q
        let forward = |t: &mut Option<DeviceSpinor<T>>, out: &mut DeviceSpinor<T>| {
            if has_alpha {
                let t = scratch(t, inp);
                self.mult_q(t, inp, false);
                self.one_pass(out, t);
            } else {
                self.one_pass(out, inp);
            }
        };
        match variant {
            Variant::D => forward(t, out),
            // Q D†, D† = C† + B† H†
            Variant::Dagger => {
                let MdwfParams {
                    ls, m5, b, c, mass, ..
                } = self.params;
                let height = 4.0 - m5;
                let h = scratch(h, inp);
                self.hops(h, inp, None, (1.0, 0.0), true, None);
                // The plain D† inp = (d+ + d− Y†) inp + (b − c Y†) h.
                let plain = |dst: &mut DeviceSpinor<T>| {
                    assert!(dst.same_shape(inp));
                    dst.parity = None;
                    // SAFETY: buffers of the same shape, the output none of the inputs.
                    unsafe {
                        T::mdwf_fifth(
                            dst.data.ptr,
                            inp.data.ptr,
                            1.0 + b * height,
                            1.0 - c * height,
                            h.data.ptr,
                            b,
                            -c,
                            (inp.vectors * padded(inp.sites())) as i32,
                            ls as i32,
                            mass,
                            1,
                        );
                    }
                };
                if has_alpha {
                    let t = scratch(t, inp);
                    plain(t);
                    self.mult_q(out, t, false);
                } else {
                    plain(out);
                }
            }
            // Γ5 D Q
            Variant::Gamma5D => {
                let d = scratch(h, inp);
                forward(t, d);
                gamma5(out, d, true);
            }
        }
    }

    /// The count Bridge++ uses for its domain-wall operator, `(464 Nc + 240) Ls`.
    fn flops_per_site(&self) -> usize {
        (464 * NC + 240) * self.params.ls
    }
}
