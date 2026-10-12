use super::{Aggregation, MultigridError};
use crate::kernel::{Cplx, Gamma, Number, Real, NC, NS, SPINOR_REALS};
use crate::{for_sites, Coarse, EvenOddIndex, Field, Half, Sites, Spinor, Storage, Vector, C64};
use std::marker::PhantomData;
use std::sync::Arc;

/// The prolongation P between a coarse lattice and the fine one, and the restriction P†.
///
/// A coarse site carries `2 n` complex numbers for `n` test vectors: component `i` is the
/// coefficient of test vector `i` on the positive-chirality aggregate of that block,
/// component `n + i` the same on the negative-chirality one.
///
/// The chiralities come from the operator's Γ5, given with the test vectors: the aggregate
/// of a block and chirality ± is spanned by (1 ± Γ5) v_i / 2 restricted to the block. Γ5
/// may act on all components of a site, including a fifth dimension (`ls > 1`): the
/// aggregate holds every s. This is how Bridge++ (`MultiGrid_Clover`) and the CUDA
/// domain-wall multigrid (`MultiGrid_Domainwall`) do it (the design notes, §7.6.1).
///
/// `T` is the storage of the fields P acts between, `S` that of P itself (by default the
/// same). A single-precision P with double-precision fields (`Basis<f64, f32>`) halves the
/// memory of P, which holds two fine-lattice vectors per test vector, while restriction and
/// prolongation still compute in `T`. With a [`Reflection`] Γ5 (the domain-wall Γ5 = γ5 R,
/// given to [`BasisBuilder::folded`]) P keeps only half of the fifth dimension: half the
/// memory again, and half the work in restriction and prolongation.
pub struct Basis<T: Storage, S: Storage = T> {
    agg: Arc<Aggregation>,
    n: usize,
    /// Reals of one vector at one site: `24 * ls`.
    reals: usize,
    /// Reals kept of one chirality part at one site: `reals`, or `reals / 2` when folded.
    kept: usize,
    fold: Option<Gamma>,
    /// `[fine site][vector][chirality][kept]`, orthonormal over every aggregate.
    data: Vec<S>,
    fields: PhantomData<T>,
}

/// Γ5 = R ⊗ g for fields with a fifth dimension of even extent: R reverses it,
/// s → Ls − 1 − s, and g is a spin matrix with g² = 1 (γ5 of the representation), the
/// colour untouched. The domain-wall Γ5 = γ5 R is of this form. The ± chirality part p of a
/// test vector then has p_(Ls−1−s) = ± g p_s, so P needs only s < Ls/2: for a field f,
/// ⟨p, f⟩ = Σ_(s<Ls/2) ⟨p_s, f_s ± g f_(Ls−1−s)⟩ (g is unitary and Hermitian), and P c is
/// a₊ + a₋ on s < Ls/2 and g (a₊ − a₋) on the mirror slice, a± the half-length sums.
#[derive(Clone, Copy, Debug)]
pub struct Reflection(pub Gamma);

/// `out = g inp` on one spinor (24 reals, the layout of [`crate::kernel::Spinor`]).
#[inline(always)]
fn spin_apply<C: Number>(g: &Gamma, inp: &[C], out: &mut [C]) {
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (c + NC * g.col[a]);
            let z = g.val[a].times(Cplx::new(inp[k], inp[k + 1]));
            let o = 2 * (c + NC * a);
            out[o] = z.re;
            out[o + 1] = z.im;
        }
    }
}

/// A stored value of P in the arithmetic of the fields.
#[inline(always)]
fn widen<S: Storage, T: Storage>(x: S) -> T::Compute {
    Number::from_f64(x.load().to_f64())
}

/// The largest `12 * ls` the prolongation keeps on the stack (Ls up to 64).
const MAX_HALF_REALS: usize = 12 * 64;

impl<T: Storage, S: Storage> Basis<T, S> {
    /// Build P from test vectors `x` (one field holding `n` vectors) and `g5x` = Γ5 x, the
    /// operator's Γ5 applied to them. Works in double precision and stores the result as
    /// `T`.
    pub fn from_vectors(
        agg: &Arc<Aggregation>,
        x: &Field<f64, Spinor>,
        g5x: &Field<f64, Spinor>,
    ) -> Result<Self, MultigridError> {
        assert!(x.outer() == 1 && x.same_shape(g5x));
        let n = x.num_vectors();
        let mut builder = BasisBuilder::<T, S>::new(agg, n);
        for i in 0..n {
            builder.add(i, &x.select(&[i]), &g5x.select(&[i]));
        }
        builder.finish()
    }

    pub fn aggregation(&self) -> &Arc<Aggregation> {
        &self.agg
    }

    /// Number of test vectors.
    pub fn num_test_vectors(&self) -> usize {
        self.n
    }

    /// Complex numbers per coarse site.
    pub fn coarse_dim(&self) -> usize {
        2 * self.n
    }

    /// A zero field on the coarse lattice holding `num_vectors` vectors.
    pub fn coarse_field(&self, num_vectors: usize) -> Field<T, Coarse> {
        Field::with_shape(
            self.agg.coarse_lattice(),
            2 * self.coarse_dim(),
            1,
            num_vectors,
            1,
        )
    }

    /// A zero field on the fine lattice of the shape the test vectors have (`ls` included),
    /// holding `num_vectors` vectors.
    pub fn fine_field(&self, num_vectors: usize) -> Field<T, Spinor> {
        Field::with_shape(self.agg.fine_lattice(), 24, self.reals / 24, num_vectors, 1)
    }

    /// Whether P keeps only half of the fifth dimension (see [`Reflection`]).
    pub fn is_folded(&self) -> bool {
        self.fold.is_some()
    }

    /// The stored reals of test vector `i`, chirality `c` (0 positive), at fine site `x`:
    /// all of them, or those of s < Ls/2 when folded.
    #[inline(always)]
    fn part(&self, x: usize, i: usize, c: usize) -> &[S] {
        let base = ((x * self.n + i) * 2 + c) * self.kept;
        &self.data[base..base + self.kept]
    }

    fn check<X: Sites>(&self, coarse: &Field<T, Coarse>, fine: &Field<T, Spinor, X>) {
        assert!(
            self.reals / 2 <= MAX_HALF_REALS,
            "Ls too large for restriction and prolongation"
        );
        assert_eq!(coarse.num_sites(), self.agg.coarse_lattice().num_sites());
        assert_eq!(coarse.internal_dof(), 2 * self.coarse_dim());
        assert_eq!(coarse.ls(), 1);
        assert_eq!(coarse.num_vectors(), fine.num_vectors());
        assert_eq!(
            fine.site_len() / fine.num_vectors(),
            self.reals,
            "the fine field has another number of reals per vector than the test vectors"
        );
    }

    /// `coarse = P† fine`.
    pub fn restrict(&self, coarse: &mut Field<T, Coarse>, fine: &Field<T, Spinor>) {
        self.check(coarse, fine);
        assert_eq!(fine.num_sites(), self.agg.fine_lattice().num_sites());
        let (dim, reals) = (self.coarse_dim(), self.reals);
        let site_len = coarse.site_len();
        for_sites(coarse.slab_mut(0), site_len, |coarse_site, out| {
            out.fill(T::default());
            let mut folded = [[T::Compute::ZERO; 2 * MAX_HALF_REALS]; 2];
            for x in self.agg.fine_sites(coarse_site) {
                let x = *x as usize;
                for (out, f) in out
                    .chunks_exact_mut(2 * dim)
                    .zip(fine.site(0, x).chunks_exact(reals))
                {
                    self.restrict_site(x, f, out, &mut folded);
                }
            }
        });
    }

    /// `coarse = P† ι half`, ι putting a field on the sites of one parity onto the whole
    /// lattice (zero on the others): only the fine sites of that parity contribute.
    /// `index` is the even-odd decomposition of the fine lattice.
    pub fn restrict_half(
        &self,
        coarse: &mut Field<T, Coarse>,
        half: &Field<T, Spinor, Half>,
        index: &EvenOddIndex,
    ) {
        self.check(coarse, half);
        assert_eq!(half.num_sites(), index.num_half_sites());
        let (dim, reals, parity) = (self.coarse_dim(), self.reals, half.parity());
        let site_len = coarse.site_len();
        for_sites(coarse.slab_mut(0), site_len, |coarse_site, out| {
            out.fill(T::default());
            let mut folded = [[T::Compute::ZERO; 2 * MAX_HALF_REALS]; 2];
            for x in self.agg.fine_sites(coarse_site) {
                let x = *x as usize;
                let (p, h) = index.half_site(x);
                if p != parity {
                    continue;
                }
                for (out, f) in out
                    .chunks_exact_mut(2 * dim)
                    .zip(half.site(0, h).chunks_exact(reals))
                {
                    self.restrict_site(x, f, out, &mut folded);
                }
            }
        });
    }

    /// Add P† restricted to fine site `x` of one vector `f` to `out` (that vector's part of
    /// the coarse site); `folded` is scratch.
    #[inline(always)]
    fn restrict_site(
        &self,
        x: usize,
        f: &[T],
        out: &mut [T],
        folded: &mut [[T::Compute; 2 * MAX_HALF_REALS]; 2],
    ) {
        let (n, reals, kept) = (self.n, self.reals, self.kept);
        // When folded: f_s ± g f_(Ls−1−s) for s < Ls/2, one for either chirality.
        if let Some(g) = &self.fold {
            let ls = reals / SPINOR_REALS;
            let mut gf = [T::Compute::ZERO; SPINOR_REALS];
            let mut far = [T::Compute::ZERO; SPINOR_REALS];
            for s in 0..ls / 2 {
                let mirror = (ls - 1 - s) * SPINOR_REALS;
                for (d, v) in far.iter_mut().zip(&f[mirror..mirror + SPINOR_REALS]) {
                    *d = v.load();
                }
                spin_apply(g, &far, &mut gf);
                let near = &f[s * SPINOR_REALS..(s + 1) * SPINOR_REALS];
                for (k, (v, w)) in near.iter().zip(&gf).enumerate() {
                    let at = s * SPINOR_REALS + k;
                    folded[0][at] = v.load() + *w;
                    folded[1][at] = v.load() - *w;
                }
            }
        }
        for i in 0..n {
            for c in 0..2 {
                let mut z = Cplx::<T::Compute>::ZERO;
                let b = self.part(x, i, c).as_chunks::<2>().0;
                if self.fold.is_some() {
                    for (b, f) in b.iter().zip(folded[c][..kept].as_chunks::<2>().0) {
                        let b = Cplx::new(widen::<S, T>(b[0]), widen::<S, T>(b[1]));
                        z += b.conj() * Cplx::new(f[0], f[1]);
                    }
                } else {
                    for (b, f) in b.iter().zip(f.as_chunks::<2>().0) {
                        let b = Cplx::new(widen::<S, T>(b[0]), widen::<S, T>(b[1]));
                        z += b.conj() * Cplx::new(f[0].load(), f[1].load());
                    }
                }
                let k = 2 * (c * n + i);
                add(&mut out[k..k + 2], z);
            }
        }
    }

    /// `fine = P coarse`.
    pub fn prolong(&self, fine: &mut Field<T, Spinor>, coarse: &Field<T, Coarse>) {
        self.check(coarse, fine);
        assert_eq!(fine.num_sites(), self.agg.fine_lattice().num_sites());
        let (dim, reals) = (self.coarse_dim(), self.reals);
        let site_len = fine.site_len();
        for_sites(fine.slab_mut(0), site_len, |x, out| {
            let coarse_site = coarse.site(0, self.agg.coarse_site(x));
            for (out, c) in out
                .chunks_exact_mut(reals)
                .zip(coarse_site.chunks_exact(2 * dim))
            {
                self.prolong_site(x, c, out);
            }
        });
    }

    /// `half = π P coarse`, π taking the sites of the parity of `half`. `index` is the
    /// even-odd decomposition of the fine lattice.
    pub fn prolong_half(
        &self,
        half: &mut Field<T, Spinor, Half>,
        coarse: &Field<T, Coarse>,
        index: &EvenOddIndex,
    ) {
        self.check(coarse, half);
        assert_eq!(half.num_sites(), index.num_half_sites());
        let (dim, reals, parity) = (self.coarse_dim(), self.reals, half.parity());
        let site_len = half.site_len();
        for_sites(half.slab_mut(0), site_len, |h, out| {
            let x = index.full_site(parity, h);
            let coarse_site = coarse.site(0, self.agg.coarse_site(x));
            for (out, c) in out
                .chunks_exact_mut(reals)
                .zip(coarse_site.chunks_exact(2 * dim))
            {
                self.prolong_site(x, c, out);
            }
        });
    }

    /// P at fine site `x` for one vector: `out` from that vector's part `c` of the coarse
    /// site.
    #[inline(always)]
    fn prolong_site(&self, x: usize, c: &[T], out: &mut [T]) {
        let (n, reals) = (self.n, self.reals);
        // On the stack: this runs once per fine site and vector. Unfolded, one sum over
        // both chiralities; folded, a₊ and a₋ over half the fifth dimension.
        let mut acc_store = [Cplx::<T::Compute>::ZERO; MAX_HALF_REALS];
        let half = self.kept / 2;
        for i in 0..n {
            for ch in 0..2 {
                let k = 2 * (ch * n + i);
                let coef = Cplx::new(c[k].load(), c[k + 1].load());
                let acc = match self.fold {
                    Some(_) => &mut acc_store[ch * half..(ch + 1) * half],
                    None => &mut acc_store[..half],
                };
                for (a, b) in acc.iter_mut().zip(self.part(x, i, ch).as_chunks::<2>().0) {
                    *a += coef * Cplx::new(widen::<S, T>(b[0]), widen::<S, T>(b[1]));
                }
            }
        }
        let store = |o: &mut [T], a: Cplx<T::Compute>| {
            o[0] = T::store(a.re);
            o[1] = T::store(a.im);
        };
        match &self.fold {
            None => {
                for (o, a) in out
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(&acc_store[..half])
                {
                    store(o, *a);
                }
            }
            Some(g) => {
                let ls = reals / SPINOR_REALS;
                let (plus, minus) = acc_store[..2 * half].split_at(half);
                let per_slice = SPINOR_REALS / 2;
                for s in 0..ls / 2 {
                    let mut diff = [T::Compute::ZERO; SPINOR_REALS];
                    for d in 0..per_slice {
                        let (p, m) = (plus[s * per_slice + d], minus[s * per_slice + d]);
                        let at = s * SPINOR_REALS + 2 * d;
                        store(&mut out[at..at + 2], p + m);
                        let q = p - m;
                        diff[2 * d] = q.re;
                        diff[2 * d + 1] = q.im;
                    }
                    let mut mirrored = [T::Compute::ZERO; SPINOR_REALS];
                    spin_apply(g, &diff, &mut mirrored);
                    let at = (ls - 1 - s) * SPINOR_REALS;
                    for (o, v) in out[at..at + SPINOR_REALS].iter_mut().zip(&mirrored) {
                        *o = T::store(*v);
                    }
                }
            }
        }
    }
}

/// Builds a [`Basis`] from test vectors given one at a time, so that a caller never needs
/// all of them, or all their Γ5 images, at once: the chirality parts (1 ± Γ5) v_i / 2 go
/// straight into the basis storage and are orthonormalised aggregate by aggregate in
/// [`finish`](Self::finish). The peak memory is then the basis itself plus one vector and
/// its image, instead of that plus all 2n vectors (`from_vectors` needs both fields whole).
pub struct BasisBuilder<T: Storage, S: Storage = T> {
    agg: Arc<Aggregation>,
    n: usize,
    /// Reals of one vector at one site, known from the first vector.
    reals: usize,
    /// Reals kept of a chirality part at one site (`reals`, or `reals / 2` folded).
    kept: usize,
    fold: Option<Gamma>,
    data: Vec<S>,
    added: Vec<bool>,
    fields: PhantomData<T>,
}

impl<T: Storage, S: Storage> BasisBuilder<T, S> {
    /// A builder for `n` test vectors on the aggregation `agg`.
    pub fn new(agg: &Arc<Aggregation>, n: usize) -> Self {
        Self {
            agg: agg.clone(),
            n,
            reals: 0,
            kept: 0,
            fold: None,
            data: Vec::new(),
            added: vec![false; n],
            fields: PhantomData,
        }
    }

    /// Keep only half of the fifth dimension: Γ5 is of the form of `reflection`, which
    /// [`add`](Self::add) checks on every test vector.
    pub fn folded(mut self, reflection: Reflection) -> Self {
        assert!(self.data.is_empty(), "fold before the first test vector");
        self.fold = Some(reflection.0);
        self
    }

    /// Test vector `i`: `x` holds one vector and `g5x` = Γ5 x, the operator's Γ5 applied
    /// to it.
    pub fn add(&mut self, i: usize, x: &Field<f64, Spinor>, g5x: &Field<f64, Spinor>) {
        let fine = self.agg.fine_lattice();
        assert_eq!(x.num_sites(), fine.num_sites());
        assert!(x.outer() == 1 && x.same_shape(g5x) && x.num_vectors() == 1);
        assert!(
            i < self.n && !self.added[i],
            "test vector {i} given twice or out of range"
        );
        let reals = x.site_len();
        if self.data.is_empty() {
            self.reals = reals;
            self.kept = match self.fold {
                Some(_) => {
                    let ls = reals / SPINOR_REALS;
                    assert!(
                        ls * SPINOR_REALS == reals && ls.is_multiple_of(2),
                        "folding needs spinors with an even fifth dimension"
                    );
                    reals / 2
                }
                None => reals,
            };
            self.data = vec![S::default(); fine.num_sites() * self.n * 2 * self.kept];
        }
        assert_eq!(reals, self.reals, "test vectors of different shapes");
        let (n, kept) = (self.n, self.kept);
        // Folded: the far half must be ± g times the near one, (x ± Γ5 x)_(Ls−1−s) =
        // ± g (x ± Γ5 x)_s; the largest deviation against the size of x.
        let (mut deviation, mut size) = (0.0_f64, 0.0_f64);
        for site in 0..fine.num_sites() {
            let (xs, gs) = (x.site(0, site), g5x.site(0, site));
            for c in 0..2 {
                let sign = if c == 0 { 1.0 } else { -1.0 };
                let base = ((site * n + i) * 2 + c) * kept;
                for (d, (v, g)) in xs.iter().zip(gs).take(kept).enumerate() {
                    self.data[base + d] = S::store(Number::from_f64(0.5 * (v + sign * g)));
                }
                if let Some(gamma) = &self.fold {
                    let ls = reals / SPINOR_REALS;
                    for s in 0..ls / 2 {
                        let part = |slice: usize| -> [f64; SPINOR_REALS] {
                            let at = slice * SPINOR_REALS;
                            core::array::from_fn(|k| 0.5 * (xs[at + k] + sign * gs[at + k]))
                        };
                        let (near, far) = (part(s), part(ls - 1 - s));
                        let mut mirrored = [0.0; SPINOR_REALS];
                        spin_apply(gamma, &near, &mut mirrored);
                        for (m, f) in mirrored.iter().zip(&far) {
                            deviation = deviation.max((sign * m - f).abs());
                        }
                    }
                }
            }
            size = xs.iter().fold(size, |m, v| m.max(v.abs()));
        }
        assert!(
            deviation <= 1e-10 * size,
            "Γ5 is not R ⊗ g for the folding given: deviation {deviation:e} against {size:e}"
        );
        self.added[i] = true;
    }

    /// Orthonormalise every chirality part over every aggregate and return P.
    pub fn finish(mut self) -> Result<Basis<T, S>, MultigridError> {
        assert!(
            self.added.iter().all(|a| *a),
            "not every test vector was given"
        );
        let (n, reals) = (self.n, self.kept);
        let volume = self.agg.block_volume();
        // Complex numbers of one vector on one aggregate (both chiralities stored).
        let dim = volume * reals / 2;
        // Folded, the stored half carries half of every inner product (|g v| = |v|): the
        // parts are orthonormalised on it and scaled by 1/√2 to be orthonormal on all of
        // the fifth dimension. A folded part is free on its half, so up to `dim` of them
        // fit, against `dim / 2` unfolded (half of a full part is fixed by its chirality).
        let (fit, scale) = match self.fold {
            Some(_) => (dim, core::f64::consts::FRAC_1_SQRT_2),
            None => (dim / 2, 1.0),
        };
        assert!(
            n <= fit,
            "{n} test vectors do not fit into aggregates of dimension {fit}"
        );
        // w[chirality][vector][site in block * reals/2 + component]
        let mut w = [vec![C64::ZERO; n * dim], vec![C64::ZERO; n * dim]];
        for coarse_site in 0..self.agg.coarse_lattice().num_sites() {
            let sites = self.agg.fine_sites(coarse_site);
            for (k, site) in sites.iter().enumerate() {
                for i in 0..n {
                    for (c, chirality) in w.iter_mut().enumerate() {
                        let base = ((*site as usize * n + i) * 2 + c) * reals;
                        let at = i * dim + k * reals / 2;
                        for d in 0..reals / 2 {
                            chirality[at + d] = C64::new(
                                self.data[base + 2 * d].load().to_f64(),
                                self.data[base + 2 * d + 1].load().to_f64(),
                            );
                        }
                    }
                }
            }
            for chirality in w.iter_mut() {
                orthonormalise(chirality, n, dim).map_err(|vector| {
                    MultigridError::DependentTestVectors {
                        coarse_site,
                        vector,
                    }
                })?;
            }
            for (k, site) in sites.iter().enumerate() {
                for i in 0..n {
                    for (c, chirality) in w.iter().enumerate() {
                        let base = ((*site as usize * n + i) * 2 + c) * reals;
                        for d in 0..reals / 2 {
                            let z = chirality[i * dim + k * reals / 2 + d].scale(scale);
                            self.data[base + 2 * d] = S::store(Number::from_f64(z.re));
                            self.data[base + 2 * d + 1] = S::store(Number::from_f64(z.im));
                        }
                    }
                }
            }
        }
        Ok(Basis {
            agg: self.agg,
            n,
            reals: self.reals,
            kept: self.kept,
            fold: self.fold,
            data: self.data,
            fields: PhantomData,
        })
    }
}

#[inline(always)]
fn add<T: Storage>(slot: &mut [T], z: Cplx<T::Compute>) {
    slot[0] = T::store(slot[0].load() + z.re);
    slot[1] = T::store(slot[1].load() + z.im);
}

/// Modified Gram-Schmidt on `n` vectors of length `dim` stored one after another. Returns
/// the index of the first vector that is (numerically) in the span of the earlier ones.
fn orthonormalise(w: &mut [C64], n: usize, dim: usize) -> Result<(), usize> {
    for i in 0..n {
        let (done, rest) = w.split_at_mut(i * dim);
        let wi = &mut rest[..dim];
        let before: f64 = wi.iter().map(|z| z.norm_sqr()).sum();
        for j in 0..i {
            let wj = &done[j * dim..(j + 1) * dim];
            let mut overlap = C64::ZERO;
            for (a, b) in wj.iter().zip(wi.iter()) {
                overlap += a.conj() * *b;
            }
            for (a, b) in wj.iter().zip(wi.iter_mut()) {
                *b = *b - overlap * *a;
            }
        }
        let after: f64 = wi.iter().map(|z| z.norm_sqr()).sum();
        if after.is_nan() || after <= 1e-24 * before {
            return Err(i);
        }
        let scale = 1.0 / after.sqrt();
        for z in wi.iter_mut() {
            *z = z.scale(scale);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_schmidt_gives_an_orthonormal_set_and_detects_dependence() {
        let (n, dim) = (4, 9);
        let mut w: Vec<C64> = (0..n * dim)
            .map(|k| {
                C64::new(
                    ((k * 7 + 1) as f64 * 0.37).sin(),
                    ((k * 5 + 2) as f64 * 0.53).cos(),
                )
            })
            .collect();
        orthonormalise(&mut w, n, dim).unwrap();
        for i in 0..n {
            for j in 0..n {
                let mut s = C64::ZERO;
                for d in 0..dim {
                    s += w[i * dim + d].conj() * w[j * dim + d];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s.re - want).abs() < 1e-13 && s.im.abs() < 1e-13);
            }
        }

        let mut dependent = w.clone();
        for d in 0..dim {
            dependent[2 * dim + d] = dependent[d].scale(2.0) - dependent[dim + d];
        }
        assert_eq!(orthonormalise(&mut dependent, n, dim), Err(2));
    }
}
