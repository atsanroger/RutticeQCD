use super::{Aggregation, MultigridError};
use crate::kernel::{Cplx, Number, Real};
use crate::{for_sites, Coarse, Field, Spinor, Storage, Vector, C64};
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
/// domain-wall multigrid (`MultiGrid_Domainwall`) do it (DESIGN.md §7.6.1).
///
/// `T` is the storage of the fields P acts between, `S` that of P itself (by default the
/// same). A single-precision P with double-precision fields (`Basis<f64, f32>`) halves the
/// memory of P, which holds two fine-lattice vectors per test vector, while restriction and
/// prolongation still compute in `T`.
pub struct Basis<T: Storage, S: Storage = T> {
    agg: Arc<Aggregation>,
    n: usize,
    /// Reals of one vector at one site: `24 * ls`.
    reals: usize,
    /// `[fine site][vector][chirality][reals]`, orthonormal over every aggregate.
    data: Vec<S>,
    fields: PhantomData<T>,
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

    /// The stored reals of test vector `i`, chirality `c` (0 positive), at fine site `x`.
    #[inline(always)]
    fn part(&self, x: usize, i: usize, c: usize) -> &[S] {
        let base = ((x * self.n + i) * 2 + c) * self.reals;
        &self.data[base..base + self.reals]
    }

    fn check(&self, coarse: &Field<T, Coarse>, fine: &Field<T, Spinor>) {
        assert_eq!(fine.num_sites(), self.agg.fine_lattice().num_sites());
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
        let (n, dim, reals) = (self.n, self.coarse_dim(), self.reals);
        let site_len = coarse.site_len();
        for_sites(coarse.slab_mut(0), site_len, |coarse_site, out| {
            out.fill(T::default());
            for x in self.agg.fine_sites(coarse_site) {
                let x = *x as usize;
                let fine_site = fine.site(0, x);
                for (out, f) in out
                    .chunks_exact_mut(2 * dim)
                    .zip(fine_site.chunks_exact(reals))
                {
                    for i in 0..n {
                        for c in 0..2 {
                            let mut z = Cplx::<T::Compute>::ZERO;
                            for (b, f) in self
                                .part(x, i, c)
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .zip(f.as_chunks::<2>().0)
                            {
                                let b = Cplx::new(widen::<S, T>(b[0]), widen::<S, T>(b[1]));
                                z += b.conj() * Cplx::new(f[0].load(), f[1].load());
                            }
                            let k = 2 * (c * n + i);
                            add(&mut out[k..k + 2], z);
                        }
                    }
                }
            }
        });
    }

    /// `fine = P coarse`.
    pub fn prolong(&self, fine: &mut Field<T, Spinor>, coarse: &Field<T, Coarse>) {
        self.check(coarse, fine);
        let (n, dim, reals) = (self.n, self.coarse_dim(), self.reals);
        let site_len = fine.site_len();
        assert!(
            reals / 2 <= MAX_HALF_REALS,
            "Ls too large for the prolongation"
        );
        for_sites(fine.slab_mut(0), site_len, |x, out| {
            let coarse_site = coarse.site(0, self.agg.coarse_site(x));
            for (out, c) in out
                .chunks_exact_mut(reals)
                .zip(coarse_site.chunks_exact(2 * dim))
            {
                // On the stack: this runs once per fine site and vector.
                let mut acc_store = [Cplx::<T::Compute>::ZERO; MAX_HALF_REALS];
                let acc = &mut acc_store[..reals / 2];
                for i in 0..n {
                    for ch in 0..2 {
                        let k = 2 * (ch * n + i);
                        let coef = Cplx::new(c[k].load(), c[k + 1].load());
                        for (a, b) in acc.iter_mut().zip(self.part(x, i, ch).as_chunks::<2>().0) {
                            *a += coef * Cplx::new(widen::<S, T>(b[0]), widen::<S, T>(b[1]));
                        }
                    }
                }
                for (o, a) in out.as_chunks_mut::<2>().0.iter_mut().zip(acc.iter()) {
                    o[0] = T::store(a.re);
                    o[1] = T::store(a.im);
                }
            }
        });
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
            data: Vec::new(),
            added: vec![false; n],
            fields: PhantomData,
        }
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
            self.data = vec![S::default(); fine.num_sites() * self.n * 2 * reals];
        }
        assert_eq!(reals, self.reals, "test vectors of different shapes");
        let n = self.n;
        for site in 0..fine.num_sites() {
            let (xs, gs) = (x.site(0, site), g5x.site(0, site));
            for c in 0..2 {
                let sign = if c == 0 { 1.0 } else { -1.0 };
                let base = ((site * n + i) * 2 + c) * reals;
                for (d, (v, g)) in xs.iter().zip(gs).enumerate() {
                    self.data[base + d] = S::store(Number::from_f64(0.5 * (v + sign * g)));
                }
            }
        }
        self.added[i] = true;
    }

    /// Orthonormalise every chirality part over every aggregate and return P.
    pub fn finish(mut self) -> Result<Basis<T, S>, MultigridError> {
        assert!(
            self.added.iter().all(|a| *a),
            "not every test vector was given"
        );
        let (n, reals) = (self.n, self.reals);
        let volume = self.agg.block_volume();
        // Complex numbers of one vector on one aggregate (both chiralities stored).
        let dim = volume * reals / 2;
        assert!(
            n <= dim / 2,
            "{n} test vectors do not fit into aggregates of dimension {}",
            dim / 2
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
                            let z = chirality[i * dim + k * reals / 2 + d];
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
            reals,
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
