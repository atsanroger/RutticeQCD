use super::{Aggregation, MultigridError};
use crate::kernel::{Cplx, Number};
use crate::{for_sites, Coarse, Field, Spinor, Storage, Vector, C64};
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
pub struct Basis<T: Storage> {
    agg: Arc<Aggregation>,
    n: usize,
    /// Reals of one vector at one site: `24 * ls`.
    reals: usize,
    /// `[fine site][vector][chirality][reals]`, orthonormal over every aggregate.
    data: Vec<T>,
}

impl<T: Storage> Basis<T> {
    /// Build P from test vectors `x` (one field holding `n` vectors) and `g5x` = Γ5 x, the
    /// operator's Γ5 applied to them. Works in double precision and stores the result as
    /// `T`.
    pub fn from_vectors(
        agg: &Arc<Aggregation>,
        x: &Field<f64, Spinor>,
        g5x: &Field<f64, Spinor>,
    ) -> Result<Self, MultigridError> {
        let fine = agg.fine_lattice();
        assert_eq!(x.num_sites(), fine.num_sites());
        assert!(x.outer() == 1 && x.same_shape(g5x));
        let n = x.num_vectors();
        let reals = x.site_len() / n;
        let volume = agg.block_volume();
        // Complex numbers of one vector on one aggregate (both chiralities stored).
        let dim = volume * reals / 2;
        assert!(
            n <= dim / 2,
            "{n} test vectors do not fit into aggregates of dimension {}",
            dim / 2
        );

        let mut data = vec![T::default(); fine.num_sites() * n * 2 * reals];
        // w[chirality][vector][site in block * reals/2 + component]
        let mut w = [vec![C64::ZERO; n * dim], vec![C64::ZERO; n * dim]];

        for coarse_site in 0..agg.coarse_lattice().num_sites() {
            let sites = agg.fine_sites(coarse_site);
            for (k, site) in sites.iter().enumerate() {
                let (xs, gs) = (x.site(0, *site as usize), g5x.site(0, *site as usize));
                for i in 0..n {
                    let at = i * dim + k * reals / 2;
                    for d in 0..reals / 2 {
                        let v = C64::new(xs[i * reals + 2 * d], xs[i * reals + 2 * d + 1]);
                        let g = C64::new(gs[i * reals + 2 * d], gs[i * reals + 2 * d + 1]);
                        w[0][at + d] = (v + g).scale(0.5);
                        w[1][at + d] = (v - g).scale(0.5);
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
                            data[base + 2 * d] = T::store(Number::from_f64(z.re));
                            data[base + 2 * d + 1] = T::store(Number::from_f64(z.im));
                        }
                    }
                }
            }
        }
        Ok(Self {
            agg: agg.clone(),
            n,
            reals,
            data,
        })
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
    fn part(&self, x: usize, i: usize, c: usize) -> &[T] {
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
                                let b = Cplx::new(b[0].load(), b[1].load());
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
        for_sites(fine.slab_mut(0), site_len, |x, out| {
            let coarse_site = coarse.site(0, self.agg.coarse_site(x));
            for (out, c) in out
                .chunks_exact_mut(reals)
                .zip(coarse_site.chunks_exact(2 * dim))
            {
                let mut acc = vec![Cplx::<T::Compute>::ZERO; reals / 2];
                for i in 0..n {
                    for ch in 0..2 {
                        let k = 2 * (ch * n + i);
                        let coef = Cplx::new(c[k].load(), c[k + 1].load());
                        for (a, b) in acc.iter_mut().zip(self.part(x, i, ch).as_chunks::<2>().0) {
                            *a += coef * Cplx::new(b[0].load(), b[1].load());
                        }
                    }
                }
                for (o, a) in out.as_chunks_mut::<2>().0.iter_mut().zip(acc) {
                    o[0] = T::store(a.re);
                    o[1] = T::store(a.im);
                }
            }
        });
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
