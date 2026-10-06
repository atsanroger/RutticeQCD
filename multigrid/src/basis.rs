use crate::{Aggregation, MultigridError};
use ruttice_core::{for_sites, Coarse, Field, Spinor, Storage, C64};
use ruttice_kernel::{
    chiral_join, chiral_split, Cplx, Real, Spinor as SiteSpinor, CHIRAL_DIM, SPINOR_REALS,
};
use std::sync::Arc;

/// Reals stored per fine site and test vector: the six complex components of each
/// chirality, positive first.
const VECTOR_REALS: usize = 2 * 2 * CHIRAL_DIM;

/// The components of one chirality at one site.
type Chiral<T> = [Cplx<T>; CHIRAL_DIM];

/// The prolongation P between a coarse lattice and the fine one, and the restriction P†.
///
/// A coarse site carries `2 n` complex numbers for `n` test vectors: component `i` is the
/// coefficient of test vector `i` on the positive-chirality aggregate of that block,
/// component `n + i` the same on the negative-chirality one.
pub struct Basis<T: Storage> {
    agg: Arc<Aggregation>,
    n: usize,
    /// `[fine site][vector][chirality][6 complex]`, orthonormal over every aggregate.
    data: Vec<T>,
}

impl<T: Storage> Basis<T> {
    /// Build P from test vectors, given as one field holding `n` vectors. Works in double
    /// precision and stores the result as `T`.
    pub fn from_vectors(
        agg: &Arc<Aggregation>,
        vectors: &Field<f64, Spinor>,
    ) -> Result<Self, MultigridError> {
        let fine = agg.fine_lattice();
        assert_eq!(vectors.num_sites(), fine.num_sites());
        assert!(vectors.outer() == 1 && vectors.ls() == 1);
        let n = vectors.num_vectors();
        let volume = agg.block_volume();
        assert!(
            n <= CHIRAL_DIM * volume,
            "{n} test vectors do not fit into aggregates of dimension {}",
            CHIRAL_DIM * volume
        );

        let mut data = vec![T::default(); fine.num_sites() * n * VECTOR_REALS];
        // w[chirality][vector][site in block * 6 + component]
        let dim = CHIRAL_DIM * volume;
        let mut w = [vec![C64::ZERO; n * dim], vec![C64::ZERO; n * dim]];

        for coarse_site in 0..agg.coarse_lattice().num_sites() {
            let sites = agg.fine_sites(coarse_site);
            for (k, x) in sites.iter().enumerate() {
                let all = vectors.site(0, *x as usize);
                for i in 0..n {
                    let psi =
                        SiteSpinor::from_reals(&all[i * SPINOR_REALS..(i + 1) * SPINOR_REALS]);
                    let (plus, minus) = chiral_split(&psi);
                    let at = i * dim + k * CHIRAL_DIM;
                    w[0][at..at + CHIRAL_DIM].copy_from_slice(&plus);
                    w[1][at..at + CHIRAL_DIM].copy_from_slice(&minus);
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

            for (k, x) in sites.iter().enumerate() {
                for i in 0..n {
                    let base = (*x as usize * n + i) * VECTOR_REALS;
                    for (c, chirality) in w.iter().enumerate() {
                        for d in 0..CHIRAL_DIM {
                            let z = chirality[i * dim + k * CHIRAL_DIM + d];
                            let at = base + 2 * (c * CHIRAL_DIM + d);
                            data[at] = T::store(Real::from_f64(z.re));
                            data[at + 1] = T::store(Real::from_f64(z.im));
                        }
                    }
                }
            }
        }
        Ok(Self {
            agg: agg.clone(),
            n,
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

    /// The stored components of test vector `i` at fine site `x`: (positive, negative).
    #[inline(always)]
    fn components(&self, x: usize, i: usize) -> (Chiral<T::Compute>, Chiral<T::Compute>) {
        let s = &self.data[(x * self.n + i) * VECTOR_REALS..(x * self.n + i + 1) * VECTOR_REALS];
        let get = |c: usize| {
            core::array::from_fn(|d| {
                let at = 2 * (c * CHIRAL_DIM + d);
                Cplx::new(s[at].load(), s[at + 1].load())
            })
        };
        (get(0), get(1))
    }

    fn check(&self, coarse: &Field<T, Coarse>, fine: &Field<T, Spinor>) {
        assert_eq!(fine.num_sites(), self.agg.fine_lattice().num_sites());
        assert_eq!(coarse.num_sites(), self.agg.coarse_lattice().num_sites());
        assert_eq!(coarse.internal_dof(), 2 * self.coarse_dim());
        assert!(
            fine.ls() == 1 && coarse.ls() == 1,
            "fifth-dimension fields are not supported yet"
        );
        assert_eq!(coarse.num_vectors(), fine.num_vectors());
    }

    /// `coarse = P† fine`.
    pub fn restrict(&self, coarse: &mut Field<T, Coarse>, fine: &Field<T, Spinor>) {
        self.check(coarse, fine);
        let (n, dim) = (self.n, self.coarse_dim());
        let site_len = coarse.site_len();
        for_sites(coarse.slab_mut(0), site_len, |coarse_site, out| {
            out.fill(T::default());
            for x in self.agg.fine_sites(coarse_site) {
                let x = *x as usize;
                let fine_site = fine.site(0, x).as_chunks::<SPINOR_REALS>().0;
                for (out, psi) in out.chunks_exact_mut(2 * dim).zip(fine_site) {
                    let psi: [T::Compute; SPINOR_REALS] = psi.map(Storage::load);
                    let (plus, minus) = chiral_split(&SiteSpinor::from_reals(&psi));
                    for i in 0..n {
                        let (b_plus, b_minus) = self.components(x, i);
                        let (mut cp, mut cm) = (Cplx::ZERO, Cplx::ZERO);
                        for d in 0..CHIRAL_DIM {
                            cp += b_plus[d].conj() * plus[d];
                            cm += b_minus[d].conj() * minus[d];
                        }
                        add(&mut out[2 * i..2 * i + 2], cp);
                        add(&mut out[2 * (n + i)..2 * (n + i) + 2], cm);
                    }
                }
            }
        });
    }

    /// `fine = P coarse`.
    pub fn prolong(&self, fine: &mut Field<T, Spinor>, coarse: &Field<T, Coarse>) {
        self.check(coarse, fine);
        let (n, dim) = (self.n, self.coarse_dim());
        let site_len = fine.site_len();
        for_sites(fine.slab_mut(0), site_len, |x, out| {
            let coarse_site = coarse.site(0, self.agg.coarse_site(x));
            let out = out.as_chunks_mut::<SPINOR_REALS>().0;
            for (out, c) in out.iter_mut().zip(coarse_site.chunks_exact(2 * dim)) {
                let mut plus = [Cplx::ZERO; CHIRAL_DIM];
                let mut minus = [Cplx::ZERO; CHIRAL_DIM];
                for i in 0..n {
                    let (b_plus, b_minus) = self.components(x, i);
                    let cp = Cplx::new(c[2 * i].load(), c[2 * i + 1].load());
                    let cm = Cplx::new(c[2 * (n + i)].load(), c[2 * (n + i) + 1].load());
                    for d in 0..CHIRAL_DIM {
                        plus[d] += cp * b_plus[d];
                        minus[d] += cm * b_minus[d];
                    }
                }
                let mut reals = [T::Compute::ZERO; SPINOR_REALS];
                chiral_join(&plus, &minus).write_reals(&mut reals);
                for (o, r) in out.iter_mut().zip(reals) {
                    *o = T::store(r);
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
