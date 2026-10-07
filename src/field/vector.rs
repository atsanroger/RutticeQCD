//! The only interface solvers need from a vector.

use crate::field::storage::Storage;
use crate::field::{Aos, Aosoa, Cpu, Field, Full, Kind, Layout, Sites};
use crate::kernel::{Cplx, Number, Real};
use crate::lattice::blocks::Blocks;
use crate::lattice::sites::{for_sites, reduce_sites_dyn};

/// Coefficients and reduction results are always double precision, whatever the field stores.
pub type C64 = Cplx<f64>;

/// A set of `num_vectors` vectors of equal length that are operated on together.
///
/// Coefficients and results are slices with one entry per vector, so a solver written
/// for one right-hand side works unchanged for several: each vector gets its own
/// coefficients. Operations between different vectors of the set (Gram matrices, block
/// updates) are not here yet.
pub trait Vector: Sized {
    fn num_vectors(&self) -> usize;
    /// A zero vector of the same shape.
    fn zeros_like(&self) -> Self;
    fn same_shape(&self, x: &Self) -> bool;
    fn copy_from(&mut self, x: &Self);
    fn set_zero(&mut self);
    /// `self_j *= a_j`.
    fn scale(&mut self, a: &[f64]);
    /// `self_j += a_j x_j`.
    fn axpy(&mut self, a: &[C64], x: &Self);
    /// `self_j = a_j self_j + x_j`.
    fn aypx(&mut self, a: &[C64], x: &Self);
    /// `out_j = <self_j | x_j>`.
    fn dot(&self, x: &Self, out: &mut [C64]);
    /// `out_j = |self_j|^2`.
    fn norm_sqr(&self, out: &mut [f64]);
    /// `self_j += a_j x_j`, then `out_j = |self_j|^2`. Implementations may do both in one sweep.
    fn axpy_norm_sqr(&mut self, a: &[C64], x: &Self, out: &mut [f64]) {
        self.axpy(a, x);
        self.norm_sqr(out);
    }
}

/// Conversion between vectors of different precision (and, later, layout).
pub trait ConvertFrom<Src> {
    fn convert_from(&mut self, src: &Src);
}

/// Reals of one vector in one block of the layout (one site for `Aos`): the `ls` copies
/// belonging to it are contiguous, and in a block layout every real comes once per lane.
fn vector_len<T: Storage, K: Kind, S: Sites, L: Layout>(f: &Field<T, K, S, L, Cpu>) -> usize {
    f.internal_dof() * f.ls() * L::LANES
}

/// The complex numbers of one vector in one block, as (real parts, imaginary parts): the
/// reals come in pairs of runs of `lanes` values, one run per lane. With one lane these are
/// the usual (re, im) pairs.
#[inline(always)]
fn complex_runs<T>(v: &[T], lanes: usize) -> impl Iterator<Item = (&[T], &[T])> {
    v.chunks_exact(2 * lanes).map(move |c| c.split_at(lanes))
}

#[inline(always)]
fn complex_runs_mut<T>(v: &mut [T], lanes: usize) -> impl Iterator<Item = (&mut [T], &mut [T])> {
    v.chunks_exact_mut(2 * lanes)
        .map(move |c| c.split_at_mut(lanes))
}

/// Apply `f(vector index, dst, src)` to every vector in every block, in parallel. The
/// outer slabs are treated as more blocks.
fn update<T, F>(dst: &mut [T], src: &[T], site_len: usize, vector_len: usize, f: F)
where
    T: Storage,
    F: Fn(usize, &mut [T], &[T]) + Sync,
{
    assert_eq!(dst.len(), src.len());
    for_sites(dst, site_len, |i, d| {
        let s = &src[i * site_len..(i + 1) * site_len];
        for (j, (d, s)) in d
            .chunks_exact_mut(vector_len)
            .zip(s.chunks_exact(vector_len))
            .enumerate()
        {
            f(j, d, s);
        }
    });
}

#[inline(always)]
fn coefficient<R: Real>(a: C64) -> Cplx<R> {
    Cplx::new(R::from_f64(a.re), R::from_f64(a.im))
}

impl<T: Storage, K: Kind, S: Sites, L: Layout> Vector for Field<T, K, S, L, Cpu> {
    fn num_vectors(&self) -> usize {
        Field::num_vectors(self)
    }

    fn zeros_like(&self) -> Self {
        let mut z = Field::with_shape(
            self.lattice(),
            self.internal_dof(),
            self.ls(),
            Field::num_vectors(self),
            self.outer(),
        );
        z.copy_parity_from(self);
        z
    }

    fn same_shape(&self, x: &Self) -> bool {
        self.internal_dof() == x.internal_dof()
            && self.ls() == x.ls()
            && Field::num_vectors(self) == Field::num_vectors(x)
            && self.outer() == x.outer()
            && self.num_sites() == x.num_sites()
            && self.parity() == x.parity()
    }

    fn copy_from(&mut self, x: &Self) {
        assert!(self.same_shape(x));
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        update(
            self.as_mut_slice(),
            x.as_slice(),
            site_len,
            vector_len,
            |_, d, s| d.copy_from_slice(s),
        );
    }

    fn set_zero(&mut self) {
        let site_len = self.block_len();
        for_sites(self.as_mut_slice(), site_len, |_, d| d.fill(T::default()));
    }

    fn scale(&mut self, a: &[f64]) {
        assert_eq!(a.len(), Field::num_vectors(self));
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        for_sites(self.as_mut_slice(), site_len, |_, d| {
            for (j, d) in d.chunks_exact_mut(vector_len).enumerate() {
                let k = T::Compute::from_f64(a[j]);
                for x in d {
                    *x = T::store(x.load() * k);
                }
            }
        });
    }

    fn axpy(&mut self, a: &[C64], x: &Self) {
        assert!(self.same_shape(x));
        assert_eq!(a.len(), Field::num_vectors(self));
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        update(
            self.as_mut_slice(),
            x.as_slice(),
            site_len,
            vector_len,
            |j, d, s| {
                let k: Cplx<T::Compute> = coefficient(a[j]);
                let runs = complex_runs_mut(d, L::LANES).zip(complex_runs(s, L::LANES));
                for ((d_re, d_im), (s_re, s_im)) in runs {
                    for l in 0..L::LANES {
                        let z = k * Cplx::new(s_re[l].load(), s_im[l].load());
                        d_re[l] = T::store(d_re[l].load() + z.re);
                        d_im[l] = T::store(d_im[l].load() + z.im);
                    }
                }
            },
        );
    }

    fn aypx(&mut self, a: &[C64], x: &Self) {
        assert!(self.same_shape(x));
        assert_eq!(a.len(), Field::num_vectors(self));
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        update(
            self.as_mut_slice(),
            x.as_slice(),
            site_len,
            vector_len,
            |j, d, s| {
                let k: Cplx<T::Compute> = coefficient(a[j]);
                let runs = complex_runs_mut(d, L::LANES).zip(complex_runs(s, L::LANES));
                for ((d_re, d_im), (s_re, s_im)) in runs {
                    for l in 0..L::LANES {
                        let z = k * Cplx::new(d_re[l].load(), d_im[l].load());
                        d_re[l] = T::store(z.re + s_re[l].load());
                        d_im[l] = T::store(z.im + s_im[l].load());
                    }
                }
            },
        );
    }

    fn dot(&self, x: &Self, out: &mut [C64]) {
        assert!(self.same_shape(x));
        let nv = Field::num_vectors(self);
        assert_eq!(out.len(), nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let (a, b) = (self.as_slice(), x.as_slice());
        let sums = reduce_sites_dyn(a.len() / site_len, 2 * nv, |i, acc| {
            let range = i * site_len..(i + 1) * site_len;
            let (a, b) = (&a[range.clone()], &b[range]);
            for (j, (a, b)) in a
                .chunks_exact(vector_len)
                .zip(b.chunks_exact(vector_len))
                .enumerate()
            {
                let runs = complex_runs(a, L::LANES).zip(complex_runs(b, L::LANES));
                for ((a_re, a_im), (b_re, b_im)) in runs {
                    for l in 0..L::LANES {
                        let (ar, ai) = (a_re[l].load().to_f64(), a_im[l].load().to_f64());
                        let (br, bi) = (b_re[l].load().to_f64(), b_im[l].load().to_f64());
                        acc[2 * j] += ar * br + ai * bi;
                        acc[2 * j + 1] += ar * bi - ai * br;
                    }
                }
            }
        });
        let mut sums = sums;
        self.lattice().comm.sum(&mut sums);
        for (o, s) in out.iter_mut().zip(sums.as_chunks::<2>().0) {
            *o = Cplx::new(s[0], s[1]);
        }
    }

    fn norm_sqr(&self, out: &mut [f64]) {
        let nv = Field::num_vectors(self);
        assert_eq!(out.len(), nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let a = self.as_slice();
        let sums = reduce_sites_dyn(a.len() / site_len, nv, |i, acc| {
            let a = &a[i * site_len..(i + 1) * site_len];
            for (j, a) in a.chunks_exact(vector_len).enumerate() {
                for x in a {
                    let x = x.load().to_f64();
                    acc[j] += x * x;
                }
            }
        });
        out.copy_from_slice(&sums);
        self.lattice().comm.sum(out);
    }
}

/// Change of precision within one layout.
impl<T, U, K, S, L> ConvertFrom<Field<U, K, S, L, Cpu>> for Field<T, K, S, L, Cpu>
where
    T: Storage,
    U: Storage,
    K: Kind,
    S: Sites,
    L: Layout,
{
    fn convert_from(&mut self, src: &Field<U, K, S, L, Cpu>) {
        assert_eq!(self.as_slice().len(), src.as_slice().len());
        assert_eq!(self.site_len(), src.site_len());
        let site_len = self.block_len();
        let s = src.as_slice();
        for_sites(self.as_mut_slice(), site_len, |i, d| {
            for (d, s) in d.iter_mut().zip(&s[i * site_len..(i + 1) * site_len]) {
                *d = T::store(Number::from_f64(s.load().to_f64()));
            }
        });
    }
}

/// The blocks of a field's lattice for `W` lanes.
fn blocks_of<const W: usize>(lat: &crate::Lattice) -> Blocks {
    Blocks::new(lat.local, W).unwrap_or_else(|| {
        panic!(
            "the lattice {:?} cannot be cut into {W} sub-lattices",
            lat.local
        )
    })
}

/// From the usual layout to blocks of `W` sites, changing the precision if asked to.
impl<T, U, K, const W: usize> ConvertFrom<Field<U, K, Full, Aos, Cpu>>
    for Field<T, K, Full, Aosoa<W>, Cpu>
where
    T: Storage,
    U: Storage,
    K: Kind,
{
    fn convert_from(&mut self, src: &Field<U, K, Full, Aos, Cpu>) {
        assert_eq!(self.as_slice().len(), src.as_slice().len());
        assert_eq!(
            (self.site_len(), self.outer()),
            (src.site_len(), src.outer())
        );
        let blocks = blocks_of::<W>(self.lattice());
        let (block_len, slab_len, site_len) = (self.block_len(), self.slab_len(), self.site_len());
        for (o, slab) in self.as_mut_slice().chunks_exact_mut(slab_len).enumerate() {
            for_sites(slab, block_len, |block, d| {
                for lane in 0..W {
                    let from = src.site(o, blocks.site(block, lane));
                    for c in 0..site_len {
                        d[lane + W * c] = T::store(Number::from_f64(from[c].load().to_f64()));
                    }
                }
            });
        }
    }
}

/// From blocks of `W` sites back to the usual layout.
impl<T, U, K, const W: usize> ConvertFrom<Field<U, K, Full, Aosoa<W>, Cpu>>
    for Field<T, K, Full, Aos, Cpu>
where
    T: Storage,
    U: Storage,
    K: Kind,
{
    fn convert_from(&mut self, src: &Field<U, K, Full, Aosoa<W>, Cpu>) {
        assert_eq!(self.as_slice().len(), src.as_slice().len());
        assert_eq!(
            (self.site_len(), self.outer()),
            (src.site_len(), src.outer())
        );
        let blocks = blocks_of::<W>(self.lattice());
        let (block_len, slab_len, site_len) = (src.block_len(), self.slab_len(), self.site_len());
        for (o, slab) in self.as_mut_slice().chunks_exact_mut(slab_len).enumerate() {
            let from = src.slab(o);
            for_sites(slab, site_len, |site, d| {
                let (block, lane) = blocks.block_lane(site);
                let from = &from[block * block_len..(block + 1) * block_len];
                for c in 0..site_len {
                    d[c] = T::store(Number::from_f64(from[lane + W * c].load().to_f64()));
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Spinor;
    use crate::lattice::Lattice;

    fn sample(seed: usize, nv: usize) -> Field<f64, Spinor> {
        let lat = Lattice::single_node([4, 2, 2, 4]);
        let mut f = Field::<f64, Spinor>::spinor(&lat, nv);
        for (i, x) in f.as_mut_slice().iter_mut().enumerate() {
            *x = ((i * (7 + 2 * seed) + seed) % 23) as f64 - 11.0;
        }
        f
    }

    /// Vector `j` of a field as a plain list of complex numbers.
    fn column(f: &Field<f64, Spinor>, j: usize) -> Vec<C64> {
        let mut v = Vec::new();
        for site in 0..f.num_sites() {
            let s = &f.site(0, site)[j * 24..(j + 1) * 24];
            v.extend(s.as_chunks::<2>().0.iter().map(|z| Cplx::new(z[0], z[1])));
        }
        v
    }

    #[test]
    fn each_vector_gets_its_own_coefficient() {
        let (x, y0) = (sample(1, 3), sample(2, 3));
        let a = [
            Cplx::new(0.5, -1.0),
            Cplx::new(0.0, 0.0),
            Cplx::new(-2.0, 0.25),
        ];

        let mut y = y0.zeros_like();
        y.copy_from(&y0);
        y.axpy(&a, &x);
        let mut z = y0.zeros_like();
        z.copy_from(&y0);
        z.aypx(&a, &x);
        for j in 0..3 {
            let (xj, y0j) = (column(&x, j), column(&y0, j));
            for (k, got) in column(&y, j).iter().enumerate() {
                assert_eq!(*got, y0j[k] + a[j] * xj[k]);
            }
            for (k, got) in column(&z, j).iter().enumerate() {
                assert_eq!(*got, a[j] * y0j[k] + xj[k]);
            }
        }
        assert_eq!(column(&y, 1), column(&y0, 1));
    }

    #[test]
    fn dot_and_norm_are_per_vector() {
        let (x, y) = (sample(3, 2), sample(4, 2));
        let mut d = [Cplx::ZERO; 2];
        x.dot(&y, &mut d);
        let mut n = [0.0; 2];
        x.norm_sqr(&mut n);
        for j in 0..2 {
            let (xj, yj) = (column(&x, j), column(&y, j));
            let mut want = Cplx::ZERO;
            let mut norm = 0.0;
            for (a, b) in xj.iter().zip(&yj) {
                want += a.conj() * *b;
                norm += a.norm_sqr();
            }
            assert_eq!(d[j], want);
            assert_eq!(n[j], norm);
        }
        assert_ne!(n[0], n[1]);
    }

    #[test]
    fn scale_zero_and_convert() {
        let x = sample(5, 2);
        let mut y = x.zeros_like();
        y.copy_from(&x);
        y.scale(&[2.0, -0.5]);
        assert_eq!(column(&y, 0)[7], column(&x, 0)[7].scale(2.0));
        assert_eq!(column(&y, 1)[7], column(&x, 1)[7].scale(-0.5));

        let mut single = Field::<f32, Spinor>::spinor(x.lattice(), 2);
        single.convert_from(&x);
        let mut back = x.zeros_like();
        back.convert_from(&single);
        assert_eq!(back.as_slice(), x.as_slice());

        y.set_zero();
        assert!(y.as_slice().iter().all(|v| *v == 0.0));
    }
}
