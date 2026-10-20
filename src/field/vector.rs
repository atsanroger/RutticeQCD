//! The only interface solvers need from a vector.

use crate::field::storage::Storage;
use crate::field::{Aos, Aosoa, Cpu, Field, Kind, Layout, Lexical, Sites};
use crate::kernel::{Cplx, Number, Real};
use crate::lattice::blocks::Blocks;
use crate::lattice::for_sites::{for_light_sites, reduce_sites_dyn, reduce_sites_mut};

/// Coefficients and reduction results are always double precision, whatever the field stores.
pub type C64 = Cplx<f64>;

/// A set of `num_vectors` vectors of equal length that are operated on together.
pub trait Vector: Sized {
    fn num_vectors(&self) -> usize;
    /// A zero vector of the same shape.
    fn zeros_like(&self) -> Self;
    /// A zero vector of the same shape except that it holds `num_vectors` vectors.
    fn zeros_with(&self, num_vectors: usize) -> Self;
    /// The vectors `which` of this set, in that order, as a set of their own.
    fn select(&self, which: &[usize]) -> Self;
    /// `self_{which[k]} = from_k`: put the vectors of a smaller set back.
    fn assign(&mut self, which: &[usize], from: &Self);
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
    // The operations below are compositions of the ones above.

    /// `self_j += a_j x_j`, then `out_j = |self_j|^2`.
    fn axpy_norm_sqr(&mut self, a: &[C64], x: &Self, out: &mut [f64]) {
        self.axpy(a, x);
        self.norm_sqr(out);
    }
    /// `self_j = x_j + a_j y_j`.
    fn xpay(&mut self, x: &Self, a: &[C64], y: &Self) {
        self.copy_from(x);
        self.axpy(a, y);
    }
    /// `self_j = x_j + a_j y_j`, then `norm_j = |self_j|^2` and `dot_j = <z_j | self_j>`.
    fn xpay_norm_sqr_dot(
        &mut self,
        x: &Self,
        a: &[C64],
        y: &Self,
        norm: &mut [f64],
        z: &Self,
        dot: &mut [C64],
    ) {
        self.xpay(x, a, y);
        self.norm_sqr(norm);
        z.dot(self, dot);
    }
    /// `self_j += a_j x_j + b_j y_j`.
    fn axpy2(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        self.axpy(a, x);
        self.axpy(b, y);
    }
    /// `self_j = c_j self_j + a_j x_j + b_j y_j`, then `out_j = |self_j|^2`: a step of a
    /// Chebyshev recurrence.
    fn scale_axpy2_norm_sqr(
        &mut self,
        c: &[f64],
        a: &[C64],
        x: &Self,
        b: &[C64],
        y: &Self,
        out: &mut [f64],
    ) {
        self.scale(c);
        self.axpy2(a, x, b, y);
        self.norm_sqr(out);
    }
    /// `self_j = y_j + b_j (self_j + a_j x_j)`.
    fn axpy_aypx(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        self.axpy(a, x);
        self.aypx(b, y);
    }
    /// `self_j += d_j`, `r_j −= ad_j`, then `d_j = c_j d_j + a_j r_j`: a step of the Chebyshev
    /// iteration for A x = b, with `self` the iterate, `r` the residual and `ad` = A d.
    fn chebyshev_step(&mut self, d: &mut Self, r: &mut Self, ad: &Self, c: &[f64], a: &[C64]) {
        let nv = self.num_vectors();
        self.axpy(&vec![C64::new(1.0, 0.0); nv], d);
        r.axpy(&vec![C64::new(-1.0, 0.0); nv], ad);
        d.scale(c);
        d.axpy(a, r);
    }
    /// `dot_j = <self_j | x_j>`, `norm_j = |self_j|^2` and `norm_x_j = |x_j|^2`.
    fn dot_norms(&self, x: &Self, dot: &mut [C64], norm: &mut [f64], norm_x: &mut [f64]) {
        self.dot(x, dot);
        self.norm_sqr(norm);
        x.norm_sqr(norm_x);
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
    for_light_sites(dst, site_len, |i, d| {
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

/// `update` with a second source: `f(vector index, dst, first, second)`.
fn update2<T, F>(dst: &mut [T], src: [&[T]; 2], site_len: usize, vector_len: usize, f: F)
where
    T: Storage,
    F: Fn(usize, &mut [T], &[T], &[T]) + Sync,
{
    assert!(dst.len() == src[0].len() && dst.len() == src[1].len());
    for_light_sites(dst, site_len, |i, d| {
        let range = i * site_len..(i + 1) * site_len;
        let (s0, s1) = (&src[0][range.clone()], &src[1][range]);
        let sources = s0.chunks_exact(vector_len).zip(s1.chunks_exact(vector_len));
        for (j, (d, (s0, s1))) in d.chunks_exact_mut(vector_len).zip(sources).enumerate() {
            f(j, d, s0, s1);
        }
    });
}

// The arithmetic of the vector operations on one vector in one block. Each is used by the
// operation of its name and by the fused operations, which is what makes a fused operation
// give the bits of the composition it stands for.

/// `d += k s`.
#[inline(always)]
fn axpy_block<T: Storage>(d: &mut [T], k: Cplx<T::Compute>, s: &[T], lanes: usize) {
    for ((d_re, d_im), (s_re, s_im)) in complex_runs_mut(d, lanes).zip(complex_runs(s, lanes)) {
        for l in 0..lanes {
            let z = k * Cplx::new(s_re[l].load(), s_im[l].load());
            d_re[l] = T::store(d_re[l].load() + z.re);
            d_im[l] = T::store(d_im[l].load() + z.im);
        }
    }
}

/// `d = k d + s`.
#[inline(always)]
fn aypx_block<T: Storage>(d: &mut [T], k: Cplx<T::Compute>, s: &[T], lanes: usize) {
    for ((d_re, d_im), (s_re, s_im)) in complex_runs_mut(d, lanes).zip(complex_runs(s, lanes)) {
        for l in 0..lanes {
            let z = k * Cplx::new(d_re[l].load(), d_im[l].load());
            d_re[l] = T::store(z.re + s_re[l].load());
            d_im[l] = T::store(z.im + s_im[l].load());
        }
    }
}

/// `acc += |a|^2`.
#[inline(always)]
fn norm_block<T: Storage>(acc: &mut f64, a: &[T]) {
    for x in a {
        let x = x.load().to_f64();
        *acc += x * x;
    }
}

/// `(re, im) += <a | b>`.
#[inline(always)]
fn dot_block<T: Storage>(re: &mut f64, im: &mut f64, a: &[T], b: &[T], lanes: usize) {
    for ((a_re, a_im), (b_re, b_im)) in complex_runs(a, lanes).zip(complex_runs(b, lanes)) {
        for l in 0..lanes {
            let (ar, ai) = (a_re[l].load().to_f64(), a_im[l].load().to_f64());
            let (br, bi) = (b_re[l].load().to_f64(), b_im[l].load().to_f64());
            *re += ar * br + ai * bi;
            *im += ar * bi - ai * br;
        }
    }
}

impl<T: Storage, K: Kind, S: Sites, L: Layout> Vector for Field<T, K, S, L, Cpu> {
    fn num_vectors(&self) -> usize {
        Field::num_vectors(self)
    }

    fn zeros_like(&self) -> Self {
        self.zeros_with(Field::num_vectors(self))
    }

    fn zeros_with(&self, num_vectors: usize) -> Self {
        let mut z = Field::with_shape(
            self.lattice(),
            self.internal_dof(),
            self.ls(),
            num_vectors,
            self.outer(),
        );
        z.copy_parity_from(self);
        z
    }

    fn select(&self, which: &[usize]) -> Self {
        assert!(which.iter().all(|j| *j < Field::num_vectors(self)));
        let mut out = self.zeros_with(which.len());
        let (len, from_len, to_len) = (vector_len(self), self.block_len(), out.block_len());
        let from = self.as_slice();
        for_light_sites(out.as_mut_slice(), to_len, |i, to| {
            let from = &from[i * from_len..(i + 1) * from_len];
            for (to, j) in to.chunks_exact_mut(len).zip(which) {
                to.copy_from_slice(&from[j * len..(j + 1) * len]);
            }
        });
        out
    }

    fn assign(&mut self, which: &[usize], from: &Self) {
        assert_eq!(which.len(), Field::num_vectors(from));
        assert!(which.iter().all(|j| *j < Field::num_vectors(self)));
        // The same kind of vector, only fewer of them.
        assert_eq!(
            from.as_slice().len() * Field::num_vectors(self),
            self.as_slice().len() * which.len()
        );
        let (len, to_len, from_len) = (vector_len(self), self.block_len(), from.block_len());
        let from = from.as_slice();
        for_light_sites(self.as_mut_slice(), to_len, |i, to| {
            let from = &from[i * from_len..(i + 1) * from_len];
            for (from, j) in from.chunks_exact(len).zip(which) {
                to[j * len..(j + 1) * len].copy_from_slice(from);
            }
        });
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
        for_light_sites(self.as_mut_slice(), site_len, |_, d| d.fill(T::default()));
    }

    fn scale(&mut self, a: &[f64]) {
        assert_eq!(a.len(), Field::num_vectors(self));
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        for_light_sites(self.as_mut_slice(), site_len, |_, d| {
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
            |j, d, s| axpy_block(d, coefficient(a[j]), s, L::LANES),
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
            |j, d, s| aypx_block(d, coefficient(a[j]), s, L::LANES),
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
                let [re, im] = &mut acc[2 * j..2 * j + 2] else {
                    unreachable!("two sums per vector");
                };
                dot_block(re, im, a, b, L::LANES);
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
                norm_block(&mut acc[j], a);
            }
        });
        out.copy_from_slice(&sums);
        self.lattice().comm.sum(out);
    }

    fn axpy_norm_sqr(&mut self, a: &[C64], x: &Self, out: &mut [f64]) {
        assert!(self.same_shape(x));
        let nv = Field::num_vectors(self);
        assert!(a.len() == nv && out.len() == nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let x = x.as_slice();
        let sums = reduce_sites_mut(self.as_mut_slice(), site_len, nv, |i, d, acc| {
            let x = &x[i * site_len..(i + 1) * site_len];
            let vectors = d
                .chunks_exact_mut(vector_len)
                .zip(x.chunks_exact(vector_len));
            for (j, (d, x)) in vectors.enumerate() {
                axpy_block(d, coefficient(a[j]), x, L::LANES);
                norm_block(&mut acc[j], d);
            }
        });
        out.copy_from_slice(&sums);
        self.lattice().comm.sum(out);
    }

    fn xpay(&mut self, x: &Self, a: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        assert_eq!(a.len(), Field::num_vectors(self));
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let sources = [x.as_slice(), y.as_slice()];
        update2(
            self.as_mut_slice(),
            sources,
            site_len,
            vector_len,
            |j, d, x, y| {
                d.copy_from_slice(x);
                axpy_block(d, coefficient(a[j]), y, L::LANES);
            },
        );
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
        let nv = Field::num_vectors(self);
        assert!(a.len() == nv && norm.len() == nv && dot.len() == nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let (x, y, z) = (x.as_slice(), y.as_slice(), z.as_slice());
        // Per vector: |self|^2, then the real and the imaginary part of <z | self>.
        let mut sums = reduce_sites_mut(self.as_mut_slice(), site_len, 3 * nv, |i, d, acc| {
            let range = i * site_len..(i + 1) * site_len;
            let (x, y, z) = (&x[range.clone()], &y[range.clone()], &z[range]);
            for (j, d) in d.chunks_exact_mut(vector_len).enumerate() {
                let at = j * vector_len..(j + 1) * vector_len;
                d.copy_from_slice(&x[at.clone()]);
                axpy_block(d, coefficient(a[j]), &y[at.clone()], L::LANES);
                let [n, re, im] = &mut acc[3 * j..3 * j + 3] else {
                    unreachable!("three sums per vector");
                };
                norm_block(n, d);
                dot_block(re, im, &z[at], d, L::LANES);
            }
        });
        self.lattice().comm.sum(&mut sums);
        for (j, s) in sums.as_chunks::<3>().0.iter().enumerate() {
            norm[j] = s[0];
            dot[j] = Cplx::new(s[1], s[2]);
        }
    }

    fn axpy2(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        let nv = Field::num_vectors(self);
        assert!(a.len() == nv && b.len() == nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let sources = [x.as_slice(), y.as_slice()];
        update2(
            self.as_mut_slice(),
            sources,
            site_len,
            vector_len,
            |j, d, x, y| {
                axpy_block(d, coefficient(a[j]), x, L::LANES);
                axpy_block(d, coefficient(b[j]), y, L::LANES);
            },
        );
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
        let nv = Field::num_vectors(self);
        assert!(c.len() == nv && a.len() == nv && b.len() == nv && out.len() == nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let (x, y) = (x.as_slice(), y.as_slice());
        let sums = reduce_sites_mut(self.as_mut_slice(), site_len, nv, |i, d, acc| {
            let range = i * site_len..(i + 1) * site_len;
            let (x, y) = (&x[range.clone()], &y[range]);
            for (j, d) in d.chunks_exact_mut(vector_len).enumerate() {
                let at = j * vector_len..(j + 1) * vector_len;
                let k = T::Compute::from_f64(c[j]);
                for v in d.iter_mut() {
                    *v = T::store(v.load() * k);
                }
                axpy_block(d, coefficient(a[j]), &x[at.clone()], L::LANES);
                axpy_block(d, coefficient(b[j]), &y[at], L::LANES);
                norm_block(&mut acc[j], d);
            }
        });
        out.copy_from_slice(&sums);
        self.lattice().comm.sum(out);
    }

    fn axpy_aypx(&mut self, a: &[C64], x: &Self, b: &[C64], y: &Self) {
        assert!(self.same_shape(x) && self.same_shape(y));
        let nv = Field::num_vectors(self);
        assert!(a.len() == nv && b.len() == nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let sources = [x.as_slice(), y.as_slice()];
        update2(
            self.as_mut_slice(),
            sources,
            site_len,
            vector_len,
            |j, d, x, y| {
                axpy_block(d, coefficient(a[j]), x, L::LANES);
                aypx_block(d, coefficient(b[j]), y, L::LANES);
            },
        );
    }

    fn dot_norms(&self, x: &Self, dot: &mut [C64], norm: &mut [f64], norm_x: &mut [f64]) {
        assert!(self.same_shape(x));
        let nv = Field::num_vectors(self);
        assert!(dot.len() == nv && norm.len() == nv && norm_x.len() == nv);
        let (site_len, vector_len) = (self.block_len(), vector_len(self));
        let (a, b) = (self.as_slice(), x.as_slice());
        // Per vector: the real and the imaginary part of <self | x>, |self|^2, |x|^2.
        let mut sums = reduce_sites_dyn(a.len() / site_len, 4 * nv, |i, acc| {
            let range = i * site_len..(i + 1) * site_len;
            let (a, b) = (&a[range.clone()], &b[range]);
            let vectors = a.chunks_exact(vector_len).zip(b.chunks_exact(vector_len));
            for (j, (a, b)) in vectors.enumerate() {
                let [re, im, na, nb] = &mut acc[4 * j..4 * j + 4] else {
                    unreachable!("four sums per vector");
                };
                dot_block(re, im, a, b, L::LANES);
                norm_block(na, a);
                norm_block(nb, b);
            }
        });
        self.lattice().comm.sum(&mut sums);
        for (j, s) in sums.as_chunks::<4>().0.iter().enumerate() {
            dot[j] = Cplx::new(s[0], s[1]);
            norm[j] = s[2];
            norm_x[j] = s[3];
        }
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
        for_light_sites(self.as_mut_slice(), site_len, |i, d| {
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
impl<T, U, K, const W: usize> ConvertFrom<Field<U, K, Lexical, Aos, Cpu>>
    for Field<T, K, Lexical, Aosoa<W>, Cpu>
where
    T: Storage,
    U: Storage,
    K: Kind,
{
    fn convert_from(&mut self, src: &Field<U, K, Lexical, Aos, Cpu>) {
        assert_eq!(self.as_slice().len(), src.as_slice().len());
        assert_eq!(
            (self.site_len(), self.outer()),
            (src.site_len(), src.outer())
        );
        let blocks = blocks_of::<W>(self.lattice());
        let (block_len, slab_len, site_len) = (self.block_len(), self.slab_len(), self.site_len());
        for (o, slab) in self.as_mut_slice().chunks_exact_mut(slab_len).enumerate() {
            for_light_sites(slab, block_len, |block, d| {
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
impl<T, U, K, const W: usize> ConvertFrom<Field<U, K, Lexical, Aosoa<W>, Cpu>>
    for Field<T, K, Lexical, Aos, Cpu>
where
    T: Storage,
    U: Storage,
    K: Kind,
{
    fn convert_from(&mut self, src: &Field<U, K, Lexical, Aosoa<W>, Cpu>) {
        assert_eq!(self.as_slice().len(), src.as_slice().len());
        assert_eq!(
            (self.site_len(), self.outer()),
            (src.site_len(), src.outer())
        );
        let blocks = blocks_of::<W>(self.lattice());
        let (block_len, slab_len, site_len) = (src.block_len(), self.slab_len(), self.site_len());
        for (o, slab) in self.as_mut_slice().chunks_exact_mut(slab_len).enumerate() {
            let from = src.slab(o);
            for_light_sites(slab, site_len, |site, d| {
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
