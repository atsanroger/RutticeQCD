//! Arithmetic on small by-value objects: reals, complex numbers, 3x3 colour matrices.
//!
//! No standard library, no allocation, no dynamic dispatch, so that the same code has a
//! chance of being accepted by a GPU code generator later (DESIGN.md §7.5).
//!
//! Written from the mathematics; not derived from any other code base.
#![no_std]

mod spin;

pub use spin::{
    hermitian6_mul, project, sink_spinor, transport, ColorVector, Dirac, Gamma, GammaRepre,
    HalfSpinor, LinkUse, Sign, Spinor, Unit, HERMITIAN6_REALS, NS, SPINOR_REALS,
};

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Number of colours. Fixed at compile time (DESIGN.md §10).
pub const NC: usize = 3;

/// A real type kernels compute with.
pub trait Real:
    Copy
    + PartialOrd
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cplx<T> {
    pub re: T,
    pub im: T,
}

impl<T: Real> Cplx<T> {
    pub const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
    pub const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };

    #[inline(always)]
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    #[inline(always)]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }
    #[inline(always)]
    pub fn scale(self, a: T) -> Self {
        Self {
            re: self.re * a,
            im: self.im * a,
        }
    }
}

impl<T: Real> Add for Cplx<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl<T: Real> Sub for Cplx<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

impl<T: Real> Mul for Cplx<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

impl<T: Real> Div for Cplx<T> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let d = o.norm_sqr();
        let n = self * o.conj();
        Self {
            re: n.re / d,
            im: n.im / d,
        }
    }
}

impl<T: Real> Neg for Cplx<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Real> AddAssign for Cplx<T> {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        self.re += o.re;
        self.im += o.im;
    }
}

/// A 3x3 complex matrix in colour space, row-major. Not necessarily in SU(3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix<T>(pub [[Cplx<T>; NC]; NC]);

impl<T: Real> ColorMatrix<T> {
    /// Number of reals in the full (uncompressed) representation.
    pub const REALS: usize = 2 * NC * NC;

    #[inline(always)]
    pub fn zero() -> Self {
        Self([[Cplx::ZERO; NC]; NC])
    }

    #[inline(always)]
    pub fn identity() -> Self {
        let mut m = Self::zero();
        for a in 0..NC {
            m.0[a][a] = Cplx::ONE;
        }
        m
    }

    /// Build from `2 * NC * NC` reals: element (a, b) is at `2 * (a * NC + b)`, real part first.
    #[inline(always)]
    pub fn from_reals(v: &[T]) -> Self {
        let mut m = Self::zero();
        for a in 0..NC {
            for b in 0..NC {
                let k = 2 * (a * NC + b);
                m.0[a][b] = Cplx::new(v[k], v[k + 1]);
            }
        }
        m
    }

    #[inline(always)]
    pub fn write_reals(&self, v: &mut [T]) {
        for a in 0..NC {
            for b in 0..NC {
                let k = 2 * (a * NC + b);
                v[k] = self.0[a][b].re;
                v[k + 1] = self.0[a][b].im;
            }
        }
    }

    /// Matrix product `self * o`.
    #[inline(always)]
    pub fn mul(&self, o: &Self) -> Self {
        let mut m = Self::zero();
        for a in 0..NC {
            for b in 0..NC {
                let mut s = Cplx::ZERO;
                for k in 0..NC {
                    s += self.0[a][k] * o.0[k][b];
                }
                m.0[a][b] = s;
            }
        }
        m
    }

    /// Hermitian conjugate.
    #[inline(always)]
    pub fn adj(&self) -> Self {
        let mut m = Self::zero();
        for a in 0..NC {
            for b in 0..NC {
                m.0[a][b] = self.0[b][a].conj();
            }
        }
        m
    }

    /// `self * v`.
    #[inline(always)]
    pub fn mul_vec(&self, v: &[Cplx<T>; NC]) -> [Cplx<T>; NC] {
        core::array::from_fn(|a| {
            let mut s = Cplx::ZERO;
            for b in 0..NC {
                s += self.0[a][b] * v[b];
            }
            s
        })
    }

    /// `self^† * v`, without forming the adjoint.
    #[inline(always)]
    pub fn adj_mul_vec(&self, v: &[Cplx<T>; NC]) -> [Cplx<T>; NC] {
        core::array::from_fn(|a| {
            let mut s = Cplx::ZERO;
            for b in 0..NC {
                s += self.0[b][a].conj() * v[b];
            }
            s
        })
    }

    #[inline(always)]
    pub fn add(&self, o: &Self) -> Self {
        Self(core::array::from_fn(|a| {
            core::array::from_fn(|b| self.0[a][b] + o.0[a][b])
        }))
    }

    #[inline(always)]
    pub fn sub(&self, o: &Self) -> Self {
        Self(core::array::from_fn(|a| {
            core::array::from_fn(|b| self.0[a][b] - o.0[a][b])
        }))
    }

    #[inline(always)]
    pub fn scale(&self, k: T) -> Self {
        Self(core::array::from_fn(|a| {
            core::array::from_fn(|b| self.0[a][b].scale(k))
        }))
    }

    #[inline(always)]
    pub fn trace(&self) -> Cplx<T> {
        let mut s = Cplx::ZERO;
        for a in 0..NC {
            s += self.0[a][a];
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ColorMatrix<f64> {
        let mut v = [0.0f64; 18];
        for (k, x) in v.iter_mut().enumerate() {
            *x = 0.1 * (k as f64 + 1.0) * if k % 3 == 0 { -1.0 } else { 1.0 };
        }
        ColorMatrix::from_reals(&v)
    }

    #[test]
    fn identity_is_neutral() {
        let a = sample();
        assert_eq!(a.mul(&ColorMatrix::identity()), a);
        assert_eq!(ColorMatrix::identity().mul(&a), a);
    }

    #[test]
    fn adjoint_of_product_reverses_order() {
        let a = sample();
        let b = a.mul(&a).adj();
        let lhs = a.mul(&b).adj();
        let rhs = b.adj().mul(&a.adj());
        for i in 0..NC {
            for j in 0..NC {
                assert!((lhs.0[i][j] - rhs.0[i][j]).norm_sqr() < 1e-24);
            }
        }
    }

    #[test]
    fn matrix_times_vector_matches_matrix_product() {
        let a = sample();
        let v = [
            Cplx::new(0.3, -1.1),
            Cplx::new(2.0, 0.5),
            Cplx::new(-0.7, 0.9),
        ];
        let mut col = ColorMatrix::zero();
        for (b, z) in v.iter().enumerate() {
            col.0[b][0] = *z;
        }
        let (p, q) = (a.mul(&col), a.adj().mul(&col));
        for b in 0..NC {
            assert_eq!(a.mul_vec(&v)[b], p.0[b][0]);
            assert_eq!(a.adj_mul_vec(&v)[b], q.0[b][0]);
        }
    }

    #[test]
    fn reals_round_trip() {
        let a = sample();
        let mut v = [0.0f64; 18];
        a.write_reals(&mut v);
        assert_eq!(ColorMatrix::from_reals(&v), a);
    }
}
