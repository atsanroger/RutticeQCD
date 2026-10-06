//! Spinors, gamma matrices, and the two halves of a hopping term: transport and sink
//! (DESIGN.md §6.1).
//!
//! The Dirac-representation matrices use the same convention as Bridge++ so that results
//! can be compared; only the matrix entries are taken from there.

use crate::{ColorMatrix, Cplx, Real, NC};

/// Number of spin components.
pub const NS: usize = 4;
/// Reals in one spinor.
pub const SPINOR_REALS: usize = 2 * NC * NS;

pub type ColorVector<T> = [Cplx<T>; NC];

/// Spin outermost, colour inside: component (s, c) is at real index `2 * (c + NC * s)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spinor<T>(pub [ColorVector<T>; NS]);

/// The two independent spin components of a projected spinor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfSpinor<T>(pub [ColorVector<T>; 2]);

impl<T: Real> Spinor<T> {
    #[inline(always)]
    pub fn zero() -> Self {
        Self([[Cplx::ZERO; NC]; NS])
    }

    #[inline(always)]
    pub fn from_reals(v: &[T]) -> Self {
        let mut p = Self::zero();
        for s in 0..NS {
            for c in 0..NC {
                let k = 2 * (c + NC * s);
                p.0[s][c] = Cplx::new(v[k], v[k + 1]);
            }
        }
        p
    }

    #[inline(always)]
    pub fn write_reals(&self, v: &mut [T]) {
        for s in 0..NS {
            for c in 0..NC {
                let k = 2 * (c + NC * s);
                v[k] = self.0[s][c].re;
                v[k + 1] = self.0[s][c].im;
            }
        }
    }

    #[inline(always)]
    pub fn add(&self, o: &Self) -> Self {
        let mut p = *self;
        for s in 0..NS {
            for c in 0..NC {
                p.0[s][c] += o.0[s][c];
            }
        }
        p
    }

    /// `self - k * o`.
    #[inline(always)]
    pub fn sub_scaled(&self, k: T, o: &Self) -> Self {
        let mut p = *self;
        for s in 0..NS {
            for c in 0..NC {
                p.0[s][c] = p.0[s][c] - o.0[s][c].scale(k);
            }
        }
        p
    }

    /// Multiply every spin component by a colour matrix.
    #[inline(always)]
    pub fn color_mul(&self, m: &ColorMatrix<T>) -> Self {
        Self(core::array::from_fn(|s| m.mul_vec(&self.0[s])))
    }
}

impl<T: Real> HalfSpinor<T> {
    #[inline(always)]
    pub fn scale_cplx(&self, z: Cplx<T>) -> Self {
        Self(core::array::from_fn(|k| {
            core::array::from_fn(|c| z * self.0[k][c])
        }))
    }
}

/// A fourth root of unity. Multiplying by one needs no floating-point multiplication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    One,
    I,
    MinusOne,
    MinusI,
}

impl Unit {
    const fn power(self) -> u8 {
        match self {
            Unit::One => 0,
            Unit::I => 1,
            Unit::MinusOne => 2,
            Unit::MinusI => 3,
        }
    }
    const fn from_power(k: u8) -> Self {
        match k % 4 {
            0 => Unit::One,
            1 => Unit::I,
            2 => Unit::MinusOne,
            _ => Unit::MinusI,
        }
    }
    pub const fn mul(self, o: Self) -> Self {
        Self::from_power(self.power() + o.power())
    }
    pub const fn neg(self) -> Self {
        Self::from_power(self.power() + 2)
    }
    pub const fn conj(self) -> Self {
        Self::from_power(4 - self.power())
    }

    #[inline(always)]
    pub fn times<T: Real>(self, z: Cplx<T>) -> Cplx<T> {
        match self {
            Unit::One => z,
            Unit::I => Cplx::new(-z.im, z.re),
            Unit::MinusOne => -z,
            Unit::MinusI => Cplx::new(z.im, -z.re),
        }
    }
}

/// A gamma matrix with exactly one non-zero entry per row: `(γ ψ)_a = val[a] * ψ_{col[a]}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gamma {
    pub col: [usize; NS],
    pub val: [Unit; NS],
}

impl Gamma {
    pub const IDENTITY: Gamma = Gamma {
        col: [0, 1, 2, 3],
        val: [Unit::One; NS],
    };

    /// The matrix product `self * o`.
    pub fn mul(&self, o: &Gamma) -> Gamma {
        // (a b psi)_r = a.val[r] * b.val[a.col[r]] * psi_{b.col[a.col[r]]}
        Gamma {
            col: core::array::from_fn(|r| o.col[self.col[r]]),
            val: core::array::from_fn(|r| self.val[r].mul(o.val[self.col[r]])),
        }
    }

    #[inline(always)]
    pub fn apply<T: Real>(&self, psi: &Spinor<T>) -> Spinor<T> {
        Spinor(core::array::from_fn(|a| {
            core::array::from_fn(|c| self.val[a].times(psi.0[self.col[a]][c]))
        }))
    }
}

/// A choice of gamma matrices. `GAMMA[mu]` for the four directions (x, y, z, t).
pub trait GammaRepre: Copy + Send + Sync + 'static {
    const GAMMA: [Gamma; 4];
    const GAMMA5: Gamma;
}

/// Dirac representation: γ4 diagonal.
#[derive(Clone, Copy, Debug)]
pub struct Dirac;

impl GammaRepre for Dirac {
    const GAMMA: [Gamma; 4] = [
        Gamma {
            col: [3, 2, 1, 0],
            val: [Unit::MinusI, Unit::MinusI, Unit::I, Unit::I],
        },
        Gamma {
            col: [3, 2, 1, 0],
            val: [Unit::MinusOne, Unit::One, Unit::One, Unit::MinusOne],
        },
        Gamma {
            col: [2, 3, 0, 1],
            val: [Unit::MinusI, Unit::I, Unit::I, Unit::MinusI],
        },
        Gamma {
            col: [0, 1, 2, 3],
            val: [Unit::One, Unit::One, Unit::MinusOne, Unit::MinusOne],
        },
    ];
    const GAMMA5: Gamma = Gamma {
        col: [2, 3, 0, 1],
        val: [Unit::One; NS],
    };
}

/// Selects the projector `1 + sign * γ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Minus,
    Plus,
}

impl Sign {
    #[inline(always)]
    fn times(self, u: Unit) -> Unit {
        match self {
            Sign::Plus => u,
            Sign::Minus => u.neg(),
        }
    }
}

/// Whether a link is used as it is or Hermitian-conjugated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkUse {
    Direct,
    Adjoint,
}

/// For a diagonal gamma, the two rows that `1 + sign * γ` does not annihilate.
#[inline(always)]
fn surviving_rows(g: &Gamma, sign: Sign) -> [usize; 2] {
    let mut rows = [0; 2];
    let mut n = 0;
    for a in 0..NS {
        if sign.times(g.val[a]) == Unit::One && n < 2 {
            rows[n] = a;
            n += 1;
        }
    }
    rows
}

/// The two independent components of `(1 + sign * γ) ψ`.
///
/// `1 + sign * γ` has rank two. For an off-diagonal γ, rows 0 and 1 determine the other
/// two; for a diagonal γ, two rows vanish and the other two are `2 ψ`.
#[inline(always)]
pub fn project<T: Real>(g: &Gamma, sign: Sign, psi: &Spinor<T>) -> HalfSpinor<T> {
    if g.col[0] != 0 {
        HalfSpinor(core::array::from_fn(|k| {
            let u = sign.times(g.val[k]);
            core::array::from_fn(|c| psi.0[k][c] + u.times(psi.0[g.col[k]][c]))
        }))
    } else {
        let rows = surviving_rows(g, sign);
        HalfSpinor(core::array::from_fn(|k| {
            core::array::from_fn(|c| psi.0[rows[k]][c] + psi.0[rows[k]][c])
        }))
    }
}

/// Transport: bring the neighbour `psi` to this site along one direction. Projects, applies
/// the boundary phase if there is one, and multiplies by the link. Does not touch any output.
#[inline(always)]
pub fn transport<T: Real>(
    g: &Gamma,
    sign: Sign,
    psi: &Spinor<T>,
    link: &ColorMatrix<T>,
    link_use: LinkUse,
    phase: Option<Cplx<T>>,
) -> HalfSpinor<T> {
    let mut h = project(g, sign, psi);
    if let Some(z) = phase {
        h = h.scale_cplx(z);
    }
    HalfSpinor(core::array::from_fn(|k| match link_use {
        LinkUse::Direct => link.mul_vec(&h.0[k]),
        LinkUse::Adjoint => link.adj_mul_vec(&h.0[k]),
    }))
}

/// Sink for Dslash and hopping terms: rebuild the four-component spinor from a transported
/// half spinor and add it to `acc`. Must be given the same `g` and `sign` as the transport.
#[inline(always)]
pub fn sink_spinor<T: Real>(acc: &mut Spinor<T>, h: &HalfSpinor<T>, g: &Gamma, sign: Sign) {
    if g.col[0] != 0 {
        for k in 0..2 {
            let b = g.col[k];
            let u = sign.times(g.val[b]);
            for c in 0..NC {
                acc.0[k][c] += h.0[k][c];
                acc.0[b][c] += u.times(h.0[k][c]);
            }
        }
    } else {
        let rows = surviving_rows(g, sign);
        for k in 0..2 {
            for c in 0..NC {
                acc.0[rows[k]][c] += h.0[k][c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Spinor<f64> {
        let mut v = [0.0f64; SPINOR_REALS];
        for (k, x) in v.iter_mut().enumerate() {
            *x = ((k * 7 + 3) % 11) as f64 - 4.5 + 0.25 * k as f64;
        }
        Spinor::from_reals(&v)
    }

    fn compose(a: &Gamma, b: &Gamma) -> Gamma {
        // (a b ψ)_r = a.val[r] * b.val[a.col[r]] * ψ_{b.col[a.col[r]]}
        Gamma {
            col: core::array::from_fn(|r| b.col[a.col[r]]),
            val: core::array::from_fn(|r| a.val[r].mul(b.val[a.col[r]])),
        }
    }

    const IDENTITY: Gamma = Gamma {
        col: [0, 1, 2, 3],
        val: [Unit::One; NS],
    };

    fn negated(g: &Gamma) -> Gamma {
        Gamma {
            col: g.col,
            val: core::array::from_fn(|r| g.val[r].neg()),
        }
    }

    #[test]
    fn dirac_matrices_satisfy_the_clifford_algebra() {
        let g = Dirac::GAMMA;
        for mu in 0..4 {
            assert_eq!(compose(&g[mu], &g[mu]), IDENTITY, "gamma_{mu}^2");
            for nu in 0..4 {
                if mu != nu {
                    assert_eq!(compose(&g[mu], &g[nu]), negated(&compose(&g[nu], &g[mu])));
                }
            }
            // Hermitian: val[col[r]] at row col[r] is the conjugate of val[r].
            for r in 0..NS {
                assert_eq!(g[mu].col[g[mu].col[r]], r);
                assert_eq!(g[mu].val[g[mu].col[r]], g[mu].val[r].conj());
            }
            assert_eq!(
                compose(&Dirac::GAMMA5, &g[mu]),
                negated(&compose(&g[mu], &Dirac::GAMMA5))
            );
        }
        assert_eq!(compose(&Dirac::GAMMA5, &Dirac::GAMMA5), IDENTITY);
    }

    #[test]
    fn project_then_sink_is_one_plus_sign_gamma() {
        let psi = sample();
        for g in &Dirac::GAMMA {
            for sign in [Sign::Minus, Sign::Plus] {
                let mut got = Spinor::zero();
                sink_spinor(&mut got, &project(g, sign, &psi), g, sign);
                let gpsi = g.apply(&psi);
                let k = if sign == Sign::Plus { -1.0 } else { 1.0 };
                let want = psi.sub_scaled(k, &gpsi);
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn transport_with_identity_link_is_projection() {
        let psi = sample();
        let id = ColorMatrix::identity();
        let g = &Dirac::GAMMA[2];
        for link_use in [LinkUse::Direct, LinkUse::Adjoint] {
            assert_eq!(
                transport(g, Sign::Plus, &psi, &id, link_use, None),
                project(g, Sign::Plus, &psi)
            );
        }
        let minus = transport(
            g,
            Sign::Plus,
            &psi,
            &id,
            LinkUse::Direct,
            Some(Cplx::new(-1.0, 0.0)),
        );
        let plain = project(g, Sign::Plus, &psi);
        for k in 0..2 {
            for c in 0..NC {
                assert_eq!(minus.0[k][c], -plain.0[k][c]);
            }
        }
    }

    #[test]
    fn reals_round_trip() {
        let psi = sample();
        let mut v = [0.0f64; SPINOR_REALS];
        psi.write_reals(&mut v);
        assert_eq!(Spinor::from_reals(&v), psi);
        // spin 2, colour 1, imaginary part
        assert_eq!(psi.0[2][1].im, v[2 * (1 + 3 * 2) + 1]);
    }
}
