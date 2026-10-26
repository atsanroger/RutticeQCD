//! The cycle of a two-grid method on any kind of vector.

use crate::solver::Chebyshev;
use crate::{fit, LinearFn, Vector, C64};

/// A coarse correction: `e` from a residual `r`.
pub trait Corrects<V> {
    fn correct(&mut self, e: &mut V, r: &V);
}

impl<V, F: FnMut(&mut V, &V)> Corrects<V> for F {
    fn correct(&mut self, e: &mut V, r: &V) {
        self(e, r);
    }
}

/// The smoothing of a cycle on the operator A, and the residual of A.
pub trait Smooths<V> {
    /// y from b before the coarse correction.
    fn pre(&mut self, y: &mut V, b: &V);
    /// r = b − A y.
    fn residual(&mut self, r: &mut V, y: &V, b: &V);
    /// e ≈ A⁻¹ r after the coarse correction.
    fn post(&mut self, e: &mut V, r: &V);
}

/// A Chebyshev polynomial in A smooths before and after; it holds A.
impl<V: Vector, A: LinearFn<V>> Smooths<V> for Chebyshev<A, V> {
    fn pre(&mut self, y: &mut V, b: &V) {
        self.mult(y, b);
    }

    fn residual(&mut self, r: &mut V, y: &V, b: &V) {
        // The operator may have it in one sweep.
        let minus = vec![C64::new(-1.0, 0.0); b.num_vectors()];
        self.operator().mult_xpay(r, y, &minus, b);
    }

    fn post(&mut self, e: &mut V, r: &V) {
        self.mult(e, r);
    }
}

/// `y = S₁ b; y += E(b − A y); y += S₂(b − A y)`: S₁ and S₂ the smoothing, E the coarse
/// correction. A preconditioner for a flexible solver on A.
pub struct TwoGrid<V, S, E> {
    smoother: S,
    correction: E,
    r: V,
    e: V,
}

impl<V: Vector, S: Smooths<V>, E: Corrects<V>> TwoGrid<V, S, E> {
    pub fn new(smoother: S, correction: E, like: &V) -> Self {
        Self {
            smoother,
            correction,
            r: like.zeros_like(),
            e: like.zeros_like(),
        }
    }

    pub(crate) fn smoother_mut(&mut self) -> &mut S {
        &mut self.smoother
    }

    pub fn correction(&self) -> &E {
        &self.correction
    }

    pub(crate) fn correction_mut(&mut self) -> &mut E {
        &mut self.correction
    }
}

impl<V: Vector, S: Smooths<V>, E: Corrects<V>> LinearFn<V> for TwoGrid<V, S, E> {
    fn mult(&mut self, y: &mut V, b: &V) {
        assert!(y.same_shape(b));
        let one = vec![C64::new(1.0, 0.0); b.num_vectors()];
        fit(&mut self.r, b);
        fit(&mut self.e, b);
        self.smoother.pre(y, b);
        self.smoother.residual(&mut self.r, y, b);
        self.correction.correct(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
        self.smoother.residual(&mut self.r, y, b);
        self.smoother.post(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
    }
}
