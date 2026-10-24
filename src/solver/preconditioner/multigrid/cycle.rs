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

/// `y = S b; y += E(b − A y); y += S(b − A y)`: S a Chebyshev polynomial in A (the smoother,
/// which holds A), E the coarse correction. A preconditioner for a flexible solver on A.
pub struct TwoGrid<V, A, E> {
    smoother: Chebyshev<A, V>,
    correction: E,
    r: V,
    e: V,
}

impl<V: Vector, A: LinearFn<V>, E: Corrects<V>> TwoGrid<V, A, E> {
    pub fn new(smoother: Chebyshev<A, V>, correction: E, like: &V) -> Self {
        Self {
            smoother,
            correction,
            r: like.zeros_like(),
            e: like.zeros_like(),
        }
    }

    pub fn smoother_mut(&mut self) -> &mut Chebyshev<A, V> {
        &mut self.smoother
    }

    pub fn correction(&self) -> &E {
        &self.correction
    }

    pub fn correction_mut(&mut self) -> &mut E {
        &mut self.correction
    }
}

impl<V: Vector, A: LinearFn<V>, E: Corrects<V>> LinearFn<V> for TwoGrid<V, A, E> {
    fn mult(&mut self, y: &mut V, b: &V) {
        let nv = b.num_vectors();
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus = vec![C64::new(-1.0, 0.0); nv];
        fit(&mut self.r, b);
        fit(&mut self.e, b);
        self.smoother.mult(y, b);
        // r = b − A y: the operator may have it in one sweep.
        self.smoother
            .operator()
            .mult_xpay(&mut self.r, y, &minus, b);
        self.correction.correct(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
        self.smoother
            .operator()
            .mult_xpay(&mut self.r, y, &minus, b);
        self.smoother.mult(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
    }
}
