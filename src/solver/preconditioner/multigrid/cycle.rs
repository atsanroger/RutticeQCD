//! The cycle of a two-grid method on any kind of vector.

use crate::solver::Chebyshev;
use crate::{fit, LinearFn, Vector, C64};

/// `y = S b; y += E(b − A y); y += S(b − A y)`: S a Chebyshev polynomial in A (the smoother,
/// which holds A), E the coarse correction, any map of a residual to a correction. A
/// preconditioner for a flexible solver on A.
pub struct TwoGrid<V, A, E> {
    smoother: Chebyshev<A, V>,
    correction: E,
    r: V,
    e: V,
}

impl<V: Vector, A: LinearFn<V>, E: FnMut(&mut V, &V)> TwoGrid<V, A, E> {
    pub fn new(smoother: Chebyshev<A, V>, correction: E, like: &V) -> Self {
        Self {
            smoother,
            correction,
            r: like.zeros_like(),
            e: like.zeros_like(),
        }
    }

    /// `r = b − A y`.
    fn residual(&mut self, y: &V, b: &V) {
        self.smoother.operator().mult(&mut self.r, y);
        self.r.aypx(&vec![C64::new(-1.0, 0.0); b.num_vectors()], b);
    }
}

impl<V: Vector, A: LinearFn<V>, E: FnMut(&mut V, &V)> LinearFn<V> for TwoGrid<V, A, E> {
    fn mult(&mut self, y: &mut V, b: &V) {
        let one = vec![C64::new(1.0, 0.0); b.num_vectors()];
        fit(&mut self.r, b);
        fit(&mut self.e, b);
        self.smoother.mult(y, b);
        self.residual(y, b);
        (self.correction)(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
        self.residual(y, b);
        self.smoother.mult(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
    }
}
