//! Linear maps and how to combine them (DESIGN.md §5.1, §7.2).

use crate::vector::{Vector, C64};

/// Takes a vector, gives a vector. Operators, preconditioners, smoothers and solvers bound
/// to an operator are all this.
pub trait LinearFn<V: Vector> {
    /// `out = A inp`.
    fn mult(&mut self, out: &mut V, inp: &V);

    /// `out = A inp` and `dot_j = <inp_j | out_j>`. A fused kernel may do both in one sweep.
    fn mult_dot(&mut self, out: &mut V, inp: &V, dot: &mut [C64]) {
        self.mult(out, inp);
        inp.dot(out, dot);
    }

    /// `out_j = b_j + a_j (A inp)_j`.
    fn mult_xpay(&mut self, out: &mut V, inp: &V, a: &[C64], b: &V) {
        self.mult(out, inp);
        out.aypx(a, b);
    }
}

/// A linear map that also knows its Hermitian conjugate.
pub trait LinearOp<V: Vector>: LinearFn<V> {
    /// `out = A^† inp`.
    fn mult_dag(&mut self, out: &mut V, inp: &V);
}

/// `A B`. Its conjugate is `B^† A^†`.
pub struct Product<A, B, V> {
    a: A,
    b: B,
    tmp: V,
}

impl<A, B, V: Vector> Product<A, B, V> {
    /// `like` only provides the shape of the intermediate vector.
    pub fn new(a: A, b: B, like: &V) -> Self {
        Self {
            a,
            b,
            tmp: like.zeros_like(),
        }
    }
}

impl<A: LinearFn<V>, B: LinearFn<V>, V: Vector> LinearFn<V> for Product<A, B, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.b.mult(&mut self.tmp, inp);
        self.a.mult(out, &self.tmp);
    }
}

impl<A: LinearOp<V>, B: LinearOp<V>, V: Vector> LinearOp<V> for Product<A, B, V> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.a.mult_dag(&mut self.tmp, inp);
        self.b.mult_dag(out, &self.tmp);
    }
}

/// `A^†`.
pub struct Adjoint<A>(pub A);

impl<A: LinearOp<V>, V: Vector> LinearFn<V> for Adjoint<A> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.0.mult_dag(out, inp);
    }
}

impl<A: LinearOp<V>, V: Vector> LinearOp<V> for Adjoint<A> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.0.mult(out, inp);
    }
}
