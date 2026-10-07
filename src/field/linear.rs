//! Linear maps and how to combine them.

use crate::field::vector::{Vector, C64};

/// Takes a vector, gives a vector. Operators, preconditioners, smoothers and solvers bound
/// to an operator are all this.
///
/// A map used as the operator of a solver must take sets of any number of vectors: the
/// solvers go on with fewer once some have converged. A map that keeps vectors of its own
/// fits them to its input (`fit`).
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

/// Make a temporary hold as many vectors as `like` does. Its content is lost when it has
/// to change.
pub fn fit<V: Vector>(tmp: &mut V, like: &V) {
    if tmp.num_vectors() != like.num_vectors() {
        *tmp = tmp.zeros_with(like.num_vectors());
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
        fit(&mut self.tmp, inp);
        self.b.mult(&mut self.tmp, inp);
        self.a.mult(out, &self.tmp);
    }
}

impl<A: LinearOp<V>, B: LinearOp<V>, V: Vector> LinearOp<V> for Product<A, B, V> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        fit(&mut self.tmp, inp);
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

/// `A^† A`, using one `A` and one intermediate vector. Hermitian by construction.
pub struct Normal<A, V> {
    a: A,
    tmp: V,
}

impl<A, V: Vector> Normal<A, V> {
    /// `like` only provides the shape of the intermediate vector.
    pub fn new(a: A, like: &V) -> Self {
        Self {
            a,
            tmp: like.zeros_like(),
        }
    }
}

impl<A: LinearOp<V>, V: Vector> LinearFn<V> for Normal<A, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        fit(&mut self.tmp, inp);
        self.a.mult(&mut self.tmp, inp);
        self.a.mult_dag(out, &self.tmp);
    }

    /// `<p, A† A p>` is `|A p|^2`: one vector to read instead of two, and real and
    /// positive whatever the rounding.
    fn mult_dot(&mut self, out: &mut V, inp: &V, dot: &mut [C64]) {
        fit(&mut self.tmp, inp);
        self.a.mult(&mut self.tmp, inp);
        let mut norm = vec![0.0; dot.len()];
        self.tmp.norm_sqr(&mut norm);
        for (d, n) in dot.iter_mut().zip(norm) {
            *d = C64::new(n, 0.0);
        }
        self.a.mult_dag(out, &self.tmp);
    }
}

impl<A: LinearOp<V>, V: Vector> LinearOp<V> for Normal<A, V> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.mult(out, inp);
    }
}

/// A mutable reference to a map is a map, so combinators can borrow instead of own.
impl<A: LinearFn<V> + ?Sized, V: Vector> LinearFn<V> for &mut A {
    fn mult(&mut self, out: &mut V, inp: &V) {
        (**self).mult(out, inp);
    }
    fn mult_dot(&mut self, out: &mut V, inp: &V, dot: &mut [C64]) {
        (**self).mult_dot(out, inp, dot);
    }
    fn mult_xpay(&mut self, out: &mut V, inp: &V, a: &[C64], b: &V) {
        (**self).mult_xpay(out, inp, a, b);
    }
}

impl<A: LinearOp<V> + ?Sized, V: Vector> LinearOp<V> for &mut A {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        (**self).mult_dag(out, inp);
    }
}
