//! Spectral transforms: operators made of A whose eigenvectors are those of A and whose
//! wanted eigenvalues are at an end of their spectrum.

use crate::{LinearFn, LinearOp, Vector, C64};

/// A − z (its adjoint A† − z̄): for the singular values of z − A (`singular`), and to move an
/// end of the spectrum.
pub struct Shifted<A> {
    pub a: A,
    pub z: C64,
}

impl<V: Vector, A: LinearFn<V>> LinearFn<V> for Shifted<A> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.a.mult(out, inp);
        out.axpy(&vec![-self.z; inp.num_vectors()], inp);
    }
}

impl<V: Vector, A: LinearOp<V>> LinearOp<V> for Shifted<A> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.a.mult_dag(out, inp);
        out.axpy(&vec![-self.z.conj(); inp.num_vectors()], inp);
    }
}

/// (A − σ)⁻¹ by the solve `solve(out, inp)` (a multigrid-preconditioned solver, say): the
/// eigenvalues of A nearest σ become the largest in magnitude of this operator, for Arnoldi
/// (`Which::LargestMagnitude`), or, for a Hermitian A and σ below its spectrum, the largest of
/// a Hermitian positive one. The solve has to be accurate enough to be the same linear map at
/// every application.
pub struct ShiftInvert<S>(pub S);

impl<V: Vector, S: FnMut(&mut V, &V)> LinearFn<V> for ShiftInvert<S> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        (self.0)(out, inp);
    }
}
