//! The generic pieces that mixed precision is assembled from:
//! defect correction, precision conversion, and a solver bound to its operator.

use super::{true_residual, SolveError, SolveInfo, Solver, Stop};
use crate::{ConvertFrom, LinearFn, Vector, C64};

/// Defect correction: `x += M (b - A x)` until the true residual is small enough. `A` is
/// the accurate operator; `M` is any approximate inverse, typically a lower-precision
/// solve. One "iteration" is one application of `M`.
pub struct Refine<V> {
    pub stop: Stop,
    work: Option<(V, V)>,
}

impl<V: Vector> Refine<V> {
    pub fn new(stop: Stop) -> Self {
        Self { stop, work: None }
    }
}

impl<V: Vector> Solver<V> for Refine<V> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        let Some(m) = m else {
            return Err(SolveError::MissingPreconditioner);
        };
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.work.as_ref().is_some_and(|w| w.0.same_shape(b)) {
            self.work = Some((b.zeros_like(), b.zeros_like()));
        }
        let Some((r, e)) = self.work.as_mut() else {
            unreachable!("the work vectors were just created");
        };

        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let one = vec![C64::new(1.0, 0.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let mut residual = vec![0.0; nv];

        let limit = self.stop.max_iterations * (self.stop.max_restarts + 1);
        for iterations in 0..=limit {
            true_residual(a, x, b, &b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: 0,
                residual_sqr: residual.clone(),
            };
            if residual.iter().all(|r| *r <= self.stop.tolerance_sqr) {
                return Ok(info);
            }
            if iterations == limit {
                return Err(SolveError::NotConverged(info));
            }
            m.mult(e, r);
            x.axpy(&one, e);
        }
        unreachable!("the loop returns on its last pass");
    }
}

/// Wraps a map acting on `V2` as a map acting on `V`: convert down, apply, convert back.
pub struct Convert<M2, V2> {
    inner: M2,
    inp: V2,
    out: V2,
}

impl<M2, V2: Vector> Convert<M2, V2> {
    /// `like` only provides the shape of the inner vectors.
    pub fn new(inner: M2, like: &V2) -> Self {
        Self {
            inner,
            inp: like.zeros_like(),
            out: like.zeros_like(),
        }
    }

    pub fn inner(&self) -> &M2 {
        &self.inner
    }
}

impl<V, V2, M2> LinearFn<V> for Convert<M2, V2>
where
    V: Vector + ConvertFrom<V2>,
    V2: Vector + ConvertFrom<V>,
    M2: LinearFn<V2>,
{
    fn mult(&mut self, out: &mut V, inp: &V) {
        // Solvers go on with fewer vectors once some have converged.
        let n = inp.num_vectors();
        if self.inp.num_vectors() != n {
            self.inp = self.inp.zeros_with(n);
            self.out = self.out.zeros_with(n);
        }
        self.inp.convert_from(inp);
        self.inner.mult(&mut self.out, &self.inp);
        out.convert_from(&self.out);
    }
}

/// A solver together with its operator: `out = A^{-1} inp` to the solver's tolerance,
/// starting from zero. This is what makes a solver usable as someone else's
/// preconditioner.
///
/// Not reaching the tolerance is normal for an inner solve and is not an error here; the
/// outcome of the most recent solve is kept in `last`.
///
/// The solver is owned; pass `&mut solver` to keep using it elsewhere. With a fixed number
/// of iterations and no tolerance to reach, this is how a Krylov solver becomes a smoother
/// (GMRES(m): `Fgmres` with `max_iterations = m`, no restarts).
pub struct Solve<S, A, M> {
    solver: S,
    a: A,
    m: Option<M>,
    pub last: Option<Result<SolveInfo, SolveError>>,
}

impl<S, A> Solve<S, A, A> {
    /// Without a preconditioner.
    pub fn new(solver: S, a: A) -> Self {
        Self {
            solver,
            a,
            m: None,
            last: None,
        }
    }
}

impl<S, A, M> Solve<S, A, M> {
    pub fn preconditioned(solver: S, a: A, m: M) -> Self {
        Self {
            solver,
            a,
            m: Some(m),
            last: None,
        }
    }
}

impl<V, S, A, M> LinearFn<V> for Solve<S, A, M>
where
    V: Vector,
    S: Solver<V>,
    A: LinearFn<V>,
    M: LinearFn<V>,
{
    fn mult(&mut self, out: &mut V, inp: &V) {
        out.set_zero();
        let m = self.m.as_mut().map(|m| m as &mut dyn LinearFn<V>);
        self.last = Some(self.solver.solve(&mut self.a, m, out, inp));
    }
}
