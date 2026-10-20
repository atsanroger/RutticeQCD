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

/// CG on `a` (Hermitian positive definite) in the precision of `V2`, with reliable updates in
/// that of `V`: the iterations apply `low`, the same operator in the lower precision (`like` gives
/// the shape of its vectors), and sum the solution of the low precision; each time |r|² of the
/// low precision has fallen by `drop_sqr` from the largest since the last update, or seems to
/// be below the tolerance, that solution is added to `x` and the residual `b − a x` recomputed
/// in the precision of `V`. CG goes on from there with the same search direction, so the solve
/// keeps the convergence of CG (unlike restarts from zero, which defect correction does), and
/// it stops only on the residual of the higher precision. `x` holds the initial guess; the
/// iterations are those of `low`, the restarts the reliable updates.
pub fn reliable_cg<V, V2>(
    a: &mut dyn LinearFn<V>,
    low: &mut dyn LinearFn<V2>,
    like: &V2,
    x: &mut V,
    b: &V,
    stop: Stop,
    drop_sqr: f64,
) -> Result<SolveInfo, SolveError>
where
    V: Vector + ConvertFrom<V2>,
    V2: Vector + ConvertFrom<V>,
{
    assert!(x.same_shape(b));
    let nv = b.num_vectors();
    let real = |x: f64| C64::new(x, 0.0);
    let minus_one = vec![real(-1.0); nv];
    let one = vec![real(1.0); nv];
    let mut b_norm = vec![0.0; nv];
    b.norm_sqr(&mut b_norm);
    let tol = stop.tolerance_sqr;

    let (mut r, mut t) = (b.zeros_like(), b.zeros_like());
    let mut residual = vec![0.0; nv];
    true_residual(a, x, b, &b_norm, &mut r, &minus_one, &mut residual);
    let mut info = SolveInfo {
        iterations: 0,
        restarts: 0,
        residual_sqr: residual.clone(),
    };
    if residual.iter().all(|r| *r <= tol) {
        return Ok(info);
    }
    let mut r2 = like.zeros_with(nv);
    r2.convert_from(&r);
    let (mut p, mut ap, mut x2) = (r2.zeros_like(), r2.zeros_like(), r2.zeros_like());
    p.copy_from(&r2);
    let (mut rr, mut rr_new) = (vec![0.0; nv], vec![0.0; nv]);
    r2.norm_sqr(&mut rr);
    // The largest |r|² of the low precision since the last update.
    let mut largest = rr.clone();
    let mut pap = vec![C64::ZERO; nv];
    let (mut alpha, mut beta) = (vec![C64::ZERO; nv], vec![C64::ZERO; nv]);

    let limit = stop.max_iterations * (stop.max_restarts + 1);
    while info.iterations < limit {
        // A vector whose residual of the higher precision is below the tolerance stays put.
        let active: Vec<bool> = residual.iter().map(|r| *r > tol).collect();
        low.mult_dot(&mut ap, &p, &mut pap);
        for j in 0..nv {
            if active[j] && !(pap[j].re > 0.0 && pap[j].re.is_finite()) {
                return Err(SolveError::Breakdown {
                    iterations: info.iterations,
                    what: "<p, A p> is not positive",
                });
            }
            alpha[j] = if active[j] {
                real(rr[j] / pap[j].re)
            } else {
                C64::ZERO
            };
        }
        x2.axpy(&alpha, &p);
        for a in alpha.iter_mut() {
            *a = -*a;
        }
        r2.axpy_norm_sqr(&alpha, &ap, &mut rr_new);
        info.iterations += 1;

        let update = (0..nv).any(|j| {
            active[j] && (rr_new[j] < drop_sqr * largest[j] || rr_new[j] <= tol * b_norm[j])
        });
        if update {
            t.convert_from(&x2);
            x.axpy(&one, &t);
            x2.set_zero();
            true_residual(a, x, b, &b_norm, &mut r, &minus_one, &mut residual);
            info.restarts += 1;
            if residual.iter().all(|r| *r <= tol) {
                info.residual_sqr = residual;
                return Ok(info);
            }
            r2.convert_from(&r);
            r2.norm_sqr(&mut rr_new);
            largest.copy_from_slice(&rr_new);
        } else {
            for (l, r) in largest.iter_mut().zip(&rr_new) {
                *l = l.max(*r);
            }
        }
        for j in 0..nv {
            beta[j] = if residual[j] > tol {
                real(rr_new[j] / rr[j])
            } else {
                C64::ZERO
            };
        }
        p.aypx(&beta, &r2);
        rr.copy_from_slice(&rr_new);
    }
    // Out of iterations: the solution as it is.
    t.convert_from(&x2);
    x.axpy(&one, &t);
    true_residual(a, x, b, &b_norm, &mut r, &minus_one, &mut residual);
    info.residual_sqr = residual;
    Err(SolveError::NotConverged(info))
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
