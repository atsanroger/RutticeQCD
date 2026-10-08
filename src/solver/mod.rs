//! Krylov solvers and preconditioners.
//!
//! The Krylov solvers know only `Vector` and `LinearFn`; nothing outside the
//! `preconditioner` module may refer to fermions or to the lattice.

mod bicgstab;
mod cg;
mod fgmres;
mod mixed;
mod polynomial;
pub mod preconditioner;

pub use bicgstab::BiCgStab;
pub use cg::Cg;
pub use fgmres::Fgmres;
pub use mixed::{Convert, Refine, Solve};
pub use polynomial::Polynomial;

use crate::{Adjoint, LinearFn, LinearOp, Normal, Vector};
use std::fmt;

/// Stopping rule shared by the solvers.
#[derive(Clone, Copy, Debug)]
pub struct Stop {
    /// Converged when `|r|^2 / |b|^2` is below this for every vector.
    pub tolerance_sqr: f64,
    /// Iterations before the true residual is recomputed and the solver restarts from the
    /// current solution.
    pub max_iterations: usize,
    pub max_restarts: usize,
}

impl Stop {
    /// No restarts: stop at the tolerance or after `max_iterations`.
    pub fn simple(tolerance_sqr: f64, max_iterations: usize) -> Self {
        Self {
            tolerance_sqr,
            max_iterations,
            max_restarts: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolveInfo {
    /// Iterations summed over all restarts.
    pub iterations: usize,
    pub restarts: usize,
    /// True `|b - A x|^2 / |b|^2` per vector, recomputed from the returned solution.
    pub residual_sqr: Vec<f64>,
}

#[derive(Debug)]
pub enum SolveError {
    /// The iteration limits were reached. The solution so far is in `x`.
    NotConverged(SolveInfo),
    /// A denominator vanished or became non-finite.
    Breakdown {
        iterations: usize,
        what: &'static str,
    },
    /// The solver cannot work without a preconditioner.
    MissingPreconditioner,
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::NotConverged(info) => write!(
                f,
                "not converged after {} iterations and {} restarts",
                info.iterations, info.restarts
            ),
            SolveError::Breakdown { iterations, what } => {
                write!(f, "breakdown at iteration {iterations}: {what}")
            }
            SolveError::MissingPreconditioner => write!(f, "this solver needs a preconditioner"),
        }
    }
}

impl std::error::Error for SolveError {}

/// Solve `A x = b`. `x` holds the initial guess on entry and the solution on return.
/// The operator and the preconditioner are passed dynamically so that solvers can be
/// chosen at run time and nested without the types growing.
pub trait Solver<V: Vector> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError>;
}

impl<V: Vector, S: Solver<V> + ?Sized> Solver<V> for &mut S {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        (**self).solve(a, m, x, b)
    }
}

/// Which vectors a solver should go on with, given which are still active: `Some` when
/// few enough are left that it pays to gather them into a smaller set, which costs a copy
/// of the solver's vectors. A quarter of the set must have converged.
pub(crate) fn fewer(active: &[bool]) -> Option<Vec<usize>> {
    let keep: Vec<usize> = (0..active.len()).filter(|j| active[*j]).collect();
    (!keep.is_empty() && 4 * keep.len() <= 3 * active.len()).then_some(keep)
}

/// Keep the entries `keep` of a list of per-vector numbers.
pub(crate) fn kept<T: Copy>(values: &mut Vec<T>, keep: &[usize]) {
    *values = keep.iter().map(|j| values[*j]).collect();
}

/// The solution of a solver that has gone on with fewer vectors: the smaller set and, for
/// each of its vectors, the vector of the caller's solution it is.
pub(crate) struct Narrowed<V> {
    pub x: V,
    pub of: Vec<usize>,
}

impl<V: Vector> Narrowed<V> {
    /// Go on with the vectors `keep` of the current set, which is `narrowed` if there is
    /// one and the caller's `x` otherwise. The vectors left out are final and are put back
    /// into `x` now.
    pub(crate) fn narrow(narrowed: Option<Self>, x: &mut V, keep: &[usize]) -> Self {
        match narrowed {
            None => Self {
                x: x.select(keep),
                of: keep.to_vec(),
            },
            Some(current) => {
                x.assign(&current.of, &current.x);
                Self {
                    x: current.x.select(keep),
                    of: keep.iter().map(|k| current.of[*k]).collect(),
                }
            }
        }
    }
}

/// `out_j = |b_j - (A x)_j|^2 / |b_j|^2`, using `r` as scratch. A zero right-hand side
/// counts as residual zero when `x` is zero too.
pub(crate) fn true_residual<V: Vector>(
    a: &mut dyn LinearFn<V>,
    x: &V,
    b: &V,
    b_norm: &[f64],
    r: &mut V,
    minus_one: &[crate::C64],
    out: &mut [f64],
) {
    a.mult_xpay(r, x, minus_one, b);
    r.norm_sqr(out);
    for (o, n) in out.iter_mut().zip(b_norm) {
        if *n > 0.0 {
            *o /= n;
        }
    }
}

/// Solve `A x = b` for a general `A` through the normal equations `A A^† y = b`,
/// `x = A^† y` (CGNE when `solver` is CG). The residual the solver sees is that of the
/// original system. Starts from zero; the content of `x` on entry is ignored.
pub fn solve_cgne<V: Vector, A: LinearOp<V>>(
    solver: &mut dyn Solver<V>,
    a: &mut A,
    x: &mut V,
    b: &V,
) -> Result<SolveInfo, SolveError> {
    let mut y = b.zeros_like();
    let info = {
        let mut a_adag = Normal::new(Adjoint(&mut *a), b);
        solver.solve(&mut a_adag, None, &mut y, b)?
    };
    a.mult_dag(x, &y);
    Ok(info)
}
