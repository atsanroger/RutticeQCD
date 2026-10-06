//! Krylov solvers. They know only `Vector` and `LinearFn`.

mod bicgstab;
mod cg;
mod mixed;

pub use bicgstab::BiCgStab;
pub use cg::Cg;
pub use mixed::{Convert, Refine, Solve};

use ruttice_core::{Adjoint, LinearFn, LinearOp, Normal, Vector};
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

/// `out_j = |b_j - (A x)_j|^2 / |b_j|^2`, using `r` as scratch. A zero right-hand side
/// counts as residual zero when `x` is zero too.
pub(crate) fn true_residual<V: Vector>(
    a: &mut dyn LinearFn<V>,
    x: &V,
    b: &V,
    b_norm: &[f64],
    r: &mut V,
    minus_one: &[ruttice_core::C64],
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
