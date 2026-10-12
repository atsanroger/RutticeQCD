//! The solves of the pseudofermion actions: x = (A†A)⁻¹ b for the operator A of an action (D,
//! or its Schur complement on the even sites), and their preconditioners.

use super::fermion_action::Psi;
use super::ForceOp;
use crate::solver::preconditioner::multigrid::{Multigrid, MultigridError, MultigridParams};
use crate::solver::{
    largest_magnitude, BiCgStab, Cg, Chebyshev, Fgmres, Gcr, SolveError, SolveInfo, Solver, Stop,
};
use crate::{Adjoint, Field, Gauge, LinearFn, LinearOp, Normal, Vector, C64};
use std::mem::ManuallyDrop;

/// How (A†A)⁻¹ b is computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormalSolver {
    /// CG on A†A.
    #[default]
    Cg,
    /// Two solves, A† y = b and then A x = y, by BiCGStab, repeated on the residual of
    /// A†A x = b until that is below the tolerance.
    BiCgStab,
    /// The same with FGMRES restarted after this many steps.
    Fgmres(usize),
    /// The same with GCR truncated to this many directions.
    Gcr(usize),
}

/// The preconditioner of one solve.
pub enum Pre<'a, V> {
    None,
    /// An approximation of (A†A)⁻¹ that is Hermitian, positive and the same map at every step:
    /// for CG.
    Normal(&'a mut dyn LinearFn<V>),
    /// An approximation `m` of A⁻¹, and Γ with A† = Γ A Γ (Wilson and clover fermions: γ5), for
    /// the two-step solvers: A† y = r is then solved as y = Γ A⁻¹ Γ r, with the same `m`.
    Hermitian {
        m: &'a mut dyn LinearFn<V>,
        gamma: &'a mut dyn LinearFn<V>,
    },
}

impl NormalSolver {
    /// `x` holds the initial guess on entry. The info has the true |b − A†A x|²/|b|² and the
    /// iterations of all the solves. Panics for a preconditioner that is not of the solver's
    /// kind.
    pub fn solve<V: Vector>(
        self,
        a: &mut dyn LinearOp<V>,
        mut pre: Pre<'_, V>,
        x: &mut V,
        b: &V,
        stop: Stop,
    ) -> Result<SolveInfo, SolveError> {
        if self == NormalSolver::Cg {
            let m = match pre {
                Pre::None => None,
                Pre::Normal(m) => Some(m),
                Pre::Hermitian { .. } => panic!("CG takes a preconditioner of A†A"),
            };
            return Cg::new(stop).solve(&mut Normal::new(a, b), m, x, b);
        }
        let n = b.num_vectors();
        let (minus, one) = (vec![C64::new(-1.0, 0.0); n], vec![C64::new(1.0, 0.0); n]);
        let mut b_norm = vec![0.0; n];
        b.norm_sqr(&mut b_norm);
        let (mut t, mut r) = (b.zeros_like(), b.zeros_like());
        let (mut y, mut e) = (b.zeros_like(), b.zeros_like());
        let mut info = SolveInfo {
            iterations: 0,
            restarts: 0,
            residual_sqr: vec![0.0; n],
        };
        for sweep in 0..=stop.max_restarts {
            // r = b − A†A x
            a.mult(&mut t, x);
            a.mult_dag(&mut r, &t);
            r.aypx(&minus, b);
            r.norm_sqr(&mut info.residual_sqr);
            for (r, b) in info.residual_sqr.iter_mut().zip(&b_norm) {
                *r = if *b > 0.0 { *r / b } else { 0.0 };
            }
            info.restarts = sweep;
            if info.residual_sqr.iter().all(|r| *r < stop.tolerance_sqr) {
                return Ok(info);
            }
            if sweep == stop.max_restarts {
                break;
            }
            y.set_zero();
            e.set_zero();
            match &mut pre {
                Pre::None => {
                    info.iterations += self.one(&mut Adjoint(&mut *a), None, &mut y, &r, stop)?;
                    info.iterations += self.one(&mut &mut *a, None, &mut e, &y, stop)?;
                }
                Pre::Hermitian { m, gamma } => {
                    gamma.mult(&mut t, &r);
                    info.iterations += self.one(&mut &mut *a, Some(&mut **m), &mut e, &t, stop)?;
                    gamma.mult(&mut y, &e);
                    e.set_zero();
                    info.iterations += self.one(&mut &mut *a, Some(&mut **m), &mut e, &y, stop)?;
                }
                Pre::Normal(_) => panic!("the two-step solvers take a preconditioner of A"),
            }
            x.axpy(&one, &e);
        }
        Err(SolveError::NotConverged(info))
    }

    /// One solve A x = b from zero; its iterations. Not reaching the tolerance is left to the
    /// residual of the normal system to find.
    fn one<V: Vector>(
        self,
        a: &mut dyn LinearFn<V>,
        m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
        stop: Stop,
    ) -> Result<usize, SolveError> {
        let solved = match self {
            NormalSolver::Cg => unreachable!("CG works on the normal system"),
            NormalSolver::BiCgStab => BiCgStab::new(stop).solve(a, m, x, b),
            NormalSolver::Fgmres(steps) => {
                let cycles = Stop {
                    max_iterations: steps,
                    max_restarts: stop.max_iterations / steps.max(1),
                    ..stop
                };
                Fgmres::new(cycles).solve(a, m, x, b)
            }
            NormalSolver::Gcr(k) => Gcr::new(stop, k).solve(a, m, x, b),
        };
        match solved {
            Ok(info) | Err(SolveError::NotConverged(info)) => Ok(info.iterations),
            Err(e) => Err(e),
        }
    }
}

/// A multigrid cycle that is set up again when the links have changed enough.
pub trait SetupOnLinks {
    /// Set up for the operator on the links `u`.
    fn setup(&mut self, u: &Field<f64, Gauge>) -> Result<(), MultigridError>;
    /// The cycle of the last setup: an approximate inverse of the operator on those links.
    fn cycle(&mut self) -> &mut dyn LinearFn<Psi>;
}

/// The preconditioner of an action's solves.
#[derive(Default)]
pub enum Preconditioner {
    #[default]
    None,
    /// For CG: `steps` steps of Chebyshev iteration on A†A for [hi/`ratio`, hi], hi from a power
    /// iteration when the pseudofermions are drawn.
    Chebyshev {
        steps: usize,
        ratio: f64,
        hi: Option<f64>,
    },
    /// For the two-step solvers of Wilson and clover fermions: a multigrid cycle, set up when
    /// the pseudofermions are drawn and again after every `refresh_every` force solves (0:
    /// never within a trajectory). In between it is the cycle of the links of its setup.
    Multigrid {
        cycle: Box<dyn SetupOnLinks>,
        refresh_every: usize,
        since: usize,
    },
}

impl Preconditioner {
    pub fn chebyshev(steps: usize, ratio: f64) -> Self {
        Preconditioner::Chebyshev {
            steps,
            ratio,
            hi: None,
        }
    }

    pub fn multigrid(cycle: impl SetupOnLinks + 'static, refresh_every: usize) -> Self {
        Preconditioner::Multigrid {
            cycle: Box::new(cycle),
            refresh_every,
            since: 0,
        }
    }
}

/// (A†A)⁻¹ b by CG with the Chebyshev polynomial of A†A as preconditioner. `again` is a second
/// instance of A, for the polynomial; `hi` is estimated when `fresh` or not known yet.
#[allow(clippy::too_many_arguments)]
pub(super) fn chebyshev_cg<V: Vector, A: LinearOp<V>>(
    a: &mut dyn LinearOp<V>,
    again: A,
    (steps, ratio, hi): (usize, f64, &mut Option<f64>),
    fresh: bool,
    x: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let mut normal = Normal::new(again, b);
    let top = match *hi {
        Some(top) if !fresh => top,
        _ => {
            let mut start = b.zeros_like();
            start.copy_from(b);
            // The links move during a trajectory: a wider margin than for one configuration.
            let top = 1.3 * largest_magnitude(&mut normal, &mut start, 20);
            *hi = Some(top);
            top
        }
    };
    let mut m = Chebyshev::new(normal, steps, top / ratio, top, b);
    NormalSolver::Cg.solve(a, Pre::Normal(&mut m), x, b, stop)
}

/// The multigrid of Wilson and clover fermions with an operator of its own, on the links of
/// the last setup.
pub struct OwnedMultigrid<O: ForceOp + 'static>
where
    O::Work: 'static,
{
    make: Box<dyn Fn() -> O>,
    params: MultigridParams,
    built: Option<Built<O>>,
}

struct Built<O: ForceOp + 'static>
where
    O::Work: 'static,
{
    cycle: ManuallyDrop<Multigrid<'static, O, f64>>,
    op: *mut O,
}

impl<O: ForceOp + 'static> Drop for Built<O>
where
    O::Work: 'static,
{
    fn drop(&mut self) {
        // SAFETY: `cycle` is the only borrower of `*op`, which came from `Box::into_raw`; it is
        // dropped first, and neither is used again.
        unsafe {
            ManuallyDrop::drop(&mut self.cycle);
            drop(Box::from_raw(self.op));
        }
    }
}

impl<O: ForceOp + 'static> OwnedMultigrid<O>
where
    O::Work: 'static,
{
    /// `make` gives the operator, without links.
    pub fn new(make: impl Fn() -> O + 'static, params: MultigridParams) -> Self {
        Self {
            make: Box::new(make),
            params,
            built: None,
        }
    }
}

impl<O: ForceOp + 'static> SetupOnLinks for OwnedMultigrid<O>
where
    O::Work: 'static,
{
    fn setup(&mut self, u: &Field<f64, Gauge>) -> Result<(), MultigridError> {
        // The old cycle first: two of them need not be held at once.
        self.built = None;
        let mut op = (self.make)();
        op.set_gauge(u).expect("the links suit the operator");
        let op = Box::into_raw(Box::new(op));
        // SAFETY: `op` points to a live box that `Built` frees only after the cycle borrowing
        // it is dropped, and nothing else gets a reference to it.
        let borrowed: &'static O = unsafe { &*op };
        match Multigrid::new(borrowed, u.lattice(), &self.params) {
            Ok(cycle) => {
                self.built = Some(Built {
                    cycle: ManuallyDrop::new(cycle),
                    op,
                });
                Ok(())
            }
            Err(e) => {
                // SAFETY: the setup failed, so nothing borrows `*op`.
                drop(unsafe { Box::from_raw(op) });
                Err(e)
            }
        }
    }

    fn cycle(&mut self) -> &mut dyn LinearFn<Psi> {
        &mut *self.built.as_mut().expect("set up").cycle
    }
}

/// A + σ.
pub(super) struct Shifted<'a, V> {
    pub a: &'a mut dyn LinearFn<V>,
    pub shift: f64,
}

impl<V: Vector> LinearFn<V> for Shifted<'_, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.a.mult(out, inp);
        out.axpy(&vec![C64::new(self.shift, 0.0); inp.num_vectors()], inp);
    }
}
