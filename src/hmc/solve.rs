//! The solves of the pseudofermion actions: x = (A†A)⁻¹ b for the operator A of an action (D,
//! or its Schur complement on the even sites), and their preconditioners.

use super::fermion_action::Psi;
use crate::fermion::HopTerms;
use crate::solver::preconditioner::multigrid::{Multigrid, MultigridParams};
use crate::solver::{
    largest_magnitude, BiCgStab, Cg, Chebyshev, Fgmres, Gcr, SolveError, SolveInfo, Solver, Stop,
};
use crate::{Adjoint, Lattice, LinearFn, LinearOp, Normal, Vector, C64};
use std::sync::Arc;

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

/// Γ with A† = Γ A Γ, and an approximate inverse `m` of A if there is one.
pub(super) struct Hermitian<'a, V> {
    pub(super) gamma: &'a mut dyn LinearFn<V>,
    pub(super) m: Option<&'a mut dyn LinearFn<V>>,
}

/// The residual r = b − (A†A + σ) x and its squares over those of b; true when all are below
/// the tolerance.
fn residual<V: Vector>(
    a: &mut dyn LinearOp<V>,
    shift: f64,
    (x, b): (&V, &V),
    (t, r): (&mut V, &mut V),
    b_norm: &[f64],
    info: &mut SolveInfo,
    tolerance_sqr: f64,
) -> bool {
    let n = b.num_vectors();
    a.mult(t, x);
    a.mult_dag(r, t);
    if shift != 0.0 {
        r.axpy(&vec![C64::new(shift, 0.0); n], x);
    }
    r.aypx(&vec![C64::new(-1.0, 0.0); n], b);
    r.norm_sqr(&mut info.residual_sqr);
    for (r, b) in info.residual_sqr.iter_mut().zip(b_norm) {
        *r = if *b > 0.0 { *r / b } else { 0.0 };
    }
    info.residual_sqr.iter().all(|r| *r < tolerance_sqr)
}

impl NormalSolver {
    /// `x` holds the initial guess on entry. The info has the true |b − A†A x|²/|b|² and the
    /// iterations of all the solves. Panics for a preconditioner that is not of the solver's
    /// kind.
    pub fn solve<V: Vector>(
        self,
        a: &mut dyn LinearOp<V>,
        pre: Pre<'_, V>,
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
        let hermitian = match pre {
            Pre::None => None,
            Pre::Hermitian { m, gamma } => Some(Hermitian { gamma, m: Some(m) }),
            Pre::Normal(_) => panic!("the two-step solvers take a preconditioner of A"),
        };
        self.two_step(a, hermitian, 0.0, x, b, stop)
    }

    /// x = (A†A + μ²)⁻¹ b through the Hermitian ΓA of an A with A† = Γ A Γ:
    /// A†A + μ² = Γ (A + iμΓ) Γ (A − iμΓ), two solves with the approximate inverse `m` of A.
    pub(super) fn shifted<V: Vector>(
        self,
        a: &mut dyn LinearOp<V>,
        hermitian: Hermitian<'_, V>,
        mu: f64,
        x: &mut V,
        b: &V,
        stop: Stop,
    ) -> Result<SolveInfo, SolveError> {
        self.two_step(a, Some(hermitian), mu, x, b, stop)
    }

    /// The two solves of (A†A + μ²) x = b, repeated on its residual until that is below the
    /// tolerance: with Γ, (A + iμΓ) w = Γ r and then (A − iμΓ) e = Γ w, both with `m`; without
    /// (μ = 0), A† w = r and then A e = w.
    fn two_step<V: Vector>(
        self,
        a: &mut dyn LinearOp<V>,
        mut hermitian: Option<Hermitian<'_, V>>,
        mu: f64,
        x: &mut V,
        b: &V,
        stop: Stop,
    ) -> Result<SolveInfo, SolveError> {
        let n = b.num_vectors();
        let mut b_norm = vec![0.0; n];
        b.norm_sqr(&mut b_norm);
        let (mut t, mut r) = (b.zeros_like(), b.zeros_like());
        let (mut w, mut e) = (b.zeros_like(), b.zeros_like());
        let mut info = SolveInfo {
            iterations: 0,
            restarts: 0,
            residual_sqr: vec![0.0; n],
        };
        for sweep in 0..=stop.max_restarts {
            info.restarts = sweep;
            let done = residual(
                a,
                mu * mu,
                (x, b),
                (&mut t, &mut r),
                &b_norm,
                &mut info,
                stop.tolerance_sqr,
            );
            if done {
                return Ok(info);
            }
            if sweep == stop.max_restarts {
                break;
            }
            match hermitian.as_mut() {
                None => {
                    w.set_zero();
                    e.set_zero();
                    info.iterations += self.one(&mut Adjoint(&mut *a), None, &mut w, &r, stop)?;
                    info.iterations += self.one(&mut &mut *a, None, &mut e, &w, stop)?;
                }
                Some(Hermitian { gamma, m }) => {
                    for sign in [1.0, -1.0] {
                        // The right-hand side of this half: Γ r, then Γ w.
                        gamma.mult(&mut t, if sign > 0.0 { &r } else { &w });
                        e.set_zero();
                        let mut twisted = Twisted {
                            a: &mut *a,
                            gamma: &mut **gamma,
                            mu: sign * mu,
                            turned: b.zeros_like(),
                        };
                        let pre = m.as_mut().map(|m| &mut **m as &mut dyn LinearFn<V>);
                        info.iterations += self.one(&mut twisted, pre, &mut e, &t, stop)?;
                        if sign > 0.0 {
                            w.copy_from(&e);
                        }
                    }
                }
            }
            x.axpy(&vec![C64::new(1.0, 0.0); n], &e);
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

/// A + iμΓ (A itself at μ = 0).
struct Twisted<'a, V> {
    a: &'a mut dyn LinearOp<V>,
    gamma: &'a mut dyn LinearFn<V>,
    mu: f64,
    turned: V,
}

impl<V: Vector> LinearFn<V> for Twisted<'_, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.a.mult(out, inp);
        if self.mu != 0.0 {
            crate::fit(&mut self.turned, inp);
            self.gamma.mult(&mut self.turned, inp);
            out.axpy(
                &vec![C64::new(0.0, self.mu); inp.num_vectors()],
                &self.turned,
            );
        }
    }
}

/// The multigrid of Wilson and clover fermions along a trajectory: the test vectors are kept,
/// and the cycle of every solve is made from them for the operator on the links of that solve.
pub struct MultigridSetup {
    pub params: MultigridParams,
    /// The test vectors are improved (a full setup) when the pseudofermions are drawn and after
    /// every this many force solves; 0: never within a trajectory.
    pub refresh_every: usize,
    /// A full setup starts from the vectors of the one before, improved this many times; none:
    /// from random vectors.
    pub reuse: Option<usize>,
    vectors: Option<Psi>,
    since: usize,
}

impl MultigridSetup {
    pub fn new(params: MultigridParams, refresh_every: usize) -> Self {
        Self {
            params,
            refresh_every,
            reuse: None,
            vectors: None,
            since: 0,
        }
    }

    /// Start every full setup but the first from the test vectors of the one before, improved
    /// `iterations` times (0: smoothed once): the links of consecutive setups are close.
    pub fn reusing_vectors(mut self, iterations: usize) -> Self {
        self.reuse = Some(iterations);
        self
    }

    /// The cycle for `op`. `fresh`: the first solve since the pseudofermions were drawn; `md`:
    /// a solve for the force.
    pub(super) fn cycle<'a, O: HopTerms<Psi>>(
        &mut self,
        op: &'a O,
        lat: &Arc<Lattice>,
        fresh: bool,
        md: bool,
    ) -> Multigrid<'a, O, f64> {
        let due = fresh || (self.refresh_every > 0 && self.since >= self.refresh_every);
        let cycle = match (self.vectors.take(), due, self.reuse) {
            (Some(vectors), false, _) => Multigrid::rebuilt(op, lat, &self.params, vectors),
            (Some(vectors), true, Some(iterations)) => {
                Multigrid::from_vectors(op, lat, &self.params, vectors, iterations)
            }
            _ => Multigrid::new(op, lat, &self.params),
        }
        .expect("the multigrid setup succeeds");
        if due {
            self.since = 0;
        }
        if md {
            self.since += 1;
        }
        self.vectors = Some(cycle.test_vectors());
        cycle
    }
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
    /// For the two-step solvers of Wilson and clover fermions.
    Multigrid(Box<MultigridSetup>),
    /// The multigrid of domain-wall fermions over Pauli-Villars fields (`hmc::dwf_multigrid`):
    /// for the Ratio action of D over D_PV of mass 1, without Neff's α.
    DomainWallMultigrid(Box<super::DwfMultigridSetup>),
}

impl Preconditioner {
    pub fn multigrid(setup: MultigridSetup) -> Self {
        Preconditioner::Multigrid(Box::new(setup))
    }

    pub fn chebyshev(steps: usize, ratio: f64) -> Self {
        Preconditioner::Chebyshev {
            steps,
            ratio,
            hi: None,
        }
    }
}

/// (A†A)⁻¹ b by CG with the Chebyshev polynomial of A†A as preconditioner. `again` is a second
/// instance of A, for the polynomial; `hi` is estimated when `fresh` or not known yet.
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
