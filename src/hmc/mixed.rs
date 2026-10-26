//! Mixed precision in the solves of the pseudofermion actions: (A†A)⁻¹ b by defect correction,
//! the residual of A†A in double precision and each correction by CG in single precision on
//! the same operator on the same links (`SingleNormal`), until the residual is below the
//! tolerance of the solve.

use crate::fermion::{Mdwf, MdwfParams, SingleAll, SingleEven, SingleOperator};
use crate::kernel::GammaRepre;
use crate::solver::{refine_normal, SolveError, SolveInfo, Stop};
use crate::{ConvertFrom, Lattice, LinearOp, Vector};
use std::sync::Arc;

/// The operator of an action in single precision: the inner solves of mixed precision.
pub(crate) trait SingleNormal<V, L> {
    /// Takes the links the operator of the action has now.
    fn set_links(&mut self, links: &L);
    /// e ≈ (A†A)⁻¹ r by CG in single precision from zero, to |r − A†A e|²/|r|² below
    /// `tolerance_sqr` or the tolerance of the inner solves, whichever is larger; its
    /// iterations.
    fn normal(&mut self, e: &mut V, r: &V, tolerance_sqr: f64, max_iterations: usize) -> usize;
}

/// D†D on all sites in single precision (`fermion::SingleAll`), on any backend.
impl<O, V> SingleNormal<V, O::Links> for SingleAll<O>
where
    O: SingleOperator,
    V: Vector + ConvertFrom<O::Whole>,
    O::Whole: ConvertFrom<V>,
{
    fn set_links(&mut self, links: &O::Links) {
        SingleAll::set_links(self, links);
    }

    fn normal(&mut self, e: &mut V, r: &V, tolerance_sqr: f64, max_iterations: usize) -> usize {
        SingleAll::normal(self, e, r, tolerance_sqr, max_iterations)
    }
}

/// K†K on the even sites in single precision, with Neff's Q if asked (`fermion::SingleEven`):
/// the even-odd Ratio action.
impl<O, V> SingleNormal<V, O::Links> for SingleEven<O>
where
    O: SingleOperator,
    V: Vector + ConvertFrom<O::Half>,
    O::Half: ConvertFrom<V>,
{
    fn set_links(&mut self, links: &O::Links) {
        SingleEven::set_links(self, links);
    }

    fn normal(&mut self, e: &mut V, r: &V, tolerance_sqr: f64, max_iterations: usize) -> usize {
        SingleEven::normal(self, e, r, tolerance_sqr, max_iterations)
    }
}

/// The domain-wall operator `params` in single precision on the CPU, with Neff's `alpha`.
pub(crate) fn single_mdwf<G: GammaRepre>(
    lat: &Arc<Lattice>,
    params: MdwfParams,
    alpha: f64,
) -> Mdwf<f32, G> {
    let mut op = Mdwf::<f32, G>::new(lat, params);
    op.set_alpha(alpha);
    op
}

/// A†A x = b by defect correction: e = (A†A)⁻¹ r in single precision on the residual r of
/// double precision, until |r|²/|b|² < `stop.tolerance_sqr` (at most `stop.max_restarts` + 1
/// residuals). Each inner solve goes no further than the outer tolerance needs (half of it
/// relative to the residual it starts from), or to the inner tolerance where that is reached
/// first, so that the last one does not overshoot. `x` holds the initial guess; the iterations
/// are those of the inner CG.
pub(crate) fn mixed_normal<V: Vector, L>(
    a: &mut dyn LinearOp<V>,
    single: &mut dyn SingleNormal<V, L>,
    x: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    refine_normal(a, 0.0, x, b, stop, &mut |_, e, r, _, residual_sqr| {
        let worst = residual_sqr.iter().fold(0.0f64, |m, r| m.max(*r));
        let needed = (0.5 * stop.tolerance_sqr / worst).min(1e-2);
        Ok(single.normal(e, r, needed, stop.max_iterations))
    })
}
