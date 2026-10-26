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
    /// iterations, or the breakdown of the solve.
    fn normal(
        &mut self,
        e: &mut V,
        r: &V,
        tolerance_sqr: f64,
        max_iterations: usize,
    ) -> Result<usize, SolveError>;
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

    fn normal(
        &mut self,
        e: &mut V,
        r: &V,
        tolerance_sqr: f64,
        max_iterations: usize,
    ) -> Result<usize, SolveError> {
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

    fn normal(
        &mut self,
        e: &mut V,
        r: &V,
        tolerance_sqr: f64,
        max_iterations: usize,
    ) -> Result<usize, SolveError> {
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
        single.normal(e, r, needed, stop.max_iterations)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::{Bound, FermionOp, Variant};
    use crate::kernel::Dirac;
    use crate::{random_fifth, Boundary};

    /// A NaN in the single-precision operator ends the solve with `Breakdown` from the outer
    /// loop, as a NaN of the double-precision operator does, rather than a panic.
    #[test]
    fn a_nan_in_single_precision_is_a_breakdown() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let params = MdwfParams {
            ls: 4,
            m5: 1.0,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary: [Boundary::Periodic; 4],
        };
        let u = crate::gauge::random_gauge(&lat, 7);
        let mut op = Mdwf::<f64, Dirac>::new(&lat, params);
        op.set_gauge(&u).unwrap();
        let mut bad = u.zeros_like();
        bad.copy_from(&u);
        bad.as_mut_slice()[100] = f64::NAN;
        let mut single = SingleAll::new(single_mdwf::<Dirac>(&lat, params, 1.0), &lat, 1e-6);
        single.set_links(&bad);
        let b = random_fifth(&lat, params.ls, 8);
        let mut x = b.zeros_like();
        let stop = Stop {
            tolerance_sqr: 1e-20,
            max_iterations: 200,
            max_restarts: 5,
        };
        let mut a = Bound::new(&op, Variant::D);
        let got = mixed_normal(&mut a, &mut single, &mut x, &b, stop);
        assert!(matches!(got, Err(SolveError::Breakdown { .. })), "{got:?}");
    }
}
