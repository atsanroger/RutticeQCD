//! The part of an operator of Wilson type that acts within a site: the site-diagonal block
//! (`DiagTerm`). For Wilson fermions it is the identity (`UnitDiag`); clover fermions put
//! the clover term here (`fermion::clover`).

use super::*;

/// The site-diagonal block of an operator of Wilson type. It must be Hermitian and commute
/// with γ5, so that D† = γ5 D γ5 keeps holding.
pub trait DiagTerm<T: Storage>: Send + Sync + 'static {
    /// True when the block is the identity; lets callers skip work.
    const IS_UNIT: bool;
    /// Nominal floating-point operations of applying the block at one site, including
    /// adding the result to the hopping term (see `FermionOp::flops_per_site`).
    const FLOPS_PER_SITE: usize;
    /// The block at `full_site` applied to `phi`.
    fn apply(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute>;
    /// Its inverse applied to `phi`.
    fn apply_inv(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute>;
    /// Rebuild whatever depends on the gauge field.
    fn set_gauge(&mut self, lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Result<(), GaugeError>;
}

/// The identity: plain Wilson fermions.
pub struct UnitDiag;

impl<T: Storage> DiagTerm<T> for UnitDiag {
    const IS_UNIT: bool = true;
    const FLOPS_PER_SITE: usize = 0;
    #[inline(always)]
    fn apply(&self, _full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        *phi
    }
    #[inline(always)]
    fn apply_inv(&self, _full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        *phi
    }
    fn set_gauge(&mut self, _lat: &Arc<Lattice>, _u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        Ok(())
    }
}
