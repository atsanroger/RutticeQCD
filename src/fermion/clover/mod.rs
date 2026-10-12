//! Clover (Sheikholeslami-Wohlert) fermions.

use super::wilson::{DiagTerm, WilsonOp, WilsonParams};
use super::GaugeError;
use crate::gauge::{field_strength, PLANES};
use crate::kernel::{
    hermitian6_mul, Cplx, Dirac, Gamma, GammaRepre, LaneOf, Number, Spinor as SiteSpinor, Unit,
    HERMITIAN6_REALS, NC, NS,
};
use crate::numerics::invert_complex;
use crate::{Aosoa, Boundary, Field, Gauge, Lattice, Stencil, Storage, ND};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct CloverParams {
    /// Hopping parameter κ.
    pub kappa: f64,
    /// Clover coefficient c_SW.
    pub csw: f64,
    pub boundary: [Boundary; ND],
}

mod site;

pub use site::CloverDiag;

/// The clover operator: the Wilson sweep with the clover term as its site-diagonal block.
pub type Clover<T, G = Dirac> = WilsonOp<T, G, CloverDiag<T>>;

/// The clover operator on fields in blocks of `W` sites (the SIMD layout). The hopping
/// term works on whole blocks; the clover term is applied to the lanes one at a time.
pub type CloverSimd<T, const W: usize, G = Dirac> = WilsonOp<T, G, CloverDiag<T>, Aosoa<W>>;

impl<T: Storage, G: GammaRepre, const W: usize> WilsonOp<T, G, CloverDiag<T>, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    /// `None` if the lattice cannot be cut into `W` sub-lattices of equal shape.
    pub fn new(lat: &Arc<Lattice>, params: CloverParams) -> Option<Self> {
        Self::with_diag(
            lat,
            WilsonParams {
                kappa: params.kappa,
                boundary: params.boundary,
            },
            CloverDiag::new::<G>(params.kappa, params.csw),
        )
    }
}

impl<T: Storage, G: GammaRepre> WilsonOp<T, G, CloverDiag<T>> {
    pub fn new(lat: &Arc<Lattice>, params: CloverParams) -> Self {
        Self::with_diag(
            lat,
            WilsonParams {
                kappa: params.kappa,
                boundary: params.boundary,
            },
            CloverDiag::new::<G>(params.kappa, params.csw),
        )
    }
}
