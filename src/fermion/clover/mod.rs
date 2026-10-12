//! Clover (Sheikholeslami-Wohlert) fermions.
//!
//! D = D_Wilson - κ c_SW Σ_{mu<nu} F_mu_nu ⊗ γ_mu γ_nu
//!
//! where F_mu_nu = (Q - Q^†) / 8 is the clover-leaf field strength (`gauge`). The
//! extra term acts within a site, so the operator is the Wilson sweep with a non-trivial
//! site-diagonal block T(x) = 1 - κ c_SW Σ F ⊗ γγ, a Hermitian 12 × 12 matrix.
//!
//! T commutes with γ5, so in the basis of γ5 eigenvectors it splits into two Hermitian
//! 6 × 6 blocks. Those are what is stored: 72 reals per site for T and 72 for its inverse,
//! instead of 288 each for the dense matrices.
//!
//! Provides: everything `Wilson` provides; T and its inverse for the even-odd blocks.
//!
//! Files: this one has the parameters and the operator, which is the Wilson operator with
//! another site term; `site.rs` that term, the clover term and its inverse. There is no
//! `full.rs` and no `even_odd.rs` here: the operator on the whole lattice and its even-odd
//! blocks are those of `wilson`, written for any site term.
//! Solvers: as for Wilson. Tools needed: field strength (`gauge`), small dense
//! inversion (`numerics`).
//!
//! The normalisation and sign conventions follow Bridge++ so that results can be compared.
//! Written from the formula above.

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
