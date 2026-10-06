//! Fermion operators (DESIGN.md §5). One fermion per file.

mod wilson;

pub use wilson::{Wilson, WilsonParams, WilsonWork};

use ruttice_core::{Field, Gauge};
use std::fmt;

/// What a fused kernel can do in a single sweep. Products such as D†D are compositions,
/// not variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    D,
    /// D† = γ5 D γ5.
    Dagger,
    /// γ5 D.
    Gamma5D,
}

#[derive(Debug)]
pub enum GaugeError {
    /// The gauge field lives on a different lattice than the operator.
    LatticeMismatch,
}

impl fmt::Display for GaugeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GaugeError::LatticeMismatch => {
                write!(f, "gauge field and operator are on different lattices")
            }
        }
    }
}

impl std::error::Error for GaugeError {}

/// A fermion operator. The operator itself is read-only while in use; scratch space lives
/// in `Work`, so several users can apply the same operator at once (DESIGN.md §7.3).
pub trait FermionOp<V> {
    type Work;
    fn work(&self) -> Self::Work;
    /// Takes the canonical `f64` gauge field and converts it to the operator's own format.
    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError>;
    /// `out = M inp` for the chosen variant.
    fn mult(&self, work: &mut Self::Work, out: &mut V, inp: &V, variant: Variant);
}

/// The pieces of the operator, one direction at a time. Needed by the fermion force and by
/// multigrid coarsening, and the reference that fused kernels are checked against.
pub trait HopTerms<V>: FermionOp<V> {
    /// `out += (1 - γ_mu) U_mu(x) inp(x + mu)`.
    fn hop_forward(&self, work: &mut Self::Work, mu: usize, out: &mut V, inp: &V);
    /// `out += (1 + γ_mu) U_mu(x - mu)^† inp(x - mu)`.
    fn hop_backward(&self, work: &mut Self::Work, mu: usize, out: &mut V, inp: &V);
    /// `out += (site-diagonal part) inp`.
    fn mult_diag(&self, work: &mut Self::Work, out: &mut V, inp: &V);
    /// `out = γ5 inp`.
    fn mult_gamma5(&self, work: &mut Self::Work, out: &mut V, inp: &V);
    /// The coefficient of the hopping terms in D: `D = diag + hop_coefficient * Σ hops`.
    fn hop_coefficient(&self) -> f64;
}
