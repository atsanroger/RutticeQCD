//! Fermion operators. One fermion per file.

mod clover;
mod evenodd;
mod wilson;

pub use clover::{Clover, CloverDiag, CloverParams};

pub use evenodd::{solve_even_odd, Dagger, EvenOdd, Schur};

pub use wilson::{DiagTerm, UnitDiag, Wilson, WilsonOp, WilsonParams, WilsonWork};

use crate::{Field, Gauge, LinearFn, LinearOp, Vector};
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

impl Variant {
    /// The variant that applies the Hermitian conjugate.
    pub fn dagger(self) -> Self {
        match self {
            Variant::D => Variant::Dagger,
            Variant::Dagger => Variant::D,
            Variant::Gamma5D => Variant::Gamma5D,
        }
    }
}

#[derive(Debug)]
pub enum GaugeError {
    /// The gauge field lives on a different lattice than the operator.
    LatticeMismatch,
    /// The site-diagonal block of the operator could not be inverted at this site.
    SingularDiagonal { site: usize },
}

impl fmt::Display for GaugeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GaugeError::LatticeMismatch => {
                write!(f, "gauge field and operator are on different lattices")
            }
            GaugeError::SingularDiagonal { site } => {
                write!(f, "the site-diagonal block is singular at site {site}")
            }
        }
    }
}

impl std::error::Error for GaugeError {}

/// A fermion operator. The operator itself is read-only while in use; scratch space lives
/// in `Work`, so several users can apply the same operator at once.
pub trait FermionOp<V> {
    type Work;
    fn work(&self) -> Self::Work;
    /// Takes the canonical `f64` gauge field and converts it to the operator's own format.
    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError>;
    /// `out = M inp` for the chosen variant.
    fn mult(&self, work: &mut Self::Work, out: &mut V, inp: &V, variant: Variant);
    /// Floating-point operations of one `mult` per site and per vector, for reporting
    /// speed as GFlops.
    ///
    /// This is a nominal count of the arithmetic the operator needs, taken from the
    /// counting Bridge++ uses for the same operator (its `flop_count`), so that the
    /// figures of the two programs can be compared. It is not a count of the instructions
    /// this implementation executes.
    fn flops_per_site(&self) -> usize;
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

/// A fermion operator fixed to one variant, with its own scratch space. This is what
/// solvers and combinators see.
///
/// It borrows the operator, so several can exist at once (an outer solver and an inner
/// smoother sharing one operator and one copy of the gauge field), and the gauge field
/// cannot be replaced while any of them is alive:
///
/// ```
/// use ruttice::{Boundary, Field, Gauge, Lattice};
/// use ruttice::fermion::{Bound, FermionOp, Variant, Wilson, WilsonParams};
///
/// let lat = Lattice::single_node([2, 2, 2, 2]);
/// let u = Field::<f64, Gauge>::gauge(&lat);
/// let params = WilsonParams { kappa: 0.1, boundary: [Boundary::Periodic; 4] };
/// let mut op = Wilson::<f64>::new(&lat, params);
/// op.set_gauge(&u).unwrap();
/// {
///     let _outer = Bound::new(&op, Variant::D);
///     let _inner = Bound::new(&op, Variant::Dagger);
/// }
/// op.set_gauge(&u).unwrap(); // fine: both bindings are gone
/// ```
///
/// ```compile_fail,E0502
/// use ruttice::{Boundary, Field, Gauge, Lattice};
/// use ruttice::fermion::{Bound, FermionOp, Variant, Wilson, WilsonParams};
///
/// let lat = Lattice::single_node([2, 2, 2, 2]);
/// let u = Field::<f64, Gauge>::gauge(&lat);
/// let params = WilsonParams { kappa: 0.1, boundary: [Boundary::Periodic; 4] };
/// let mut op = Wilson::<f64>::new(&lat, params);
/// op.set_gauge(&u).unwrap();
/// let bound = Bound::new(&op, Variant::D);
/// op.set_gauge(&u).unwrap(); // rejected: `op` is still borrowed by `bound`
/// drop(bound);
/// ```
pub struct Bound<'a, F: FermionOp<V>, V> {
    op: &'a F,
    work: F::Work,
    variant: Variant,
}

impl<'a, F: FermionOp<V>, V> Bound<'a, F, V> {
    pub fn new(op: &'a F, variant: Variant) -> Self {
        Self {
            op,
            work: op.work(),
            variant,
        }
    }
}

impl<F: FermionOp<V>, V: Vector> LinearFn<V> for Bound<'_, F, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.op.mult(&mut self.work, out, inp, self.variant);
    }
}

impl<F: FermionOp<V>, V: Vector> LinearOp<V> for Bound<'_, F, V> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.op
            .mult(&mut self.work, out, inp, self.variant.dagger());
    }
}
