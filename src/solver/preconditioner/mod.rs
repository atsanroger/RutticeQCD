//! Preconditioners: approximate inverses passed to a solver as its `m` argument.
//!
//! - `chebyshev`, `polynomial`: fixed polynomials of the operator (Chebyshev iteration, a
//!   fitted minimal-residual polynomial). They know only `Vector` and `LinearFn`, like the
//!   Krylov solvers, and work for any operator. The Chebyshev low-pass filter that makes
//!   multigrid test vectors is with them.
//! - `multigrid`: these know what the operator is made of, its hopping terms and the
//!   lattice it lives on. The multigrid for Wilson-type operators and for domain-wall
//!   fermions, and the domain decomposition (`Sap`) it smooths with, which is a
//!   preconditioner of its own as well.

mod chebyshev;
pub mod multigrid;
mod polynomial;

pub use chebyshev::{largest_magnitude, Chebyshev, ChebyshevFilter};
pub use polynomial::Polynomial;
