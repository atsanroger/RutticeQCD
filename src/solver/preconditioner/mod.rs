//! Preconditioners: approximate inverses passed to a solver as its `m` argument.

mod chebyshev;
pub mod multigrid;
mod polynomial;

pub use chebyshev::{largest_magnitude, Chebyshev, ChebyshevFilter};
pub use polynomial::Polynomial;
