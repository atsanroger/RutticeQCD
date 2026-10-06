//! Aggregation-based algebraic multigrid for operators of Wilson type.
//!
//! The construction is the standard one (Babich et al., Phys. Rev. Lett. 105 (2010)
//! 201602; Frommer et al., SIAM J. Sci. Comput. 36 (2014) A1581): the lattice is cut into
//! blocks, each block is split by chirality into two aggregates, and a set of test vectors
//! restricted to each aggregate and orthonormalised there forms the columns of the
//! prolongation P. Restriction is P†, and the coarse operator is P† D P.
//!
//! Keeping the two chiralities apart makes P commute with γ5, so the coarse operator
//! inherits γ5-hermiticity from the fine one.

mod aggregation;
mod basis;
mod coarse;

pub use aggregation::Aggregation;
pub use basis::Basis;
pub use coarse::CoarseOp;

use std::fmt;

#[derive(Debug)]
pub enum MultigridError {
    /// The test vectors are linearly dependent within one aggregate, so they cannot be
    /// orthonormalised there. Fewer vectors or larger blocks are needed.
    DependentTestVectors { coarse_site: usize, vector: usize },
}

impl fmt::Display for MultigridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultigridError::DependentTestVectors { coarse_site, vector } => write!(
                f,
                "test vector {vector} depends linearly on the earlier ones in the block at coarse site {coarse_site}"
            ),
        }
    }
}

impl std::error::Error for MultigridError {}
