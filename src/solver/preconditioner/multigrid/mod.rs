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
mod cycle;
mod levels;
mod mdwf;
mod pv_eo;
mod sap;

pub use aggregation::Aggregation;
pub use basis::{Basis, BasisBuilder, Reflection};
pub use coarse::CoarseOp;
pub use cycle::{
    CoarseSolverChoice, CycleParts, Multigrid, MultigridParams, SetupHooks, SmootherChoice,
};
pub use levels::{LevelBootstrap, SignedMultilevel};
pub use mdwf::{mdwf_multigrid, MdwfCoarse, MdwfMultigridParams, MdwfSmoother};
pub use pv_eo::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, BootstrapParams,
    CoarseKrylov, CycleLayout, LowModeParams, PvEoBasis, PvEoCycle, PvEoParams, SharedOperator,
};
pub use sap::Sap;

use std::fmt;

#[derive(Debug)]
pub enum MultigridError {
    /// The test vectors are linearly dependent within one aggregate, so they cannot be
    /// orthonormalised there. Fewer vectors or larger blocks are needed.
    DependentTestVectors { coarse_site: usize, vector: usize },
    /// The diagonal block of the coarse operator cannot be inverted at this coarse site.
    SingularCoarseDiagonal { coarse_site: usize },
    /// The powers of the operator that a polynomial smoother of this degree is fitted to
    /// are linearly dependent to working precision. A lower degree is needed.
    DependentPowers { degree: usize },
}

impl fmt::Display for MultigridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultigridError::DependentTestVectors { coarse_site, vector } => write!(
                f,
                "test vector {vector} depends linearly on the earlier ones in the block at coarse site {coarse_site}"
            ),
            MultigridError::SingularCoarseDiagonal { coarse_site } => write!(
                f,
                "the diagonal block of the coarse operator is singular at coarse site {coarse_site}"
            ),
            MultigridError::DependentPowers { degree } => write!(
                f,
                "a polynomial smoother of degree {degree} cannot be fitted: the powers of the operator are linearly dependent"
            ),
        }
    }
}

impl std::error::Error for MultigridError {}
