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
//!
//! Files: what every multigrid here is made of (`aggregation`, `basis` for P, `coarse` for
//! the coarse operator and its even-odd form, `levels` below the coarse lattice, `sap` the
//! domain-decomposition smoother), and one file per kind of operator: `wilson` the two-level
//! cycle for operators of Wilson type (Wilson, clover); `mdwf_even_odd` the multigrid for
//! domain-wall fermions on the even sites (the one in use); `mdwf_lexical` the earlier one
//! on the whole lattice.

mod aggregation;
mod basis;
mod coarse;
mod levels;
mod mdwf_even_odd;
mod mdwf_lexical;
mod sap;
mod wilson;

pub use aggregation::Aggregation;
pub use basis::{Basis, BasisBuilder, Reflection};
pub use coarse::CoarseOp;
pub use levels::{LevelBootstrap, SignedMultilevel};
pub use mdwf_even_odd::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, BootstrapParams,
    CoarseKrylov, CycleLayout, LowModeParams, PvEoBasis, PvEoCycle, PvEoParams, SharedOperator,
};
pub use mdwf_lexical::{mdwf_multigrid, MdwfCoarse, MdwfMultigridParams, MdwfSmoother};
pub use sap::Sap;
pub use wilson::{
    CoarseSolverChoice, CycleParts, Multigrid, MultigridParams, SetupHooks, SmootherChoice,
};

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
