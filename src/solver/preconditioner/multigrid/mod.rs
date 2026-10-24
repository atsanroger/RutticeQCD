//! Aggregation-based algebraic multigrid for operators of Wilson type.

mod aggregation;
mod basis;
mod coarse;
mod correction;
mod cycle;
mod levels;
mod mdwf_backend;
mod mdwf_even_odd;
mod mdwf_lexical;
mod sap;
mod wilson;

pub use aggregation::Aggregation;
pub use basis::{Basis, BasisBuilder, Reflection};
#[cfg(feature = "cuda-bridge")]
pub(crate) use coarse::coarse_neighbours;
pub use coarse::CoarseOp;
pub use correction::{solve_shared, CoarseSolve, FnMap, NeffQ};
pub use correction::{CoarseOperator, Correction, Levels, Transfer};
pub use cycle::{Corrects, TwoGrid};
pub use levels::{AbsMultilevel, BlockMap, LevelBootstrap, SignedMultilevel};
pub use mdwf_backend::{top_of as pv_eo_top, AnyParts, CpuParts, CpuVectors, Outer, PartsIn};
pub use mdwf_backend::{
    CoarseLevels, Cycles, MdwfBackend, MdwfMultigridSettings, OnCpu, SetupStage,
};
pub use mdwf_even_odd::{
    bootstrap_round, bootstrap_vector, coarse_solve, low_modes, orthonormalise_all,
    pv_eo_basis_stored, pv_eo_bootstrap_stored, pv_eo_start, BasisStorage, CoarseParts,
    CycleLayout, SharedOperator,
};
pub use mdwf_even_odd::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, random_even,
    BootstrapParams, CoarseKrylov, LowModeParams, PvEoBasis, PvEoCycle, PvEoParams,
};
pub use mdwf_lexical::{mdwf_multigrid, MdwfCoarse, MdwfMultigridParams, MdwfSmoother};
pub use sap::{CpuBlocks, Sap, SapBlocks};
pub use wilson::{
    random_vectors, CoarseSolverChoice, CycleLevels, CycleParts, Multigrid, MultigridBackend,
    MultigridParams, SetupHooks, SmootherChoice, WilsonMultigrid, WilsonOnCpu,
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
    /// What a backend could not do (the device: memory, a kernel), or does not have.
    Backend(String),
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
            MultigridError::Backend(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for MultigridError {}
