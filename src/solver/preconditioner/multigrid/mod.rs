//! Aggregation-based algebraic multigrid for operators of Wilson type.

mod aggregation;
mod basis;
mod coarse;
mod correction;
pub(crate) mod count;
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
pub(crate) use correction::solve_shared;
pub(crate) use correction::CoarseOperator;
pub use correction::{Levels, Transfer};

pub use levels::{AbsMultilevel, BlockMap, LevelBootstrap, SignedMultilevel};
pub use mdwf_backend::MdwfMultigridSettings;
pub(crate) use mdwf_backend::{AnyParts, CpuParts, PartsIn};
pub(crate) use mdwf_backend::{CoarseLevels, Cycles, MdwfBackend, OnCpu, SetupStage};
pub(crate) use mdwf_even_odd::LevelLists;
// What only the device backend uses.
#[cfg(feature = "cuda-bridge")]
pub(crate) use correction::{Correction, FnMap, NeffQ};
#[cfg(feature = "cuda-bridge")]
pub(crate) use cycle::TwoGrid;
#[cfg(feature = "cuda-bridge")]
pub(crate) use mdwf_backend::Outer;
#[cfg(feature = "cuda-bridge")]
pub(crate) use mdwf_even_odd::{bootstrap_rounds, lifted_vectors, low_modes, Waiting};
pub use mdwf_even_odd::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, random_even,
    BootstrapParams, CoarseKrylov, LowModeParams, PvEoBasis, PvEoCycle, PvEoParams,
};
pub(crate) use mdwf_even_odd::{
    pv_eo_basis_stored, pv_eo_bootstrap_stored, BasisStorage, CoarseParts, CycleLayout,
    SharedOperator,
};
pub use mdwf_lexical::{mdwf_multigrid, MdwfCoarse, MdwfMultigridParams, MdwfSmoother};
pub use sap::Sap;
#[cfg(feature = "cuda-bridge")]
pub(crate) use sap::SapBlocks;
#[cfg(feature = "cuda-bridge")]
pub(crate) use wilson::CycleLevels;

pub use wilson::{
    random_vectors, CycleParts, Multigrid, MultigridParams, SetupHooks, WilsonMultigrid,
};
pub(crate) use wilson::{CoarseSolverChoice, MultigridBackend, SmootherChoice, WilsonOnCpu};

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
