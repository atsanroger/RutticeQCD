//! The RutticeQCD library.
//!
//! One directory per subject:
//!
//! - `field`: fields, storage types, the vector and linear-operator interfaces (re-exported
//!   at the top level)
//! - `lattice`: lattice geometry, stencils, even-odd decomposition, site-parallel entry
//!   points (re-exported at the top level)
//! - `comm`: communication between processes
//! - `backend`: backends that need an external library; `backend::mpi` with the feature `mpi`
//! - `kernel`: arithmetic on one site (complex numbers, colour matrices, spinors, gammas)
//! - `numerics`: small dense linear algebra
//! - `io`: reading and writing gauge configurations
//! - `gauge`: plaquette, field strength, gauge transformations and gauge fixing, smearing,
//!   products of links along paths
//! - `fermion`: Wilson, clover and Möbius domain-wall operators, even-odd preconditioning
//! - `solver`: Krylov solvers, mixed precision, preconditioners (polynomials, multigrid)
//! - `hmc`: hybrid Monte Carlo: actions and their forces, integrators
//! - `measure`: sources and correlators
//! - `app`: parameter files, building what they name, running and reporting
//!
//! Fields are in the usual layout (`Aos`) or in blocks of W sites for SIMD (`Aosoa<W>`), on
//! the whole lattice or on one parity; the only backend is `Cpu`. Several processes are
//! supported by the stencil, `shift`, and the reductions.

pub mod app;
pub mod backend;
pub mod comm;
pub mod fermion;
pub mod gauge;
pub mod hmc;
pub mod io;
pub mod kernel;
pub mod measure;
pub mod numerics;
pub mod solver;

mod field;
mod lattice;

pub use comm::{opposite_face, Comm, HaloExchange, HaloSpec, LocalComm, SingleComm, FACES};
pub use field::blas::{dot, fill, norm_sqr};
pub use field::linear::{fit, Adjoint, LinearFn, LinearOp, Normal, Product};
pub use field::random::{random_spinor, TestRng};
pub use field::storage::{stream_fence, Storage};
pub use field::vector::{ConvertFrom, Vector, C64};
pub use field::{
    Aos, Aosoa, Backend, Coarse, Cpu, Field, Full, Gauge, Half, Kind, Layout, Sites, Spinor,
};
pub use lattice::blocks::{BlockStencil, Blocks, Edge, NOT_RECEIVED};
pub use lattice::evenodd::{extract_parity, insert_parity, EvenOddIndex, Parity, ParityBlocks};
pub use lattice::parts::{assemble, local_part};
pub use lattice::pool::{schedule, Schedule};
pub use lattice::sites::{
    for_light_sites, for_site_runs, for_sites, for_sites_as, for_sites_then, min_sites_per_task,
    reduce_sites, reduce_sites_dyn, reduce_sites_n, shift, stream_min_bytes, streams, SiteRuns,
    REDUCE_CHUNK,
};
pub use lattice::stencil::{Boundary, Direction, Source, Stencil};
pub use lattice::{Lattice, ND};
