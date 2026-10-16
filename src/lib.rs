//! The RutticeQCD library.

pub mod app;
pub mod backend;
pub mod comm;
pub mod eigen;
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
pub use field::linear::{fit, Adjoint, LinearFn, LinearOp, Normal, Product, Subspace};
pub use field::random::{random_spinor, TestRng};
pub use field::storage::{stream_fence, widen_halves, Storage, F16};
pub use field::vector::{ConvertFrom, Vector, C64};
pub use field::{
    Aos, Aosoa, Backend, Coarse, Cpu, Field, Gauge, Half, Kind, Layout, Lexical, Sites, Spinor,
};
pub use lattice::blocks::{BlockStencil, Blocks, Edge, NOT_RECEIVED};
pub use lattice::even_odd::{extract_parity, insert_parity, EvenOddIndex, Parity, ParityBlocks};
pub use lattice::for_sites::{
    for_light_sites, for_site_runs, for_sites, for_sites_as, for_sites_then, min_sites_per_task,
    reduce_sites, reduce_sites_dyn, reduce_sites_n, stream_min_bytes, streams, SiteRuns,
    REDUCE_CHUNK,
};
pub use lattice::parts::{assemble, local_part};
pub use lattice::pool::{schedule, Schedule};
pub use lattice::stencil::{shift, Boundary, Direction, Source, Stencil};
pub use lattice::{Lattice, ND};
