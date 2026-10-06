//! Fields, lattice geometry, stencils and site-parallel entry points (DESIGN.md §2–§4).
//!
//! Stage 0 scope: `Aos` layout, `Full` sites, `Cpu` backend, single node.

mod blas;
mod comm;
mod evenodd;
mod field;
mod lattice;
mod linear;
mod sites;
mod stencil;
mod storage;
mod vector;

pub use blas::{dot, fill, norm_sqr};
pub use comm::{Comm, SingleComm};
pub use evenodd::{extract_parity, insert_parity, EvenOddIndex, Parity};
pub use field::{Aos, Backend, Coarse, Cpu, Field, Full, Gauge, Half, Kind, Layout, Sites, Spinor};
pub use lattice::{Lattice, ND};
pub use linear::{Adjoint, LinearFn, LinearOp, Normal, Product};
pub use sites::{for_sites, reduce_sites, reduce_sites_dyn, reduce_sites_n, shift, REDUCE_CHUNK};
pub use stencil::{Boundary, Direction, Source, Stencil};
pub use storage::Storage;
pub use vector::{ConvertFrom, Vector, C64};
