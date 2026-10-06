//! Fields, lattice geometry, stencils and site-parallel entry points (DESIGN.md §2–§4).
//!
//! Stage 0 scope: `Aos` layout, `Full` sites, `Cpu` backend, single node.

mod comm;
mod field;
mod lattice;
mod sites;
mod stencil;
mod storage;

pub use comm::{Comm, SingleComm};
pub use field::{Aos, Backend, Coarse, Cpu, Field, Full, Gauge, Half, Kind, Layout, Sites, Spinor};
pub use lattice::{Lattice, ND};
pub use sites::{for_sites, reduce_sites, shift, REDUCE_CHUNK};
pub use stencil::{Boundary, Direction, Source, Stencil};
pub use storage::Storage;
