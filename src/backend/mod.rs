//! Backends that need an external library, each behind a feature of the same name.
//!
//! - `mpi`: the communicator on MPI (feature `mpi`)

#[cfg(feature = "mpi")]
pub mod mpi;
