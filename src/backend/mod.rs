//! Backends that need an external library, each behind a feature of the same name.
//!
//! - `mpi`: the communicator on MPI (feature `mpi`)
//! - `cuda_bridge`: the CUDA kernels of `cuda_bridge/` (feature `cuda-bridge`)

#[cfg(feature = "mpi")]
pub mod mpi;

#[cfg(feature = "cuda-bridge")]
pub mod cuda_bridge;
