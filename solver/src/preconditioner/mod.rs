//! Preconditioners: approximate inverses passed to a solver as its `m` argument.
//!
//! Unlike the Krylov solvers, these know what the operator is made of: its hopping
//! terms and the lattice it lives on.

pub mod multigrid;
