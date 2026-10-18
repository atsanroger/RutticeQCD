//! Eigenvalues, singular values and pseudospectra of large sparse operators, on any [`Vector`]
//! (on the host or on a device), by Krylov methods that only apply the operator:
//!
//! | what | method | module |
//! |---|---|---|
//! | the lowest eigenpairs of a Hermitian positive A | thick-restart Lanczos, optionally on a Chebyshev polynomial of A | [`lanczos`] ([`lowest`]) |
//! | eigenpairs of a non-Hermitian A at an end of its spectrum | Krylov-Schur | [`arnoldi`] ([`arnoldi()`]) |
//! | the smallest singular value of A, and σ_min(z − A) for points z (the norm of the resolvent, 1/‖(z − A)⁻¹‖: the ε-pseudospectrum is where it is below ε) | Lanczos on (z − A)†(z − A) | [`singular`] |
//! | spectral transforms: A − z, and (A − σ)⁻¹ by a given solve (shift and invert, for eigenvalues in the interior or a crowded low end) | | [`transform`] |
//!
//! The small dense problems inside (the Schur form of a projection, its reordering, triangular
//! eigenvectors) are in [`schur`]; Hermitian ones use `numerics::hermitian_eigen`. Every
//! method keeps all its Krylov vectors and orthogonalises twice (no loss of orthogonality): the
//! memory is the Krylov dimension in vectors.

pub mod arnoldi;
pub mod lanczos;
pub mod schur;
pub mod singular;
pub mod transform;

pub use arnoldi::{arnoldi, ArnoldiParams, NonHermitianPairs, Which};
pub use lanczos::{lowest, Chebyshev, Eigenpairs, LanczosParams};
pub use singular::{resolvent_norms, smallest_singular, Resolvent, Singular};
pub use transform::{ShiftInvert, Shifted};

use crate::{Vector, C64};

fn dot<V: Vector>(a: &V, b: &V) -> C64 {
    let mut d = [C64::ZERO];
    a.dot(b, &mut d);
    d[0]
}

fn norm<V: Vector>(a: &V) -> f64 {
    let mut n = [0.0];
    a.norm_sqr(&mut n);
    n[0].sqrt()
}

/// Σ_j basis[j] c(j).
fn combine<V: Vector>(basis: &[V], c: impl Fn(usize) -> C64) -> V {
    let mut x = basis[0].zeros_like();
    for (j, v) in basis.iter().enumerate() {
        x.axpy(&[c(j)], v);
    }
    x
}
