//! The work of the setup of the multigrids (`meter::Work::Coarse`, `Vector`): one formula for
//! each operation, which the CPU and the device both count. The operator applications of a
//! setup (probing, the test vectors) count themselves; these are the dense parts around them.
//! Copies, signs and casts count their bytes as vector operations.

use crate::meter::{add, gauss_jordan, Work};

/// The sweeps of the Jacobi method assumed for |H|⁻¹: the number is not known before; six is
/// what 12 × 12 to 48 × 48 Hermitian blocks take to 1e-15 (an estimate).
const JACOBI_SWEEPS: f64 = 6.0;

fn count(work: Work, real_bytes: usize, flops: f64, reals: f64) {
    add(work, real_bytes, flops, reals * real_bytes as f64);
}

/// The inverses of `sites` diagonal blocks of `n` × `n` complex numbers, by Gauss–Jordan.
pub(crate) fn invert_blocks(sites: usize, n: usize) {
    let s = sites as f64;
    count(
        Work::Coarse,
        8,
        s * gauss_jordan(n),
        s * 4.0 * (n * n) as f64,
    );
}

/// |H|⁻¹ = V |Λ|⁻¹ V† of `sites` Hermitian blocks of size `n` (H = A Γ5 first): the Jacobi
/// method (`JACOBI_SWEEPS` sweeps of n (n − 1)/2 rotations, each of two rows and two columns of
/// H and two columns of V, 12 n flops), then V |Λ|⁻¹ V† (8 n³).
pub(crate) fn abs_inverse_blocks(sites: usize, n: usize) {
    let (s, m) = (sites as f64, n as f64);
    let jacobi = JACOBI_SWEEPS * m * (m - 1.0) / 2.0 * 3.0 * 12.0 * m;
    count(
        Work::Coarse,
        8,
        s * (jacobi + 8.0 * m * m * m),
        s * 4.0 * m * m,
    );
}

/// A dense complex n × n matrix applied to `vectors` vectors.
pub(crate) fn dense_apply(n: usize, vectors: usize, real_bytes: usize) {
    let (m, v) = (n as f64, vectors as f64);
    count(
        Work::Coarse,
        real_bytes,
        8.0 * m * m * v,
        2.0 * m * m + 4.0 * m * v,
    );
}

/// The orthonormal basis of `vectors` vectors over `blocks` aggregates of `m` complex numbers
/// each, by Gram–Schmidt: a projection and a subtraction a pair (8 m each), a norm and a scaling a
/// vector (8 m).
pub(crate) fn gram_schmidt(blocks: usize, m: usize, vectors: usize, real_bytes: usize) {
    let (b, m, n) = (blocks as f64, m as f64, vectors as f64);
    let flops = b * (n * (n - 1.0) / 2.0 * 16.0 * m + n * 8.0 * m);
    count(Work::Vector, real_bytes, flops, b * 4.0 * m * n);
}

/// The local vectors of a level (`level_basis`): in every super-block of `m` complex numbers,
/// the `nq / 2` vectors of each chirality made orthonormal by Gram–Schmidt run twice.
pub(crate) fn level_basis(blocks: usize, m: usize, nq: usize) {
    let (b, m, n) = (blocks as f64, m as f64, (nq / 2) as f64);
    let one = 2.0 * n * (n - 1.0) / 2.0 * 16.0 * m + n * 8.0 * m;
    count(Work::Vector, 8, b * 2.0 * one, b * 4.0 * m * nq as f64);
}

/// Q† v or Q y between a level and the one above (`parent_sites` sites of `pd` complex numbers
/// there, `nq` local vectors a super-block): 8 flops for every complex number above and local
/// vector; Q and the two fields read or written once.
pub(crate) fn level_transfer(
    parent_sites: usize,
    pd: usize,
    nq: usize,
    vectors: usize,
    real_bytes: usize,
) {
    let above = (parent_sites * pd) as f64;
    let v = vectors as f64;
    add(
        Work::Transfer,
        real_bytes,
        8.0 * above * nq as f64 * v,
        (2.0 * above * nq as f64 + 2.0 * above * v + 2.0 * above / pd as f64 * nq as f64 * v)
            * real_bytes as f64,
    );
}

/// A copy or rearrangement of `reals` reals (cast, sign, transpose): read and written once.
pub(crate) fn copy(reals: usize, real_bytes: usize) {
    count(Work::Vector, real_bytes, 0.0, 2.0 * reals as f64);
}
