//! The work of the fermion forces (`meter::Work::Force`) and of the clover blocks
//! (`meter::Work::Gauge`): one formula for each operation, which the CPU and the device both
//! count (`fermion/force.cuh`, `fermion/clover_links.cuh` follow `hmc/fermion_action.rs` and
//! `fermion/clover`). Where the two compute an operation in another way (the insertion of the
//! log det: the device reads T⁻¹, the CPU applies it to twelve unit vectors), both count the
//! device's arithmetic and run their pieces uncounted (`meter::uncounted`). Products of 3 × 3
//! complex matrices are 198 flops, sums 18 (`meter`).

use crate::meter::{add, gauss_jordan, Work, MATRIX_PRODUCT as P, MATRIX_REALS, MATRIX_SUM as S};

/// One leg of the hops of a spinor, Bridge++'s count of the eight (1320) shared out.
const LEG: f64 = 1320.0 / 8.0;
/// a b† summed into a colour matrix, for the four spins: 9 entries, the products and the
/// coefficient of the two terms (20 each).
const OUTER: f64 = 4.0 * 9.0 * 20.0;
/// γ_μ γ_ν of a plane from the table (4 complex products).
const GAMMA_PAIR: f64 = 24.0;
/// B += coef (A − A†): the difference, the coefficient and the sum.
const INSERT: f64 = 3.0 * S;
/// The planes, the leaves of a plane, the links of a leaf.
const PLANES: f64 = 6.0;
/// A 6 × 6 complex block of T.
const BLOCK: usize = 6;

fn count(work: Work, sites: usize, flops: f64, reals: f64) {
    let n = sites as f64;
    add(work, 8, n * flops, n * reals * 8.0);
}

/// The derivative by the links of the hops of a Wilson-type operator acting on η (`ls` spinors a
/// site): for every link and spinor, the forward leg of η and of γ5 ζ, and the two outer products.
/// Reads ζ, η, γ5 ζ, γ5 η and the links, writes M.
pub(crate) fn hop_force(sites: usize, ls: usize) {
    let l = ls as f64;
    count(
        Work::Force,
        sites,
        4.0 * l * (2.0 * LEG + OUTER),
        4.0 * 24.0 * l + 2.0 * 4.0 * MATRIX_REALS,
    );
}

/// The clover insertion of one pair (ζ, η): for the six planes, A = Σ_spin γ_μγ_ν η ζ† (for
/// every spin and colour i, (γη)_i and its three products with conj ζ_j: 6 + 24) and B += coef
/// (A − A†). Reads the two spinors, reads and writes B.
pub(crate) fn clover_insertion(sites: usize) {
    let plane = GAMMA_PAIR + 4.0 * 3.0 * (6.0 + 24.0) + INSERT;
    count(
        Work::Force,
        sites,
        PLANES * plane,
        2.0 * 24.0 + 2.0 * PLANES * MATRIX_REALS,
    );
}

/// The insertion of the derivative of −2 ln |det T| on `sites` sites (those of one parity): for
/// the six planes and every (spin, i, j), an entry of T⁻¹ from the two blocks (8) times γ_μγ_ν
/// (6), then B += coef (A − A†). Reads the blocks of T⁻¹ (72 reals), reads and writes B.
pub(crate) fn log_det_insertion(sites: usize) {
    let plane = GAMMA_PAIR + 36.0 * 14.0 + INSERT;
    count(
        Work::Force,
        sites,
        PLANES * plane,
        72.0 + 2.0 * PLANES * MATRIX_REALS,
    );
}

/// The leaves through every link with B inserted at their corner: three planes a link, eight
/// terms a plane (the two links of each leaf in the direction of the link), each three links,
/// B and the sum (4 products and a sum). Reads the links and the two fields of B, writes M.
pub(crate) fn clover_leaves(sites: usize) {
    count(
        Work::Force,
        sites,
        4.0 * 3.0 * 8.0 * (4.0 * P + S),
        4.0 * 4.0 * MATRIX_REALS,
    );
}

/// T, T⁻¹ and ln |det| of the clover blocks from the links: F_μν of the six planes (four leaves
/// of three products, their sum, (Q − Q†)/8), the two blocks of every plane (36 entries of 10
/// flops each), Gauss–Jordan on both, and the logarithms of the pivots. Reads the links, writes T,
/// T⁻¹ and two numbers.
pub(crate) fn clover_blocks(sites: usize) {
    let field_strength = PLANES * (4.0 * 3.0 * P + 3.0 * S + 2.0 * S);
    let assembly = 2.0 * PLANES * 36.0 * 10.0;
    let inverse = 2.0 * gauss_jordan(BLOCK);
    count(
        Work::Gauge,
        sites,
        field_strength + assembly + inverse + 2.0 * BLOCK as f64,
        4.0 * MATRIX_REALS + 2.0 * 72.0 + 2.0,
    );
}

/// ln |det T| of the CPU on `sites` sites (one parity): the twelve columns T e_k (the site block,
/// Bridge++'s count) and Gaussian elimination of the 12 × 12 matrix. The device takes it from
/// the blocks it made (`clover_blocks`), so only the CPU counts this.
pub(crate) fn log_det_from_columns(sites: usize) {
    let n = 12;
    let elimination: f64 = (0..n)
        .map(|col| {
            let rows = (n - 1 - col) as f64;
            rows * (8.0 + (n - col) as f64 * 8.0)
        })
        .sum();
    let columns = 12.0 * (12.0 * (2.0 + 48.0 * 3.0));
    count(
        Work::Gauge,
        sites,
        columns + elimination,
        72.0 + 12.0 * 24.0,
    );
}
