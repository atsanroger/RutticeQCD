//! The work on the links (`meter::Work::Gauge`): one formula for each operation, which the CPU
//! and the device both count, since they do the same arithmetic (`gauge/stout.cuh` and
//! `gauge/links.cuh` follow `gauge/smearing.rs` and `hmc/`). A product of two 3 × 3 complex
//! matrices is 198 flops, a sum or a real multiple 18 (`meter`); a path of n links is n − 1
//! products. The bytes are a model: every field an operation reads or writes, once (72 reals a
//! site). The halos of several processes are communication and are not counted, as those of the
//! fermion fields.

use crate::meter::{MATRIX_PRODUCT as P, MATRIX_REALS, MATRIX_SUM as S};

/// (M − M†)/2 − tr(M − M†)/(2 N_c): the difference and the half (18 each), the trace and the
/// shift of the diagonal (6).
const TA: f64 = 2.0 * S + 6.0;
/// Re tr of a matrix: two additions.
const TRACE: f64 = 2.0;
/// The reunitarisation of `exp_update`: the rows normalised and made orthogonal (81), the third
/// the cross product (42).
const REUNITARIZE: f64 = 123.0;
/// exp(iQ) of a stout step: C U† and its projection, Q² and Q³ (3 products and TA), |Q|² (35),
/// and the scalars u, w, f_j (about 100 with the trigonometric functions).
const STOUT_EXP: f64 = 3.0 * P + TA + 35.0 + 100.0;
/// a + b Q + c Q²: two complex products and a sum a matrix entry, a on the diagonal.
const POLY: f64 = 9.0 * 14.0 + 6.0;
/// iΛ of Morningstar–Peardon (73): the scalars r, b (about 250), two polynomials, five
/// products and two traces, Γ entry by entry (38 each) and its projection.
const LAMBDA: f64 = 250.0 + 2.0 * POLY + 5.0 * P + 8.0 + 9.0 * 38.0 + TA;
/// The SU(2) matrix of one subgroup that maximises Re tr(A v) (`su2_max`).
const SU2_MAX: f64 = 45.0;

fn path(links: usize) -> f64 {
    (links - 1) as f64 * P
}

/// The planes (μ, ν ≠ μ) whose staples have a weight.
fn weighted_planes(rho: &[[f64; 4]; 4]) -> f64 {
    let mut n = 0;
    for (mu, row) in rho.iter().enumerate() {
        n += row
            .iter()
            .enumerate()
            .filter(|&(nu, r)| nu != mu && *r != 0.0)
            .count();
    }
    n as f64
}

/// Counts `flops` a site over the `sites` of the lattice, reading or writing `fields` fields of
/// links, in double precision.
fn count(sites: usize, flops: f64, fields: f64) {
    let n = sites as f64;
    let bytes = n * fields * 4.0 * MATRIX_REALS * 8.0;
    crate::meter::add(crate::meter::Work::Gauge, 8, n * flops, bytes);
}

/// The weighted staples of every link: two paths of three links, their sum, the weight and the
/// sum over the planes, a site.
fn staples_flops(rho: &[[f64; 4]; 4]) -> f64 {
    weighted_planes(rho) * (2.0 * path(3) + 3.0 * S)
}

/// One stout step (`Stout::apply`, `stout_step_kernel`): the staples, exp(iQ), exp(iQ) U.
pub(crate) fn stout_step(sites: usize, rho: &[[f64; 4]; 4]) {
    count(
        sites,
        staples_flops(rho) + 4.0 * (STOUT_EXP + POLY + P),
        2.0,
    );
}

/// One step of the chain rule of stout smearing (`Stout::chain`, the two chain kernels): the
/// staples, exp(iQ) and iΛ, Σ' exp(iQ) + C† iΛ and iΘ = iΛ U, then the six terms of every
/// weighted plane.
pub(crate) fn stout_chain(sites: usize, rho: &[[f64; 4]; 4]) {
    let local = 4.0 * (STOUT_EXP + LAMBDA + POLY + 3.0 * P + S);
    let planes = weighted_planes(rho) * (6.0 * path(3) + 7.0 * S);
    count(sites, staples_flops(rho) + local + planes, 8.0);
}

/// APE smearing (`Ape::apply`): the staples and (1 − ρ) U + C.
pub(crate) fn ape_step(sites: usize, rho: &[[f64; 4]; 4]) {
    count(sites, staples_flops(rho) + 4.0 * 2.0 * S, 3.0);
}

/// One sweep of APE's SU(3) projection: three subgroups, each chosen and multiplied into A, G
/// and the change, and the trace of the change.
pub(crate) fn ape_sweep(sites: usize) {
    count(sites, 4.0 * (3.0 * (SU2_MAX + 3.0 * P) + TRACE), 2.0);
}

/// A product of two fields of links, link by link.
pub(crate) fn products(sites: usize) {
    count(sites, 4.0 * P, 3.0);
}

/// s TA(M) of every link.
pub(crate) fn projections(sites: usize) {
    count(sites, 4.0 * (TA + S), 2.0);
}

/// y += a x for fields of links.
pub(crate) fn axpy(sites: usize) {
    count(sites, 4.0 * 2.0 * MATRIX_REALS, 3.0);
}

/// |P|² of momenta.
pub(crate) fn norm(sites: usize) {
    count(sites, 4.0 * 2.0 * MATRIX_REALS, 1.0);
}

/// U ← exp(ε iP) U by the nested Taylor series to `order` (a product, a scaling and a sum each),
/// then reunitarised.
pub(crate) fn exp_update(sites: usize, order: usize) {
    count(
        sites,
        4.0 * (order as f64 * (P + 2.0 * S) + REUNITARIZE),
        3.0,
    );
}

/// The plaquettes of every site: six loops of four links and their traces.
pub(crate) fn plaquettes(sites: usize) {
    count(sites, 6.0 * (path(4) + TRACE), 1.0);
}

/// The gauge action: the plaquettes, and with rectangles twelve loops of six links a site.
pub(crate) fn gauge_action(sites: usize, rectangles: bool) {
    let rect = if rectangles {
        12.0 * (path(6) + TRACE)
    } else {
        0.0
    };
    count(sites, 6.0 * (path(4) + TRACE) + rect, 1.0);
}

/// The gauge force: for every link and plane, the weighted staples of the plaquettes (two
/// paths of three links) and of the rectangles (six of five), then U Σ†, TA and the factor.
pub(crate) fn gauge_force(sites: usize, plaquettes: bool, rectangles: bool) {
    let mut plane = 0.0;
    if plaquettes {
        plane += 2.0 * (path(3) + 2.0 * S);
    }
    if rectangles {
        plane += 6.0 * (path(5) + 2.0 * S);
    }
    count(sites, 4.0 * (3.0 * plane + P + TA + S), 2.0);
}
