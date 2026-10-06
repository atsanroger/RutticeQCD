//! Coulomb gauge fixing.
//!
//! The gauge is chosen to maximise F = Σ_x Σ_{i spatial} Re tr U_i(x) on every time slice,
//! which makes the lattice divergence of the spatial gauge potential vanish. The algorithm
//! is the usual one: sweep over the even sites and then the odd ones; at each site take
//! the sum W of the spatial links touching it, find the gauge transformation g that
//! increases Re tr(g W) by maximising it in the three SU(2) subgroups in turn
//! (Cabibbo-Marinari), overrelax it to g^ω, and apply it. Sites of one parity do not share
//! links, so all of them are updated at once.
//!
//! Parameters, the overrelaxation step and the convergence measure follow the Bridge++
//! class `GaugeFixing_Coulomb`, so that the same parameters stop at the same step.

use super::{link, shifted, BWD, FWD};
use crate::kernel::{ColorMatrix, Cplx, NC};
use crate::{for_sites, reduce_sites_dyn, Boundary, Field, Gauge, Lattice, Parity, Stencil, ND};
use std::fmt;
use std::sync::Arc;

/// The Coulomb condition involves the spatial directions 0 .. TIME.
const TIME: usize = ND - 1;

#[derive(Clone, Copy, Debug)]
pub struct CoulombParams {
    /// Give up after this many sweeps.
    pub max_iterations: usize,
    /// Sweeps without overrelaxation at the start and after every reset.
    pub naive_iterations: usize,
    /// Measure the violation every this many sweeps.
    pub measure_interval: usize,
    /// Every this many sweeps, apply a random gauge transformation to the time slices that
    /// have not converged yet, to get out of a slow region.
    pub reset_interval: usize,
    /// Stop when the violation is below this on every time slice (see `coulomb_violation`).
    pub tolerance_sqr: f64,
    /// Overrelaxation parameter ω; 1 is no overrelaxation.
    pub overrelaxation: f64,
    /// Seed for the random gauge transformations of a reset. The transformation at a site
    /// depends only on the seed, the global site and the reset count, not on the process grid.
    pub seed: u64,
}

/// The parameters of the Bridge++ test `Spectrum_Clover_Hadron2ptFunction`.
impl Default for CoulombParams {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            naive_iterations: 50,
            measure_interval: 10,
            reset_interval: 1000,
            tolerance_sqr: 1e-24,
            overrelaxation: 1.6,
            seed: 1234567,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GaugeFixReport {
    /// Sweeps done.
    pub iterations: usize,
    /// The violation on each global time slice when it stopped.
    pub violation: Vec<f64>,
}

#[derive(Debug)]
pub enum GaugeFixError {
    NotConverged {
        iterations: usize,
        largest_violation: f64,
    },
}

impl fmt::Display for GaugeFixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GaugeFixError::NotConverged {
                iterations,
                largest_violation,
            } => write!(
                f,
                "gauge fixing did not converge in {iterations} sweeps (largest violation {largest_violation:.3e})"
            ),
        }
    }
}

impl std::error::Error for GaugeFixError {}

/// Transform `u` to Coulomb gauge in place. Works on any number of processes.
///
/// Every extent must be even, so that the checkerboard is consistent across the periodic
/// boundaries. Neighbours are always periodic: gauge fixing acts on the configuration, not
/// on a fermion operator, so the fermion boundary conditions play no part.
pub fn fix_coulomb(
    u: &mut Field<f64, Gauge>,
    params: &CoulombParams,
) -> Result<GaugeFixReport, GaugeFixError> {
    let lat = u.lattice().clone();
    assert!(
        lat.global.iter().all(|l| l % 2 == 0),
        "gauge fixing needs even extents, got {:?}",
        lat.global
    );
    assert!(params.measure_interval > 0 && params.reset_interval > 0);
    let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
    let parity: Vec<Parity> = (0..lat.num_sites())
        .map(|x| Parity::of(lat.global_coords(x)))
        .collect();
    let mut last = vec![f64::INFINITY; lat.global[TIME]];
    for iter in 0..params.max_iterations {
        // Measured only every few sweeps; the reset below sees only this sweep's measurement.
        let measured = if iter % params.measure_interval == 0 {
            let v = coulomb_violation(u, &st);
            if largest(&v) < params.tolerance_sqr {
                return Ok(GaugeFixReport {
                    iterations: iter,
                    violation: v,
                });
            }
            last = v.clone();
            Some(v)
        } else {
            None
        };
        let omega = if iter % params.reset_interval < params.naive_iterations {
            1.0
        } else {
            params.overrelaxation
        };
        for p in [Parity::Even, Parity::Odd] {
            let g = sweep_transformation(u, &st, &parity, p, omega);
            gauge_transform(u, &g, &st);
        }
        if iter > 0 && iter % params.reset_interval == 0 {
            if let Some(v) = &measured {
                let round = (iter / params.reset_interval) as u64;
                let g = random_transformation(&lat, v, params.tolerance_sqr, params.seed, round);
                gauge_transform(u, &g, &st);
            }
        }
    }
    Err(GaugeFixError::NotConverged {
        iterations: params.max_iterations,
        largest_violation: largest(&last),
    })
}

fn largest(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(*x))
}

/// How far `u` is from Coulomb gauge, per global time slice:
///
/// σ(t) = Σ_{x in slice t} |Δ(x)|² / (4 Nc V_t),  Δ(x) = [M(x) − M(x)^†]_traceless,
/// M(x) = Σ_i [U_i(x − i) − U_i(x)],
///
/// where |·|² is the sum of the squares of all real components and V_t is the number of
/// sites in a time slice. The factor 4 (the number of dimensions, not of spatial ones)
/// matches the normalisation Bridge++ uses for its convergence criterion.
///
/// The stencil must be periodic in every direction. The sum runs over all processes.
pub fn coulomb_violation(u: &Field<f64, Gauge>, st: &Stencil) -> Vec<f64> {
    let lat = u.lattice();
    let back: [_; TIME] = core::array::from_fn(|i| shifted(u, st, i, BWD));
    let nt = lat.local[TIME];
    let local = reduce_sites_dyn(lat.num_sites(), nt, |x, acc| {
        let mut m = ColorMatrix::<f64>::zero();
        for (i, b) in back.iter().enumerate() {
            m = m.add(&link(b, i, x)).sub(&link(u, i, x));
        }
        acc[lat.coords(x)[TIME]] += norm_sqr(&traceless_antihermitian(&m));
    });
    let mut sigma = vec![0.0; lat.global[TIME]];
    sigma[lat.origin[TIME]..lat.origin[TIME] + nt].copy_from_slice(&local);
    lat.comm.sum(&mut sigma);
    let slice_sites = lat.num_global_sites() / lat.global[TIME];
    for s in &mut sigma {
        *s /= (ND * NC * slice_sites) as f64;
    }
    sigma
}

/// `U_mu(x) -> g(x) U_mu(x) g(x + mu)^†` for every link. `g` holds one matrix per site.
/// The stencil must be periodic in every direction.
pub fn gauge_transform(u: &mut Field<f64, Gauge>, g: &Field<f64, Gauge>, st: &Stencil) {
    assert_eq!(g.outer(), 1, "one gauge transformation matrix per site");
    let mut out = Field::<f64, Gauge>::gauge(u.lattice());
    for mu in 0..ND {
        let g_up = shifted(g, st, mu, FWD);
        let u_ref = &*u;
        for_sites(out.slab_mut(mu), 18, |x, o| {
            link(g, 0, x)
                .mul(&link(u_ref, mu, x))
                .mul(&link(&g_up, 0, x).adj())
                .write_reals(o);
        });
    }
    *u = out;
}

/// A field of one 3 × 3 matrix per site.
fn matrix_field(lat: &Arc<Lattice>) -> Field<f64, Gauge> {
    Field::<f64, Gauge>::with_shape(lat, 18, 1, 1, 1)
}

/// The transformation of one half-sweep: on sites of parity `p`, the overrelaxed g that
/// increases Re tr(g W); the identity elsewhere.
fn sweep_transformation(
    u: &Field<f64, Gauge>,
    st: &Stencil,
    parity: &[Parity],
    p: Parity,
    omega: f64,
) -> Field<f64, Gauge> {
    let back: [_; TIME] = core::array::from_fn(|i| shifted(u, st, i, BWD));
    let mut g = matrix_field(u.lattice());
    for_sites(g.slab_mut(0), 18, |x, out| {
        if parity[x] != p {
            ColorMatrix::<f64>::identity().write_reals(out);
            return;
        }
        // W = Σ_i [U_i(x) + U_i(x − i)^†]: g(x) multiplies the first from the left and the
        // second, as g(x)^†, from the right.
        let mut w = ColorMatrix::<f64>::zero();
        for (i, b) in back.iter().enumerate() {
            w = w.add(&link(u, i, x)).add(&link(b, i, x).adj());
        }
        let g = maximise_trace(w);
        // g^ω, to first order: ω g + (1 − ω), projected back onto SU(3).
        let g = g
            .scale(omega)
            .add(&ColorMatrix::identity().scale(1.0 - omega));
        reunitarise(&g).write_reals(out);
    });
    g
}

/// An SU(3) matrix g that increases Re tr(g W): the product of the maximising SU(2)
/// rotations in the subgroups (0,1), (0,2), (1,2), each applied to W before the next is
/// found.
fn maximise_trace(mut w: ColorMatrix<f64>) -> ColorMatrix<f64> {
    let mut g = ColorMatrix::identity();
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        // The SU(2) element maximising Re tr(h w) over the (i, j) block of w is the
        // conjugate of the block's SU(2)-like part α 1 + β (…), normalised.
        let alpha = w.0[i][i] + w.0[j][j].conj();
        let beta = w.0[i][j] - w.0[j][i].conj();
        let norm = (alpha.norm_sqr() + beta.norm_sqr()).sqrt();
        let mut h = ColorMatrix::identity();
        h.0[i][i] = alpha.conj().scale(1.0 / norm);
        h.0[i][j] = (-beta).scale(1.0 / norm);
        h.0[j][i] = beta.conj().scale(1.0 / norm);
        h.0[j][j] = alpha.scale(1.0 / norm);
        w = h.mul(&w);
        g = h.mul(&g);
    }
    g
}

/// Project onto SU(3): normalise the first row, orthonormalise the second against it, and
/// take the third as the complex conjugate of their cross product.
fn reunitarise(m: &ColorMatrix<f64>) -> ColorMatrix<f64> {
    type Row = [Cplx<f64>; NC];
    let normalise = |r: Row| {
        let n = 1.0 / r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        r.map(|c| c.scale(n))
    };
    let r0 = normalise(m.0[0]);
    let mut overlap = Cplx::ZERO;
    for k in 0..NC {
        overlap += r0[k].conj() * m.0[1][k];
    }
    let r1 = normalise(core::array::from_fn(|k| m.0[1][k] - overlap * r0[k]));
    let r2 = [
        (r0[1] * r1[2] - r0[2] * r1[1]).conj(),
        (r0[2] * r1[0] - r0[0] * r1[2]).conj(),
        (r0[0] * r1[1] - r0[1] * r1[0]).conj(),
    ];
    ColorMatrix([r0, r1, r2])
}

/// (M − M^†) with its trace removed.
fn traceless_antihermitian(m: &ColorMatrix<f64>) -> ColorMatrix<f64> {
    let mut a = m.sub(&m.adj());
    let tr = a.trace().scale(1.0 / NC as f64);
    for c in 0..NC {
        a.0[c][c] = a.0[c][c] - tr;
    }
    a
}

fn norm_sqr(m: &ColorMatrix<f64>) -> f64 {
    m.0.iter().flatten().map(|c| c.norm_sqr()).sum()
}

/// A random SU(3) matrix on every site of a time slice whose violation is above `tolerance`,
/// the identity elsewhere.
fn random_transformation(
    lat: &Arc<Lattice>,
    violation: &[f64],
    tolerance: f64,
    seed: u64,
    round: u64,
) -> Field<f64, Gauge> {
    let mut g = matrix_field(lat);
    for_sites(g.slab_mut(0), 18, |x, out| {
        let gx = lat.global_coords(x);
        if violation[gx[TIME]] <= tolerance {
            ColorMatrix::<f64>::identity().write_reals(out);
            return;
        }
        let global_site =
            gx[0] + lat.global[0] * (gx[1] + lat.global[1] * (gx[2] + lat.global[2] * gx[3]));
        let mut state = seed
            ^ round.wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ (global_site as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut m = ColorMatrix::<f64>::zero();
        for row in &mut m.0 {
            for c in row.iter_mut() {
                *c = Cplx::new(uniform(&mut state), uniform(&mut state));
            }
        }
        reunitarise(&m).write_reals(out);
    });
    g
}

/// SplitMix64, mapped to (−1, 1).
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}
