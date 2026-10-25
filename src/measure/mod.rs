//! Sources and hadron correlators.
//!
//! A propagator from one source point is a spinor field with twelve vectors: vector
//! `k = colour + 3 * spin` is the solution for a unit source in that spin and colour.
//! `source` makes the sources, `meson` and `baryon` the correlators; this module holds what
//! they share: the view of a propagator and the sum over time slices with a momentum.

mod baryon;
mod meson;
mod source;

pub use baryon::proton_correlator;

pub use meson::{
    channel, meson_correlator, meson_correlator_with_momentum, pion_correlator, CHANNELS,
};
pub use source::{exponential_source, point_source};

use crate::kernel::{Cplx, GammaRepre, Real, NC, NS, SPINOR_REALS};
use crate::{reduce_sites_dyn, Field, Lattice, Spinor, Storage, C64, ND};
use std::sync::Arc;

/// Number of source components of a point-to-all propagator.
pub const NUM_SOURCES: usize = NC * NS;

/// A point-to-all propagator: a field of `NUM_SOURCES` vectors, checked when made.
#[derive(Clone, Copy)]
pub(crate) struct Propagator<'a, T: Storage> {
    field: &'a Field<T, Spinor>,
}

impl<'a, T: Storage> Propagator<'a, T> {
    pub(crate) fn new(field: &'a Field<T, Spinor>) -> Self {
        assert_eq!(
            field.copies(),
            NUM_SOURCES,
            "a propagator has {NUM_SOURCES} vectors"
        );
        Self { field }
    }

    pub(crate) fn lattice(&self) -> &'a Arc<Lattice> {
        self.field.lattice()
    }

    pub(crate) fn num_sites(&self) -> usize {
        self.field.num_sites()
    }

    /// The twelve vectors at one site, to read with `element`.
    pub(crate) fn site(&self, site: usize) -> &'a [T] {
        self.field.site(0, site)
    }
}

/// Element of the propagator at one site: sink (spin `a`, colour `i`), source vector `k`.
#[inline(always)]
pub(crate) fn element<T: Storage>(site: &[T], a: usize, i: usize, k: usize) -> C64 {
    let at = k * SPINOR_REALS + 2 * (i + NC * a);
    Cplx::new(site[at].load().to_f64(), site[at + 1].load().to_f64())
}

/// Σ over each time slice of the global lattice of e^{+i p·(x − x0)} f(site), p = 2π n / L,
/// summed over all processes: the phase of Bridge++'s `contract_at_t`. For n = 0 the values
/// of f are added unchanged.
pub(crate) fn slice_sum(
    lat: &Lattice,
    n: [i64; 3],
    origin: [usize; ND],
    f: impl Fn(usize) -> C64 + Sync,
) -> Vec<C64> {
    let nt = lat.global[3];
    let unit: [f64; 3] =
        std::array::from_fn(|mu| 2.0 * std::f64::consts::PI * n[mu] as f64 / lat.global[mu] as f64);
    let moving = n != [0; 3];
    let mut sums = reduce_sites_dyn(lat.num_sites(), 2 * nt, |site, acc| {
        let x = lat.global_coords(site);
        let mut c = f(site);
        if moving {
            let angle: f64 = (0..3)
                .map(|mu| unit[mu] * (x[mu] as f64 - origin[mu] as f64))
                .sum();
            c = c * Cplx::new(angle.cos(), angle.sin());
        }
        acc[2 * x[3]] += c.re;
        acc[2 * x[3] + 1] += c.im;
    });
    lat.comm.sum(&mut sums);
    sums.as_chunks::<2>()
        .0
        .iter()
        .map(|z| Cplx::new(z[0], z[1]))
        .collect()
}

/// ⟨ψ|γ5 ψ⟩ / ⟨ψ|ψ⟩ for a field of `ls` spinors per site (summed over all processes): with
/// `reflect`, Γ5 = γ5 R of domain-wall fermions, R reversing the fifth dimension; else γ5 on
/// every spinor. For the four-dimensional quark field of a near-zero mode, |chirality| ≈ 1 is
/// the mark of a topological zero mode, its sign that of the charge.
pub fn chirality<G: GammaRepre>(psi: &Field<f64, Spinor>, reflect: bool) -> f64 {
    let lat = psi.lattice();
    let g5 = G::GAMMA5;
    let (mut num, mut den) = (0.0, 0.0);
    for x in 0..lat.num_sites() {
        let site = psi.site(0, x);
        let ls = site.len() / 24;
        for s in 0..ls {
            let (a_at, b_at) = (24 * s, 24 * if reflect { ls - 1 - s } else { s });
            for spin in 0..4 {
                let (to, v) = (g5.col[spin], g5.val[spin]);
                for c in 0..3 {
                    let i = a_at + 2 * (c + 3 * spin);
                    let j = b_at + 2 * (c + 3 * to);
                    let (ar, ai) = (site[i], site[i + 1]);
                    let w = v.times(crate::kernel::Cplx::new(site[j], site[j + 1]));
                    // Re conj(a) w
                    num += ar * w.re + ai * w.im;
                    den += ar * ar + ai * ai;
                }
            }
        }
    }
    let mut sums = [num, den];
    lat.comm.sum(&mut sums);
    sums[0] / sums[1]
}
