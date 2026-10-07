//! Sources and hadron correlators.
//!
//! A propagator from one source point is a spinor field with twelve vectors: vector
//! `k = colour + 3 * spin` is the solution for a unit source in that spin and colour.

use crate::kernel::{Cplx, Gamma, GammaRepre, Real, Unit, NC, NS, SPINOR_REALS};
use crate::{reduce_sites_dyn, Field, Lattice, Spinor, Storage, C64, ND};
use std::sync::Arc;

/// Number of source components of a point-to-all propagator.
pub const NUM_SOURCES: usize = NC * NS;

/// Twelve unit sources at one site: vector `k` has a real 1 in component `k`. `position`
/// is in global coordinates; the process that holds the site sets it.
pub fn point_source<T: Storage>(lat: &Arc<Lattice>, position: [usize; ND]) -> Field<T, Spinor> {
    for mu in 0..ND {
        assert!(
            position[mu] < lat.global[mu],
            "source position outside the lattice"
        );
    }
    let mut b = Field::<T, Spinor>::spinor(lat, NUM_SOURCES);
    if let Some(site) = lat.local_site(position) {
        let site = b.site_mut(0, site);
        for k in 0..NUM_SOURCES {
            site[k * SPINOR_REALS + 2 * k] = T::store(Real::from_f64(1.0));
        }
    }
    b
}

/// Twelve smeared sources on one time slice: vector `k` has the same real profile in
/// component `k` at every site of the time slice of `position`.
///
/// The profile at displacement d from `position` is `exp(-slope * |d|^power)`, summed over
/// all displacements with each component in (-L, L) so that the tail is folded back into
/// the periodic box, and normalised to unit 2-norm. This is not gauge invariant: results
/// depend on the gauge the configuration is in.
pub fn exponential_source<T: Storage>(
    lat: &Arc<Lattice>,
    position: [usize; ND],
    slope: f64,
    power: f64,
) -> Field<T, Spinor> {
    for mu in 0..ND {
        assert!(
            position[mu] < lat.global[mu],
            "source position outside the lattice"
        );
    }
    // The profile is computed on the whole time slice by every process.
    let l = [lat.global[0], lat.global[1], lat.global[2]];
    let mut profile = vec![0.0f64; l[0] * l[1] * l[2]];
    let range = |n: usize| -(n as i64) + 1..n as i64;
    let wrap = |x0: usize, d: i64, n: usize| (x0 as i64 + d).rem_euclid(n as i64) as usize;
    for dz in range(l[2]) {
        for dy in range(l[1]) {
            for dx in range(l[0]) {
                let r = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                let x = wrap(position[0], dx, l[0]);
                let y = wrap(position[1], dy, l[1]);
                let z = wrap(position[2], dz, l[2]);
                profile[x + l[0] * (y + l[1] * z)] += (-slope * r.powf(power)).exp();
            }
        }
    }
    let norm = profile.iter().map(|w| w * w).sum::<f64>().sqrt();

    let mut b = Field::<T, Spinor>::spinor(lat, NUM_SOURCES);
    for site in 0..lat.num_sites() {
        let [x, y, z, t] = lat.global_coords(site);
        if t != position[3] {
            continue;
        }
        let w = profile[x + l[0] * (y + l[1] * z)] / norm;
        let site = b.site_mut(0, site);
        for k in 0..NUM_SOURCES {
            site[k * SPINOR_REALS + 2 * k] = T::store(Real::from_f64(w));
        }
    }
    b
}

/// Names of the meson channels `channel` knows, in the order Bridge++ prints them.
pub const CHANNELS: [&str; 13] = [
    "PS", "V1", "V2", "V3", "A4_PS", "PS_A4", "S", "G51", "G52", "G53", "S12", "S23", "S31",
];

/// Channels `channel` knows beyond those of Bridge++. They are computed only when asked
/// for by name, and have no reference values from another program.
pub const EXTRA_CHANNELS: [&str; 1] = ["A4_A4"];

/// The (sink, source) gamma matrices of a named meson channel, to pass to
/// `meson_correlator`:
///
/// - `PS`: γ5 at both ends; `S`: the identity; `V1`..`V3`: γ_k
/// - `A4_PS`: γ4 γ5 at the sink, γ5 at the source; `PS_A4`: γ5 at the sink, γ5 γ4 at the source
/// - `A4_A4`: the axial current at both ends, γ4 γ5 at the sink and γ5 γ4 at the source,
///   which makes the correlator `Σ_x Tr[γ4 S γ4 S†]`
/// - `G51`..`G53`: γ5 γ_k; `S12`, `S23`, `S31`: i γ2 γ1, i γ3 γ2, i γ1 γ3
///
/// The composite matrices are defined as in the Bridge++ gamma-matrix set.
pub fn channel<G: GammaRepre>(name: &str) -> Option<(Gamma, Gamma)> {
    let g = G::GAMMA;
    let g5 = G::GAMMA5;
    let both = |m: Gamma| Some((m, m));
    match name {
        "PS" => both(g5),
        "V1" => both(g[0]),
        "V2" => both(g[1]),
        "V3" => both(g[2]),
        "A4_PS" => Some((g[3].mul(&g5), g5)),
        "PS_A4" => Some((g5, g5.mul(&g[3]))),
        "A4_A4" => Some((g[3].mul(&g5), g5.mul(&g[3]))),
        "S" => both(Gamma::IDENTITY),
        "G51" => both(g5.mul(&g[0])),
        "G52" => both(g5.mul(&g[1])),
        "G53" => both(g5.mul(&g[2])),
        "S12" => both(g[1].mul(&g[0]).scaled(Unit::I)),
        "S23" => both(g[2].mul(&g[1]).scaled(Unit::I)),
        "S31" => both(g[0].mul(&g[2]).scaled(Unit::I)),
        _ => None,
    }
}

/// Element of the propagator at one site: sink (spin `a`, colour `i`), source vector `k`.
#[inline(always)]
fn element<T: Storage>(site: &[T], a: usize, i: usize, k: usize) -> C64 {
    let at = k * SPINOR_REALS + 2 * (i + NC * a);
    Cplx::new(site[at].load().to_f64(), site[at + 1].load().to_f64())
}

fn check_propagator<T: Storage>(prop: &Field<T, Spinor>) {
    assert_eq!(
        prop.copies(),
        NUM_SOURCES,
        "a propagator has {NUM_SOURCES} vectors"
    );
}

/// Zero-momentum meson two-point function, one value per time slice of the global lattice:
///
/// `C(t) = Σ_x Tr[ (γ5 Γ_sink) S1(x, t) (Γ_source γ5) S2(x, t)^† ]`
///
/// with the trace over spin and colour. `prop1` and `prop2` are the propagators of the two
/// quarks from the same source; pass the same field twice for degenerate quarks. This uses
/// γ5-hermiticity to express the backward propagator through the forward one. No further
/// sign or normalisation factors are applied; the convention agrees with Bridge++ for all
/// thirteen channels its spectrum test prints (docs/COMPARISON.md).
pub fn meson_correlator<T: Storage>(
    prop1: &Field<T, Spinor>,
    prop2: &Field<T, Spinor>,
    sink: &Gamma,
    source: &Gamma,
    gamma5: &Gamma,
) -> Vec<C64> {
    check_propagator(prop1);
    check_propagator(prop2);
    assert_eq!(prop1.num_sites(), prop2.num_sites());
    let lat = prop1.lattice();
    let nt = lat.global[3];
    let (a_mat, b_mat) = (gamma5.mul(sink), source.mul(gamma5));

    let mut sums = reduce_sites_dyn(prop1.num_sites(), 2 * nt, |site, acc| {
        let t = lat.global_coords(site)[3];
        let (s1, s2) = (prop1.site(0, site), prop2.site(0, site));
        let mut c = Cplx::ZERO;
        // (A S)_{(a,i),k} = A.val[a] S_{(A.col[a], i), k}
        // (X B)_{r,(b,j)} = X_{r,(b',j)} B.val[b']  with  b = B.col[b']
        for a in 0..NS {
            for bp in 0..NS {
                let coefficient = a_mat.val[a].mul(b_mat.val[bp]);
                for i in 0..NC {
                    for j in 0..NC {
                        let lhs = element(s1, a_mat.col[a], i, j + NC * bp);
                        let rhs = element(s2, a, i, j + NC * b_mat.col[bp]);
                        c += coefficient.times(lhs * rhs.conj());
                    }
                }
            }
        }
        acc[2 * t] += c.re;
        acc[2 * t + 1] += c.im;
    });
    lat.comm.sum(&mut sums);
    sums.as_chunks::<2>()
        .0
        .iter()
        .map(|z| Cplx::new(z[0], z[1]))
        .collect()
}

/// The pseudoscalar (pion) correlator: the sum of |S|^2 over each time slice. Equal to
/// `meson_correlator` with γ5 at source and sink, without the spin algebra.
pub fn pion_correlator<T: Storage>(prop: &Field<T, Spinor>) -> Vec<f64> {
    check_propagator(prop);
    let lat = prop.lattice();
    let mut sums = reduce_sites_dyn(prop.num_sites(), lat.global[3], |site, acc| {
        let t = lat.global_coords(site)[3];
        for x in prop.site(0, site) {
            let x = x.load().to_f64();
            acc[t] += x * x;
        }
    });
    lat.comm.sum(&mut sums);
    sums
}
