//! Meson two-point functions: the channels of Bridge++ and their correlators.

use super::{element, slice_sum, Propagator};
use crate::kernel::{Cplx, Gamma, GammaRepre, Real, Unit, NC, NS};
use crate::{reduce_sites_dyn, Field, Spinor, Storage, C64, ND};

/// Names of the meson channels `channel` knows, in the order they are printed: that of
/// Bridge++, with `A4_A4`, which Bridge++ does not have, after the other axial channels.
pub const CHANNELS: [&str; 14] = [
    "PS", "V1", "V2", "V3", "A4_PS", "PS_A4", "A4_A4", "S", "G51", "G52", "G53", "S12", "S23",
    "S31",
];

/// The (sink, source) gamma matrices of a named meson channel, to pass to
/// `meson_correlator`:
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

/// Zero-momentum meson two-point function, one value per time slice of the global lattice:
pub fn meson_correlator<T: Storage>(
    prop1: &Field<T, Spinor>,
    prop2: &Field<T, Spinor>,
    sink: &Gamma,
    source: &Gamma,
    gamma5: &Gamma,
) -> Vec<C64> {
    let (s1, s2) = (Propagator::new(prop1), Propagator::new(prop2));
    assert_eq!(s1.num_sites(), s2.num_sites());
    let (a_mat, b_mat) = (gamma5.mul(sink), source.mul(gamma5));
    slice_sum(s1.lattice(), [0; 3], [0; ND], |site| {
        let (s1, s2) = (s1.site(site), s2.site(site));
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
        c
    })
}

/// The pseudoscalar (pion) correlator: the sum of |S|^2 over each time slice. Equal to
/// `meson_correlator` with γ5 at source and sink, without the spin algebra.
pub fn pion_correlator<T: Storage>(prop: &Field<T, Spinor>) -> Vec<f64> {
    let s = Propagator::new(prop);
    let lat = s.lattice();
    let mut sums = reduce_sites_dyn(s.num_sites(), lat.global[3], |site, acc| {
        let t = lat.global_coords(site)[3];
        for x in s.site(site) {
            let x = x.load().to_f64();
            acc[t] += x * x;
        }
    });
    lat.comm.sum(&mut sums);
    sums
}
