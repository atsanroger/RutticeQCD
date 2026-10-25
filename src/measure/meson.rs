//! Meson two-point functions: the channels of Bridge++ and their correlators.

use super::three_point::{Sink, SiteMatrix};
use super::{element, slice_sum, Propagator};
use crate::kernel::{Cplx, Gamma, GammaRepre, Real, Unit, NC, NS};
use crate::{reduce_sites_dyn, Field, Spinor, Storage, C64, ND};
use std::marker::PhantomData;

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
    meson_correlator_with_momentum(prop1, prop2, sink, source, gamma5, [0; 3], [0; ND])
}

/// `meson_correlator` with the sink projected on momentum p = 2π n / L: each site weighted by
/// e^{+i p·(x − x0)}, x0 the source position (Bridge++'s `meson_momentum_correlator`; the
/// state annihilated at the sink has momentum −p).
pub fn meson_correlator_with_momentum<T: Storage>(
    prop1: &Field<T, Spinor>,
    prop2: &Field<T, Spinor>,
    sink: &Gamma,
    source: &Gamma,
    gamma5: &Gamma,
    n: [i64; 3],
    source_position: [usize; ND],
) -> Vec<C64> {
    let (s1, s2) = (Propagator::new(prop1), Propagator::new(prop2));
    assert_eq!(s1.num_sites(), s2.num_sites());
    let (a_mat, b_mat) = (gamma5.mul(sink), source.mul(gamma5));
    slice_sum(s1.lattice(), n, source_position, |site| {
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

/// A meson at the sink, `sink` and `source` as in `meson_correlator`, the current on the line
/// of the first propagator. The propagators are `[prop1, prop2]` of `meson_correlator`.
pub struct MesonSink<G: GammaRepre> {
    pub sink: Gamma,
    pub source: Gamma,
    pub gammas: PhantomData<G>,
}

impl<G: GammaRepre> Sink for MesonSink<G> {
    fn two_point(
        &self,
        props: [&Field<f64, Spinor>; 2],
        n: [i64; 3],
        source_position: [usize; ND],
    ) -> Vec<C64> {
        let g5 = G::GAMMA5;
        meson_correlator_with_momentum(
            props[0],
            props[1],
            &self.sink,
            &self.source,
            &g5,
            n,
            source_position,
        )
    }

    /// c = Tr[A S1 B S2†] with A = γ5 Γ_sink, B = Γ_source γ5, so ∂c/∂(S1)_{ik} = (B S2† A)_{ki}:
    /// G_{(s_i, c_i), (s_k, c_k)} = B.val[s_k] Σ_{s': A.col[s'] = s_i} A.val[s'] conj(S2_{(s', c_i), (B.col[s_k], c_k)}).
    fn derivative(&self, sites: [&[f64]; 2], g: &mut SiteMatrix) {
        let g5 = G::GAMMA5;
        let (a, b) = (g5.mul(&self.sink), self.source.mul(&g5));
        let s2 = sites[1];
        for sp in 0..NS {
            let si = a.col[sp];
            for ci in 0..NC {
                for sk in 0..NS {
                    for ck in 0..NC {
                        let v = element(s2, sp, ci, ck + NC * b.col[sk]).conj();
                        g[ci + NC * si][ck + NC * sk] += b.val[sk].times(a.val[sp].times(v));
                    }
                }
            }
        }
    }
}
