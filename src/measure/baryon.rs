//! Baryon two-point functions: the proton of Bridge++ (`Corr2pt_4spinor::proton_correlator`),
//! and its sink for three-point functions.
//!
//! The operator is N = ε_abc (u_a^T Cγ5 d_b) u_c with C = γ4 γ2. Unlike a meson, the
//! contraction takes no complex conjugate: all three quarks propagate from the source to the
//! sink, so no γ5-hermiticity is needed.

use super::three_point::{Flavour, Sink, SiteMatrix};
use super::{element, slice_sum, Propagator};
use crate::kernel::{Cplx, Gamma, GammaRepre, Unit, NC, NS};
use crate::{Field, Spinor, Storage, C64, ND};
use std::marker::PhantomData;

/// The permutations of the colours (a, b, c) with the sign of ε_abc, in Bridge++'s order.
const EPSILON: [([usize; 3], f64); 6] = [
    ([0, 1, 2], 1.0),
    ([1, 2, 0], 1.0),
    ([2, 0, 1], 1.0),
    ([2, 1, 0], -1.0),
    ([1, 0, 2], -1.0),
    ([0, 2, 1], -1.0),
];

/// Cγ5 with C = γ4 γ2, the charge conjugation matrix of Bridge++ in both representations.
fn c_gamma5<G: GammaRepre>() -> Gamma {
    G::GAMMA[3].mul(&G::GAMMA[1]).mul(&G::GAMMA5)
}

/// An element of a propagator: sink index `colour + 3 spin`, source vector.
type At = (usize, usize);

/// Every term of the proton contraction at one site, in Bridge++'s order: the term is
/// `unit * sign * u[first] d[second] u[third]`. The loops are those of Bridge++'s
/// `proton_correlator` and its baryon `contract_at_t`: Γ_{αα'} (Cγ5)_{α1' α2'} ε_{a'b'c'} at the
/// source, Σ_d1 (Cγ5)_{d1 d2} ε_abc at the sink, u entering twice (the second pairing
/// subtracted).
#[inline(always)]
fn proton_terms(projection: &Gamma, cg5: &Gamma, mut term: impl FnMut(Unit, f64, At, At, At)) {
    for alpha in 0..NS {
        // The source spin of the third quark; its sink spin is alpha.
        let alpha_p = projection.col[alpha];
        for a1p in 0..NS {
            let a2p = cg5.col[a1p];
            let spins = projection.val[alpha].mul(cg5.val[a1p]);
            for ([c1p, c2p, c3p], outer) in EPSILON {
                let (k1, k2, k3) = (c1p + NC * a1p, c2p + NC * a2p, c3p + NC * alpha_p);
                for d1 in 0..NS {
                    let d2 = cg5.col[d1];
                    let unit = spins.mul(cg5.val[d1]);
                    for ([a, b, c], inner) in EPSILON {
                        let (i1, i2, i3) = (a + NC * d1, b + NC * d2, c + NC * alpha);
                        let sign = outer * inner;
                        term(unit, sign, (i1, k1), (i2, k2), (i3, k3));
                        term(unit, -sign, (i1, k3), (i2, k2), (i3, k1));
                    }
                }
            }
        }
    }
}

/// An element of a propagator site by sink index `colour + 3 spin` and source vector.
#[inline(always)]
fn at<T: Storage>(site: &[T], (i, k): At) -> C64 {
    element(site, i / NC, i % NC, k)
}

/// The proton two-point function C(t) = Σ_x e^{+i p·(x − x0)} Σ_{αα'} Γ_{αα'} ⟨N_α(x) N̄_α'(x0)⟩,
/// u and d from the propagators `prop_u`, `prop_d`, p = 2π n / L (Bridge++'s
/// `proton_correlator` for n = 0). Bridge++ prints Γ = 1, Γ = γ4 and their mean, the
/// positive-parity projection (1 + γ4)/2.
pub fn proton_correlator<G: GammaRepre, T: Storage>(
    prop_u: &Field<T, Spinor>,
    prop_d: &Field<T, Spinor>,
    projection: &Gamma,
    n: [i64; 3],
    source_position: [usize; ND],
) -> Vec<C64> {
    let (u, d) = (Propagator::new(prop_u), Propagator::new(prop_d));
    assert_eq!(u.num_sites(), d.num_sites());
    let cg5 = c_gamma5::<G>();
    slice_sum(u.lattice(), n, source_position, |site| {
        let (su, sd) = (u.site(site), d.site(site));
        let mut sum = Cplx::ZERO;
        proton_terms(projection, &cg5, |unit, sign, x, y, z| {
            sum += unit.times(at(su, x) * at(sd, y) * at(su, z)).scale(sign);
        });
        sum
    })
}

/// The proton at the sink, projected by Σ_j w_j Γ_j (for example (1 + γ4)/2 as
/// `[(1, 0.5), (γ4, 0.5)]`), with the current on the u or the d quark. The propagators are
/// `[u, d]`.
pub struct ProtonSink<G: GammaRepre> {
    pub projection: Vec<(Gamma, C64)>,
    pub current: Flavour,
    pub gammas: PhantomData<G>,
}

impl<G: GammaRepre> Sink for ProtonSink<G> {
    fn two_point(
        &self,
        props: [&Field<f64, Spinor>; 2],
        n: [i64; 3],
        source_position: [usize; ND],
    ) -> Vec<C64> {
        let mut sum = vec![Cplx::ZERO; props[0].lattice().global[3]];
        for (g, w) in &self.projection {
            let c = proton_correlator::<G, f64>(props[0], props[1], g, n, source_position);
            for (s, c) in sum.iter_mut().zip(c) {
                *s += *w * c;
            }
        }
        sum
    }

    fn derivative(&self, sites: [&[f64]; 2], g: &mut SiteMatrix) {
        let (su, sd) = (sites[0], sites[1]);
        let cg5 = c_gamma5::<G>();
        for (projection, w) in &self.projection {
            proton_terms(projection, &cg5, |unit, sign, x, y, z| {
                let c = *w * unit.times(Cplx::new(sign, 0.0));
                match self.current {
                    // u enters twice: the derivative is the sum of the two places.
                    Flavour::U => {
                        let d = at(sd, y);
                        g[x.0][x.1] += c * d * at(su, z);
                        g[z.0][z.1] += c * at(su, x) * d;
                    }
                    Flavour::D => g[y.0][y.1] += c * at(su, x) * at(su, z),
                }
            });
        }
    }
}
