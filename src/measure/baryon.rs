//! Baryon two-point functions: the proton of Bridge++ (`Corr2pt_4spinor::proton_correlator`).
//!
//! The operator is N = ε_abc (u_a^T Cγ5 d_b) u_c with C = γ4 γ2. Unlike a meson, the
//! contraction takes no complex conjugate: all three quarks propagate from the source to the
//! sink, so no γ5-hermiticity is needed.

use super::{element, slice_sum, Propagator};
use crate::kernel::{Cplx, Gamma, GammaRepre, NC, NS};
use crate::{Field, Spinor, Storage, C64, ND};

/// The permutations of the colours (a, b, c) with the sign of ε_abc, in Bridge++'s order.
pub(crate) const EPSILON: [([usize; 3], f64); 6] = [
    ([0, 1, 2], 1.0),
    ([1, 2, 0], 1.0),
    ([2, 0, 1], 1.0),
    ([2, 1, 0], -1.0),
    ([1, 0, 2], -1.0),
    ([0, 2, 1], -1.0),
];

/// Cγ5 with C = γ4 γ2, the charge conjugation matrix of Bridge++ in both representations.
pub(crate) fn c_gamma5<G: GammaRepre>() -> Gamma {
    G::GAMMA[3].mul(&G::GAMMA[1]).mul(&G::GAMMA5)
}

/// The diquark-quark contraction at one site of three propagator columns `k1`, `k2`, `k3`
/// (Bridge++'s baryon `contract_at_t`): Σ_d1 (Cγ5)_{d1 d2} ε_abc v1_{d1 a} v2_{d2 b} v3_{d3 c},
/// d2 the column of row d1 of Cγ5, v_{s c} the element of spin s and colour c.
#[inline(always)]
fn diquark_quark<T: Storage>(
    (s1, k1): (&[T], usize),
    (s2, k2): (&[T], usize),
    (s3, k3): (&[T], usize),
    cg5: &Gamma,
    d3: usize,
) -> C64 {
    let mut corr = Cplx::ZERO;
    for d1 in 0..NS {
        let d2 = cg5.col[d1];
        let mut c = Cplx::ZERO;
        for ([a, b, cc], sign) in EPSILON {
            let pair = element(s1, d1, a, k1) * element(s2, d2, b, k2);
            c += (pair * element(s3, d3, cc, k3)).scale(sign);
        }
        corr += cg5.val[d1].times(c);
    }
    corr
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
        for alpha in 0..NS {
            // The source spin of the third quark.
            let alpha_p = projection.col[alpha];
            for a1p in 0..NS {
                let a2p = cg5.col[a1p];
                let spins = projection.val[alpha].mul(cg5.val[a1p]);
                for ([c1p, c2p, c3p], sign) in EPSILON {
                    let (k1, k2, k3) = (c1p + NC * a1p, c2p + NC * a2p, c3p + NC * alpha_p);
                    // u enters twice: the two ways to pair the u at the sink with those at the source.
                    let first = diquark_quark((su, k1), (sd, k2), (su, k3), &cg5, alpha);
                    let second = diquark_quark((su, k3), (sd, k2), (su, k1), &cg5, alpha);
                    sum += spins.times(first - second).scale(sign);
                }
            }
        }
        sum
    })
}
