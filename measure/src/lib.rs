//! Sources and hadron correlators (FLOW.md §B).
//!
//! A propagator from one source point is a spinor field with twelve vectors: vector
//! `k = colour + 3 * spin` is the solution for a unit source in that spin and colour.

use ruttice_core::{reduce_sites_dyn, Field, Lattice, Spinor, Storage, C64, ND};
use ruttice_kernel::{Cplx, Gamma, Real, NC, NS, SPINOR_REALS};
use std::sync::Arc;

/// Number of source components of a point-to-all propagator.
pub const NUM_SOURCES: usize = NC * NS;

/// Twelve unit sources at one site: vector `k` has a real 1 in component `k`.
pub fn point_source<T: Storage>(lat: &Arc<Lattice>, position: [usize; ND]) -> Field<T, Spinor> {
    for mu in 0..ND {
        assert!(
            position[mu] < lat.local[mu],
            "source position outside the lattice"
        );
    }
    let mut b = Field::<T, Spinor>::spinor(lat, NUM_SOURCES);
    let site = b.site_mut(0, lat.site(position));
    for k in 0..NUM_SOURCES {
        site[k * SPINOR_REALS + 2 * k] = T::store(Real::from_f64(1.0));
    }
    b
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
    let lat = prop.lattice();
    assert_eq!(
        lat.local, lat.global,
        "correlators on more than one process are not implemented"
    );
}

/// Zero-momentum meson two-point function, one value per time slice:
///
/// `C(t) = Σ_x Tr[ (γ5 Γ_sink) S(x, t) (Γ_source γ5) S(x, t)^† ]`
///
/// with the trace over spin and colour. This uses γ5-hermiticity of the propagator to
/// express the backward propagator through the forward one. No further sign or
/// normalisation factors are applied.
pub fn meson_correlator<T: Storage>(
    prop: &Field<T, Spinor>,
    sink: &Gamma,
    source: &Gamma,
    gamma5: &Gamma,
) -> Vec<C64> {
    check_propagator(prop);
    let lat = prop.lattice();
    let nt = lat.local[3];
    let (a_mat, b_mat) = (gamma5.mul(sink), source.mul(gamma5));

    let sums = reduce_sites_dyn(prop.num_sites(), 2 * nt, |site, acc| {
        let t = lat.coords(site)[3];
        let s = prop.site(0, site);
        let mut c = Cplx::ZERO;
        // (A S)_{(a,i),k} = A.val[a] S_{(A.col[a], i), k}
        // (X B)_{r,(b,j)} = X_{r,(b',j)} B.val[b']  with  b = B.col[b']
        for a in 0..NS {
            for bp in 0..NS {
                let coefficient = a_mat.val[a].mul(b_mat.val[bp]);
                for i in 0..NC {
                    for j in 0..NC {
                        let lhs = element(s, a_mat.col[a], i, j + NC * bp);
                        let rhs = element(s, a, i, j + NC * b_mat.col[bp]);
                        c += coefficient.times(lhs * rhs.conj());
                    }
                }
            }
        }
        acc[2 * t] += c.re;
        acc[2 * t + 1] += c.im;
    });
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
    reduce_sites_dyn(prop.num_sites(), lat.local[3], |site, acc| {
        let t = lat.coords(site)[3];
        for x in prop.site(0, site) {
            let x = x.load().to_f64();
            acc[t] += x * x;
        }
    })
}
