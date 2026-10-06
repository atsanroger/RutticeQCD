//! Gauge-field observables and operations.

use ruttice_core::{reduce_sites, Direction, Field, Gauge, Source, Stencil, Storage, ND};
use ruttice_kernel::{ColorMatrix, Real, NC};

/// The link U_mu(site), converted to the compute type.
#[inline]
pub fn link<T: Storage>(u: &Field<T, Gauge>, mu: usize, site: usize) -> ColorMatrix<T::Compute> {
    let s = u.site(mu, site);
    let mut v = [T::Compute::ZERO; 18];
    for k in 0..18 {
        v[k] = s[k].load();
    }
    ColorMatrix::from_reals(&v)
}

#[inline]
pub fn set_link<T: Storage>(
    u: &mut Field<T, Gauge>,
    mu: usize,
    site: usize,
    m: &ColorMatrix<T::Compute>,
) {
    let mut v = [T::Compute::ZERO; 18];
    m.write_reals(&mut v);
    let s = u.site_mut(mu, site);
    for k in 0..18 {
        s[k] = T::store(v[k]);
    }
}

/// Set every link to the identity.
pub fn set_unit<T: Storage>(u: &mut Field<T, Gauge>) {
    let id = ColorMatrix::identity();
    for mu in 0..ND {
        for site in 0..u.num_sites() {
            set_link(u, mu, site, &id);
        }
    }
}

/// Average plaquette, normalised so that a unit gauge field gives 1:
/// the mean over sites and the six planes of Re tr[U_mu(x) U_nu(x+mu) U_mu(x+nu)^† U_nu(x)^†] / Nc.
///
/// Planes with a `Zero` neighbour (Dirichlet boundary) are skipped and still counted in
/// the normalisation.
pub fn plaquette<T: Storage>(u: &Field<T, Gauge>, st: &Stencil) -> f64 {
    let num_sites = u.num_sites();
    assert_eq!(st.num_sites(), num_sites);
    let forward = |site: usize, mu: usize| match st.source(site, mu, Direction::Forward) {
        Source::Local { site, .. } => Some(site),
        Source::Zero => None,
        Source::Buffer { .. } => unreachable!("no communication buffers on a single node"),
    };
    let sum = reduce_sites(num_sites, |x| {
        let mut s = 0.0;
        for mu in 0..ND {
            for nu in mu + 1..ND {
                let (Some(x_mu), Some(x_nu)) = (forward(x, mu), forward(x, nu)) else {
                    continue;
                };
                let loop_ = link(u, mu, x)
                    .mul(&link(u, nu, x_mu))
                    .mul(&link(u, mu, x_nu).adj())
                    .mul(&link(u, nu, x).adj());
                s += loop_.trace().re.to_f64();
            }
        }
        s
    });
    let planes = ND * (ND - 1) / 2;
    sum / (num_sites * planes * NC) as f64
}
