//! Gauge-field observables and operations: plaquette, field strength, gauge fixing,
//! smearing (`smear`) and products of links along paths (`paths`).

use crate::kernel::{ColorMatrix, Number, Real, NC};
use crate::{
    for_sites, reduce_sites, shift, Boundary, Direction, Field, Gauge, Stencil, Storage, ND,
};

mod fixing;
pub mod flow;
pub mod paths;
mod random;
pub mod smearing;

pub use fixing::{
    coulomb_violation, fix_coulomb, gauge_transform, CoulombParams, GaugeFixError, GaugeFixReport,
};
pub use random::{random_gauge, random_su3};
pub use smearing::{weighted_staples, GaugeSmearing, Smear};

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
pub fn plaquette<T: Storage>(u: &Field<T, Gauge>, st: &Stencil) -> f64 {
    let lat = u.lattice();
    let num_sites = u.num_sites();
    assert_eq!(st.num_sites(), num_sites);
    // up[mu] holds U(x + mu); zero where the stencil has no neighbour.
    let up: [Field<T, Gauge>; ND] = core::array::from_fn(|mu| shifted(u, st, mu, FWD));
    let sum = reduce_sites(num_sites, |x| {
        let mut s = 0.0;
        for mu in 0..ND {
            for nu in mu + 1..ND {
                let loop_ = link(u, mu, x)
                    .mul(&link(&up[mu], nu, x))
                    .mul(&link(&up[nu], mu, x).adj())
                    .mul(&link(u, nu, x).adj());
                s += loop_.trace().re.to_f64();
            }
        }
        s
    });
    let mut sum = [sum];
    lat.comm.sum(&mut sum);
    let planes = ND * (ND - 1) / 2;
    sum[0] / (lat.num_global_sites() * planes * NC) as f64
}

const FWD: Direction = Direction::Forward;
const BWD: Direction = Direction::Backward;

/// A copy of a field of colour matrices moved by one site: the result at x is the field at
/// x ± mu.
fn shifted<T: Storage>(
    u: &Field<T, Gauge>,
    st: &Stencil,
    mu: usize,
    dir: Direction,
) -> Field<T, Gauge> {
    let mut out = Field::<T, Gauge>::with_shape(
        u.lattice(),
        u.internal_dof(),
        u.ls(),
        u.num_vectors(),
        u.outer(),
    );
    shift(&mut out, u, st, mu, dir);
    out
}

/// The six planes, in the order the field strength is stored.
pub const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Clover-leaf field strength, one 3 × 3 matrix per site and plane (the outer index runs
/// over `PLANES`):
///
/// The stencil must give a neighbour in every direction (no Dirichlet boundaries).
pub fn field_strength(u: &Field<f64, Gauge>, st: &Stencil) -> Field<f64, Gauge> {
    assert!(
        (0..ND).all(|mu| st.boundary(mu) != Boundary::Dirichlet),
        "field_strength needs a neighbour in every direction"
    );
    // The gauge field at x + d and at x - d.
    let plus: [_; ND] = core::array::from_fn(|d| shifted(u, st, d, FWD));
    let minus: [_; ND] = core::array::from_fn(|d| shifted(u, st, d, BWD));
    let mut f = Field::<f64, Gauge>::with_shape(u.lattice(), 18, 1, 1, PLANES.len());
    for (plane, (mu, nu)) in PLANES.into_iter().enumerate() {
        let (p_mu, p_nu, m_mu, m_nu) = (&plus[mu], &plus[nu], &minus[mu], &minus[nu]);
        // The gauge field at x - mu + nu, x + mu - nu and x - mu - nu.
        let m_mu_p_nu = &shifted(m_mu, st, nu, FWD);
        let p_mu_m_nu = &shifted(p_mu, st, nu, BWD);
        let m_mu_m_nu = &shifted(m_mu, st, nu, BWD);
        for_sites(f.slab_mut(plane), 18, |x, out| {
            // The link in direction d of a shifted copy, at this site.
            let at = |field: &Field<f64, Gauge>, d: usize| link(field, d, x);

            // x -> x+mu -> x+mu+nu -> x+nu -> x
            let q1 = at(u, mu)
                .mul(&at(p_mu, nu))
                .mul(&at(p_nu, mu).adj())
                .mul(&at(u, nu).adj());
            // x -> x+nu -> x-mu+nu -> x-mu -> x
            let q2 = at(u, nu)
                .mul(&at(m_mu_p_nu, mu).adj())
                .mul(&at(m_mu, nu).adj())
                .mul(&at(m_mu, mu));
            // x -> x-mu -> x-mu-nu -> x-nu -> x
            let q3 = at(m_mu, mu)
                .adj()
                .mul(&at(m_mu_m_nu, nu).adj())
                .mul(&at(m_mu_m_nu, mu))
                .mul(&at(m_nu, nu));
            // x -> x-nu -> x+mu-nu -> x+mu -> x
            let q4 = at(m_nu, nu)
                .adj()
                .mul(&at(m_nu, mu))
                .mul(&at(p_mu_m_nu, nu))
                .mul(&at(u, mu).adj());

            let q = q1.add(&q2).add(&q3).add(&q4);
            q.sub(&q.adj()).scale(0.125).write_reals(out);
        });
    }
    f
}
