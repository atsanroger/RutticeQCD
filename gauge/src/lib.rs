//! Gauge-field observables and operations.

use ruttice_core::{
    for_sites, reduce_sites, Direction, Field, Gauge, Source, Stencil, Storage, ND,
};
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
        Source::Buffer { .. } => {
            panic!("the plaquette across processes is not implemented yet")
        }
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

/// The six planes, in the order the field strength is stored.
pub const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Clover-leaf field strength, one 3 × 3 matrix per site and plane (the outer index runs
/// over `PLANES`):
///
/// `F_mu_nu(x) = (Q - Q^†) / 8`, where Q is the sum of the four plaquettes in the mu-nu
/// plane that start and end at x, all with the same orientation. F is anti-Hermitian;
/// its trace is not removed.
///
/// The stencil must give a neighbour in every direction (no Dirichlet boundaries).
pub fn field_strength(u: &Field<f64, Gauge>, st: &Stencil) -> Field<f64, Gauge> {
    let step = |site: usize, mu: usize, dir: Direction| match st.source(site, mu, dir) {
        Source::Local { site, .. } => site,
        _ => panic!("field_strength needs a neighbour in every direction"),
    };
    let (fwd, bwd) = (Direction::Forward, Direction::Backward);
    let mut f = Field::<f64, Gauge>::with_shape(u.lattice(), 18, 1, 1, PLANES.len());
    for (plane, (mu, nu)) in PLANES.into_iter().enumerate() {
        for_sites(f.slab_mut(plane), 18, |x, out| {
            let (xp_mu, xp_nu) = (step(x, mu, fwd), step(x, nu, fwd));
            let (xm_mu, xm_nu) = (step(x, mu, bwd), step(x, nu, bwd));
            let (xm_mu_p_nu, xp_mu_m_nu) = (step(xm_mu, nu, fwd), step(xp_mu, nu, bwd));
            let xm_mu_m_nu = step(xm_mu, nu, bwd);
            let l = |d: usize, site: usize| link(u, d, site);

            // x -> x+mu -> x+mu+nu -> x+nu -> x
            let q1 = l(mu, x)
                .mul(&l(nu, xp_mu))
                .mul(&l(mu, xp_nu).adj())
                .mul(&l(nu, x).adj());
            // x -> x+nu -> x-mu+nu -> x-mu -> x
            let q2 = l(nu, x)
                .mul(&l(mu, xm_mu_p_nu).adj())
                .mul(&l(nu, xm_mu).adj())
                .mul(&l(mu, xm_mu));
            // x -> x-mu -> x-mu-nu -> x-nu -> x
            let q3 = l(mu, xm_mu)
                .adj()
                .mul(&l(nu, xm_mu_m_nu).adj())
                .mul(&l(mu, xm_mu_m_nu))
                .mul(&l(nu, xm_nu));
            // x -> x-nu -> x+mu-nu -> x+mu -> x
            let q4 = l(nu, xm_nu)
                .adj()
                .mul(&l(mu, xm_nu))
                .mul(&l(nu, xp_mu_m_nu))
                .mul(&l(mu, x).adj());

            let q = q1.add(&q2).add(&q3).add(&q4);
            q.sub(&q.adj()).scale(0.125).write_reals(out);
        });
    }
    f
}
