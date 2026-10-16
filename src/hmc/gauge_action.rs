//! The gauge actions and their forces: the plaquette action, and the improved actions of
//! plaquettes and rectangles. The staples and the products of links are those of `gauge`.

use super::{LinkField, Links, MdAction, Mseries};
use crate::gauge::paths::Extended;
use crate::gauge::smearing::uniform;
use crate::gauge::{plaquette, weighted_staples};
use crate::kernel::ColorMatrix;
use crate::{for_sites, Boundary, Field, Gauge, Stencil, ND};
use rayon::prelude::*;

pub use crate::gauge::paths::{path_product, Step};
pub use crate::gauge::smearing::traceless_antihermitian;

/// The plaquette action S_G = β Σ_P (1 − Re tr U_P / 3), with the normalisation of Bridge++
/// (`Action_G_Plaq`): β (1 − ⟨plaquette⟩) times the number of plaquettes.
#[derive(Clone, Copy, Debug)]
pub struct PlaquetteAction {
    pub beta: f64,
}

/// The sum of the six staples of every link: for U_μ(x), Σ_(ν ≠ μ) of U_ν(x) U_μ(x+ν) U_ν(x+μ)†
/// and U_ν(x−ν)† U_μ(x−ν) U_ν(x−ν+μ), so that U_μ(x) Σ_μ(x)† is the sum of the plaquettes
/// through the link. Periodic boundary conditions.
pub fn staples(u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
    weighted_staples(u, &uniform(1.0))
}

impl PlaquetteAction {
    /// S_G (summed over all processes).
    pub fn action(&self, u: &Field<f64, Gauge>) -> f64 {
        let lat = u.lattice();
        let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
        let planes = (ND * (ND - 1) / 2) as f64;
        self.beta * (1.0 - plaquette(u, &st)) * lat.num_global_sites() as f64 * planes
    }

    /// The force on the momenta, −∂S_G/∂U as the momenta are normalised (iP = i P^a T^a,
    /// H = ½ |iP|²): F_μ(x) = −(β/3) TA[U_μ(x) Σ_μ(x)†], TA the traceless anti-Hermitian part.
    pub fn force(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        // The staples walked site by site through the links and a halo one site deep, with
        // no shifted copies of the whole field.
        let ext = Extended::new(u, 1);
        let mut f = Field::<f64, Gauge>::gauge(u.lattice());
        let k = -self.beta / 3.0;
        for mu in 0..ND {
            let paths: Vec<[Step; 3]> = (0..ND)
                .filter(|nu| *nu != mu)
                .flat_map(|nu| {
                    [
                        [(nu, true), (mu, true), (nu, false)],
                        [(nu, false), (mu, true), (nu, true)],
                    ]
                })
                .collect();
            for_sites(f.slab_mut(mu), 18, |x, out| {
                let e = ext.at(x);
                let sigma = paths
                    .iter()
                    .map(|p| ext.walk(e, p))
                    .reduce(|s, m| s.add(&m))
                    .expect("staples");
                let m = ext.link(mu, e).mul(&sigma.adj());
                traceless_antihermitian(&m).scale(k).write_reals(out);
            });
        }
        f
    }
}

/// (average, largest) of |F_μ(x)| over the links, and the standard deviation, as Bridge++
/// prints them for a force (`Field::stat`): a check of a force against its logs.
pub fn force_statistics(f: &Field<f64, Gauge>) -> (f64, f64, f64) {
    let lat = f.lattice();
    let (mut sum, mut sum2, mut max) = (0.0, 0.0, 0.0_f64);
    for mu in 0..ND {
        for x in 0..f.num_sites() {
            let s: f64 = f.site(mu, x).iter().map(|v| v * v).sum();
            sum2 += s;
            sum += s.sqrt();
            max = max.max(s.sqrt());
        }
    }
    let mut sums = [sum, sum2];
    lat.comm.sum(&mut sums);
    let n = (lat.num_global_sites() * ND) as f64;
    let ave = sums[0] / n;
    // The largest is this process's.
    (ave, max, (sums[1] / n - ave * ave).sqrt())
}

/// The improved gauge action of plaquettes and 1 × 2 rectangles,
/// S_G = β [c_plaq Σ_P (1 − Re tr U_P / 3) + c_rect Σ_R (1 − Re tr U_R / 3)], over the 6 V
/// plaquettes and the 12 V rectangles (Bridge++'s `Action_G_Rectangle`; c_plaq = 1, c_rect
/// = 0 is the plaquette action, c_plaq = 3.648, c_rect = −0.331 Iwasaki's).
#[derive(Clone, Copy, Debug)]
pub struct RectangleAction {
    pub beta: f64,
    pub c_plaq: f64,
    pub c_rect: f64,
}

/// The paths from x to x + μ̂ that close a plaquette (the first two) or a rectangle (the
/// other six) with the link U_μ(x): its staples.
fn staple_paths(mu: usize, nu: usize) -> [Vec<Step>; 8] {
    let (f, b) = (true, false);
    [
        vec![(nu, f), (mu, f), (nu, b)],
        vec![(nu, b), (mu, f), (nu, f)],
        vec![(nu, f), (mu, f), (mu, f), (nu, b), (mu, b)],
        vec![(mu, b), (nu, f), (mu, f), (mu, f), (nu, b)],
        vec![(nu, f), (nu, f), (mu, f), (nu, b), (nu, b)],
        vec![(nu, b), (mu, f), (mu, f), (nu, f), (mu, b)],
        vec![(mu, b), (nu, b), (mu, f), (mu, f), (nu, f)],
        vec![(nu, b), (nu, b), (mu, f), (nu, f), (nu, f)],
    ]
}

impl RectangleAction {
    /// S_G (summed over all processes).
    pub fn action(&self, u: &Field<f64, Gauge>) -> f64 {
        let lat = u.lattice();
        let ext = Extended::new(u, 2);
        let (f, b) = (true, false);
        let with_rect = self.c_rect != 0.0;
        let (plaq, rect) = (0..u.num_sites())
            .into_par_iter()
            .map(|x| {
                let e = ext.at(x);
                let (mut p, mut r) = (0.0, 0.0);
                for mu in 0..ND {
                    for nu in (0..ND).filter(|nu| *nu != mu) {
                        if mu < nu {
                            p += ext
                                .walk(e, &[(mu, f), (nu, f), (mu, b), (nu, b)])
                                .trace()
                                .re;
                        }
                        if with_rect {
                            r += ext
                                .walk(e, &[(mu, f), (mu, f), (nu, f), (mu, b), (mu, b), (nu, b)])
                                .trace()
                                .re;
                        }
                    }
                }
                (p, r)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let mut sums = [plaq, rect];
        lat.comm.sum(&mut sums);
        let v = lat.num_global_sites() as f64;
        self.beta
            * (self.c_plaq * (6.0 * v - sums[0] / 3.0) + self.c_rect * (12.0 * v - sums[1] / 3.0))
    }

    /// F_μ(x) = −(β/3) TA[U_μ(x) Σ_μ(x)†] with Σ the staples weighted by c_plaq and c_rect.
    pub fn force(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        let lat = u.lattice();
        let ext = Extended::new(u, 2);
        let mut f = Field::<f64, Gauge>::gauge(lat);
        let k = -self.beta / 3.0;
        for mu in 0..ND {
            let paths: Vec<(f64, Vec<Step>)> = (0..ND)
                .filter(|nu| *nu != mu)
                .flat_map(|nu| {
                    staple_paths(mu, nu)
                        .into_iter()
                        .enumerate()
                        .map(|(i, p)| (if i < 2 { self.c_plaq } else { self.c_rect }, p))
                })
                .filter(|(c, _)| *c != 0.0)
                .collect();
            for_sites(f.slab_mut(mu), 18, |x, out| {
                let e = ext.at(x);
                let mut sigma = ColorMatrix::zero();
                for (c, p) in &paths {
                    sigma = sigma.add(&ext.walk(e, p).scale(*c));
                }
                let m = ext.link(mu, e).mul(&sigma.adj());
                traceless_antihermitian(&m).scale(k).write_reals(out);
            });
        }
        f
    }
}

impl<U: LinkField> MdAction<U> for PlaquetteAction {
    fn langevin(&mut self, links: &Links<U>, _rng: &mut Mseries) -> f64 {
        links.thin.gauge_action(self.beta, 1.0, 0.0)
    }
    fn hamiltonian(&mut self, links: &Links<U>) -> f64 {
        links.thin.gauge_action(self.beta, 1.0, 0.0)
    }
    fn force(&mut self, links: &Links<U>) -> U {
        links.thin.gauge_force(self.beta, 1.0, 0.0)
    }
}

impl<U: LinkField> MdAction<U> for RectangleAction {
    fn langevin(&mut self, links: &Links<U>, _rng: &mut Mseries) -> f64 {
        links.thin.gauge_action(self.beta, self.c_plaq, self.c_rect)
    }
    fn hamiltonian(&mut self, links: &Links<U>) -> f64 {
        links.thin.gauge_action(self.beta, self.c_plaq, self.c_rect)
    }
    fn force(&mut self, links: &Links<U>) -> U {
        links.thin.gauge_force(self.beta, self.c_plaq, self.c_rect)
    }
}
