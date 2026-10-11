//! The Wilson plaquette action of the gauge field and its force.

use crate::gauge::smear::uniform;
use crate::gauge::{link, plaquette, weighted_staples};
use crate::kernel::ColorMatrix;
use crate::{for_sites, shift, Boundary, Direction, Field, Gauge, Stencil, ND};

/// The plaquette action S_G = β Σ_P (1 − Re tr U_P / 3), with the normalisation of Bridge++
/// (`Action_G_Plaq`): β (1 − ⟨plaquette⟩) times the number of plaquettes.
#[derive(Clone, Copy, Debug)]
pub struct PlaquetteAction {
    pub beta: f64,
}

pub use crate::gauge::smear::traceless_antihermitian;

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
        let sigma = staples(u);
        let mut f = Field::<f64, Gauge>::gauge(u.lattice());
        let k = -self.beta / 3.0;
        for mu in 0..ND {
            for_sites(f.slab_mut(mu), 18, |x, out| {
                let m = link(u, mu, x).mul(&link(&sigma, mu, x).adj());
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

/// A step of a path of links: direction μ, forward (to x + μ̂, through U_μ(x)) or backward
/// (to x − μ̂, through U_μ(x − μ̂)†).
pub type Step = (usize, bool);

/// The product of the links along the path `steps` from every site, as one colour matrix per
/// site (in the first slab of a gauge field). Periodic boundary conditions.
pub fn path_product(u: &Field<f64, Gauge>, steps: &[Step]) -> Field<f64, Gauge> {
    let lat = u.lattice();
    let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
    // U at x − μ̂, for the backward steps.
    let downs: [Field<f64, Gauge>; ND] = core::array::from_fn(|mu| {
        let mut d = Field::<f64, Gauge>::gauge(lat);
        shift(&mut d, u, &st, mu, Direction::Backward);
        d
    });
    let at = |x: usize, (mu, forward): Step| {
        if forward {
            link(u, mu, x)
        } else {
            link(&downs[mu], mu, x).adj()
        }
    };
    let (&last, rest) = steps.split_last().expect("a path has steps");
    let mut p = Field::<f64, Gauge>::gauge(lat);
    for_sites(p.slab_mut(0), 18, |x, out| at(x, last).write_reals(out));
    for &(mu, forward) in rest.iter().rev() {
        // The rest of the path, from the site this step leads to.
        let mut moved = Field::<f64, Gauge>::gauge(lat);
        let dir = if forward {
            Direction::Forward
        } else {
            Direction::Backward
        };
        shift(&mut moved, &p, &st, mu, dir);
        for_sites(p.slab_mut(0), 18, |x, out| {
            at(x, (mu, forward))
                .mul(&link(&moved, 0, x))
                .write_reals(out);
        });
    }
    p
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
        let (mut plaq, mut rect) = (0.0, 0.0);
        let (f, b) = (true, false);
        for mu in 0..ND {
            for nu in (0..ND).filter(|nu| *nu != mu) {
                if mu < nu {
                    let p = path_product(u, &[(mu, f), (nu, f), (mu, b), (nu, b)]);
                    plaq += (0..u.num_sites())
                        .map(|x| link(&p, 0, x).trace().re)
                        .sum::<f64>();
                }
                if self.c_rect != 0.0 {
                    let r =
                        path_product(u, &[(mu, f), (mu, f), (nu, f), (mu, b), (mu, b), (nu, b)]);
                    rect += (0..u.num_sites())
                        .map(|x| link(&r, 0, x).trace().re)
                        .sum::<f64>();
                }
            }
        }
        let mut sums = [plaq, rect];
        lat.comm.sum(&mut sums);
        let v = lat.num_global_sites() as f64;
        self.beta
            * (self.c_plaq * (6.0 * v - sums[0] / 3.0) + self.c_rect * (12.0 * v - sums[1] / 3.0))
    }

    /// F_μ(x) = −(β/3) TA[U_μ(x) Σ_μ(x)†] with Σ the staples weighted by c_plaq and c_rect.
    pub fn force(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        let lat = u.lattice();
        let mut f = Field::<f64, Gauge>::gauge(lat);
        let k = -self.beta / 3.0;
        for mu in 0..ND {
            let mut sigma = Field::<f64, Gauge>::gauge(lat);
            for nu in (0..ND).filter(|nu| *nu != mu) {
                for (i, path) in staple_paths(mu, nu).iter().enumerate() {
                    let c = if i < 2 { self.c_plaq } else { self.c_rect };
                    if c == 0.0 {
                        continue;
                    }
                    let p = path_product(u, path);
                    for_sites(sigma.slab_mut(0), 18, |x, out| {
                        ColorMatrix::from_reals(out)
                            .add(&link(&p, 0, x).scale(c))
                            .write_reals(out);
                    });
                }
            }
            for_sites(f.slab_mut(mu), 18, |x, out| {
                let m = link(u, mu, x).mul(&link(&sigma, 0, x).adj());
                traceless_antihermitian(&m).scale(k).write_reals(out);
            });
        }
        f
    }
}
