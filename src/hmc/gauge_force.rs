//! The Wilson plaquette action of the gauge field and its force.

use crate::gauge::{link, plaquette};
use crate::kernel::{ColorMatrix, Cplx};
use crate::{for_sites, shift, Boundary, Direction, Field, Gauge, Stencil, ND};

/// The plaquette action S_G = β Σ_P (1 − Re tr U_P / 3), with the normalisation of Bridge++
/// (`Action_G_Plaq`): β (1 − ⟨plaquette⟩) times the number of plaquettes.
#[derive(Clone, Copy, Debug)]
pub struct PlaquetteAction {
    pub beta: f64,
}

/// The traceless anti-Hermitian part of a matrix, (M − M†)/2 − tr(M − M†)/(2 N_c).
pub fn traceless_antihermitian(m: &ColorMatrix<f64>) -> ColorMatrix<f64> {
    let mut a = m.sub(&m.adj()).scale(0.5);
    let shift = a.trace().im / 3.0;
    for c in 0..3 {
        a.0[c][c] = Cplx::new(0.0, a.0[c][c].im - shift);
    }
    a
}

/// The sum of the six staples of every link: for U_μ(x), Σ_(ν ≠ μ) of U_ν(x) U_μ(x+ν) U_ν(x+μ)†
/// and U_ν(x−ν)† U_μ(x−ν) U_ν(x−ν+μ), so that U_μ(x) Σ_μ(x)† is the sum of the plaquettes
/// through the link. Periodic boundary conditions.
pub fn staples(u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
    let lat = u.lattice();
    let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
    let shifted = |f: &Field<f64, Gauge>, mu: usize, dir: Direction| {
        let mut out = Field::<f64, Gauge>::gauge(f.lattice());
        shift(&mut out, f, &st, mu, dir);
        out
    };
    // u at x + mu for every mu.
    let up: [Field<f64, Gauge>; ND] = core::array::from_fn(|mu| shifted(u, mu, Direction::Forward));
    let mut sum = Field::<f64, Gauge>::gauge(lat);
    for mu in 0..ND {
        for nu in (0..ND).filter(|nu| *nu != mu) {
            // The lower staple at x − ν, built at x and shifted back: U_ν(x)† U_μ(x) U_ν(x+μ).
            let mut lower = Field::<f64, Gauge>::gauge(lat);
            for_sites(lower.slab_mut(mu), 18, |x, out| {
                link(u, nu, x)
                    .adj()
                    .mul(&link(u, mu, x).mul(&link(&up[mu], nu, x)))
                    .write_reals(out);
            });
            let lower = shifted(&lower, nu, Direction::Backward);
            let (up_nu, up_mu) = (&up[nu], &up[mu]);
            for_sites(sum.slab_mut(mu), 18, |x, out| {
                let upper = link(u, nu, x).mul(&link(up_nu, mu, x).mul(&link(up_mu, nu, x).adj()));
                let total = ColorMatrix::from_reals(out)
                    .add(&upper)
                    .add(&link(&lower, mu, x));
                total.write_reals(out);
            });
        }
    }
    sum
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
