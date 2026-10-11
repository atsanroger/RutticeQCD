//! Gauge smearing: every link replaced by a smoother one built from its staples, as the links
//! a fermion operator is made of (stout links in HMC) or for gauge observables.
//!
//! This is the smearing of the gauge field. The smearing of quark sources and sinks
//! (operator smearing, e.g. exponential or Gaussian wave functions) is a different thing and
//! lives with the sources (`measure`); the parameter files keep the two apart
//! (`gauge_smearing` and the source type).
//!
//! Two kinds, with the conventions of Bridge++ (`Smear_APE` with `Projection_Stout_SU3` or
//! `Projection_Maximum_SU_N`) so that its runs can be repeated:
//!
//! - stout (Morningstar–Peardon, hep-lat/0311018): V_μ = exp(iQ_μ) U_μ, iQ_μ the traceless
//!   anti-Hermitian part of C_μ U_μ†, C_μ = Σ_(ν≠μ) ρ_μν (staples in the μν plane);
//!   differentiable, so HMC can use it (`Stout::chain`);
//! - APE: V_μ = the SU(3) matrix that maximises Re tr[V† ((1 − ρ_μμ) U_μ + C_μ)], found by
//!   SU(2) subgroups (Cabibbo–Marinari) until the change is below a tolerance.
//!
//! Written from the formulas of the paper and the structure of the Bridge++ code.

use super::link;
use super::paths::Neighbors;
use crate::kernel::{ColorMatrix, Cplx};
use crate::{for_sites, shift, Boundary, Direction, Field, Gauge, Stencil, ND};
use rayon::prelude::*;

/// ρ_μν: the weight of the staples in the μν plane in the new U_μ (ρ_μμ is used by APE only).
pub type Weights = [[f64; ND]; ND];

/// The same weight in every plane.
pub fn uniform(rho: f64) -> Weights {
    [[rho; ND]; ND]
}

/// The weight in the spatial planes only: the time links stay as they are and the spatial
/// links take spatial staples only (to smear the links a smeared source is built from).
pub fn spatial(rho: f64) -> Weights {
    core::array::from_fn(|mu| {
        core::array::from_fn(|nu| if mu < ND - 1 && nu < ND - 1 { rho } else { 0.0 })
    })
}

/// One step of gauge smearing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smear {
    Stout(Stout),
    Ape(Ape),
}

/// Stout smearing with weights ρ_μν.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stout {
    pub rho: Weights,
}

/// APE smearing with weights ρ_μν, projected back to SU(3) by maximising the trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ape {
    pub rho: Weights,
    /// The projection stops when 1 − ⟨Re tr ΔG⟩/3 over all sites, ΔG the change of a sweep
    /// over the subgroups, is below this ...
    pub tolerance: f64,
    /// ... and fails after this many sweeps.
    pub max_iterations: usize,
}

/// Gauge smearing: `steps` times the same step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaugeSmearing {
    pub smear: Smear,
    pub steps: usize,
}

impl GaugeSmearing {
    /// The smeared fields after each step, the last the one to use. Empty for no steps.
    pub fn levels(&self, u: &Field<f64, Gauge>) -> Vec<Field<f64, Gauge>> {
        let mut levels: Vec<Field<f64, Gauge>> = Vec::with_capacity(self.steps);
        for _ in 0..self.steps {
            let next = self.smear.apply(levels.last().unwrap_or(u));
            levels.push(next);
        }
        levels
    }

    /// The smeared field.
    pub fn apply(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        self.levels(u).pop().unwrap_or_else(|| copy(u))
    }

    /// The chain rule through all the steps: given Σ' with dS = Re tr Σ' δV for the smeared
    /// links V (one matrix per link), the Σ of the links `u` (`levels` = `self.levels(u)`).
    /// Stout smearing only.
    pub fn chain(
        &self,
        sigma: Field<f64, Gauge>,
        u: &Field<f64, Gauge>,
        levels: &[Field<f64, Gauge>],
    ) -> Field<f64, Gauge> {
        let Smear::Stout(stout) = self.smear else {
            panic!("only stout smearing is differentiable");
        };
        let mut sigma = sigma;
        for k in (0..self.steps).rev() {
            let below = if k == 0 { u } else { &levels[k - 1] };
            sigma = stout.chain(&sigma, below);
        }
        sigma
    }
}

impl Smear {
    pub fn apply(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        match self {
            Smear::Stout(s) => s.apply(u),
            Smear::Ape(a) => a.apply(u),
        }
    }
}

fn copy(u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
    let mut v = Field::<f64, Gauge>::gauge(u.lattice());
    v.as_mut_slice().copy_from_slice(u.as_slice());
    v
}

fn shifted(f: &Field<f64, Gauge>, st: &Stencil, mu: usize, dir: Direction) -> Field<f64, Gauge> {
    let mut out = Field::<f64, Gauge>::gauge(f.lattice());
    shift(&mut out, f, st, mu, dir);
    out
}

/// C_μ = Σ_(ν≠μ) ρ_μν [U_ν(x) U_μ(x+ν) U_ν(x+μ)† + U_ν(x−ν)† U_μ(x−ν) U_ν(x−ν+μ)], the
/// weighted sum of the staples of every link; with all weights 1, U_μ C_μ† is the sum of the
/// plaquettes through U_μ. Periodic boundary conditions.
pub fn weighted_staples(u: &Field<f64, Gauge>, rho: &Weights) -> Field<f64, Gauge> {
    let lat = u.lattice();
    if let Some(nb) = Neighbors::new(lat) {
        let mut sum = Field::<f64, Gauge>::gauge(lat);
        for mu in 0..ND {
            for_sites(sum.slab_mut(mu), 18, |x, out| {
                let mut c = ColorMatrix::zero();
                for nu in (0..ND).filter(|nu| *nu != mu) {
                    let r = rho[mu][nu];
                    if r == 0.0 {
                        continue;
                    }
                    let upper = nb.walk(u, x, &[(nu, true), (mu, true), (nu, false)]);
                    let lower = nb.walk(u, x, &[(nu, false), (mu, true), (nu, true)]);
                    c = c.add(&upper.add(&lower).scale(r));
                }
                c.write_reals(out);
            });
        }
        return sum;
    }
    let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
    // u at x + mu for every mu.
    let up: [Field<f64, Gauge>; ND] =
        core::array::from_fn(|mu| shifted(u, &st, mu, Direction::Forward));
    let mut sum = Field::<f64, Gauge>::gauge(lat);
    for mu in 0..ND {
        for nu in (0..ND).filter(|nu| *nu != mu) {
            let r = rho[mu][nu];
            if r == 0.0 {
                continue;
            }
            // The lower staple at x − ν, built at x and shifted back: U_ν(x)† U_μ(x) U_ν(x+μ).
            let mut lower = Field::<f64, Gauge>::gauge(lat);
            for_sites(lower.slab_mut(mu), 18, |x, out| {
                link(u, nu, x)
                    .adj()
                    .mul(&link(u, mu, x).mul(&link(&up[mu], nu, x)))
                    .write_reals(out);
            });
            let lower = shifted(&lower, &st, nu, Direction::Backward);
            let (up_nu, up_mu) = (&up[nu], &up[mu]);
            for_sites(sum.slab_mut(mu), 18, |x, out| {
                let upper = link(u, nu, x).mul(&link(up_nu, mu, x).mul(&link(up_mu, nu, x).adj()));
                let total = ColorMatrix::from_reals(out)
                    .add(&upper.scale(r))
                    .add(&link(&lower, mu, x).scale(r));
                total.write_reals(out);
            });
        }
    }
    sum
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

/// The matrices of exp(iQ) for one link: iQ, (iQ)², the coefficients f_j of
/// exp(iQ) = f0 + f1 Q + f2 Q² (Morningstar–Peardon (29)–(34)) and u, w.
struct StoutExp {
    iq: ColorMatrix<f64>,
    iq2: ColorMatrix<f64>,
    f: [Cplx<f64>; 3],
    u: f64,
    w: f64,
    /// Whether Q is (numerically) zero: then exp(iQ) = 1.
    zero: bool,
}

fn cplx(re: f64, im: f64) -> Cplx<f64> {
    Cplx::new(re, im)
}

fn xi0(w: f64) -> f64 {
    if w == 0.0 {
        1.0
    } else {
        w.sin() / w
    }
}

fn xi1(w: f64) -> f64 {
    if w < 0.25 {
        let w2 = w * w;
        let (c0, c1, c2, c3, c4) = (
            -1.0 / 3.0,
            1.0 / 30.0,
            -1.0 / 840.0,
            1.0 / 45360.0,
            -1.0 / 3991680.0,
        );
        c0 + w2 * (c1 + w2 * (c2 + w2 * (c3 + w2 * c4)))
    } else {
        (w * w.cos() - w.sin()) / (w * w * w)
    }
}

/// a 1 + b Q + c Q², with Q = −i iQ and Q² = −(iQ)².
fn poly(
    a: Cplx<f64>,
    b: Cplx<f64>,
    c: Cplx<f64>,
    iq: &ColorMatrix<f64>,
    iq2: &ColorMatrix<f64>,
) -> ColorMatrix<f64> {
    ColorMatrix(core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let one = if i == j { a } else { Cplx::new(0.0, 0.0) };
            let q = iq.0[i][j];
            // Q = −i iQ: (re, im) → (im, −re).
            one + b * cplx(q.im, -q.re) - c * iq2.0[i][j]
        })
    }))
}

fn trace(m: &ColorMatrix<f64>) -> Cplx<f64> {
    m.trace()
}

impl StoutExp {
    /// For the link `u` with weighted staple sum `c`.
    fn new(c: &ColorMatrix<f64>, u: &ColorMatrix<f64>) -> Self {
        let iq = traceless_antihermitian(&c.mul(&u.adj()));
        let iq2 = iq.mul(&iq);
        let iq3 = iq.mul(&iq2);
        let norm: f64 = iq.0.iter().flatten().map(|z| z.norm_sqr()).sum();
        if norm <= 1e-10 {
            let z = Cplx::new(0.0, 0.0);
            return Self {
                iq,
                iq2,
                f: [z; 3],
                u: 0.0,
                w: 0.0,
                zero: true,
            };
        }
        let c0 = -trace(&iq3).im / 3.0;
        let c1 = -0.5 * trace(&iq2).re;
        let c13r = (c1 / 3.0).sqrt();
        let c0max = 2.0 * c13r * c13r * c13r;
        let theta = (c0 / c0max).acos();
        let u = c13r * (theta / 3.0).cos();
        let w = c1.sqrt() * (theta / 3.0).sin();
        let (u2, w2) = (u * u, w * w);
        let (cos_w, cos_u, sin_u) = (w.cos(), u.cos(), u.sin());
        let x0 = xi0(w);
        let emiu = cplx(cos_u, -sin_u);
        let e2iu = cplx(cos_u * cos_u - sin_u * sin_u, 2.0 * sin_u * cos_u);
        let h0 = e2iu * cplx(u2 - w2, 0.0)
            + emiu * cplx(8.0 * u2 * cos_w, 2.0 * u * (3.0 * u2 + w2) * x0);
        let h1 = cplx(2.0 * u, 0.0) * e2iu - emiu * cplx(2.0 * u * cos_w, -(3.0 * u2 - w2) * x0);
        let h2 = e2iu - emiu * cplx(cos_w, 3.0 * u * x0);
        let fden = 1.0 / (9.0 * u2 - w2);
        Self {
            iq,
            iq2,
            f: [h0.scale(fden), h1.scale(fden), h2.scale(fden)],
            u,
            w,
            zero: false,
        }
    }

    fn exp(&self) -> ColorMatrix<f64> {
        if self.zero {
            return ColorMatrix::identity();
        }
        poly(self.f[0], self.f[1], self.f[2], &self.iq, &self.iq2)
    }

    /// iΛ of Morningstar–Peardon (73) for Σ' at this link (`u` the link): the traceless
    /// anti-Hermitian part of i Γ, Γ = tr(Σ'U B1) Q + tr(Σ'U B2) Q² + f1 UΣ' + f2 (Q UΣ' + UΣ' Q).
    fn lambda(&self, sigma: &ColorMatrix<f64>, u: &ColorMatrix<f64>) -> ColorMatrix<f64> {
        if self.zero {
            return ColorMatrix::zero();
        }
        let (u_, w) = (self.u, self.w);
        let [f0, f1, f2] = self.f;
        let x0 = xi0(w);
        let x1 = xi1(w);
        let (u2, w2) = (u_ * u_, w * w);
        let cos_w = w.cos();
        let emiu = cplx(u_.cos(), -u_.sin());
        let e2iu = cplx((2.0 * u_).cos(), (2.0 * u_).sin());
        let r01 = cplx(2.0 * u_, 2.0 * (u2 - w2)) * e2iu
            + emiu
                * cplx(
                    16.0 * u_ * cos_w + 2.0 * u_ * (3.0 * u2 + w2) * x0,
                    -8.0 * u2 * cos_w + 2.0 * (9.0 * u2 + w2) * x0,
                );
        let r11 = cplx(2.0, 4.0 * u_) * e2iu
            + emiu
                * cplx(
                    -2.0 * cos_w + (3.0 * u2 - w2) * x0,
                    2.0 * u_ * cos_w + 6.0 * u_ * x0,
                );
        let r21 = cplx(0.0, 2.0) * e2iu + emiu * cplx(-3.0 * u_ * x0, cos_w - 3.0 * x0);
        let r02 = cplx(-2.0, 0.0) * e2iu
            + emiu * cplx(-8.0 * u2 * x0, 2.0 * u_ * (cos_w + x0 + 3.0 * u2 * x1));
        let r12 = emiu * cplx(2.0 * u_ * x0, -cos_w - x0 + 3.0 * u2 * x1);
        let r22 = emiu * cplx(x0, -3.0 * u_ * x1);
        let fden = 1.0 / (2.0 * (9.0 * u2 - w2) * (9.0 * u2 - w2));
        let k1 = cplx(2.0 * u_, 0.0);
        let k2 = cplx(3.0 * u2 - w2, 0.0);
        let k3 = cplx(30.0 * u2 + 2.0 * w2, 0.0);
        let b10 = (k1 * r01 + k2 * r02 - k3 * f0).scale(fden);
        let b11 = (k1 * r11 + k2 * r12 - k3 * f1).scale(fden);
        let b12 = (k1 * r21 + k2 * r22 - k3 * f2).scale(fden);
        let k4 = cplx(3.0 * u_, 0.0);
        let k5 = cplx(24.0 * u_, 0.0);
        let b20 = (r01 - k4 * r02 - k5 * f0).scale(fden);
        let b21 = (r11 - k4 * r12 - k5 * f1).scale(fden);
        let b22 = (r21 - k4 * r22 - k5 * f2).scale(fden);
        let (iq, iq2) = (&self.iq, &self.iq2);
        let b1 = poly(b10, b11, b12, iq, iq2);
        let b2 = poly(b20, b21, b22, iq, iq2);
        let us = u.mul(sigma);
        let tr1 = trace(&us.mul(&b1));
        let tr2 = trace(&us.mul(&b2));
        let iqus = iq.mul(&us);
        let iusq = us.mul(iq);
        let gamma = ColorMatrix(core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                let q = iq.0[i][j];
                let a = iqus.0[i][j];
                let b = iusq.0[i][j];
                let qt = tr1 * cplx(q.im, -q.re) - tr2 * iq2.0[i][j]
                    + f1 * us.0[i][j]
                    + f2 * cplx(a.im, -a.re)
                    + f2 * cplx(b.im, -b.re);
                // i Γ.
                cplx(-qt.im, qt.re)
            })
        }));
        traceless_antihermitian(&gamma)
    }
}

impl Stout {
    pub fn uniform(rho: f64) -> Self {
        Self { rho: uniform(rho) }
    }

    pub fn apply(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        let c = weighted_staples(u, &self.rho);
        let mut v = Field::<f64, Gauge>::gauge(u.lattice());
        for mu in 0..ND {
            for_sites(v.slab_mut(mu), 18, |x, out| {
                let ux = link(u, mu, x);
                StoutExp::new(&link(&c, mu, x), &ux)
                    .exp()
                    .mul(&ux)
                    .write_reals(out);
            });
        }
        v
    }

    /// One step of the chain rule (Morningstar–Peardon (75)): from Σ' of the smeared links
    /// V = exp(iQ) U (dS = Re tr Σ' δV) to Σ of the links `u`.
    pub fn chain(
        &self,
        sigma_prime: &Field<f64, Gauge>,
        u: &Field<f64, Gauge>,
    ) -> Field<f64, Gauge> {
        let lat = u.lattice();
        let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
        let c = weighted_staples(u, &self.rho);
        // Σ' exp(iQ) + C† iΛ, and iΛ U.
        let mut sigma = Field::<f64, Gauge>::gauge(lat);
        let mut itheta = Field::<f64, Gauge>::gauge(lat);
        for mu in 0..ND {
            let s = sigma.slab_mut(mu);
            let t = itheta.slab_mut(mu);
            s.as_chunks_mut::<18>()
                .0
                .par_iter_mut()
                .zip(t.as_chunks_mut::<18>().0.par_iter_mut())
                .enumerate()
                .for_each(|(x, (s, t))| {
                    let (cx, ux, sx) = (link(&c, mu, x), link(u, mu, x), link(sigma_prime, mu, x));
                    let e = StoutExp::new(&cx, &ux);
                    let lambda = e.lambda(&sx, &ux);
                    lambda.mul(&ux).write_reals(t);
                    sx.mul(&e.exp()).add(&cx.adj().mul(&lambda)).write_reals(s);
                });
        }
        // The terms from the links whose staples contain U_μ(x).
        if let Some(nb) = Neighbors::new(lat) {
            let th = &itheta;
            for mu in 0..ND {
                for_sites(sigma.slab_mut(mu), 18, |x, out| {
                    let mut acc = ColorMatrix::from_reals(out);
                    for nu in (0..ND).filter(|nu| *nu != mu) {
                        let r = self.rho[mu][nu];
                        if r == 0.0 {
                            continue;
                        }
                        let (xm, xn, y) = (nb.up(x, mu), nb.up(x, nu), nb.down(x, nu));
                        let ym = nb.up(y, mu);
                        let vnu_xm = link(u, nu, xm);
                        let vmu_xn_d = link(u, mu, xn).adj();
                        let vnu_d = link(u, nu, x).adj();
                        let vnu_ym_d = link(u, nu, ym).adj();
                        let (vmu_y_d, vnu_y) = (link(u, mu, y).adj(), link(u, nu, y));
                        let each = vnu_xm
                            .mul(&vmu_xn_d)
                            .mul(&link(th, nu, x).adj())
                            .add(&vnu_ym_d.mul(&link(th, mu, y).adj()).mul(&vnu_y))
                            .add(&vnu_ym_d.mul(&vmu_y_d).mul(&link(th, nu, y)))
                            .add(&link(th, nu, ym).adj().mul(&vmu_y_d).mul(&vnu_y))
                            .add(&link(th, nu, xm).mul(&vmu_xn_d).mul(&vnu_d))
                            .add(&vnu_xm.mul(&link(th, mu, xn).adj()).mul(&vnu_d));
                        acc = acc.add(&each.scale(r));
                    }
                    acc.write_reals(out);
                });
            }
            return sigma;
        }
        let up: [Field<f64, Gauge>; ND] =
            core::array::from_fn(|d| shifted(u, &st, d, Direction::Forward));
        let tup: [Field<f64, Gauge>; ND] =
            core::array::from_fn(|d| shifted(&itheta, &st, d, Direction::Forward));
        for mu in 0..ND {
            for nu in (0..ND).filter(|nu| *nu != mu) {
                let r = self.rho[mu][nu];
                if r == 0.0 {
                    continue;
                }
                // The three terms that are built at x − ν.
                let mut low = Field::<f64, Gauge>::gauge(lat);
                for_sites(low.slab_mut(0), 18, |y, out| {
                    let vnu_mu = link(&up[mu], nu, y).adj(); // V_ν(y+μ)†
                    let (vmu, vnu) = (link(u, mu, y), link(u, nu, y));
                    let (tmu, tnu) = (link(&itheta, mu, y), link(&itheta, nu, y));
                    let tnu_mu = link(&tup[mu], nu, y); // iΘ_ν(y+μ)
                    let g = vnu_mu
                        .mul(&tmu.adj().mul(&vnu))
                        .add(&vnu_mu.mul(&vmu.adj().mul(&tnu)))
                        .add(&tnu_mu.adj().mul(&vmu.adj()).mul(&vnu));
                    g.write_reals(out);
                });
                let low = shifted(&low, &st, nu, Direction::Backward);
                for_sites(sigma.slab_mut(mu), 18, |x, out| {
                    let vnu_mu = link(&up[mu], nu, x); // V_ν(x+μ)
                    let vmu_nu = link(&up[nu], mu, x); // V_μ(x+ν)
                    let vnu = link(u, nu, x);
                    let tnu = link(&itheta, nu, x);
                    let tnu_mu = link(&tup[mu], nu, x); // iΘ_ν(x+μ)
                    let tmu_nu = link(&tup[nu], mu, x); // iΘ_μ(x+ν)
                    let each = vnu_mu
                        .mul(&vmu_nu.adj())
                        .mul(&tnu.adj())
                        .add(&link(&low, 0, x))
                        .add(&tnu_mu.mul(&vmu_nu.adj()).mul(&vnu.adj()))
                        .add(&vnu_mu.mul(&tmu_nu.adj()).mul(&vnu.adj()));
                    ColorMatrix::from_reals(out)
                        .add(&each.scale(r))
                        .write_reals(out);
                });
            }
        }
        sigma
    }
}

impl Ape {
    pub fn uniform(rho: f64, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            rho: uniform(rho),
            tolerance,
            max_iterations,
        }
    }

    pub fn apply(&self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        let lat = u.lattice();
        let c = weighted_staples(u, &self.rho);
        let mut v = Field::<f64, Gauge>::gauge(lat);
        for mu in 0..ND {
            let keep = 1.0 - self.rho[mu][mu];
            let a: Vec<ColorMatrix<f64>> = (0..u.num_sites())
                .map(|x| link(&c, mu, x).add(&link(u, mu, x).scale(keep)))
                .collect();
            let g = self.max_trace(a, lat);
            for_sites(v.slab_mut(mu), 18, |x, out| g[x].adj().write_reals(out));
        }
        v
    }

    /// G maximising Re tr(A G) site by site, by sweeps over the three SU(2) subgroups until
    /// the sweep changes G by less than the tolerance on average over all processes.
    fn max_trace(
        &self,
        mut a: Vec<ColorMatrix<f64>>,
        lat: &crate::Lattice,
    ) -> Vec<ColorMatrix<f64>> {
        let mut g = vec![ColorMatrix::identity(); a.len()];
        let total = (3 * lat.num_global_sites()) as f64;
        for _ in 0..self.max_iterations {
            let retr: f64 = a
                .par_iter_mut()
                .zip(g.par_iter_mut())
                .map(|(ax, gx)| {
                    let mut delta = ColorMatrix::identity();
                    for i1 in 0..3 {
                        let i2 = (i1 + 1) % 3;
                        let v = su2_max(ax, i1, i2);
                        *ax = ax.mul(&v);
                        *gx = gx.mul(&v);
                        delta = delta.mul(&v);
                    }
                    delta.trace().re
                })
                .sum();
            let mut sum = [retr];
            lat.comm.sum(&mut sum);
            if 1.0 - sum[0] / total < self.tolerance {
                return g;
            }
        }
        panic!(
            "APE smearing: the SU(3) projection did not converge in {} sweeps",
            self.max_iterations
        );
    }
}

/// The SU(2) matrix in rows and columns i1, i2 that maximises Re tr(A v).
fn su2_max(a: &ColorMatrix<f64>, i1: usize, i2: usize) -> ColorMatrix<f64> {
    let (j1, j2, k1, k2) = (a.0[i1][i1], a.0[i2][i2], a.0[i1][i2], a.0[i2][i1]);
    let lam = j1.norm_sqr() + 2.0 * j1.re * j2.re + k1.norm_sqr() - 2.0 * j1.im * j2.im
        + k2.norm_sqr()
        - 2.0 * k1.re * k2.re
        + j2.norm_sqr()
        + 2.0 * k1.im * k2.im;
    let lam = 1.0 / lam.sqrt();
    let mut v = ColorMatrix::identity();
    v.0[i1][i1] = cplx((j1.re + j2.re) * lam, (-j1.im + j2.im) * lam);
    v.0[i1][i2] = cplx((k2.re - k1.re) * lam, (-k2.im - k1.im) * lam);
    v.0[i2][i1] = cplx((k1.re - k2.re) * lam, (-k1.im - k2.im) * lam);
    v.0[i2][i2] = cplx((j1.re + j2.re) * lam, (j1.im - j2.im) * lam);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauge::{plaquette, random_gauge};
    use crate::Lattice;

    fn is_su3(m: &ColorMatrix<f64>) -> bool {
        let d = m.mul(&m.adj()).sub(&ColorMatrix::identity());
        let det = {
            let a = &m.0;
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        };
        d.0.iter().flatten().all(|z| z.norm_sqr() < 1e-24)
            && (det - cplx(1.0, 0.0)).norm_sqr() < 1e-24
    }

    /// exp(iQ) from the Cayley–Hamilton coefficients is the series.
    #[test]
    fn stout_exponential_is_the_series() {
        let lat = Lattice::single_node([2, 2, 2, 2]);
        let u = random_gauge(&lat, 3);
        let c = weighted_staples(&u, &uniform(0.13));
        let (cx, ux) = (link(&c, 2, 5), link(&u, 2, 5));
        let e = StoutExp::new(&cx, &ux);
        let mut series = ColorMatrix::identity();
        let mut term = ColorMatrix::identity();
        for k in 1..30 {
            term = e.iq.mul(&term).scale(1.0 / k as f64);
            series = series.add(&term);
        }
        let d = e.exp().sub(&series);
        assert!(d.0.iter().flatten().all(|z| z.norm_sqr() < 1e-26));
    }

    /// Smeared links are in SU(3) and smoother: the plaquette goes up.
    #[test]
    fn smeared_links_are_su3_and_smoother() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 7);
        // Close to the identity, so that the plaquette is that of a rough but ordered field.
        let mut w = Field::<f64, Gauge>::gauge(&lat);
        for mu in 0..ND {
            for_sites(w.slab_mut(mu), 18, |x, out| {
                let m = link(&u, mu, x);
                let a = traceless_antihermitian(&m).scale(0.3);
                let mut e = ColorMatrix::identity();
                let mut t = ColorMatrix::identity();
                for k in 1..20 {
                    t = a.mul(&t).scale(1.0 / k as f64);
                    e = e.add(&t);
                }
                e.write_reals(out);
            });
        }
        let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
        let p0 = plaquette(&w, &st);
        for smear in [
            Smear::Stout(Stout::uniform(0.1)),
            Smear::Ape(Ape::uniform(0.1, 1e-14, 100)),
        ] {
            let v = GaugeSmearing { smear, steps: 2 }.apply(&w);
            for mu in 0..ND {
                for x in 0..v.num_sites() {
                    assert!(is_su3(&link(&v, mu, x)), "{smear:?}");
                }
            }
            let p = plaquette(&v, &st);
            assert!(p > p0 + 0.01, "{smear:?}: {p0} → {p}");
        }
        // Spatial weights leave the time links alone.
        let v = Stout { rho: spatial(0.1) }.apply(&w);
        assert_eq!(v.slab(ND - 1), w.slab(ND - 1));
    }
}
