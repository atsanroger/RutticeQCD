//! Two flavours of fermions in HMC: the pseudofermion action S = φ† (D†D)⁻¹ φ and its force,
//! for operators of Wilson type made of (optionally smeared) links.
//!
//! The force: with η = (D†D)⁻¹ φ and ζ = D η, a change δU of the links changes S by
//! −2 Re ζ† δD η. For D = diag + c Σ_μ [(1 − γ_μ) U_μ(x) δ_(x+μ,y) + (1 + γ_μ) U_μ(x−μ)† δ_(x−μ,y)]
//! and δU_μ(x) = X U_μ(x) this is Re tr X M_μ(x) with
//! M_μ(x) = −2c Σ_spin [h_μ(η)(x) ζ(x)† − γ5 η(x) h_μ(γ5 ζ)(x)†],
//! h_μ(ψ)(x) = (1 − γ_μ) U_μ(x) ψ(x+μ) the forward hopping term (`HopTerms::hop_forward`,
//! boundary phases included); the backward hop is turned into a forward one with γ5. The
//! force on the momenta is then F = TA(M) (`gauge_force`'s convention).
//!
//! With smeared links V(U) the derivative is taken for V, Σ' = V† M (dS = Re tr Σ' δV), and
//! carried back to U by the chain rule of the smearing (`GaugeSmearing::chain`).

use super::{traceless_antihermitian, Mseries};
use crate::fermion::{Bound, HopTerms, Variant};
use crate::gauge::{link, GaugeSmearing};
use crate::kernel::{ColorMatrix, Cplx, NC, NS};
use crate::solver::{Cg, SolveError, Solver, Stop};
use crate::{for_sites, Field, Gauge, Normal, Spinor, Vector, C64, ND};

/// A spinor field of one vector.
type Psi = Field<f64, Spinor>;

/// Σ_spin a(x) b(x)†, a colour matrix per site, accumulated into `m` times `k`.
fn add_outer(m: &mut [f64], a: &Psi, b: &Psi, k: f64) {
    let (sa, sb) = (a.slab(0), b.slab(0));
    for_sites(m, 18, |x, out| {
        let (va, vb) = (&sa[24 * x..24 * (x + 1)], &sb[24 * x..24 * (x + 1)]);
        let mut acc = ColorMatrix::from_reals(out);
        for s in 0..NS {
            for i in 0..NC {
                let ai = Cplx::new(va[2 * (i + NC * s)], va[2 * (i + NC * s) + 1]);
                for j in 0..NC {
                    let bj = Cplx::new(vb[2 * (j + NC * s)], -vb[2 * (j + NC * s) + 1]);
                    acc.0[i][j] += (ai * bj).scale(k);
                }
            }
        }
        acc.write_reals(out);
    });
}

/// M_μ(x) of the module documentation for the operator `op` (its links): Re tr X M is the
/// change of S = −2 Re ζ† δD η … per unit of X, i.e. dS for δU = X U.
pub fn hop_derivative<O: HopTerms<Psi>>(op: &O, zeta: &Psi, eta: &Psi) -> Field<f64, Gauge> {
    let mut work = op.work();
    let lat = eta.lattice();
    let c = op.hop_coefficient();
    let mut g5_eta = eta.zeros_like();
    let mut g5_zeta = eta.zeros_like();
    op.mult_gamma5(&mut work, &mut g5_eta, eta);
    op.mult_gamma5(&mut work, &mut g5_zeta, zeta);
    let mut m = Field::<f64, Gauge>::gauge(lat);
    let mut h = eta.zeros_like();
    for mu in 0..ND {
        h.set_zero();
        op.hop_forward(&mut work, mu, &mut h, eta);
        add_outer(m.slab_mut(mu), &h, zeta, -2.0 * c);
        h.set_zero();
        op.hop_forward(&mut work, mu, &mut h, &g5_zeta);
        add_outer(m.slab_mut(mu), &g5_eta, &h, 2.0 * c);
    }
    m
}

/// F = TA(M) for every link.
pub fn force_from(m: &Field<f64, Gauge>) -> Field<f64, Gauge> {
    let mut f = Field::<f64, Gauge>::gauge(m.lattice());
    for mu in 0..ND {
        for_sites(f.slab_mut(mu), 18, |x, out| {
            traceless_antihermitian(&link(m, mu, x)).write_reals(out);
        });
    }
    f
}

/// The force of an action whose derivative M (dS = Re tr X M for δV = X V) is known for the
/// smeared links V of `u`: carried back to `u` and projected.
pub fn smeared_force(
    m_v: &Field<f64, Gauge>,
    u: &Field<f64, Gauge>,
    smearing: Option<&GaugeSmearing>,
    levels: &[Field<f64, Gauge>],
) -> Field<f64, Gauge> {
    let Some(smearing) = smearing.filter(|s| s.steps > 0) else {
        return force_from(m_v);
    };
    let v = levels.last().expect("the smeared links");
    let lat = u.lattice();
    // Σ' = V† M.
    let mut sigma = Field::<f64, Gauge>::gauge(lat);
    for mu in 0..ND {
        for_sites(sigma.slab_mut(mu), 18, |x, out| {
            link(v, mu, x).adj().mul(&link(m_v, mu, x)).write_reals(out);
        });
    }
    let sigma = smearing.chain(sigma, u, levels);
    let mut m = Field::<f64, Gauge>::gauge(lat);
    for mu in 0..ND {
        for_sites(m.slab_mut(mu), 18, |x, out| {
            link(u, mu, x).mul(&link(&sigma, mu, x)).write_reals(out);
        });
    }
    force_from(&m)
}

/// Two degenerate flavours, S = φ† (D†D)⁻¹ φ, with D the operator `op` on the links smeared
/// by `smearing` (none: the links themselves). (D†D)⁻¹ by CG to |r|²/|b|² < `stop`.
pub struct TwoFlavor<O> {
    pub op: O,
    pub smearing: Option<GaugeSmearing>,
    /// The solves for the force and for the energy.
    pub stop_md: Stop,
    pub stop_h: Stop,
    phi: Option<Psi>,
    /// CG iterations of the last solve.
    pub last_iterations: usize,
}

impl<O: HopTerms<Psi>> TwoFlavor<O> {
    pub fn new(op: O, smearing: Option<GaugeSmearing>, stop_md: Stop, stop_h: Stop) -> Self {
        Self {
            op,
            smearing,
            stop_md,
            stop_h,
            phi: None,
            last_iterations: 0,
        }
    }

    /// The operator on the (smeared) links of `u`; the smeared levels, for the force.
    fn set_links(&mut self, u: &Field<f64, Gauge>) -> Vec<Field<f64, Gauge>> {
        let levels = match &self.smearing {
            Some(s) => s.levels(u),
            None => Vec::new(),
        };
        self.op
            .set_gauge(levels.last().unwrap_or(u))
            .expect("the links suit the operator");
        levels
    }

    /// (D†D)⁻¹ φ.
    fn solve(&mut self, stop: Stop) -> Result<Psi, SolveError> {
        let phi = self.phi.as_ref().expect("pseudofermions drawn");
        let mut x = phi.zeros_like();
        let mut cg = Cg::new(stop);
        let mut d = Bound::new(&self.op, Variant::D);
        let mut normal = Normal::new(&mut d, phi);
        let info = cg.solve(&mut normal, None, &mut x, phi)?;
        self.last_iterations = info.iterations;
        Ok(x)
    }

    /// The heat bath: φ = D† ξ, ξ Gaussian (variance ½ per real component) drawn in Bridge++'s
    /// order; returns |ξ|², the action of φ.
    pub fn langevin(&mut self, u: &Field<f64, Gauge>, rng: &mut Mseries) -> f64 {
        self.set_links(u);
        let lat = u.lattice().clone();
        let mut xi = Psi::spinor(&lat, 1);
        rng.gauss_lex_global(&lat, 24, 1, |_, x, h| {
            xi.site_mut(0, x).copy_from_slice(h);
        });
        let mut phi = xi.zeros_like();
        let mut work = self.op.work();
        self.op.mult(&mut work, &mut phi, &xi, Variant::Dagger);
        self.phi = Some(phi);
        let mut n = [0.0];
        xi.norm_sqr(&mut n);
        n[0]
    }

    /// S = φ† (D†D)⁻¹ φ.
    pub fn hamiltonian(&mut self, u: &Field<f64, Gauge>) -> f64 {
        self.set_links(u);
        let v = self
            .solve(self.stop_h)
            .expect("the solve for the energy converges");
        let mut d = [C64::ZERO];
        v.dot(self.phi.as_ref().expect("drawn"), &mut d);
        d[0].re
    }

    pub fn force(&mut self, u: &Field<f64, Gauge>) -> Field<f64, Gauge> {
        let levels = self.set_links(u);
        let eta = self
            .solve(self.stop_md)
            .expect("the solve for the force converges");
        let mut zeta = eta.zeros_like();
        let mut work = self.op.work();
        self.op.mult(&mut work, &mut zeta, &eta, Variant::D);
        let m = hop_derivative(&self.op, &zeta, &eta);
        smeared_force(&m, u, self.smearing.as_ref(), &levels)
    }
}
