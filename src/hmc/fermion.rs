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

use super::{traceless_antihermitian, Links, MdAction, Mseries};
use crate::fermion::{Bound, HopTerms, Variant};
use crate::gauge::{link, GaugeSmearing};
use crate::kernel::{ColorMatrix, Cplx, NC, NS};
use crate::solver::{Cg, SolveError, Solver, Stop};
use crate::{for_sites, Field, Gauge, Normal, Spinor, Vector, C64, ND};
use std::rc::Rc;

/// A spinor field of one vector.
type Psi = Field<f64, Spinor>;

/// Σ_spin a(x) b(x)†, a colour matrix per site, accumulated into `m` times `k`; with
/// several spinors per site (a fifth dimension), summed over them too.
fn add_outer(m: &mut [f64], a: &Psi, b: &Psi, k: f64) {
    let (sa, sb) = (a.slab(0), b.slab(0));
    let per = sa.len() / (m.len() / 18);
    for_sites(m, 18, |x, out| {
        let (va, vb) = (&sa[per * x..per * (x + 1)], &sb[per * x..per * (x + 1)]);
        let mut acc = ColorMatrix::from_reals(out);
        for (va, vb) in va.as_chunks::<24>().0.iter().zip(vb.as_chunks::<24>().0) {
            for s in 0..NS {
                for i in 0..NC {
                    let ai = Cplx::new(va[2 * (i + NC * s)], va[2 * (i + NC * s) + 1]);
                    for j in 0..NC {
                        let bj = Cplx::new(vb[2 * (j + NC * s)], -vb[2 * (j + NC * s) + 1]);
                        acc.0[i][j] += (ai * bj).scale(k);
                    }
                }
            }
        }
        acc.write_reals(out);
    });
}

/// M_μ(x) of the module documentation for the operator `op` (its links): Re tr X M is the
/// change of S = −2 Re ζ† δD η … per unit of X, i.e. dS for δU = X U.
pub fn hop_derivative<O: HopTerms<Psi> + ?Sized>(
    op: &O,
    zeta: &Psi,
    eta: &Psi,
) -> Field<f64, Gauge> {
    kernel_derivative(op, op.hop_coefficient(), zeta, eta)
}

/// `hop_derivative` for an operator whose hops are those of the Wilson kernel `op` (γ5 its
/// four-dimensional one) with coefficient `c`, acting on `eta`.
pub fn kernel_derivative<O: HopTerms<Psi> + ?Sized>(
    op: &O,
    c: f64,
    zeta: &Psi,
    eta: &Psi,
) -> Field<f64, Gauge> {
    let mut work = op.work();
    let lat = eta.lattice();
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

/// The part of M that comes from the clover term −κ c_SW Σ_(μ<ν) F_μν ⊗ γ_μ γ_ν of D, for
/// the links `u` of the operator (F_μν = (Q − Q†)/8, Q the four leaves at x; `clover`'s
/// convention). With A_μν(x) = Σ_spin γ_μγ_ν η(x) ζ(x)† and B = κ c_SW (A − A†)/4, a change of
/// the links changes S by Σ_x Σ_(μ<ν) Re tr δQ_μν(x) B_μν(x). A leaf through the link U_ρ(y)
/// contributes the leaf turned to start at y with B inserted where the leaf starts: the
/// product of its links from y round to y, plus for a forward step and minus for a backward
/// one.
pub fn clover_derivative(
    u: &Field<f64, Gauge>,
    kappa_csw: f64,
    zeta: &Psi,
    eta: &Psi,
) -> Field<f64, Gauge> {
    use crate::gauge::paths::{Extended, Step};
    use crate::gauge::PLANES;
    use crate::kernel::{Dirac, GammaRepre};
    let lat = u.lattice();
    // B for the six planes, in two fields of four matrices per site.
    let mut b = [
        Field::<f64, Gauge>::gauge(lat),
        Field::<f64, Gauge>::gauge(lat),
    ];
    let (sz, se) = (zeta.slab(0), eta.slab(0));
    for (plane, &(mu, nu)) in PLANES.iter().enumerate() {
        let g = Dirac::GAMMA[mu].mul(&Dirac::GAMMA[nu]);
        for_sites(b[plane / ND].slab_mut(plane % ND), 18, |x, out| {
            let (z, e) = (&sz[24 * x..24 * (x + 1)], &se[24 * x..24 * (x + 1)]);
            let mut a = ColorMatrix::zero();
            for s in 0..NS {
                let t = g.col[s];
                for i in 0..NC {
                    let gi =
                        g.val[s].times(Cplx::new(e[2 * (i + NC * t)], e[2 * (i + NC * t) + 1]));
                    for j in 0..NC {
                        let zj = Cplx::new(z[2 * (j + NC * s)], -z[2 * (j + NC * s) + 1]);
                        a.0[i][j] += gi * zj;
                    }
                }
            }
            a.sub(&a.adj()).scale(kappa_csw / 4.0).write_reals(out);
        });
    }
    let eu = Extended::new(u, 2);
    // The same depth as the links: the indices are the same.
    let eb = [Extended::new(&b[0], 2), Extended::new(&b[1], 2)];
    let (f, bk) = (true, false);
    let walk = |e: usize, path: &[Step]| -> (ColorMatrix<f64>, usize) {
        let mut m = ColorMatrix::identity();
        let mut y = e;
        for &s in path {
            let (l, z) = eu.step(y, s);
            m = m.mul(&l);
            y = z;
        }
        (m, y)
    };
    let mut m = Field::<f64, Gauge>::gauge(lat);
    for rho in 0..ND {
        for_sites(m.slab_mut(rho), 18, |y, out| {
            let e = eu.at(y);
            let mut acc = ColorMatrix::zero();
            for (plane, &(mu, nu)) in PLANES.iter().enumerate() {
                if rho != mu && rho != nu {
                    continue;
                }
                let leaves: [[Step; 4]; 4] = [
                    [(mu, f), (nu, f), (mu, bk), (nu, bk)],
                    [(nu, f), (mu, bk), (nu, bk), (mu, f)],
                    [(mu, bk), (nu, bk), (mu, f), (nu, f)],
                    [(nu, bk), (mu, f), (nu, f), (mu, bk)],
                ];
                for leaf in &leaves {
                    for (k, &(d, forward)) in leaf.iter().enumerate() {
                        if d != rho {
                            continue;
                        }
                        let cut = if forward { k } else { k + 1 };
                        let (w1, ex) = walk(e, &leaf[cut..]);
                        let bx = eb[plane / ND].link(plane % ND, ex);
                        let (w2, _) = walk(ex, &leaf[..cut]);
                        let term = w1.mul(&bx).mul(&w2);
                        acc = if forward {
                            acc.add(&term)
                        } else {
                            acc.sub(&term)
                        };
                    }
                }
            }
            acc.write_reals(out);
        });
    }
    m
}

/// What the pseudofermion actions need of an operator besides applying it.
pub trait ForceOp: HopTerms<Psi> + Sized {
    /// M of −2 Re ζ† δD η (dS = Re tr X M for δU = X U).
    fn derivative(&self, zeta: &Psi, eta: &Psi) -> Field<f64, Gauge> {
        hop_derivative(self, zeta, eta)
    }
    /// A zero field of the shape the operator acts on (one vector).
    fn new_field(&self, lat: &std::sync::Arc<crate::Lattice>) -> Psi {
        Psi::spinor(lat, 1)
    }
}

impl<G: crate::kernel::GammaRepre, D: crate::fermion::DiagTerm<f64>> ForceOp
    for crate::fermion::WilsonOp<f64, G, D>
{
}

/// Domain-wall fermions: the hops are those of the Wilson kernel acting on b η − c Y η.
impl<G: crate::kernel::GammaRepre> ForceOp for crate::fermion::Mdwf<f64, G> {
    fn derivative(&self, zeta: &Psi, eta: &Psi) -> Field<f64, Gauge> {
        kernel_derivative(
            self.kernel(),
            self.hop_coefficient(),
            zeta,
            &self.kernel_input(eta),
        )
    }
    fn new_field(&self, lat: &std::sync::Arc<crate::Lattice>) -> Psi {
        Field::with_shape(lat, 24, self.params().ls, 1, 1)
    }
}

/// Gaussian numbers of variance ½ for every real of `xi`, drawn as Bridge++ draws them for a
/// field with the spinors of a site's fifth dimension in its outer index.
fn draw(xi: &mut Psi, rng: &mut Mseries) {
    let lat = xi.lattice().clone();
    let ls = xi.slab(0).len() / (24 * lat.num_sites());
    rng.gauss_lex_global(&lat, 24, ls, |s, x, h| {
        xi.site_mut(0, x)[24 * s..24 * (s + 1)].copy_from_slice(h);
    });
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
    /// κ c_SW when the operator has a clover term (`with_clover`).
    clover: Option<f64>,
}

impl<O: ForceOp> TwoFlavor<O> {
    pub fn new(op: O, smearing: Option<GaugeSmearing>, stop_md: Stop, stop_h: Stop) -> Self {
        Self {
            op,
            smearing,
            stop_md,
            stop_h,
            phi: None,
            last_iterations: 0,
            clover: None,
        }
    }

    /// The operator has the clover term with coefficient κ c_SW: its force is added.
    pub fn with_clover(mut self, kappa_csw: f64) -> Self {
        self.clover = Some(kappa_csw);
        self
    }

    /// The operator on the (smeared) links; the smeared levels, for the force.
    fn set_links(&mut self, links: &Links) -> Rc<Vec<Field<f64, Gauge>>> {
        let levels = links.levels(self.smearing.as_ref());
        self.op
            .set_gauge(levels.last().unwrap_or(links.thin))
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
    pub fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let mut xi = self.op.new_field(links.thin.lattice());
        draw(&mut xi, rng);
        let mut phi = xi.zeros_like();
        let mut work = self.op.work();
        self.op.mult(&mut work, &mut phi, &xi, Variant::Dagger);
        self.phi = Some(phi);
        let mut n = [0.0];
        xi.norm_sqr(&mut n);
        n[0]
    }

    /// S = φ† (D†D)⁻¹ φ.
    pub fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.set_links(links);
        let v = self
            .solve(self.stop_h)
            .expect("the solve for the energy converges");
        let mut d = [C64::ZERO];
        v.dot(self.phi.as_ref().expect("drawn"), &mut d);
        d[0].re
    }

    pub fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        let levels = self.set_links(links);
        let eta = self
            .solve(self.stop_md)
            .expect("the solve for the force converges");
        let mut zeta = eta.zeros_like();
        let mut work = self.op.work();
        self.op.mult(&mut work, &mut zeta, &eta, Variant::D);
        let mut m = self.op.derivative(&zeta, &eta);
        if let Some(kc) = self.clover {
            let v = levels.last().unwrap_or(links.thin);
            let c = clover_derivative(v, kc, &zeta, &eta);
            for (a, b) in m.as_mut_slice().iter_mut().zip(c.as_slice()) {
                *a += b;
            }
        }
        smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

/// Two flavours over two Pauli-Villars flavours, det(D†D)/det(D_PV†D_PV), as Bridge++'s
/// `Action_F_Ratio_lex` writes it with H = Γ5 D: S = φ† H_PV (H†H)⁻¹ H_PV† φ, φ drawn as
/// H_PV (H_PV†H_PV)⁻¹ H† ξ so that S = |ξ|² at the start.
pub struct TwoFlavorRatio<O> {
    pub op: O,
    pub pv: O,
    pub smearing: Option<GaugeSmearing>,
    pub stop_md: Stop,
    pub stop_h: Stop,
    phi: Option<Psi>,
}

impl<O: ForceOp> TwoFlavorRatio<O> {
    pub fn new(op: O, pv: O, smearing: Option<GaugeSmearing>, stop_md: Stop, stop_h: Stop) -> Self {
        Self {
            op,
            pv,
            smearing,
            stop_md,
            stop_h,
            phi: None,
        }
    }

    fn set_links(&mut self, links: &Links) -> Rc<Vec<Field<f64, Gauge>>> {
        let levels = links.levels(self.smearing.as_ref());
        let v = levels.last().unwrap_or(links.thin);
        self.op.set_gauge(v).expect("the links suit the operator");
        self.pv.set_gauge(v).expect("the links suit the operator");
        levels
    }

    /// (D†D)⁻¹ b for the operator `op`.
    fn solve(op: &O, b: &Psi, stop: Stop) -> Psi {
        let mut x = b.zeros_like();
        let mut d = Bound::new(op, Variant::D);
        let mut normal = Normal::new(&mut d, b);
        Cg::new(stop)
            .solve(&mut normal, None, &mut x, b)
            .expect("the solve converges");
        x
    }

    fn apply(op: &O, v: &Psi, variant: Variant) -> Psi {
        let mut out = v.zeros_like();
        op.mult(&mut op.work(), &mut out, v, variant);
        out
    }

    /// Γ5 of the operator (γ5 R for domain-wall fermions).
    fn gamma5(op: &O, v: &Psi) -> Psi {
        let mut out = v.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut out, v);
        out
    }

    pub fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let mut xi = self.op.new_field(links.thin.lattice());
        draw(&mut xi, rng);
        // H† ξ = D† Γ5 ξ.
        let v2 = Self::apply(&self.op, &Self::gamma5(&self.op, &xi), Variant::Dagger);
        let v1 = Self::solve(&self.pv, &v2, self.stop_h);
        self.phi = Some(Self::apply(&self.pv, &v1, Variant::Gamma5D));
        let mut n = [0.0];
        xi.norm_sqr(&mut n);
        n[0]
    }

    /// H_PV† φ = D_PV† Γ5 φ.
    fn v1(&self) -> Psi {
        let phi = self.phi.as_ref().expect("pseudofermions drawn");
        Self::apply(&self.pv, &Self::gamma5(&self.pv, phi), Variant::Dagger)
    }

    pub fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.set_links(links);
        let v1 = self.v1();
        let v2 = Self::solve(&self.op, &v1, self.stop_h);
        let mut d = [C64::ZERO];
        v1.dot(&v2, &mut d);
        d[0].re
    }

    /// dS = −2 Re (D v2)† δD v2 + 2 Re (Γ5 φ)† δD_PV v2, v2 = (D†D)⁻¹ H_PV† φ.
    pub fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        let levels = self.set_links(links);
        let v1 = self.v1();
        let v2 = Self::solve(&self.op, &v1, self.stop_md);
        let dv2 = Self::apply(&self.op, &v2, Variant::D);
        let mut m = self.op.derivative(&dv2, &v2);
        let g5phi = Self::gamma5(&self.pv, self.phi.as_ref().expect("drawn"));
        let m_pv = self.pv.derivative(&g5phi, &v2);
        for (a, b) in m.as_mut_slice().iter_mut().zip(m_pv.as_slice()) {
            *a -= b;
        }
        smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

impl<O: ForceOp> MdAction for TwoFlavor<O> {
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        TwoFlavor::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        TwoFlavor::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        TwoFlavor::force(self, links)
    }
}

impl<O: ForceOp> MdAction for TwoFlavorRatio<O> {
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        TwoFlavorRatio::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        TwoFlavorRatio::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        TwoFlavorRatio::force(self, links)
    }
}
