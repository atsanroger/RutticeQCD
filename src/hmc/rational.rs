//! Fractional powers of the fermion determinant by rational approximation (RHMC): one
//! flavour of Wilson-type fermions is det(D†D)^(1/2), the strange quark of 2 + 1 flavours.
//! Bridge++'s `Action_F_Rational` with `Fopr_Rational` and `Force_F_Rational`:
//!
//! The coefficients are those of the minimax relative approximation (`RationalApprox`), on
//! an interval [λ_min², λ_max²] that must hold the spectrum of D†D; Bridge++ gives the bounds
//! λ of |D| and reads its coefficients from AlgRemez's files, which `RationalApprox::minimax`
//! reproduces to the last digit.

use super::fermion_action::Gamma5;
use super::solve::{Hermitian, Shifted};
use super::{ForceOp, LinkField, Links, Mseries};
use super::{NormalSolver, Preconditioner};
use crate::fermion::{Bound, Variant};
use crate::gauge::GaugeSmearing;
use crate::numerics::RationalApprox;
use crate::solver::{Cg, MultiShiftCg, SolveError, Solver, Stop};
use crate::{Normal, Vector, C64};
use std::rc::Rc;

/// S = φ† r_md(D†D) φ, r_md ≈ (D†D)^(−a), φ drawn as r_hb(D†D) ξ, r_hb ≈ (D†D)^(a/2), with D
/// the operator `op` on the links smeared by `smearing`.
pub struct Rational<O: ForceOp> {
    pub op: O,
    pub smearing: Option<GaugeSmearing>,
    /// The approximation of the action and the force, and that of the heat bath.
    pub md: RationalApprox,
    pub(crate) heat_bath: RationalApprox,
    /// The multi-shift solves for the force, and for the heat bath and the energy.
    pub(crate) stop_md: Stop,
    pub(crate) stop_h: Stop,
    /// One CG solve per pole instead of the multi-shift solver.
    pub(crate) pole_by_pole: bool,
    /// Solve every pole by two solves with D ± i√pole γ5 (`with_two_step`).
    two_step: Option<NormalSolver>,
    /// The multigrid of the two-step solves; none otherwise.
    pub preconditioner: Preconditioner,
    fresh: bool,
    phi: Option<O::Field>,
    /// κ c_SW when the operator has a clover term.
    clover: Option<f64>,
    /// Iterations of the last multi-shift solve.
    pub(crate) last_iterations: usize,
    meter: super::solve::CostMeter,
}

impl<O: ForceOp> Rational<O> {
    pub fn new(
        op: O,
        smearing: Option<GaugeSmearing>,
        md: RationalApprox,
        heat_bath: RationalApprox,
        stop_md: Stop,
        stop_h: Stop,
    ) -> Self {
        Self {
            op,
            smearing,
            md,
            heat_bath,
            stop_md,
            stop_h,
            pole_by_pole: false,
            two_step: None,
            preconditioner: Preconditioner::None,
            fresh: true,
            phi: None,
            clover: None,
            last_iterations: 0,
            meter: Default::default(),
        }
    }

    /// Solve for every pole with a CG of its own (each to the tolerance on its own residual).
    pub(crate) fn with_pole_by_pole(mut self, on: bool) -> Self {
        self.pole_by_pole = on;
        self
    }

    /// Solve pole by pole with `solver` (not CG) through the Hermitian γ5 D:
    /// D†D + σ = γ5 (D + i√σ γ5) γ5 (D − i√σ γ5), two solves that take the multigrid of D as
    /// preconditioner (`with_preconditioner`).
    pub(crate) fn with_two_step(mut self, solver: NormalSolver) -> Self {
        self.two_step = Some(solver);
        self
    }

    /// The multigrid of the two-step solves.
    pub(crate) fn with_preconditioner(mut self, preconditioner: Preconditioner) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    /// The operator has the clover term with coefficient κ c_SW: its force is added.
    pub fn with_clover(mut self, kappa_csw: f64) -> Self {
        self.clover = Some(kappa_csw);
        self
    }

    fn set_links(&mut self, links: &Links<O::Links>) -> Rc<Vec<O::Links>> {
        let levels = links.levels(self.smearing.as_ref());
        self.op
            .set_links(levels.last().unwrap_or(links.thin))
            .expect("the links suit the operator");
        levels
    }

    /// x_k = (D†D + pole_k)⁻¹ b for the poles of `r`; `md`: for the force.
    fn shifted(
        &mut self,
        r: &RationalApprox,
        b: &O::Field,
        stop: Stop,
        md: bool,
    ) -> Result<Vec<O::Field>, SolveError> {
        if let Some(solver) = self.two_step {
            let fresh = std::mem::take(&mut self.fresh);
            let mut cycle = match &mut self.preconditioner {
                Preconditioner::Multigrid(setup) => self.op.multigrid(setup, fresh, md),
                _ => None,
            };
            let mut d = Bound::new(&self.op, Variant::D);
            let mut gamma = Gamma5 {
                op: &self.op,
                work: self.op.work(),
            };
            let mut xs = Vec::new();
            self.last_iterations = 0;
            for pole in &r.poles {
                let mut x = b.zeros_like();
                let hermitian = Hermitian {
                    gamma: &mut gamma,
                    m: cycle
                        .as_mut()
                        .map(|m| &mut **m as &mut dyn crate::LinearFn<O::Field>),
                };
                let start = std::time::Instant::now();
                let info = solver.shifted(&mut d, hermitian, pole.sqrt(), &mut x, b, stop)?;
                self.meter.add(start, info.iterations);
                self.last_iterations += info.iterations;
                xs.push(x);
            }
            return Ok(xs);
        }
        let mut d = Bound::new(&self.op, Variant::D);
        let mut normal = Normal::new(&mut d, b);
        let mut xs = Vec::new();
        if self.pole_by_pole {
            self.last_iterations = 0;
            for pole in &r.poles {
                let mut x = b.zeros_like();
                let mut shifted = Shifted {
                    a: &mut normal,
                    shift: *pole,
                };
                let start = std::time::Instant::now();
                let info = Cg::new(stop).solve(&mut shifted, None, &mut x, b)?;
                self.meter.add(start, info.iterations);
                self.last_iterations += info.iterations;
                xs.push(x);
            }
            return Ok(xs);
        }
        let start = std::time::Instant::now();
        let info = MultiShiftCg::new(stop).solve(&mut normal, &r.poles, &mut xs, b)?;
        self.meter.add(start, info.iterations);
        self.last_iterations = info.iterations;
        Ok(xs)
    }

    /// r(D†D) b = a0 b + Σ_k res_k x_k, in Bridge++'s order of the sum.
    fn apply(&mut self, r: &RationalApprox, b: &O::Field, stop: Stop) -> O::Field {
        let xs = self
            .shifted(r, b, stop, false)
            .expect("the multi-shift solve converges");
        let mut out = b.zeros_like();
        out.copy_from(b);
        out.scale(&[r.norm]);
        for (x, res) in xs.iter().zip(&r.residues) {
            out.axpy(&[C64::new(*res, 0.0)], x);
        }
        out
    }

    /// The heat bath: φ = r_hb(D†D) ξ, ξ Gaussian (variance ½ per real) in Bridge++'s order;
    /// returns |ξ|².
    pub fn langevin(&mut self, links: &Links<O::Links>, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let mut xi = self.op.new_field(LinkField::lattice(links.thin));
        self.op.draw(&mut xi, rng);
        let r = self.heat_bath.clone();
        self.fresh = true;
        self.phi = Some(self.apply(&r, &xi, self.stop_h));
        let mut n = [0.0];
        xi.norm_sqr(&mut n);
        n[0]
    }

    /// S = φ† r_md(D†D) φ.
    pub fn hamiltonian(&mut self, links: &Links<O::Links>) -> f64 {
        self.set_links(links);
        let phi = self.phi.take().expect("pseudofermions drawn");
        let r = self.md.clone();
        let v = self.apply(&r, &phi, self.stop_h);
        let mut d = [C64::ZERO];
        v.dot(&phi, &mut d);
        self.phi = Some(phi);
        d[0].re
    }

    /// Σ_k res_k M(D x_k, x_k), the clover term's part of all poles walked once.
    pub fn force(&mut self, links: &Links<O::Links>) -> O::Links {
        let levels = self.set_links(links);
        let phi = self.phi.take().expect("pseudofermions drawn");
        let r = self.md.clone();
        let v = levels.last().unwrap_or(links.thin);
        let xs = self
            .shifted(&r, &phi, self.stop_md, true)
            .expect("the multi-shift solve for the force converges");
        self.phi = Some(phi);
        let mut m = links.thin.zeros_like();
        let mut work = self.op.work();
        let mut dxs = Vec::with_capacity(xs.len());
        for (x, res) in xs.iter().zip(&r.residues) {
            let mut dx = x.zeros_like();
            self.op.mult(&mut work, &mut dx, x, Variant::D);
            m.add_scaled(*res, &self.op.derivative(&dx, x));
            dxs.push(dx);
        }
        if let Some(kc) = self.clover {
            let pairs: Vec<_> = (r.residues.iter().zip(&dxs).zip(&xs))
                .map(|((res, dx), x)| (*res, dx, x))
                .collect();
            m.add_scaled(1.0, &self.op.clover_derivative(kc, v, &pairs));
        }
        O::Links::smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

impl<O: ForceOp> super::MdAction<O::Links> for Rational<O> {
    fn langevin(&mut self, links: &Links<O::Links>, rng: &mut Mseries) -> f64 {
        Rational::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links<O::Links>) -> f64 {
        Rational::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links<O::Links>) -> O::Links {
        Rational::force(self, links)
    }
    fn take_cost(&mut self) -> Option<super::SolveCost> {
        Some(self.meter.take(&self.preconditioner))
    }
}
