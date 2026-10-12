//! Fractional powers of the fermion determinant by rational approximation (RHMC): one
//! flavour of Wilson-type fermions is det(D†D)^(1/2), the strange quark of 2 + 1 flavours.
//! Bridge++'s `Action_F_Rational` with `Fopr_Rational` and `Force_F_Rational`:
//!
//! The coefficients are those of the minimax relative approximation (`RationalApprox`), on
//! an interval [λ_min², λ_max²] that must hold the spectrum of D†D; Bridge++ gives the bounds
//! λ of |D| and reads its coefficients from AlgRemez's files, which `RationalApprox::minimax`
//! reproduces to the last digit.

use super::fermion_action::{draw, full_derivative, Psi};
use super::solve::Shifted;
use super::{smeared_force, ForceOp, Links, Mseries};
use crate::fermion::{Bound, Variant};
use crate::gauge::GaugeSmearing;
use crate::numerics::RationalApprox;
use crate::solver::{Cg, MultiShiftCg, SolveError, Solver, Stop};
use crate::{Field, Gauge, Normal, Vector, C64};
use std::rc::Rc;

/// S = φ† r_md(D†D) φ, r_md ≈ (D†D)^(−a), φ drawn as r_hb(D†D) ξ, r_hb ≈ (D†D)^(a/2), with D
/// the operator `op` on the links smeared by `smearing`.
pub struct Rational<O> {
    pub op: O,
    pub smearing: Option<GaugeSmearing>,
    /// The approximation of the action and the force, and that of the heat bath.
    pub md: RationalApprox,
    pub heat_bath: RationalApprox,
    /// The multi-shift solves for the force, and for the heat bath and the energy.
    pub stop_md: Stop,
    pub stop_h: Stop,
    /// One CG solve per pole instead of the multi-shift solver.
    pub pole_by_pole: bool,
    phi: Option<Psi>,
    /// κ c_SW when the operator has a clover term.
    clover: Option<f64>,
    /// Iterations of the last multi-shift solve.
    pub last_iterations: usize,
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
            phi: None,
            clover: None,
            last_iterations: 0,
        }
    }

    /// Solve for every pole with a CG of its own (each to the tolerance on its own residual).
    pub fn with_pole_by_pole(mut self, on: bool) -> Self {
        self.pole_by_pole = on;
        self
    }

    /// The operator has the clover term with coefficient κ c_SW: its force is added.
    pub fn with_clover(mut self, kappa_csw: f64) -> Self {
        self.clover = Some(kappa_csw);
        self
    }

    fn set_links(&mut self, links: &Links) -> Rc<Vec<Field<f64, Gauge>>> {
        let levels = links.levels(self.smearing.as_ref());
        self.op
            .set_gauge(levels.last().unwrap_or(links.thin))
            .expect("the links suit the operator");
        levels
    }

    /// x_k = (D†D + pole_k)⁻¹ b for the poles of `r`.
    fn shifted(&mut self, r: &RationalApprox, b: &Psi, stop: Stop) -> Result<Vec<Psi>, SolveError> {
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
                let info = Cg::new(stop).solve(&mut shifted, None, &mut x, b)?;
                self.last_iterations += info.iterations;
                xs.push(x);
            }
            return Ok(xs);
        }
        let info = MultiShiftCg::new(stop).solve(&mut normal, &r.poles, &mut xs, b)?;
        self.last_iterations = info.iterations;
        Ok(xs)
    }

    /// r(D†D) b = a0 b + Σ_k res_k x_k, in Bridge++'s order of the sum.
    fn apply(&mut self, r: &RationalApprox, b: &Psi, stop: Stop) -> Psi {
        let xs = self
            .shifted(r, b, stop)
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
    pub fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let mut xi = self.op.new_field(links.thin.lattice());
        draw(&mut xi, rng);
        let r = self.heat_bath.clone();
        self.phi = Some(self.apply(&r, &xi, self.stop_h));
        let mut n = [0.0];
        xi.norm_sqr(&mut n);
        n[0]
    }

    /// S = φ† r_md(D†D) φ.
    pub fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.set_links(links);
        let phi = self.phi.take().expect("pseudofermions drawn");
        let r = self.md.clone();
        let v = self.apply(&r, &phi, self.stop_h);
        let mut d = [C64::ZERO];
        v.dot(&phi, &mut d);
        self.phi = Some(phi);
        d[0].re
    }

    pub fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        let levels = self.set_links(links);
        let phi = self.phi.take().expect("pseudofermions drawn");
        let r = self.md.clone();
        let xs = self
            .shifted(&r, &phi, self.stop_md)
            .expect("the multi-shift solve for the force converges");
        self.phi = Some(phi);
        let v = levels.last().unwrap_or(links.thin);
        let mut m = Field::<f64, Gauge>::gauge(links.thin.lattice());
        let mut work = self.op.work();
        for (x, res) in xs.iter().zip(&r.residues) {
            let mut dx = x.zeros_like();
            self.op.mult(&mut work, &mut dx, x, Variant::D);
            let mk = full_derivative(&self.op, self.clover, v, &dx, x);
            for (a, b) in m.as_mut_slice().iter_mut().zip(mk.as_slice()) {
                *a += res * b;
            }
        }
        smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

impl<O: ForceOp> super::MdAction for Rational<O> {
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        Rational::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        Rational::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        Rational::force(self, links)
    }
}
