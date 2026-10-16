//! The fields HMC moves: the links, and the fields of matrices that go with them (momenta,
//! forces, derivatives), kept on the host (`Field<f64, Gauge>`) or elsewhere (a device), with
//! what the integrator, the gauge actions and the smeared fermion actions do with them. The
//! integrator and the actions are written once over this trait.

use super::{draw_momenta, exp_update, kinetic_energy, smeared_force, Mseries};
use super::{PlaquetteAction, RectangleAction};
use crate::gauge::{plaquette, GaugeSmearing};
use crate::{Boundary, Field, Gauge, Lattice, Stencil, ND};
use std::sync::Arc;

/// A field of 3 × 3 complex matrices, one for each link.
pub trait LinkField: Sized + 'static {
    fn lattice(&self) -> &Arc<Lattice>;
    /// A zero field of the same lattice.
    fn zeros_like(&self) -> Self;
    fn duplicate(&self) -> Self;
    /// self += a x.
    fn add_scaled(&mut self, a: f64, x: &Self);
    /// ½ |iP|² of momenta `self`, summed over all processes.
    fn kinetic_energy(&self) -> f64;
    /// U ← exp(ε iP) U for the links `self` (`exp_update`).
    fn exp_update(&mut self, eps: f64, p: &Self, order: usize);
    /// The average plaquette of the links `self`.
    fn plaquette(&self) -> f64;
    /// Momenta drawn from `rng` as Bridge++ draws them (`draw_momenta`).
    fn draw_momenta(lat: &Arc<Lattice>, rng: &mut Mseries) -> Self;
    /// The fields after each step of `smearing` of the links `self`.
    fn smear_levels(&self, smearing: &GaugeSmearing) -> Vec<Self>;
    /// The force of an action whose derivative `m_v` for the smeared links is known, carried
    /// back to the links `u` and projected (`smeared_force`).
    fn smeared_force(
        m_v: &Self,
        u: &Self,
        smearing: Option<&GaugeSmearing>,
        levels: &[Self],
    ) -> Self;
    /// S of the gauge action of plaquettes and rectangles at the links `self`
    /// (`RectangleAction::action`; `PlaquetteAction::action` for c_rect = 0, c_plaq = 1).
    fn gauge_action(&self, beta: f64, c_plaq: f64, c_rect: f64) -> f64;
    /// Its force (`RectangleAction::force`, `PlaquetteAction::force`).
    fn gauge_force(&self, beta: f64, c_plaq: f64, c_rect: f64) -> Self;
    /// The field on the host, and from it.
    fn to_host(&self) -> Field<f64, Gauge>;
    fn from_host(u: &Field<f64, Gauge>) -> Self;
}

impl LinkField for Field<f64, Gauge> {
    fn lattice(&self) -> &Arc<Lattice> {
        Field::lattice(self)
    }
    fn zeros_like(&self) -> Self {
        Field::<f64, Gauge>::gauge(Field::lattice(self))
    }
    fn duplicate(&self) -> Self {
        let mut v = LinkField::zeros_like(self);
        v.as_mut_slice().copy_from_slice(self.as_slice());
        v
    }
    fn add_scaled(&mut self, a: f64, x: &Self) {
        for (s, v) in self.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *s += a * v;
        }
    }
    fn kinetic_energy(&self) -> f64 {
        kinetic_energy(self)
    }
    fn exp_update(&mut self, eps: f64, p: &Self, order: usize) {
        exp_update(eps, p, self, order);
    }
    fn plaquette(&self) -> f64 {
        let st = Stencil::nearest(Field::lattice(self), [Boundary::Periodic; ND]);
        plaquette(self, &st)
    }
    fn draw_momenta(lat: &Arc<Lattice>, rng: &mut Mseries) -> Self {
        draw_momenta(lat, rng)
    }
    fn smear_levels(&self, smearing: &GaugeSmearing) -> Vec<Self> {
        smearing.levels(self)
    }
    fn smeared_force(
        m_v: &Self,
        u: &Self,
        smearing: Option<&GaugeSmearing>,
        levels: &[Self],
    ) -> Self {
        smeared_force(m_v, u, smearing, levels)
    }
    fn gauge_action(&self, beta: f64, c_plaq: f64, c_rect: f64) -> f64 {
        if c_rect == 0.0 && c_plaq == 1.0 {
            PlaquetteAction { beta }.action(self)
        } else {
            RectangleAction {
                beta,
                c_plaq,
                c_rect,
            }
            .action(self)
        }
    }
    fn gauge_force(&self, beta: f64, c_plaq: f64, c_rect: f64) -> Self {
        if c_rect == 0.0 && c_plaq == 1.0 {
            PlaquetteAction { beta }.force(self)
        } else {
            RectangleAction {
                beta,
                c_plaq,
                c_rect,
            }
            .force(self)
        }
    }
    fn to_host(&self) -> Field<f64, Gauge> {
        self.duplicate()
    }
    fn from_host(u: &Field<f64, Gauge>) -> Self {
        u.duplicate()
    }
}
