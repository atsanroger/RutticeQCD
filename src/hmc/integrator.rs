//! HMC with any actions on several time scales: Bridge++'s `HMC_General` with the integrators
//! of its `Builder_Integrator`.
//!
//! The integrator has levels 0 … n−1: level ℓ makes `steps[ℓ]` steps of leapfrog or Omelyan
//! over the time it is given, updating the momenta with the forces of the actions on level ℓ;
//! its update of the links is the whole of level ℓ + 1 over the step (below the last level, the
//! links themselves: U ← exp(ε iP) U). Level 0 runs over the trajectory length. The force of a
//! level is kept until the links change, as Bridge++ does (two half steps in a row cost one).

use super::{
    draw_momenta, exp_update, kinetic_energy, Links, Mseries, PlaquetteAction, RectangleAction,
    SmearingCache,
};
use crate::gauge::plaquette;
use crate::{Boundary, Field, Gauge, Stencil, ND};

/// An action in the molecular dynamics.
pub trait MdAction {
    /// The start of a trajectory: draw what the action draws (pseudofermions) for the
    /// links; returns its energy.
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64;
    /// The energy at the links.
    fn hamiltonian(&mut self, links: &Links) -> f64;
    /// The force on the momenta.
    fn force(&mut self, links: &Links) -> Field<f64, Gauge>;
}

impl MdAction for PlaquetteAction {
    fn langevin(&mut self, links: &Links, _rng: &mut Mseries) -> f64 {
        self.action(links.thin)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.action(links.thin)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        PlaquetteAction::force(self, links.thin)
    }
}

impl MdAction for RectangleAction {
    fn langevin(&mut self, links: &Links, _rng: &mut Mseries) -> f64 {
        self.action(links.thin)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.action(links.thin)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        RectangleAction::force(self, links.thin)
    }
}

impl<O: super::ForceOp> MdAction for super::TwoFlavor<O> {
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        super::TwoFlavor::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        super::TwoFlavor::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        super::TwoFlavor::force(self, links)
    }
}

/// The scheme of every level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scheme {
    /// P(ε/2) [U(ε) P(ε)]… U(ε) P(ε/2).
    Leapfrog,
    /// P(λε) [U(ε/2) P((1−2λ)ε) U(ε/2) P(2λε)]… with P(λε) at the end (Omelyan, λ ≈ 0.193).
    Omelyan { lambda: f64 },
}

/// One action and the level of the integrator it is on.
pub struct Term<'a> {
    pub level: usize,
    pub action: Box<dyn MdAction + 'a>,
}

/// HMC with the actions `terms`, in the order Bridge++ lists them (it matters for the order
/// of the random numbers drawn: the momenta first, then the actions in this order).
pub struct Hmc<'a> {
    pub terms: Vec<Term<'a>>,
    pub scheme: Scheme,
    /// Steps of each level.
    pub steps: Vec<usize>,
    /// Order of the Taylor series of exp(ε iP).
    pub exp_order: usize,
    pub trajectory_length: f64,
    pub metropolis: bool,
    /// The force of each level at the current links.
    cache: Vec<Option<Field<f64, Gauge>>>,
    /// The smearings of the current links, shared by the actions.
    smeared: SmearingCache,
}

/// What one trajectory did.
#[derive(Clone, Debug)]
pub struct HmcTrajectory {
    /// ½|iP|² and the energy of each action, at the start and at the end.
    pub kinetic: [f64; 2],
    pub actions: Vec<[f64; 2]>,
    /// H(end) − H(start).
    pub delta_h: f64,
    /// The number compared with exp(−ΔH) (−1 without the Metropolis test).
    pub random: f64,
    pub accepted: bool,
    /// The plaquette at the end of the trajectory and after the test.
    pub plaquette_trial: f64,
    pub plaquette: f64,
}

impl<'a> Hmc<'a> {
    pub fn new(
        terms: Vec<Term<'a>>,
        scheme: Scheme,
        steps: Vec<usize>,
        exp_order: usize,
        trajectory_length: f64,
    ) -> Self {
        assert!(
            terms.iter().all(|t| t.level < steps.len()),
            "a level without steps"
        );
        let levels = steps.len();
        Self {
            terms,
            scheme,
            steps,
            exp_order,
            trajectory_length,
            metropolis: true,
            cache: (0..levels).map(|_| None).collect(),
            smeared: SmearingCache::new(),
        }
    }

    fn invalidate(&mut self) {
        self.cache.iter_mut().for_each(|c| *c = None);
        self.smeared.clear();
    }

    /// P ← P + ε F_ℓ(U).
    fn update_p(
        &mut self,
        level: usize,
        eps: f64,
        p: &mut Field<f64, Gauge>,
        u: &Field<f64, Gauge>,
    ) {
        if self.cache[level].is_none() {
            let mut sum = Field::<f64, Gauge>::gauge(u.lattice());
            let links = Links::new(u, &self.smeared);
            for t in self.terms.iter_mut().filter(|t| t.level == level) {
                let f = t.action.force(&links);
                for (s, v) in sum.as_mut_slice().iter_mut().zip(f.as_slice()) {
                    *s += v;
                }
            }
            self.cache[level] = Some(sum);
        }
        let f = self.cache[level].as_ref().expect("just computed");
        for (pv, fv) in p.as_mut_slice().iter_mut().zip(f.as_slice()) {
            *pv += eps * fv;
        }
    }

    /// Level `level` over the time `tau`.
    fn evolve(
        &mut self,
        level: usize,
        tau: f64,
        p: &mut Field<f64, Gauge>,
        u: &mut Field<f64, Gauge>,
    ) {
        if level == self.steps.len() {
            exp_update(tau, p, u, self.exp_order);
            self.invalidate();
            return;
        }
        let n = self.steps[level];
        if n == 0 {
            return;
        }
        let eps = tau / n as f64;
        match self.scheme {
            Scheme::Leapfrog => {
                self.update_p(level, 0.5 * eps, p, u);
                for i in 1..=n {
                    self.evolve(level + 1, eps, p, u);
                    if i < n {
                        self.update_p(level, eps, p, u);
                    }
                }
                self.update_p(level, 0.5 * eps, p, u);
            }
            Scheme::Omelyan { lambda } => {
                self.update_p(level, eps * lambda, p, u);
                for i in 1..=n {
                    self.evolve(level + 1, eps * 0.5, p, u);
                    self.update_p(level, eps * (1.0 - 2.0 * lambda), p, u);
                    self.evolve(level + 1, eps * 0.5, p, u);
                    if i < n {
                        self.update_p(level, eps * lambda * 2.0, p, u);
                    }
                }
                self.update_p(level, eps * lambda, p, u);
            }
        }
    }

    /// One trajectory from `u`, which becomes the new configuration if it is accepted.
    pub fn trajectory(&mut self, u: &mut Field<f64, Gauge>, rng: &mut Mseries) -> HmcTrajectory {
        let lat = u.lattice().clone();
        let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
        let mut trial = Field::<f64, Gauge>::gauge(&lat);
        trial.as_mut_slice().copy_from_slice(u.as_slice());
        self.invalidate();
        let mut p = draw_momenta(&lat, rng);
        let k0 = kinetic_energy(&p);
        let a0: Vec<f64> = {
            let links = Links::new(&trial, &self.smeared);
            self.terms
                .iter_mut()
                .map(|t| t.action.langevin(&links, rng))
                .collect()
        };
        self.evolve(0, self.trajectory_length, &mut p, &mut trial);
        let k1 = kinetic_energy(&p);
        let a1: Vec<f64> = {
            let links = Links::new(&trial, &self.smeared);
            self.terms
                .iter_mut()
                .map(|t| t.action.hamiltonian(&links))
                .collect()
        };
        let h0 = k0 + a0.iter().sum::<f64>();
        let h1 = k1 + a1.iter().sum::<f64>();
        let delta_h = h1 - h0;
        let random = if self.metropolis { rng.uniform() } else { -1.0 };
        let accepted = random <= (-delta_h).exp();
        let plaquette_trial = plaquette(&trial, &st);
        if accepted {
            u.as_mut_slice().copy_from_slice(trial.as_slice());
        }
        HmcTrajectory {
            kinetic: [k0, k1],
            actions: a0.into_iter().zip(a1).map(|(a, b)| [a, b]).collect(),
            delta_h,
            random,
            accepted,
            plaquette_trial,
            plaquette: plaquette(u, &st),
        }
    }
}

impl<O: super::ForceOp> MdAction for super::TwoFlavorRatio<O> {
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        super::TwoFlavorRatio::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        super::TwoFlavorRatio::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        super::TwoFlavorRatio::force(self, links)
    }
}
