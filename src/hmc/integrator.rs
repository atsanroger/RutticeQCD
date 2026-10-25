//! HMC with any actions on several time scales: Bridge++'s `HMC_General` with the integrators
//! of its `Builder_Integrator`.

use super::{LinkField, Links, Mseries, SmearingCache};
use crate::{Field, Gauge};

/// An action in the molecular dynamics, on links of the kind `U` (on the host by default).
pub trait MdAction<U: LinkField = Field<f64, Gauge>> {
    /// The start of a trajectory: draw what the action draws (pseudofermions) for the
    /// links; returns its energy.
    fn langevin(&mut self, links: &Links<U>, rng: &mut Mseries) -> f64;
    /// The energy at the links.
    fn hamiltonian(&mut self, links: &Links<U>) -> f64;
    /// The force on the momenta.
    fn force(&mut self, links: &Links<U>) -> U;
    /// What the solves of the action have cost since the last time this was asked; None for
    /// an action without solves.
    fn take_cost(&mut self) -> Option<SolveCost> {
        None
    }
}

/// What the solves of an action cost (`MdAction::take_cost`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveCost {
    /// Solves: for the forces, the energies and the heat bath (a multi-shift solve is one; a
    /// pole solved on its own is one).
    pub solves: usize,
    /// Iterations of the solvers: CG, GCR on the even sites with the multigrid, the CG in
    /// single precision with mixed precision.
    pub iterations: usize,
    /// Seconds in the solves.
    pub seconds: f64,
    /// The multigrid: seconds of its setups and bootstraps, and of making P and the coarse
    /// parts for new links (both within `seconds`); how often those were made and reused.
    pub(crate) setup_seconds: f64,
    pub(crate) rebuild_seconds: f64,
    pub(crate) rebuilds: usize,
    pub(crate) reuses: usize,
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
pub struct Term<'a, U: LinkField = Field<f64, Gauge>> {
    pub level: usize,
    pub action: Box<dyn MdAction<U> + 'a>,
}

/// HMC with the actions `terms`, in the order Bridge++ lists them (it matters for the order
/// of the random numbers drawn: the momenta first, then the actions in this order).
pub struct Hmc<'a, U: LinkField = Field<f64, Gauge>> {
    pub terms: Vec<Term<'a, U>>,
    pub(crate) scheme: Scheme,
    /// Steps of each level.
    pub steps: Vec<usize>,
    /// Order of the Taylor series of exp(ε iP).
    pub exp_order: usize,
    pub trajectory_length: f64,
    pub metropolis: bool,
    /// The force of each level at the current links.
    cache: Vec<Option<U>>,
    /// The smearings of the current links, shared by the actions.
    smeared: SmearingCache<U>,
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

/// What `Hmc::reversibility` found.
#[derive(Clone, Debug)]
pub struct Reversibility {
    /// ΔH of the trajectory.
    pub delta_h: f64,
    /// H after the trajectory and back − H at the start: zero for a reversible integrator.
    pub(crate) delta_h_back: f64,
    /// The largest difference of a real of the links, and of the momenta, from the start.
    pub(crate) max_link: f64,
    pub(crate) max_momentum: f64,
}

impl<'a, U: LinkField> Hmc<'a, U> {
    pub fn new(
        terms: Vec<Term<'a, U>>,
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
    fn update_p(&mut self, level: usize, eps: f64, p: &mut U, u: &U) {
        if self.cache[level].is_none() {
            let mut sum = u.zeros_like();
            let links = Links::new(u, &self.smeared);
            for t in self.terms.iter_mut().filter(|t| t.level == level) {
                let f = t.action.force(&links);
                sum.add_scaled(1.0, &f);
            }
            self.cache[level] = Some(sum);
        }
        let f = self.cache[level].as_ref().expect("just computed");
        p.add_scaled(eps, f);
    }

    /// Level `level` over the time `tau`.
    fn evolve(&mut self, level: usize, tau: f64, p: &mut U, u: &mut U) {
        if level == self.steps.len() {
            u.exp_update(tau, p, self.exp_order);
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

    /// The reversibility of the molecular dynamics from `u` (which is not changed): momenta and
    /// pseudofermions drawn as for a trajectory, the trajectory, P → −P, the trajectory again,
    /// P → −P. A reversible integrator is back at (U, P) and at H, to rounding and the tolerance
    /// of the solves; the initial guess of the force solves (`Chrono`) breaks it at the level of
    /// that tolerance.
    pub(crate) fn reversibility(&mut self, u: &U, rng: &mut Mseries) -> Reversibility {
        let lat = u.lattice().clone();
        let mut trial = u.duplicate();
        self.invalidate();
        let mut p = U::draw_momenta(&lat, rng);
        let p0 = p.duplicate();
        let energy = |hmc: &mut Self, links: &U, p: &U| -> f64 {
            let l = Links::new(links, &hmc.smeared);
            p.kinetic_energy()
                + hmc
                    .terms
                    .iter_mut()
                    .map(|t| t.action.hamiltonian(&l))
                    .sum::<f64>()
        };
        let h0 = {
            let links = Links::new(&trial, &self.smeared);
            p.kinetic_energy()
                + self
                    .terms
                    .iter_mut()
                    .map(|t| t.action.langevin(&links, rng))
                    .sum::<f64>()
        };
        self.evolve(0, self.trajectory_length, &mut p, &mut trial);
        let h1 = energy(self, &trial, &p);
        let flip = |p: &mut U| {
            let q = p.duplicate();
            p.add_scaled(-2.0, &q);
        };
        flip(&mut p);
        self.evolve(0, self.trajectory_length, &mut p, &mut trial);
        flip(&mut p);
        let h2 = energy(self, &trial, &p);
        let worst = |a: &U, b: &U| {
            a.to_host()
                .as_slice()
                .iter()
                .zip(b.to_host().as_slice())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        self.invalidate();
        Reversibility {
            delta_h: h1 - h0,
            delta_h_back: h2 - h0,
            max_link: worst(&trial, u),
            max_momentum: worst(&p, &p0),
        }
    }

    /// One trajectory from `u`, which becomes the new configuration if it is accepted.
    pub fn trajectory(&mut self, u: &mut U, rng: &mut Mseries) -> HmcTrajectory {
        let lat = u.lattice().clone();
        let mut trial = u.duplicate();
        self.invalidate();
        let mut p = U::draw_momenta(&lat, rng);
        let k0 = p.kinetic_energy();
        let a0: Vec<f64> = {
            let links = Links::new(&trial, &self.smeared);
            self.terms
                .iter_mut()
                .map(|t| t.action.langevin(&links, rng))
                .collect()
        };
        self.evolve(0, self.trajectory_length, &mut p, &mut trial);
        let k1 = p.kinetic_energy();
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
        let plaquette_trial = trial.plaquette();
        if accepted {
            *u = trial;
        }
        HmcTrajectory {
            kinetic: [k0, k1],
            actions: a0.into_iter().zip(a1).map(|(a, b)| [a, b]).collect(),
            delta_h,
            random,
            accepted,
            plaquette_trial,
            plaquette: u.plaquette(),
        }
    }
}
