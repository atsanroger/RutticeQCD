//! Hybrid Monte Carlo from a parameter file ([`super::hmc_params`] has its sections): the start
//! configuration, the actions on the levels of the integrator, the trajectories, and where the
//! configurations go. The structures here are what the run is made from.
//!
//! The conventions are those of Bridge++'s `HMC_General` (momenta, pseudofermions and the
//! Metropolis test from one M-series generator; the integrator levels of its
//! `Builder_Integrator`), so a run can be repeated there.

use super::config::{BoundaryConfig, GaugeFormat, GaugeSmearingConfig};
use super::hmc_params::{is_hmc_text, HmcParams};
use super::CliError;
use crate::fermion::{Clover, CloverParams, Mdwf, MdwfParams, Wilson, WilsonParams};
use crate::gauge::flow::{flow_to_t0, AtT0, WilsonFlow};
use crate::gauge::GaugeSmearing;
use crate::gauge::{plaquette, random_gauge};
use crate::hmc::{
    cold_start, EvenOddForceOp, EvenOddHalves, ForceOp, Hmc, LinkField, MdAction, Mseries,
    MultigridSetup, NormalSolver, PlaquetteAction, Preconditioner, Rational, RectangleAction,
    Scheme, Term, TwoFlavor, TwoFlavorEo, TwoFlavorRatio, TwoFlavorRatioEo,
};
use crate::hmc::{DwfMultigridSetup, HalfPsi, Single, SingleMdwf, SingleMdwfEo, SingleNormal};
use crate::kernel::Dirac;
use crate::numerics::{RationalApprox, RationalSpec};
use crate::solver::preconditioner::multigrid::{
    BootstrapParams, LowModeParams, MultigridParams, PvEoParams,
};
use crate::solver::Stop;
use crate::{Boundary, Comm, Field, Gauge, Lattice, Stencil, ND};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct HmcConfig {
    pub lattice: [usize; ND],
    pub process_grid: Option<[usize; ND]>,
    /// Threads of each process; without it `RAYON_NUM_THREADS`, or every core.
    pub cpu_thread: Option<usize>,
    pub start: StartConfig,
    /// The smearing of the links of the fermion actions that ask for it (`smeared`).
    pub gauge_smearing: Option<GaugeSmearingConfig>,
    pub hmc: HmcBlock,
    pub actions: Vec<ActionConfig>,
    /// The section of each action in the file (`Action_G`, `Action_F_1`, ...), for messages.
    pub action_names: Vec<String>,
    pub output: HmcOutput,
    /// The Wilson flow measured along the run.
    pub flow: Option<FlowMonitor>,
    /// The domain-wall operators of the fermion actions on the device, with their solves and
    /// the derivative of the actions (`Main.backend : cuda_bridge`; the Ratio action only).
    pub device: bool,
}

/// t0 and the topological charge by the Wilson flow of a copy of the configuration, after every
/// `interval` trajectories; at the end, the Γ-method for their series.
#[derive(Debug, Clone, Copy)]
pub struct FlowMonitor {
    pub interval: usize,
    pub flow: WilsonFlow,
    /// The flow stops here if t² E has not reached 0.3.
    pub t_max: f64,
    /// The first this many trajectories of the run (thermalization) are measured and printed
    /// but left out of the Γ-method at the end.
    pub skip: usize,
}

/// The configuration the run starts from.
#[derive(Debug, Clone)]
pub enum StartConfig {
    /// Read from a file; relative paths from the parameter file's directory.
    File { file: PathBuf, format: GaugeFormat },
    /// Every link the identity.
    Cold,
    /// Random links.
    Hot { seed: u64 },
}

#[derive(Debug, Clone)]
pub struct HmcBlock {
    pub trajectories: usize,
    /// The number of the first trajectory: the random numbers of the run are those of the
    /// M-series generator made from it (Bridge++'s `trajectory_number`).
    pub first_trajectory: u32,
    pub trajectory_length: f64,
    pub integrator: IntegratorConfig,
    pub metropolis: bool,
    /// Checks of the reversibility before the trajectories (`HMC_General.reversibility_checks`).
    pub reversibility_checks: usize,
}

#[derive(Debug, Clone)]
pub struct IntegratorConfig {
    pub scheme: SchemeConfig,
    /// Steps of each level, the outermost first.
    pub steps: Vec<usize>,
    /// λ of the Omelyan scheme.
    pub lambda: f64,
    /// Order of the Taylor series of exp(ε iP).
    pub exp_order: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeConfig {
    Leapfrog,
    Omelyan,
}

/// An action and the level of the integrator it is on.
#[derive(Debug, Clone)]
pub enum ActionConfig {
    Plaquette {
        beta: f64,
        level: usize,
    },
    /// Plaquettes and 1 × 2 rectangles (Iwasaki: c_plaq = 3.648, c_rect = −0.331).
    Rectangle {
        beta: f64,
        c_plaq: f64,
        c_rect: f64,
        level: usize,
    },
    /// Two flavours, S = φ† (D†D)⁻¹ φ.
    TwoFlavor {
        fermion: HmcFermion,
        level: usize,
        /// On the links smeared by `gauge_smearing`.
        smeared: bool,
        /// Even-odd preconditioned: φ on the even sites, S = φ_e† (Ŝ†Ŝ)⁻¹ φ_e − 2 ln |det T_oo|
        /// with Ŝ the Schur complement (`hmc::TwoFlavorEo`; Bridge++'s `Action_F_alt_Standard_eo`
        /// for Wilson fermions).
        even_odd: bool,
        solver: HmcSolver,
    },
    /// Two flavours over two of `denominator` (Hasenbusch), S = φ† H_d (H†H)⁻¹ H_d† φ; for
    /// domain-wall fermions without `denominator`, over two Pauli-Villars flavours.
    TwoFlavorRatio {
        fermion: HmcFermion,
        denominator: Option<HmcFermion>,
        level: usize,
        smeared: bool,
        /// Even-odd preconditioned, H = Γ5 T_ee⁻¹ Ŝ on the even sites (`hmc::TwoFlavorRatioEo`;
        /// Bridge++'s `Action_F_alt_Ratio_eo`): Wilson and domain-wall fermions.
        even_odd: bool,
        solver: HmcSolver,
    },
    /// det(D†D)^(p/q) by rational approximation (RHMC), p/q = `power` (one flavour: [1, 2]): S = φ†
    /// r(D†D) φ, r ≈ x^(−p/q), φ = r'(D†D) ξ, r' ≈ x^(p/2q), minimax with `poles` poles on [lower²,
    /// upper²], `bounds` = [lower, upper] those of the eigenvalues of |D| = √(D†D), as Bridge++'s
    /// `lower_bound` and `upper_bound`.
    Rational {
        fermion: HmcFermion,
        level: usize,
        poles: usize,
        bounds: [f64; 2],
        power: [i32; 2],
        coefficients: Option<PathBuf>,
        smeared: bool,
        solver: HmcSolver,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HmcFermion {
    Wilson {
        kappa: f64,
        boundary: [BoundaryConfig; ND],
    },
    Clover {
        kappa: f64,
        csw: f64,
        boundary: [BoundaryConfig; ND],
    },
    /// Möbius domain-wall fermions (b = 1, c = 0: Shamir's), the normalisation of Bridge++.
    DomainWall {
        ls: usize,
        m5: f64,
        mass: f64,
        b: f64,
        c: f64,
        boundary: [BoundaryConfig; ND],
    },
}

/// The solver of an action's forces and energies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HmcSolverKind {
    /// CG on D†D: the default of the two-flavour actions.
    Cg,
    /// D† y = φ and then D x = y, by BiCGStab.
    Bicgstab,
    /// The same by FGMRES restarted after `restart` steps.
    Fgmres,
    /// The same by GCR truncated to `restart` directions.
    Gcr,
    /// Multi-shift CG: the default of `rational`, which with `cg` solves pole by pole and with
    /// the two-step solvers pole by pole through γ5 D (D ± i√pole γ5).
    MultishiftCg,
}

/// The preconditioner of an action's solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HmcPreconditioner {
    None,
    /// A Chebyshev polynomial of D†D (`chebyshev`), with `cg`; any fermions.
    Chebyshev,
    /// The multigrid (`multigrid`), with `bicgstab`, `fgmres` or `gcr`; Wilson and clover
    /// fermions.
    Multigrid,
}

#[derive(Debug, Clone, Copy)]
pub struct HmcChebyshev {
    /// Steps of the Chebyshev iteration: a polynomial of degree `steps` − 1.
    pub steps: usize,
    /// The interval is [hi/ratio, hi], hi the largest eigenvalue of D†D.
    pub ratio: f64,
}

impl Default for HmcChebyshev {
    fn default() -> Self {
        Self {
            steps: chebyshev_steps(),
            ratio: chebyshev_ratio(),
        }
    }
}

/// The multigrid of an action's solves.
#[derive(Debug, Clone, Copy)]
pub enum HmcMultigrid {
    /// Wilson and clover fermions (`hmc::MultigridSetup`): set up again after `refresh_every`
    /// force solves (0: when a trajectory starts), from the test vectors of the setup before
    /// improved `reuse` times, or from random vectors.
    Wilson {
        params: MultigridParams,
        refresh_every: usize,
        reuse: Option<usize>,
    },
    /// Domain-wall fermions over Pauli-Villars fields (`hmc::DwfMultigridSetup`): `test_vectors`
    /// kept (P has twice as many columns), each solve on the even sites to
    /// `inner_tolerance_squared`.
    DomainWall {
        block: [usize; ND],
        test_vectors: usize,
        cycle: PvEoParams,
        low: LowModeParams,
        boot: BootstrapParams,
        refresh_every: usize,
        inner_tolerance_squared: f64,
        /// The coarse solves in double precision (single by default).
        coarse_double: bool,
        /// Keep P and the coarse parts until the iterations grow this much; 0: never kept.
        rebuild_growth: f64,
        /// The cycle in single precision (on the device).
        fine_single: bool,
        /// Levels below the coarse lattice (on the device).
        levels: crate::hmc::CoarseLevels,
    },
}

impl HmcMultigrid {
    /// The aggregates of the coarse lattice.
    pub fn block(&self) -> [usize; ND] {
        match self {
            HmcMultigrid::Wilson { params, .. } => params.block,
            HmcMultigrid::DomainWall { block, .. } => *block,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HmcSolver {
    pub preconditioner: Option<HmcPreconditioner>,
    pub chebyshev: HmcChebyshev,
    pub multigrid: Option<HmcMultigrid>,
    pub kind: Option<HmcSolverKind>,
    /// Domain-wall fermions: solve with Neff's better conditioned D Q at this α (1: D itself).
    pub neff_alpha: f64,
    /// Steps of FGMRES before a restart; directions GCR keeps.
    pub restart: usize,
    pub tolerance_squared: f64,
    /// The energies at the start and the end of a trajectory: as precise as the force or more.
    pub energy_tolerance_squared: f64,
    pub max_iterations: usize,
    /// Mixed precision: CG in single precision to this |r|²/|b|², corrected in double.
    pub mixed: Option<f64>,
    /// Start each force solve from the best combination of the solutions of the last this
    /// many force solves of the trajectory (`hmc::Chrono`); 0 (the default): from zero, as
    /// Bridge++ does, so that its runs are repeated number by number. Two flavours only.
    pub initial_guess: usize,
}

impl HmcSolver {
    /// The two-step solver of a `rational` action, if it asks for one.
    fn two_step(&self) -> Option<NormalSolver> {
        matches!(
            self.kind,
            Some(HmcSolverKind::Bicgstab | HmcSolverKind::Fgmres | HmcSolverKind::Gcr)
        )
        .then(|| self.normal())
    }

    /// The solver of a two-flavour action.
    fn normal(&self) -> NormalSolver {
        match self.kind {
            None | Some(HmcSolverKind::Cg) => NormalSolver::Cg,
            Some(HmcSolverKind::Bicgstab) => NormalSolver::BiCgStab,
            Some(HmcSolverKind::Fgmres) => NormalSolver::Fgmres(self.restart),
            Some(HmcSolverKind::Gcr) => NormalSolver::Gcr(self.restart),
            Some(HmcSolverKind::MultishiftCg) => unreachable!("checked when the file is read"),
        }
    }
}

impl HmcSolver {
    pub(super) fn asks_for(&self, p: HmcPreconditioner) -> bool {
        self.preconditioner == Some(p)
    }

    /// The Chebyshev preconditioner if the file asks for it, none otherwise.
    fn chebyshev(&self) -> Preconditioner {
        if self.asks_for(HmcPreconditioner::Chebyshev) {
            Preconditioner::chebyshev(self.chebyshev.steps, self.chebyshev.ratio)
        } else {
            Preconditioner::None
        }
    }

    /// The preconditioner the file asks for.
    fn preconditioner(&self) -> Preconditioner {
        match (self.preconditioner, self.multigrid) {
            (
                Some(HmcPreconditioner::Multigrid),
                Some(HmcMultigrid::Wilson {
                    params,
                    refresh_every,
                    reuse,
                }),
            ) => {
                let mut setup = MultigridSetup::new(params, refresh_every);
                if let Some(iterations) = reuse {
                    setup = setup.reusing_vectors(iterations);
                }
                Preconditioner::multigrid(setup)
            }
            (
                Some(HmcPreconditioner::Multigrid),
                Some(HmcMultigrid::DomainWall {
                    block,
                    test_vectors,
                    cycle,
                    low,
                    boot,
                    refresh_every,
                    inner_tolerance_squared,
                    coarse_double,
                    rebuild_growth,
                    fine_single,
                    levels,
                }),
            ) => {
                let inner = Stop::simple(inner_tolerance_squared, self.max_iterations);
                let mut setup = DwfMultigridSetup::new(block, test_vectors, cycle, inner);
                setup.low_modes = low;
                setup.bootstrap = boot;
                setup.directions = self.restart;
                setup.refresh_every = refresh_every;
                setup.coarse_double = coarse_double;
                setup.rebuild_growth = rebuild_growth;
                setup.fine_single = fine_single;
                setup.levels = levels;
                setup.alpha = self.neff_alpha;
                Preconditioner::DomainWallMultigrid(Box::new(setup))
            }
            _ => self.chebyshev(),
        }
    }

    /// What is wrong with the preconditioner asked for, for an action of the given kind.
    fn check_preconditioner(
        &self,
        lattice: [usize; ND],
        wilson_like: bool,
        all_sites: bool,
    ) -> Result<(), String> {
        match self.preconditioner {
            None | Some(HmcPreconditioner::None) => Ok(()),
            Some(HmcPreconditioner::Chebyshev) => {
                if !matches!(self.kind, None | Some(HmcSolverKind::Cg)) {
                    return Err("preconditioner_type Chebyshev goes with solver_type CG".into());
                }
                let c = self.chebyshev;
                if c.steps < 2 || c.ratio.is_nan() || c.ratio <= 1.0 {
                    return Err(
                        "Chebyshev: number_of_steps at least 2 and interval_ratio above 1".into(),
                    );
                }
                Ok(())
            }
            Some(HmcPreconditioner::Multigrid) => {
                if !matches!(
                    self.kind,
                    Some(HmcSolverKind::Bicgstab | HmcSolverKind::Fgmres | HmcSolverKind::Gcr)
                ) {
                    return Err(
                        "preconditioner_type MG goes with solver_type BiCGStab, FGMRES or GCR"
                            .into(),
                    );
                }
                if !wilson_like || !all_sites {
                    return Err("preconditioner_type MG: for Wilson and clover fermions, and the domain-wall Ratio action on all sites".into());
                }
                let Some(mg) = self.multigrid else {
                    return Err("preconditioner_type MG needs a MultiGrid section".into());
                };
                let block = mg.block();
                if (0..ND).any(|mu| block[mu] == 0 || !lattice[mu].is_multiple_of(block[mu])) {
                    return Err(format!(
                        "MultiGrid.MultiGrid_Level1.block_size {block:?} must divide the lattice {lattice:?}"
                    ));
                }
                if let HmcMultigrid::Wilson { params, .. } = mg {
                    let most = 6 * block.iter().product::<usize>();
                    if params.num_test_vectors == 0 || params.num_test_vectors > most {
                        return Err(format!(
                            "MultiGrid.MultiGrid_Level1.number_of_test_vectors: between 1 and {most}"
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

impl Default for HmcSolver {
    fn default() -> Self {
        Self {
            preconditioner: None,
            chebyshev: HmcChebyshev::default(),
            multigrid: None,
            kind: None,
            restart: restart(),
            mixed: None,
            neff_alpha: 1.0,
            tolerance_squared: md_tolerance(),
            energy_tolerance_squared: md_tolerance(),
            max_iterations: md_iterations(),
            initial_guess: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HmcOutput {
    /// Write the configuration after every this many trajectories (and after the last).
    pub save_every: Option<usize>,
    /// The file name, `{n}` replaced by the trajectory number; relative to the parameter
    /// file's directory. On several processes rank 0 writes it.
    pub file: Option<String>,
    pub format: GaugeFormat,
}

fn chebyshev_steps() -> usize {
    8
}
fn chebyshev_ratio() -> f64 {
    30.0
}
fn restart() -> usize {
    32
}
fn md_tolerance() -> f64 {
    1e-24
}
fn md_iterations() -> usize {
    10000
}

fn boundary(b: &[BoundaryConfig; ND]) -> [Boundary; ND] {
    b.map(Boundary::from)
}

fn invalid<T>(m: String) -> Result<T, CliError> {
    Err(CliError::Invalid(m))
}

/// Whether a parameter file is an HMC file (it has an `hmc` section).
pub fn is_hmc_file(path: &Path) -> Result<bool, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let toml = path.extension().and_then(|e| e.to_str()) == Some("toml");
    Ok(is_hmc_text(&text, toml))
}

impl HmcConfig {
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let params = match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => HmcParams::from_toml(&text)?,
            _ => HmcParams::from_yaml(&text)?,
        };
        Self::checked(params)
    }

    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        Self::checked(HmcParams::from_yaml(text)?)
    }

    fn checked(params: HmcParams) -> Result<Self, CliError> {
        let c = params.to_config()?;
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<(), CliError> {
        let levels = self.hmc.integrator.steps.len();
        if levels == 0 {
            return invalid("Builder_Integrator.number_of_steps: at least one level".into());
        }
        if self.actions.is_empty() {
            return invalid("Action_G, Action_F_1, ...: at least one action".into());
        }
        for (i, a) in self.actions.iter().enumerate() {
            let at = &self.action_names[i];
            let (level, smeared, ratio) = match a {
                ActionConfig::Plaquette { level, .. } | ActionConfig::Rectangle { level, .. } => {
                    (*level, false, None)
                }
                ActionConfig::TwoFlavor { level, smeared, .. } => (*level, *smeared, None),
                ActionConfig::Rational {
                    level,
                    smeared,
                    fermion,
                    poles,
                    bounds,
                    power,
                    ..
                } => {
                    if matches!(fermion, HmcFermion::DomainWall { .. }) {
                        return invalid(format!(
                            "{at}: Rational is for Wilson or clover fermions (domain-wall fermions need the Ratio over Pauli-Villars fields)"
                        ));
                    }
                    let [p, q] = *power;
                    if !(p > 0 && q > p) {
                        return invalid(format!(
                            "{at}.Rational: exponent_numerator/exponent_denominator {p}/{q} must be in (0, 1)"
                        ));
                    }
                    if *poles == 0 || !(bounds[0] > 0.0 && bounds[1] > bounds[0]) {
                        return invalid(format!(
                            "{at}.Rational: number_of_poles > 0 and 0 < lower_bound < upper_bound"
                        ));
                    }
                    (*level, *smeared, None)
                }
                ActionConfig::TwoFlavorRatio {
                    level,
                    smeared,
                    fermion,
                    denominator,
                    ..
                } => (*level, *smeared, Some((fermion, denominator))),
            };
            if level >= levels {
                return invalid(format!(
                    "{at}.level_of_action {level}: Builder_Integrator.number_of_steps has {levels} levels"
                ));
            }
            let _ = smeared;
            match a {
                ActionConfig::TwoFlavor { solver, .. }
                | ActionConfig::TwoFlavorRatio { solver, .. } => {
                    if solver.kind == Some(HmcSolverKind::MultishiftCg) {
                        return invalid(format!(
                            "{at}.Solver.solver_type: CG, BiCGStab, FGMRES or GCR"
                        ));
                    }
                    if solver.neff_alpha.is_nan() || solver.neff_alpha <= 0.0 {
                        return invalid(format!("{at}.Fopr.neff_alpha: positive"));
                    }
                    // The multigrid of domain-wall fermions: the Ratio action on all sites over
                    // Pauli-Villars fields, without Neff's α, with GCR on the even sites.
                    let dwf_multigrid = solver.asks_for(HmcPreconditioner::Multigrid)
                        && matches!(
                            a,
                            ActionConfig::TwoFlavorRatio {
                                fermion: HmcFermion::DomainWall { .. },
                                ..
                            }
                        );
                    if dwf_multigrid {
                        let ActionConfig::TwoFlavorRatio {
                            denominator,
                            even_odd,
                            ..
                        } = a
                        else {
                            unreachable!()
                        };
                        let _ = even_odd;
                        if denominator.is_some() {
                            return invalid(format!(
                                "{at}.Solver.preconditioner_type: the multigrid of domain-wall fermions is for the Ratio action over Pauli-Villars fields"
                            ));
                        }
                        if solver.kind != Some(HmcSolverKind::Gcr) {
                            return invalid(format!(
                                "{at}.Solver.solver_type: GCR with the multigrid of domain-wall fermions"
                            ));
                        }
                    }
                    let (wilson_like, all_sites) = match a {
                        ActionConfig::TwoFlavor {
                            fermion, even_odd, ..
                        } => {
                            let _ = even_odd;
                            (!matches!(fermion, HmcFermion::DomainWall { .. }), true)
                        }
                        ActionConfig::TwoFlavorRatio { fermion, .. } => (
                            dwf_multigrid || !matches!(fermion, HmcFermion::DomainWall { .. }),
                            true,
                        ),
                        _ => unreachable!(),
                    };
                    if let Err(e) =
                        solver.check_preconditioner(self.lattice, wilson_like, all_sites)
                    {
                        return invalid(format!("{at}.Solver: {e}"));
                    }
                    if solver.restart == 0 {
                        return invalid(format!(
                            "{at}.Solver.number_of_kept_directions: at least 1"
                        ));
                    }
                    if solver.mixed.is_some() {
                        let standard_eo =
                            matches!(a, ActionConfig::TwoFlavor { even_odd: true, .. });
                        let wall = matches!(
                            a,
                            ActionConfig::TwoFlavor {
                                fermion: HmcFermion::DomainWall { .. },
                                ..
                            } | ActionConfig::TwoFlavorRatio {
                                fermion: HmcFermion::DomainWall { .. },
                                ..
                            }
                        );
                        if !wall || standard_eo {
                            return invalid(format!(
                                "{at}.Solver.precision: mixed is for the domain-wall Standard (Lexical) and Ratio actions"
                            ));
                        }
                        if !matches!(solver.kind, None | Some(HmcSolverKind::Cg))
                            || !matches!(
                                solver.preconditioner,
                                None | Some(HmcPreconditioner::None)
                            )
                        {
                            return invalid(format!(
                                "{at}.Solver.precision: mixed goes with solver_type CG and no preconditioner"
                            ));
                        }
                    }
                    let domain_wall = matches!(
                        a,
                        ActionConfig::TwoFlavor {
                            fermion: HmcFermion::DomainWall { .. },
                            ..
                        } | ActionConfig::TwoFlavorRatio {
                            fermion: HmcFermion::DomainWall { .. },
                            ..
                        }
                    );
                    if !domain_wall && solver.neff_alpha != 1.0 {
                        return invalid(format!("{at}.Fopr.neff_alpha: for domain-wall fermions"));
                    }
                    if domain_wall
                        && !dwf_multigrid
                        && !matches!(solver.kind, None | Some(HmcSolverKind::Cg))
                    {
                        return invalid(format!(
                            "{at}.Solver.solver_type: CG for domain-wall fermions (BiCGStab, FGMRES and GCR do not converge on D itself)"
                        ));
                    }
                }
                ActionConfig::Rational { solver, .. } => {
                    if solver.asks_for(HmcPreconditioner::Chebyshev) {
                        return invalid(format!(
                            "{at}.Solver.preconditioner_type: MG (with BiCGStab, FGMRES or GCR) or None for Rational"
                        ));
                    }
                    if let Err(e) = solver.check_preconditioner(self.lattice, true, true) {
                        return invalid(format!("{at}.Solver: {e}"));
                    }
                    if solver.restart == 0 {
                        return invalid(format!(
                            "{at}.Solver.number_of_kept_directions: at least 1"
                        ));
                    }
                }
                _ => {}
            }
            if let ActionConfig::TwoFlavorRatio {
                solver,
                even_odd,
                fermion,
                ..
            } = a
            {
                if solver.initial_guess > 0 && !even_odd {
                    return invalid(format!(
                        "{at}.Solver.number_of_initial_guess: for the Standard actions and the even-odd Ratio"
                    ));
                }
                if *even_odd {
                    if matches!(fermion, HmcFermion::Clover { .. }) {
                        return invalid(format!(
                            "{at}: the even-odd Ratio of clover fermions (its det T term) is not written"
                        ));
                    }
                    if solver.neff_alpha != 1.0
                        && !solver.asks_for(HmcPreconditioner::Multigrid)
                    {
                        return invalid(format!(
                            "{at}.Fopr.neff_alpha: not with CG on the even-odd Ratio yet (the multigrid takes it)"
                        ));
                    }
                    if solver.asks_for(HmcPreconditioner::Multigrid)
                        && !matches!(fermion, HmcFermion::DomainWall { .. })
                    {
                        return invalid(format!(
                            "{at}.Solver.preconditioner_type: None or Chebyshev for the even-odd Ratio of Wilson fermions (the multigrid is that of domain-wall fermions)"
                        ));
                    }
                }
            }
            if let Some((f, d)) = ratio {
                match (f, d) {
                    (HmcFermion::DomainWall { .. }, None) => {}
                    (_, None) => {
                        return invalid(format!(
                            "{at}: a Ratio of Wilson or clover fermions needs Fopr_denominator"
                        ))
                    }
                    (a, Some(b)) if std::mem::discriminant(a) != std::mem::discriminant(b) => {
                        return invalid(format!(
                            "{at}.Fopr_denominator: the same kind of fermion as Fopr"
                        ))
                    }
                    _ => {}
                }
            }
        }
        if self.output.save_every.is_some() != self.output.file.is_some() {
            return invalid(
                "HMC: save_config_interval and config_filename_output go together".into(),
            );
        }
        Ok(())
    }

    /// The blocks of the multigrids must divide the lattice of a process (`local`).
    fn check_blocks(&self, local: [usize; ND]) -> Result<(), CliError> {
        for (i, a) in self.actions.iter().enumerate() {
            let solver = match a {
                ActionConfig::TwoFlavor { solver, .. }
                | ActionConfig::TwoFlavorRatio { solver, .. }
                | ActionConfig::Rational { solver, .. } => solver,
                _ => continue,
            };
            let (true, Some(mg)) = (
                solver.asks_for(HmcPreconditioner::Multigrid),
                solver.multigrid,
            ) else {
                continue;
            };
            let block = mg.block();
            if (0..ND).any(|mu| !local[mu].is_multiple_of(block[mu])) {
                return invalid(format!(
                    "{}.MultiGrid.MultiGrid_Level1.block_size {block:?} must divide the lattice {local:?} of a process",
                    self.action_names[i]
                ));
            }
        }
        Ok(())
    }

    /// The actions on the lattice `lat`; files named relative to `base`.
    fn terms(&self, lat: &Arc<Lattice>, base: &Path) -> Result<Vec<Term<'static>>, CliError> {
        let smearing = self.gauge_smearing.map(|g| g.smearing());
        let stops = |s: &HmcSolver| {
            let stop = |t: f64| Stop {
                tolerance_sqr: t,
                max_iterations: s.max_iterations,
                max_restarts: 10,
            };
            (stop(s.tolerance_squared), stop(s.energy_tolerance_squared))
        };
        self.actions
            .iter()
            .map(|a| -> Result<Term<'static>, CliError> {
                let (level, action): (usize, Box<dyn MdAction>) = match a {
                    ActionConfig::Plaquette { beta, level } => {
                        (*level, Box::new(PlaquetteAction { beta: *beta }))
                    }
                    ActionConfig::Rectangle {
                        beta,
                        c_plaq,
                        c_rect,
                        level,
                    } => (
                        *level,
                        Box::new(RectangleAction {
                            beta: *beta,
                            c_plaq: *c_plaq,
                            c_rect: *c_rect,
                        }),
                    ),
                    ActionConfig::TwoFlavor {
                        fermion,
                        level,
                        smeared,
                        even_odd,
                        solver,
                    } => {
                        let (md, h) = stops(solver);
                        let c = Common {
                            smearing: smearing.filter(|_| *smeared),
                            md,
                            h,
                            solver,
                        };
                        let action = match *fermion {
                            HmcFermion::Wilson { kappa, boundary: b } => {
                                two_flavor(wilson(lat, kappa, &b), None, None, None, *even_odd, c)
                            }
                            HmcFermion::Clover {
                                kappa,
                                csw,
                                boundary: b,
                            } => {
                                let op = clover(lat, kappa, csw, &b);
                                two_flavor(op, Some(kappa * csw), None, None, *even_odd, c)
                            }
                            HmcFermion::DomainWall { .. } => {
                                let p = mdwf_params(fermion);
                                let alpha = with_alpha(lat, p, solver.neff_alpha);
                                let single = single_mdwf(lat, p, solver);
                                two_flavor(
                                    Mdwf::<f64>::new(lat, p),
                                    None,
                                    alpha,
                                    single,
                                    *even_odd,
                                    c,
                                )
                            }
                        };
                        (*level, action)
                    }
                    ActionConfig::TwoFlavorRatio {
                        fermion,
                        denominator,
                        level,
                        smeared,
                        even_odd,
                        solver,
                    } => {
                        let (md, h) = stops(solver);
                        let c = Common {
                            smearing: smearing.filter(|_| *smeared),
                            md,
                            h,
                            solver,
                        };
                        let action = match (*fermion, *denominator) {
                            (HmcFermion::DomainWall { .. }, d) if *even_odd => {
                                let p = mdwf_params(fermion);
                                let pd = d.map_or(p.pauli_villars(), |d| mdwf_params(&d));
                                let ops = (Mdwf::<f64>::new(lat, p), Mdwf::<f64>::new(lat, pd));
                                ratio_eo(ops, single_mdwf_eo(lat, p, solver), c)
                            }
                            (
                                HmcFermion::Wilson { kappa, boundary: b },
                                Some(HmcFermion::Wilson {
                                    kappa: kd,
                                    boundary: bd,
                                }),
                            ) if *even_odd => {
                                ratio_eo((wilson(lat, kappa, &b), wilson(lat, kd, &bd)), None, c)
                            }
                            (HmcFermion::DomainWall { .. }, d) => {
                                let p = mdwf_params(fermion);
                                let pd = d.map_or(p.pauli_villars(), |d| mdwf_params(&d));
                                let ops = (Mdwf::<f64>::new(lat, p), Mdwf::<f64>::new(lat, pd));
                                match (
                                    solver.asks_for(HmcPreconditioner::Multigrid),
                                    solver.multigrid,
                                ) {
                                    (true, Some(_)) => ratio(ops, None, None, None, c),
                                    _ => {
                                        let alpha = with_alpha(lat, p, solver.neff_alpha);
                                        let single = single_mdwf(lat, p, solver);
                                        ratio(ops, None, alpha, single, c)
                                    }
                                }
                            }
                            (
                                HmcFermion::Wilson { kappa, boundary: b },
                                Some(HmcFermion::Wilson {
                                    kappa: kd,
                                    boundary: bd,
                                }),
                            ) => ratio(
                                (wilson(lat, kappa, &b), wilson(lat, kd, &bd)),
                                None,
                                None,
                                None,
                                c,
                            ),
                            (
                                HmcFermion::Clover {
                                    kappa,
                                    csw,
                                    boundary: b,
                                },
                                Some(HmcFermion::Clover {
                                    kappa: kd,
                                    csw: cd,
                                    boundary: bd,
                                }),
                            ) => {
                                let ops = (clover(lat, kappa, csw, &b), clover(lat, kd, cd, &bd));
                                ratio(ops, Some((kappa * csw, kd * cd)), None, None, c)
                            }
                            _ => unreachable!("checked when the file is read"),
                        };
                        (*level, action)
                    }
                    ActionConfig::Rational {
                        fermion,
                        level,
                        poles,
                        bounds,
                        power: [p, q],
                        coefficients,
                        smeared,
                        solver,
                    } => {
                        let (md, h) = stops(solver);
                        let c = Common {
                            smearing: smearing.filter(|_| *smeared),
                            md,
                            h,
                            solver,
                        };
                        // S = φ† (D†D)^(−p/q) φ, φ = (D†D)^(p/2q) ξ.
                        let spec = RationalSpec {
                            poles: *poles,
                            p: -p,
                            q: *q,
                            x_min: bounds[0] * bounds[0],
                            x_max: bounds[1] * bounds[1],
                        };
                        let g = gcd(*p, 2 * q);
                        let approximations = (
                            approximation(&spec, coefficients.as_deref(), base)?,
                            approximation(
                                &spec.with_power(p / g, 2 * q / g),
                                coefficients.as_deref(),
                                base,
                            )?,
                        );
                        let action = match *fermion {
                            HmcFermion::Wilson { kappa, boundary: b } => {
                                rational(wilson(lat, kappa, &b), None, approximations, c)
                            }
                            HmcFermion::Clover {
                                kappa,
                                csw,
                                boundary: b,
                            } => {
                                let op = clover(lat, kappa, csw, &b);
                                rational(op, Some(kappa * csw), approximations, c)
                            }
                            HmcFermion::DomainWall { .. } => {
                                unreachable!("checked when the file is read")
                            }
                        };
                        (*level, action)
                    }
                };
                Ok(Term { level, action })
            })
            .collect()
    }
}

#[cfg(feature = "cuda-bridge")]
impl HmcConfig {
    /// The actions with the links on the device: the gauge actions and the domain-wall Ratio
    /// actions (`HmcParams::to_config` refuses others).
    fn device_terms(
        &self,
        lat: &Arc<Lattice>,
    ) -> Result<Vec<Term<'static, crate::backend::cuda_bridge::DeviceLinks>>, CliError> {
        use crate::backend::cuda_bridge::DeviceLinks;
        let smearing = self.gauge_smearing.map(|g| g.smearing());
        self.actions
            .iter()
            .map(|a| -> Result<Term<'static, DeviceLinks>, CliError> {
                let (level, action): (usize, Box<dyn MdAction<DeviceLinks>>) = match a {
                    ActionConfig::Plaquette { beta, level } => {
                        (*level, Box::new(PlaquetteAction { beta: *beta }))
                    }
                    ActionConfig::Rectangle {
                        beta,
                        c_plaq,
                        c_rect,
                        level,
                    } => (
                        *level,
                        Box::new(RectangleAction {
                            beta: *beta,
                            c_plaq: *c_plaq,
                            c_rect: *c_rect,
                        }),
                    ),
                    ActionConfig::TwoFlavorRatio {
                        fermion: fermion @ HmcFermion::DomainWall { .. },
                        denominator,
                        level,
                        smeared,
                        even_odd,
                        solver,
                    } => {
                        let stop = |t: f64| Stop {
                            tolerance_sqr: t,
                            max_iterations: solver.max_iterations,
                            max_restarts: 10,
                        };
                        let c = Common {
                            smearing: smearing.filter(|_| *smeared),
                            md: stop(solver.tolerance_squared),
                            h: stop(solver.energy_tolerance_squared),
                            solver,
                        };
                        let p = mdwf_params(fermion);
                        let pd = denominator.map_or(p.pauli_villars(), |d| mdwf_params(&d));
                        let action = match even_odd {
                            true => device_ratio_eo(lat, (p, pd), c)?,
                            false => device_ratio(lat, (p, pd), solver.neff_alpha, c)?,
                        };
                        (*level, action)
                    }
                    ActionConfig::TwoFlavor {
                        fermion: fermion @ HmcFermion::DomainWall { .. },
                        level,
                        smeared,
                        solver,
                        ..
                    } => {
                        let stop = |t: f64| Stop {
                            tolerance_sqr: t,
                            max_iterations: solver.max_iterations,
                            max_restarts: 10,
                        };
                        let c = Common {
                            smearing: smearing.filter(|_| *smeared),
                            md: stop(solver.tolerance_squared),
                            h: stop(solver.energy_tolerance_squared),
                            solver,
                        };
                        let p = mdwf_params(fermion);
                        (*level, device_two_flavor(lat, p, solver.neff_alpha, c)?)
                    }
                    _ => unreachable!("refused when the file is read"),
                };
                Ok(Term { level, action })
            })
            .collect()
    }
}

/// What an action is made from besides its operator.
struct Common<'a> {
    smearing: Option<GaugeSmearing>,
    /// The stopping rules of the force solves and of the energy solves.
    md: Stop,
    h: Stop,
    solver: &'a HmcSolver,
}

/// Two flavours of `op`, on all sites or even-odd: with the clover term's coefficient κ c_SW
/// for an operator that has one, and the better conditioned operator to solve with if any.
fn two_flavor<O: EvenOddForceOp + 'static>(
    op: O,
    clover: Option<f64>,
    conditioned: Option<O>,
    single: Option<Box<dyn SingleNormal<O::Field, O::Links>>>,
    even_odd: bool,
    c: Common,
) -> Box<dyn MdAction> {
    let s = c.solver;
    if even_odd {
        assert!(single.is_none(), "refused when the file is read");
        let mut a = TwoFlavorEo::new(op, c.smearing, c.md, c.h)
            .with_guess(s.initial_guess)
            .with_solver(s.normal())
            .with_preconditioner(s.preconditioner());
        if let Some(kc) = clover {
            a = a.with_clover(kc);
        }
        if let Some(o) = conditioned {
            a = a.with_conditioned(o);
        }
        Box::new(a)
    } else {
        let mut a = TwoFlavor::new(op, c.smearing, c.md, c.h)
            .with_guess(s.initial_guess)
            .with_solver(s.normal())
            .with_preconditioner(s.preconditioner());
        if let Some(kc) = clover {
            a = a.with_clover(kc);
        }
        if let Some(s) = single {
            a = a.with_single(s);
        }
        if let Some(o) = conditioned {
            a = a.with_conditioned(o);
        }
        Box::new(a)
    }
}

/// Two flavours of the first operator over two of the second.
fn ratio<O: ForceOp + 'static>(
    (op, denominator): (O, O),
    clover: Option<(f64, f64)>,
    conditioned: Option<O>,
    single: Option<Box<dyn SingleNormal<O::Field, O::Links>>>,
    c: Common,
) -> Box<dyn MdAction<O::Links>> {
    let mut a = TwoFlavorRatio::new(op, denominator, c.smearing, c.md, c.h)
        .with_solver(c.solver.normal())
        .with_preconditioner(c.solver.preconditioner());
    if let Some((kc, kc_denominator)) = clover {
        a = a.with_clover(kc, kc_denominator);
    }
    if let Some(o) = conditioned {
        a = a.with_conditioned(o);
    }
    if let Some(s) = single {
        a = a.with_single(s);
    }
    Box::new(a)
}

/// Two flavours of the first operator over two of the second, even-odd.
fn ratio_eo<O: EvenOddHalves + 'static>(
    (op, denominator): (O, O),
    single: Option<Box<dyn SingleNormal<O::Half, O::Links>>>,
    c: Common,
) -> Box<dyn MdAction<O::Links>> {
    let s = c.solver;
    let mut a = TwoFlavorRatioEo::new(op, denominator, c.smearing, c.md, c.h)
        .with_guess(s.initial_guess)
        .with_solver(s.normal())
        .with_preconditioner(s.preconditioner());
    if let Some(s) = single {
        a = a.with_single(s);
    }
    Box::new(a)
}

/// The domain-wall operator `p` (with Neff's α of the solver) in single precision on all sites,
/// when the solver asks for mixed precision.
fn single_mdwf(lat: &Arc<Lattice>, p: MdwfParams, s: &HmcSolver) -> Option<Single<Mdwf<f64>>> {
    s.mixed
        .map(|tol| Box::new(SingleMdwf::<Dirac>::new(lat, p, s.neff_alpha, tol)) as Box<_>)
}

/// The same on the even sites, for the even-odd Ratio action.
fn single_mdwf_eo(
    lat: &Arc<Lattice>,
    p: MdwfParams,
    s: &HmcSolver,
) -> Option<Box<dyn SingleNormal<HalfPsi, Field<f64, Gauge>>>> {
    s.mixed
        .map(|tol| Box::new(SingleMdwfEo::<Dirac>::new(lat, p, tol)) as Box<_>)
}

/// `ratio` of domain-wall operators on the device: D of `p` over D of `pd`, solved with Neff's
/// `alpha` if it is not 1.
#[cfg(feature = "cuda-bridge")]
fn device_ratio(
    lat: &Arc<Lattice>,
    (p, pd): (MdwfParams, MdwfParams),
    alpha: f64,
    c: Common,
) -> Result<Box<dyn MdAction<crate::backend::cuda_bridge::DeviceLinks>>, CliError> {
    use crate::backend::cuda_bridge::CudaMdwf;
    let make = |p: MdwfParams| {
        CudaMdwf::<f64>::new(lat, p).map_err(|e| CliError::Invalid(format!("the device: {e}")))
    };
    // The multigrid takes Neff's α itself (`DwfMultigridSetup::alpha`), with the plain D.
    let multigrid = c.solver.asks_for(HmcPreconditioner::Multigrid);
    let conditioned = match alpha != 1.0 && !multigrid {
        true => {
            let mut op = make(p)?;
            op.set_alpha(alpha);
            Some(op)
        }
        false => None,
    };
    let single = device_single(lat, p, alpha, c.solver, false)?;
    Ok(ratio((make(p)?, make(pd)?), None, conditioned, single, c))
}

/// The domain-wall operator `p` in single precision on the device, when the solver asks for
/// mixed precision (`even_odd`: for K of the even-odd Ratio action).
#[cfg(feature = "cuda-bridge")]
fn device_single(
    lat: &Arc<Lattice>,
    p: MdwfParams,
    alpha: f64,
    s: &HmcSolver,
    even_odd: bool,
) -> Result<Option<Single<crate::backend::cuda_bridge::CudaMdwf<f64>>>, CliError> {
    use crate::backend::cuda_bridge::hmc_mixed::SingleCuda;
    s.mixed
        .map(|tol| {
            SingleCuda::new(lat, p, alpha, tol, even_odd)
                .map(|op| Box::new(op) as Single<crate::backend::cuda_bridge::CudaMdwf<f64>>)
                .map_err(|e| CliError::Invalid(format!("the device: {e}")))
        })
        .transpose()
}

/// `ratio_eo` of domain-wall operators on the device: D of `p` over D of `pd`, even-odd.
#[cfg(feature = "cuda-bridge")]
fn device_ratio_eo(
    lat: &Arc<Lattice>,
    (p, pd): (MdwfParams, MdwfParams),
    c: Common,
) -> Result<Box<dyn MdAction<crate::backend::cuda_bridge::DeviceLinks>>, CliError> {
    use crate::backend::cuda_bridge::CudaMdwf;
    let make = |p: MdwfParams| {
        CudaMdwf::<f64>::new(lat, p).map_err(|e| CliError::Invalid(format!("the device: {e}")))
    };
    let single = device_single(lat, p, 1.0, c.solver, true)?;
    Ok(ratio_eo((make(p)?, make(pd)?), single, c))
}

/// Two flavours of the domain-wall operator of `p` on the device, solved with Neff's `alpha`
/// if it is not 1: `two_flavor` on all sites.
#[cfg(feature = "cuda-bridge")]
fn device_two_flavor(
    lat: &Arc<Lattice>,
    p: MdwfParams,
    alpha: f64,
    c: Common,
) -> Result<Box<dyn MdAction<crate::backend::cuda_bridge::DeviceLinks>>, CliError> {
    use crate::backend::cuda_bridge::CudaMdwf;
    let make = |p: MdwfParams| {
        CudaMdwf::<f64>::new(lat, p).map_err(|e| CliError::Invalid(format!("the device: {e}")))
    };
    let s = c.solver;
    let mut a = TwoFlavor::new(make(p)?, c.smearing, c.md, c.h)
        .with_guess(s.initial_guess)
        .with_solver(s.normal())
        .with_preconditioner(s.preconditioner());
    if alpha != 1.0 {
        let mut op = make(p)?;
        op.set_alpha(alpha);
        a = a.with_conditioned(op);
    }
    if let Some(single) = device_single(lat, p, alpha, s, false)? {
        a = a.with_single(single);
    }
    Ok(Box::new(a))
}

/// A power of the determinant of `op` by the rational approximations of the action and of
/// the heat bath.
fn rational<O: ForceOp<Links = Field<f64, Gauge>> + 'static>(
    op: O,
    clover: Option<f64>,
    (md, heat_bath): (RationalApprox, RationalApprox),
    c: Common,
) -> Box<dyn MdAction> {
    let s = c.solver;
    let mut a = Rational::new(op, c.smearing, md, heat_bath, c.md, c.h)
        .with_pole_by_pole(s.kind == Some(HmcSolverKind::Cg));
    if let Some(kc) = clover {
        a = a.with_clover(kc);
    }
    if let Some(two) = s.two_step() {
        a = a.with_two_step(two).with_preconditioner(s.preconditioner());
    }
    Box::new(a)
}

fn wilson(lat: &Arc<Lattice>, kappa: f64, b: &[BoundaryConfig; ND]) -> Wilson<f64> {
    Wilson::<f64>::new(
        lat,
        WilsonParams {
            kappa,
            boundary: boundary(b),
        },
    )
}

fn clover(lat: &Arc<Lattice>, kappa: f64, csw: f64, b: &[BoundaryConfig; ND]) -> Clover<f64> {
    Clover::<f64>::new(
        lat,
        CloverParams {
            kappa,
            csw,
            boundary: boundary(b),
        },
    )
}

/// The domain-wall operator with Neff's α, for the solves; none at α = 1.
fn with_alpha(lat: &Arc<Lattice>, p: MdwfParams, alpha: f64) -> Option<Mdwf<f64>> {
    (alpha != 1.0).then(|| {
        let mut op = Mdwf::<f64>::new(lat, p);
        op.set_alpha(alpha);
        op
    })
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// The rational approximation `spec`: read from Bridge++'s file in `dir` (relative to `base`)
/// if a directory is given, computed otherwise.
fn approximation(
    spec: &RationalSpec,
    dir: Option<&Path>,
    base: &Path,
) -> Result<RationalApprox, CliError> {
    match dir {
        Some(dir) => {
            RationalApprox::read_bridge(&base.join(dir).join(spec.bridge_file_name()), spec)
        }
        None => RationalApprox::minimax(spec),
    }
    .map_err(|e| CliError::Invalid(format!("rational approximation: {e}")))
}

impl HmcConfig {}

fn mdwf_params(f: &HmcFermion) -> MdwfParams {
    match *f {
        HmcFermion::DomainWall {
            ls,
            m5,
            mass,
            b,
            c,
            boundary: bd,
        } => MdwfParams {
            ls,
            m5,
            mass,
            b,
            c,
            boundary: boundary(&bd),
        },
        _ => unreachable!("a domain-wall fermion"),
    }
}

/// What a run did, trajectory by trajectory.
#[derive(Debug, Clone)]
pub struct HmcReport {
    pub plaquette_start: f64,
    /// (trajectory number, ΔH, accepted, plaquette after it).
    pub trajectories: Vec<(u32, f64, bool, f64)>,
}

/// Run the HMC the file describes on the processes of `comm`; rank 0 prints a line per
/// trajectory to `out`.
pub fn run_on(
    config: &HmcConfig,
    base: &Path,
    out: &mut dyn Write,
    comm: Arc<dyn Comm>,
) -> Result<HmcReport, CliError> {
    let grid = super::grid_for(config.lattice, config.process_grid, comm.size())?;
    let lat = Lattice::new(config.lattice, grid, comm);
    config.check_blocks(lat.local)?;
    let root = lat.comm.rank() == 0;
    let u = match &config.start {
        StartConfig::File { file, format } => super::gauge_file(*format)?
            .read(&base.join(file), &lat)
            .map_err(CliError::Gauge)?,
        StartConfig::Cold => cold_start(&lat),
        StartConfig::Hot { seed } => random_gauge(&lat, *seed),
    };
    let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
    let plaquette_start = plaquette(&u, &st);
    let ig = &config.hmc.integrator;
    let scheme = match ig.scheme {
        SchemeConfig::Leapfrog => Scheme::Leapfrog,
        SchemeConfig::Omelyan => Scheme::Omelyan { lambda: ig.lambda },
    };
    let make = |terms| {
        let mut hmc = Hmc::new(
            terms,
            scheme,
            ig.steps.clone(),
            ig.exp_order,
            config.hmc.trajectory_length,
        );
        hmc.metropolis = config.hmc.metropolis;
        hmc
    };
    if config.device {
        #[cfg(feature = "cuda-bridge")]
        {
            use crate::backend::cuda_bridge::DeviceLinks;
            let hmc = make_device(config, &lat, scheme)?;
            let u = <DeviceLinks as LinkField>::from_host(&u);
            return chain(config, base, out, root, u, hmc, plaquette_start);
        }
    }
    let hmc = make(config.terms(&lat, base)?);
    chain(config, base, out, root, u, hmc, plaquette_start)
}

/// The HMC of the file with the links on the device.
#[cfg(feature = "cuda-bridge")]
fn make_device(
    config: &HmcConfig,
    lat: &Arc<Lattice>,
    scheme: Scheme,
) -> Result<Hmc<'static, crate::backend::cuda_bridge::DeviceLinks>, CliError> {
    let ig = &config.hmc.integrator;
    let mut hmc = Hmc::new(
        config.device_terms(lat)?,
        scheme,
        ig.steps.clone(),
        ig.exp_order,
        config.hmc.trajectory_length,
    );
    hmc.metropolis = config.hmc.metropolis;
    Ok(hmc)
}

/// The trajectories of the run from the links `u`, on the host or on a device; the flow and
/// the saved configurations from a copy on the host.
fn chain<U: LinkField>(
    config: &HmcConfig,
    base: &Path,
    out: &mut dyn Write,
    root: bool,
    mut u: U,
    mut hmc: Hmc<'_, U>,
    plaquette_start: f64,
) -> Result<HmcReport, CliError> {
    let io = |e| CliError::Io {
        path: PathBuf::from("<stdout>"),
        source: e,
    };
    let first = config.hmc.first_trajectory;
    let mut rng = Mseries::new(first);
    if root {
        writeln!(
            out,
            "# HMC on {:?}, {} actions, plaquette {plaquette_start:.15}",
            config.lattice,
            config.actions.len()
        )
        .map_err(io)?;
        writeln!(
            out,
            "# trajectory  plaquette  dH  exp(-dH)  accepted  seconds"
        )
        .map_err(io)?;
    }
    // The reversibility checks, from the first configuration, with random numbers of their own.
    let mut check_rng = Mseries::new(first.wrapping_add(0x5eed_0000));
    for k in 0..config.hmc.reversibility_checks {
        let start = Instant::now();
        let r = hmc.reversibility(&u, &mut check_rng);
        if root {
            writeln!(
                out,
                "# reversibility {}: dH {:+.6e}, H back - H {:+.3e}, max |dU| {:.3e}, max |dP| {:.3e} ({:.1} s)",
                k + 1,
                r.delta_h,
                r.delta_h_back,
                r.max_link,
                r.max_momentum,
                start.elapsed().as_secs_f64()
            )
            .map_err(io)?;
        }
    }
    // The cost of the solves is that of each trajectory: what the checks cost is left out.
    for term in hmc.terms.iter_mut() {
        term.action.take_cost();
    }
    let mut report = HmcReport {
        plaquette_start,
        trajectories: Vec::new(),
    };
    let (mut accepted, mut exp_sum) = (0usize, 0.0);
    let mut flowed: Vec<AtT0> = Vec::new();
    for k in 0..config.hmc.trajectories {
        let n = first + k as u32 + 1;
        let start = Instant::now();
        let t = hmc.trajectory(&mut u, &mut rng);
        accepted += usize::from(t.accepted);
        exp_sum += (-t.delta_h).exp();
        if root {
            writeln!(
                out,
                "{n:8}  {:.15}  {:+.6e}  {:.6}  {}  {:.1}",
                t.plaquette,
                t.delta_h,
                (-t.delta_h).exp(),
                u8::from(t.accepted),
                start.elapsed().as_secs_f64()
            )
            .map_err(io)?;
        }
        for (i, term) in hmc.terms.iter_mut().enumerate() {
            let Some(c) = term.action.take_cost() else {
                continue;
            };
            if root {
                let name = config.action_names.get(i).map_or("action", String::as_str);
                let multigrid = match c.rebuilds + c.reuses {
                    0 => String::new(),
                    _ => format!(
                        "; multigrid setup {:.2} s, rebuild {:.2} s, {} rebuilds, {} reuses",
                        c.setup_seconds, c.rebuild_seconds, c.rebuilds, c.reuses
                    ),
                };
                writeln!(
                    out,
                    "# cost {n:8} {name}: {} solves, {} iterations, {:.2} s{multigrid}",
                    c.solves, c.iterations, c.seconds
                )
                .map_err(io)?;
            }
        }
        report
            .trajectories
            .push((n, t.delta_h, t.accepted, t.plaquette));
        if let Some(m) = config.flow.filter(|m| (k + 1) % m.interval == 0) {
            // The flow of a copy: the chain goes on from `u`.
            let mut v = u.to_host();
            let at = flow_to_t0(&mut v, &m.flow, m.t_max);
            if root {
                writeln!(
                    out,
                    "# flow {n:8}  t0/a^2 t Q_clover Q_imp  {}",
                    super::flow::t0_columns(at.as_ref())
                )
                .map_err(io)?;
            }
            if k + 1 > m.skip {
                flowed.extend(at);
            }
        }
        if let (Some(every), Some(file)) = (config.output.save_every, &config.output.file) {
            if (k + 1) % every == 0 || k + 1 == config.hmc.trajectories {
                let path = base.join(file.replace("{n}", &format!("{n:06}")));
                // Every process takes part; rank 0 writes the file.
                super::gauge_file(config.output.format)?
                    .write(&path, &u.to_host())
                    .map_err(CliError::Gauge)?;
            }
        }
    }
    if root {
        let n = config.hmc.trajectories.max(1) as f64;
        writeln!(
            out,
            "# acceptance {:.3}, <exp(-dH)> {:.4}",
            accepted as f64 / n,
            exp_sum / n
        )
        .map_err(io)?;
        if let Some(m) = config.flow {
            if m.skip > 0 {
                writeln!(out, "# without the first {} trajectories:", m.skip).map_err(io)?;
            }
            super::flow::gamma_lines(out, &flowed, m.interval, "trajectories").map_err(io)?;
        }
    }
    Ok(report)
}
