//! The parameter file of an HMC run. The sections and keys are those of Bridge++'s HMC files
//! (`HMC_General`, `Builder_Integrator`, `Action_G`, `Fopr`, …) where it has them; where
//! Bridge++ fixes the actions in the test program, the file lists them: `Action_F_1`,
//! `Action_F_2`, … each with its operator, its level and its solver. `Main`, `Fopr`,
//! `GaugeSmearing` and `Chebyshev` are the sections of the propagator files
//! ([`super::params`]).
//!
//! ```yaml
//! Main:
//!   lattice_size : [4, 4, 4, 8]
//! HMC:
//!   gauge_config_status   : Continue          # Continue | Cold_start | Hot_start
//!   config_filename_input : conf_04040408.txt
//!   trajectory_number     : 200000            # of the first one: the M-series random numbers
//!   number_of_trajectories : 10
//!   save_config_interval  : 5                 # with config_filename_output: "conf_{n}.txt"
//!   rand_filename_output  : "rand_{n}.txt"    # the random numbers, saved with the configurations
//!   ## rand_filename_input : rand_000200.txt  # go on with them instead of trajectory_number
//! HMC_General:
//!   trajectory_length : 1.0
//! Builder_Integrator:
//!   integrator      : Omelyan                 # Leapfrog | Omelyan
//!   number_of_steps : [5, 4]                  # one per level, the outermost first
//! Action_G:
//!   action_type : Action_G_Rectangle          # Action_G_Plaq | Action_G_Rectangle
//!   beta : 2.3
//!   c_plaq : 3.648
//!   c_rect : -0.331
//!   level_of_action : 1
//! Action_F_1:
//!   action_type : Standard                    # Standard | Ratio | Rational
//!   operator_type : Even_odd                  # Lexical (default) | Even_odd (Standard, Ratio)
//!   level_of_action : 0
//!   Fopr:
//!     fermion_type : Clover
//!     hopping_parameter : 0.12
//!     clover_coefficient : 1.0
//!     boundary_condition : [1, 1, 1, -1]
//!   Solver:
//!     solver_type : CG
//!     convergence_criterion_squared : 1.0e-24
//! ```
//!
//! With `Solver.preconditioner_type : MG` the action has the `MultiGrid` section of the
//! propagator files (`MultiGrid_Level1`, `Setup`, `Smoother`, the coarse solvers), and
//! `MultiGrid_HMC` for what only HMC has: when the multigrid is set up again.

use super::config::GaugeFormat;
use super::hmc::{
    ActionConfig, FlowMonitor, HmcBlock, HmcChebyshev, HmcConfig, HmcFermion, HmcMultigrid,
    HmcOutput, HmcPreconditioner, HmcSolver, HmcSolverKind, IntegratorConfig, SchemeConfig,
    StartConfig,
};
use super::params::{
    Backend, ChebyshevBlock, FermionType, FoprBlock, GaugeSmearingBlock, GaugeType, MainBlock,
    MultigridBlock, OperatorType, PrecisionType, PreconditionerType, SolverPrecision, SolverType,
};
use super::CliError;
use crate::gauge::flow::WilsonFlow;
use crate::{keyword_enum, ND};
use serde::Deserialize;
use std::path::PathBuf;

keyword_enum! {
    pub(crate) enum StartStatus {
        /// From the file `config_filename_input`.
        Continue => ["Continue", "File"],
        /// Every link the identity.
        ColdStart => ["Cold_start", "Cold"],
        /// Random links, from `seed_for_random_number`.
        HotStart => ["Hot_start", "Hot"],
    }
}

keyword_enum! {
    pub(crate) enum IntegratorType {
        Leapfrog => ["Leapfrog"],
        Omelyan => ["Omelyan"],
    }
}

keyword_enum! {
    pub(crate) enum GaugeActionType {
        Plaquette => ["Action_G_Plaq", "Plaquette", "Plaq"],
        Rectangle => ["Action_G_Rectangle", "Rectangle"],
    }
}

keyword_enum! {
    pub(crate) enum FermionActionType {
        /// Two flavours, φ† (D†D)⁻¹ φ.
        Standard => ["Standard", "Action_F_Standard", "Action_F_Standard_lex", "Two_flavor"],
        /// Two flavours over two of `Fopr_denominator` (domain-wall fermions without one: over
        /// Pauli-Villars).
        Ratio => ["Ratio", "Action_F_Ratio", "Action_F_Ratio_lex"],
        /// A power of the determinant by a rational approximation (RHMC).
        Rational => ["Rational", "Action_F_Rational"],
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HmcParams {
    #[serde(rename = "Main")]
    pub main: MainBlock,
    #[serde(rename = "HMC")]
    pub hmc: RunBlock,
    #[serde(rename = "HMC_General")]
    pub general: GeneralBlock,
    #[serde(rename = "Builder_Integrator")]
    pub integrator: IntegratorBlock,
    /// The smearing of the links of the fermion actions that ask for it (`smeared`).
    #[serde(rename = "GaugeSmearing")]
    pub gauge_smearing: Option<GaugeSmearingBlock>,
    #[serde(rename = "Action_G")]
    pub(crate) action_g: Option<GaugeActionBlock>,
    #[serde(rename = "Action_F_1")]
    pub(crate) action_f_1: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_2")]
    pub(crate) action_f_2: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_3")]
    pub(crate) action_f_3: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_4")]
    pub(crate) action_f_4: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_5")]
    pub(crate) action_f_5: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_6")]
    pub(crate) action_f_6: Option<FermionActionBlock>,
    /// The Wilson flow measured every `measurement_interval` trajectories.
    #[serde(rename = "GradientFlow")]
    pub(crate) gradient_flow: Option<FlowBlock>,
}

/// The Wilson flow along the run: t0 and the topological charge after every
/// `measurement_interval` trajectories (keys of Bridge++'s `GradientFlow` where it has them).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlowBlock {
    pub measurement_interval: usize,
    #[serde(default = "flow_step")]
    pub step_size: f64,
    #[serde(default = "flow_t_max")]
    pub max_flow_time: f64,
    #[serde(rename = "order_of_approx_for_exp_iP", default = "flow_exp_order")]
    pub(crate) order_of_exp_ip: usize,
    /// Only 3 (Lüscher's scheme).
    #[serde(rename = "order_of_RungeKutta", default = "flow_rk_order")]
    pub(crate) order_of_runge_kutta: usize,
    /// The first this many trajectories of the run are left out of the Γ-method at the end.
    #[serde(default)]
    pub skip_trajectories: usize,
}

fn flow_step() -> f64 {
    0.01
}

fn flow_t_max() -> f64 {
    10.0
}

fn flow_exp_order() -> usize {
    8
}

fn flow_rk_order() -> usize {
    3
}

impl FlowBlock {
    fn to_config(self) -> Result<FlowMonitor, CliError> {
        if self.order_of_runge_kutta != 3 {
            return invalid("GradientFlow.order_of_RungeKutta: only 3".into());
        }
        if self.measurement_interval == 0
            || self.step_size.is_nan()
            || self.step_size <= 0.0
            || self.order_of_exp_ip == 0
        {
            return invalid(
                "GradientFlow: measurement_interval, step_size and order_of_approx_for_exp_iP \
                 must be positive"
                    .into(),
            );
        }
        Ok(FlowMonitor {
            interval: self.measurement_interval,
            flow: WilsonFlow {
                step: self.step_size,
                exp_order: self.order_of_exp_ip,
            },
            t_max: self.max_flow_time,
            skip: self.skip_trajectories,
        })
    }
}

/// Where the run starts, how long it is and what it saves.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunBlock {
    pub gauge_config_status: StartStatus,
    #[serde(default = "text")]
    pub gauge_config_type_input: GaugeType,
    /// Relative to the parameter file's directory.
    pub config_filename_input: Option<PathBuf>,
    pub seed_for_random_number: Option<u64>,
    /// The number of the first trajectory: the random numbers of the run are those of the
    /// M-series generator made from it.
    #[serde(default)]
    pub trajectory_number: u32,
    pub number_of_trajectories: usize,
    /// Save the configuration after every this many trajectories (and after the last), in
    /// `config_filename_output` with `{n}` replaced by the trajectory number.
    pub save_config_interval: Option<usize>,
    pub config_filename_output: Option<String>,
    #[serde(default = "text")]
    pub gauge_config_type_output: GaugeType,
    /// The state of the random numbers, saved with every configuration (`{n}` as in
    /// `config_filename_output`), in the format of Bridge++'s `RandomNumbers_Mseries`.
    pub rand_filename_output: Option<String>,
    /// Go on with the random numbers of this state instead of those made from
    /// `trajectory_number`; relative to the parameter file's directory.
    pub rand_filename_input: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GeneralBlock {
    pub trajectory_length: f64,
    #[serde(rename = "Metropolis_test", default = "yes")]
    pub(crate) metropolis_test: bool,
    /// Not in Bridge++: checks of the reversibility of the molecular dynamics from the first
    /// configuration before the trajectories (`hmc::Hmc::reversibility`), with random numbers
    /// of their own; the trajectories are as without them.
    #[serde(default)]
    pub(crate) reversibility_checks: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IntegratorBlock {
    pub integrator: IntegratorType,
    /// Steps of each level, the outermost first.
    pub number_of_steps: Vec<usize>,
    #[serde(rename = "order_of_exp_iP", default = "exp_order")]
    pub(crate) order_of_exp_ip: usize,
    #[serde(rename = "lambda_Omelyan", default = "omelyan_lambda")]
    pub(crate) lambda_omelyan: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GaugeActionBlock {
    pub action_type: GaugeActionType,
    pub beta: f64,
    pub c_plaq: Option<f64>,
    pub c_rect: Option<f64>,
    #[serde(default)]
    pub level_of_action: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FermionActionBlock {
    pub action_type: FermionActionType,
    /// `Standard` and `Ratio`: on all sites (`Lexical`, the default) or the Schur complement on
    /// the even ones (`Even_odd`).
    pub operator_type: Option<OperatorType>,
    #[serde(default)]
    pub level_of_action: usize,
    /// On the links smeared by `GaugeSmearing`.
    #[serde(default = "yes")]
    pub smeared: bool,
    #[serde(rename = "Fopr")]
    pub fopr: FoprBlock,
    #[serde(rename = "Fopr_denominator", alias = "Fopr_prec")]
    pub(crate) fopr_denominator: Option<FoprBlock>,
    #[serde(rename = "Solver", default)]
    pub solver: HmcSolverBlock,
    /// The multigrid of `Solver.preconditioner_type : MG`: the `MultiGrid` section of the
    /// propagator files.
    #[serde(rename = "MultiGrid")]
    pub multigrid: Option<MultigridBlock>,
    #[serde(rename = "MultiGrid_HMC", default)]
    pub(crate) multigrid_hmc: MultigridHmcBlock,
    #[serde(rename = "Rational")]
    pub rational: Option<RationalBlock>,
}

/// The solver of an action's forces and energies.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HmcSolverBlock {
    /// `CG` (the default of `Standard` and `Ratio`), `BiCGStab`, `FGMRES`, `GCR`. `Rational`
    /// without it: multi-shift CG; with `CG`: one solve per pole.
    pub solver_type: Option<SolverType>,
    /// Of the force solves.
    #[serde(default = "md_tolerance")]
    pub convergence_criterion_squared: f64,
    /// Of the energies at the ends of a trajectory; the same as the forces if not given.
    #[serde(rename = "convergence_criterion_squared_H")]
    pub(crate) convergence_criterion_squared_h: Option<f64>,
    #[serde(default = "md_iterations")]
    pub maximum_number_of_iteration: usize,
    /// The cycle of FGMRES, the directions GCR keeps.
    #[serde(default = "kept")]
    pub number_of_kept_directions: usize,
    /// `Standard`: start each force solve from the last this many solutions.
    #[serde(default)]
    pub number_of_initial_guess: usize,
    /// What the solver works on, as in the propagator files; only what the action solves is
    /// accepted (`Lexical`, `Even_odd`, `Normal_left` for the even-odd domain-wall ratio,
    /// `PV_right` for the domain-wall multigrid).
    pub operator_type: Option<OperatorType>,
    #[serde(default = "no_preconditioner")]
    pub preconditioner_type: PreconditionerType,
    /// The polynomial preconditioner of CG (the smoother of the multigrid is in `MultiGrid`).
    #[serde(rename = "Chebyshev")]
    pub chebyshev: Option<ChebyshevBlock>,
    /// `double` (the default) or `mixed`: CG in single precision corrected in double, for
    /// domain-wall fermions with CG and no preconditioner.
    pub precision: Option<SolverPrecision>,
    /// Mixed precision: where each CG in single precision stops, |r|²/|b|² (1e-10).
    pub convergence_criterion_squared_inner: Option<f64>,
    /// The place of the multigrid before it moved to the action: refused, with where it went.
    #[serde(rename = "MultiGrid")]
    pub(crate) moved_multigrid: Option<serde::de::IgnoredAny>,
}

impl Default for HmcSolverBlock {
    fn default() -> Self {
        Self {
            solver_type: None,
            convergence_criterion_squared: md_tolerance(),
            convergence_criterion_squared_h: None,
            maximum_number_of_iteration: md_iterations(),
            number_of_kept_directions: kept(),
            number_of_initial_guess: 0,
            operator_type: None,
            precision: None,
            convergence_criterion_squared_inner: None,
            preconditioner_type: PreconditionerType::None,
            chebyshev: None,
            moved_multigrid: None,
        }
    }
}

/// What only HMC has of the multigrid.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MultigridHmcBlock {
    /// Improve the test vectors again after this many force solves; 0: only when the
    /// pseudofermions are drawn (a trajectory starts).
    #[serde(default)]
    pub(crate) refresh_interval: usize,
    /// Wilson and clover: start every setup but the first from the test vectors of the one
    /// before, improved `number_of_reuse_iterations` times; false: from random vectors.
    /// Domain-wall fermions always go on from their vectors (by the bootstrap of `Setup`).
    #[serde(default = "yes")]
    pub(crate) reuse_vectors: bool,
    #[serde(default)]
    pub(crate) number_of_reuse_iterations: usize,
    /// Domain-wall fermions: where each of the two solves on the even sites (through Ã† and
    /// Ã) stops, |r|²/|b|² (1e-10); then they are repeated on the residual of D†D until
    /// `Solver.convergence_criterion_squared`.
    pub convergence_criterion_squared_inner: Option<f64>,
    /// Domain-wall fermions: keep P and the coarse operators from solve to solve, and make them
    /// again when the GCR iterations per solve have grown to this many times those after the
    /// last rebuild (`hmc::DwfMultigridSetup::rebuild_growth`); left out: for every solve.
    pub rebuild_iteration_growth: Option<f64>,
}

impl Default for MultigridHmcBlock {
    fn default() -> Self {
        Self {
            refresh_interval: 0,
            reuse_vectors: true,
            number_of_reuse_iterations: 0,
            convergence_criterion_squared_inner: None,
            rebuild_iteration_growth: None,
        }
    }
}

/// det(D†D)^(numerator/denominator) by a rational approximation with `number_of_poles` poles
/// for the eigenvalues of |D| in [`lower_bound`, `upper_bound`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RationalBlock {
    pub number_of_poles: usize,
    #[serde(default = "one_i")]
    pub exponent_numerator: i32,
    #[serde(default = "two_i")]
    pub(crate) exponent_denominator: i32,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// A directory with Bridge++'s coefficient files, to read instead of computing them.
    pub coefficients: Option<PathBuf>,
}

fn text() -> GaugeType {
    GaugeType::Text
}
fn yes() -> bool {
    true
}
fn exp_order() -> usize {
    8
}
fn omelyan_lambda() -> f64 {
    0.1931833275
}
fn md_tolerance() -> f64 {
    1e-24
}
fn md_iterations() -> usize {
    10000
}
fn kept() -> usize {
    32
}
fn no_preconditioner() -> PreconditionerType {
    PreconditionerType::None
}
fn one_i() -> i32 {
    1
}
fn two_i() -> i32 {
    2
}

fn invalid<T>(message: String) -> Result<T, CliError> {
    Err(CliError::Invalid(message))
}

/// Whether the text has the sections of an HMC file.
pub(crate) fn is_hmc_text(text: &str, toml: bool) -> bool {
    const KEY: &str = "HMC_General";
    if toml {
        text.parse::<toml::Table>()
            .is_ok_and(|t| t.contains_key(KEY))
    } else {
        serde_yaml_ng::from_str::<serde_yaml_ng::Value>(text)
            .ok()
            .and_then(|v| v.as_mapping().map(|m| m.contains_key(KEY)))
            .unwrap_or(false)
    }
}

impl HmcParams {
    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))
    }

    pub fn from_toml(text: &str) -> Result<Self, CliError> {
        toml::from_str(text).map_err(|e| CliError::Parse(e.to_string()))
    }

    /// The run the file describes.
    pub fn to_config(&self) -> Result<HmcConfig, CliError> {
        let device =
            match self.main.backend {
                Backend::Cpu => false,
                Backend::CudaBridge if cfg!(feature = "cuda-bridge") => true,
                Backend::CudaBridge => return invalid(
                    "Main.backend cuda_bridge needs a program built with the feature cuda-bridge"
                        .into(),
                ),
                _ => return invalid("Main.backend: HMC runs on cpu or cuda_bridge".into()),
            };
        let run = &self.hmc;
        let start = match run.gauge_config_status {
            StartStatus::Continue => StartConfig::File {
                file: run.config_filename_input.clone().ok_or_else(|| {
                    CliError::Invalid("HMC.config_filename_input: needed to continue".into())
                })?,
                format: GaugeFormat::from(run.gauge_config_type_input),
            },
            StartStatus::ColdStart => StartConfig::Cold,
            StartStatus::HotStart => StartConfig::Hot {
                seed: run.seed_for_random_number.ok_or_else(|| {
                    CliError::Invalid("HMC.seed_for_random_number: needed for a hot start".into())
                })?,
            },
        };
        let (mut actions, mut names) = (Vec::new(), Vec::new());
        if let Some(g) = self.action_g {
            names.push("Action_G".to_string());
            actions.push(match g.action_type {
                GaugeActionType::Plaquette => ActionConfig::Plaquette {
                    beta: g.beta,
                    level: g.level_of_action,
                },
                GaugeActionType::Rectangle => match (g.c_plaq, g.c_rect) {
                    (Some(c_plaq), Some(c_rect)) => ActionConfig::Rectangle {
                        beta: g.beta,
                        c_plaq,
                        c_rect,
                        level: g.level_of_action,
                    },
                    _ => {
                        return invalid(
                            "Action_G: Action_G_Rectangle needs c_plaq and c_rect".into(),
                        )
                    }
                },
            });
        }
        let fermions = [
            &self.action_f_1,
            &self.action_f_2,
            &self.action_f_3,
            &self.action_f_4,
            &self.action_f_5,
            &self.action_f_6,
        ];
        let mut ended = false;
        for (i, f) in fermions.into_iter().enumerate() {
            let name = format!("Action_F_{}", i + 1);
            match f {
                Some(_) if ended => {
                    return invalid(format!("{name}: Action_F_{i} is missing"));
                }
                Some(f) => {
                    actions.push(f.to_config(&name, self.main.lattice_size)?);
                    names.push(name);
                }
                None => ended = true,
            }
        }
        // The force through the smearing needs its derivative: stout only.
        let ape = matches!(
            self.gauge_smearing.map(|g| g.to_config()),
            Some(super::config::GaugeSmearingConfig::Ape { .. })
        );
        if let Some(i) = actions.iter().position(|a| {
            ape && matches!(
                a,
                ActionConfig::TwoFlavor { smeared: true, .. }
                    | ActionConfig::TwoFlavorRatio { smeared: true, .. }
                    | ActionConfig::Rational { smeared: true, .. }
            )
        }) {
            return invalid(format!(
                "{}: the force on smeared links needs stout smearing; GaugeSmearing.smear_type APE is not differentiable",
                names[i]
            ));
        }
        let grid = self.main.grid_size;
        if device {
            if grid != [1; ND] {
                let local: Vec<usize> = (0..ND)
                    .map(|mu| self.main.lattice_size[mu] / grid[mu].max(1))
                    .collect();
                if local.iter().any(|l| l % 2 != 0) {
                    return invalid(format!(
                        "Main.grid_size {grid:?}: the local lattice {local:?} of a process must have even extents on the device"
                    ));
                }
            }
            if grid != [1; ND] {
                let local = core::array::from_fn(|mu| self.main.lattice_size[mu] / grid[mu].max(1));
                if let Some(why) = super::hmc::refused_on_several(&actions, local) {
                    return invalid(format!("Main.grid_size {grid:?}: {why}"));
                }
            }
            for (i, a) in actions.iter().enumerate() {
                match a {
                    ActionConfig::Plaquette { .. } | ActionConfig::Rectangle { .. } => {}
                    ActionConfig::TwoFlavorRatio {
                        fermion: HmcFermion::DomainWall { .. },
                        ..
                    } => {}
                    ActionConfig::TwoFlavor {
                        fermion: HmcFermion::DomainWall { .. },
                        ..
                    } => {}
                    ActionConfig::TwoFlavor { solver, .. }
                    | ActionConfig::TwoFlavorRatio { solver, .. }
                    | ActionConfig::Rational { solver, .. } => {
                        if solver.asks_for(super::hmc::HmcPreconditioner::Multigrid) {
                            if let Some(super::hmc::HmcMultigrid::Wilson { params, .. }) =
                                &solver.multigrid
                            {
                                if params.levels[0] > 0 {
                                    return invalid(format!(
                                        "{}.MultiGrid.MultiGrid_Level2: the levels below the coarse lattice of the Wilson and clover multigrid are not on the device yet",
                                        names[i]
                                    ));
                                }
                                if params.precondition_coarse_diagonal {
                                    return invalid(format!(
                                        "{}.MultiGrid.MultiGrid_Level1.precondition_diagonal: the coarse operator preconditioned by its blocks is not on the device for Wilson and clover fermions; leave it false",
                                        names[i]
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(HmcConfig {
            lattice: self.main.lattice_size,
            process_grid: (grid != [1; ND]).then_some(grid),
            cpu_thread: self.main.cpu_thread,
            start,
            gauge_smearing: self.gauge_smearing.map(|g| g.to_config()),
            hmc: HmcBlock {
                trajectories: run.number_of_trajectories,
                first_trajectory: run.trajectory_number,
                random_state: run.rand_filename_input.clone(),
                trajectory_length: self.general.trajectory_length,
                integrator: IntegratorConfig {
                    scheme: match self.integrator.integrator {
                        IntegratorType::Leapfrog => SchemeConfig::Leapfrog,
                        IntegratorType::Omelyan => SchemeConfig::Omelyan,
                    },
                    steps: self.integrator.number_of_steps.clone(),
                    lambda: self.integrator.lambda_omelyan,
                    exp_order: self.integrator.order_of_exp_ip,
                },
                metropolis: self.general.metropolis_test,
                reversibility_checks: self.general.reversibility_checks,
            },
            actions,
            action_names: names,
            output: HmcOutput {
                save_every: run.save_config_interval,
                file: run.config_filename_output.clone(),
                format: GaugeFormat::from(run.gauge_config_type_output),
                random_state: run.rand_filename_output.clone(),
            },
            flow: self.gradient_flow.map(FlowBlock::to_config).transpose()?,
            device,
        })
    }
}

/// The operator of an action from its `Fopr` section.
fn fermion(f: &FoprBlock, name: &str) -> Result<HmcFermion, CliError> {
    let need = |v: Option<f64>, key: &str| {
        v.ok_or_else(|| {
            CliError::Invalid(format!("{name}: {key} is needed for {}", f.fermion_type))
        })
    };
    let boundary = f.boundary_config();
    Ok(match f.fermion_type {
        FermionType::Wilson => HmcFermion::Wilson {
            kappa: need(f.hopping_parameter, "hopping_parameter")?,
            boundary,
        },
        FermionType::Clover => HmcFermion::Clover {
            kappa: need(f.hopping_parameter, "hopping_parameter")?,
            csw: need(f.clover_coefficient, "clover_coefficient")?,
            boundary,
        },
        FermionType::DomainWall => HmcFermion::DomainWall {
            ls: f.extent_of_5th_dimension.ok_or_else(|| {
                CliError::Invalid(format!("{name}: extent_of_5th_dimension is needed"))
            })?,
            m5: need(f.domain_wall_height, "domain_wall_height")?,
            mass: need(f.quark_mass, "quark_mass")?,
            b: f.coefficient_b,
            c: f.coefficient_c,
            boundary,
        },
    })
}

impl FermionActionBlock {
    fn to_config(&self, name: &str, lattice: [usize; ND]) -> Result<ActionConfig, CliError> {
        let s = &self.solver;
        if s.moved_multigrid.is_some() {
            return invalid(format!(
                "{name}.Solver.MultiGrid: the multigrid is the section {name}.MultiGrid now, that of the propagator files (MultiGrid_Level1: block_size, number_of_test_vectors; Setup: bootstrap_number_of_rounds for the setup iterations), and {name}.MultiGrid_HMC has refresh_interval, reuse_vectors and number_of_reuse_iterations"
            ));
        }
        let kind = match s.solver_type {
            None => None,
            Some(SolverType::Cg) => Some(HmcSolverKind::Cg),
            Some(SolverType::BiCgStab) => Some(HmcSolverKind::Bicgstab),
            Some(SolverType::Fgmres) => Some(HmcSolverKind::Fgmres),
            Some(SolverType::Gcr) => Some(HmcSolverKind::Gcr),
            Some(other) => {
                return invalid(format!(
                    "{name}.Solver.solver_type: CG, BiCGStab, FGMRES or GCR, not {other}"
                ))
            }
        };
        if s.chebyshev.is_some() && s.preconditioner_type != PreconditionerType::Chebyshev {
            return invalid(format!(
                "{name}.Solver.Chebyshev: for preconditioner_type Chebyshev (the smoother of the multigrid is MultiGrid.Smoother)"
            ));
        }
        let chebyshev = s
            .chebyshev
            .as_ref()
            .map_or_else(HmcChebyshev::default, |c| HmcChebyshev {
                steps: c.number_of_steps,
                ratio: c.interval_ratio,
            });
        let solver = HmcSolver {
            preconditioner: Some(match s.preconditioner_type {
                PreconditionerType::None => HmcPreconditioner::None,
                PreconditionerType::Chebyshev => HmcPreconditioner::Chebyshev,
                PreconditionerType::Multigrid => HmcPreconditioner::Multigrid,
            }),
            chebyshev,
            multigrid: match (&self.multigrid, s.preconditioner_type) {
                (Some(mg), PreconditionerType::Multigrid) => {
                    Some(self.hmc_multigrid(mg, name, lattice)?)
                }
                (None, PreconditionerType::Multigrid) => {
                    return invalid(format!(
                        "{name}.Solver.preconditioner_type: MG needs the section {name}.MultiGrid"
                    ))
                }
                (Some(_), _) => {
                    return invalid(format!(
                        "{name}.MultiGrid: for Solver.preconditioner_type MG"
                    ))
                }
                (None, _) => None,
            },
            kind,
            mixed: match s.precision {
                Some(SolverPrecision::Mixed) => {
                    Some(s.convergence_criterion_squared_inner.unwrap_or(1e-10))
                }
                _ if s.convergence_criterion_squared_inner.is_some() => {
                    return invalid(format!(
                        "{name}.Solver.convergence_criterion_squared_inner: for precision mixed"
                    ))
                }
                _ => None,
            },
            neff_alpha: self.fopr.neff_alpha,
            restart: s.number_of_kept_directions,
            tolerance_squared: s.convergence_criterion_squared,
            energy_tolerance_squared: s
                .convergence_criterion_squared_h
                .unwrap_or(s.convergence_criterion_squared),
            max_iterations: s.maximum_number_of_iteration,
            initial_guess: s.number_of_initial_guess,
        };
        let fermion_of = fermion(&self.fopr, &format!("{name}.Fopr"))?;
        let level = self.level_of_action;
        let even_odd = match self.operator_type {
            None | Some(OperatorType::Lexical) => false,
            Some(OperatorType::EvenOdd) => true,
            Some(other) => {
                return invalid(format!(
                    "{name}.operator_type: Lexical or Even_odd, not {other}"
                ))
            }
        };
        let domain_wall = self.fopr.fermion_type.is_domain_wall();
        let solved = match (domain_wall, even_odd, s.preconditioner_type) {
            (true, _, PreconditionerType::Multigrid) => OperatorType::PvRight,
            (true, true, _) => OperatorType::NormalLeft,
            (false, true, _) => OperatorType::EvenOdd,
            _ => OperatorType::Lexical,
        };
        if let Some(o) = s.operator_type.filter(|o| *o != solved) {
            return invalid(format!(
                "{name}.Solver.operator_type: this action solves {solved}, not {o}"
            ));
        }
        if even_odd && self.action_type == FermionActionType::Rational {
            return invalid(format!(
                "{name}.operator_type: Even_odd is for action_type Standard or Ratio"
            ));
        }
        if self.fopr_denominator.is_some() && self.action_type != FermionActionType::Ratio {
            return invalid(format!("{name}.Fopr_denominator: for action_type Ratio"));
        }
        if self.rational.is_some() != (self.action_type == FermionActionType::Rational) {
            return invalid(format!(
                "{name}: the section Rational goes with action_type Rational"
            ));
        }
        Ok(match self.action_type {
            FermionActionType::Standard => ActionConfig::TwoFlavor {
                fermion: fermion_of,
                level,
                smeared: self.smeared,
                even_odd,
                solver,
            },
            FermionActionType::Ratio => ActionConfig::TwoFlavorRatio {
                fermion: fermion_of,
                denominator: match &self.fopr_denominator {
                    Some(d) => Some(fermion(d, &format!("{name}.Fopr_denominator"))?),
                    None => None,
                },
                level,
                smeared: self.smeared,
                even_odd,
                solver,
            },
            FermionActionType::Rational => {
                let r = self.rational.as_ref().expect("checked above");
                ActionConfig::Rational {
                    fermion: fermion_of,
                    level,
                    poles: r.number_of_poles,
                    bounds: [r.lower_bound, r.upper_bound],
                    power: [r.exponent_numerator, r.exponent_denominator],
                    coefficients: r.coefficients.clone(),
                    smeared: self.smeared,
                    solver,
                }
            }
        })
    }
}

impl FermionActionBlock {
    /// The multigrid of the action from its `MultiGrid` and `MultiGrid_HMC` sections.
    fn hmc_multigrid(
        &self,
        mg: &MultigridBlock,
        name: &str,
        lattice: [usize; ND],
    ) -> Result<HmcMultigrid, CliError> {
        let at = format!("{name}.MultiGrid");
        let h = &self.multigrid_hmc;
        let refused = |what: &str, why: &str| invalid(format!("{at}.{what}: {why}"));
        if mg.setup.checkpoint.is_some() {
            return refused(
                "Setup.checkpoint",
                "HMC makes its test vectors for the links of each trajectory",
            );
        }
        if self.fopr.fermion_type.is_domain_wall() {
            if !h.reuse_vectors || h.number_of_reuse_iterations != 0 {
                return invalid(format!(
                    "{name}.MultiGrid_HMC: reuse_vectors and number_of_reuse_iterations are for Wilson and clover fermions; domain-wall fermions go on from their test vectors by the bootstrap of {at}.Setup"
                ));
            }
            mg.check_domain_wall(&at, lattice, false)?;
            Ok(HmcMultigrid::DomainWall {
                settings: mg.mdwf_settings(PrecisionType::Double),
                refresh_every: h.refresh_interval,
                inner_tolerance_squared: h.convergence_criterion_squared_inner.unwrap_or(1e-10),
                rebuild_growth: match h.rebuild_iteration_growth {
                    Some(g) if g.is_nan() || g <= 1.0 => {
                        return invalid(format!(
                            "{name}.MultiGrid_HMC.rebuild_iteration_growth: above 1"
                        ))
                    }
                    g => g.unwrap_or(0.0),
                },
            })
        } else {
            if h.convergence_criterion_squared_inner.is_some()
                || h.rebuild_iteration_growth.is_some()
            {
                return invalid(format!(
                    "{name}.MultiGrid_HMC: convergence_criterion_squared_inner and rebuild_iteration_growth are for domain-wall fermions"
                ));
            }
            let mut unused = Vec::new();
            mg.check_wilson(&at, self.fopr.fermion_type, lattice, &mut unused)?;
            if let Some(note) = unused.into_iter().next() {
                return Err(CliError::Invalid(note));
            }
            Ok(HmcMultigrid::Wilson {
                params: mg.wilson().params(false),
                refresh_every: h.refresh_interval,
                reuse: h.reuse_vectors.then_some(h.number_of_reuse_iterations),
            })
        }
    }
}
