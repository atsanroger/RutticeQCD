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

use super::config::GaugeFormat;
use super::hmc::{
    ActionConfig, FlowMonitor, HmcBlock, HmcChebyshev, HmcConfig, HmcFermion, HmcMultigrid,
    HmcOutput, HmcPreconditioner, HmcSolver, HmcSolverKind, IntegratorConfig, SchemeConfig,
    StartConfig,
};
use super::params::{
    Backend, ChebyshevBlock, FermionType, FoprBlock, GaugeSmearingBlock, GaugeType, MainBlock,
    OperatorType, PreconditionerType, SolverType,
};
use super::CliError;
use crate::gauge::flow::WilsonFlow;
use crate::{keyword_enum, ND};
use serde::Deserialize;
use std::path::PathBuf;

keyword_enum! {
    pub enum StartStatus {
        /// From the file `config_filename_input`.
        Continue => ["Continue", "File"],
        /// Every link the identity.
        ColdStart => ["Cold_start", "Cold"],
        /// Random links, from `seed_for_random_number`.
        HotStart => ["Hot_start", "Hot"],
    }
}

keyword_enum! {
    pub enum IntegratorType {
        Leapfrog => ["Leapfrog"],
        Omelyan => ["Omelyan"],
    }
}

keyword_enum! {
    pub enum GaugeActionType {
        Plaquette => ["Action_G_Plaq", "Plaquette", "Plaq"],
        Rectangle => ["Action_G_Rectangle", "Rectangle"],
    }
}

keyword_enum! {
    pub enum FermionActionType {
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
pub struct HmcParams {
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
    pub action_g: Option<GaugeActionBlock>,
    #[serde(rename = "Action_F_1")]
    pub action_f_1: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_2")]
    pub action_f_2: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_3")]
    pub action_f_3: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_4")]
    pub action_f_4: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_5")]
    pub action_f_5: Option<FermionActionBlock>,
    #[serde(rename = "Action_F_6")]
    pub action_f_6: Option<FermionActionBlock>,
    /// The Wilson flow measured every `measurement_interval` trajectories.
    #[serde(rename = "GradientFlow")]
    pub gradient_flow: Option<FlowBlock>,
}

/// The Wilson flow along the run: t0 and the topological charge after every
/// `measurement_interval` trajectories (keys of Bridge++'s `GradientFlow` where it has them).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowBlock {
    pub measurement_interval: usize,
    #[serde(default = "flow_step")]
    pub step_size: f64,
    #[serde(default = "flow_t_max")]
    pub max_flow_time: f64,
    #[serde(rename = "order_of_approx_for_exp_iP", default = "flow_exp_order")]
    pub order_of_exp_ip: usize,
    /// Only 3 (Lüscher's scheme).
    #[serde(rename = "order_of_RungeKutta", default = "flow_rk_order")]
    pub order_of_runge_kutta: usize,
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
        if self.measurement_interval == 0 || self.step_size <= 0.0 {
            return invalid(
                "GradientFlow: measurement_interval and step_size must be positive".into(),
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
pub struct RunBlock {
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
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralBlock {
    pub trajectory_length: f64,
    #[serde(rename = "Metropolis_test", default = "yes")]
    pub metropolis_test: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegratorBlock {
    pub integrator: IntegratorType,
    /// Steps of each level, the outermost first.
    pub number_of_steps: Vec<usize>,
    #[serde(rename = "order_of_exp_iP", default = "exp_order")]
    pub order_of_exp_ip: usize,
    #[serde(rename = "lambda_Omelyan", default = "omelyan_lambda")]
    pub lambda_omelyan: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeActionBlock {
    pub action_type: GaugeActionType,
    pub beta: f64,
    pub c_plaq: Option<f64>,
    pub c_rect: Option<f64>,
    #[serde(default)]
    pub level_of_action: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FermionActionBlock {
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
    pub fopr_denominator: Option<FoprBlock>,
    #[serde(rename = "Solver", default)]
    pub solver: HmcSolverBlock,
    #[serde(rename = "Rational")]
    pub rational: Option<RationalBlock>,
}

/// The solver of an action's forces and energies.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmcSolverBlock {
    /// `CG` (the default of `Standard` and `Ratio`), `BiCGStab`, `FGMRES`, `GCR`. `Rational`
    /// without it: multi-shift CG; with `CG`: one solve per pole.
    pub solver_type: Option<SolverType>,
    /// Of the force solves.
    #[serde(default = "md_tolerance")]
    pub convergence_criterion_squared: f64,
    /// Of the energies at the ends of a trajectory; the same as the forces if not given.
    #[serde(rename = "convergence_criterion_squared_H")]
    pub convergence_criterion_squared_h: Option<f64>,
    #[serde(default = "md_iterations")]
    pub maximum_number_of_iteration: usize,
    /// The cycle of FGMRES, the directions GCR keeps.
    #[serde(default = "kept")]
    pub number_of_kept_directions: usize,
    /// `Standard`: start each force solve from the last this many solutions.
    #[serde(default)]
    pub number_of_initial_guess: usize,
    #[serde(default = "no_preconditioner")]
    pub preconditioner_type: PreconditionerType,
    #[serde(rename = "Chebyshev")]
    pub chebyshev: Option<ChebyshevBlock>,
    #[serde(rename = "MultiGrid")]
    pub multigrid: Option<HmcMultigridBlock>,
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
            preconditioner_type: PreconditionerType::None,
            chebyshev: None,
            multigrid: None,
        }
    }
}

/// The multigrid of Wilson and clover fermions in the two-step solves.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmcMultigridBlock {
    /// The blocks of the coarse lattice; they must divide the lattice of a process.
    pub block_size: [usize; ND],
    #[serde(default = "mg_vectors")]
    pub number_of_test_vectors: usize,
    #[serde(default = "mg_setup")]
    pub number_of_setup_iterations: usize,
    /// Improve the test vectors again after this many force solves; 0: only when a
    /// trajectory starts.
    #[serde(default)]
    pub refresh_interval: usize,
    /// Start every setup but the first from the test vectors of the one before.
    #[serde(default = "yes")]
    pub reuse_vectors: bool,
    #[serde(default)]
    pub number_of_reuse_iterations: usize,
}

/// det(D†D)^(numerator/denominator) by a rational approximation with `number_of_poles` poles
/// for the eigenvalues of |D| in [`lower_bound`, `upper_bound`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalBlock {
    pub number_of_poles: usize,
    #[serde(default = "one_i")]
    pub exponent_numerator: i32,
    #[serde(default = "two_i")]
    pub exponent_denominator: i32,
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
fn mg_vectors() -> usize {
    16
}
fn mg_setup() -> usize {
    4
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
pub fn is_hmc_text(text: &str, toml: bool) -> bool {
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
        let device = match self.main.backend {
            Backend::Cpu => false,
            Backend::CudaBridge => true,
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
        let mut actions = Vec::new();
        if let Some(g) = self.action_g {
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
                Some(f) => actions.push(f.to_config(&name)?),
                None => ended = true,
            }
        }
        let grid = self.main.grid_size;
        if device {
            if grid != [1; ND] {
                return invalid("Main.backend: cuda_bridge runs on one process".into());
            }
            for (i, a) in actions.iter().enumerate() {
                match a {
                    ActionConfig::Plaquette { .. } | ActionConfig::Rectangle { .. } => {}
                    ActionConfig::TwoFlavorRatio {
                        fermion: HmcFermion::DomainWall { .. },
                        solver,
                        ..
                    }
                    | ActionConfig::TwoFlavor {
                        fermion: HmcFermion::DomainWall { .. },
                        even_odd: false,
                        solver,
                        ..
                    } if !solver.asks_for(HmcPreconditioner::Multigrid) => {}
                    _ => {
                        return invalid(format!(
                            "fermion action {}: with Main.backend cuda_bridge only the Standard (Lexical) and the Ratio (Lexical or Even_odd) actions of domain-wall fermions, without the multigrid, are on the device",
                            i
                        ))
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
            },
            actions,
            output: HmcOutput {
                save_every: run.save_config_interval,
                file: run.config_filename_output.clone(),
                format: GaugeFormat::from(run.gauge_config_type_output),
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
    fn to_config(&self, name: &str) -> Result<ActionConfig, CliError> {
        let s = &self.solver;
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
            multigrid: s.multigrid.map(|m| HmcMultigrid {
                block: m.block_size,
                test_vectors: m.number_of_test_vectors,
                setup_iterations: m.number_of_setup_iterations,
                refresh_every: m.refresh_interval,
                reuse_vectors: m.reuse_vectors,
                reuse_iterations: m.number_of_reuse_iterations,
            }),
            kind,
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
