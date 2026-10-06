//! The parameter file. The same structure is accepted as TOML (`.toml`) or YAML
//! (`.yaml`, `.yml`); unknown keys are errors, so a misspelt name is reported when the file
//! is read rather than silently ignored.

use super::CliError;
use crate::gauge::CoulombParams;
use crate::solver::preconditioner::multigrid::MultigridParams;
use crate::solver::Stop;
use crate::{Boundary, ND};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Lattice extents [Lx, Ly, Lz, Lt].
    pub lattice: [usize; ND],
    pub gauge: GaugeConfig,
    pub solver: SolverConfig,
    /// One entry per quark flavour. Correlators are computed for every pair (i, j), i <= j.
    pub quark: Vec<QuarkConfig>,
    #[serde(default)]
    pub measure: MeasureConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeConfig {
    /// Path of the configuration; relative paths are taken from the parameter file's directory.
    pub file: PathBuf,
    #[serde(default)]
    pub format: GaugeFormat,
    /// Fix the gauge of the configuration before anything else; no fixing when absent.
    pub fixing: Option<GaugeFixingConfig>,
}

/// Gauge fixing. The defaults are those of the Bridge++ test `Spectrum_Clover_Hadron2ptFunction`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum GaugeFixingConfig {
    Coulomb {
        /// Stop when the violation is below this on every time slice.
        #[serde(default = "coulomb_tolerance")]
        tolerance_squared: f64,
        #[serde(default = "coulomb_max_iterations")]
        max_iterations: usize,
        /// Sweeps without overrelaxation at the start and after every reset.
        #[serde(default = "coulomb_naive_iterations")]
        naive_iterations: usize,
        /// Measure the violation every this many sweeps.
        #[serde(default = "coulomb_measure_interval")]
        measure_interval: usize,
        /// Apply a random gauge transformation to time slices not converged after this
        /// many sweeps.
        #[serde(default = "coulomb_reset_interval")]
        reset_interval: usize,
        #[serde(default = "coulomb_overrelaxation")]
        overrelaxation: f64,
        /// Seed for the random gauge transformations of a reset.
        #[serde(default = "coulomb_seed")]
        seed: u64,
    },
}

impl GaugeFixingConfig {
    pub fn coulomb_params(&self) -> CoulombParams {
        match *self {
            GaugeFixingConfig::Coulomb {
                tolerance_squared,
                max_iterations,
                naive_iterations,
                measure_interval,
                reset_interval,
                overrelaxation,
                seed,
            } => CoulombParams {
                max_iterations,
                naive_iterations,
                measure_interval,
                reset_interval,
                tolerance_sqr: tolerance_squared,
                overrelaxation,
                seed,
            },
        }
    }
}

fn coulomb_tolerance() -> f64 {
    CoulombParams::default().tolerance_sqr
}
fn coulomb_max_iterations() -> usize {
    CoulombParams::default().max_iterations
}
fn coulomb_naive_iterations() -> usize {
    CoulombParams::default().naive_iterations
}
fn coulomb_measure_interval() -> usize {
    CoulombParams::default().measure_interval
}
fn coulomb_reset_interval() -> usize {
    CoulombParams::default().reset_interval
}
fn coulomb_overrelaxation() -> f64 {
    CoulombParams::default().overrelaxation
}
fn coulomb_seed() -> u64 {
    CoulombParams::default().seed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GaugeFormat {
    /// Plain text, one real per line (the Bridge++ text format).
    #[default]
    BridgeText,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuarkConfig {
    pub fermion: FermionConfig,
    pub source: SourceConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FermionConfig {
    Wilson {
        kappa: f64,
        boundary: [BoundaryConfig; ND],
    },
    Clover {
        kappa: f64,
        csw: f64,
        boundary: [BoundaryConfig; ND],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryConfig {
    Periodic,
    Antiperiodic,
    Dirichlet,
}

impl From<BoundaryConfig> for Boundary {
    fn from(b: BoundaryConfig) -> Self {
        match b {
            BoundaryConfig::Periodic => Boundary::Periodic,
            BoundaryConfig::Antiperiodic => Boundary::Antiperiodic,
            BoundaryConfig::Dirichlet => Boundary::Dirichlet,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceConfig {
    /// Unit sources at one site.
    Point { position: [usize; ND] },
    /// Smeared sources on the time slice of `position`: exp(-slope * r^power), normalised.
    /// Gauge dependent; see `gauge.fixing`.
    Exponential {
        position: [usize; ND],
        slope: f64,
        power: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    #[serde(rename = "type")]
    pub kind: SolverKind,
    /// Solve through the Schur complement on the even sites.
    #[serde(default = "yes")]
    pub even_odd: bool,
    /// Stop when |r|^2 / |b|^2 is below this for every right-hand side.
    pub tolerance_squared: f64,
    /// Iterations before the true residual is recomputed and the solver restarts.
    pub max_iterations: usize,
    #[serde(default = "default_restarts")]
    pub max_restarts: usize,
    /// Precondition with two-level multigrid; needs `type = fgmres` and `even_odd = false`.
    pub multigrid: Option<MultigridConfig>,
}

/// The multigrid preconditioner. Everything except `block` defaults to the parameters of
/// the Bridge++ test `test_alt_Multigrid.yaml`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultigridConfig {
    /// Block size [bx, by, bz, bt]: the aggregates of the coarse lattice and the blocks of
    /// the smoother. Must divide the lattice.
    pub block: [usize; ND],
    #[serde(default = "mg_test_vectors")]
    pub test_vectors: usize,
    #[serde(default = "mg_setup_iterations")]
    pub setup_iterations: usize,
    /// The coarse solve (BiCGStab) stops at this |r|^2 / |b|^2 ...
    #[serde(default = "mg_coarse_tolerance")]
    pub coarse_tolerance_squared: f64,
    /// ... or after this many iterations.
    #[serde(default = "mg_coarse_iterations")]
    pub coarse_max_iterations: usize,
    #[serde(default = "mg_smoother_cycles")]
    pub smoother_cycles: usize,
    /// Minimal-residual steps per block in each smoother cycle.
    #[serde(default = "mg_smoother_block_steps")]
    pub smoother_block_steps: usize,
    /// Precision the preconditioner works in; the outer solve is always double.
    #[serde(default)]
    pub precision: Precision,
    /// Seed of the random initial test vectors.
    #[serde(default = "mg_seed")]
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    Single,
    Double,
}

impl MultigridConfig {
    pub fn params(&self) -> MultigridParams {
        MultigridParams {
            block: self.block,
            num_test_vectors: self.test_vectors,
            setup_iterations: self.setup_iterations,
            coarse_stop: Stop::simple(self.coarse_tolerance_squared, self.coarse_max_iterations),
            smoother_cycles: self.smoother_cycles,
            smoother_block_steps: self.smoother_block_steps,
            seed: self.seed,
        }
    }
}

fn bridge_mg() -> MultigridParams {
    MultigridParams::bridge([1; ND])
}
fn mg_test_vectors() -> usize {
    bridge_mg().num_test_vectors
}
fn mg_setup_iterations() -> usize {
    bridge_mg().setup_iterations
}
fn mg_coarse_tolerance() -> f64 {
    bridge_mg().coarse_stop.tolerance_sqr
}
fn mg_coarse_iterations() -> usize {
    bridge_mg().coarse_stop.max_iterations
}
fn mg_smoother_cycles() -> usize {
    bridge_mg().smoother_cycles
}
fn mg_smoother_block_steps() -> usize {
    bridge_mg().smoother_block_steps
}
fn mg_seed() -> u64 {
    bridge_mg().seed
}

fn yes() -> bool {
    true
}

fn default_restarts() -> usize {
    3
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverKind {
    /// Conjugate gradient on the normal equations.
    CgNormal,
    /// BiCGStab on the operator itself.
    Bicgstab,
    /// Flexible GMRES on the operator itself; `max_iterations` is the size of the Krylov
    /// space built before a restart. The solver to use with `multigrid`.
    Fgmres,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasureConfig {
    /// Meson channels to compute; all of them when absent.
    pub channels: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Also write the correlator table to this file; relative paths are taken from the
    /// parameter file's directory.
    pub file: Option<PathBuf>,
}

impl Config {
    /// Read a parameter file; the format is chosen by the file extension.
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match extension {
            "toml" => Self::from_toml(&text),
            "yaml" | "yml" => Self::from_yaml(&text),
            other => Err(CliError::Invalid(format!(
                "parameter file must end in .toml, .yaml or .yml, not {other:?}"
            ))),
        }
    }

    pub fn from_toml(text: &str) -> Result<Self, CliError> {
        toml::from_str(text).map_err(|e| CliError::Parse(e.to_string()))
    }

    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))
    }
}
