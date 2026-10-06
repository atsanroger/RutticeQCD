//! The parameter file. The same structure is accepted as TOML (`.toml`) or YAML
//! (`.yaml`, `.yml`); unknown keys are errors, so a misspelt name is reported when the file
//! is read rather than silently ignored.

use super::CliError;
use crate::gauge::CoulombParams;
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
