//! The parameter file. The same structure is accepted as TOML (`.toml`) or YAML
//! (`.yaml`, `.yml`); unknown keys are errors, so a misspelt name is reported when the file
//! is read rather than silently ignored.

use super::CliError;
use crate::gauge::CoulombParams;
use crate::solver::preconditioner::multigrid::{
    CoarseSolverChoice, MultigridParams, SmootherChoice,
};
use crate::solver::Stop;
use crate::{Boundary, ND};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Lattice extents [Lx, Ly, Lz, Lt].
    pub lattice: [usize; ND],
    /// Number of processes in each direction when the program runs on several. Their
    /// product must be the number of processes. Chosen automatically when absent.
    #[serde(default)]
    pub process_grid: Option<[usize; ND]>,
    pub gauge: GaugeConfig,
    /// Gauge smearing: the fermion operators are made of smeared links (stout or APE).
    /// Not to be confused with the smearing of quark sources (operator smearing), which is
    /// `quark.source`; a key called just `smearing` or `smear` is refused for that reason.
    #[serde(default)]
    pub gauge_smearing: Option<GaugeSmearingConfig>,
    /// Refused names (see `gauge_smearing`).
    #[serde(default)]
    pub(crate) smearing: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    pub(crate) smear: Option<serde::de::IgnoredAny>,
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
    /// Refused: gauge smearing is the top-level `gauge_smearing`.
    #[serde(default)]
    pub(crate) smearing: Option<serde::de::IgnoredAny>,
}

/// Gauge smearing, applied `steps` times to the links (after gauge fixing) before the
/// fermion operators are made of them. ρ is the weight of the staples of every plane, or of
/// the spatial planes only (`spatial_only`: the time links are kept).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum GaugeSmearingConfig {
    /// V = exp(iQ) U, iQ the traceless anti-Hermitian part of ρ (staples) U†.
    Stout {
        rho: f64,
        steps: usize,
        #[serde(default)]
        spatial_only: bool,
    },
    /// V = the SU(3) matrix closest to (1 − ρ) U + ρ (staples), found by SU(2) subgroups.
    Ape {
        rho: f64,
        steps: usize,
        #[serde(default)]
        spatial_only: bool,
        /// The projection to SU(3) stops when a sweep over the subgroups changes the links
        /// by less than this (1 − ⟨Re tr ΔG⟩/3) ...
        #[serde(default = "ape_tolerance")]
        tolerance: f64,
        /// ... and fails after this many sweeps.
        #[serde(default = "ape_max_iterations")]
        max_iterations: usize,
    },
}

fn ape_tolerance() -> f64 {
    1.0e-15
}
fn ape_max_iterations() -> usize {
    100
}

impl GaugeSmearingConfig {
    pub fn smearing(&self) -> crate::gauge::GaugeSmearing {
        use crate::gauge::smear::{spatial, uniform, Ape, Smear, Stout};
        let weights = |rho: f64, spatial_only: bool| {
            if spatial_only {
                spatial(rho)
            } else {
                uniform(rho)
            }
        };
        match *self {
            GaugeSmearingConfig::Stout {
                rho,
                steps,
                spatial_only,
            } => crate::gauge::GaugeSmearing {
                smear: Smear::Stout(Stout {
                    rho: weights(rho, spatial_only),
                }),
                steps,
            },
            GaugeSmearingConfig::Ape {
                rho,
                steps,
                spatial_only,
                tolerance,
                max_iterations,
            } => crate::gauge::GaugeSmearing {
                smear: Smear::Ape(Ape {
                    rho: weights(rho, spatial_only),
                    tolerance,
                    max_iterations,
                }),
                steps,
            },
        }
    }
}

/// The message for a key that says "smearing" without saying of what.
pub(crate) const AMBIGUOUS_SMEARING: &str = "\"smearing\" alone is ambiguous: gauge (link) smearing is `gauge_smearing` (type stout or ape); the smearing of quark sources (operator smearing) is the source type of each quark (`quark.source`, e.g. exponential)";

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
    /// The ILDG format: binary data in a LIME container.
    Ildg,
    /// A lattice-colour-matrix record written by the SciDAC QIO library, in a LIME
    /// container.
    Scidac,
    /// The dataset `gauge` of an HDF5 file; needs a program built with the feature `hdf5`.
    Hdf5,
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
    /// Smeared sources (source smearing, not gauge smearing) on the time slice of
    /// `position`: the wave function exp(-slope * r^power), normalised.
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
    /// Solve through the Schur complement on the even sites. With the multigrid its coarse
    /// system is then solved on the even coarse sites too (where the coarse lattice has
    /// even extents); `false` is the whole lattice on both.
    #[serde(default = "yes")]
    pub even_odd: bool,
    /// Stop when |r|^2 / |b|^2 is below this for every right-hand side.
    pub tolerance_squared: f64,
    /// Iterations before the true residual is recomputed and the solver restarts.
    pub max_iterations: usize,
    #[serde(default = "default_restarts")]
    pub max_restarts: usize,
    /// Lattice sites computed together, as the lanes of one SIMD register: 1 (the usual
    /// layout of the fields), 2, 4 or 8. In double precision 4 fills a 256-bit register
    /// and 8 a 512-bit one.
    #[serde(default = "one")]
    pub simd: usize,
    /// Precondition with two-level multigrid.
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
    /// The coarse solver.
    #[serde(default)]
    pub coarse_solver: CoarseSolverKind,
    /// The coarse solve stops at this |r|^2 / |b|^2 ...
    #[serde(default = "mg_coarse_tolerance")]
    pub coarse_tolerance_squared: f64,
    /// ... or after this many iterations (for GMRES: the m of GMRES(m), no restarts).
    #[serde(default = "mg_coarse_iterations")]
    pub coarse_max_iterations: usize,
    /// Precondition the coarse operator by the inverse of its diagonal blocks.
    #[serde(default)]
    pub precondition_coarse_diagonal: bool,
    /// The smoother.
    #[serde(default)]
    pub smoother: SmootherKind,
    /// Smooth before the coarse correction too, instead of starting from x = b.
    #[serde(default)]
    pub pre_smoothing: bool,
    /// SAP: cycles ...
    #[serde(default = "mg_smoother_cycles")]
    pub smoother_cycles: usize,
    /// ... and minimal-residual steps per block in each cycle.
    #[serde(default = "mg_smoother_block_steps")]
    pub smoother_block_steps: usize,
    /// GMRES: the m of GMRES(m) ...
    #[serde(default = "mg_smoother_iterations")]
    pub smoother_iterations: usize,
    /// ... and an earlier stop at this |r|^2 / |b|^2.
    #[serde(default = "mg_smoother_tolerance")]
    pub smoother_tolerance_squared: f64,
    /// Precision the preconditioner works in; the outer solve is always double.
    #[serde(default)]
    pub precision: Precision,
    /// Seed of the random initial test vectors.
    #[serde(default = "mg_seed")]
    pub seed: u64,
    /// Levels below the coarse lattice for the coarse solves: the super-block sizes of
    /// the second and the third level (coarse sites per direction).
    pub level2_super_block: Option<usize>,
    pub level3_super_block: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmootherKind {
    /// Schwarz alternating procedure (the Bridge++ clover multigrid).
    #[default]
    Sap,
    /// GMRES(m) on the operator (the CUDA domain-wall multigrid).
    Gmres,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoarseSolverKind {
    #[default]
    Bicgstab,
    Gmres,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    Single,
    Double,
}

impl MultigridConfig {
    /// `even_odd`: that of the solver (`SolverConfig::even_odd`), which the coarse solves
    /// follow.
    pub fn params(&self, even_odd: bool) -> MultigridParams {
        MultigridParams {
            block: self.block,
            num_test_vectors: self.test_vectors,
            setup_iterations: self.setup_iterations,
            coarse_stop: Stop::simple(self.coarse_tolerance_squared, self.coarse_max_iterations),
            coarse_solver: match self.coarse_solver {
                CoarseSolverKind::Bicgstab => CoarseSolverChoice::BiCgStab,
                CoarseSolverKind::Gmres => CoarseSolverChoice::Gmres,
            },
            smoother: match self.smoother {
                SmootherKind::Sap => SmootherChoice::Sap {
                    cycles: self.smoother_cycles,
                    block_steps: self.smoother_block_steps,
                },
                SmootherKind::Gmres => SmootherChoice::Gmres {
                    iterations: self.smoother_iterations,
                    tolerance_sqr: self.smoother_tolerance_squared,
                },
            },
            pre_smoothing: self.pre_smoothing,
            precondition_coarse_diagonal: self.precondition_coarse_diagonal,
            coarse_even_odd: even_odd,
            seed: self.seed,
            levels: [
                self.level2_super_block.unwrap_or(0),
                self.level2_super_block
                    .and(self.level3_super_block)
                    .unwrap_or(0),
                0,
            ],
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
fn bridge_sap() -> (usize, usize) {
    match bridge_mg().smoother {
        SmootherChoice::Sap {
            cycles,
            block_steps,
        } => (cycles, block_steps),
        SmootherChoice::Gmres { .. } => unreachable!("the Bridge++ clover smoother is SAP"),
    }
}
fn mg_smoother_cycles() -> usize {
    bridge_sap().0
}
fn mg_smoother_block_steps() -> usize {
    bridge_sap().1
}
/// GMRES(m) smoother of the CUDA domain-wall test: `smoother_number_of_iteration: 4`,
/// `smoother_convergence_criterion_squared: 1.0e-3`.
fn mg_smoother_iterations() -> usize {
    4
}
fn mg_smoother_tolerance() -> f64 {
    1e-3
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

fn one() -> usize {
    1
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
    /// Meson channels to compute: `"all"` or a list of names. All of them when absent.
    #[serde(default)]
    pub channels: ChannelsConfig,
}

/// Which meson channels to compute.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(try_from = "ChannelsText")]
pub enum ChannelsConfig {
    /// Every channel the program knows.
    #[default]
    All,
    List(Vec<String>),
}

/// What may be written for `channels` in a parameter file.
#[derive(Deserialize)]
#[serde(untagged)]
enum ChannelsText {
    Word(String),
    List(Vec<String>),
}

impl TryFrom<ChannelsText> for ChannelsConfig {
    type Error = String;

    fn try_from(text: ChannelsText) -> Result<Self, String> {
        match text {
            ChannelsText::Word(word) if word == "all" => Ok(ChannelsConfig::All),
            ChannelsText::Word(word) => Err(format!(
                "channels must be \"all\" or a list of channel names, not {word:?}"
            )),
            ChannelsText::List(names) => Ok(ChannelsConfig::List(names)),
        }
    }
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
        let c: Self = toml::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        c.check_names()
    }

    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        let c: Self = serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        c.check_names()
    }

    /// Refuse the keys that do not say what is smeared.
    fn check_names(self) -> Result<Self, CliError> {
        if self.smearing.is_some() || self.smear.is_some() || self.gauge.smearing.is_some() {
            return Err(CliError::Invalid(AMBIGUOUS_SMEARING.into()));
        }
        Ok(self)
    }
}
