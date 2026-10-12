//! Parameter files in the style of Bridge++: sections `Main`, `Gauge`, `Fopr` and
//! `Solver`, keys spelt out (`maximum_number_of_iteration`, `convergence_criterion_squared`),
//! read from YAML (`.yaml`, `.yml`) or TOML (`.toml`). The values of choices are keywords
//! read without regard to case or `_` (see [`super::keyword`]); a misspelt key is an error.
//!
//! The multigrid is a section of its own, `MultiGrid`, next to `Solver`, which uses it with
//! `preconditioner_type: MG` and may keep it when it does not (switching is one keyword).
//! It can also come from a file of its own (`MultiGrid_file`), to share one multigrid among
//! many parameter files. `MultiGrid_Level1` is the coarse lattice; it names a solver for each
//! coarse factor, with the keys of `Solver`. `MultiGrid_Level2` to `MultiGrid_Level4`, if
//! given, are levels below it (`super_block_size`) in the solves of the quark factor, and with
//! `levels_for_pauli_villars : true` in those of the second factor too.
//!
//! ```yaml
//! Main:
//!   lattice_size : [12, 12, 12, 12]
//!   backend      : cpu
//! Gauge:
//!   gauge_config_type_input : Text
//!   config_filename_input   : ./conf.txt
//! Fopr:
//!   fermion_type            : Domainwall_5din
//!   quark_mass              : 0.005
//!   domain_wall_height      : 1.0
//!   extent_of_5th_dimension : 8
//!   neff_alpha              : 0.6
//! Solver:
//!   solver_type                   : GCR
//!   operator_type                 : PV_left
//!   convergence_criterion_squared : 1.0e-18
//!   preconditioner_type           : MG
//! MultiGrid:
//!   MultiGrid_Level1:
//!     block_size : [4, 4, 4, 4]
//! ```

use super::config::{
    BoundaryConfig, CoarseSolverKind, Config, FermionConfig, GaugeConfig, GaugeFixingConfig,
    GaugeFormat, GaugeSmearingConfig, MeasureConfig, MultigridConfig, OutputConfig, Precision,
    QuarkConfig, SmootherKind, SolverConfig, SolverKind, SourceConfig,
};
use super::CliError;
use crate::fermion::{EoForm, MdwfParams};
use crate::gauge::CoulombParams;
use crate::keyword_enum;
use crate::solver::preconditioner::multigrid::{
    BootstrapParams, CoarseKrylov, LevelBootstrap, LowModeParams, PvEoParams,
};
use crate::solver::Stop;
use crate::{Boundary, ND};
use serde::Deserialize;
use std::fmt;
use std::path::{Path, PathBuf};

keyword_enum! {
    /// Where the fields live and how the kernels run.
    pub enum Backend {
        Cpu => ["cpu"],
        /// Several sites in the lanes of one SIMD register (`simd_lanes`).
        Simd => ["simd"],
        /// Not implemented yet: the keyword is reserved.
        Gpu => ["gpu", "cuda"],
    }
}

keyword_enum! {
    pub enum VerboseLevel {
        Crucial => ["Crucial"],
        General => ["General"],
        Detailed => ["Detailed"],
        Paranoiac => ["Paranoiac"],
    }
}

keyword_enum! {
    pub enum GaugeType {
        /// One real per line, the Bridge++ text format.
        Text => ["Text", "Bridge_text", "Text_4x4x4x8"],
        /// Big-endian doubles without a header, the Bridge++ binary format.
        Binary => ["Binary"],
        /// The NERSC format: a text header, then the links.
        Nersc => ["NERSC"],
        Ildg => ["ILDG"],
        /// A lattice-colour-matrix record of the SciDAC QIO library, in a LIME container.
        Scidac => ["SciDAC", "QIO"],
        /// The dataset `gauge` of an HDF5 file (a program built with the feature `hdf5`).
        Hdf5 => ["HDF5"],
    }
}

keyword_enum! {
    pub enum FermionType {
        /// Möbius domain-wall fermions with a Wilson kernel (b = 1, c = 0: Shamir).
        DomainWall => [
            "Domainwall_5din",
            "Domainwall",
            "Domain_wall",
            "DWF",
            "MDWF",
            "Mobius",
            "Mobius_DWF",
            "Mobius_domainwall",
            "Mobius_domain_wall",
            "Domainwall_Mobius",
        ],
        /// Wilson fermions (`hopping_parameter`).
        Wilson => ["Wilson"],
        /// Wilson fermions with the clover term (`hopping_parameter`, `clover_coefficient`).
        Clover => ["Clover", "Wilson_clover", "Clover_Wilson"],
    }
}

impl FermionType {
    pub fn is_domain_wall(self) -> bool {
        self == FermionType::DomainWall
    }
}

keyword_enum! {
    pub enum GammaType {
        Dirac => ["Dirac"],
        Chiral => ["Chiral"],
    }
}

keyword_enum! {
    /// Which operator the solver works on, on the even sites.
    pub enum OperatorType {
        /// A′ = (D_PV C_PV⁻¹)† D C⁻¹ in its even-odd form; Neff's α cancels in it.
        PvRight => ["PV_right", "PV", "A_prime", "Aprime", "right"],
        /// The left-preconditioned even-odd A′, which sees α.
        PvLeft => ["PV_left", "left"],
        /// The normal equations L† L of the left Schur complement: CGNE, which sees α.
        NormalLeft => ["Normal_left", "Normal", "CGNE"],
        /// Wilson and clover fermions: the Schur complement on the even sites (CG: its
        /// normal equations). With the multigrid, the coarse system is solved on the even
        /// coarse sites too.
        EvenOdd => ["Even_odd", "eo", "Standard_eo", "Schur"],
        /// Wilson and clover fermions: D on the whole lattice (CG: its normal equations),
        /// and with the multigrid the coarse system on the whole coarse lattice.
        Lexical => ["Lexical", "Full", "Standard", "Whole"],
    }
}

keyword_enum! {
    pub enum SolverType {
        /// Flexible GCR keeping `number_of_kept_directions` directions.
        Gcr => ["GCR"],
        BiCgStab => ["BiCGStab", "BiCGStab_Cmplx", "BiCG_Stab"],
        /// Flexible GMRES; `number_of_kept_directions` is the size of its Krylov space.
        Fgmres => ["FGMRES", "GMRES"],
        /// Conjugate gradients, on the normal equations (`operator_type: Normal_left`).
        Cg => ["CG", "CGNE", "CG_normal"],
        /// MINRES on a Γ5-Hermitian form: for the second coarse factor only.
        Minres => ["MINRES"],
    }
}

keyword_enum! {
    pub enum PreconditionerType {
        None => ["None", "No"],
        /// A Chebyshev polynomial of the operator (block `Chebyshev`).
        Chebyshev => ["Chebyshev"],
        /// The even-odd Pauli-Villars multigrid (block `MultiGrid`).
        Multigrid => ["MG", "MultiGrid", "Multi_Grid"],
    }
}

keyword_enum! {
    pub enum PrecisionType {
        Single => ["single", "float", "f32"],
        Double => ["double", "f64"],
    }
}

keyword_enum! {
    pub enum SourceType {
        /// A unit source at one site, twelve columns (colour × spin).
        Local => ["Local", "Point"],
        /// Wilson and clover: smeared sources on the time slice of `source_position`, the
        /// wave function exp(−slope r^power), normalised (`slope`, `power`). Gauge dependent:
        /// see `GaugeFixing`.
        Exponential => ["Exponential"],
    }
}

keyword_enum! {
    pub enum GaugeFixingType {
        Coulomb => ["Coulomb"],
    }
}

keyword_enum! {
    pub enum SmootherType {
        /// A Chebyshev polynomial of Ã (domain-wall).
        Chebyshev => ["Chebyshev"],
        /// The Schwarz alternating procedure on the blocks (Wilson, clover).
        Sap => ["SAP", "Schwarz"],
        /// GMRES(m) on D (Wilson, clover).
        Gmres => ["GMRES"],
    }
}

/// The whole file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    #[serde(rename = "Main")]
    pub main: MainBlock,
    #[serde(rename = "Gauge")]
    pub gauge: GaugeBlock,
    #[serde(rename = "Fopr")]
    pub fopr: FoprBlock,
    #[serde(rename = "Solver")]
    pub solver: SolverBlock,
    /// The multigrid, used when `Solver.preconditioner_type` is `MG`; it may stay in the
    /// file when it is not (switching is one keyword).
    #[serde(rename = "MultiGrid")]
    pub multigrid: Option<MultigridBlock>,
    /// The `MultiGrid` block from a file of its own (holding a `MultiGrid:` section), to
    /// share one multigrid among many parameter files; relative to this file's directory.
    /// A `MultiGrid` block in this file takes precedence.
    #[serde(rename = "MultiGrid_file")]
    pub multigrid_file: Option<PathBuf>,
    /// With a source, the propagator is solved for and the meson correlators and the
    /// residual mass are printed; without one, random right-hand sides are solved.
    #[serde(rename = "Source")]
    pub source: Option<SourceBlock>,
    /// Wilson and clover: further quarks, each with its own `Fopr` and `Source` (`Fopr` and
    /// `Source` at the top are the first quark). The solver is the same for all of them;
    /// correlators are printed for every pair (i, j), i ≤ j.
    #[serde(rename = "Quark_2")]
    pub quark_2: Option<QuarkBlock>,
    #[serde(rename = "Quark_3")]
    pub quark_3: Option<QuarkBlock>,
    #[serde(rename = "Quark_4")]
    pub quark_4: Option<QuarkBlock>,
    /// Wilson and clover: fix the gauge of the configuration before anything else.
    #[serde(rename = "GaugeFixing")]
    pub gauge_fixing: Option<GaugeFixingBlock>,
    /// Wilson and clover: the meson channels to print (`channels: all`, the default, or a
    /// list of names).
    #[serde(rename = "Measure", default)]
    pub measure: MeasureConfig,
    /// Wilson and clover: also write the table to `file`.
    #[serde(rename = "Output", default)]
    pub output: OutputConfig,
    /// Gauge smearing: the fermion operator is made of smeared links. (The smearing of
    /// sources, operator smearing, is a `Source.source_type`.)
    #[serde(rename = "GaugeSmearing")]
    pub gauge_smearing: Option<GaugeSmearingBlock>,
    /// Bridge++'s blocks for gauge smearing, refused because `Smear` does not say what is
    /// smeared; see `GaugeSmearing`.
    #[serde(rename = "Smear", default)]
    bridge_smear: Option<serde::de::IgnoredAny>,
    #[serde(rename = "Director_Smear", default)]
    bridge_director_smear: Option<serde::de::IgnoredAny>,
    #[serde(rename = "Projection", default)]
    bridge_projection: Option<serde::de::IgnoredAny>,
    /// Filled when the file is read and checked: blocks given but not used.
    #[serde(skip)]
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MainBlock {
    pub lattice_size: [usize; ND],
    #[serde(default = "one_grid")]
    pub grid_size: [usize; ND],
    #[serde(default = "cpu")]
    pub backend: Backend,
    /// Sites per SIMD register with `backend: simd`.
    #[serde(default = "four")]
    pub simd_lanes: usize,
    #[serde(default = "general")]
    pub verbose_level: VerboseLevel,
    /// Threads of each process; without it `RAYON_NUM_THREADS`, or every core.
    #[serde(default)]
    pub cpu_thread: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeBlock {
    #[serde(default = "text")]
    pub gauge_config_type_input: GaugeType,
    /// Relative to the parameter file's directory.
    pub config_filename_input: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoprBlock {
    pub fermion_type: FermionType,
    #[serde(default = "dirac")]
    pub gamma_matrix_type: GammaType,
    /// Wilson and clover: κ.
    pub hopping_parameter: Option<f64>,
    /// Clover: c_SW.
    pub clover_coefficient: Option<f64>,
    /// Domain-wall: the quark mass, the height of the wall and Ls (the keys after it are
    /// domain-wall only too).
    pub quark_mass: Option<f64>,
    #[serde(
        default = "unit",
        rename = "quark_mass_PauliVillars",
        alias = "quark_mass_pauli_villars"
    )]
    pub quark_mass_pauli_villars: f64,
    pub domain_wall_height: Option<f64>,
    pub extent_of_5th_dimension: Option<usize>,
    #[serde(default = "mobius_b")]
    pub coefficient_b: f64,
    #[serde(default = "mobius_c")]
    pub coefficient_c: f64,
    /// [x, y, z, t]: 1 periodic, −1 antiperiodic, 0 Dirichlet (Wilson and clover).
    #[serde(default = "antiperiodic_time")]
    pub boundary_condition: [i32; ND],
    /// Neff's α (1: the plain operator). It cancels in `operator_type: PV_right`.
    #[serde(default = "unit")]
    pub neff_alpha: f64,
    /// α of the Pauli-Villars operator. 1 by default: on 12⁴ α in D alone is what helps the
    /// multigrid (the working log (branch `notes`)).
    #[serde(
        default = "unit",
        rename = "neff_alpha_PauliVillars",
        alias = "neff_alpha_pauli_villars"
    )]
    pub neff_alpha_pauli_villars: f64,
}

/// A further quark of a Wilson or clover run.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuarkBlock {
    #[serde(rename = "Fopr")]
    pub fopr: FoprBlock,
    #[serde(rename = "Source")]
    pub source: SourceBlock,
}

/// Gauge fixing, with the keys of Bridge++; the defaults are those of its test
/// `Spectrum_Clover_Hadron2ptFunction`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeFixingBlock {
    pub gauge_fixing_type: GaugeFixingType,
    #[serde(default = "fixing_iterations")]
    pub maximum_number_of_iteration: usize,
    /// Sweeps without overrelaxation at the start and after every reset.
    #[serde(default = "fixing_naive")]
    pub number_of_naive_iteration: usize,
    /// The violation is measured every this many sweeps.
    #[serde(default = "fixing_interval")]
    pub interval_of_measurement: usize,
    /// Time slices not converged after this many sweeps get a random gauge transformation.
    #[serde(default = "fixing_reset")]
    pub iteration_to_reset: usize,
    /// Stop when the violation is below this on every time slice.
    #[serde(default = "fixing_criterion")]
    pub convergence_criterion_squared: f64,
    #[serde(default = "fixing_overrelaxation")]
    pub overrelaxation_parameter: f64,
    /// Seed of the random gauge transformations of a reset.
    #[serde(default = "fixing_seed")]
    pub seed: u64,
}

fn fixing_iterations() -> usize {
    CoulombParams::default().max_iterations
}
fn fixing_naive() -> usize {
    CoulombParams::default().naive_iterations
}
fn fixing_interval() -> usize {
    CoulombParams::default().measure_interval
}
fn fixing_reset() -> usize {
    CoulombParams::default().reset_interval
}
fn fixing_criterion() -> f64 {
    CoulombParams::default().tolerance_sqr
}
fn fixing_overrelaxation() -> f64 {
    CoulombParams::default().overrelaxation
}
fn fixing_seed() -> u64 {
    CoulombParams::default().seed
}

impl GaugeFixingBlock {
    pub fn to_config(&self) -> GaugeFixingConfig {
        GaugeFixingConfig::Coulomb {
            tolerance_squared: self.convergence_criterion_squared,
            max_iterations: self.maximum_number_of_iteration,
            naive_iterations: self.number_of_naive_iteration,
            measure_interval: self.interval_of_measurement,
            reset_interval: self.iteration_to_reset,
            overrelaxation: self.overrelaxation_parameter,
            seed: self.seed,
        }
    }
}

/// A solver: the outer one, or that of a coarse factor of a multigrid level.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverBlock {
    pub solver_type: SolverType,
    /// The outer solver only; by default `Normal_left` for CG and `PV_right` otherwise.
    pub operator_type: Option<OperatorType>,
    /// GCR: the directions kept (memory 2k + 4 vectors); FGMRES: its Krylov space.
    #[serde(default = "two")]
    pub number_of_kept_directions: usize,
    #[serde(default = "iterations")]
    pub maximum_number_of_iteration: usize,
    #[serde(default)]
    pub maximum_number_of_restart: usize,
    pub convergence_criterion_squared: f64,
    #[serde(default = "no_preconditioner")]
    pub preconditioner_type: PreconditionerType,
    #[serde(rename = "Chebyshev")]
    pub chebyshev: Option<ChebyshevBlock>,
}

/// A file holding only the multigrid block.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MultigridFile {
    #[serde(rename = "MultiGrid")]
    multigrid: MultigridBlock,
}

keyword_enum! {
    pub enum GaugeSmearType {
        /// Stout smearing (Bridge++: `Smear_APE` with `Projection_Stout_SU3`).
        Stout => ["Stout", "Stout_SU3"],
        /// APE smearing projected to SU(3) by maximising the trace (Bridge++: `Smear_APE` with
        /// `Projection_Maximum_SU_N`).
        Ape => ["APE"],
    }
}

/// Gauge smearing: `number_of_smearing` steps of `smear_type` with weight `rho_uniform` in
/// every plane (or in the spatial planes only).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeSmearingBlock {
    pub smear_type: GaugeSmearType,
    pub rho_uniform: f64,
    pub number_of_smearing: usize,
    #[serde(default)]
    pub spatial_only: bool,
    /// APE: the projection to SU(3) (Bridge++'s `Projection_Maximum_SU_N` keys).
    #[serde(default = "ape_criterion")]
    pub convergence_criterion: f64,
    #[serde(default = "ape_iterations")]
    pub maximum_number_of_iteration: usize,
}

fn ape_criterion() -> f64 {
    1.0e-15
}
fn ape_iterations() -> usize {
    100
}

impl GaugeSmearingBlock {
    pub fn to_config(&self) -> GaugeSmearingConfig {
        match self.smear_type {
            GaugeSmearType::Stout => GaugeSmearingConfig::Stout {
                rho: self.rho_uniform,
                steps: self.number_of_smearing,
                spatial_only: self.spatial_only,
            },
            GaugeSmearType::Ape => GaugeSmearingConfig::Ape {
                rho: self.rho_uniform,
                steps: self.number_of_smearing,
                spatial_only: self.spatial_only,
                tolerance: self.convergence_criterion,
                max_iterations: self.maximum_number_of_iteration,
            },
        }
    }
}

/// The source of the propagator.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBlock {
    #[serde(default = "local")]
    pub source_type: SourceType,
    /// [x, y, z, t]
    #[serde(default)]
    pub source_position: [usize; ND],
    /// `Exponential`: the wave function exp(−slope r^power).
    pub slope: Option<f64>,
    pub power: Option<f64>,
    /// Columns of the propagator solved at once (1 to 12): more share the coarse
    /// operator reads, fewer need less memory.
    #[serde(default = "one")]
    pub number_of_rhs_per_solve: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChebyshevBlock {
    #[serde(default = "cheb_steps")]
    pub number_of_steps: usize,
    /// The polynomial is small on [hi/ratio, hi], hi = 1.1 |λ_max|.
    #[serde(default = "cheb_ratio")]
    pub interval_ratio: f64,
}

impl Default for ChebyshevBlock {
    fn default() -> Self {
        Self {
            number_of_steps: cheb_steps(),
            interval_ratio: cheb_ratio(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultigridBlock {
    #[serde(default = "single")]
    pub coarse_precision: PrecisionType,
    #[serde(rename = "Setup", default)]
    pub setup: SetupBlock,
    #[serde(rename = "Smoother", default)]
    pub smoother: SmootherBlock,
    #[serde(rename = "MultiGrid_Level1")]
    pub level1: LevelBlock,
    /// Levels below the coarse lattice, for the solves of the quark factor P†DP.
    #[serde(rename = "MultiGrid_Level2")]
    pub level2: Option<SubLevelBlock>,
    #[serde(rename = "MultiGrid_Level3")]
    pub level3: Option<SubLevelBlock>,
    #[serde(rename = "MultiGrid_Level4")]
    pub level4: Option<SubLevelBlock>,
    /// The levels below the coarse lattice also for the solves of the second factor
    /// P†D_PV†P (with MINRES for that factor).
    #[serde(default)]
    pub levels_for_pauli_villars: bool,
}

impl MultigridBlock {
    /// The super-block sizes of the levels below the coarse lattice, in order.
    pub fn super_blocks(&self) -> Vec<usize> {
        [&self.level2, &self.level3, &self.level4]
            .into_iter()
            .map_while(|l| l.as_ref().map(|l| l.super_block_size))
            .collect()
    }

    /// The refinement of the test vectors on the lattice of each level below the coarse
    /// one, in the same order.
    pub fn level_bootstrap(&self) -> Vec<LevelBootstrap> {
        [&self.level2, &self.level3, &self.level4]
            .into_iter()
            .map_while(|l| {
                l.as_ref().map(|l| LevelBootstrap {
                    rounds: l.bootstrap_rounds,
                    iterations: l.bootstrap_iterations,
                })
            })
            .collect()
    }
}

/// A level below the coarse lattice: super-blocks of `super_block_size` sites of the level
/// above in every direction, with block-Jacobi smoothing; the last level is solved exactly
/// (densely, so it has to be small: two test vectors per test vector of level 1 and
/// chirality on each of its sites).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubLevelBlock {
    pub super_block_size: usize,
    /// Before the next level is built from the test vectors on this level's lattice, they
    /// are refined by inverse iteration with this level's operator: this many rounds (1 by
    /// default; it costs seconds and no coarse applications) of ...
    #[serde(default = "one")]
    pub bootstrap_rounds: usize,
    /// ... this many GCR steps preconditioned by the level's V-cycle. With the bootstrap
    /// basis the vectors are smooth only within blocks; a level of few super-blocks below
    /// needs them closer to the low modes.
    #[serde(default = "four")]
    pub bootstrap_iterations: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetupBlock {
    /// Chebyshev-filtered subspace iteration on Ã†Ã: vectors, rounds, vectors at a time.
    #[serde(default = "filter_vectors")]
    pub filter_number_of_vectors: usize,
    #[serde(default = "two")]
    pub filter_number_of_rounds: usize,
    #[serde(default = "four")]
    pub filter_number_in_block: usize,
    /// The multigrid bootstrap after it: rounds, FGMRES steps per vector and round (2 by
    /// default). Wilson and clover: the setup iterations of Bridge++ (4 by default).
    pub bootstrap_number_of_rounds: Option<usize>,
    #[serde(default = "four")]
    pub bootstrap_number_of_steps: usize,
    #[serde(default = "setup_seed")]
    pub seed: u64,
    /// Low modes of Ã from a checkpoint file instead of the filter (see `app::mdwf`).
    pub checkpoint: Option<PathBuf>,
}

impl Default for SetupBlock {
    fn default() -> Self {
        Self {
            filter_number_of_vectors: filter_vectors(),
            filter_number_of_rounds: two(),
            filter_number_in_block: four(),
            bootstrap_number_of_rounds: None,
            bootstrap_number_of_steps: four(),
            seed: setup_seed(),
            checkpoint: None,
        }
    }
}

/// The smoother. Its type defaults to Chebyshev for domain-wall fermions and to SAP for
/// Wilson and clover; `number_of_steps` is the degree of the polynomial, the
/// minimal-residual steps per block of SAP, or the m of GMRES(m).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmootherBlock {
    pub smoother_type: Option<SmootherType>,
    pub number_of_steps: Option<usize>,
    #[serde(default = "cheb_ratio")]
    pub interval_ratio: f64,
    /// SAP: cycles.
    pub number_of_cycles: Option<usize>,
    /// GMRES: an earlier stop at this |r|²/|b|².
    pub convergence_criterion_squared: Option<f64>,
    /// Wilson and clover: smooth before the coarse correction too, instead of starting the
    /// cycle from x = b.
    #[serde(default)]
    pub pre_smoothing: bool,
}

impl Default for SmootherBlock {
    fn default() -> Self {
        Self {
            smoother_type: None,
            number_of_steps: None,
            interval_ratio: cheb_ratio(),
            number_of_cycles: None,
            convergence_criterion_squared: None,
            pre_smoothing: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelBlock {
    pub block_size: [usize; ND],
    /// 32 by default for domain-wall fermions, 16 (Bridge++) for Wilson and clover.
    pub number_of_test_vectors: Option<usize>,
    /// Wilson and clover: the coarse operator multiplied by the inverses of its diagonal
    /// blocks (then solved on the whole coarse lattice).
    #[serde(default)]
    pub precondition_diagonal: bool,
    /// The quark factor P†DP (Wilson, clover: the coarse operator).
    #[serde(rename = "Coarse_Solver_D")]
    pub coarse_d: Option<SolverBlock>,
    /// Domain-wall: the second factor P†D_PV†P, or P†D†P with `operator_type: Normal_left`.
    #[serde(rename = "Coarse_Solver_PV")]
    pub coarse_pv: Option<SolverBlock>,
}

impl LevelBlock {
    pub fn test_vectors(&self, domain_wall: bool) -> usize {
        self.number_of_test_vectors
            .unwrap_or(if domain_wall { test_vectors() } else { 16 })
    }

    /// The coarse solver of the quark factor; the defaults are those of the domain-wall
    /// log, or of the Bridge++ clover multigrid (BiCGStab to 2e-3, at most 200 iterations).
    pub fn solver_d(&self, domain_wall: bool) -> SolverBlock {
        self.coarse_d.clone().unwrap_or_else(|| {
            if domain_wall {
                coarse_d()
            } else {
                SolverBlock {
                    convergence_criterion_squared: 2e-3,
                    maximum_number_of_iteration: 200,
                    ..coarse_d()
                }
            }
        })
    }

    pub fn solver_pv(&self) -> SolverBlock {
        self.coarse_pv.clone().unwrap_or_else(coarse_pv)
    }
}

impl SetupBlock {
    pub fn rounds(&self, domain_wall: bool) -> usize {
        self.bootstrap_number_of_rounds
            .unwrap_or(if domain_wall { 2 } else { 4 })
    }
}

impl FoprBlock {
    /// The domain-wall quark mass (checked to be there when the file is read).
    pub fn mass(&self) -> f64 {
        self.quark_mass.expect("checked for domain-wall fermions")
    }

    /// Ls.
    pub fn ls(&self) -> usize {
        self.extent_of_5th_dimension
            .expect("checked for domain-wall fermions")
    }

    fn boundary(&self) -> [Boundary; ND] {
        self.boundary_condition.map(|b| match b {
            0 => Boundary::Dirichlet,
            b if b < 0 => Boundary::Antiperiodic,
            _ => Boundary::Periodic,
        })
    }
}

fn local() -> SourceType {
    SourceType::Local
}
fn one() -> usize {
    1
}
fn one_grid() -> [usize; ND] {
    [1; ND]
}
fn cpu() -> Backend {
    Backend::Cpu
}
fn general() -> VerboseLevel {
    VerboseLevel::General
}
fn text() -> GaugeType {
    GaugeType::Text
}
fn dirac() -> GammaType {
    GammaType::Dirac
}
fn unit() -> f64 {
    1.0
}
fn mobius_b() -> f64 {
    1.5
}
fn mobius_c() -> f64 {
    0.5
}
fn antiperiodic_time() -> [i32; ND] {
    [1, 1, 1, -1]
}
fn two() -> usize {
    2
}
fn four() -> usize {
    4
}
fn iterations() -> usize {
    2000
}
fn no_preconditioner() -> PreconditionerType {
    PreconditionerType::None
}
fn single() -> PrecisionType {
    PrecisionType::Single
}
fn cheb_steps() -> usize {
    PvEoParams::default().chebyshev_steps
}
fn cheb_ratio() -> f64 {
    PvEoParams::default().chebyshev_ratio
}
fn filter_vectors() -> usize {
    LowModeParams::default().vectors
}
fn setup_seed() -> u64 {
    LowModeParams::default().seed
}
fn test_vectors() -> usize {
    32
}
fn coarse_block(solver_type: SolverType) -> SolverBlock {
    SolverBlock {
        solver_type,
        operator_type: None,
        number_of_kept_directions: 2,
        maximum_number_of_iteration: PvEoParams::default().coarse_max_iterations,
        maximum_number_of_restart: 0,
        convergence_criterion_squared: PvEoParams::default().coarse_tolerance_sqr,
        preconditioner_type: PreconditionerType::None,
        chebyshev: None,
    }
}
fn coarse_d() -> SolverBlock {
    coarse_block(SolverType::BiCgStab)
}
fn coarse_pv() -> SolverBlock {
    coarse_block(SolverType::Minres)
}

fn invalid(what: String) -> CliError {
    CliError::Invalid(what)
}

impl SolverBlock {
    /// The operator the outer solver works on.
    pub fn operator(&self) -> OperatorType {
        self.operator_type.unwrap_or(match self.solver_type {
            SolverType::Cg => OperatorType::NormalLeft,
            _ => OperatorType::PvRight,
        })
    }

    pub fn stop(&self) -> Stop {
        Stop {
            tolerance_sqr: self.convergence_criterion_squared,
            max_iterations: self.maximum_number_of_iteration,
            max_restarts: self.maximum_number_of_restart,
        }
    }

    /// The block of the preconditioner chosen must be there; the blocks of the others may
    /// stay in the file (switching is one keyword) and are named in the notes.
    fn check_preconditioner(
        &self,
        name: &str,
        mg: bool,
        notes: &mut Vec<String>,
    ) -> Result<(), CliError> {
        let cheb = self.chebyshev.is_some();
        let chosen = self.preconditioner_type;
        if chosen == PreconditionerType::Multigrid && !mg {
            return Err(invalid(format!(
                "{name}: preconditioner_type is MG but there is neither a MultiGrid block nor a MultiGrid_file"
            )));
        }
        if mg && chosen != PreconditionerType::Multigrid {
            notes.push(format!(
                "the MultiGrid block is not used ({name}.preconditioner_type: {chosen})"
            ));
        }
        if cheb && chosen != PreconditionerType::Chebyshev {
            notes.push(format!(
                "{name}: the Chebyshev block is not used (preconditioner_type: {chosen})"
            ));
        }
        Ok(())
    }

    /// The Chebyshev preconditioner's settings (the defaults without a block).
    pub fn chebyshev_settings(&self) -> ChebyshevBlock {
        self.chebyshev.unwrap_or_default()
    }

    /// As a coarse solver: no operator, no preconditioner of its own (the inverses of the
    /// diagonal blocks are built in; further levels go into the MultiGrid block).
    fn check_coarse(&self, name: &str, second: bool, normal: bool) -> Result<(), CliError> {
        if self.operator_type.is_some() {
            return Err(invalid(format!(
                "{name}: operator_type belongs to the outer solver only"
            )));
        }
        if self.preconditioner_type != PreconditionerType::None || self.chebyshev.is_some() {
            return Err(invalid(format!(
                "{name}: coarse solvers are preconditioned by the inverses of their diagonal blocks"
            )));
        }
        match self.solver_type {
            SolverType::BiCgStab | SolverType::Gcr => Ok(()),
            SolverType::Minres if second && !normal => Ok(()),
            SolverType::Minres if second => Err(invalid(format!(
                "{name}: with operator_type Normal_left the second factor is P†D†P, far from Γ5-Hermitian; use BiCGStab or GCR"
            ))),
            SolverType::Minres => Err(invalid(format!(
                "{name}: MINRES needs a Hermitian operator, which the quark factor is not"
            ))),
            other => Err(invalid(format!(
                "{name}: solver_type {other} is not available on the coarse lattice; use BiCGStab, GCR or MINRES"
            ))),
        }
    }

    fn krylov(&self) -> CoarseKrylov {
        match self.solver_type {
            SolverType::Gcr => CoarseKrylov::Gcr {
                k: self.number_of_kept_directions,
            },
            SolverType::Minres => CoarseKrylov::Minres,
            _ => CoarseKrylov::BiCgStab,
        }
    }
}

/// Whether a file is a parameter file in the style of Bridge++: one with a `Main` section.
pub fn is_bridge_style(path: &Path) -> Result<bool, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(match path.extension().and_then(|e| e.to_str()) {
        Some("toml") => text
            .parse::<toml::Table>()
            .is_ok_and(|t| t.contains_key("Main")),
        _ => serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text)
            .ok()
            .and_then(|v| v.as_mapping().map(|m| m.contains_key("Main")))
            .unwrap_or(false),
    })
}

impl Params {
    /// The parameters written as YAML. A `MultiGrid_file` is not read: that needs the
    /// directory of the file (`load`).
    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        let params = serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        Self::checked(params)
    }

    /// The parameters written as TOML; see `from_yaml`.
    pub fn from_toml(text: &str) -> Result<Self, CliError> {
        let params = toml::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        Self::checked(params)
    }

    fn checked(mut params: Self) -> Result<Self, CliError> {
        if params.multigrid.is_none() {
            if let Some(file) = &params.multigrid_file {
                return Err(invalid(format!(
                    "MultiGrid_file {file:?} is read relative to the parameter file: load the file itself"
                )));
            }
        }
        params.validate()?;
        Ok(params)
    }

    /// Read a parameter file; the format is chosen by the file extension.
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let params: Self = match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => toml::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?,
            Some("yaml" | "yml") => {
                serde_yaml_ng::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?
            }
            other => {
                return Err(invalid(format!(
                    "parameter file must end in .toml, .yaml or .yml, not {other:?}"
                )))
            }
        };
        let mut params = params;
        if let Some(file) = params.multigrid_file.clone() {
            if params.multigrid.is_some() {
                params.notes.push(format!(
                    "both a MultiGrid block and MultiGrid_file {file:?}; the block is used"
                ));
            } else {
                let file = path.parent().map_or(file.clone(), |d| d.join(&file));
                let text = std::fs::read_to_string(&file).map_err(|e| CliError::Io {
                    path: file.clone(),
                    source: e,
                })?;
                let block: MultigridFile = match file.extension().and_then(|e| e.to_str()) {
                    Some("toml") => {
                        toml::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?
                    }
                    _ => serde_yaml_ng::from_str(&text)
                        .map_err(|e| CliError::Parse(e.to_string()))?,
                };
                params.multigrid = Some(block.multigrid);
            }
        }
        params.validate()?;
        Ok(params)
    }

    /// What was given but is not used, worth a line in the output.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// Checks what the types cannot; what is only odd goes to the notes.
    pub fn validate(&mut self) -> Result<(), CliError> {
        let mut notes = std::mem::take(&mut self.notes);
        let result = self.check(&mut notes);
        self.notes = notes;
        result
    }

    fn check(&self, notes: &mut Vec<String>) -> Result<(), CliError> {
        if self.bridge_smear.is_some()
            || self.bridge_director_smear.is_some()
            || self.bridge_projection.is_some()
        {
            return Err(invalid(
                "Smear / Director_Smear / Projection: \"smear\" alone does not say what is smeared; gauge (link) smearing is the block GaugeSmearing (smear_type: Stout or APE, rho_uniform, number_of_smearing); the smearing of sources (operator smearing) is Source.source_type".into(),
            ));
        }
        self.check_common()?;
        if !self.fopr.fermion_type.is_domain_wall() {
            return self.check_wilson(notes);
        }
        let f = &self.fopr;
        for (key, missing) in [
            ("quark_mass", f.quark_mass.is_none()),
            ("domain_wall_height", f.domain_wall_height.is_none()),
            (
                "extent_of_5th_dimension",
                f.extent_of_5th_dimension.is_none(),
            ),
        ] {
            if missing {
                return Err(invalid(format!(
                    "Fopr.{key} is needed for domain-wall fermions"
                )));
            }
        }
        for (key, given) in [
            ("hopping_parameter", f.hopping_parameter.is_some()),
            ("clover_coefficient", f.clover_coefficient.is_some()),
        ] {
            if given {
                notes.push(format!(
                    "Fopr.{key} is for Wilson and clover fermions; not used"
                ));
            }
        }
        match self.main.backend {
            Backend::Cpu => {}
            Backend::Simd => {
                let lanes = self.main.simd_lanes;
                if ![2, 4, 8].contains(&lanes) {
                    return Err(invalid(format!("Main.simd_lanes: 2, 4 or 8, not {lanes}")));
                }
                let cut = crate::Blocks::new(self.main.lattice_size, lanes);
                if cut.as_ref().and_then(|b| b.even_odd_index()).is_none() {
                    return Err(invalid(format!(
                        "Main.simd_lanes: the lattice {:?} cannot be cut into {lanes} sub-lattices whose lattice of blocks has even extents (needed for even-odd); use fewer lanes",
                        self.main.lattice_size
                    )));
                }
            }
            Backend::Gpu => {
                return Err(invalid(
                    "Main.backend: gpu is not implemented yet; use cpu".into(),
                ))
            }
        }
        let s = &self.solver;
        s.check_preconditioner("Solver", self.multigrid.is_some(), notes)?;
        let operator = s.operator();
        if matches!(operator, OperatorType::EvenOdd | OperatorType::Lexical) {
            return Err(invalid(format!(
                "Solver.operator_type {operator} is for Wilson and clover fermions; use PV_right, PV_left or Normal_left"
            )));
        }
        match (s.solver_type, operator) {
            (SolverType::Cg, OperatorType::PvRight | OperatorType::PvLeft) => {
                return Err(invalid(format!(
                    "Solver: CG needs a Hermitian positive operator; {operator} is not (use operator_type: Normal_left, which is CGNE)"
                )))
            }
            (SolverType::Minres, _) => {
                return Err(invalid(
                    "Solver: MINRES is for the second coarse factor only".into(),
                ))
            }
            (SolverType::Cg, _) if s.preconditioner_type == PreconditionerType::Multigrid => {
                return Err(invalid(
                    "Solver: the multigrid cycle is not a fixed symmetric map, which CG needs; use GCR on Normal_left".into(),
                ))
            }
            _ => {}
        }
        if let Some(mg) = self
            .multigrid
            .as_ref()
            .filter(|_| s.preconditioner_type == PreconditionerType::Multigrid)
        {
            let normal = operator == OperatorType::NormalLeft;
            let level = &mg.level1;
            if let Some(t) = mg
                .smoother
                .smoother_type
                .filter(|t| *t != SmootherType::Chebyshev)
            {
                return Err(invalid(format!(
                    "MultiGrid.Smoother: {t} is for Wilson and clover fermions; the domain-wall smoother is Chebyshev"
                )));
            }
            level.solver_d(true).check_coarse(
                "MultiGrid.MultiGrid_Level1.Coarse_Solver_D",
                false,
                normal,
            )?;
            level.solver_pv().check_coarse(
                "MultiGrid.MultiGrid_Level1.Coarse_Solver_PV",
                true,
                normal,
            )?;
            let n = level.test_vectors(true);
            if !n.is_multiple_of(2) || n / 2 > mg.setup.filter_number_of_vectors {
                return Err(invalid(
                    "MultiGrid.MultiGrid_Level1: number_of_test_vectors must be even and at most twice Setup.filter_number_of_vectors".into(),
                ));
            }
            let levels = mg.super_blocks();
            let given = [&mg.level2, &mg.level3, &mg.level4]
                .iter()
                .filter(|l| l.is_some())
                .count();
            if given != levels.len() {
                return Err(invalid(
                    "MultiGrid: MultiGrid_Level3 needs MultiGrid_Level2, and MultiGrid_Level4 needs MultiGrid_Level3".into(),
                ));
            }
            let mut ext: Vec<usize> = (0..ND)
                .map(|mu| self.main.lattice_size[mu] / level.block_size[mu])
                .collect();
            for (k, &sb) in levels.iter().enumerate() {
                if sb == 0 || ext.iter().any(|e| e % sb != 0) {
                    return Err(invalid(format!(
                        "MultiGrid.MultiGrid_Level{}: super_block_size {sb} does not divide the extent {ext:?} of level {}",
                        k + 2,
                        k + 1
                    )));
                }
                ext.iter_mut().for_each(|e| *e /= sb);
            }
        }
        Ok(())
    }

    /// What every fermion needs: the gamma matrices, the boundary, the source.
    fn check_common(&self) -> Result<(), CliError> {
        if self.fopr.gamma_matrix_type != GammaType::Dirac {
            return Err(invalid(
                "Fopr.gamma_matrix_type: only Dirac is implemented".into(),
            ));
        }
        let domain_wall = self.fopr.fermion_type.is_domain_wall();
        if let Some(b) = self
            .fopr
            .boundary_condition
            .iter()
            .find(|b| b.abs() > 1 || (domain_wall && **b == 0))
        {
            return Err(invalid(format!(
                "Fopr.boundary_condition: 1 (periodic) or -1 (antiperiodic){}, not {b}",
                if domain_wall { "" } else { " or 0 (Dirichlet)" }
            )));
        }
        if domain_wall {
            if self.quark_2.is_some() || self.quark_3.is_some() || self.quark_4.is_some() {
                return Err(invalid(
                    "Quark_2 … Quark_4 are for Wilson and clover runs; a domain-wall run has one quark".into(),
                ));
            }
            if self.gauge_fixing.is_some() {
                return Err(invalid("GaugeFixing is for Wilson and clover runs".into()));
            }
            if self
                .source
                .is_some_and(|s| s.source_type != SourceType::Local)
            {
                return Err(invalid(
                    "Source.source_type: domain-wall runs take a Local source".into(),
                ));
            }
        }
        if self.main.backend == Backend::Gpu {
            return Err(invalid(
                "Main.backend: gpu is not implemented yet; use cpu".into(),
            ));
        }
        if let Some(src) = &self.source {
            self.check_source(src, "Source")?;
        }
        Ok(())
    }

    /// A source: inside the lattice, with what its type needs.
    fn check_source(&self, src: &SourceBlock, name: &str) -> Result<(), CliError> {
        let n = src.number_of_rhs_per_solve;
        if !(1..=12).contains(&n) {
            return Err(invalid(format!(
                "{name}.number_of_rhs_per_solve: 1 to 12, not {n}"
            )));
        }
        if src
            .source_position
            .iter()
            .zip(&self.main.lattice_size)
            .any(|(x, l)| x >= l)
        {
            return Err(invalid(format!(
                "{name}.source_position {:?} is outside the lattice {:?}",
                src.source_position, self.main.lattice_size
            )));
        }
        if src.source_type == SourceType::Exponential
            && (src.slope.is_none() || src.power.is_none())
        {
            return Err(invalid(format!(
                "{name}: an Exponential source needs slope and power (exp(-slope r^power))"
            )));
        }
        Ok(())
    }

    /// What a Wilson or clover quark needs of its `Fopr`.
    fn check_wilson_fopr(f: &FoprBlock, at: &str, notes: &mut Vec<String>) -> Result<(), CliError> {
        let name = f.fermion_type;
        if name.is_domain_wall() {
            return Err(invalid(format!(
                "{at}.fermion_type: the quarks of one run are Wilson or clover fermions, not {name}"
            )));
        }
        if f.gamma_matrix_type != GammaType::Dirac {
            return Err(invalid(format!(
                "{at}.gamma_matrix_type: only Dirac is implemented"
            )));
        }
        if let Some(b) = f.boundary_condition.iter().find(|b| b.abs() > 1) {
            return Err(invalid(format!(
                "{at}.boundary_condition: 1 (periodic), -1 (antiperiodic) or 0 (Dirichlet), not {b}"
            )));
        }
        if f.hopping_parameter.is_none() {
            return Err(invalid(format!(
                "{at}.hopping_parameter is needed for {name} fermions"
            )));
        }
        match (name, f.clover_coefficient.is_some()) {
            (FermionType::Clover, false) => {
                return Err(invalid(format!(
                    "{at}.clover_coefficient is needed for clover fermions"
                )))
            }
            (FermionType::Wilson, true) => notes.push(format!(
                "{at}.clover_coefficient is not used by Wilson fermions (fermion_type: Clover uses it)"
            )),
            _ => {}
        }
        for (key, given) in [
            ("quark_mass", f.quark_mass.is_some()),
            ("domain_wall_height", f.domain_wall_height.is_some()),
            (
                "extent_of_5th_dimension",
                f.extent_of_5th_dimension.is_some(),
            ),
            ("neff_alpha", f.neff_alpha != 1.0),
        ] {
            if given {
                notes.push(format!("{at}.{key} is for domain-wall fermions; not used"));
            }
        }
        Ok(())
    }

    /// Wilson and clover fermions: what the program of `ruttice::app::run` needs, which
    /// is then checked again there (lattice, blocks, SIMD widths, processes).
    fn check_wilson(&self, notes: &mut Vec<String>) -> Result<(), CliError> {
        let name = self.fopr.fermion_type;
        Self::check_wilson_fopr(&self.fopr, "Fopr", notes)?;
        if (self.quark_3.is_some() && self.quark_2.is_none())
            || (self.quark_4.is_some() && self.quark_3.is_none())
        {
            return Err(invalid(
                "Quark_3 needs Quark_2, and Quark_4 needs Quark_3".into(),
            ));
        }
        for (k, q) in self.more_quarks().into_iter().enumerate() {
            let at = format!("Quark_{}", k + 2);
            Self::check_wilson_fopr(&q.fopr, &format!("{at}.Fopr"), notes)?;
            self.check_source(&q.source, &format!("{at}.Source"))?;
        }
        if self.source.is_none() {
            return Err(invalid(format!(
                "Source: {name} runs solve for the propagator of a source; add a Source block"
            )));
        }
        let s = &self.solver;
        s.check_preconditioner("Solver", self.multigrid.is_some(), notes)?;
        match s.solver_type {
            SolverType::Cg | SolverType::BiCgStab | SolverType::Fgmres => {}
            other => {
                return Err(invalid(format!(
                    "Solver: {other} is not available for {name} fermions; use CG (on the normal equations), BiCGStab or FGMRES"
                )))
            }
        }
        if let Some(o) = s
            .operator_type
            .filter(|o| !matches!(o, OperatorType::EvenOdd | OperatorType::Lexical))
        {
            return Err(invalid(format!(
                "Solver.operator_type {o} is for domain-wall fermions; use Even_odd or Lexical"
            )));
        }
        match s.preconditioner_type {
            PreconditionerType::Chebyshev => {
                return Err(invalid(format!(
                    "Solver: the Chebyshev preconditioner is for domain-wall fermions; use MG or None for {name}"
                )))
            }
            PreconditionerType::Multigrid if s.solver_type == SolverType::Cg => {
                return Err(invalid(
                    "Solver: the multigrid needs a flexible solver: BiCGStab or FGMRES".into(),
                ))
            }
            _ => {}
        }
        if let Some(mg) = self
            .multigrid
            .as_ref()
            .filter(|_| s.preconditioner_type == PreconditionerType::Multigrid)
        {
            if mg.smoother.smoother_type == Some(SmootherType::Chebyshev) {
                return Err(invalid(format!(
                    "MultiGrid.Smoother: Chebyshev is the domain-wall smoother; use SAP or GMRES for {name}"
                )));
            }
            match mg.level1.solver_d(false).solver_type {
                SolverType::BiCgStab | SolverType::Fgmres => {}
                other => {
                    return Err(invalid(format!(
                        "MultiGrid.MultiGrid_Level1.Coarse_Solver_D: {other} is not available for {name}; use BiCGStab or GMRES"
                    )))
                }
            }
            if mg.level1.coarse_pv.is_some() {
                notes.push(format!(
                    "MultiGrid.MultiGrid_Level1.Coarse_Solver_PV is for domain-wall fermions; not used for {name}"
                ));
            }
            if mg.level4.is_some() {
                return Err(invalid(format!(
                    "MultiGrid.MultiGrid_Level4: {name} fermions take levels down to MultiGrid_Level3"
                )));
            }
            if mg.level3.is_some() && mg.level2.is_none() {
                return Err(invalid(
                    "MultiGrid: MultiGrid_Level3 needs MultiGrid_Level2".into(),
                ));
            }
            let mut ext: Vec<usize> = (0..ND)
                .map(|mu| self.main.lattice_size[mu] / mg.level1.block_size[mu].max(1))
                .collect();
            for (k, sb) in mg.super_blocks().into_iter().enumerate() {
                if sb == 0 || ext.iter().any(|e| e % sb != 0) {
                    return Err(invalid(format!(
                        "MultiGrid.MultiGrid_Level{}: super_block_size {sb} does not divide the extent {ext:?} of level {}",
                        k + 2,
                        k + 1
                    )));
                }
                ext.iter_mut().for_each(|e| *e /= sb);
            }
        }
        Ok(())
    }

    /// Wilson and clover fermions: the parameter file of `ruttice::app::run` this one
    /// describes (the propagator of the source, all meson channels).
    pub fn to_config(&self) -> Config {
        let quark = |f: &FoprBlock, src: &SourceBlock| {
            let boundary = f.boundary().map(|b| match b {
                Boundary::Antiperiodic => BoundaryConfig::Antiperiodic,
                Boundary::Dirichlet => BoundaryConfig::Dirichlet,
                _ => BoundaryConfig::Periodic,
            });
            let kappa = f.hopping_parameter.expect("checked");
            let position = src.source_position;
            QuarkConfig {
                fermion: match f.fermion_type {
                    FermionType::Wilson => FermionConfig::Wilson { kappa, boundary },
                    FermionType::Clover => FermionConfig::Clover {
                        kappa,
                        csw: f.clover_coefficient.expect("checked"),
                        boundary,
                    },
                    FermionType::DomainWall => {
                        unreachable!("domain-wall runs do not go through Config")
                    }
                },
                source: match src.source_type {
                    SourceType::Local => SourceConfig::Point { position },
                    SourceType::Exponential => SourceConfig::Exponential {
                        position,
                        slope: src.slope.expect("checked"),
                        power: src.power.expect("checked"),
                    },
                },
            }
        };
        let s = &self.solver;
        let multigrid = self
            .multigrid
            .as_ref()
            .filter(|_| s.preconditioner_type == PreconditionerType::Multigrid)
            .map(|mg| {
                let (l, sm) = (&mg.level1, &mg.smoother);
                let coarse = l.solver_d(false);
                let gmres = sm.smoother_type == Some(SmootherType::Gmres);
                MultigridConfig {
                    block: l.block_size,
                    test_vectors: l.test_vectors(false),
                    setup_iterations: mg.setup.rounds(false),
                    coarse_solver: match coarse.solver_type {
                        SolverType::Fgmres => CoarseSolverKind::Gmres,
                        _ => CoarseSolverKind::Bicgstab,
                    },
                    coarse_tolerance_squared: coarse.convergence_criterion_squared,
                    coarse_max_iterations: coarse.maximum_number_of_iteration,
                    precondition_coarse_diagonal: l.precondition_diagonal,
                    smoother: if gmres {
                        SmootherKind::Gmres
                    } else {
                        SmootherKind::Sap
                    },
                    pre_smoothing: sm.pre_smoothing,
                    smoother_cycles: sm.number_of_cycles.unwrap_or(4),
                    smoother_block_steps: sm.number_of_steps.unwrap_or(6),
                    smoother_iterations: sm.number_of_steps.unwrap_or(4),
                    smoother_tolerance_squared: sm.convergence_criterion_squared.unwrap_or(1e-3),
                    precision: match mg.coarse_precision {
                        PrecisionType::Single => Precision::Single,
                        PrecisionType::Double => Precision::Double,
                    },
                    seed: mg.setup.seed,
                    level2_super_block: mg.level2.map(|l| l.super_block_size),
                    level3_super_block: mg.level3.map(|l| l.super_block_size),
                }
            });
        let first = quark(&self.fopr, self.source.as_ref().expect("checked"));
        let more = self
            .more_quarks()
            .into_iter()
            .map(|q| quark(&q.fopr, &q.source));
        Config {
            lattice: self.main.lattice_size,
            process_grid: (self.main.grid_size != [1; ND]).then_some(self.main.grid_size),
            gauge: GaugeConfig {
                file: self.gauge.config_filename_input.clone(),
                format: match self.gauge.gauge_config_type_input {
                    GaugeType::Text => GaugeFormat::BridgeText,
                    GaugeType::Binary => GaugeFormat::Binary,
                    GaugeType::Nersc => GaugeFormat::Nersc,
                    GaugeType::Ildg => GaugeFormat::Ildg,
                    GaugeType::Scidac => GaugeFormat::Scidac,
                    GaugeType::Hdf5 => GaugeFormat::Hdf5,
                },
                fixing: self.gauge_fixing.map(|g| g.to_config()),
            },
            gauge_smearing: self.gauge_smearing.map(|g| g.to_config()),
            solver: SolverConfig {
                kind: match s.solver_type {
                    SolverType::Cg => SolverKind::CgNormal,
                    SolverType::BiCgStab => SolverKind::Bicgstab,
                    _ => SolverKind::Fgmres,
                },
                even_odd: s.operator_type != Some(OperatorType::Lexical),
                tolerance_squared: s.convergence_criterion_squared,
                // FGMRES keeps its Krylov space of number_of_kept_directions vectors and
                // restarts until maximum_number_of_iteration iterations are done.
                max_iterations: if s.solver_type == SolverType::Fgmres {
                    s.number_of_kept_directions
                } else {
                    s.maximum_number_of_iteration
                },
                max_restarts: if s.solver_type == SolverType::Fgmres {
                    (s.maximum_number_of_iteration / s.number_of_kept_directions.max(1)).max(1)
                } else {
                    s.maximum_number_of_restart.max(1)
                },
                simd: match self.main.backend {
                    Backend::Simd => self.main.simd_lanes,
                    _ => 1,
                },
                multigrid,
            },
            quark: std::iter::once(first).chain(more).collect(),
            measure: self.measure.clone(),
            output: self.output.clone(),
        }
    }

    /// The quarks after the first (`Quark_2` …), in order.
    pub fn more_quarks(&self) -> Vec<&QuarkBlock> {
        [&self.quark_2, &self.quark_3, &self.quark_4]
            .into_iter()
            .map_while(|q| q.as_ref())
            .collect()
    }

    /// The domain-wall operator of `Fopr` (α = 1; set α with `Mdwf::set_alpha`).
    pub fn mdwf_params(&self, mass: f64) -> MdwfParams {
        let f = &self.fopr;
        let boundary = f.boundary();
        MdwfParams {
            ls: f.ls(),
            m5: f
                .domain_wall_height
                .expect("checked for domain-wall fermions"),
            mass,
            b: f.coefficient_b,
            c: f.coefficient_c,
            boundary,
        }
    }

    /// The operator the outer solver works on: by default `Normal_left` for CG and
    /// `PV_right` otherwise for domain-wall fermions, `Even_odd` for Wilson and clover.
    pub fn operator(&self) -> OperatorType {
        if self.fopr.fermion_type.is_domain_wall() {
            self.solver.operator()
        } else {
            self.solver.operator_type.unwrap_or(OperatorType::EvenOdd)
        }
    }

    /// The even-odd form of the domain-wall operator, and whether D_PV is replaced by D
    /// (CGNE).
    pub fn form(&self) -> (EoForm, bool) {
        match self.solver.operator() {
            OperatorType::PvRight => (EoForm::Right, false),
            OperatorType::PvLeft => (EoForm::Left, false),
            OperatorType::NormalLeft => (EoForm::Left, true),
            other => unreachable!("{other} was refused for domain-wall fermions"),
        }
    }

    /// The settings of the multigrid cycle, setup and bootstrap, when it is the
    /// preconditioner.
    pub fn multigrid(&self) -> Option<(PvEoParams, LowModeParams, BootstrapParams)> {
        if self.solver.preconditioner_type != PreconditionerType::Multigrid {
            return None;
        }
        let mg = self.multigrid.as_ref()?;
        let level = &mg.level1;
        let (coarse_d, coarse_pv) = (level.solver_d(true), level.solver_pv());
        let cycle = PvEoParams {
            chebyshev_steps: mg.smoother.number_of_steps.unwrap_or(cheb_steps()),
            chebyshev_ratio: mg.smoother.interval_ratio,
            coarse_tolerance_sqr: coarse_d.convergence_criterion_squared,
            coarse_max_iterations: coarse_d.maximum_number_of_iteration,
            pv_tolerance_sqr: coarse_pv.convergence_criterion_squared,
            pv_max_iterations: coarse_pv.maximum_number_of_iteration,
            d_solver: coarse_d.krylov(),
            pv_solver: coarse_pv.krylov(),
            ..PvEoParams::default()
        };
        let low = LowModeParams {
            vectors: mg.setup.filter_number_of_vectors,
            rounds: mg.setup.filter_number_of_rounds,
            chunk: mg.setup.filter_number_in_block,
            seed: mg.setup.seed,
        };
        let boot = BootstrapParams {
            rounds: mg.setup.rounds(true),
            steps: mg.setup.bootstrap_number_of_steps,
            cycle,
        };
        Some((cycle, low, boot))
    }
}

/// What will run, in a few lines: printed at the start so that the effect of the defaults
/// and of `operator_type` is visible.
impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (m, fo, s) = (&self.main, &self.fopr, &self.solver);
        writeln!(
            f,
            "lattice {:?} on {}{}",
            m.lattice_size,
            m.backend,
            if m.backend == Backend::Simd {
                format!(" ({} lanes)", m.simd_lanes)
            } else {
                String::new()
            }
        )?;
        let shown = |x: Option<f64>| x.map_or("?".to_string(), |x| x.to_string());
        match fo.fermion_type {
            FermionType::DomainWall => writeln!(
                f,
                "{}: Ls = {}, M5 = {}, m = {}, m_PV = {}, b = {}, c = {}, α = {} (D_PV: {})",
                fo.fermion_type,
                fo.extent_of_5th_dimension
                    .map_or("?".to_string(), |l| l.to_string()),
                shown(fo.domain_wall_height),
                shown(fo.quark_mass),
                fo.quark_mass_pauli_villars,
                fo.coefficient_b,
                fo.coefficient_c,
                fo.neff_alpha,
                fo.neff_alpha_pauli_villars
            )?,
            FermionType::Wilson => writeln!(
                f,
                "{}: κ = {}",
                fo.fermion_type,
                shown(fo.hopping_parameter)
            )?,
            FermionType::Clover => writeln!(
                f,
                "{}: κ = {}, c_SW = {}",
                fo.fermion_type,
                shown(fo.hopping_parameter),
                shown(fo.clover_coefficient)
            )?,
        }
        for (k, q) in self.more_quarks().into_iter().enumerate() {
            writeln!(
                f,
                "quark {}: {}, κ = {}{}; {} source at {:?}",
                k + 2,
                q.fopr.fermion_type,
                shown(q.fopr.hopping_parameter),
                q.fopr
                    .clover_coefficient
                    .map_or(String::new(), |c| format!(", c_SW = {c}")),
                q.source.source_type,
                q.source.source_position
            )?;
        }
        if let Some(g) = &self.gauge_fixing {
            writeln!(
                f,
                "{} gauge fixing to {:e}",
                g.gauge_fixing_type, g.convergence_criterion_squared
            )?;
        }
        let operator = self.operator();
        let cg = s.solver_type == SolverType::Cg;
        write!(
            f,
            "{} on {}{}, to |r|²/|b|² < {:e}",
            s.solver_type,
            operator,
            match operator {
                OperatorType::PvRight => " (A′, α cancels)",
                OperatorType::PvLeft => " (left-preconditioned A′)",
                OperatorType::NormalLeft => " (CGNE)",
                OperatorType::EvenOdd if cg => " (the Schur complement on the even sites, CGNE)",
                OperatorType::EvenOdd => " (the Schur complement on the even sites)",
                OperatorType::Lexical if cg => " (D on the whole lattice, CGNE)",
                OperatorType::Lexical => " (D on the whole lattice)",
            },
            s.convergence_criterion_squared
        )?;
        match (s.preconditioner_type, &self.multigrid) {
            (PreconditionerType::Multigrid, Some(mg)) if !fo.fermion_type.is_domain_wall() => {
                let (l, sm) = (&mg.level1, &mg.smoother);
                let coarse = l.solver_d(false);
                let smoother = match sm.smoother_type {
                    Some(SmootherType::Gmres) => {
                        format!("GMRES({})", sm.number_of_steps.unwrap_or(4))
                    }
                    _ => format!(
                        "SAP {} cycles × {} steps",
                        sm.number_of_cycles.unwrap_or(4),
                        sm.number_of_steps.unwrap_or(6)
                    ),
                };
                write!(
                    f,
                    "\npreconditioned by the multigrid of the whole lattice: {:?} blocks, {} test vectors, {} setup iterations; coarse {} to {:e} in {}; smoother {smoother}",
                    l.block_size,
                    l.test_vectors(false),
                    mg.setup.rounds(false),
                    coarse.solver_type,
                    coarse.convergence_criterion_squared,
                    mg.coarse_precision,
                )
            }
            (PreconditionerType::Multigrid, Some(mg)) => {
                let l = &mg.level1;
                write!(
                    f,
                    "\npreconditioned by the multigrid: {:?} blocks, {} test vectors; coarse {} (P†DP) to {:e} and {} ({}) to {:e} in {}; smoother Chebyshev {}/{}",
                    l.block_size,
                    l.test_vectors(true),
                    l.solver_d(true).solver_type,
                    l.solver_d(true).convergence_criterion_squared,
                    l.solver_pv().solver_type,
                    if operator == OperatorType::NormalLeft { "P†D†P" } else { "P†D_PV†P" },
                    l.solver_pv().convergence_criterion_squared,
                    mg.coarse_precision,
                    mg.smoother.number_of_steps.unwrap_or(cheb_steps()),
                    mg.smoother.interval_ratio
                )?;
                let levels = mg.super_blocks();
                if !levels.is_empty() {
                    write!(
                        f,
                        "\nlevels below the coarse lattice for P†DP: super-blocks {levels:?} (V-cycle, the last solved exactly)"
                    )?;
                    if mg.levels_for_pauli_villars {
                        write!(f, "; for P†D_PV†P too (|H|⁻¹, added to the blocks)")?;
                    }
                }
                Ok(())
            }
            (PreconditionerType::Chebyshev, _) => {
                let c = s.chebyshev_settings();
                write!(
                    f,
                    "\npreconditioned by Chebyshev {}/{}",
                    c.number_of_steps, c.interval_ratio
                )
            }
            _ => write!(f, "\nno preconditioner"),
        }?;
        if let Some(src) = &self.source {
            write!(
                f,
                "\n{} source at {:?}, {} columns per solve: meson correlators{}",
                src.source_type,
                src.source_position,
                src.number_of_rhs_per_solve,
                if fo.fermion_type.is_domain_wall() {
                    " and the residual mass"
                } else {
                    ""
                }
            )?;
        }
        for note in &self.notes {
            write!(f, "\nnote: {note}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "
Main:
  lattice_size : [12, 12, 12, 12]
Gauge:
  config_filename_input : conf.txt
Fopr:
  fermion_type            : domainwall_5din
  quark_mass              : 0.005
  domain_wall_height      : 1.0
  extent_of_5th_dimension : 8
Solver:
  solver_type                   : cgne
  convergence_criterion_squared : 1.0e-18
";

    const MG: &str = "MultiGrid:\n  MultiGrid_Level1:\n    block_size : [4, 4, 4, 4]\n";

    /// MINIMAL with its solver lines replaced, and the multigrid section appended.
    fn with(solver: &str, mg: bool) -> String {
        let text = MINIMAL.replace("  solver_type                   : cgne\n", solver);
        if mg {
            text + MG
        } else {
            text
        }
    }

    fn parse(text: &str) -> Result<Params, CliError> {
        let mut p: Params =
            serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        p.validate()?;
        Ok(p)
    }

    #[test]
    fn cgne_without_multigrid_needs_no_more() {
        let p = parse(MINIMAL).unwrap();
        assert_eq!(p.solver.operator(), OperatorType::NormalLeft);
        assert_eq!(p.form(), (EoForm::Left, true));
        assert!(p.multigrid().is_none());
        assert!(p.to_string().contains("CG on Normal_left (CGNE)"), "{p}");
    }

    #[test]
    fn multigrid_block_with_defaults_and_keywords_in_any_case() {
        let text = with(
            "  solver_type : gcr\n  operator_type : pv_LEFT\n  preconditioner_type : multi_grid\n",
            true,
        );
        let p = parse(&text).unwrap();
        let (cycle, low, boot) = p.multigrid().unwrap();
        assert_eq!(cycle.d_solver, CoarseKrylov::BiCgStab);
        assert_eq!(cycle.pv_solver, CoarseKrylov::Minres);
        assert_eq!(low.vectors, 16);
        assert_eq!(boot.rounds, 2);
        assert_eq!(p.form(), (EoForm::Left, false));
        assert!(p.multigrid.as_ref().unwrap().super_blocks().is_empty());
    }

    #[test]
    fn levels_below_the_coarse_lattice() {
        let solver = "  solver_type : gcr\n  preconditioner_type : MG\n";
        // 12⁴ in blocks of 4⁴ is 3⁴ coarse sites; 16⁴ is 4⁴.
        let sixteen = |levels: &str| {
            with(solver, true).replace("[12, 12, 12, 12]", "[16, 16, 16, 16]") + levels
        };
        let p = parse(&sixteen(
            "  MultiGrid_Level2:\n    super_block_size : 2\n  MultiGrid_Level3:\n    super_block_size : 2\n",
        ))
        .unwrap();
        assert_eq!(p.multigrid.as_ref().unwrap().super_blocks(), vec![2, 2]);
        assert!(p.to_string().contains("super-blocks [2, 2]"), "{p}");
        let e = parse(&(with(solver, true) + "  MultiGrid_Level2:\n    super_block_size : 2\n"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("does not divide the extent [3, 3, 3, 3]"), "{e}");
        let e = parse(&sixteen("  MultiGrid_Level3:\n    super_block_size : 2\n"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("Level3 needs MultiGrid_Level2"), "{e}");
    }

    #[test]
    fn confusing_combinations_are_explained() {
        let mg_block_unused = with("  solver_type : gcr\n", true);
        // Not an error: switching the multigrid off is one keyword; the block is noted.
        let p = parse(&mg_block_unused).unwrap();
        assert!(p.multigrid().is_none());
        assert!(p.to_string().contains("no preconditioner"), "{p}");
        assert!(
            p.notes()[0].contains("MultiGrid block is not used"),
            "{:?}",
            p.notes()
        );
        let mg_missing = with("  solver_type : gcr\n  preconditioner_type : MG\n", false);
        let e = parse(&mg_missing).unwrap_err().to_string();
        assert!(e.contains("neither a MultiGrid block"), "{e}");
        let cg_on_a = with("  solver_type : CG\n  operator_type : PV_right\n", false);
        let e = parse(&cg_on_a).unwrap_err().to_string();
        assert!(e.contains("Hermitian positive"), "{e}");
        let normal_minres = with(
            "  solver_type : GCR\n  operator_type : normal\n  preconditioner_type : MG\n",
            true,
        );
        let e = parse(&normal_minres).unwrap_err().to_string();
        assert!(e.contains("P†D†P"), "{e}");
        let gpu = MINIMAL.replace(
            "  lattice_size : [12, 12, 12, 12]\n",
            "  lattice_size : [12, 12, 12, 12]\n  backend : GPU\n",
        );
        let e = parse(&gpu).unwrap_err().to_string();
        assert!(e.contains("not implemented"), "{e}");
        let typo = MINIMAL.replace("quark_mass  ", "quark_mas   ");
        assert!(parse(&typo).is_err());
    }

    #[test]
    fn toml_reads_the_same() {
        let text = "
[Main]
lattice_size = [12, 12, 12, 12]
[Gauge]
config_filename_input = \"conf.txt\"
[Fopr]
fermion_type = \"MDWF\"
quark_mass = 0.005
domain_wall_height = 1.0
extent_of_5th_dimension = 8
neff_alpha = 0.6
[Solver]
solver_type = \"BiCGStab\"
convergence_criterion_squared = 1e-18
";
        let mut p: Params = toml::from_str(text).unwrap();
        p.validate().unwrap();
        assert_eq!(p.solver.operator(), OperatorType::PvRight);
        assert_eq!(p.fopr.neff_alpha, 0.6);
    }

    /// One multigrid file shared by two parameter files, one of them YAML, the other TOML.
    #[test]
    fn multigrid_from_a_file_of_its_own() {
        let dir = std::env::temp_dir().join(format!("ruttice_params_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("mg.yaml"),
            "MultiGrid:\n  coarse_precision : DOUBLE\n  MultiGrid_Level1:\n    block_size : [2, 2, 2, 2]\n    Coarse_Solver_PV:\n      solver_type : gcr\n      convergence_criterion_squared : 1.0e-2\n",
        )
        .unwrap();
        let yaml = with("  solver_type : GCR\n  preconditioner_type : MG\n", false)
            + "MultiGrid_file : mg.yaml\n";
        std::fs::write(dir.join("run.yaml"), yaml).unwrap();
        let toml_text = "MultiGrid_file = \"mg.yaml\"\n[Main]\nlattice_size = [12, 12, 12, 12]\n[Gauge]\nconfig_filename_input = \"conf.txt\"\n[Fopr]\nfermion_type = \"mobius\"\nquark_mass = 0.005\ndomain_wall_height = 1.0\nextent_of_5th_dimension = 8\n[Solver]\nsolver_type = \"gcr\"\npreconditioner_type = \"mg\"\nconvergence_criterion_squared = 1e-18\n";
        std::fs::write(dir.join("run.toml"), toml_text).unwrap();
        for name in ["run.yaml", "run.toml"] {
            let p = Params::load(&dir.join(name)).unwrap();
            let (cycle, _, _) = p.multigrid().unwrap();
            assert_eq!(cycle.pv_solver, CoarseKrylov::Gcr { k: 2 }, "{name}");
            let mg = p.multigrid.as_ref().unwrap();
            assert_eq!(mg.coarse_precision, PrecisionType::Double, "{name}");
            assert_eq!(mg.level1.block_size, [2; ND], "{name}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wilson_and_clover_files() {
        let p = parse(include_str!("../../examples/clover_mg.yaml")).unwrap();
        assert_eq!(p.operator(), OperatorType::EvenOdd);
        let c = p.to_config();
        assert_eq!(
            c.quark[0].fermion,
            FermionConfig::Clover {
                kappa: 0.12,
                csw: 1.0,
                boundary: [
                    BoundaryConfig::Periodic,
                    BoundaryConfig::Periodic,
                    BoundaryConfig::Periodic,
                    BoundaryConfig::Antiperiodic
                ]
            }
        );
        assert!(c.solver.even_odd);
        assert_eq!(c.solver.kind, SolverKind::Fgmres);
        assert_eq!((c.solver.max_iterations, c.solver.max_restarts), (30, 100));
        let mg = c.solver.multigrid.unwrap();
        assert_eq!(
            (mg.block, mg.test_vectors, mg.setup_iterations),
            ([2; ND], 8, 4)
        );
        assert_eq!(
            (mg.smoother, mg.smoother_cycles, mg.smoother_block_steps),
            (SmootherKind::Sap, 4, 6)
        );
        assert_eq!(
            (mg.coarse_tolerance_squared, mg.coarse_max_iterations),
            (2e-3, 200)
        );

        let wilson = "Main:\n  lattice_size : [4, 4, 4, 8]\nGauge:\n  config_filename_input : c.txt\nFopr:\n  fermion_type : wilson\n  hopping_parameter : 0.13\nSolver:\n  solver_type : cg\n  convergence_criterion_squared : 1.0e-20\nSource:\n  source_position : [0, 0, 0, 0]\n";
        // Gauge smearing is the block GaugeSmearing; Bridge++'s Smear blocks are refused.
        let p = parse(&format!(
            "{wilson}GaugeSmearing:\n  smear_type : stout\n  rho_uniform : 0.1\n  number_of_smearing : 3\n"
        ))
        .unwrap();
        assert_eq!(
            p.to_config().gauge_smearing,
            Some(GaugeSmearingConfig::Stout {
                rho: 0.1,
                steps: 3,
                spatial_only: false
            })
        );
        for block in [
            "Smear:\n  smear_type : APE\n",
            "Director_Smear:\n  number_of_smearing : 1\n",
            "Projection:\n  projection_type : Stout_SU3\n",
        ] {
            let e = parse(&format!("{wilson}{block}")).unwrap_err().to_string();
            assert!(
                e.contains("GaugeSmearing") && e.contains("Source.source_type"),
                "{e}"
            );
        }
        let p = parse(wilson).unwrap();
        let c = p.to_config();
        assert_eq!(c.solver.kind, SolverKind::CgNormal);
        assert!(p.to_string().contains("Wilson: κ = 0.13"), "{p}");
        assert!(p.to_string().contains("CGNE"), "{p}");
        // The Bridge++ defaults of the multigrid when the block names only the blocks.
        let p = parse(
            &(wilson.replace(
                "solver_type : cg",
                "solver_type : bicgstab\n  preconditioner_type : MG",
            ) + "MultiGrid:\n  MultiGrid_Level1:\n    block_size : [2, 2, 2, 2]\n"),
        )
        .unwrap();
        let mg = p.to_config().solver.multigrid.unwrap();
        assert_eq!(
            (
                mg.test_vectors,
                mg.setup_iterations,
                mg.coarse_max_iterations
            ),
            (16, 4, 200)
        );

        let err = |text: String| parse(&text).unwrap_err().to_string();
        let e = err(wilson.replace("  hopping_parameter : 0.13\n", ""));
        assert!(e.contains("hopping_parameter is needed"), "{e}");
        let e = err(wilson.replace("fermion_type : wilson", "fermion_type : clover"));
        assert!(e.contains("clover_coefficient is needed"), "{e}");
        let e = err(wilson.replace("solver_type : cg", "solver_type : gcr"));
        assert!(e.contains("not available for Wilson"), "{e}");
        let e = err(wilson.replace(
            "solver_type : cg",
            "solver_type : cg\n  operator_type : PV_right",
        ));
        assert!(e.contains("for domain-wall fermions"), "{e}");
        let e = err(wilson.replace("Source:\n  source_position : [0, 0, 0, 0]\n", ""));
        assert!(e.contains("add a Source block"), "{e}");
        let p = parse(&wilson.replace(
            "hopping_parameter : 0.13",
            "hopping_parameter : 0.13\n  quark_mass : 0.1",
        ))
        .unwrap();
        assert!(
            p.notes()[0].contains("quark_mass is for domain-wall"),
            "{:?}",
            p.notes()
        );
        // And the other way round.
        let e = err(MINIMAL.replace("  quark_mass              : 0.005\n", ""));
        assert!(e.contains("quark_mass is needed for domain-wall"), "{e}");
    }
}
