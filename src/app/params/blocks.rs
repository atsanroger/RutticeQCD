//! The sections of a parameter file and their defaults.

use super::*;

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
    pub(super) bridge_smear: Option<serde::de::IgnoredAny>,
    #[serde(rename = "Director_Smear", default)]
    pub(super) bridge_director_smear: Option<serde::de::IgnoredAny>,
    #[serde(rename = "Projection", default)]
    pub(super) bridge_projection: Option<serde::de::IgnoredAny>,
    /// Filled when the file is read and checked: blocks given but not used.
    #[serde(skip)]
    pub(super) notes: Vec<String>,
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
pub(super) struct MultigridFile {
    #[serde(rename = "MultiGrid")]
    pub(super) multigrid: MultigridBlock,
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
    /// The precision of the coarse solves (`coarse_precision()`), as the file gives it.
    #[serde(rename = "coarse_precision")]
    pub coarse_precision_given: Option<PrecisionType>,
    /// The precision of the multigrid above the coarse lattice, its setup and its cycle (the
    /// outer solver is in double precision either way), as given (`fine_precision()`).
    #[serde(rename = "fine_precision")]
    pub fine_precision_given: Option<PrecisionType>,
    /// On a device: how the prolongation is kept, the largest thing of the multigrid: in half
    /// precision unless this says single. The coarse operators are built from it as it is kept.
    #[serde(rename = "basis_precision")]
    pub basis_precision_given: Option<BasisPrecision>,
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
    /// Single unless the file says double.
    pub fn coarse_precision(&self) -> PrecisionType {
        self.coarse_precision_given.unwrap_or(PrecisionType::Single)
    }

    pub fn fine_precision(&self) -> PrecisionType {
        self.fine_precision_given.unwrap_or(PrecisionType::Single)
    }

    /// Half unless the file says single.
    pub fn basis_precision(&self) -> BasisPrecision {
        self.basis_precision_given.unwrap_or(BasisPrecision::Half)
    }

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

    /// The boundary conditions as the internal parameter structures hold them.
    pub(in crate::app) fn boundary_config(&self) -> [BoundaryConfig; ND] {
        self.boundary().map(|b| match b {
            Boundary::Antiperiodic => BoundaryConfig::Antiperiodic,
            Boundary::Dirichlet => BoundaryConfig::Dirichlet,
            _ => BoundaryConfig::Periodic,
        })
    }

    pub(super) fn boundary(&self) -> [Boundary; ND] {
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
pub(super) fn cheb_steps() -> usize {
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
    pub(super) fn check_preconditioner(
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
    pub(super) fn check_coarse(
        &self,
        name: &str,
        second: bool,
        normal: bool,
    ) -> Result<(), CliError> {
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

    pub(super) fn krylov(&self) -> CoarseKrylov {
        match self.solver_type {
            SolverType::Gcr => CoarseKrylov::Gcr {
                k: self.number_of_kept_directions,
            },
            SolverType::Minres => CoarseKrylov::Minres,
            _ => CoarseKrylov::BiCgStab,
        }
    }
}
