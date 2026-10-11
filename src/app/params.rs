//! Parameter files in the style of Bridge++: sections `Main`, `Gauge`, `Fopr` and
//! `Solver`, keys spelt out (`maximum_number_of_iteration`, `convergence_criterion_squared`),
//! read from YAML (`.yaml`, `.yml`) or TOML (`.toml`). The values of choices are keywords
//! read without regard to case or `_` (see [`super::keyword`]); a misspelt key is an error.
//!
//! The multigrid is a section of its own, `MultiGrid`, next to `Solver`, which uses it with
//! `preconditioner_type: MG` and may keep it when it does not (switching is one keyword).
//! It can also come from a file of its own (`MultiGrid_file`), to share one multigrid among
//! many parameter files. Its levels are `MultiGrid_Level1`, `MultiGrid_Level2`, ... (only the
//! first is implemented so far); each names a solver for each coarse factor, with the keys
//! of `Solver`.
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

use super::CliError;
use crate::fermion::{EoForm, MdwfParams};
use crate::keyword_enum;
use crate::solver::preconditioner::multigrid::{
    BootstrapParams, CoarseKrylov, LowModeParams, PvEoParams,
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
        Ildg => ["ILDG"],
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
    }
}

keyword_enum! {
    pub enum SmootherType {
        Chebyshev => ["Chebyshev"],
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
    pub quark_mass: f64,
    #[serde(
        default = "unit",
        rename = "quark_mass_PauliVillars",
        alias = "quark_mass_pauli_villars"
    )]
    pub quark_mass_pauli_villars: f64,
    pub domain_wall_height: f64,
    pub extent_of_5th_dimension: usize,
    #[serde(default = "mobius_b")]
    pub coefficient_b: f64,
    #[serde(default = "mobius_c")]
    pub coefficient_c: f64,
    /// [x, y, z, t]: 1 periodic, −1 antiperiodic.
    #[serde(default = "antiperiodic_time")]
    pub boundary_condition: [i32; ND],
    /// Neff's α (1: the plain operator). It cancels in `operator_type: PV_right`.
    #[serde(default = "unit")]
    pub neff_alpha: f64,
    /// α of the Pauli-Villars operator. 1 by default: on 12⁴ α in D alone is what helps the
    /// multigrid (docs/MDWF_MULTIGRID_LOG.md).
    #[serde(
        default = "unit",
        rename = "neff_alpha_PauliVillars",
        alias = "neff_alpha_pauli_villars"
    )]
    pub neff_alpha_pauli_villars: f64,
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

/// The source of the propagator.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBlock {
    #[serde(default = "local")]
    pub source_type: SourceType,
    /// [x, y, z, t]
    #[serde(default)]
    pub source_position: [usize; ND],
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
    /// The multigrid bootstrap after it: rounds, FGMRES steps per vector and round.
    #[serde(default = "two")]
    pub bootstrap_number_of_rounds: usize,
    #[serde(default = "four")]
    pub bootstrap_number_of_steps: usize,
    #[serde(default = "setup_seed")]
    pub seed: u64,
    /// Low modes of Ã from `mdwf_evenodd_coherence` instead of the filter.
    pub checkpoint: Option<PathBuf>,
}

impl Default for SetupBlock {
    fn default() -> Self {
        Self {
            filter_number_of_vectors: filter_vectors(),
            filter_number_of_rounds: two(),
            filter_number_in_block: four(),
            bootstrap_number_of_rounds: two(),
            bootstrap_number_of_steps: four(),
            seed: setup_seed(),
            checkpoint: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmootherBlock {
    #[serde(default = "chebyshev_smoother")]
    pub smoother_type: SmootherType,
    #[serde(default = "cheb_steps")]
    pub number_of_steps: usize,
    #[serde(default = "cheb_ratio")]
    pub interval_ratio: f64,
}

impl Default for SmootherBlock {
    fn default() -> Self {
        Self {
            smoother_type: SmootherType::Chebyshev,
            number_of_steps: cheb_steps(),
            interval_ratio: cheb_ratio(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelBlock {
    pub block_size: [usize; ND],
    #[serde(default = "test_vectors")]
    pub number_of_test_vectors: usize,
    /// The quark factor P†DP.
    #[serde(rename = "Coarse_Solver_D", default = "coarse_d")]
    pub coarse_d: SolverBlock,
    /// The second factor: P†D_PV†P, or P†D†P with `operator_type: Normal_left`.
    #[serde(rename = "Coarse_Solver_PV", default = "coarse_pv")]
    pub coarse_pv: SolverBlock,
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
fn chebyshev_smoother() -> SmootherType {
    SmootherType::Chebyshev
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

impl Params {
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
        match self.main.backend {
            Backend::Cpu => {}
            Backend::Simd => {
                return Err(invalid(
                    "Main.backend: simd has no domain-wall operator yet; use cpu".into(),
                ))
            }
            Backend::Gpu => {
                return Err(invalid(
                    "Main.backend: gpu is not implemented yet; use cpu".into(),
                ))
            }
        }
        if self.main.grid_size != [1; ND] {
            return Err(invalid(
                "Main.grid_size: the domain-wall multigrid runs on one process so far".into(),
            ));
        }
        if self.fopr.gamma_matrix_type != GammaType::Dirac {
            return Err(invalid(
                "Fopr.gamma_matrix_type: only Dirac is implemented for domain-wall fermions".into(),
            ));
        }
        if let Some(src) = &self.source {
            let n = src.number_of_rhs_per_solve;
            if !(1..=12).contains(&n) {
                return Err(invalid(format!(
                    "Source.number_of_rhs_per_solve: 1 to 12, not {n}"
                )));
            }
            if src
                .source_position
                .iter()
                .zip(&self.main.lattice_size)
                .any(|(x, l)| x >= l)
            {
                return Err(invalid(format!(
                    "Source.source_position {:?} is outside the lattice {:?}",
                    src.source_position, self.main.lattice_size
                )));
            }
        }
        if let Some(b) = self.fopr.boundary_condition.iter().find(|b| b.abs() != 1) {
            return Err(invalid(format!(
                "Fopr.boundary_condition: 1 (periodic) or -1 (antiperiodic), not {b}"
            )));
        }
        let s = &self.solver;
        s.check_preconditioner("Solver", self.multigrid.is_some(), notes)?;
        let operator = s.operator();
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
            level.coarse_d.check_coarse(
                "MultiGrid.MultiGrid_Level1.Coarse_Solver_D",
                false,
                normal,
            )?;
            level.coarse_pv.check_coarse(
                "MultiGrid.MultiGrid_Level1.Coarse_Solver_PV",
                true,
                normal,
            )?;
            let n = level.number_of_test_vectors;
            if !n.is_multiple_of(2) || n / 2 > mg.setup.filter_number_of_vectors {
                return Err(invalid(
                    "MultiGrid.MultiGrid_Level1: number_of_test_vectors must be even and at most twice Setup.filter_number_of_vectors".into(),
                ));
            }
        }
        Ok(())
    }

    /// The domain-wall operator of `Fopr` (α = 1; set α with `Mdwf::set_alpha`).
    pub fn mdwf_params(&self, mass: f64) -> MdwfParams {
        let f = &self.fopr;
        let boundary = f.boundary_condition.map(|b| {
            if b < 0 {
                Boundary::Antiperiodic
            } else {
                Boundary::Periodic
            }
        });
        MdwfParams {
            ls: f.extent_of_5th_dimension,
            m5: f.domain_wall_height,
            mass,
            b: f.coefficient_b,
            c: f.coefficient_c,
            boundary,
        }
    }

    /// The even-odd form of the operator, and whether D_PV is replaced by D (CGNE).
    pub fn form(&self) -> (EoForm, bool) {
        match self.solver.operator() {
            OperatorType::PvRight => (EoForm::Right, false),
            OperatorType::PvLeft => (EoForm::Left, false),
            OperatorType::NormalLeft => (EoForm::Left, true),
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
        let cycle = PvEoParams {
            chebyshev_steps: mg.smoother.number_of_steps,
            chebyshev_ratio: mg.smoother.interval_ratio,
            coarse_tolerance_sqr: level.coarse_d.convergence_criterion_squared,
            coarse_max_iterations: level.coarse_d.maximum_number_of_iteration,
            pv_tolerance_sqr: level.coarse_pv.convergence_criterion_squared,
            pv_max_iterations: level.coarse_pv.maximum_number_of_iteration,
            d_solver: level.coarse_d.krylov(),
            pv_solver: level.coarse_pv.krylov(),
            ..PvEoParams::default()
        };
        let low = LowModeParams {
            vectors: mg.setup.filter_number_of_vectors,
            rounds: mg.setup.filter_number_of_rounds,
            chunk: mg.setup.filter_number_in_block,
            seed: mg.setup.seed,
        };
        let boot = BootstrapParams {
            rounds: mg.setup.bootstrap_number_of_rounds,
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
        writeln!(
            f,
            "{}: Ls = {}, M5 = {}, m = {}, m_PV = {}, b = {}, c = {}, α = {} (D_PV: {})",
            fo.fermion_type,
            fo.extent_of_5th_dimension,
            fo.domain_wall_height,
            fo.quark_mass,
            fo.quark_mass_pauli_villars,
            fo.coefficient_b,
            fo.coefficient_c,
            fo.neff_alpha,
            fo.neff_alpha_pauli_villars
        )?;
        let operator = s.operator();
        write!(
            f,
            "{} on {}{}, to |r|²/|b|² < {:e}",
            s.solver_type,
            operator,
            match operator {
                OperatorType::PvRight => " (A′, α cancels)",
                OperatorType::PvLeft => " (left-preconditioned A′)",
                OperatorType::NormalLeft => " (CGNE)",
            },
            s.convergence_criterion_squared
        )?;
        match (s.preconditioner_type, &self.multigrid) {
            (PreconditionerType::Multigrid, Some(mg)) => {
                let l = &mg.level1;
                write!(
                    f,
                    "\npreconditioned by the multigrid: {:?} blocks, {} test vectors; coarse {} (P†DP) to {:e} and {} ({}) to {:e} in {}; smoother Chebyshev {}/{}",
                    l.block_size,
                    l.number_of_test_vectors,
                    l.coarse_d.solver_type,
                    l.coarse_d.convergence_criterion_squared,
                    l.coarse_pv.solver_type,
                    if operator == OperatorType::NormalLeft { "P†D†P" } else { "P†D_PV†P" },
                    l.coarse_pv.convergence_criterion_squared,
                    mg.coarse_precision,
                    mg.smoother.number_of_steps,
                    mg.smoother.interval_ratio
                )
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
                "\n{} source at {:?}, {} columns per solve: meson correlators and the residual mass",
                src.source_type, src.source_position, src.number_of_rhs_per_solve
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
}
