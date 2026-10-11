//! The application layer: read a parameter file, build the pieces it names, run, report.
//! Nothing below this crate knows that parameter files exist.

mod bench;
mod config;
pub mod keyword;
pub mod params;

pub use bench::wilson_benchmark;

pub use config::{
    BoundaryConfig, ChannelsConfig, Config, FermionConfig, GaugeConfig, GaugeFixingConfig,
    GaugeFormat, MeasureConfig, MultigridConfig, OutputConfig, Precision, QuarkConfig,
    SolverConfig, SolverKind, SourceConfig,
};

use crate::fermion::{
    solve_even_odd, Bound, Clover, CloverParams, CloverSimd, DiagTerm, EvenOdd, FermionOp,
    GaugeError, Variant, Wilson, WilsonOp, WilsonParams, WilsonSimd,
};
use crate::gauge::{fix_coulomb, plaquette, GaugeFixError};
use crate::io::{BridgeText, GaugeFile, Ildg, IoError, Scidac};
use crate::kernel::{Dirac, GammaRepre, LaneOf, SPINOR_REALS};
use crate::measure::{
    channel, exponential_source, meson_correlator, point_source, CHANNELS, NUM_SOURCES,
};
use crate::solver::preconditioner::multigrid::{Multigrid, MultigridError};
use crate::solver::{
    solve_cgne, BiCgStab, Cg, Convert, Fgmres, SolveError, SolveInfo, Solver, Stop,
};
use crate::{
    Aosoa, Blocks, Boundary, Comm, ConvertFrom, EvenOddIndex, Field, Full, Gauge, Half, Lattice,
    Layout, LinearFn, SingleComm, Spinor, Stencil, Vector, C64, ND,
};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type V = Field<f64, Spinor>;

#[derive(Debug)]
pub enum CliError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The parameter file is not valid TOML/YAML or does not have the expected structure.
    Parse(String),
    /// The parameter file is well formed but asks for something impossible.
    Invalid(String),
    /// The lattice cannot be split over the processes the way that was asked for.
    Grid(String),
    Gauge(IoError),
    GaugeFixing(GaugeFixError),
    Operator {
        quark: usize,
        source: GaugeError,
    },
    Solve {
        quark: usize,
        source: SolveError,
    },
    /// The multigrid setup failed.
    Multigrid {
        quark: usize,
        source: MultigridError,
    },
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            CliError::Parse(m) => write!(f, "parameter file: {m}"),
            CliError::Invalid(m) => write!(f, "parameter file: {m}"),
            CliError::Grid(m) => write!(f, "process grid: {m}"),
            CliError::Gauge(e) => write!(f, "gauge configuration: {e}"),
            CliError::GaugeFixing(e) => write!(f, "gauge configuration: {e}"),
            CliError::Operator { quark, source } => write!(f, "quark {quark}: {source}"),
            CliError::Solve { quark, source } => write!(f, "quark {quark}: solver: {source}"),
            CliError::Multigrid { quark, source } => {
                write!(f, "quark {quark}: multigrid setup: {source}")
            }
        }
    }
}

impl std::error::Error for CliError {}

/// What happened when one quark's propagator was computed.
#[derive(Debug, Clone)]
pub struct SolveReport {
    /// 1-based index of the quark in the parameter file.
    pub quark: usize,
    pub iterations: usize,
    pub restarts: usize,
    /// Worst |D S - b|^2 / |b|^2 over the twelve right-hand sides, from the full operator.
    pub worst_residual_sqr: f64,
    pub seconds: f64,
    /// The peak memory of the process so far, in GB (where the system reports it).
    pub peak_memory_gb: Option<f64>,
}

/// The memory of this process in GB: (now, peak so far), from `/proc/self/status`
/// (VmRSS, VmHWM) on Linux; `None` elsewhere.
pub fn memory_usage() -> Option<(f64, f64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = |name: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
            .map(|kb| kb / 1048576.0)
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}

/// `memory 1.23 GB, peak 2.34 GB`, or nothing where the system does not say.
pub fn memory_line() -> String {
    memory_usage().map_or(String::new(), |(now, peak)| {
        format!("memory {now:.2} GB, peak {peak:.2} GB")
    })
}

#[derive(Debug, Clone)]
pub struct Correlator {
    /// 1-based indices of the two quarks.
    pub quarks: (usize, usize),
    pub channel: String,
    /// One value per time slice.
    pub values: Vec<C64>,
}

#[derive(Debug, Clone)]
pub struct Report {
    /// Processes per direction.
    pub grid: [usize; ND],
    pub plaquette: f64,
    /// Sweeps the gauge fixing took, if the parameter file asked for it.
    pub gauge_fixing_sweeps: Option<usize>,
    pub solves: Vec<SolveReport>,
    pub correlators: Vec<Correlator>,
}

fn invalid<T>(message: String) -> Result<T, CliError> {
    Err(CliError::Invalid(message))
}

/// Split the lattice over `processes` processes: each prime factor goes to the direction
/// whose local extent is largest among those it divides (the later direction on a tie).
fn choose_grid(lattice: [usize; ND], processes: usize) -> Option<[usize; ND]> {
    let (mut grid, mut local, mut left) = ([1; ND], lattice, processes);
    let mut factors = Vec::new();
    let mut p = 2;
    while left > 1 {
        if left.is_multiple_of(p) {
            factors.push(p);
            left /= p;
        } else {
            p += 1;
        }
    }
    for p in factors.into_iter().rev() {
        let mu = (0..ND)
            .filter(|mu| local[*mu].is_multiple_of(p))
            .max_by_key(|mu| local[*mu])?;
        grid[mu] *= p;
        local[mu] /= p;
    }
    Some(grid)
}

/// The process grid to use: the one asked for if it fits, otherwise one chosen here.
fn process_grid(config: &Config, processes: usize) -> Result<[usize; ND], CliError> {
    let lattice = config.lattice;
    if lattice.contains(&0) {
        return invalid(format!("lattice extents must be positive, got {lattice:?}"));
    }
    match config.process_grid {
        Some(grid) => {
            let fits = (0..ND).all(|mu| grid[mu] > 0 && lattice[mu].is_multiple_of(grid[mu]));
            let needed: usize = grid.iter().product();
            if needed != processes {
                // One process where several were asked for usually means the launcher
                // belongs to another MPI installation than the one the program was
                // built with: every process then believes it is alone.
                let hint = if processes == 1 {
                    "; if the program was started with mpirun, check that mpirun comes \
                     from the MPI installation the program was built with"
                } else {
                    ""
                };
                Err(CliError::Grid(format!(
                    "process_grid = {grid:?} needs {needed} processes, but there are \
                     {processes}{hint}"
                )))
            } else if !fits {
                Err(CliError::Grid(format!(
                    "{grid:?} does not divide the lattice {lattice:?}"
                )))
            } else {
                Ok(grid)
            }
        }
        None => choose_grid(lattice, processes).ok_or_else(|| {
            CliError::Grid(format!(
                "the lattice {lattice:?} cannot be split evenly over {processes} processes"
            ))
        }),
    }
}

/// Checks that do not need the gauge configuration.
fn validate(config: &Config, grid: [usize; ND]) -> Result<Vec<String>, CliError> {
    let local: [usize; ND] = core::array::from_fn(|mu| config.lattice[mu] / grid[mu]);
    if config.solver.even_odd && local.iter().any(|l| l % 2 != 0) {
        return if grid == [1; ND] {
            invalid(format!(
                "even_odd needs even lattice extents, got {:?}; set solver.even_odd = false",
                config.lattice
            ))
        } else {
            Err(CliError::Grid(format!(
                "even_odd needs even extents on every process, but {grid:?} gives {local:?}; \
                 set process_grid in the parameter file or set solver.even_odd = false"
            )))
        };
    }
    if config.quark.is_empty() {
        return invalid("at least one quark is needed".to_string());
    }
    if config.quark.len() > 9 {
        return invalid("at most nine quarks are supported".to_string());
    }
    if let Some(fixing) = &config.gauge.fixing {
        let p = fixing.coulomb_params();
        if config.lattice.iter().any(|l| l % 2 != 0) {
            return invalid(format!(
                "gauge fixing needs even lattice extents, got {:?}",
                config.lattice
            ));
        }
        if p.tolerance_sqr.is_nan()
            || p.tolerance_sqr <= 0.0
            || p.overrelaxation.is_nan()
            || p.overrelaxation <= 0.0
            || p.measure_interval == 0
            || p.reset_interval == 0
        {
            return invalid(
                "gauge.fixing: tolerance_squared, overrelaxation, measure_interval and \
                 reset_interval must be positive"
                    .to_string(),
            );
        }
    }
    if let Some(mg) = &config.solver.multigrid {
        validate_multigrid(config, mg, grid)?;
    }
    validate_simd(config, grid, local)?;
    let tolerance = config.solver.tolerance_squared;
    if tolerance.is_nan() || tolerance <= 0.0 || config.solver.max_iterations == 0 {
        return invalid(
            "solver.tolerance_squared and solver.max_iterations must be positive".to_string(),
        );
    }
    for (i, q) in config.quark.iter().enumerate() {
        let position = match q.source {
            SourceConfig::Point { position } | SourceConfig::Exponential { position, .. } => {
                position
            }
        };
        if (0..ND).any(|mu| position[mu] >= config.lattice[mu]) {
            return invalid(format!(
                "quark {}: source position {position:?} is outside the lattice {:?}",
                i + 1,
                config.lattice
            ));
        }
    }
    let channels = match &config.measure.channels {
        ChannelsConfig::All => CHANNELS.iter().map(|c| c.to_string()).collect(),
        ChannelsConfig::List(list) => list.clone(),
    };
    for name in &channels {
        if channel::<Dirac>(name).is_none() {
            return invalid(format!(
                "unknown channel {name:?}; known channels: {}",
                CHANNELS.join(", ")
            ));
        }
    }
    Ok(channels)
}

/// Checks on `solver.simd` that do not need the gauge configuration.
fn validate_simd(config: &Config, grid: [usize; ND], local: [usize; ND]) -> Result<(), CliError> {
    let lanes = config.solver.simd;
    if lanes == 1 {
        return Ok(());
    }
    if !SIMD_WIDTHS.contains(&lanes) {
        return invalid(format!(
            "solver.simd must be 1 or one of {SIMD_WIDTHS:?}, not {lanes}"
        ));
    }
    if config.solver.multigrid.is_some() {
        return invalid(
            "solver.multigrid works on fields in the usual layout; set solver.simd = 1".to_string(),
        );
    }
    // On several processes it is the part of the lattice on one process that is cut.
    let on_one = if grid == [1; ND] {
        String::new()
    } else {
        format!(" on each process (process_grid {grid:?})")
    };
    let Some(blocks) = Blocks::new(local, lanes) else {
        return invalid(format!(
            "solver.simd = {lanes}: the lattice {local:?}{on_one} cannot be halved often enough \
             to give {lanes} equal parts"
        ));
    };
    if config.solver.even_odd && blocks.even_odd_index().is_none() {
        return invalid(format!(
            "solver.simd = {lanes} with even_odd: the lattice {local:?}{on_one} is cut into \
             {lanes} parts of extents {:?}, which must all be even; use a smaller solver.simd \
             or set solver.even_odd = false",
            blocks.reduced()
        ));
    }
    Ok(())
}

/// The values of `solver.simd` above 1 that `run_on` has an operator for.
const SIMD_WIDTHS: [usize; 3] = [2, 4, 8];

type Spinors<L> = Field<f64, Spinor, Full, L>;

/// Solve D S = b for the twelve right-hand sides of `b`, the way the parameter file asks.
/// The fields may be in any layout the operator works on; `index` is the operator's
/// even-odd numbering.
fn propagator<L, F>(
    op: &F,
    index: Option<&EvenOddIndex>,
    b: &Spinors<L>,
    solver: &SolverConfig,
    m: Option<&mut dyn LinearFn<Spinors<L>>>,
) -> Result<(Spinors<L>, SolveInfo), SolveError>
where
    L: Layout,
    F: FermionOp<Spinors<L>> + EvenOdd<Field<f64, Spinor, Half, L>>,
{
    let stop = Stop {
        tolerance_sqr: solver.tolerance_squared,
        max_iterations: solver.max_iterations,
        max_restarts: solver.max_restarts,
    };
    let mut s = b.zeros_like();
    let info = match (solver.even_odd, solver.kind) {
        (true, kind) => {
            let index = index.expect("even extents were checked when the file was read");
            solve_even_odd(op, index, &mut s, b, |schur, x_e, rhs| match kind {
                SolverKind::CgNormal => solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
                SolverKind::Bicgstab => BiCgStab::new(stop).solve(schur, None, x_e, rhs),
                SolverKind::Fgmres => Fgmres::new(stop).solve(schur, None, x_e, rhs),
            })?
        }
        (false, SolverKind::CgNormal) => solve_cgne(
            &mut Cg::new(stop),
            &mut Bound::new(op, Variant::D),
            &mut s,
            b,
        )?,
        (false, SolverKind::Bicgstab) => {
            BiCgStab::new(stop).solve(&mut Bound::new(op, Variant::D), m, &mut s, b)?
        }
        (false, SolverKind::Fgmres) => {
            Fgmres::new(stop).solve(&mut Bound::new(op, Variant::D), m, &mut s, b)?
        }
    };
    Ok((s, info))
}

/// Solve for one quark with `op`, building the multigrid preconditioner first when the
/// parameter file asks for one. `single` builds the single-precision copy of `op` that a
/// single-precision preconditioner works with. Returns the worst residual, the solution
/// and what the solver reported.
fn solve_quark<D, D32>(
    lat: &Arc<Lattice>,
    op: &WilsonOp<f64, Dirac, D>,
    single: impl FnOnce() -> Result<WilsonOp<f32, Dirac, D32>, GaugeError>,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError>
where
    D: DiagTerm<f64>,
    D32: DiagTerm<f32>,
{
    let setup_failed = |e| CliError::Multigrid { quark, source: e };
    let index = op.even_odd_index().map(|index| &**index);
    let solved = match &solver.multigrid {
        None => propagator(op, index, b, solver, None),
        Some(mg) => match mg.precision {
            Precision::Double => {
                let mut m = Multigrid::new(op, lat, &mg.params()).map_err(setup_failed)?;
                propagator(op, index, b, solver, Some(&mut m))
            }
            Precision::Single => {
                let op32 = single().map_err(|e| CliError::Operator { quark, source: e })?;
                let m32 = Multigrid::new(&op32, lat, &mg.params()).map_err(setup_failed)?;
                let like = Field::<f32, Spinor>::spinor(lat, b.num_vectors());
                let mut m = Convert::new(m32, &like);
                // Bound to a name so that the preconditioner, which borrows op32, is dropped
                // before op32 is.
                #[allow(clippy::let_and_return)]
                let solved = propagator(op, index, b, solver, Some(&mut m));
                solved
            }
        },
    };
    let (s, info) = solved.map_err(|e| CliError::Solve { quark, source: e })?;
    Ok((worst_residual(op, &s, b), s, info))
}

/// Checks on the multigrid section that do not need the gauge configuration.
fn validate_multigrid(
    config: &Config,
    mg: &MultigridConfig,
    grid: [usize; ND],
) -> Result<(), CliError> {
    // The cycle is not a fixed linear map: the outer solver must be a flexible one.
    if !matches!(
        config.solver.kind,
        SolverKind::Fgmres | SolverKind::Bicgstab
    ) {
        return invalid(
            "solver.multigrid needs solver.type = fgmres or bicgstab (flexible with a preconditioner)"
                .to_string(),
        );
    }
    if config.solver.even_odd {
        return invalid(
            "solver.multigrid works on the whole lattice; set solver.even_odd = false".to_string(),
        );
    }
    if grid != [1; ND] {
        return Err(CliError::Grid(
            "multigrid runs on one process only so far".to_string(),
        ));
    }
    if (0..ND).any(|mu| mg.block[mu] == 0 || !config.lattice[mu].is_multiple_of(mg.block[mu])) {
        return invalid(format!(
            "solver.multigrid.block {:?} must divide the lattice {:?}",
            mg.block, config.lattice
        ));
    }
    // Six complex components per chirality and site.
    let aggregate_dim = 6 * mg.block.iter().product::<usize>();
    if mg.test_vectors == 0 || mg.test_vectors > aggregate_dim {
        return invalid(format!(
            "solver.multigrid.test_vectors must be between 1 and {aggregate_dim} for blocks of {:?}",
            mg.block
        ));
    }
    let smoother_tolerance = mg.smoother_tolerance_squared;
    if smoother_tolerance.is_nan() || smoother_tolerance < 0.0 || mg.smoother_iterations == 0 {
        return invalid(
            "solver.multigrid.smoother_iterations must be positive and smoother_tolerance_squared not negative"
                .to_string(),
        );
    }
    let tolerance = mg.coarse_tolerance_squared;
    if tolerance.is_nan() || tolerance <= 0.0 || mg.coarse_max_iterations == 0 {
        return invalid(
            "solver.multigrid.coarse_tolerance_squared and coarse_max_iterations must be positive"
                .to_string(),
        );
    }
    Ok(())
}

/// `solve_quark` with the fields in blocks of `W` sites (`solver.simd`). The source is
/// rearranged into blocks, the solvers and the operator work on blocks throughout, and the
/// solution is rearranged back.
fn solve_quark_simd<D: DiagTerm<f64>, const W: usize>(
    op: Option<WilsonOp<f64, Dirac, D, Aosoa<W>>>,
    u: &Field<f64, Gauge>,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError>
where
    f64: LaneOf<W>,
{
    let mut op = op.expect("that the lattice can be cut was checked when the file was read");
    op.set_gauge(u)
        .map_err(|e| CliError::Operator { quark, source: e })?;
    let mut rhs = Spinors::<Aosoa<W>>::with_shape(b.lattice(), SPINOR_REALS, 1, NUM_SOURCES, 1);
    rhs.convert_from(b);
    let index = op.even_odd_index().map(|index| &**index);
    let (blocks, info) = propagator(&op, index, &rhs, solver, None)
        .map_err(|e| CliError::Solve { quark, source: e })?;
    let mut s = b.zeros_like();
    s.convert_from(&blocks);
    Ok((worst_residual(&op, &blocks, &rhs), s, info))
}

/// Worst |D S - b|^2 / |b|^2 over the right-hand sides, using the full operator.
fn worst_residual<X: Vector, F: FermionOp<X>>(op: &F, s: &X, b: &X) -> f64 {
    let mut r = b.zeros_like();
    Bound::new(op, Variant::D).mult(&mut r, s);
    r.scale(&[-1.0; NUM_SOURCES]);
    r.axpy(&[C64::new(1.0, 0.0); NUM_SOURCES], b);
    let (mut rr, mut bb) = ([0.0; NUM_SOURCES], [0.0; NUM_SOURCES]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    rr.iter().zip(&bb).fold(0.0f64, |m, (r, b)| m.max(r / b))
}

/// Run what the parameter file describes on one process. Relative paths in it are
/// resolved against `base`.
pub fn run(config: &Config, base: &Path) -> Result<Report, CliError> {
    run_on(config, base, Arc::new(SingleComm))
}

/// Run what the parameter file describes on the processes of `comm`; every one of them
/// must make this call and gets the same report.
pub fn run_on(config: &Config, base: &Path, comm: Arc<dyn Comm>) -> Result<Report, CliError> {
    let grid = process_grid(config, comm.size())?;
    let channels = validate(config, grid)?;
    let lat = Lattice::new(config.lattice, grid, comm);

    let gauge_path = base.join(&config.gauge.file);
    let format: &dyn GaugeFile = match config.gauge.format {
        GaugeFormat::BridgeText => &BridgeText,
        GaugeFormat::Ildg => &Ildg::default(),
        GaugeFormat::Scidac => &Scidac::default(),
        #[cfg(feature = "hdf5")]
        GaugeFormat::Hdf5 => &crate::io::Hdf5::default(),
        #[cfg(not(feature = "hdf5"))]
        GaugeFormat::Hdf5 => {
            return invalid(
                "gauge.format = hdf5 needs a program built with the feature hdf5".to_string(),
            )
        }
    };
    let mut u = format.read(&gauge_path, &lat).map_err(CliError::Gauge)?;
    let plaq = plaquette(&u, &Stencil::nearest(&lat, [Boundary::Periodic; ND]));
    let gauge_fixing_sweeps = match &config.gauge.fixing {
        None => None,
        Some(fixing) => Some(
            fix_coulomb(&mut u, &fixing.coulomb_params())
                .map_err(CliError::GaugeFixing)?
                .iterations,
        ),
    };

    let mut solves = Vec::new();
    let mut props: Vec<V> = Vec::new();
    for (i, q) in config.quark.iter().enumerate() {
        let quark = i + 1;
        let b = match q.source {
            SourceConfig::Point { position } => point_source::<f64>(&lat, position),
            SourceConfig::Exponential {
                position,
                slope,
                power,
            } => exponential_source::<f64>(&lat, position, slope, power),
        };
        let start = Instant::now();
        // The number of sites per block is part of the type of the operator: one call for
        // each value `solver.simd` may have.
        macro_rules! in_blocks {
            ($operator:ident, $params:expr) => {
                match config.solver.simd {
                    2 => solve_quark_simd(
                        $operator::<f64, 2>::new(&lat, $params),
                        &u,
                        &b,
                        &config.solver,
                        quark,
                    )?,
                    4 => solve_quark_simd(
                        $operator::<f64, 4>::new(&lat, $params),
                        &u,
                        &b,
                        &config.solver,
                        quark,
                    )?,
                    8 => solve_quark_simd(
                        $operator::<f64, 8>::new(&lat, $params),
                        &u,
                        &b,
                        &config.solver,
                        quark,
                    )?,
                    lanes => unreachable!("solver.simd = {lanes} was refused by validate"),
                }
            };
        }
        // The two fermion types differ only in the operator that is built; everything after
        // that is the same generic code.
        let solved = match q.fermion {
            FermionConfig::Wilson { kappa, boundary } if config.solver.simd > 1 => {
                let params = WilsonParams {
                    kappa,
                    boundary: boundary.map(Into::into),
                };
                in_blocks!(WilsonSimd, params)
            }
            FermionConfig::Clover {
                kappa,
                csw,
                boundary,
            } if config.solver.simd > 1 => {
                let params = CloverParams {
                    kappa,
                    csw,
                    boundary: boundary.map(Into::into),
                };
                in_blocks!(CloverSimd, params)
            }
            FermionConfig::Wilson { kappa, boundary } => {
                let mut op = Wilson::<f64>::new(
                    &lat,
                    WilsonParams {
                        kappa,
                        boundary: boundary.map(Into::into),
                    },
                );
                op.set_gauge(&u)
                    .map_err(|e| CliError::Operator { quark, source: e })?;
                let single = || {
                    let mut op = Wilson::<f32>::new(&lat, *op.params());
                    op.set_gauge(&u).map(|()| op)
                };
                solve_quark(&lat, &op, single, &b, &config.solver, quark)?
            }
            FermionConfig::Clover {
                kappa,
                csw,
                boundary,
            } => {
                let params = CloverParams {
                    kappa,
                    csw,
                    boundary: boundary.map(Into::into),
                };
                let mut op = Clover::<f64>::new(&lat, params);
                op.set_gauge(&u)
                    .map_err(|e| CliError::Operator { quark, source: e })?;
                let single = || {
                    let mut op = Clover::<f32>::new(&lat, params);
                    op.set_gauge(&u).map(|()| op)
                };
                solve_quark(&lat, &op, single, &b, &config.solver, quark)?
            }
        };
        let (worst_residual_sqr, s, info) = solved;
        solves.push(SolveReport {
            quark,
            iterations: info.iterations,
            restarts: info.restarts,
            worst_residual_sqr,
            seconds: start.elapsed().as_secs_f64(),
            peak_memory_gb: memory_usage().map(|(_, peak)| peak),
        });
        props.push(s);
    }

    let mut correlators = Vec::new();
    for i in 0..props.len() {
        for j in i..props.len() {
            for name in &channels {
                let (sink, source) = channel::<Dirac>(name).expect("channel names were checked");
                correlators.push(Correlator {
                    quarks: (i + 1, j + 1),
                    channel: name.clone(),
                    values: meson_correlator(&props[i], &props[j], &sink, &source, &Dirac::GAMMA5),
                });
            }
        }
    }
    Ok(Report {
        grid,
        plaquette: plaq,
        gauge_fixing_sweeps,
        solves,
        correlators,
    })
}

/// Write the report: comment lines starting with `#`, then one line per quark pair,
/// channel and time slice: `<i><j> <channel> <t> <re> <im>`.
pub fn write_report(report: &Report, config: &Config, out: &mut dyn Write) -> std::io::Result<()> {
    writeln!(out, "# lattice    {:?}", config.lattice)?;
    if report.grid != [1; ND] {
        writeln!(out, "# processes  {:?}", report.grid)?;
    }
    writeln!(out, "# plaquette  {:.15}", report.plaquette)?;
    if let Some(sweeps) = report.gauge_fixing_sweeps {
        writeln!(out, "# gauge      Coulomb gauge after {sweeps} sweeps")?;
    }
    for s in &report.solves {
        writeln!(
            out,
            "# quark {}    {} iterations, {} restarts, worst |D S - b|^2/|b|^2 = {:.3e}, {:.3} s{}",
            s.quark,
            s.iterations,
            s.restarts,
            s.worst_residual_sqr,
            s.seconds,
            s.peak_memory_gb
                .map_or(String::new(), |p| format!(", peak memory {p:.2} GB"))
        )?;
    }
    writeln!(out, "# quarks channel t re im")?;
    for c in &report.correlators {
        for (t, z) in c.values.iter().enumerate() {
            writeln!(
                out,
                "{}{} {} {} {:.15e} {:.15e}",
                c.quarks.0, c.quarks.1, c.channel, t, z.re, z.im
            )?;
        }
    }
    Ok(())
}

/// Load, run on one process, print to `stdout`, and write the output file if the
/// parameter file names one.
pub fn run_file(path: &Path, stdout: &mut dyn Write) -> Result<(), CliError> {
    run_file_on(path, stdout, Arc::new(SingleComm))
}

/// `run_file` on the processes of `comm`. Only rank 0 prints and writes.
pub fn run_file_on(
    path: &Path,
    stdout: &mut dyn Write,
    comm: Arc<dyn Comm>,
) -> Result<(), CliError> {
    let config = Config::load(path)?;
    let base = path.parent().unwrap_or(Path::new("."));
    let root = comm.rank() == 0;
    let report = run_on(&config, base, comm)?;
    if !root {
        return Ok(());
    }
    let io_error = |p: &Path| {
        let p = p.to_path_buf();
        move |e| CliError::Io { path: p, source: e }
    };
    match write_report(&report, &config, stdout) {
        // The reader went away (`ruttice ... | head`): not an error.
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
        other => other.map_err(io_error(Path::new("<stdout>")))?,
    }
    if let Some(file) = &config.output.file {
        let target = base.join(file);
        let mut f = std::fs::File::create(&target).map_err(io_error(&target))?;
        write_report(&report, &config, &mut f).map_err(io_error(&target))?;
    }
    Ok(())
}

const USAGE: &str = "\
usage: ruttice PARAMETER_FILE

Reads a parameter file (.toml, .yaml or .yml), solves for the quark propagators it
describes and prints meson two-point functions. See examples/ for parameter files.";

/// The whole program: interpret the command line (without the program name), run, and
/// return the exit code. Every process of `comm` makes this call; only rank 0 prints.
pub fn command_line(args: &[String], comm: Arc<dyn Comm>, stdout: &mut dyn Write) -> u8 {
    let root = comm.rank() == 0;
    let file = match args {
        [flag] if flag == "-h" || flag == "--help" => {
            if root {
                // Nothing useful can be done if printing the help fails.
                let _ = writeln!(stdout, "{USAGE}");
            }
            return 0;
        }
        [file] if !file.starts_with('-') => file,
        _ => {
            if root {
                eprintln!("{USAGE}");
            }
            return 1;
        }
    };
    match run_file_on(Path::new(file), stdout, comm) {
        Ok(()) => 0,
        Err(e) => {
            if root {
                eprintln!("ruttice: {e}");
            }
            1
        }
    }
}
