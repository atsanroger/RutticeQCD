//! The application layer: read a parameter file, build the pieces it names, run, report.
//! Nothing below this crate knows that parameter files exist.

mod bench;
mod config;
pub mod eigen;
pub mod flow;
pub mod hmc;
pub mod hmc_params;
pub mod keyword;
pub mod mdwf;
#[cfg(feature = "cuda-bridge")]
mod mdwf_cuda;
pub mod params;
mod wilson;
#[cfg(feature = "cuda-bridge")]
mod wilson_cuda;

pub use bench::wilson_benchmark;
pub use wilson::{run, run_on, write_report, Correlator, Report, SolveReport};

pub use config::{
    BoundaryConfig, ChannelsConfig, Config, FermionConfig, GaugeConfig, GaugeFixingConfig,
    GaugeFormat, GaugeSmearingConfig, MeasureConfig, MultigridConfig, OutputConfig, Precision,
    QuarkConfig, SolverConfig, SolverKind, SourceConfig,
};

use crate::fermion::{
    solve_even_odd, Bound, Clover, CloverParams, CloverSimd, DiagTerm, EvenOdd, FermionOp,
    GaugeError, OnParity, SplitParity, Variant, Wilson, WilsonOp, WilsonParams, WilsonSimd,
};
use crate::gauge::{fix_coulomb, plaquette, GaugeFixError};
use crate::io::{Binary, BridgeText, GaugeFile, Ildg, IoError, Nersc, Scidac};
use crate::kernel::{Dirac, GammaRepre, LaneOf, SPINOR_REALS};
use crate::measure::{
    channel, exponential_source, meson_correlator, point_source, CHANNELS, NUM_SOURCES,
};
use crate::solver::preconditioner::multigrid::{Multigrid, MultigridError};
use crate::solver::{
    solve_cgne, BiCgStab, Cg, Convert, Fgmres, SolveError, SolveInfo, Solver, Stop,
};
use crate::{
    Aosoa, Blocks, Boundary, Comm, ConvertFrom, EvenOddIndex, Field, Gauge, Half, Lattice, Layout,
    Lexical, LinearFn, SingleComm, Spinor, Stencil, Vector, C64, ND,
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
    /// A domain-wall run failed.
    Run(String),
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
            CliError::Run(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CliError {}

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

/// The gauge-field format of a name on a command line: text, binary, nersc, ildg or scidac
/// (any case).
pub fn gauge_format_named(name: &str) -> Option<GaugeFormat> {
    Some(match name.to_ascii_lowercase().as_str() {
        "text" => GaugeFormat::BridgeText,
        "binary" => GaugeFormat::Binary,
        "nersc" => GaugeFormat::Nersc,
        "ildg" => GaugeFormat::Ildg,
        "scidac" => GaugeFormat::Scidac,
        _ => return None,
    })
}

/// The links of the file `path` in the format named `name` (`gauge_format_named`).
pub fn read_gauge_named(
    name: &str,
    path: &std::path::Path,
    lat: &std::sync::Arc<crate::Lattice>,
) -> Result<crate::Field<f64, crate::Gauge>, CliError> {
    let format = gauge_format_named(name)
        .ok_or_else(|| CliError::Invalid(format!("{name}: text, binary, nersc, ildg or scidac")))?;
    gauge_file(format)?.read(path, lat).map_err(CliError::Gauge)
}

/// The reader and writer of the gauge-field format `f`.
pub(crate) fn gauge_file(f: GaugeFormat) -> Result<Box<dyn GaugeFile>, CliError> {
    Ok(match f {
        GaugeFormat::BridgeText => Box::new(BridgeText),
        GaugeFormat::Binary => Box::new(Binary),
        GaugeFormat::Nersc => Box::new(Nersc),
        GaugeFormat::Ildg => Box::new(Ildg::default()),
        GaugeFormat::Scidac => Box::new(Scidac::default()),
        #[cfg(feature = "hdf5")]
        GaugeFormat::Hdf5 => Box::new(crate::io::Hdf5::default()),
        #[cfg(not(feature = "hdf5"))]
        GaugeFormat::Hdf5 => {
            return invalid("format hdf5 needs a program built with the feature hdf5".into())
        }
    })
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

/// The process grid for `lattice` on `processes` processes: `requested` if it fits,
/// otherwise one chosen here.
pub(crate) fn grid_for(
    lattice: [usize; ND],
    requested: Option<[usize; ND]>,
    processes: usize,
) -> Result<[usize; ND], CliError> {
    if lattice.contains(&0) {
        return invalid(format!("lattice extents must be positive, got {lattice:?}"));
    }
    match requested {
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
                    "the process grid {grid:?} needs {needed} processes, but there are \
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
    let root = comm.rank() == 0;
    let io_error = |e| CliError::Io {
        path: PathBuf::from("<stdout>"),
        source: e,
    };
    if hmc::is_hmc_file(path)? {
        let config = hmc::HmcConfig::load(path)?;
        set_threads(config.cpu_thread)?;
        let base = path.parent().unwrap_or(Path::new("."));
        let mut sink = std::io::sink();
        let out: &mut dyn Write = if root { stdout } else { &mut sink };
        return hmc::run_on(&config, base, out, comm).map(|_| ());
    }
    if params::EigenParams::is_eigen_file(path)? {
        let p = params::EigenParams::load(path)?;
        set_threads(p.main.cpu_thread)?;
        let mut sink = std::io::sink();
        let out: &mut dyn Write = if root { stdout } else { &mut sink };
        for line in p.to_string().lines() {
            writeln!(out, "# {line}").map_err(io_error)?;
        }
        return eigen::run_on(&p, path, out, comm);
    }
    if !params::is_bridge_style(path)? {
        return invalid(format!(
            "{}: no Main section. A parameter file has the sections of Bridge++ (Main, Gauge, \
             Fopr, Solver, Source, …; see examples/), or is an HMC file (hmc, actions)",
            path.display()
        ));
    }
    let p = params::Params::load(path)?;
    set_threads(p.main.cpu_thread)?;
    if p.fopr.fermion_type.is_domain_wall() {
        // Only rank 0 prints.
        let mut sink = std::io::sink();
        let out: &mut dyn Write = if root { stdout } else { &mut sink };
        writeln!(out, "{p}").map_err(io_error)?;
        return mdwf::run_on(&p, path, (91, 91), out, comm)
            .map_err(|e| CliError::Run(e.to_string()));
    }
    if root {
        for line in p.to_string().lines() {
            writeln!(stdout, "# {line}").map_err(io_error)?;
        }
    }
    wilson::run_config(&p.to_config(), path, stdout, comm)
}

/// Make the thread pool of this process with `threads` threads (`cpu_thread` of a
/// parameter file). The pool is made once: asking for another size later is an error.
fn set_threads(threads: Option<usize>) -> Result<(), CliError> {
    let Some(n) = threads else {
        return Ok(());
    };
    if n == 0 {
        return invalid("cpu_thread: at least 1".to_string());
    }
    let made = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    match (made, rayon::current_num_threads()) {
        (Err(_), have) if have != n => invalid(format!(
            "cpu_thread {n}: the thread pool already has {have} threads"
        )),
        _ => Ok(()),
    }
}

const USAGE: &str = "\
usage: ruttice PARAMETER_FILE

Reads a parameter file (.yaml, .yml or .toml) in the style of Bridge++ and solves for the
quark propagators it describes (Wilson, clover or domain-wall fermions), printing meson
two-point functions; with an Eigensolver section, the eigenvalues of the fermion operator; or
an HMC file, and generates gauge configurations. See examples/.";

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

/// With `Main.verbose_level : Detailed`: the residual history of GCR, one line per step.
pub(crate) fn history_lines(out: &mut dyn Write, label: &str, history: Option<Vec<Vec<f64>>>) {
    for (i, r) in history.into_iter().flatten().enumerate() {
        let values: Vec<String> = r.iter().map(|x| format!("{x:.3e}")).collect();
        let _ = writeln!(
            out,
            "    {label} GCR step {i:4}: |r|^2/|b|^2 {}",
            values.join(" ")
        );
    }
}
