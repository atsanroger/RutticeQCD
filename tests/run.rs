//! The parameter file and the run it drives.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::app::{
    run, BoundaryConfig, CliError, Config, FermionConfig, SolverKind, SourceConfig,
};
use ruttice::gauge::random_gauge;
use ruttice::io::write_bridge_text;
use ruttice::Lattice;
use std::path::{Path, PathBuf};

const WILSON_TOML: &str = include_str!("../examples/wilson_pion.toml");
const CLOVER_YAML: &str = include_str!("../examples/clover_two_flavours.yaml");

/// A fresh directory under the system temporary directory.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ruttice_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_example_files_parse() {
    let wilson = Config::from_toml(WILSON_TOML).unwrap();
    assert_eq!(wilson.lattice, [4, 4, 4, 8]);
    assert_eq!(wilson.solver.kind, SolverKind::CgNormal);
    assert!(wilson.solver.even_odd);
    assert_eq!(wilson.quark.len(), 1);
    assert_eq!(
        wilson.quark[0].fermion,
        FermionConfig::Wilson {
            kappa: 0.12,
            boundary: [BoundaryConfig::Antiperiodic; 4]
        }
    );
    assert_eq!(wilson.measure.channels, Some(vec!["PS".to_string()]));

    let clover = Config::from_yaml(CLOVER_YAML).unwrap();
    assert_eq!(clover.quark.len(), 2);
    assert_eq!(clover.solver.max_restarts, 3, "default");
    assert_eq!(clover.measure.channels, None, "all channels when absent");
    assert_eq!(
        clover.quark[1].source,
        SourceConfig::Exponential {
            position: [0; 4],
            slope: 0.5,
            power: 2.0
        }
    );
}

/// The same parameters written in TOML and in YAML give the same configuration.
#[test]
fn toml_and_yaml_are_equivalent() {
    let yaml = "
lattice: [4, 4, 4, 8]
gauge: {file: conf_04040408.txt, format: bridge_text}
solver: {type: cg_normal, even_odd: true, tolerance_squared: 1.0e-28, max_iterations: 2000, max_restarts: 3}
quark:
  - fermion: {type: wilson, kappa: 0.12, boundary: [antiperiodic, antiperiodic, antiperiodic, antiperiodic]}
    source: {type: point, position: [0, 0, 0, 0]}
measure: {channels: [PS]}
output: {file: wilson_pion.out}
";
    assert_eq!(
        Config::from_yaml(yaml).unwrap(),
        Config::from_toml(WILSON_TOML).unwrap()
    );
}

#[test]
fn mistakes_in_the_file_are_reported() {
    let message = |text: String| match Config::from_toml(&text) {
        Err(CliError::Parse(m)) => m,
        other => panic!("expected a parse error, got {other:?}"),
    };
    // A misspelt key is an error, not silently ignored.
    let m = message(WILSON_TOML.replace("max_iterations", "max_iteration"));
    assert!(m.contains("max_iteration"), "{m}");
    // An unknown fermion type names the alternatives.
    let m = message(WILSON_TOML.replace("\"wilson\"", "\"wilsonn\""));
    assert!(m.contains("wilsonn") && m.contains("clover"), "{m}");
    // A key that belongs to another fermion type.
    let m = message(WILSON_TOML.replace("kappa = 0.12,", "kappa = 0.12, csw = 1.0,"));
    assert!(m.contains("csw"), "{m}");
    // A misspelt gauge-fixing parameter.
    let m = message(WILSON_TOML.replace(
        "[gauge]\n",
        "[gauge]\nfixing = { type = \"coulomb\", overrelax = 1.6 }\n",
    ));
    assert!(m.contains("overrelax"), "{m}");
}

#[test]
fn impossible_requests_are_rejected_before_any_work() {
    let invalid = |text: String| match run(&Config::from_toml(&text).unwrap(), Path::new(".")) {
        Err(CliError::Invalid(m)) => m,
        other => panic!(
            "expected an invalid-parameter error, got {:?}",
            other.map(|_| ())
        ),
    };
    let m = invalid(WILSON_TOML.replace("channels = [\"PS\"]", "channels = [\"PION\"]"));
    assert!(m.contains("PION") && m.contains("A4_PS"), "{m}");
    let m = invalid(WILSON_TOML.replace("lattice = [4, 4, 4, 8]", "lattice = [4, 3, 4, 8]"));
    assert!(m.contains("even"), "{m}");
    let m = invalid(WILSON_TOML.replace("position = [0, 0, 0, 0]", "position = [0, 0, 0, 8]"));
    assert!(m.contains("outside"), "{m}");
    let m = invalid(WILSON_TOML.replace(
        "[gauge]\n",
        "[gauge]\nfixing = { type = \"coulomb\", overrelaxation = 0.0 }\n",
    ));
    assert!(m.contains("overrelaxation"), "{m}");
    // Multigrid needs FGMRES on the whole lattice and blocks that divide it.
    let with_multigrid = |block: &str| {
        WILSON_TOML.replace(
            "max_restarts = 3 ",
            &format!("multigrid = {{ block = {block} }}\nmax_restarts = 3 "),
        )
    };
    let m = invalid(with_multigrid("[2, 2, 2, 2]"));
    assert!(m.contains("fgmres"), "{m}");
    let fgmres = |text: String| text.replace("\"cg_normal\"", "\"fgmres\"");
    let m = invalid(fgmres(with_multigrid("[2, 2, 2, 2]")));
    assert!(m.contains("even_odd"), "{m}");
    let whole = |text: String| text.replace("even_odd = true", "even_odd = false");
    let m = invalid(whole(fgmres(with_multigrid("[3, 2, 2, 2]"))));
    assert!(m.contains("block"), "{m}");
}

#[test]
fn a_missing_gauge_file_is_an_error_not_a_panic() {
    let config = Config::from_toml(WILSON_TOML).unwrap();
    assert!(matches!(
        run(&config, &scratch("missing")),
        Err(CliError::Gauge(_))
    ));
}

/// End to end on a configuration written by this code: every way of solving that the
/// parameter file offers gives the same correlator, and the output file is written.
#[test]
fn every_solver_choice_gives_the_same_correlator() {
    let dir = scratch("solvers");
    let lat = Lattice::single_node([4, 4, 4, 8]);
    write_bridge_text(&dir.join("conf_04040408.txt"), &random_gauge(&lat, 500)).unwrap();

    let correlator = |text: String| {
        let report = run(&Config::from_toml(&text).unwrap(), &dir).unwrap();
        assert!(
            report.solves[0].worst_residual_sqr < 1e-24,
            "{:?}",
            report.solves[0]
        );
        assert_eq!(report.correlators.len(), 1);
        report.correlators[0].values.clone()
    };
    let base = WILSON_TOML.replace("tolerance_squared = 1.0e-28", "tolerance_squared = 1.0e-26");
    let reference = correlator(base.clone());
    assert_eq!(reference.len(), 8);
    assert!(reference[0].re > reference[1].re && reference[1].re > 0.0);

    // FGMRES keeps its whole Krylov space, so it gets a small one and restarts.
    let fgmres = base
        .replace("\"cg_normal\"", "\"fgmres\"")
        .replace("max_iterations = 2000", "max_iterations = 30")
        .replace("max_restarts = 3 ", "max_restarts = 100 ");
    let multigrid = |precision: &str| {
        fgmres.replace("even_odd = true", "even_odd = false").replace(
            "max_restarts = 100 ",
            &format!("multigrid = {{ block = [2, 2, 2, 2], test_vectors = 8, precision = \"{precision}\" }}\nmax_restarts = 100 "),
        )
    };
    for variant in [
        base.replace("even_odd = true", "even_odd = false"),
        base.replace("\"cg_normal\"", "\"bicgstab\""),
        base.replace("\"cg_normal\"", "\"bicgstab\"")
            .replace("even_odd = true", "even_odd = false"),
        fgmres.clone(),
        fgmres.replace("even_odd = true", "even_odd = false"),
        multigrid("single"),
        multigrid("double"),
        // BiCGStab is flexible with a preconditioner.
        multigrid("single").replace("\"fgmres\"", "\"bicgstab\""),
        // The choices of the CUDA domain-wall multigrid.
        multigrid("single").replace(
            "precision = \"single\"",
            "precision = \"single\", smoother = \"gmres\", pre_smoothing = true, coarse_solver = \"gmres\", coarse_max_iterations = 5, coarse_tolerance_squared = 1.0e-2, precondition_coarse_diagonal = true",
        ),
    ] {
        for (a, b) in correlator(variant).iter().zip(&reference) {
            assert!(
                (a.re - b.re).abs() < 1e-10 * reference[0].re,
                "{a:?} vs {b:?}"
            );
        }
    }

    let file = dir.join("wilson_pion.toml");
    std::fs::write(&file, &base).unwrap();
    let mut printed = Vec::new();
    ruttice::app::run_file(&file, &mut printed).unwrap();
    let printed = String::from_utf8(printed).unwrap();
    assert!(
        printed.contains("# plaquette") && printed.contains("\n11 PS 0 "),
        "{printed}"
    );
    let written = std::fs::read_to_string(dir.join("wilson_pion.out")).unwrap();
    assert_eq!(
        written.lines().filter(|l| l.starts_with("11 PS ")).count(),
        8
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Compare every row of a reference table in validation/tests/data with the report.
fn compare_with_table(report: &ruttice::app::Report, table: &str) {
    let mut compared = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let pair = (
            f[0][0..1].parse::<usize>().unwrap(),
            f[0][1..2].parse::<usize>().unwrap(),
        );
        let t: usize = f[2].parse().unwrap();
        let (re, im): (f64, f64) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        let ours = report
            .correlators
            .iter()
            .find(|c| c.quarks == pair && c.channel == f[1])
            .unwrap_or_else(|| panic!("no correlator for {line}"))
            .values[t];
        assert!(
            (ours.re - re).abs() < 4e-11 && (ours.im - im).abs() < 4e-11,
            "{line}: ours {ours:?}"
        );
        compared += 1;
    }
    assert_eq!(compared, 312);
    assert!((report.plaquette - 0.625129946974942).abs() < 1e-13);
}

/// The YAML example, run on the Bridge++ test configuration, reproduces the reference
/// table produced with Bridge++ (validation/tests/data).
#[test]
fn clover_example_reproduces_the_reference_table() {
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let report = run(
        &Config::from_yaml(CLOVER_YAML).unwrap(),
        conf.parent().unwrap(),
    )
    .unwrap();
    assert_eq!(report.gauge_fixing_sweeps, None);
    compare_with_table(
        &report,
        include_str!("data/bridge_spectrum_clover_nogfix.txt"),
    );
}

/// The same with Coulomb gauge fixing at its default parameters: the unmodified Bridge++
/// test `Spectrum_Clover_Hadron2ptFunction`.
#[test]
fn clover_example_in_coulomb_gauge_reproduces_the_reference_table() {
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let text = CLOVER_YAML.replace(
        "  file: conf_04040408.txt\n",
        "  file: conf_04040408.txt\n  fixing:\n    type: coulomb\n",
    );
    let config = Config::from_yaml(&text).unwrap();
    assert!(config.gauge.fixing.is_some());
    let report = run(&config, conf.parent().unwrap()).unwrap();
    assert_eq!(report.gauge_fixing_sweeps, Some(120));
    compare_with_table(
        &report,
        include_str!("data/bridge_spectrum_clover_coulomb.txt"),
    );
}
