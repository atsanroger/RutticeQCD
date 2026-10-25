//! The parameter file and the run it drives.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::app::params::Params;
use ruttice::app::{
    run, BoundaryConfig, ChannelsConfig, CliError, Config, FermionConfig, SolverKind, SourceConfig,
};
use ruttice::gauge::random_gauge;
use ruttice::io::{read_binary, read_bridge_text, write_bridge_text};
use ruttice::Lattice;
use std::path::{Path, PathBuf};

const WILSON_YAML: &str = include_str!("../examples/wilson_pion.yaml");
const CLOVER_YAML: &str = include_str!("../examples/clover_two_flavours.yaml");
const CLOVER_MG_YAML: &str = include_str!("../examples/clover_mg.yaml");
const MEASUREMENTS_YAML: &str = include_str!("../examples/clover_measurements.yaml");

/// A fresh directory under the system temporary directory.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ruttice_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The run a parameter file describes.
fn config(text: &str) -> Config {
    Params::from_yaml(text).unwrap().to_config()
}

/// `text` with `old` replaced by `new`; `old` must be there, so that a variant of a file is
/// the variant it is meant to be.
fn with(text: &str, old: &str, new: &str) -> String {
    assert!(text.contains(old), "{old:?} is not in the file");
    text.replace(old, new)
}

#[test]
fn the_example_files_parse() {
    let wilson = config(WILSON_YAML);
    assert_eq!(wilson.lattice, [4, 4, 4, 8]);
    assert_eq!(wilson.process_grid, None);
    assert_eq!(wilson.solver.kind, SolverKind::CgNormal);
    assert!(wilson.solver.even_odd);
    assert_eq!(wilson.solver.simd, 1, "simd_lanes is for backend simd");
    assert_eq!(wilson.solver.max_restarts, 3);
    assert_eq!(wilson.quark.len(), 1);
    assert_eq!(
        wilson.quark[0].fermion,
        FermionConfig::Wilson {
            kappa: 0.12,
            boundary: [BoundaryConfig::Antiperiodic; 4]
        }
    );
    assert_eq!(
        wilson.measure.channels,
        ChannelsConfig::List(vec!["PS".to_string()])
    );
    assert_eq!(wilson.output.file, Some(PathBuf::from("wilson_pion.out")));

    let clover = config(CLOVER_YAML);
    assert_eq!(clover.quark.len(), 2);
    assert!(
        clover.solver.even_odd,
        "Even_odd when operator_type is absent"
    );
    assert_eq!(
        clover.measure.channels,
        ChannelsConfig::All,
        "all channels when absent"
    );
    assert_eq!(
        clover.quark[1].fermion,
        FermionConfig::Clover {
            kappa: 0.115,
            csw: 1.0,
            boundary: [
                BoundaryConfig::Periodic,
                BoundaryConfig::Periodic,
                BoundaryConfig::Periodic,
                BoundaryConfig::Antiperiodic
            ]
        }
    );
    assert_eq!(
        clover.quark[1].source,
        SourceConfig::Exponential {
            position: [0; 4],
            slope: 0.5,
            power: 2.0
        }
    );
    assert!(clover.gauge.fixing.is_none());
}

/// Gauge smearing is the block `GaugeSmearing`; Bridge++'s `Smear` and its companions are
/// refused, because "smear" alone does not say whether links or sources are smeared.
#[test]
fn gauge_smearing_has_its_own_name() {
    use ruttice::app::GaugeSmearingConfig;
    let c = config(&format!(
        "{WILSON_YAML}GaugeSmearing:\n  smear_type : Stout\n  rho_uniform : 0.1\n  number_of_smearing : 3\n"
    ));
    assert_eq!(
        c.gauge_smearing,
        Some(GaugeSmearingConfig::Stout {
            rho: 0.1,
            steps: 3,
            spatial_only: false
        })
    );
    let c = config(&format!(
        "{WILSON_YAML}GaugeSmearing:\n  smear_type : APE\n  rho_uniform : 0.5\n  number_of_smearing : 2\n  spatial_only : true\n"
    ));
    assert!(matches!(
        c.gauge_smearing,
        Some(GaugeSmearingConfig::Ape {
            spatial_only: true,
            max_iterations: 100,
            ..
        })
    ));
    for bad in [
        "Smear:\n  rho_uniform : 0.1\n",
        "Director_Smear:\n  number_of_smearing : 3\n",
        "Projection:\n  projection_type : Stout_SU3\n",
    ] {
        match Params::from_yaml(&format!("{WILSON_YAML}{bad}")) {
            Err(CliError::Invalid(e)) => {
                assert!(e.contains("GaugeSmearing") && e.contains("Source"), "{e}")
            }
            other => panic!("{bad}: {:?}", other.map(|_| ())),
        }
    }
}

/// A run on gauge-smeared links reports the plaquette before and after the smearing.
#[test]
fn a_run_on_smeared_links() {
    let dir = scratch("smeared");
    let lat = Lattice::single_node([4, 4, 4, 8]);
    write_bridge_text(&dir.join("conf_04040408.txt"), &random_gauge(&lat, 5)).unwrap();
    let mut c = config(&format!(
        "{WILSON_YAML}GaugeSmearing:\n  smear_type : Stout\n  rho_uniform : 0.1\n  number_of_smearing : 2\n"
    ));
    c.output.file = None;
    let r = run(&c, &dir).unwrap();
    let smeared = r.plaquette_smeared.unwrap();
    assert!(smeared > r.plaquette + 0.05, "{} → {smeared}", r.plaquette);
    let mut text = Vec::new();
    ruttice::app::write_report(&r, &c, &mut text).unwrap();
    assert!(String::from_utf8(text)
        .unwrap()
        .contains("after gauge smearing"));
}

/// The same parameters written in TOML and in YAML describe the same run, and keywords are
/// read in any case, with or without `_`.
#[test]
fn toml_and_yaml_are_equivalent() {
    let toml = r#"
[Main]
lattice_size = [4, 4, 4, 8]
[Gauge]
gauge_config_type_input = "text"
config_filename_input = "conf_04040408.txt"
[Fopr]
fermion_type = "WILSON"
hopping_parameter = 0.12
boundary_condition = [-1, -1, -1, -1]
[Solver]
solver_type = "cg"
operator_type = "evenodd"
convergence_criterion_squared = 1.0e-28
maximum_number_of_iteration = 2000
maximum_number_of_restart = 3
[Source]
source_type = "point"
source_position = [0, 0, 0, 0]
[Measure]
channels = ["PS"]
[Output]
file = "wilson_pion.out"
"#;
    assert_eq!(
        Params::from_toml(toml).unwrap().to_config(),
        config(WILSON_YAML)
    );
}

#[test]
fn mistakes_in_the_file_are_reported() {
    let message = |text: String| match Params::from_yaml(&text) {
        Err(CliError::Parse(m)) => m,
        other => panic!("expected a parse error, got {:?}", other.map(|_| ())),
    };
    // A misspelt key is an error, not silently ignored.
    let m = message(with(
        WILSON_YAML,
        "maximum_number_of_iteration ",
        "maximum_number_of_iterations",
    ));
    assert!(m.contains("maximum_number_of_iterations"), "{m}");
    // An unknown fermion type names the alternatives.
    let m = message(with(WILSON_YAML, ": Wilson\n", ": Wilsonn\n"));
    assert!(m.contains("Wilsonn") && m.contains("Clover"), "{m}");
    // A misspelt gauge-fixing parameter.
    let m = message(format!(
        "{WILSON_YAML}GaugeFixing:\n  gauge_fixing_type : Coulomb\n  overrelax : 1.6\n"
    ));
    assert!(m.contains("overrelax"), "{m}");
    // A key that belongs to another fermion type is read and said not to be used.
    let p = Params::from_yaml(&with(
        WILSON_YAML,
        "  hopping_parameter  : 0.12\n",
        "  hopping_parameter  : 0.12\n  clover_coefficient : 1.0\n",
    ))
    .unwrap();
    assert!(
        p.notes().iter().any(|n| n.contains("clover_coefficient")),
        "{:?}",
        p.notes()
    );
    // A file without a Main section is not a parameter file.
    let dir = scratch("no_main");
    let file = dir.join("old.yaml");
    std::fs::write(&file, "lattice: [4, 4, 4, 8]\n").unwrap();
    match ruttice::app::run_file(&file, &mut Vec::new()) {
        Err(CliError::Invalid(m)) => assert!(m.contains("Main"), "{m}"),
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn impossible_requests_are_rejected_before_any_work() {
    // Refused when the file is read, or when the run is set up, before the gauge
    // configuration is looked for.
    let invalid = |text: String| {
        let refused = Params::from_yaml(&text).and_then(|p| run(&p.to_config(), Path::new(".")));
        match refused {
            Err(CliError::Invalid(m)) => m,
            other => panic!(
                "expected an invalid-parameter error, got {:?}",
                other.map(|_| ())
            ),
        }
    };
    let m = invalid(with(WILSON_YAML, "channels : [PS]", "channels : [PION]"));
    assert!(m.contains("PION") && m.contains("A4_PS"), "{m}");
    let odd = |text: &str| {
        with(
            text,
            "lattice_size : [4, 4, 4, 8]",
            "lattice_size : [4, 3, 4, 8]",
        )
    };
    let m = invalid(odd(WILSON_YAML));
    assert!(m.contains("even"), "{m}");
    let m = invalid(with(
        WILSON_YAML,
        "source_position : [0, 0, 0, 0]",
        "source_position : [0, 0, 0, 8]",
    ));
    assert!(m.contains("outside"), "{m}");
    let m = invalid(format!(
        "{WILSON_YAML}GaugeFixing:\n  gauge_fixing_type : Coulomb\n  overrelaxation_parameter : 0.0\n"
    ));
    assert!(m.contains("overrelaxation"), "{m}");
    // Multigrid needs a flexible solver (FGMRES, BiCGStab) and blocks that divide the
    // lattice.
    let with_multigrid = |block: &str| {
        with(
            WILSON_YAML,
            "  maximum_number_of_restart     : 3\n",
            "  maximum_number_of_restart     : 3\n  preconditioner_type : MG\n",
        ) + &format!("MultiGrid:\n  MultiGrid_Level1:\n    block_size : {block}\n")
    };
    let m = invalid(with_multigrid("[2, 2, 2, 2]"));
    assert!(m.contains("FGMRES"), "{m}");
    let fgmres = |text: String| with(&text, ": CG  ", ": FGMRES");
    let whole = |text: String| with(&text, ": Even_odd", ": Lexical ");
    let m = invalid(whole(fgmres(with_multigrid("[3, 2, 2, 2]"))));
    assert!(m.contains("block"), "{m}");
    // Blocks of sites: a number of lanes there is an operator for, a lattice that can be
    // cut, parts with even extents for even-odd.
    let simd = |lanes: usize| {
        with(
            &with(WILSON_YAML, "backend    : cpu ", "backend    : simd"),
            "simd_lanes : 4",
            &format!("simd_lanes : {lanes}"),
        )
    };
    let m = invalid(simd(3));
    assert!(m.contains("Main.simd_lanes must be"), "{m}");
    let m = invalid(with(
        &simd(8),
        "lattice_size : [4, 4, 4, 8]",
        "lattice_size : [4, 4, 2, 8]",
    ));
    assert!(m.contains("which must all be even"), "{m}");
    let m = invalid(whole(with(
        &simd(4),
        "lattice_size : [4, 4, 4, 8]",
        "lattice_size : [3, 3, 3, 6]",
    )));
    assert!(m.contains("cannot be halved"), "{m}");
    // More quarks: in order, and of the kind this run is for.
    let m = invalid(format!(
        "{WILSON_YAML}Quark_3:\n  Fopr: {{fermion_type: Wilson, hopping_parameter: 0.1}}\n  Source: {{source_type: Local}}\n"
    ));
    assert!(m.contains("Quark_3 needs Quark_2"), "{m}");
    let m = invalid(format!(
        "{WILSON_YAML}Quark_2:\n  Fopr: {{fermion_type: Wilson, hopping_parameter: 0.1}}\n  Source: {{source_type: Exponential}}\n"
    ));
    assert!(m.contains("slope"), "{m}");
}

#[test]
fn a_missing_gauge_file_is_an_error_not_a_panic() {
    assert!(matches!(
        run(&config(WILSON_YAML), &scratch("missing")),
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
        let report = run(&config(&text), &dir).unwrap();
        assert!(
            report.solves[0].worst_residual_sqr < 1e-24,
            "{:?}",
            report.solves[0]
        );
        assert_eq!(report.correlators.len(), 1);
        report.correlators[0].values.clone()
    };
    let base = with(
        WILSON_YAML,
        "convergence_criterion_squared : 1.0e-28",
        "convergence_criterion_squared : 1.0e-26",
    );
    let reference = correlator(base.clone());
    assert_eq!(reference.len(), 8);
    assert!(reference[0].re > reference[1].re && reference[1].re > 0.0);

    let lexical = |text: &str| with(text, ": Even_odd", ": Lexical ");
    let bicgstab = |text: &str| with(text, ": CG  ", ": BiCGStab");
    let lanes = |text: &str, n: usize| {
        with(
            &with(text, "backend    : cpu ", "backend    : simd"),
            "simd_lanes : 4",
            &format!("simd_lanes : {n}"),
        )
    };
    // FGMRES keeps its whole Krylov space, so it gets a small one and restarts.
    let fgmres = with(
        &with(&base, ": CG  ", ": FGMRES"),
        "  maximum_number_of_iteration   : 2000",
        "  number_of_kept_directions     : 30\n  maximum_number_of_iteration   : 3000",
    );
    // The multigrid of the whole lattice (`lexical`), or of the even sites.
    let multigrid = |precision: &str, more: &str| {
        with(
            &fgmres,
            "  maximum_number_of_restart     : 3\n",
            "  maximum_number_of_restart     : 3\n  preconditioner_type : MG\n",
        ) + &format!(
            "MultiGrid:\n  coarse_precision : {precision}\n  MultiGrid_Level1:\n    block_size : [2, 2, 2, 2]\n    number_of_test_vectors : 8\n{more}"
        )
    };
    for variant in [
        lexical(&base),
        bicgstab(&base),
        lexical(&bicgstab(&base)),
        fgmres.clone(),
        lexical(&fgmres),
        lexical(&multigrid("single", "")),
        lexical(&multigrid("double", "")),
        // BiCGStab is flexible with a preconditioner.
        with(&lexical(&multigrid("single", "")), ": FGMRES", ": BiCGStab"),
        // The multigrid under a solver on the even sites (the coarse system on the even
        // coarse sites too), and with the fields in blocks.
        multigrid("single", ""),
        lanes(&lexical(&multigrid("double", "")), 4),
        lanes(&multigrid("single", ""), 2),
        // Fields in blocks of sites.
        lanes(&base, 2),
        lanes(&base, 8),
        lanes(&lexical(&base), 4),
        lanes(&fgmres, 4),
        lanes(&bicgstab(&base), 4),
        // The choices of the CUDA domain-wall multigrid: GMRES smoothing before and after,
        // GMRES(5) on the coarse lattice, the coarse operator preconditioned by its
        // diagonal blocks.
        lexical(&multigrid(
            "single",
            "    precondition_diagonal : true\n    Coarse_Solver_D:\n      solver_type : GMRES\n      maximum_number_of_iteration : 5\n      convergence_criterion_squared : 1.0e-2\n  Smoother:\n    smoother_type : GMRES\n    pre_smoothing : true\n",
        )),
    ] {
        for (a, b) in correlator(variant).iter().zip(&reference) {
            assert!(
                (a.re - b.re).abs() < 1e-10 * reference[0].re,
                "{a:?} vs {b:?}"
            );
        }
    }

    let file = dir.join("wilson_pion.yaml");
    std::fs::write(&file, &base).unwrap();
    let mut printed = Vec::new();
    ruttice::app::run_file(&file, &mut printed).unwrap();
    let printed = String::from_utf8(printed).unwrap();
    assert!(
        printed.contains("# Wilson: κ = 0.12")
            && printed.contains("# plaquette")
            && printed.contains("\n11 PS 0 "),
        "{printed}"
    );
    let written = std::fs::read_to_string(dir.join("wilson_pion.out")).unwrap();
    assert_eq!(
        written.lines().filter(|l| l.starts_with("11 PS ")).count(),
        8
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Compare every row of a reference table in tests/data with the report.
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

/// Bridge++'s binary file of its test configuration reads as its text file does.
#[test]
fn the_binary_file_of_bridge_is_its_text_file() {
    let (Some(text), Some(binary)) = (
        bridge_test_file("conf_04040408.txt"),
        bridge_test_file("conf_binary.dat"),
    ) else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_binary.dat");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let from_text = read_bridge_text(&text, &lat).unwrap();
    let from_binary = read_binary(&binary, &lat).unwrap();
    let worst = from_text
        .as_slice()
        .iter()
        .zip(from_binary.as_slice())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-14, "{worst:e}");
}

/// The two-flavour example, run on the Bridge++ test configuration, reproduces the
/// reference table produced with Bridge++ (tests/data).
#[test]
fn clover_example_reproduces_the_reference_table() {
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let report = run(&config(CLOVER_YAML), conf.parent().unwrap()).unwrap();
    assert_eq!(report.gauge_fixing_sweeps, None);
    compare_with_table(
        &report,
        include_str!("data/bridge_spectrum_clover_nogfix.txt"),
    );
}

/// The same run with the fields in blocks of eight sites.
#[test]
fn clover_example_in_blocks_reproduces_the_reference_table() {
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let text = with(
        CLOVER_YAML,
        "  lattice_size : [4, 4, 4, 8]\n",
        "  lattice_size : [4, 4, 4, 8]\n  backend : simd\n  simd_lanes : 8\n",
    );
    let config = config(&text);
    assert_eq!(config.solver.simd, 8);
    let report = run(&config, conf.parent().unwrap()).unwrap();
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
    let config = config(&format!(
        "{CLOVER_YAML}GaugeFixing:\n  gauge_fixing_type : Coulomb\n"
    ));
    assert!(config.gauge.fixing.is_some());
    let report = run(&config, conf.parent().unwrap()).unwrap();
    assert_eq!(report.gauge_fixing_sweeps, Some(120));
    compare_with_table(
        &report,
        include_str!("data/bridge_spectrum_clover_coulomb.txt"),
    );
}

/// The multigrid example for clover fermions (FGMRES on the even sites with the multigrid)
/// runs through `run_file`, says what it runs in comment lines, and gives the pion correlator
/// of CG on the same operator.
#[test]
fn the_multigrid_example_runs_clover_fermions() {
    let dir = scratch("bridge_style");
    let lat = Lattice::single_node([4, 4, 4, 8]);
    write_bridge_text(&dir.join("conf_04040408.txt"), &random_gauge(&lat, 600)).unwrap();
    let file = dir.join("clover_mg.yaml");
    std::fs::write(&file, CLOVER_MG_YAML).unwrap();
    let mut printed = Vec::new();
    ruttice::app::run_file(&file, &mut printed).unwrap();
    let printed = String::from_utf8(printed).unwrap();
    assert!(
        printed.contains("# Clover: κ = 0.12, c_SW = 1"),
        "{printed}"
    );
    assert!(
        printed.contains("# preconditioned by the multigrid of the whole lattice: [2, 2, 2, 2] blocks, 8 test vectors, 4 setup iterations"),
        "{printed}"
    );
    let pion: Vec<f64> = printed
        .lines()
        .filter(|l| l.starts_with("11 PS "))
        .map(|l| l.split_whitespace().nth(3).unwrap().parse().unwrap())
        .collect();
    assert_eq!(pion.len(), 8);

    let reference = "
Main: {lattice_size: [4, 4, 4, 8]}
Gauge: {config_filename_input: conf_04040408.txt}
Fopr: {fermion_type: Clover, hopping_parameter: 0.12, clover_coefficient: 1.0, boundary_condition: [1, 1, 1, -1]}
Solver: {solver_type: CG, convergence_criterion_squared: 1.0e-28, maximum_number_of_iteration: 2000, maximum_number_of_restart: 3}
Source: {source_type: Local, source_position: [0, 0, 0, 0]}
Measure: {channels: [PS]}
";
    let report = run(&config(reference), &dir).unwrap();
    for (a, b) in pion.iter().zip(&report.correlators[0].values) {
        assert!(
            (a - b.re).abs() < 1e-10 * b.re.abs().max(1e-30),
            "{a} vs {}",
            b.re
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The HMC example file: one trajectory of Bridge++'s test_HMC_Domainwall_Nf2_PV, through the
/// program; also its file is recognised as an HMC file.
#[test]
fn the_hmc_example_repeats_bridge() {
    use ruttice::app::hmc::{is_hmc_file, run_on, HmcConfig};
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let dir = scratch("hmc");
    std::fs::copy(&conf, dir.join("conf_04040408.txt")).unwrap();
    let file = dir.join("hmc.yaml");
    std::fs::write(&file, include_str!("../examples/hmc_domainwall_pv.yaml")).unwrap();
    assert!(is_hmc_file(&file).unwrap());
    let config = HmcConfig::load(&file).unwrap();
    let mut out = Vec::new();
    let report = run_on(
        &config,
        &dir,
        &mut out,
        std::sync::Arc::new(ruttice::SingleComm),
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    eprintln!("{text}");
    let (n, _, _, plaq) = report.trajectories[0];
    assert_eq!(n, 200001);
    assert!(
        ((plaq - 0.62754640357188) / 0.62754640357188).abs() < 1e-11,
        "{plaq}"
    );
    // A section the file does not have is refused.
    let bad =
        include_str!("../examples/hmc_domainwall_pv.yaml").replace("GaugeSmearing:", "Smearing:");
    assert!(HmcConfig::from_yaml(&bad).is_err());
}

/// The HMC runner saves configurations on several processes (threads standing in for them):
/// the files of a run on [1, 1, 2, 2] are the bytes of those of a run on one process, in the
/// text format and in ILDG (whose date of writing is left out of the comparison).
#[test]
fn hmc_saves_configurations_on_several_processes() {
    use ruttice::app::hmc::{run_on, HmcConfig};
    use ruttice::LocalComm;
    let without_date = |path: &Path| {
        let mut bytes = std::fs::read(path).unwrap();
        let find = |b: &[u8], what: &[u8]| b.windows(what.len()).position(|w| w == what);
        if let Some(open) = find(&bytes, b"<date>") {
            let close = open + find(&bytes[open..], b"</date>").unwrap();
            bytes[open..close].fill(b'-');
        }
        bytes
    };
    for (format, name) in [("Text", "conf_{n}.txt"), ("ILDG", "conf_{n}.lime")] {
        let yaml = |grid: &str| {
            format!(
                "Main:
  lattice_size : [4, 4, 4, 8]
  grid_size : {grid}
HMC:
  gauge_config_status : Cold_start
  trajectory_number : 100
  number_of_trajectories : 2
  save_config_interval : 1
  config_filename_output : \"{name}\"
  gauge_config_type_output : {format}
HMC_General:
  trajectory_length : 0.5
  Metropolis_test : false
Builder_Integrator:
  integrator : Omelyan
  number_of_steps : [5]
Action_G:
  action_type : Action_G_Plaq
  beta : 5.5
"
            )
        };
        let one = scratch(&format!("hmc_save_{format}_1"));
        let config = HmcConfig::from_yaml(&yaml("[1, 1, 1, 1]")).unwrap();
        let alone = run_on(
            &config,
            &one,
            &mut Vec::new(),
            std::sync::Arc::new(ruttice::SingleComm),
        )
        .unwrap();
        // Without the Metropolis test every trajectory is kept: the links move away from the
        // cold start, so the files hold more than unit matrices.
        assert!(
            alone.trajectories.iter().all(|t| t.3 < 0.999),
            "{:?}",
            alone.trajectories
        );

        let four = scratch(&format!("hmc_save_{format}_4"));
        let config = HmcConfig::from_yaml(&yaml("[1, 1, 2, 2]")).unwrap();
        let split = LocalComm::run([4, 4, 4, 8], [1, 1, 2, 2], |lat| {
            run_on(&config, &four, &mut Vec::new(), lat.comm.clone()).unwrap()
        });
        for report in &split {
            for (a, b) in report.trajectories.iter().zip(&alone.trajectories) {
                assert_eq!((a.0, a.2), (b.0, b.2), "the same trajectories accepted");
            }
        }
        for n in [101, 102] {
            let file = name.replace("{n}", &format!("{n:06}"));
            assert!(
                without_date(&four.join(&file)) == without_date(&one.join(&file)),
                "{file}: written on [1, 1, 2, 2] it differs from the one of one process"
            );
        }
        let _ = std::fs::remove_dir_all(one);
        let _ = std::fs::remove_dir_all(four);
    }
}

/// The Wilson flow along an HMC run: a line after every `measurement_interval` trajectories
/// with what `flow_to_t0` finds on that configuration, the same on several processes, and the
/// Γ-method for the series at the end.
#[test]
fn hmc_measures_the_flow_along_the_run() {
    use ruttice::app::hmc::{run_on, HmcConfig};
    use ruttice::LocalComm;
    let yaml = |grid: &str| {
        format!(
            "Main:
  lattice_size : [4, 4, 4, 8]
  grid_size : {grid}
HMC:
  gauge_config_status : Hot_start
  seed_for_random_number : 5
  number_of_trajectories : 6
HMC_General:
  trajectory_length : 0.5
  Metropolis_test : false
Builder_Integrator:
  integrator : Omelyan
  number_of_steps : [5]
Action_G:
  action_type : Action_G_Plaq
  beta : 5.5
GradientFlow:
  measurement_interval : 2
  step_size : 0.02
  max_flow_time : 3.0
"
        )
    };
    let flow_lines = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|l| {
                l.starts_with("# flow") || l.starts_with("# t0/a^2") || l.starts_with("# Q_")
            })
            .map(str::to_string)
            .collect()
    };
    let config = HmcConfig::from_yaml(&yaml("[1, 1, 1, 1]")).unwrap();
    let mut one = Vec::new();
    run_on(
        &config,
        Path::new("."),
        &mut one,
        std::sync::Arc::new(ruttice::SingleComm),
    )
    .unwrap();
    let one = flow_lines(&String::from_utf8(one).unwrap());
    assert_eq!(
        one.iter().filter(|l| l.starts_with("# flow")).count(),
        3,
        "{one:?}"
    );
    assert!(one.iter().any(|l| l.starts_with("# t0/a^2")), "{one:?}");

    let config = HmcConfig::from_yaml(&yaml("[1, 1, 2, 2]")).unwrap();
    let split = LocalComm::run([4, 4, 4, 8], [1, 1, 2, 2], |lat| {
        let mut out = Vec::new();
        run_on(&config, Path::new("."), &mut out, lat.comm.clone()).unwrap();
        String::from_utf8(out).unwrap()
    });
    // Rank 0 prints; the numbers to the digits printed are those of one process.
    assert_eq!(flow_lines(&split[0]), one);

    // Without the first two trajectories: the measurements after 4 and 6 only.
    let skipping = yaml("[1, 1, 1, 1]").replace(
        "max_flow_time : 3.0",
        "max_flow_time : 3.0\n  skip_trajectories : 2",
    );
    let config = HmcConfig::from_yaml(&skipping).unwrap();
    let mut out = Vec::new();
    run_on(
        &config,
        Path::new("."),
        &mut out,
        std::sync::Arc::new(ruttice::SingleComm),
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(
        text.contains("# without the first 2 trajectories:"),
        "{text}"
    );
    assert!(
        text.lines()
            .filter(|l| l.starts_with("# t0/a^2"))
            .all(|l| l.ends_with("; 2 measurements)")),
        "{text}"
    );

    let bad = yaml("[1, 1, 1, 1]").replace("step_size : 0.02", "order_of_RungeKutta : 4");
    assert!(HmcConfig::from_yaml(&bad)
        .map(|c| c.flow.is_some())
        .is_err());
}

/// HMC with the multigrid in the force solves on several processes: the trajectory of one
/// process, to the tolerance of the solves; blocks that a process cuts are refused.
#[test]
fn hmc_with_the_multigrid_on_several_processes() {
    use ruttice::app::hmc::{run_on, HmcConfig};
    use ruttice::LocalComm;
    let yaml = |grid: &str, block: &str| {
        format!(
            "Main:
  lattice_size : [4, 4, 4, 8]
  grid_size : {grid}
HMC:
  gauge_config_status : Hot_start
  seed_for_random_number : 7
  trajectory_number : 100
  number_of_trajectories : 1
HMC_General:
  trajectory_length : 0.05
Builder_Integrator:
  integrator : Leapfrog
  number_of_steps : [2]
Action_G:
  action_type : Action_G_Plaq
  beta : 5.5
Action_F_1:
  action_type : Standard
  Fopr:
    fermion_type : Clover
    hopping_parameter : 0.11
    clover_coefficient : 1.0
    boundary_condition : [1, 1, 1, -1]
  Solver:
    solver_type : BiCGStab
    convergence_criterion_squared : 1.0e-24
    preconditioner_type : MG
  MultiGrid:
    Setup:
      bootstrap_number_of_rounds : 2
    MultiGrid_Level1:
      block_size : {block}
      number_of_test_vectors : 8
"
        )
    };
    let dir = std::env::temp_dir();
    let config = HmcConfig::from_yaml(&yaml("[1, 1, 1, 1]", "[2, 2, 2, 2]")).unwrap();
    let alone = run_on(
        &config,
        &dir,
        &mut Vec::new(),
        std::sync::Arc::new(ruttice::SingleComm),
    )
    .unwrap();
    let config = HmcConfig::from_yaml(&yaml("[1, 1, 2, 2]", "[2, 2, 2, 2]")).unwrap();
    let split = LocalComm::run([4, 4, 4, 8], [1, 1, 2, 2], |lat| {
        run_on(&config, &dir, &mut Vec::new(), lat.comm.clone()).unwrap()
    });
    for report in &split {
        let (a, b) = (report.trajectories[0], alone.trajectories[0]);
        assert_eq!((a.0, a.2), (b.0, b.2));
        assert!((a.1 - b.1).abs() < 1e-7, "ΔH {} against {}", a.1, b.1);
        assert!(
            (a.3 - b.3).abs() < 1e-10,
            "plaquette {} against {}",
            a.3,
            b.3
        );
    }
    // 4 sites in z on each process: blocks of 4 in t fit (8 / 2), blocks of 4 in z do not.
    let config = HmcConfig::from_yaml(&yaml("[1, 1, 2, 2]", "[2, 2, 4, 4]")).unwrap();
    let refused = LocalComm::run([4, 4, 4, 8], [1, 1, 2, 2], |lat| {
        run_on(&config, &dir, &mut Vec::new(), lat.comm.clone()).is_err()
    });
    assert!(refused.iter().all(|r| *r));
}

/// The RHMC example file: one trajectory of Bridge++'s test_HMC_Clover_RHMC_Nf2p1 (2 + 1
/// flavours, the strange quark by a rational approximation computed by the program), to
/// Bridge++'s relative 1e-11.
#[test]
fn the_rhmc_example_repeats_bridge() {
    use ruttice::app::hmc::{run_on, HmcConfig};
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let dir = scratch("rhmc");
    std::fs::copy(&conf, dir.join("conf_04040408.txt")).unwrap();
    let text = include_str!("../examples/hmc_clover_rhmc_nf2p1.yaml");
    let config = HmcConfig::from_yaml(text).unwrap();
    let mut out = Vec::new();
    let report = run_on(
        &config,
        &dir,
        &mut out,
        std::sync::Arc::new(ruttice::SingleComm),
    )
    .unwrap();
    eprintln!("{}", String::from_utf8(out).unwrap());
    let (_, _, _, plaq) = report.trajectories[0];
    let expected = 0.627668186553629;
    assert!(((plaq - expected) / expected).abs() < 1e-11, "{plaq}");
    // Powers outside (0, 1), empty intervals and domain-wall fermions are refused.
    for bad in [
        text.replace("exponent_numerator   : 1", "exponent_numerator   : 2"),
        text.replace(
            "lower_bound          : 1.0e-2",
            "lower_bound          : 5.0",
        ),
        text.replace(
            "    hopping_parameter  : 0.12\n    clover_coefficient : 1.0\n",
            "    extent_of_5th_dimension : 4\n    domain_wall_height : 1.6\n    quark_mass : 0.1\n",
        )
        .replace(
            "    fermion_type       : Clover\n    extent_of_5th_dimension",
            "    fermion_type       : Domainwall\n    extent_of_5th_dimension",
        ),
    ] {
        assert!(matches!(
            HmcConfig::from_yaml(&bad),
            Err(CliError::Invalid(_))
        ));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `rational` action (one flavour of Wilson fermions from a cold start, no Bridge++ input):
/// the trajectory conserves H, and the coefficients read from Bridge++-format files in
/// `coefficients` give the run of the computed ones.
#[test]
fn rational_action_from_a_parameter_file() {
    use ruttice::app::hmc::{run_on, HmcConfig};
    use ruttice::numerics::{RationalApprox, RationalSpec};
    let dir = scratch("rhmc_wilson");
    let text = "
Main:
  lattice_size : [4, 4, 4, 4]
HMC:
  gauge_config_status : Hot_start
  seed_for_random_number : 3
  number_of_trajectories : 1
HMC_General:
  trajectory_length : 0.2
Builder_Integrator:
  integrator : Omelyan
  number_of_steps : [3, 2]
Action_G:
  action_type : Action_G_Plaq
  beta : 5.5
  level_of_action : 1
Action_F_1:
  action_type : Rational
  level_of_action : 0
  Fopr:
    fermion_type : Wilson
    hopping_parameter : 0.11
    boundary_condition : [1, 1, 1, -1]
  Rational:
    number_of_poles : 10
    lower_bound : 0.05
    upper_bound : 2.5
";
    let run = |text: &str| {
        let config = HmcConfig::from_yaml(text).unwrap();
        let mut out = Vec::new();
        let report = run_on(
            &config,
            &dir,
            &mut out,
            std::sync::Arc::new(ruttice::SingleComm),
        )
        .unwrap();
        eprintln!("{}", String::from_utf8(out).unwrap());
        report.trajectories[0]
    };
    let (_, dh, _, plaq) = run(text);
    assert!(dh.abs() < 0.05, "dH {dh}");
    // D†D in [0.05², 2.5²]: x^(−1/2) and x^(1/4), written as Bridge++ writes them.
    for (p, q) in [(-1, 2), (1, 4)] {
        let spec = RationalSpec {
            poles: 10,
            p,
            q,
            x_min: 0.05 * 0.05,
            x_max: 2.5 * 2.5,
        };
        let r = RationalApprox::minimax(&spec).unwrap();
        std::fs::write(dir.join(spec.bridge_file_name()), r.to_bridge(&spec)).unwrap();
    }
    let (_, dh_file, _, plaq_file) = run(&text.replace(
        "number_of_poles : 10",
        "number_of_poles : 10\n    coefficients : .",
    ));
    assert!((plaq_file - plaq).abs() < 1e-13 && (dh_file - dh).abs() < 1e-9);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `even_odd` and `initial_guess` of a two-flavour action in the HMC file: one trajectory of
/// even-odd clover fermions through the program is the trajectory of `TwoFlavorEo` built by
/// hand, number by number; with the guess the same to the precision of the solves; and a
/// guess on a ratio action is refused.
#[test]
fn hmc_file_even_odd_and_initial_guess() {
    use ruttice::app::hmc::{run_on, HmcConfig};
    use ruttice::fermion::{Clover, CloverParams};
    use ruttice::hmc::{Hmc, Mseries, RectangleAction, Scheme, Term, TwoFlavorEo};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    let yaml = |guess: usize| {
        format!(
            "Main:
  lattice_size : [4, 4, 4, 4]
HMC:
  gauge_config_status : Hot_start
  seed_for_random_number : 7
  trajectory_number : 5
  number_of_trajectories : 1
HMC_General:
  trajectory_length : 0.2
Builder_Integrator:
  integrator : Leapfrog
  number_of_steps : [4]
GaugeSmearing:
  smear_type : Stout
  rho_uniform : 0.1
  number_of_smearing : 1
Action_G:
  action_type : Action_G_Rectangle
  beta : 2.3
  c_plaq : 3.648
  c_rect : -0.331
Action_F_1:
  action_type : Standard
  operator_type : Even_odd
  Fopr:
    fermion_type : Clover
    hopping_parameter : 0.12
    clover_coefficient : 1.5
    boundary_condition : [1, 1, 1, -1]
  Solver:
    convergence_criterion_squared : 1.0e-26
    number_of_initial_guess : {guess}
"
        )
    };
    let dir = scratch("hmc_eo");
    let run = |text: &str| {
        let config = HmcConfig::from_yaml(text).unwrap();
        let mut out = Vec::new();
        let report = run_on(
            &config,
            &dir,
            &mut out,
            std::sync::Arc::new(ruttice::SingleComm),
        )
        .unwrap();
        eprintln!("{}", String::from_utf8(out).unwrap());
        report.trajectories[0]
    };
    let (_, dh, _, plaq) = run(&yaml(0));
    let (_, dh_g, _, plaq_g) = run(&yaml(3));

    let lat = Lattice::single_node([4; 4]);
    let mut u = random_gauge(&lat, 7);
    let params = CloverParams {
        kappa: 0.12,
        csw: 1.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let stop = Stop {
        tolerance_sqr: 1e-26,
        max_iterations: 10000,
        max_restarts: 10,
    };
    let smearing = ruttice::gauge::GaugeSmearing {
        smear: ruttice::gauge::Smear::Stout(ruttice::gauge::smearing::Stout::uniform(0.1)),
        steps: 1,
    };
    let fermions = TwoFlavorEo::new(Clover::<f64>::new(&lat, params), Some(smearing), stop, stop)
        .with_clover(0.12 * 1.5);
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut hmc = Hmc::new(
        vec![
            Term {
                level: 0,
                action: Box::new(gauge),
            },
            Term {
                level: 0,
                action: Box::new(fermions),
            },
        ],
        Scheme::Leapfrog,
        vec![4],
        8,
        0.2,
    );
    let t = hmc.trajectory(&mut u, &mut Mseries::new(5));
    assert_eq!(plaq, t.plaquette);
    assert_eq!(dh, t.delta_h);
    assert!(
        (plaq_g - plaq).abs() < 1e-11 && (dh_g - dh).abs() < 1e-7,
        "{plaq_g} {dh_g}"
    );

    let bad = "Main:
  lattice_size : [4, 4, 4, 4]
HMC:
  gauge_config_status : Cold_start
  number_of_trajectories : 1
HMC_General:
  trajectory_length : 0.1
Builder_Integrator:
  integrator : Leapfrog
  number_of_steps : [2]
Action_F_1:
  action_type : Ratio
  Fopr:
    fermion_type : Domainwall
    extent_of_5th_dimension : 4
    domain_wall_height : 1.6
    quark_mass : 0.1
    boundary_condition : [1, 1, 1, -1]
  Solver:
    number_of_initial_guess : 2
";
    assert!(matches!(
        HmcConfig::from_yaml(bad),
        Err(CliError::Invalid(_))
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The solver keys of an HMC file: which solvers and preconditioners go with which action.
#[test]
fn hmc_solver_keys_are_checked() {
    use ruttice::app::hmc::HmcConfig;
    // `action`: the lines of the action after its type; `solver`: the lines of its Solver.
    let file = |fopr: &str, action: &str, solver: &[&str]| {
        let mut text = format!(
            "Main:
  lattice_size : [4, 4, 4, 8]
HMC:
  gauge_config_status : Cold_start
  number_of_trajectories : 1
HMC_General:
  trajectory_length : 0.1
Builder_Integrator:
  integrator : Leapfrog
  number_of_steps : [2]
Action_F_1:
{action}  Fopr:
{fopr}    boundary_condition : [1, 1, 1, -1]
"
        );
        // The lines from `MultiGrid` on are a section of the action, beside its Solver.
        let split = solver
            .iter()
            .position(|l| l.starts_with("MultiGrid"))
            .unwrap_or(solver.len());
        if split > 0 {
            text += "  Solver:\n";
            for line in &solver[..split] {
                text += &format!("    {line}\n");
            }
        }
        for line in &solver[split..] {
            text += &format!("  {line}\n");
        }
        text
    };
    let clover =
        "    fermion_type : Clover\n    hopping_parameter : 0.12\n    clover_coefficient : 1.0\n";
    let wall = "    fermion_type : Domainwall\n    extent_of_5th_dimension : 4\n    domain_wall_height : 1.6\n    quark_mass : 0.1\n";
    let standard = "  action_type : Standard\n";
    let even_odd = "  action_type : Standard\n  operator_type : Even_odd\n";
    let rational = "  action_type : Rational\n  Rational:\n    number_of_poles : 8\n    lower_bound : 0.01\n    upper_bound : 3.2\n";
    let mg = [
        "preconditioner_type : MG",
        "MultiGrid:",
        "  MultiGrid_Level1:",
        "    block_size : [2, 2, 2, 2]",
    ];
    let with = |first: &'static str, more: &[&'static str]| -> Vec<&'static str> {
        [&[first][..], more].concat()
    };
    let ok = |fopr: &str, action: &str, solver: &[&str]| {
        let parsed = HmcConfig::from_yaml(&file(fopr, action, solver));
        if let Err(e) = &parsed {
            eprintln!("{solver:?}: {e}");
        }
        parsed.is_ok()
    };
    for solver in ["CG", "BiCGStab", "FGMRES", "GCR"] {
        assert!(ok(clover, standard, &[&format!("solver_type : {solver}")]));
    }
    assert!(ok(clover, standard, &[]));
    assert!(ok(clover, standard, &["preconditioner_type : Chebyshev"]));
    assert!(ok(clover, standard, &with("solver_type : BiCGStab", &mg)));
    assert!(ok(clover, even_odd, &with("solver_type : GCR", &mg)));
    assert!(ok(
        &format!("{wall}    neff_alpha : 0.6\n"),
        standard,
        &["preconditioner_type : Chebyshev"]
    ));
    assert!(ok(clover, rational, &[]));
    assert!(ok(clover, rational, &["solver_type : CG"]));
    assert!(ok(clover, rational, &["solver_type : BiCGStab"]));
    assert!(ok(clover, rational, &with("solver_type : GCR", &mg)));
    // What does not go together.
    assert!(!ok(clover, standard, &["solver_type : MINRES"]));
    assert!(!ok(clover, standard, &with("solver_type : CG", &mg)));
    assert!(!ok(
        clover,
        standard,
        &["solver_type : BiCGStab", "preconditioner_type : MG"]
    ));
    assert!(!ok(
        clover,
        standard,
        &["solver_type : BiCGStab", "preconditioner_type : Chebyshev"]
    ));
    assert!(!ok(
        &format!("{clover}    neff_alpha : 0.6\n"),
        standard,
        &[]
    ));
    assert!(!ok(
        clover,
        standard,
        &[
            "solver_type : BiCGStab",
            "preconditioner_type : MG",
            "MultiGrid:",
            "  MultiGrid_Level1:",
            "    block_size : [3, 2, 2, 2]"
        ]
    ));
    // The domain-wall multigrid: the Ratio action over Pauli-Villars fields, GCR, PV_right.
    let ratio = "  action_type : Ratio\n";
    assert!(ok(wall, ratio, &with("solver_type : GCR", &mg)));
    assert!(ok(
        wall,
        ratio,
        &with(
            "solver_type : GCR",
            &[&["operator_type : PV_right"][..], &mg].concat()
        )
    ));
    assert!(!ok(
        wall,
        ratio,
        &with(
            "solver_type : GCR",
            &[&["operator_type : PV_left"][..], &mg].concat()
        )
    ));
    assert!(ok(
        wall,
        ratio,
        &with(
            "solver_type : GCR",
            &[&mg[..], &["  coarse_precision : single"]].concat()
        )
    ));
    // The multigrid of HMC takes the keys of the propagator files on the CPU as on the device.
    for keys in [
        &["  fine_precision : single"][..],
        &["  basis_precision : single"],
        &["  MultiGrid_Level2:", "    super_block_size : 2"],
    ] {
        assert!(ok(
            wall,
            ratio,
            &with("solver_type : GCR", &[&mg[..], keys].concat())
        ));
    }
    assert!(ok(
        &format!("{wall}    neff_alpha : 0.6\n"),
        ratio,
        &with("solver_type : GCR", &mg)
    ));
    assert!(ok(
        wall,
        ratio,
        &with(
            "solver_type : GCR",
            &[&mg[..], &["  coarse_precision : FP64"]].concat()
        )
    ));
    assert!(!ok(wall, even_odd, &with("solver_type : GCR", &mg)));
    // The even-odd Ratio of domain-wall fermions takes the multigrid too; of clover fermions not.
    let ratio_eo = "  action_type : Ratio\n  operator_type : Even_odd\n";
    assert!(ok(wall, ratio_eo, &with("solver_type : GCR", &mg)));
    assert!(!ok(clover, ratio_eo, &with("solver_type : GCR", &mg)));
    // The old place of the multigrid, in the Solver, is refused with where it went.
    let old = file(
        clover,
        standard,
        &["solver_type : BiCGStab", "preconditioner_type : MG"],
    ) + "    MultiGrid:\n      block_size : [2, 2, 2, 2]\n";
    let refused = HmcConfig::from_yaml(&old).unwrap_err().to_string();
    assert!(refused.contains("MultiGrid_HMC"), "{refused}");
    assert!(ok(wall, standard, &["operator_type : Lexical"]));
    // Mixed precision: domain-wall fermions, CG, no preconditioner, on the CPU.
    assert!(ok(wall, ratio, &["precision : mixed"]));
    assert!(ok(
        wall,
        ratio_eo,
        &[
            "precision : Mixed",
            "convergence_criterion_squared_inner : 1.0e-9"
        ]
    ));
    assert!(ok(wall, standard, &["precision : mixed"]));
    assert!(ok(wall, ratio, &["precision : FP64"]));
    assert!(!ok(wall, even_odd, &["precision : mixed"]));
    assert!(!ok(clover, standard, &["precision : mixed"]));
    assert!(!ok(
        wall,
        ratio,
        &["precision : mixed", "preconditioner_type : Chebyshev"]
    ));
    assert!(!ok(
        wall,
        ratio,
        &with(
            "solver_type : GCR",
            &[&["precision : mixed"][..], &mg].concat()
        )
    ));
    assert!(!ok(
        wall,
        ratio,
        &["convergence_criterion_squared_inner : 1.0e-9"]
    ));
    assert!(!ok(wall, standard, &["operator_type : Normal_left"]));
    assert!(!ok(wall, standard, &["solver_type : BiCGStab"]));
    assert!(!ok(wall, standard, &with("solver_type : BiCGStab", &mg)));
    assert!(!ok(clover, rational, &with("solver_type : CG", &mg)));
    assert!(!ok(clover, rational, &["preconditioner_type : Chebyshev"]));
    // The sections of an action must suit its type.
    assert!(!ok(clover, "  action_type : Rational\n", &[]));
    assert!(!ok(
        clover,
        "  action_type : Ratio\n  operator_type : Even_odd\n",
        &[]
    ));
}

/// The measurements of the `Measure` section through the program
/// (examples/clover_measurements.yaml) on the Bridge++ test configuration: the mesons at momenta and the proton against
/// Bridge++'s numbers, and the charge of the conserved current in three-point functions (a
/// meson sink with the current on u, a proton sink with the current on d).
#[test]
fn measurements_of_the_measure_section() {
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let base = conf.parent().unwrap();
    let report = run(&config(MEASUREMENTS_YAML), base).unwrap();
    let find = |report: &ruttice::app::Report, quarks: (usize, usize), name: &str| {
        report
            .correlators
            .iter()
            .find(|c| c.quarks == quarks && c.channel == name)
            .unwrap_or_else(|| panic!("no {name} for {quarks:?}"))
            .values
            .clone()
    };
    let pair = |s: &str| (s[0..1].parse().unwrap(), s[1..2].parse().unwrap());
    let rows = |table: &str| -> Vec<Vec<String>> {
        table
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| l.split_whitespace().map(String::from).collect())
            .collect()
    };
    let close = |ours: ruttice::C64, re: &str, im: &str, what: &str| {
        let (re, im): (f64, f64) = (re.parse().unwrap(), im.parse().unwrap());
        assert!(
            (ours.re - re).abs() < 4e-11 && (ours.im - im).abs() < 4e-11,
            "{what}: ours {ours:?}, Bridge++ {re} {im}"
        );
    };
    let mut compared = 0;
    for f in rows(include_str!("data/bridge_momentum_clover.txt")) {
        let n = match f[2].as_str() {
            "100" => "[1,0,0]",
            "011" => "[0,1,1]",
            _ => continue,
        };
        let t: usize = f[3].parse().unwrap();
        let ours = find(&report, pair(&f[0]), &format!("{}{n}", f[1]))[t];
        close(ours, &f[4], &f[5], &f.join(" "));
        compared += 1;
    }
    assert_eq!(compared, 2 * 3 * 4 * 8);
    for f in rows(include_str!("data/bridge_proton_clover.txt")) {
        if f[0] != "12" {
            continue;
        }
        let name = match f[1].as_str() {
            "UNITY" => "proton_unity",
            "GAMMA0" => "proton_gamma4",
            _ => "proton_upper",
        };
        let t: usize = f[2].parse().unwrap();
        close(find(&report, (1, 2), name)[t], &f[3], &f[4], &f.join(" "));
    }

    // The charge on the links 0..3 (between the source at t = 0 and the sink at t = 4) and on
    // the others is constant; the jump is `quarks` times the two-point function at the sink.
    let charge = |report: &ruttice::app::Report, quarks: (usize, usize), sink: &str, n: f64| {
        let q = find(report, quarks, "C3_conserved_V4");
        let c2 = find(report, quarks, &format!("C2_{sink}"))[4];
        let size = c2.norm_sqr().sqrt();
        for t in 0..8 {
            let expected = if t < 4 { q[0] } else { q[4] };
            assert!(
                (q[t] - expected).norm_sqr().sqrt() < 1e-8 * size,
                "{sink}: link {t}: {q:?}"
            );
        }
        let jump = q[0] - q[4];
        assert!(
            (jump - c2.scale(n)).norm_sqr().sqrt() < 1e-8 * size,
            "{sink}: jump {jump:?}, C2 {c2:?}"
        );
    };
    charge(&report, (1, 2), "PS", 1.0);
    assert_eq!(report.sequential_solves.len(), 1);
    let proton = with(
        &with(
            MEASUREMENTS_YAML,
            "sink       : PS",
            "sink       : proton_upper",
        ),
        "current_on : u",
        "current_on : d",
    );
    charge(
        &run(&config(&proton), base).unwrap(),
        (1, 2),
        "proton_upper",
        1.0,
    );
}

/// The proton at zero momentum is printed once, `momenta` listing [0, 0, 0] or not.
#[test]
fn the_proton_at_zero_momentum_once() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let text = format!(
        "{CLOVER_YAML}Measure:\n  channels : [PS]\n  momenta : [[0, 0, 0]]\n  proton : true\n"
    );
    let report = run(&config(&text), &examples).unwrap();
    let n = report
        .correlators
        .iter()
        .filter(|c| c.quarks == (1, 2) && c.channel == "proton_upper")
        .count();
    assert_eq!(n, 1);
}

/// Mistakes in the `Measure` section are refused before any work.
#[test]
fn mistakes_in_the_measure_section_are_refused() {
    let dir = scratch("measure_mistakes");
    let three_point = |body: &str| format!("{CLOVER_YAML}Measure:\n  three_point:\n{body}");
    let ok = "    t_sink : 4\n    sink : PS\n    current_on : u\n    currents : [V4]\n";
    for (text, expected) in [
        (
            three_point(&ok.replace("t_sink : 4", "t_sink : 8")),
            "between 1 and 7",
        ),
        (
            three_point(&ok.replace("sink : PS", "sink : XX")),
            "proton_upper",
        ),
        (three_point(&ok.replace("[V4]", "[V5]")), "unknown current"),
        (
            with(
                &three_point(ok),
                "    source_position : [0, 0, 0, 0]\n    slope           : 0.5",
                "    source_position : [1, 0, 0, 0]\n    slope           : 0.5",
            ),
            "at one place",
        ),
    ] {
        let e = run(&config(&text), &dir).unwrap_err().to_string();
        assert!(e.contains(expected), "{e}");
    }
    // The momenta need one source position; quark 2 elsewhere is refused.
    let moved = with(
        &format!("{CLOVER_YAML}Measure:\n  momenta : [[1, 0, 0]]\n"),
        "    source_position : [0, 0, 0, 0]\n    slope           : 0.5",
        "    source_position : [1, 0, 0, 0]\n    slope           : 0.5",
    );
    let e = run(&config(&moved), &dir).unwrap_err().to_string();
    assert!(
        e.contains("Measure.momenta") && e.contains("quark 2"),
        "{e}"
    );
    let typo = format!("{CLOVER_YAML}Measure:\n  protn : true\n");
    assert!(Params::from_yaml(&typo).is_err());
    // Domain-wall runs refuse what they would not compute.
    let mdwf = "Main:\n  lattice_size : [4, 4, 4, 8]\nGauge:\n  config_filename_input : c.txt\nFopr:\n  fermion_type : Domainwall_5din\n  quark_mass : 0.05\n  domain_wall_height : 1.0\n  extent_of_5th_dimension : 8\nSolver:\n  solver_type : CG\n  convergence_criterion_squared : 1.0e-20\nSource:\n  source_position : [0, 0, 0, 0]\nMeasure:\n  proton : true\n";
    let e = Params::from_yaml(mdwf).unwrap_err().to_string();
    assert!(e.contains("for Wilson and clover fermions so far"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
