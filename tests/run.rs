//! The parameter file and the run it drives.

mod common;

use common::{bridge_test_file, BRIDGE_TESTS_ENV};
use ruttice::app::{
    run, BoundaryConfig, ChannelsConfig, CliError, Config, FermionConfig, SolverKind, SourceConfig,
};
use ruttice::gauge::random_gauge;
use ruttice::io::write_bridge_text;
use ruttice::Lattice;
use std::path::{Path, PathBuf};

const WILSON_TOML: &str = include_str!("../examples/wilson_pion.toml");
const CLOVER_YAML: &str = include_str!("../examples/clover_two_flavours.yaml");
const CLOVER_MG_YAML: &str = include_str!("../examples/clover_mg.yaml");

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
    assert_eq!(
        wilson.measure.channels,
        ChannelsConfig::List(vec!["PS".to_string()])
    );

    let clover = Config::from_yaml(CLOVER_YAML).unwrap();
    assert_eq!(clover.quark.len(), 2);
    assert_eq!(clover.solver.max_restarts, 3, "default");
    assert_eq!(
        clover.measure.channels,
        ChannelsConfig::All,
        "all channels when absent"
    );
    assert_eq!(
        clover.quark[1].source,
        SourceConfig::Exponential {
            position: [0; 4],
            slope: 0.5,
            power: 2.0
        }
    );
}

/// Gauge smearing is `gauge_smearing`; a key that says only "smearing" is refused, in either
/// place one might write it, because it does not say whether links or sources are smeared.
#[test]
fn gauge_smearing_has_its_own_name() {
    use ruttice::app::GaugeSmearingConfig;
    // A table goes at the end (keys after it would belong to it), a key at the start.
    let with = |extra: &str| {
        if extra.starts_with('[') {
            format!("{WILSON_TOML}\n{extra}\n")
        } else {
            format!("{extra}\n{WILSON_TOML}")
        }
    };
    let c = Config::from_toml(&with(
        "[gauge_smearing]\ntype = \"stout\"\nrho = 0.1\nsteps = 3",
    ))
    .unwrap();
    assert_eq!(
        c.gauge_smearing,
        Some(GaugeSmearingConfig::Stout {
            rho: 0.1,
            steps: 3,
            spatial_only: false
        })
    );
    let c = Config::from_toml(&with(
        "[gauge_smearing]\ntype = \"ape\"\nrho = 0.5\nsteps = 2\nspatial_only = true",
    ))
    .unwrap();
    assert!(matches!(
        c.gauge_smearing,
        Some(GaugeSmearingConfig::Ape {
            spatial_only: true,
            max_iterations: 100,
            ..
        })
    ));
    for bad in ["[smearing]\ntype = \"stout\"", "smear = 3"] {
        match Config::from_toml(&with(bad)) {
            Err(CliError::Invalid(e)) => {
                assert!(e.contains("gauge_smearing") && e.contains("source"), "{e}")
            }
            other => panic!("{bad}: {other:?}"),
        }
    }
    let inside = WILSON_TOML.replacen("[gauge]\n", "[gauge]\nsmearing = 1\n", 1);
    assert!(inside != WILSON_TOML, "the example has a [gauge] table");
    assert!(matches!(
        Config::from_toml(&inside),
        Err(CliError::Invalid(_))
    ));
}

/// A run on gauge-smeared links reports the plaquette before and after the smearing.
#[test]
fn a_run_on_smeared_links() {
    let dir = scratch("smeared");
    let lat = Lattice::single_node([4, 4, 4, 8]);
    write_bridge_text(&dir.join("conf_04040408.txt"), &random_gauge(&lat, 5)).unwrap();
    let mut c = Config::from_toml(&format!(
        "{WILSON_TOML}\n[gauge_smearing]\ntype = \"stout\"\nrho = 0.1\nsteps = 2\n"
    ))
    .unwrap();
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
    // Multigrid needs a flexible solver (FGMRES, BiCGStab) and blocks that divide the
    // lattice.
    let with_multigrid = |block: &str| {
        WILSON_TOML.replace(
            "max_restarts = 3 ",
            &format!("multigrid = {{ block = {block} }}\nmax_restarts = 3 "),
        )
    };
    let m = invalid(with_multigrid("[2, 2, 2, 2]"));
    assert!(m.contains("fgmres"), "{m}");
    let fgmres = |text: String| text.replace("\"cg_normal\"", "\"fgmres\"");
    let whole = |text: String| text.replace("even_odd = true", "even_odd = false");
    let m = invalid(whole(fgmres(with_multigrid("[3, 2, 2, 2]"))));
    assert!(m.contains("block"), "{m}");
    // Blocks of sites: a width there is an operator for, a lattice that can be cut, parts
    // with even extents for even-odd.
    let simd = |lanes: usize| WILSON_TOML.replace("simd = 1 ", &format!("simd = {lanes} "));
    let m = invalid(simd(3));
    assert!(m.contains("solver.simd must be"), "{m}");
    let m = invalid(simd(8).replace("lattice = [4, 4, 4, 8]", "lattice = [4, 4, 2, 8]"));
    assert!(m.contains("which must all be even"), "{m}");
    let m = invalid(whole(
        simd(4).replace("lattice = [4, 4, 4, 8]", "lattice = [3, 3, 3, 6]"),
    ));
    assert!(m.contains("cannot be halved"), "{m}");
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
        // The multigrid under a solver on the even sites, and with the fields in blocks.
        multigrid("single").replace("even_odd = false", "even_odd = true"),
        multigrid("double").replace("simd = 1 ", "simd = 4 "),
        multigrid("single")
            .replace("even_odd = false", "even_odd = true")
            .replace("simd = 1 ", "simd = 2 "),
        // Fields in blocks of sites.
        base.replace("simd = 1 ", "simd = 2 "),
        base.replace("simd = 1 ", "simd = 8 "),
        base.replace("simd = 1 ", "simd = 4 ")
            .replace("even_odd = true", "even_odd = false"),
        fgmres.replace("simd = 1 ", "simd = 4 "),
        base.replace("simd = 1 ", "simd = 4 ")
            .replace("\"cg_normal\"", "\"bicgstab\""),
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

/// The YAML example, run on the Bridge++ test configuration, reproduces the reference
/// table produced with Bridge++ (tests/data).
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

/// The same run with the fields in blocks of eight sites.
#[test]
fn clover_example_in_blocks_reproduces_the_reference_table() {
    let Some(conf) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let text = CLOVER_YAML.replace(
        "  type: cg_normal
",
        "  type: cg_normal
  simd: 8
",
    );
    let config = Config::from_yaml(&text).unwrap();
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

/// A parameter file in the style of Bridge++ for clover fermions (FGMRES on the even sites
/// with the multigrid) runs through `run_file`, says what it runs in comment lines, and gives
/// the pion correlator of CG on the same operator from a file of the other format.
#[test]
fn a_file_in_the_style_of_bridge_runs_clover_fermions() {
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

    let reference = WILSON_TOML
        .replace(
            "fermion = { type = \"wilson\", kappa = 0.12, boundary = [\"antiperiodic\", \"antiperiodic\", \"antiperiodic\", \"antiperiodic\"] }",
            "fermion = { type = \"clover\", kappa = 0.12, csw = 1.0, boundary = [\"periodic\", \"periodic\", \"periodic\", \"antiperiodic\"] }",
        )
        .replace("file = \"wilson_pion.out\"", "");
    let report = run(&Config::from_toml(&reference).unwrap(), &dir).unwrap();
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
    // "smearing" alone is refused here too.
    let bad =
        include_str!("../examples/hmc_domainwall_pv.yaml").replace("gauge_smearing:", "smearing:");
    assert!(matches!(
        HmcConfig::from_yaml(&bad),
        Err(CliError::Invalid(_))
    ));
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
    for (format, name) in [("bridge_text", "conf_{n}.txt"), ("ildg", "conf_{n}.lime")] {
        let yaml = |grid: &str| {
            format!(
                "lattice: [4, 4, 4, 8]\n\
                 process_grid: {grid}\n\
                 start: {{type: cold}}\n\
                 hmc:\n  trajectories: 2\n  first_trajectory: 100\n  trajectory_length: 0.5\n  \
                 integrator: {{scheme: omelyan, steps: [5]}}\n  metropolis: false\n\
                 actions:\n  - {{type: plaquette, beta: 5.5, level: 0}}\n\
                 output: {{save_every: 1, file: \"{name}\", format: {format}}}\n"
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
