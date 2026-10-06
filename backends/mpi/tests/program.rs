//! Checks: `ruttice-mpi` under `mpirun` prints the same correlators as one process.
//! Skipped, with a message, if there is no `mpirun`.

use ruttice_cli::{run, Config};
use ruttice_core::Lattice;
use ruttice_io::write_bridge_text;
use ruttice_validation::random_gauge;
use std::io::ErrorKind;
use std::process::Command;

/// A Wilson and a clover quark, a point and a smeared source away from the origin, so
/// that the sources and the clover term all cross process boundaries.
const PARAMETERS: &str = r#"
lattice = [4, 4, 4, 8]

[gauge]
file = "conf.txt"

[solver]
type = "bicgstab"
even_odd = true
tolerance_squared = 1.0e-26
max_iterations = 2000

[[quark]]
fermion = { type = "wilson", kappa = 0.12, boundary = ["periodic", "periodic", "periodic", "antiperiodic"] }
source = { type = "point", position = [1, 2, 3, 5] }

[[quark]]
fermion = { type = "clover", kappa = 0.11, csw = 1.2, boundary = ["periodic", "periodic", "periodic", "antiperiodic"] }
source = { type = "exponential", position = [3, 0, 1, 5], slope = 0.5, power = 1.0 }

[measure]
channels = ["PS", "V1", "A4_PS"]
"#;

/// The table lines of the program's output: (label, re, im).
fn table(text: &str) -> Vec<(String, f64, f64)> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(f.len(), 5, "unexpected line: {l}");
            (
                f[..3].join(" "),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn the_program_gives_the_same_correlators_on_several_processes() {
    let dir = std::env::temp_dir().join(format!("ruttice_mpi_program_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let whole = Lattice::single_node([4, 4, 4, 8]);
    write_bridge_text(&dir.join("conf.txt"), &random_gauge(&whole, 700)).unwrap();
    let file = dir.join("run.toml");
    std::fs::write(&file, PARAMETERS).unwrap();

    let reference = run(&Config::from_toml(PARAMETERS).unwrap(), &dir).unwrap();
    let scale = reference.correlators[0].values[5].re.abs();
    assert_eq!(reference.correlators.len(), 9);

    // The last one leaves the choice of the grid to the program.
    for (processes, grid) in [(4, Some("2,1,1,2")), (8, Some("1,2,2,2")), (4, None)] {
        let mut command = Command::new("mpirun");
        command
            .args(["--oversubscribe", "-np", &processes.to_string()])
            .arg(env!("CARGO_BIN_EXE_ruttice-mpi"));
        if let Some(grid) = grid {
            command.args(["--grid", grid]);
        }
        let output = match command.arg(&file).output() {
            Ok(output) => output,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                eprintln!("SKIPPED: mpirun was not found");
                return;
            }
            Err(e) => panic!("could not launch mpirun: {e}"),
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{processes} processes, grid {grid:?}:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(stdout.contains("# processes"), "{stdout}");
        let plaquette = stdout
            .lines()
            .find_map(|l| l.strip_prefix("# plaquette"))
            .map(|v| v.trim().parse::<f64>().unwrap())
            .expect("the output has a plaquette line");
        assert!((plaquette - reference.plaquette).abs() < 1e-13);

        let got = table(&stdout);
        let want: Vec<_> = reference
            .correlators
            .iter()
            .flat_map(|c| c.values.iter())
            .collect();
        assert_eq!(got.len(), want.len());
        for ((label, re, im), z) in got.iter().zip(want) {
            assert!(
                (re - z.re).abs() < 1e-9 * scale && (im - z.im).abs() < 1e-9 * scale,
                "{processes} processes, grid {grid:?}, {label}: {re} {im} vs {z:?}"
            );
        }
    }

    // A grid that does not fit is an error from every process, not a hang.
    let output = Command::new("mpirun")
        .args(["--oversubscribe", "-np", "2"])
        .arg(env!("CARGO_BIN_EXE_ruttice-mpi"))
        .args(["--grid", "1,1,1,4"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("process grid"));
    let _ = std::fs::remove_dir_all(&dir);
}
