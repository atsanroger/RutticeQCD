//! Checks: the MPI communicator, by launching `ruttice-mpi-check` under `mpirun` for several
//! process grids. Skipped, with a message, if there is no `mpirun`.

use std::io::ErrorKind;
use std::process::Command;

#[test]
fn distributed_results_match_a_single_process() {
    for grid in [
        [1, 1, 1, 1],
        [2, 1, 1, 1],
        [1, 2, 1, 2],
        [1, 1, 2, 4],
        [2, 2, 2, 2],
    ] {
        let processes: usize = grid.iter().product();
        let output = Command::new("mpirun")
            .args(["--oversubscribe", "-np", &processes.to_string()])
            .arg(env!("CARGO_BIN_EXE_ruttice-mpi-check"))
            .args(grid.map(|n| n.to_string()))
            .output();
        let output = match output {
            Ok(output) => output,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                eprintln!("SKIPPED: mpirun was not found");
                return;
            }
            Err(e) => panic!("could not launch mpirun: {e}"),
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("{stdout}");
        assert!(
            output.status.success() && stdout.contains("PASSED"),
            "grid {grid:?} failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
