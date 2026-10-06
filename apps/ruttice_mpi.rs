//! The `ruttice` program for several processes:
//!
//!     mpirun -np 4 ruttice-mpi --grid 1,1,2,2 PARAMETER_FILE
//!
//! Same parameter files and output as `ruttice`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = ruttice::backend::mpi::run(|comm| {
        ruttice::app::command_line(&args, comm, &mut std::io::stdout().lock())
    });
    ExitCode::from(code)
}
