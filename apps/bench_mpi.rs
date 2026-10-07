//! Times the Wilson operator on several processes:
//!
//!     mpirun -np 2 ruttice-bench-mpi LX LY LZ LT [NUMBER_OF_MULTS [PX PY PZ PT]]
//!
//! See `ruttice::app::wilson_benchmark` for what is measured and printed.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = ruttice::backend::mpi::run(|comm| {
        ruttice::app::wilson_benchmark(&args, comm, &mut std::io::stdout().lock())
    });
    ExitCode::from(code)
}
