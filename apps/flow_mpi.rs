//! `ruttice-flow` on several processes:
//!
//!     mpirun -np 4 ruttice-flow-mpi LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T]
//!
//! Same arguments and output as `ruttice-flow`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = ruttice::backend::mpi::run(|comm| {
        ruttice::app::flow::flow_measure(&args, comm, &mut std::io::stdout().lock())
    });
    ExitCode::from(code)
}
