//! Times the Wilson operator on one process:
//!
//!     ruttice-bench LX LY LZ LT [NUMBER_OF_MULTS]
//!
//! See `ruttice::app::wilson_benchmark` for what is measured and printed.

use ruttice::SingleComm;
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code =
        ruttice::app::wilson_benchmark(&args, Arc::new(SingleComm), &mut std::io::stdout().lock());
    ExitCode::from(code)
}
