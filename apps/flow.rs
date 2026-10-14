//! The Wilson flow of gauge configurations: t0 and the topological charge of each.
//!
//!     ruttice-flow LX LY LZ LT FORMAT FILE... [--step EPS] [--t-max T]
//!
//! See `ruttice::app::flow` for what is printed.

use ruttice::SingleComm;
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = ruttice::app::flow::flow_measure(
        &args,
        Arc::new(SingleComm),
        &mut std::io::stdout().lock(),
    );
    ExitCode::from(code)
}
