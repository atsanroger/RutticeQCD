use ruttice_core::SingleComm;
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = ruttice_cli::command_line(&args, Arc::new(SingleComm), &mut std::io::stdout().lock());
    ExitCode::from(code)
}
