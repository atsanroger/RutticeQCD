use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "\
usage: ruttice PARAMETER_FILE

Reads a parameter file (.toml, .yaml or .yml), solves for the quark propagators it
describes and prints meson two-point functions. See cli/examples/ for parameter files.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "-h" || flag == "--help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        [file] => match ruttice_cli::run_file(Path::new(file), &mut std::io::stdout().lock()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("ruttice: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}
