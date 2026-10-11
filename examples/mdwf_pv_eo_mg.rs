//! Möbius domain-wall solves on the even sites, with or without the even-odd
//! Pauli-Villars multigrid, from a parameter file in the style of Bridge++: what
//! `ruttice::app::mdwf::run` does (and `ruttice PARAMS.yaml` runs), here with a choice of
//! the seeds of the random right-hand sides when the file has no `Source` block.
//!
//! ```text
//! cargo run --release --example mdwf_pv_eo_mg -- examples/mdwf_pv_eo_mg.yaml [seeds:A-B]
//! ```

use ruttice::app::params::Params;
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_pv_eo_mg PARAMS.yaml|PARAMS.toml [seeds:A-B]";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = PathBuf::from(args.first().ok_or(USAGE)?);
    let seeds: (u64, u64) = args
        .iter()
        .find_map(|a| a.strip_prefix("seeds:")?.split_once('-'))
        .map_or(Ok((91, 91)), |(a, b)| -> Result<_, Box<dyn Error>> {
            Ok((a.parse()?, b.parse()?))
        })?;
    let params = Params::load(&file)?;
    println!("{params}");
    ruttice::app::mdwf::run(&params, &file, seeds, &mut std::io::stdout())
}
