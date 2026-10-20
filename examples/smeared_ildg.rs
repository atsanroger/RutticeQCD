//! A configuration in Bridge++'s binary format, stout-smeared as `GaugeSmearing` does in a run,
//! written as an ILDG file in double precision: the links of the fermion operators of such a
//! run, for another code that reads ILDG (QUDA's tests through QIO, for one). Prints the
//! plaquette before and after, as a run does.
//!
//!   cargo run --release --example smeared_ildg -- <in.bin> <nx> <ny> <nz> <nt> <rho> <steps> <out.ildg>

use ruttice::gauge::smearing::Stout;
use ruttice::gauge::{plaquette, GaugeSmearing, Smear};
use ruttice::io::{read_binary, read_lime, write_ildg, FilePrecision};
use ruttice::{Boundary, Lattice, Stencil};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        8,
        "usage: smeared_ildg <in.bin> <nx> <ny> <nz> <nt> <rho> <steps> <out.ildg>"
    );
    let extent = |k: usize| args[k].parse::<usize>().expect("a lattice extent");
    let lat = Lattice::single_node([extent(1), extent(2), extent(3), extent(4)]);
    let u = read_binary(Path::new(&args[0]), &lat).expect("the configuration");
    let smearing = GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(args[5].parse().expect("rho"))),
        steps: args[6].parse().expect("the number of steps"),
    };
    let v = smearing.apply(&u);
    let st = Stencil::nearest(&lat, [Boundary::Periodic; 4]);
    println!(
        "plaquette {:.15} -> {:.15}",
        plaquette(&u, &st),
        plaquette(&v, &st)
    );
    let out = Path::new(&args[7]);
    write_ildg(out, &v, FilePrecision::Double).expect("the ILDG file");
    // Read back: the same links.
    let back = read_lime(out, &lat).expect("the ILDG file just written");
    println!(
        "written {}, read back: plaquette {:.15}",
        out.display(),
        plaquette(&back, &st)
    );
}
