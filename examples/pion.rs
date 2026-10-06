//! Pion correlator of one gauge configuration, start to finish.
//!
//! ```text
//! cargo run --release --example pion -- CONFIG LX LY LZ LT KAPPA
//! ```
//!
//! Reads a plain-text configuration, prints the plaquette, solves for the propagator from
//! a point source at the origin (twelve right-hand sides together, even-odd preconditioned
//! CG on the normal equations) and prints the pseudoscalar correlator for every time slice.
//! Fermions are antiperiodic in all four directions.

use ruttice::fermion::{solve_even_odd, Bound, FermionOp, Variant, Wilson, WilsonParams};
use ruttice::gauge::plaquette;
use ruttice::io::read_bridge_text;
use ruttice::measure::{pion_correlator, point_source, NUM_SOURCES};
use ruttice::solver::{solve_cgne, Cg, Stop};
use ruttice::{Boundary, Lattice, LinearFn, Stencil, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: pion CONFIG LX LY LZ LT KAPPA";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let mut dims = [0usize; ND];
    for mu in 0..ND {
        dims[mu] = args[1 + mu].parse().map_err(|_| USAGE)?;
    }
    let kappa: f64 = args[5].parse().map_err(|_| USAGE)?;
    if dims.iter().any(|l| *l == 0 || l % 2 != 0) {
        return Err("every extent must be even and non-zero".into());
    }

    let lat = Lattice::single_node(dims);
    let u = read_bridge_text(&path, &lat)?;
    let periodic = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
    println!("lattice    {dims:?}");
    println!("plaquette  {:.15}", plaquette(&u, &periodic));

    let boundary = [Boundary::Antiperiodic; ND];
    let mut op = Wilson::<f64>::new(&lat, WilsonParams { kappa, boundary });
    op.set_gauge(&u)?;
    let index = op.even_odd_index().ok_or("even-odd needs even extents")?;

    let b = point_source::<f64>(&lat, [0; ND]);
    let mut s = b.zeros_like();
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 2000,
        max_restarts: 3,
    };
    let start = Instant::now();
    let info = solve_even_odd(&op, index, &mut s, &b, |schur, x_e, rhs| {
        solve_cgne(&mut Cg::new(stop), schur, x_e, rhs)
    })?;
    let seconds = start.elapsed().as_secs_f64();

    // Residual of the full system, from the full operator.
    let mut r = b.zeros_like();
    Bound::new(&op, Variant::D).mult(&mut r, &s);
    r.scale(&[-1.0; NUM_SOURCES]);
    r.axpy(&[C64::new(1.0, 0.0); NUM_SOURCES], &b);
    let mut rr = [0.0; NUM_SOURCES];
    r.norm_sqr(&mut rr);
    let worst = rr.iter().fold(0.0, |m: f64, x| m.max(*x));

    println!("kappa      {kappa}");
    println!(
        "solve      {} CG steps, {} restarts, {seconds:.3} s",
        info.iterations, info.restarts
    );
    println!("residual   worst |D S - b|^2 over {NUM_SOURCES} columns = {worst:.3e}");
    println!("# t  C_PS(t)");
    for (t, c) in pion_correlator(&s).iter().enumerate() {
        println!("{t:3}  {c:.15e}");
    }
    Ok(())
}
