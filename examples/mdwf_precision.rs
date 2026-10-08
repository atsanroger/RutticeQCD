//! BiCGStab on the Pauli-Villars preconditioned domain-wall operator A in single and in
//! double precision, from a random and from a point source: does it break down?
//!
//! ```text
//! cargo run --release --example mdwf_precision -- CONFIG L M5 MASS [LS] [TOL]
//! ```
//!
//! CONFIG is a plain-text configuration on an L^4 lattice. Möbius b = 1.5, c = 0.5,
//! boundary conditions (1, 1, 1, -1), LS = 8 unless given. BiCGStab solves A x = P b
//! without a preconditioner and without restarts, to |r|^2/|b|^2 < TOL (1e-10, the
//! single-precision target of the reference spectrum test). Each line: precision, source,
//! the outcome, iterations, the solver's own residual and |D x - b|^2/|b|^2 of the
//! solution, computed in double precision.

use ruttice::fermion::{FermionOp, Mdwf, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::kernel::{Number, Real};
use ruttice::solver::{BiCgStab, SolveError, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, Spinor, Storage, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

const USAGE: &str = "usage: mdwf_precision CONFIG L M5 MASS [LS] [TOL]";

/// The source in double precision: random, or a unit vector at the origin.
fn source(lat: &Arc<Lattice>, ls: usize, point: bool) -> Field<f64, Spinor> {
    let mut b = Field::<f64, Spinor>::with_shape(lat, 24, ls, 1, 1);
    if point {
        b.as_mut_slice()[0] = 1.0;
    } else {
        b.as_mut_slice()
            .copy_from_slice(random_spinor(lat, ls, 91).as_slice());
    }
    b
}

/// Solve A x = P b in precision T; the solution comes back in double precision.
fn solve<T: Storage>(
    lat: &Arc<Lattice>,
    params: MdwfParams,
    u: &Field<f64, ruttice::Gauge>,
    b64: &Field<f64, Spinor>,
    stop: Stop,
) -> Result<(String, Field<f64, Spinor>), Box<dyn Error>> {
    let mut op = Mdwf::<T>::new(lat, params);
    op.set_gauge(u)?;
    let mut b = Field::<T, Spinor>::with_shape(lat, 24, params.ls, 1, 1);
    for (to, from) in b.as_mut_slice().iter_mut().zip(b64.as_slice()) {
        *to = T::store(T::Compute::from_f64(*from));
    }
    let mut rhs = b.zeros_like();
    PvPreconditioned::new(&op, &b).precondition(&mut rhs, &b);
    let mut x = b.zeros_like();
    let outcome = match BiCgStab::new(stop).solve(
        &mut PvPreconditioned::new(&op, &b),
        None,
        &mut x,
        &rhs,
    ) {
        Ok(info) => format!(
            "converged      {:5}  own {:8.1e}",
            info.iterations, info.residual_sqr[0]
        ),
        Err(SolveError::NotConverged(info)) => format!(
            "not converged  {:5}  own {:8.1e}",
            info.iterations, info.residual_sqr[0]
        ),
        Err(SolveError::Breakdown { iterations, what }) => {
            format!("BREAKDOWN      {iterations:5}  ({what})")
        }
        Err(e) => format!("failed: {e}"),
    };
    let mut x64 = b64.zeros_like();
    for (to, from) in x64.as_mut_slice().iter_mut().zip(x.as_slice()) {
        *to = from.load().to_f64();
    }
    Ok((outcome, x64))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args.get(4).map_or(Ok(8), |s| s.parse()).map_err(|_| USAGE)?;
    let tol: f64 = args.get(5).map_or(Ok(1e-10), |s| s.parse()).map_err(|_| USAGE)?;

    let lat = Lattice::single_node([l; ND]);
    let u = read_bridge_text(&path, &lat)?;
    let params = MdwfParams {
        ls,
        m5,
        mass,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut op = Mdwf::<f64>::new(&lat, params);
    op.set_gauge(&u)?;
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, BiCGStab on A to {tol:.0e}, no restarts");
    let stop = Stop::simple(tol, 4000);
    for point in [false, true] {
        let b = source(&lat, ls, point);
        let physical = |x: &Field<f64, Spinor>| {
            let mut r = b.zeros_like();
            op.mult(&mut op.work(), &mut r, x, Variant::D);
            r.aypx(&[C64::new(-1.0, 0.0)], &b);
            let (mut left, mut size) = ([0.0], [0.0]);
            r.norm_sqr(&mut left);
            b.norm_sqr(&mut size);
            left[0] / size[0]
        };
        let name = if point { "point " } else { "random" };
        for (precision, result) in [
            ("f32", solve::<f32>(&lat, params, &u, &b, stop)?),
            ("f64", solve::<f64>(&lat, params, &u, &b, stop)?),
        ] {
            let (outcome, x) = result;
            println!(
                "{precision} {name}  {outcome}  |Dx-b|^2/|b|^2 {:.1e}",
                physical(&x)
            );
        }
    }
    Ok(())
}
