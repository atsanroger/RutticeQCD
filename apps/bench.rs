//! Times the Wilson operator on a random gauge field:
//!
//!     ruttice-bench LX LY LZ LT [NUMBER_OF_MULTS]
//!
//! Prints the time of one application of D and the speed in GFlops per core, in double and
//! in single precision, for the number of threads the thread pool has (set with
//! RAYON_NUM_THREADS), with the links kept in full and as two rows. The flops are the operator's nominal count
//! (`FermionOp::flops_per_site`), the one Bridge++ uses, so the figure can be set next to
//! the "GFlops / core" that Bridge++ prints. The operator, the
//! hopping parameter and the boundary condition are those of the Bridge++ test
//! `Mult_Wilson`, so that the two programs can be timed on the same problem.

use ruttice::fermion::{FermionOp, LinkFormat, Variant, Wilson, WilsonParams};
use ruttice::gauge::random_gauge;
use ruttice::{random_spinor, Boundary, ConvertFrom, Field, Lattice, Spinor, Storage, ND};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

/// How often the timed loop is repeated; the fastest repetition is reported next to the mean.
const REPEATS: usize = 5;

struct Timing {
    best: f64,
    mean: f64,
    norm_sqr: f64,
    /// Nominal floating-point operations of one application on the whole lattice.
    flops: f64,
}

/// Seconds per application of D, from `REPEATS` loops of `mults` applications each.
fn time_mult<T: Storage>(lat: &Arc<Lattice>, mults: usize, links: LinkFormat) -> Timing
where
    Field<T, Spinor>: ConvertFrom<Field<f64, Spinor>>,
{
    let mut op = Wilson::<T>::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary: [Boundary::Antiperiodic; ND],
        },
    )
    .with_link_format(links);
    op.set_gauge(&random_gauge(lat, 1))
        .expect("the gauge field is on this lattice");
    let mut work = op.work();
    let mut a = Field::<T, Spinor>::spinor(lat, 1);
    a.convert_from(&random_spinor(lat, 1, 2));
    let mut b = Field::<T, Spinor>::spinor(lat, 1);

    // Warm up: touch the memory and wake the threads.
    for _ in 0..2 {
        op.mult(&mut work, &mut b, &a, Variant::D);
        op.mult(&mut work, &mut a, &b, Variant::D);
    }
    a.convert_from(&random_spinor(lat, 1, 2));

    let mut seconds = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..mults {
            op.mult(&mut work, &mut b, &a, Variant::D);
        }
        seconds.push(start.elapsed().as_secs_f64() / mults as f64);
    }
    Timing {
        best: seconds.iter().copied().fold(f64::INFINITY, f64::min),
        mean: seconds.iter().sum::<f64>() / REPEATS as f64,
        norm_sqr: ruttice::norm_sqr(&b),
        flops: (op.flops_per_site() * lat.num_sites()) as f64,
    }
}

fn main() -> ExitCode {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let (dims, mults) = match args.as_slice() {
        [x, y, z, t] => ([*x, *y, *z, *t], 100),
        [x, y, z, t, n] if *n > 0 => ([*x, *y, *z, *t], *n),
        _ => {
            eprintln!("usage: ruttice-bench LX LY LZ LT [NUMBER_OF_MULTS]");
            return ExitCode::from(2);
        }
    };
    if dims.contains(&0) {
        eprintln!("ruttice-bench: lattice extents must be positive");
        return ExitCode::from(2);
    }
    let lat = Lattice::single_node(dims);
    let sites = lat.num_sites() as f64;
    println!(
        "lattice {dims:?}, {} threads, {mults} mults x {REPEATS} repeats, at least {} sites per task",
        rayon::current_num_threads(),
        ruttice::min_sites_per_task()
    );
    let threads = rayon::current_num_threads() as f64;
    println!(
        "precision  links     ms/mult(best)  ms/mult(mean)  ns/site(best)  GFlops/core  |D psi|^2"
    );
    let (full, two) = (LinkFormat::Full, LinkFormat::TwoRows);
    for (name, t) in [
        ("double     full    ", time_mult::<f64>(&lat, mults, full)),
        ("double     two rows", time_mult::<f64>(&lat, mults, two)),
        ("single     full    ", time_mult::<f32>(&lat, mults, full)),
        ("single     two rows", time_mult::<f32>(&lat, mults, two)),
    ] {
        println!(
            "{name}  {:>13.4}  {:>13.4}  {:>13.2}  {:>11.4}  {:.12e}",
            t.best * 1e3,
            t.mean * 1e3,
            t.best * 1e9 / sites,
            t.flops / t.best / threads * 1e-9,
            t.norm_sqr
        );
    }
    ExitCode::SUCCESS
}
