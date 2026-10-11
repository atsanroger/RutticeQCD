//! Timing of the Wilson operator, shared by `ruttice-bench` and `ruttice-bench-mpi`.
//!
//! Applies D on a random gauge field and prints the time of one application and the speed
//! in GFlops per core, in double and in single precision, with the links kept in full and
//! as two rows, and for fields in blocks of sites (the SIMD layout). The
//! flops are the operator's nominal count (`FermionOp::flops_per_site`), the one Bridge++
//! uses, so the figure can be set next to the "GFlops / core" that Bridge++ prints. The operator, the hopping parameter and the boundary condition are
//! those of the Bridge++ test `Mult_Wilson`, so that the two programs can be timed on the
//! same problem.

use super::choose_grid;
use crate::fermion::{Bound, FermionOp, LinkFormat, Variant, Wilson, WilsonParams, WilsonSimd};
use crate::gauge::random_gauge;
use crate::kernel::LaneOf;
use crate::solver::{solve_cgne, BiCgStab, Cg, SolveError, Solver, Stop};
use crate::{
    min_sites_per_task, norm_sqr, random_spinor, Aosoa, Boundary, Comm, ConvertFrom, Field, Full,
    Lattice, Spinor, Storage, Vector, ND,
};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

/// How often the timed loop is repeated; the fastest repetition is reported next to the mean.
const REPEATS: usize = 5;

const USAGE: &str = "usage: ruttice-bench LX LY LZ LT [NUMBER_OF_MULTS [PX PY PZ PT]]

Times the Wilson operator on a lattice of the given size. With several processes
(ruttice-bench-mpi under mpirun) PX PY PZ PT is the number of processes in each direction;
without it the lattice is split automatically. Threads per process: RAYON_NUM_THREADS.";

struct Timing {
    best: f64,
    mean: f64,
    norm_sqr: f64,
    /// Nominal floating-point operations of one application on the whole lattice.
    flops: f64,
}

/// Seconds per application of D, from `REPEATS` loops of `mults` applications each. Every
/// process makes the same calls; the times are those of the calling process.
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
    // Each process draws its own part: the links do not join up into one field that a
    // single process would have drawn, which does not matter for the timing.
    let rank = u64::try_from(lat.comm.rank()).expect("ranks fit in 64 bits");
    op.set_gauge(&random_gauge(lat, 1 + 2 * rank))
        .expect("the gauge field is on this lattice");
    let mut work = op.work();
    let mut a = Field::<T, Spinor>::spinor(lat, 1);
    a.convert_from(&random_spinor(lat, 1, 2 + 2 * rank));
    let mut b = Field::<T, Spinor>::spinor(lat, 1);

    // Warm up: touch the memory and wake the threads.
    for _ in 0..2 {
        op.mult(&mut work, &mut b, &a, Variant::D);
        op.mult(&mut work, &mut a, &b, Variant::D);
    }
    a.convert_from(&random_spinor(lat, 1, 2 + 2 * rank));

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
        norm_sqr: norm_sqr(&b),
        flops: (op.flops_per_site() * lat.num_global_sites()) as f64,
    }
}

/// `time_mult` for fields in blocks of `W` sites (the SIMD layout). `None` if the lattice
/// cannot be cut into `W` sub-lattices.
fn time_simd<T: Storage, const W: usize>(lat: &Arc<Lattice>, mults: usize) -> Option<Timing>
where
    T::Compute: LaneOf<W>,
{
    let params = WilsonParams {
        kappa: 0.12,
        boundary: [Boundary::Antiperiodic; ND],
    };
    let mut op = WilsonSimd::<T, W>::new(lat, params)?;
    let rank = u64::try_from(lat.comm.rank()).expect("ranks fit in 64 bits");
    op.set_gauge(&random_gauge(lat, 1 + 2 * rank))
        .expect("the gauge field is on this lattice");
    let mut a = Field::<T, Spinor, Full, Aosoa<W>>::with_shape(lat, 24, 1, 1, 1);
    a.convert_from(&random_spinor(lat, 1, 2 + 2 * rank));
    let mut b = a.zeros_like();
    let mut work = op.work();
    for _ in 0..2 {
        op.mult(&mut work, &mut b, &a, Variant::D);
        op.mult(&mut work, &mut a, &b, Variant::D);
    }
    a.convert_from(&random_spinor(lat, 1, 2 + 2 * rank));

    let mut seconds = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..mults {
            op.mult(&mut work, &mut b, &a, Variant::D);
        }
        seconds.push(start.elapsed().as_secs_f64() / mults as f64);
    }
    let mut norm = [0.0];
    b.norm_sqr(&mut norm);
    Some(Timing {
        best: seconds.iter().copied().fold(f64::INFINITY, f64::min),
        mean: seconds.iter().sum::<f64>() / REPEATS as f64,
        norm_sqr: norm[0],
        flops: (op.flops_per_site() * lat.num_global_sites()) as f64,
    })
}

/// Milliseconds per iteration of CG on the normal equations and of BiCGStab on `op`, from
/// a fixed number of iterations on a random right-hand side: (CG, BiCGStab). An iteration
/// of either applies the operator twice; the rest is the solver's own work on its vectors.
fn time_solvers<V: Vector, F: FermionOp<V>>(op: &F, b: &V, iterations: usize) -> (f64, f64) {
    // A tolerance of zero is never met: the solver runs for the iterations it is given.
    let stop = Stop::simple(0.0, iterations);
    let per_iteration = |solve: &mut dyn FnMut(&mut V) -> Result<(), SolveError>| {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let mut x = b.zeros_like();
            let start = Instant::now();
            let done = match solve(&mut x) {
                Err(SolveError::NotConverged(info)) => info.iterations,
                // Anything else did not run the iterations asked for.
                _ => return f64::NAN,
            };
            best = best.min(start.elapsed().as_secs_f64() * 1e3 / done as f64);
        }
        best
    };
    // The solvers are kept across the repetitions, so that from the second one on no
    // work vector is allocated inside the time measured.
    let (mut cg, mut bicgstab) = (Cg::new(stop), BiCgStab::new(stop));
    let cg = per_iteration(&mut |x| {
        solve_cgne(&mut cg, &mut Bound::new(op, Variant::D), x, b).map(|_| ())
    });
    let bicgstab = per_iteration(&mut |x| {
        bicgstab
            .solve(&mut Bound::new(op, Variant::D), None, x, b)
            .map(|_| ())
    });
    (cg, bicgstab)
}

/// The whole benchmark program: interpret the arguments (without the program name), run on
/// the processes of `comm`, and return the exit code. Only rank 0 prints.
pub fn wilson_benchmark(args: &[String], comm: Arc<dyn Comm>, out: &mut dyn Write) -> u8 {
    let root = comm.rank() == 0;
    let numbers: Option<Vec<usize>> = args.iter().map(|a| a.parse().ok()).collect();
    let parsed = match numbers.as_deref() {
        Some([x, y, z, t]) => Some(([*x, *y, *z, *t], 100, None)),
        Some([x, y, z, t, n]) => Some(([*x, *y, *z, *t], *n, None)),
        Some([x, y, z, t, n, px, py, pz, pt]) => {
            Some(([*x, *y, *z, *t], *n, Some([*px, *py, *pz, *pt])))
        }
        _ => None,
    };
    let Some((dims, mults, asked)) = parsed.filter(|(d, n, _)| !d.contains(&0) && *n > 0) else {
        if root {
            eprintln!("{USAGE}");
        }
        return 2;
    };
    let processes = comm.size();
    let fits = |grid: &[usize; ND]| {
        grid.iter().product::<usize>() == processes
            && (0..ND).all(|mu| grid[mu] > 0 && dims[mu].is_multiple_of(grid[mu]))
    };
    let Some(grid) = asked.or_else(|| choose_grid(dims, processes)).filter(fits) else {
        if root {
            eprintln!(
                "ruttice-bench: the lattice {dims:?} cannot be split over {processes} processes as asked"
            );
        }
        return 2;
    };
    let lat = Lattice::new(dims, grid, comm);
    let sites = lat.num_global_sites() as f64;
    let threads = rayon::current_num_threads();
    let cores = (threads * processes) as f64;

    // Printing only fails if the reader went away; the timings are still made.
    let mut line = |text: String| {
        if root {
            let _ = writeln!(out, "{text}");
        }
    };
    line(format!(
        "lattice {dims:?}, {processes} processes {grid:?} x {threads} threads, {mults} mults x {REPEATS} repeats, schedule {:?}, at least {} sites per task",
        crate::schedule(),
        min_sites_per_task()
    ));
    line(
        "precision  links     ms/mult(best)  ms/mult(mean)  ns/site(best)  GFlops/core  |D psi|^2"
            .to_string(),
    );
    let (full, two) = (LinkFormat::Full, LinkFormat::TwoRows);
    let mut rows = vec![
        ("double     full    ", time_mult::<f64>(&lat, mults, full)),
        ("double     two rows", time_mult::<f64>(&lat, mults, two)),
        ("single     full    ", time_mult::<f32>(&lat, mults, full)),
        ("single     two rows", time_mult::<f32>(&lat, mults, two)),
    ];
    // The block layout, for the widths that fill a 256-bit and a 512-bit register in each
    // precision.
    let simd = [
        ("double     blocks 4", time_simd::<f64, 4>(&lat, mults)),
        ("double     blocks 8", time_simd::<f64, 8>(&lat, mults)),
        ("single     blocks 8", time_simd::<f32, 8>(&lat, mults)),
        ("single     blocks16", time_simd::<f32, 16>(&lat, mults)),
    ];
    rows.extend(simd.into_iter().filter_map(|(name, t)| Some((name, t?))));
    for (name, t) in rows {
        line(format!(
            "{name}  {:>13.4}  {:>13.4}  {:>13.2}  {:>11.4}  {:.12e}",
            t.best * 1e3,
            t.mean * 1e3,
            t.best * 1e9 / sites,
            t.flops / t.best / cores * 1e-9,
            t.norm_sqr
        ));
    }
    line(format!("peak memory: {}", super::memory_line()));

    // The solvers, on one right-hand side: what an iteration costs beyond its two
    // applications of the operator is the work on the vectors.
    let params = WilsonParams {
        kappa: 0.12,
        boundary: [Boundary::Antiperiodic; ND],
    };
    let iterations = mults.max(20);
    let rank = u64::try_from(lat.comm.rank()).expect("ranks fit in 64 bits");
    let u = random_gauge(&lat, 1 + 2 * rank);
    let b = random_spinor(&lat, 1, 2 + 2 * rank);
    let mut op = Wilson::<f64>::new(&lat, params);
    op.set_gauge(&u)
        .expect("the gauge field is on this lattice");
    let on_sites = time_solvers(&op, &b, iterations);
    let on_blocks = WilsonSimd::<f64, 4>::new(&lat, params).map(|mut op| {
        op.set_gauge(&u)
            .expect("the gauge field is on this lattice");
        let mut blocks = Field::<f64, Spinor, Full, Aosoa<4>>::with_shape(&lat, 24, 1, 1, 1);
        blocks.convert_from(&b);
        time_solvers(&op, &blocks, iterations)
    });
    line(format!(
        "solvers, double, ms/iteration over {iterations} iterations (two applications of D each)"
    ));
    line(format!(
        "cg_normal  sites     {:>10.4}
bicgstab   sites     {:>10.4}",
        on_sites.0, on_sites.1
    ));
    if let Some((cg, bicgstab)) = on_blocks {
        line(format!(
            "cg_normal  blocks 4  {cg:>10.4}
bicgstab   blocks 4  {bicgstab:>10.4}"
        ));
    }
    0
}
