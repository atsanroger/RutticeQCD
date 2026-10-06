//! Compares a Wilson operator and a solve on a lattice split over MPI processes with the
//! same computation on one process. Run as
//!
//!     mpirun -np 4 ruttice-mpi-check 2 1 1 2
//!
//! where the arguments are the process grid. Every process also computes the whole lattice
//! by itself as the reference, so this is for small lattices only.

use ruttice::fermion::{solve_even_odd, Bound, FermionOp, Variant, Wilson, WilsonParams};
use ruttice::gauge::random_gauge;
use ruttice::solver::{solve_cgne, Cg, Stop};
use ruttice::{local_part, random_spinor};
use ruttice::{Boundary, Field, Gauge, Lattice, Spinor, Storage, Vector, ND};
use std::process::ExitCode;
use std::sync::Arc;

const GLOBAL: [usize; ND] = [4, 4, 4, 8];
const BOUNDARY: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];

fn wilson<T: Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Wilson<T> {
    let mut op = Wilson::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary: BOUNDARY,
        },
    );
    op.set_gauge(u).expect("the gauge field is on this lattice");
    op
}

/// Number of reals that differ, summed over all processes.
fn mismatches<T: Storage + PartialEq>(
    lat: &Lattice,
    got: &Field<T, Spinor>,
    want: &Field<T, Spinor>,
) -> f64 {
    let differing = got
        .as_slice()
        .iter()
        .zip(want.as_slice())
        .filter(|(a, b)| a != b);
    let mut n = [differing.count() as f64];
    lat.comm.sum(&mut n);
    n[0]
}

fn main() -> ExitCode {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let Ok(grid) = <[usize; ND]>::try_from(args) else {
        eprintln!("usage: ruttice-mpi-check PX PY PZ PT");
        return ExitCode::from(2);
    };

    let passed = ruttice::backend::mpi::run(|comm| {
        let lat = Lattice::new(GLOBAL, grid, comm);
        let whole = Lattice::single_node(GLOBAL);
        let root = lat.comm.rank() == 0;
        let mut passed = true;
        let mut report = |name: &str, ok: bool, detail: String| {
            passed &= ok;
            if root {
                println!("{} {name}: {detail}", if ok { "ok  " } else { "FAIL" });
            }
        };

        // The same global fields on every process.
        let u = random_gauge(&whole, 601);
        let psi = random_spinor(&whole, 2, 602);
        let (u_local, psi_local) = (local_part(&u, &lat), local_part(&psi, &lat));

        let reference = wilson::<f64>(&whole, &u);
        let op = wilson::<f64>(&lat, &u_local);
        let mut work = op.work();
        for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
            let mut want = psi.zeros_like();
            reference.mult(&mut reference.work(), &mut want, &psi, variant);
            let mut got = psi_local.zeros_like();
            op.mult(&mut work, &mut got, &psi_local, variant);
            let n = mismatches(&lat, &got, &local_part(&want, &lat));
            report(
                &format!("Wilson {variant:?}"),
                n == 0.0,
                format!("{n} reals differ from the single-process result"),
            );
        }

        let (mut want, mut got) = ([0.0; 2], [0.0; 2]);
        psi.norm_sqr(&mut want);
        psi_local.norm_sqr(&mut got);
        let error = (got[0] - want[0]).abs() / want[0];
        report(
            "global norm",
            error < 1e-13,
            format!("relative difference {error:e}"),
        );

        // An even-odd preconditioned solve, checked by its residual against the global norm.
        let stop = Stop {
            tolerance_sqr: 1e-24,
            max_iterations: 1000,
            max_restarts: 3,
        };
        let solved = op.even_odd_index().map(|index| {
            let mut x = psi_local.zeros_like();
            solve_even_odd(&op, index, &mut x, &psi_local, |schur, x_e, rhs| {
                solve_cgne(&mut Cg::new(stop), schur, x_e, rhs)
            })
            .map(|info| (x, info.iterations))
        });
        match solved {
            None => report(
                "even-odd solve",
                true,
                "skipped: a local extent is odd".into(),
            ),
            Some(Err(e)) => report("even-odd solve", false, format!("{e:?}")),
            Some(Ok((x, iterations))) => {
                let mut r = x.zeros_like();
                op.mult(&mut work, &mut r, &x, Variant::D);
                r.scale(&[-1.0; 2]);
                r.axpy(&[ruttice::C64::new(1.0, 0.0); 2], &psi_local);
                let mut rr = [0.0; 2];
                r.norm_sqr(&mut rr);
                let worst = (rr[0] / got[0]).max(rr[1] / got[1]);
                report(
                    "even-odd solve",
                    worst < 1e-22,
                    format!("{iterations} iterations, |r|^2/|b|^2 = {worst:e}"),
                );
            }
        }
        // A Bound keeps its own work object; make sure a second user of the operator works.
        let mut d = Bound::new(&op, Variant::D);
        let mut twice = psi_local.zeros_like();
        ruttice::LinearFn::mult(&mut d, &mut twice, &psi_local);
        let mut once = psi_local.zeros_like();
        op.mult(&mut work, &mut once, &psi_local, Variant::D);
        let n = mismatches(&lat, &twice, &once);
        report("repeat", n == 0.0, format!("{n} reals differ"));

        // Every process must agree on the verdict.
        let mut failures = [if passed { 0.0 } else { 1.0 }];
        lat.comm.sum(&mut failures);
        if root {
            println!(
                "grid {grid:?}, {} processes: {}",
                lat.comm.size(),
                if failures[0] == 0.0 {
                    "PASSED"
                } else {
                    "FAILED"
                }
            );
        }
        failures[0] == 0.0
    });
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
