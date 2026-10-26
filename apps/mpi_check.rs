//! Compares a Wilson operator and a solve on a lattice split over MPI processes with the
//! same computation on one process. Run as
//!
//!     mpirun -np 4 ruttice-mpi-check 2 1 1 2
//!
//! where the arguments are the process grid.

use ruttice::comm::{HaloSpec, FACES};
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

        // The sum adds in rank order (these numbers sum to something else in pairs), and every
        // process gets the same bits.
        let parts = [1.0, 1e16, -1e16, 1.0];
        let size = lat.comm.size();
        let mut x = [parts[lat.comm.rank() % 4]];
        lat.comm.sum(&mut x);
        let in_order = (0..size).fold(0.0, |s, r| s + parts[r % 4]);
        let seen = lat.comm.gather(&x);
        let same = seen.iter().all(|y| y[0].to_bits() == x[0].to_bits());
        let mut wrong = [f64::from(u8::from(
            x[0].to_bits() != in_order.to_bits() || !same,
        ))];
        lat.comm.sum(&mut wrong);
        report(
            "sum in rank order",
            wrong[0] == 0.0,
            format!("{} against {in_order} in rank order", x[0]),
        );

        // Gather: rank r passes r % 3 numbers, -0.0 among them; rank 0 gets them in order.
        let passed_by = |r: usize| -> Vec<f64> {
            (0..r % 3)
                .map(|i| if i == 1 { -0.0 } else { (10 * r + i) as f64 })
                .collect()
        };
        let got = lat.comm.gather(&passed_by(lat.comm.rank()));
        let gathered = !root
            || (got.len() == size
                && got.iter().enumerate().all(|(r, g)| {
                    let want = passed_by(r);
                    g.len() == want.len()
                        && g.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits())
                }));
        report("gather", gathered, format!("{size} processes to rank 0"));

        // A single-precision halo around a ring of the ranks, through the t faces.
        if size > 1 {
            let me = lat.comm.rank();
            let mut spec = HaloSpec {
                neighbour: [me; FACES],
                len: [0; FACES],
            };
            spec.neighbour[3] = (me + 1) % size;
            spec.neighbour[ND + 3] = (me + size - 1) % size;
            spec.len[3] = 3;
            spec.len[ND + 3] = 3;
            let mut ex = lat.comm.halo_f32(&spec);
            ex.send_buffer(3).fill(me as f32);
            ex.send_buffer(ND + 3).fill(me as f32 + 0.5);
            ex.start();
            ex.wait();
            // Forward face: what the next rank sent backward; backward face: what the one
            // before sent forward.
            let ok = ex.recv_buffer(3) == [spec.neighbour[3] as f32 + 0.5; 3]
                && ex.recv_buffer(ND + 3) == [spec.neighbour[ND + 3] as f32; 3];
            let mut wrong = [f64::from(u8::from(!ok))];
            lat.comm.sum(&mut wrong);
            report(
                "halo f32",
                wrong[0] == 0.0,
                format!("{} ranks with a wrong face", wrong[0]),
            );
        }

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
