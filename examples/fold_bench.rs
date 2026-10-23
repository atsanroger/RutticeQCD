//! Times the prolongation P of the domain-wall multigrid on the device and its adjoint
//! (`DeviceBasis::prolong`, `restrict`: the kernels `fold_prolong` and `fold_restrict_tiled`),
//! on the whole lattice and on the even sites, for P in half precision made from N random test
//! vectors (the HMC runs: 20 on 24⁴, Ls 8, aggregates of 4⁴). The printed checksums (of the fine
//! results) and norms (of the coarse ones) check that a change of the kernels leaves them alone.
//!
//! ```text
//! cargo run --release --features cuda-bridge --example fold_bench -- [L] [Ls] [N] [REPS]
//! ```

#[cfg(feature = "cuda-bridge")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[cfg(not(feature = "cuda-bridge"))]
fn main() {
    eprintln!("fold_bench needs the feature cuda-bridge");
}

#[cfg(feature = "cuda-bridge")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use ruttice::backend::cuda_bridge::{CudaMdwf, DeviceBasis, DeviceParities, DeviceSpinor};
    use ruttice::fermion::{FermionOp, MdwfParams, SplitParity};
    use ruttice::gauge::random_gauge;
    use ruttice::solver::preconditioner::multigrid::{random_even, Aggregation};
    use ruttice::{
        random_spinor, Boundary, EvenOddIndex, Field, Lattice, Parity, Spinor, Vector, ND,
    };
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (l, ls, n, reps) = (arg(0, 24), arg(1, 8), arg(2, 20), arg(3, 50));
    let lat = Lattice::single_node([l; ND]);
    let u = random_gauge(&lat, 7);
    let params = MdwfParams {
        ls,
        m5: 1.0,
        mass: 0.001,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut d = CudaMdwf::<f32>::new(&lat, params)?;
    d.set_gauge(&u)?;
    let mut pv = CudaMdwf::<f32>::new(&lat, params.pauli_villars())?;
    pv.set_gauge(&u)?;
    let index = EvenOddIndex::new(&lat);
    let y = DeviceSpinor::<f32>::from_host_half(&random_even(&lat, &index, ls, n, 11), &index)?;
    let agg = Aggregation::new(&lat, [4; ND]);
    let start = Instant::now();
    let basis = DeviceBasis::pv_eo(&d, &pv, &agg, &y, true)?;
    println!("P made in {:.2} s", start.elapsed().as_secs_f64());
    drop(y);

    // A coarse field that P has made: P† of a random fine field.
    // Ls random four-dimensional spinors as one five-dimensional one.
    let flat = random_spinor(&lat, ls, 3);
    let mut psi = Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
    psi.as_mut_slice().copy_from_slice(flat.as_slice());
    let whole = DeviceSpinor::<f32>::from_host(&psi)?;
    let mut even = DeviceParities.half_like(&whole, Parity::Even);
    DeviceParities.extract(&mut even, &whole, Parity::Even);
    let mut coarse = basis.coarse_field::<f32>(1);
    basis.restrict(&mut coarse, &whole);

    // Milliseconds per call, the best of three runs of `reps`.
    let time = |f: &mut dyn FnMut()| {
        f();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..reps {
                f();
            }
            // A norm waits for the device to finish.
            coarse.norm_sqr(&mut [0.0]);
            best = best.min(start.elapsed().as_secs_f64() / reps as f64);
        }
        best * 1e3
    };
    let mut out_whole = whole.zeros_like();
    let mut out_even = even.zeros_like();
    let t_pw = time(&mut || basis.prolong(&mut out_whole, &coarse));
    let t_pe = time(&mut || basis.prolong(&mut out_even, &coarse));
    let mut c_out = basis.coarse_field::<f32>(1);
    let t_rw = time(&mut || basis.restrict(&mut c_out, &whole));
    let norm = |c: &ruttice::backend::cuda_bridge::DeviceCoarse<f32>| {
        let mut x = [0.0];
        c.norm_sqr(&mut x);
        x[0]
    };
    let c_whole = norm(&c_out);
    let t_re = time(&mut || basis.restrict(&mut c_out, &even));
    println!("{l}^4 Ls {ls}, {n} test vectors, P in half precision:");
    println!(
        "  prolong  whole {t_pw:7.3} ms (checksum {:016x}), even {t_pe:7.3} ms (checksum {:016x})",
        out_whole.checksum()?,
        out_even.checksum()?
    );
    println!(
        "  restrict whole {t_rw:7.3} ms (|P† f|² {c_whole:.15e}), even {t_re:7.3} ms ({:.15e})",
        norm(&c_out)
    );
    Ok(())
}
