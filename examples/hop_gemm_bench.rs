//! Times the even-odd hop of the Möbius operator on the device with the product within the
//! site in the same pass (`EvenOdd::hop_then` with C⁻¹ after it: the kernel `hop_gemm`), in
//! single and in double precision, with and without the dagger, on random links. Nothing runs
//! on the CPU but the making of the fields. The printed checksums check that a change of the
//! kernel leaves the results alone; where it changes their rounding, the relative difference of
//! the single-precision result from the double-precision one should stay what it was.
//!
//! ```text
//! cargo run --release --features cuda-bridge --example hop_gemm_bench -- [L] [Ls] [REPS]
//! ```

#[cfg(feature = "cuda-bridge")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}

#[cfg(not(feature = "cuda-bridge"))]
fn main() {
    eprintln!("hop_gemm_bench needs the feature cuda-bridge");
}

#[cfg(feature = "cuda-bridge")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use ruttice::backend::cuda_bridge::{CudaMdwf, DeviceParities, DeviceReal, DeviceSpinor};
    use ruttice::fermion::{Dagger, EvenOdd, FermionOp, MdwfParams, SplitParity, Then};
    use ruttice::gauge::random_gauge;
    use ruttice::{
        random_spinor, Boundary, ConvertFrom, Field, Lattice, Parity, Spinor, Vector, C64, ND,
    };
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (l, ls, reps) = (arg(0, 24), arg(1, 8), arg(2, 200));
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
    let flat = random_spinor(&lat, ls, 2);
    let mut psi = Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
    psi.as_mut_slice().copy_from_slice(flat.as_slice());

    // Milliseconds per application (the best of three runs of `reps`), the result and its
    // checksum, for the operator in the precision T.
    fn time<T: DeviceReal>(
        op: &CudaMdwf<T>,
        psi: &DeviceSpinor<T>,
        dagger: Dagger,
        reps: usize,
    ) -> Result<(f64, DeviceSpinor<T>, u64), Box<dyn std::error::Error>> {
        let (even, odd) = (Parity::Even, Parity::Odd);
        let mut inp = DeviceParities.half_like(psi, odd);
        DeviceParities.extract(&mut inp, psi, odd);
        let (mut out, mut scratch) = (DeviceParities.half_like(psi, even), DeviceParities.half_like(psi, even));
        let mut work = op.even_odd_work();
        let mut apply = |out: &mut DeviceSpinor<T>| {
            op.hop_then(&mut work, out, &inp, even, dagger, Then::INVERSE, &mut scratch)
        };
        apply(&mut out);
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..reps {
                apply(&mut out);
            }
            // The norm waits for the device to finish.
            out.norm_sqr(&mut [0.0]);
            best = best.min(start.elapsed().as_secs_f64() / reps as f64);
        }
        let checksum = out.checksum()?;
        Ok((best * 1e3, out, checksum))
    }

    let mut double = CudaMdwf::<f64>::new(&lat, params)?;
    double.set_gauge(&u)?;
    let mut single = CudaMdwf::<f32>::new(&lat, params)?;
    single.set_gauge(&u)?;
    let psi64 = DeviceSpinor::<f64>::from_host(&psi)?;
    let psi32 = DeviceSpinor::<f32>::from_host(&psi)?;
    for dagger in [Dagger::No, Dagger::Yes] {
        let (t32, out32, c32) = time(&single, &psi32, dagger, reps)?;
        let (t64, out64, c64) = time(&double, &psi64, dagger, reps)?;
        let mut diff = out64.zeros_like();
        diff.convert_from(&out32);
        diff.axpy(&[C64::new(-1.0, 0.0)], &out64);
        let (mut dd, mut nn) = ([0.0], [0.0]);
        diff.norm_sqr(&mut dd);
        out64.norm_sqr(&mut nn);
        println!(
            "{l}^4 Ls {ls} {dagger:?}: single {t32:.3} ms (checksum {c32:016x}), double {t64:.3} ms (checksum {c64:016x}); |single - double|/|double| {:.2e}",
            (dd[0] / nn[0]).sqrt()
        );
    }
    Ok(())
}
