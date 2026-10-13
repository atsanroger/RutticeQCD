//! The operators on the device against those on the CPU, in time: one application of D, and
//! the even-odd solve with CG on the normal equations.
//!
//!   ruttice-bench-cuda <gauge file (Bridge++ text)> <L> [Ls]
//!
//! on an L⁴ lattice. The CPU runs on the threads of rayon (RAYON_NUM_THREADS).

use ruttice::backend::cuda_bridge::{
    CudaClover, CudaMdwf, CudaWilson, DeviceParities, DeviceSpinor,
};
use ruttice::fermion::{
    solve_even_odd, Clover, CloverParams, EvenOdd, FermionOp, Mdwf, MdwfParams, Variant, Wilson,
    WilsonParams,
};
use ruttice::io::read_bridge_text;
use ruttice::solver::{solve_cgne, Cg, Stop};
use ruttice::{random_spinor, Boundary, EvenOddIndex, Field, Half, Spinor, Vector};
use std::path::Path;
use std::time::Instant;

type Psi = Field<f64, Spinor>;
type HalfPsi = Field<f64, Spinor, Half>;

const APPLICATIONS: usize = 50;

/// Seconds of one application of D, after one to warm up.
fn time_mult<V: Vector, F: FermionOp<V>>(op: &F, psi: &V) -> f64 {
    let (mut work, mut out) = (op.work(), psi.zeros_like());
    op.mult(&mut work, &mut out, psi, Variant::D);
    let start = Instant::now();
    for _ in 0..APPLICATIONS {
        op.mult(&mut work, &mut out, psi, Variant::D);
    }
    // The norm waits for the device to finish.
    out.norm_sqr(&mut [0.0]);
    start.elapsed().as_secs_f64() / APPLICATIONS as f64
}

fn compare<C, G>(name: &str, cpu: &C, gpu: &G, index: &EvenOddIndex, psi: &Psi, stop: Stop)
where
    C: FermionOp<Psi> + EvenOdd<HalfPsi>,
    G: FermionOp<DeviceSpinor> + EvenOdd<DeviceSpinor>,
{
    let device_psi = DeviceSpinor::from_host(psi).expect("memory on the device");
    let sites = psi.lattice().num_sites() as f64;
    let gflops = |seconds: f64| cpu.flops_per_site() as f64 * sites / seconds * 1e-9;
    let (t_cpu, t_gpu) = (time_mult(cpu, psi), time_mult(gpu, &device_psi));
    println!(
        "{name}: D  CPU {:.3} ms ({:.1} GFlops)  device {:.3} ms ({:.1} GFlops)  ratio {:.1}",
        t_cpu * 1e3,
        gflops(t_cpu),
        t_gpu * 1e3,
        gflops(t_gpu),
        t_cpu / t_gpu
    );

    let mut x = psi.zeros_like();
    let start = Instant::now();
    let on_cpu = solve_even_odd(cpu, index, &mut x, psi, |schur, x_e, rhs| {
        solve_cgne(&mut Cg::new(stop), schur, x_e, rhs)
    });
    let t_cpu = start.elapsed().as_secs_f64();

    let mut device_x = device_psi.zeros_like();
    let start = Instant::now();
    let on_gpu = solve_even_odd(
        gpu,
        &DeviceParities,
        &mut device_x,
        &device_psi,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    );
    let t_gpu = start.elapsed().as_secs_f64();
    let iterations = |r: &Result<ruttice::solver::SolveInfo, ruttice::solver::SolveError>| match r {
        Ok(info) => format!("{} iterations", info.iterations),
        Err(e) => format!("not converged ({e:?})"),
    };
    println!(
        "{name}: even-odd CG  CPU {t_cpu:.2} s ({})  device {t_gpu:.2} s ({})  ratio {:.1}",
        iterations(&on_cpu),
        iterations(&on_gpu),
        t_cpu / t_gpu
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ruttice-bench-cuda <gauge file (Bridge++ text)> <L> [Ls]");
        std::process::exit(2);
    }
    let l: usize = args[2].parse().expect("L is a number");
    let ls: usize = args
        .get(3)
        .map_or(8, |s| s.parse().expect("Ls is a number"));
    let lat = ruttice::Lattice::single_node([l; 4]);
    let u = read_bridge_text(Path::new(&args[1]), &lat).expect("the gauge file");
    let boundary = [
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Antiperiodic,
    ];
    let stop = Stop::simple(1e-16, 5000);
    println!(
        "{l}^4, {} threads on the CPU, |r|/|b| < 1e-8",
        rayon::current_num_threads()
    );

    let params = WilsonParams {
        kappa: 0.13,
        boundary,
    };
    let mut cpu = Wilson::<f64>::new(&lat, params);
    cpu.set_gauge(&u).expect("the links");
    let mut gpu = CudaWilson::new(&lat, params).expect("memory on the device");
    gpu.set_gauge(&u).expect("the links");
    let index = cpu.even_odd_index().expect("even extents").clone();
    let psi = random_spinor(&lat, 1, 1);
    compare("Wilson", &cpu, &gpu, &index, &psi, stop);

    let params = CloverParams {
        kappa: 0.13,
        csw: 1.0,
        boundary,
    };
    let mut cpu = Clover::<f64>::new(&lat, params);
    cpu.set_gauge(&u).expect("the links");
    let mut gpu = CudaClover::new(&lat, params).expect("memory on the device");
    gpu.set_gauge(&u).expect("the links");
    compare("clover", &cpu, &gpu, &index, &psi, stop);

    let params = MdwfParams {
        ls,
        m5: 1.8,
        mass: 0.01,
        b: 1.5,
        c: 0.5,
        boundary,
    };
    let mut cpu = Mdwf::<f64>::new(&lat, params);
    cpu.set_gauge(&u).expect("the links");
    let mut gpu = CudaMdwf::new(&lat, params).expect("memory on the device");
    gpu.set_gauge(&u).expect("the links");
    let flat = random_spinor(&lat, ls, 2);
    let mut psi = Psi::with_shape(&lat, 24, ls, 1, 1);
    psi.as_mut_slice().copy_from_slice(flat.as_slice());
    compare(&format!("MDWF Ls={ls}"), &cpu, &gpu, &index, &psi, stop);
}
