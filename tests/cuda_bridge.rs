//! The CUDA kernels against the operators on the CPU (feature `cuda-bridge`; needs a GPU).
#![cfg(feature = "cuda-bridge")]

use ruttice::backend::cuda_bridge::{CudaClover, CudaMdwf, CudaWilson, DeviceSpinor};
use ruttice::fermion::{
    Bound, Clover, CloverParams, Dagger, EvenOdd, FermionOp, Mdwf, MdwfParams, Variant, Wilson,
    WilsonParams,
};
use ruttice::gauge::random_gauge;
use ruttice::solver::{Cg, Solver, Stop};
use ruttice::{
    extract_parity, random_spinor, Boundary, Field, Gauge, Half, Lattice, LinearFn, Normal, Parity,
    Sites, Spinor, Vector, C64,
};
use std::sync::Arc;

type Psi = Field<f64, Spinor>;

const PERIODIC: [Boundary; 4] = [Boundary::Periodic; 4];
const ANTIPERIODIC_T: [Boundary; 4] = [
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];
const MIXED: [Boundary; 4] = [
    Boundary::Antiperiodic,
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];

/// The largest difference between two fields.
fn worst<S: Sites>(a: &Field<f64, Spinor, S>, b: &Field<f64, Spinor, S>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
}

/// A random field with `ls` spinors per site.
fn random_fifth(lat: &Arc<Lattice>, ls: usize, seed: u64) -> Psi {
    let flat = random_spinor(lat, ls, seed);
    let mut psi = Psi::with_shape(lat, 24, ls, 1, 1);
    psi.as_mut_slice().copy_from_slice(flat.as_slice());
    psi
}

/// D, D† and γ5 D on the device are those on the CPU.
fn same_operator<C: FermionOp<Psi>, G: FermionOp<DeviceSpinor>>(
    cpu: &C,
    gpu: &G,
    psi: &Psi,
    what: &str,
) {
    let inp = DeviceSpinor::from_host(psi).unwrap();
    let mut out = inp.zeros_like();
    let (mut cpu_work, mut gpu_work) = (cpu.work(), gpu.work());
    for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
        let mut expected = psi.zeros_like();
        cpu.mult(&mut cpu_work, &mut expected, psi, variant);
        gpu.mult(&mut gpu_work, &mut out, &inp, variant);
        let d = worst(&out.to_host().unwrap(), &expected);
        assert!(d < 1e-13, "{what}: {variant:?} differs by {d:e}");
    }
}

/// CG on D†D on the device gives the solution CG on the CPU gives, and the CPU operator
/// agrees that it is a solution.
fn same_solution<C: FermionOp<Psi>, G: FermionOp<DeviceSpinor>>(
    cpu: &C,
    gpu: &G,
    b: &Psi,
    what: &str,
) {
    let stop = Stop::simple(1e-22, 20000);
    let mut expected = b.zeros_like();
    let mut cpu_normal = Normal::new(Bound::new(cpu, Variant::D), b);
    Cg::new(stop)
        .solve(&mut cpu_normal, None, &mut expected, b)
        .unwrap_or_else(|e| panic!("{what}: the CPU solve failed: {e:?}"));

    let device_b = DeviceSpinor::from_host(b).unwrap();
    let mut x = device_b.zeros_like();
    let mut gpu_normal = Normal::new(Bound::new(gpu, Variant::D), &device_b);
    let info = Cg::new(stop)
        .solve(&mut gpu_normal, None, &mut x, &device_b)
        .unwrap_or_else(|e| panic!("{what}: the solve on the device failed: {e:?}"));
    let x = x.to_host().unwrap();

    let mut r = b.zeros_like();
    cpu_normal.mult(&mut r, &x);
    r.aypx(&[C64::new(-1.0, 0.0)], b);
    let (mut rr, mut bb) = ([0.0], [0.0]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    assert!(
        rr[0] / bb[0] < 1e-20,
        "{what}: |b - D†D x|² / |b|² = {:e} after {} iterations",
        rr[0] / bb[0],
        info.iterations
    );
    let d = worst(&x, &expected);
    assert!(d < 1e-8, "{what}: the solutions differ by {d:e}");
    println!("{what}: {} iterations on the device", info.iterations);
}

/// The operations of `Vector` on the device are those on the CPU, on a lattice whose sites do
/// not fill the blocks of 32.
#[test]
fn linear_algebra_on_the_device_is_that_on_the_cpu() {
    let lat = Lattice::single_node([2, 2, 2, 6]);
    let (mut x, y) = (random_fifth(&lat, 3, 341), random_fifth(&lat, 3, 342));
    let (mut dx, dy) = (
        DeviceSpinor::from_host(&x).unwrap(),
        DeviceSpinor::from_host(&y).unwrap(),
    );
    let same = |dx: &DeviceSpinor, x: &Psi, what: &str| {
        let d = worst(&dx.to_host().unwrap(), x);
        assert!(d < 1e-14, "{what} differs by {d:e}");
    };
    let a = [C64::new(0.3, -1.1)];

    let (mut dot, mut device_dot) = ([C64::ZERO], [C64::ZERO]);
    x.dot(&y, &mut dot);
    dx.dot(&dy, &mut device_dot);
    assert!((dot[0] - device_dot[0]).norm_sqr().sqrt() < 1e-11 * dot[0].norm_sqr().sqrt());
    let (mut norm, mut device_norm) = ([0.0], [0.0]);
    x.norm_sqr(&mut norm);
    dx.norm_sqr(&mut device_norm);
    assert!((norm[0] - device_norm[0]).abs() < 1e-12 * norm[0]);

    x.axpy(&a, &y);
    dx.axpy(&a, &dy);
    same(&dx, &x, "axpy");
    x.aypx(&a, &y);
    dx.aypx(&a, &dy);
    same(&dx, &x, "aypx");
    x.scale(&[-0.7]);
    dx.scale(&[-0.7]);
    same(&dx, &x, "scale");
    x.copy_from(&y);
    dx.copy_from(&dy);
    same(&dx, &x, "copy_from");
    let (z, dz) = (x.zeros_like(), dx.zeros_like());
    same(&dz, &z, "zeros_like");
    dx.set_zero();
    same(&dx, &z, "set_zero");
    // The padding stayed zero: the norm counts the sites of the lattice only.
    dy.norm_sqr(&mut device_norm);
    y.norm_sqr(&mut norm);
    assert!((norm[0] - device_norm[0]).abs() < 1e-12 * norm[0]);
}

fn wilson_pair(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    params: WilsonParams,
) -> (Wilson<f64>, CudaWilson) {
    let mut cpu = Wilson::<f64>::new(lat, params);
    cpu.set_gauge(u).unwrap();
    let mut gpu = CudaWilson::new(lat, params).unwrap();
    gpu.set_gauge(u).unwrap();
    (cpu, gpu)
}

/// The Wilson operator, for the three kinds of boundary.
#[test]
fn wilson_on_the_device_is_wilson_on_the_cpu() {
    for (dims, boundary) in [
        ([4, 4, 4, 8], PERIODIC),
        ([4, 4, 4, 8], ANTIPERIODIC_T),
        ([6, 4, 2, 6], MIXED),
    ] {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 301);
        let params = WilsonParams {
            kappa: 0.123,
            boundary,
        };
        let (cpu, gpu) = wilson_pair(&lat, &u, params);
        let what = format!("Wilson {dims:?} {boundary:?}");
        same_operator(&cpu, &gpu, &random_spinor(&lat, 1, 302), &what);
    }
}

fn clover_pair(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    params: CloverParams,
) -> (Clover<f64>, CudaClover) {
    let mut cpu = Clover::<f64>::new(lat, params);
    cpu.set_gauge(u).unwrap();
    let mut gpu = CudaClover::new(lat, params).unwrap();
    gpu.set_gauge(u).unwrap();
    (cpu, gpu)
}

/// The clover operator, and T⁻¹ T = 1.
#[test]
fn clover_on_the_device_is_clover_on_the_cpu() {
    for (dims, csw, boundary) in [([4, 4, 4, 8], 1.0, PERIODIC), ([6, 4, 2, 6], 1.7, MIXED)] {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 331);
        let psi = random_spinor(&lat, 1, 332);
        let params = CloverParams {
            kappa: 0.123,
            csw,
            boundary,
        };
        let (cpu, gpu) = clover_pair(&lat, &u, params);
        let what = format!("clover {dims:?} csw {csw}");
        same_operator(&cpu, &gpu, &psi, &what);

        let inp = DeviceSpinor::from_host(&psi).unwrap();
        let (mut t, mut back) = (inp.zeros_like(), inp.zeros_like());
        gpu.site(&mut t, &inp, false);
        gpu.site(&mut back, &t, true);
        let d = worst(&back.to_host().unwrap(), &psi);
        assert!(d < 1e-12, "{what}: T⁻¹ T differs from 1 by {d:e}");
    }
}

/// The lattices and parameters of the domain-wall tests: Shamir and Möbius, with a mass, an Ls
/// that does not fill the blocks of four of the kernels, and the three kinds of boundary.
fn mdwf_cases() -> Vec<([usize; 4], MdwfParams)> {
    [
        ([4, 4, 4, 8], 8, 1.0, 0.0, 1.0, 0.05, PERIODIC),
        ([4, 4, 4, 8], 6, 1.5, 0.5, 1.8, 0.02, ANTIPERIODIC_T),
        ([6, 4, 2, 6], 5, 2.0, 1.0, 1.4, 0.3, MIXED),
        ([2, 4, 4, 4], 2, 1.5, 0.5, 1.8, 1.0, ANTIPERIODIC_T),
    ]
    .into_iter()
    .map(|(dims, ls, b, c, m5, mass, boundary)| {
        let params = MdwfParams {
            ls,
            m5,
            mass,
            b,
            c,
            boundary,
        };
        (dims, params)
    })
    .collect()
}

fn mdwf_pair(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    params: MdwfParams,
) -> (Mdwf<f64>, CudaMdwf) {
    let mut cpu = Mdwf::<f64>::new(lat, params);
    cpu.set_gauge(u).unwrap();
    let mut gpu = CudaMdwf::new(lat, params).unwrap();
    gpu.set_gauge(u).unwrap();
    (cpu, gpu)
}

#[test]
fn mdwf_on_the_device_is_mdwf_on_the_cpu() {
    for (dims, params) in mdwf_cases() {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 311);
        let (cpu, gpu) = mdwf_pair(&lat, &u, params);
        let psi = random_fifth(&lat, params.ls, 312);
        same_operator(&cpu, &gpu, &psi, &format!("MDWF {dims:?} {params:?}"));
    }
}

/// The even-odd blocks of the domain-wall operator: the hop to each parity and its adjoint,
/// C⁻¹ and C†⁻¹, and the Schur factor made of them.
#[test]
fn mdwf_even_odd_on_the_device_is_that_on_the_cpu() {
    for (dims, params) in mdwf_cases() {
        let what = format!("MDWF {dims:?} {params:?}");
        let ls = params.ls;
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 321);
        let psi = random_fifth(&lat, ls, 322);
        let (cpu, gpu) = mdwf_pair(&lat, &u, params);
        let index = cpu.even_odd_index().unwrap().clone();
        let mut work = cpu.even_odd_work();
        let mut gpu_work = None;
        let half = |parity| {
            let mut f = Field::<f64, Spinor, Half>::with_shape(&lat, 24, ls, 1, 1);
            f.set_parity(parity);
            f
        };
        let device = |parity| DeviceSpinor::zeros_half(&lat, ls, parity).unwrap();

        for to in [Parity::Even, Parity::Odd] {
            let from = to.flip();
            let mut inp = half(from);
            extract_parity(&mut inp, &psi, &index, from);
            let device_inp = DeviceSpinor::from_host_half(&inp, &index).unwrap();
            for dagger in [Dagger::No, Dagger::Yes] {
                let mut expected = half(to);
                cpu.hop(&mut work, &mut expected, &inp, to, dagger);
                let mut out = device(to);
                gpu.hop(&mut gpu_work, &mut out, &device_inp, to, dagger);
                let d = worst(&out.to_host_half(&index).unwrap(), &expected);
                assert!(
                    d < 1e-13,
                    "{what}: the hop to {to:?} ({dagger:?}) differs by {d:e}"
                );

                let mut expected = half(from);
                cpu.diag_inv(&mut work, &mut expected, &inp, from, dagger);
                let mut out = device(from);
                gpu.site_inv(&mut out, &device_inp, dagger);
                let d = worst(&out.to_host_half(&index).unwrap(), &expected);
                assert!(
                    d < 1e-13,
                    "{what}: C⁻¹ ({dagger:?}) on {from:?} differs by {d:e}"
                );
            }
        }

        // 1 − D_eo C_o⁻¹ D_oe C_e⁻¹
        let (even, odd) = (Parity::Even, Parity::Odd);
        let mut inp = half(even);
        extract_parity(&mut inp, &psi, &index, even);
        let (mut e, mut o1, mut o2) = (half(even), half(odd), half(odd));
        cpu.diag_inv(&mut work, &mut e, &inp, even, Dagger::No);
        cpu.hop(&mut work, &mut o1, &e, odd, Dagger::No);
        cpu.diag_inv(&mut work, &mut o2, &o1, odd, Dagger::No);
        cpu.hop(&mut work, &mut e, &o2, even, Dagger::No);
        let mut expected = inp.zeros_like();
        expected.copy_from(&inp);
        expected.axpy(&[C64::new(-1.0, 0.0)], &e);
        let device_inp = DeviceSpinor::from_host_half(&inp, &index).unwrap();
        let mut out = device(even);
        gpu.schur_factor(&mut out, &device_inp, &mut [device(even), device(odd)]);
        let d = worst(&out.to_host_half(&index).unwrap(), &expected);
        assert!(d < 1e-12, "{what}: the Schur factor differs by {d:e}");
    }
}

/// The solvers run on the device as they are: CG on D†D for the three operators.
#[test]
fn conjugate_gradient_on_the_device_solves_as_on_the_cpu() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 351);
    let b = random_spinor(&lat, 1, 352);

    let wilson = WilsonParams {
        kappa: 0.1,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = wilson_pair(&lat, &u, wilson);
    same_solution(&cpu, &gpu, &b, "Wilson");

    let clover = CloverParams {
        kappa: 0.1,
        csw: 1.0,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = clover_pair(&lat, &u, clover);
    same_solution(&cpu, &gpu, &b, "clover");

    let mdwf = MdwfParams {
        ls: 6,
        m5: 1.8,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = mdwf_pair(&lat, &u, mdwf);
    same_solution(&cpu, &gpu, &random_fifth(&lat, mdwf.ls, 353), "MDWF");
}
