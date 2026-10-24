//! The CUDA kernels against the operators on the CPU (feature `cuda-bridge`; needs a GPU).
#![cfg(feature = "cuda-bridge")]

use ruttice::backend::cuda_bridge::{
    CudaClover, CudaMdwf, CudaWilson, DeviceParities, DeviceReal, DeviceSpinor,
};
use ruttice::fermion::{
    solve_even_odd, Bound, Clover, CloverParams, Dagger, EoForm, EvenOdd, FermionOp, Mdwf,
    MdwfParams, PvEvenOdd, PvRatio, Schur, Variant, Wilson, WilsonParams,
};
use ruttice::gauge::random_gauge;
use ruttice::solver::{solve_cgne, Cg, Convert, Refine, Solve, Solver, Stop};
use ruttice::{
    extract_parity, random_spinor, Boundary, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn,
    LinearOp, Normal, Parity, Sites, Spinor, Vector, C64,
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

/// D, D† and γ5 D on the device are those on the CPU, to `tolerance` (the device may be in
/// single precision).
fn same_operator<T: DeviceReal, C: FermionOp<Psi>, G: FermionOp<DeviceSpinor<T>>>(
    cpu: &C,
    gpu: &G,
    psi: &Psi,
    tolerance: f64,
    what: &str,
) {
    let inp = DeviceSpinor::<T>::from_host(psi).unwrap();
    let mut out = inp.zeros_like();
    let (mut cpu_work, mut gpu_work) = (cpu.work(), gpu.work());
    for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
        let mut expected = psi.zeros_like();
        cpu.mult(&mut cpu_work, &mut expected, psi, variant);
        gpu.mult(&mut gpu_work, &mut out, &inp, variant);
        let d = worst(&out.to_host().unwrap(), &expected);
        assert!(d < tolerance, "{what}: {variant:?} differs by {d:e}");
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

/// Links in SU(3) are kept as two rows on the device and others whole: the operators agree
/// with the CPU on both.
#[test]
fn links_outside_su3_on_the_device() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut u = random_gauge(&lat, 401);
    assert!(ruttice::fermion::links_in_su3(&u));
    for (k, v) in u.as_mut_slice().iter_mut().enumerate() {
        *v *= 1.0 + 0.05 * ((k % 7) as f64 - 3.0);
    }
    assert!(!ruttice::fermion::links_in_su3(&u));
    let wilson = WilsonParams {
        kappa: 0.123,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = wilson_pair(&lat, &u, wilson);
    same_operator(
        &cpu,
        &gpu,
        &random_spinor(&lat, 1, 402),
        1e-13,
        "Wilson, links not in SU(3)",
    );
    let (_, params) = mdwf_cases().swap_remove(1);
    let (cpu, gpu) = mdwf_pair(&lat, &u, params);
    let psi = random_fifth(&lat, params.ls, 403);
    same_operator(&cpu, &gpu, &psi, 1e-12, "MDWF, links not in SU(3)");
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
        same_operator(&cpu, &gpu, &random_spinor(&lat, 1, 302), 1e-13, &what);
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
        same_operator(&cpu, &gpu, &psi, 1e-13, &what);

        let inp = DeviceSpinor::from_host(&psi).unwrap();
        let (mut t, mut back) = (inp.zeros_like(), inp.zeros_like());
        gpu.site(&mut t, &inp, false);
        gpu.site(&mut back, &t, true);
        let d = worst(&back.to_host().unwrap(), &psi);
        assert!(d < 1e-12, "{what}: T⁻¹ T differs from 1 by {d:e}");
    }
}

/// The lattices and parameters of the domain-wall tests: Shamir and Möbius, with a mass, small
/// and large Ls (the kernels shape their blocks of threads by it), odd Ls, and the three kinds
/// of boundary.
fn mdwf_cases() -> Vec<([usize; 4], MdwfParams)> {
    [
        ([4, 4, 4, 8], 8, 1.0, 0.0, 1.0, 0.05, PERIODIC),
        ([4, 4, 4, 8], 6, 1.5, 0.5, 1.8, 0.02, ANTIPERIODIC_T),
        ([6, 4, 2, 6], 5, 2.0, 1.0, 1.4, 0.3, MIXED),
        ([2, 4, 4, 4], 2, 1.5, 0.5, 1.8, 1.0, ANTIPERIODIC_T),
        ([4, 4, 4, 4], 16, 1.5, 0.5, 1.8, 0.01, ANTIPERIODIC_T),
        ([4, 2, 4, 4], 12, 2.0, 1.0, 1.4, 0.1, PERIODIC),
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
        same_operator(
            &cpu,
            &gpu,
            &psi,
            1e-13,
            &format!("MDWF {dims:?} {params:?}"),
        );
    }
}

type HalfPsi = Field<f64, Spinor, Half>;

/// The even-odd blocks on the device are those on the CPU: the hop to each parity, the block
/// within a site and its inverse, each with its adjoint, and the Schur complement made of them.
fn same_blocks<C: EvenOdd<HalfPsi>, G: EvenOdd<DeviceSpinor>>(
    cpu: &C,
    gpu: &G,
    index: &EvenOddIndex,
    psi: &Psi,
    what: &str,
) {
    let (lat, ls, nv) = (psi.lattice(), psi.ls(), psi.num_vectors());
    let half = |parity| {
        let mut f = HalfPsi::with_shape(lat, 24, ls, nv, 1);
        f.set_parity(parity);
        f
    };
    let device = |parity| DeviceSpinor::from_host_half(&half(parity), index).unwrap();
    let (mut work, mut gpu_work) = (cpu.even_odd_work(), gpu.even_odd_work());
    let same = |out: &DeviceSpinor, expected: &HalfPsi, which: &str| {
        let d = worst(&out.to_host_half(index).unwrap(), expected);
        assert!(d < 1e-13, "{what}: {which} differs by {d:e}");
    };

    for to in [Parity::Even, Parity::Odd] {
        let from = to.flip();
        let mut inp = half(from);
        extract_parity(&mut inp, psi, index, from);
        let device_inp = DeviceSpinor::from_host_half(&inp, index).unwrap();
        for dagger in [Dagger::No, Dagger::Yes] {
            let (mut expected, mut out) = (half(to), device(to));
            cpu.hop(&mut work, &mut expected, &inp, to, dagger);
            gpu.hop(&mut gpu_work, &mut out, &device_inp, to, dagger);
            same(&out, &expected, &format!("the hop to {to:?} ({dagger:?})"));

            let (mut expected, mut out) = (half(from), device(from));
            cpu.diag(&mut work, &mut expected, &inp, from, dagger);
            gpu.diag(&mut gpu_work, &mut out, &device_inp, from, dagger);
            same(
                &out,
                &expected,
                &format!("the block on {from:?} ({dagger:?})"),
            );
            cpu.diag_inv(&mut work, &mut expected, &inp, from, dagger);
            gpu.diag_inv(&mut gpu_work, &mut out, &device_inp, from, dagger);
            same(
                &out,
                &expected,
                &format!("its inverse on {from:?} ({dagger:?})"),
            );
        }
    }

    let (even, odd) = (Parity::Even, Parity::Odd);
    let mut inp = half(even);
    extract_parity(&mut inp, psi, index, even);
    let device_inp = DeviceSpinor::from_host_half(&inp, index).unwrap();
    let mut cpu_schur = Schur::new(cpu, even, &inp, &half(odd));
    let mut gpu_schur = Schur::new(gpu, even, &device_inp, &device(odd));
    let (mut expected, mut out) = (half(even), device(even));
    cpu_schur.mult(&mut expected, &inp);
    gpu_schur.mult(&mut out, &device_inp);
    let d = worst(&out.to_host_half(index).unwrap(), &expected);
    assert!(d < 1e-12, "{what}: the Schur complement differs by {d:e}");
    cpu_schur.mult_dag(&mut expected, &inp);
    gpu_schur.mult_dag(&mut out, &device_inp);
    let d = worst(&out.to_host_half(index).unwrap(), &expected);
    assert!(
        d < 1e-12,
        "{what}: the adjoint of the Schur complement differs by {d:e}"
    );
}

/// The even-odd solve on the device (CG on the normal equations of the Schur complement)
/// gives a solution of D x = b, as the operator on the CPU sees it.
fn solves_even_odd<C: FermionOp<Psi>, G: EvenOdd<DeviceSpinor>>(
    cpu: &C,
    gpu: &G,
    b: &Psi,
    what: &str,
) {
    let stop = Stop::simple(1e-22, 20000);
    let device_b = DeviceSpinor::from_host(b).unwrap();
    let mut x = device_b.zeros_like();
    let info = solve_even_odd(
        gpu,
        &DeviceParities,
        &mut x,
        &device_b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap_or_else(|e| panic!("{what}: the even-odd solve on the device failed: {e:?}"));
    let x = x.to_host().unwrap();

    let nv = b.num_vectors();
    let mut r = b.zeros_like();
    cpu.mult(&mut cpu.work(), &mut r, &x, Variant::D);
    r.aypx(&vec![C64::new(-1.0, 0.0); nv], b);
    let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    let residual = rr.iter().zip(&bb).fold(0.0f64, |m, (r, b)| m.max(r / b));
    assert!(
        residual < 1e-18,
        "{what}: |b - D x|² / |b|² = {residual:e} after {} iterations",
        info.iterations
    );
    println!(
        "{what}: {} iterations of the even-odd solve on the device",
        info.iterations
    );
}

#[test]
fn even_odd_blocks_on_the_device_are_those_on_the_cpu() {
    for (dims, boundary) in [([4, 4, 4, 8], ANTIPERIODIC_T), ([6, 4, 2, 6], MIXED)] {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 361);
        let psi = random_spinor(&lat, 1, 362);

        let wilson = WilsonParams {
            kappa: 0.123,
            boundary,
        };
        let (cpu, gpu) = wilson_pair(&lat, &u, wilson);
        let index = cpu.even_odd_index().unwrap().clone();
        same_blocks(&cpu, &gpu, &index, &psi, &format!("Wilson {dims:?}"));

        let clover = CloverParams {
            kappa: 0.123,
            csw: 1.3,
            boundary,
        };
        let (cpu, gpu) = clover_pair(&lat, &u, clover);
        same_blocks(&cpu, &gpu, &index, &psi, &format!("clover {dims:?}"));
    }
    for (dims, params) in mdwf_cases() {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 321);
        let psi = random_fifth(&lat, params.ls, 322);
        let (cpu, gpu) = mdwf_pair(&lat, &u, params);
        let index = cpu.even_odd_index().unwrap().clone();
        same_blocks(
            &cpu,
            &gpu,
            &index,
            &psi,
            &format!("MDWF {dims:?} {params:?}"),
        );
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

/// The even-odd solve runs on the device as it is, for the three operators.
#[test]
fn even_odd_solve_on_the_device_solves() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 351);
    let b = random_spinor(&lat, 1, 352);

    let wilson = WilsonParams {
        kappa: 0.1,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = wilson_pair(&lat, &u, wilson);
    solves_even_odd(&cpu, &gpu, &b, "Wilson");

    let clover = CloverParams {
        kappa: 0.1,
        csw: 1.0,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = clover_pair(&lat, &u, clover);
    solves_even_odd(&cpu, &gpu, &b, "clover");

    let mdwf = MdwfParams {
        ls: 6,
        m5: 1.8,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = mdwf_pair(&lat, &u, mdwf);
    solves_even_odd(&cpu, &gpu, &random_fifth(&lat, mdwf.ls, 353), "MDWF");
}

/// The operators in single precision on the device are those in double precision on the CPU,
/// as far as single precision goes.
#[test]
fn single_precision_on_the_device() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 371);
    let psi = random_spinor(&lat, 1, 372);

    let params = WilsonParams {
        kappa: 0.123,
        boundary: ANTIPERIODIC_T,
    };
    let mut cpu = Wilson::<f64>::new(&lat, params);
    cpu.set_gauge(&u).unwrap();
    let mut gpu = CudaWilson::<f32>::new(&lat, params).unwrap();
    gpu.set_gauge(&u).unwrap();
    same_operator(&cpu, &gpu, &psi, 2e-6, "Wilson in single precision");

    let params = CloverParams {
        kappa: 0.123,
        csw: 1.3,
        boundary: ANTIPERIODIC_T,
    };
    let mut cpu = Clover::<f64>::new(&lat, params);
    cpu.set_gauge(&u).unwrap();
    let mut gpu = CudaClover::<f32>::new(&lat, params).unwrap();
    gpu.set_gauge(&u).unwrap();
    same_operator(&cpu, &gpu, &psi, 2e-6, "clover in single precision");

    let (_, params) = mdwf_cases().swap_remove(1);
    let mut cpu = Mdwf::<f64>::new(&lat, params);
    cpu.set_gauge(&u).unwrap();
    let mut gpu = CudaMdwf::<f32>::new(&lat, params).unwrap();
    gpu.set_gauge(&u).unwrap();
    let psi = random_fifth(&lat, params.ls, 373);
    same_operator(&cpu, &gpu, &psi, 5e-6, "MDWF in single precision");
}

/// The five-dimensional source of a four-dimensional one and the four-dimensional fields of a
/// five-dimensional solution, made on the device, are those of the CPU.
#[test]
fn four_dimensional_fields_of_mdwf_on_the_device() {
    let lat = Lattice::single_node([4, 4, 2, 6]);
    let u = random_gauge(&lat, 421);
    let (ls, nv) = (6, 3);
    let params = MdwfParams {
        ls,
        m5: 1.8,
        mass: 0.2,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = mdwf_pair(&lat, &u, params);
    let eta = random_spinor(&lat, nv, 422);
    let mut b = Psi::with_shape(&lat, 24, ls, nv, 1);
    cpu.source_5d(&mut cpu.work(), &mut b, &eta);
    let device_eta = DeviceSpinor::from_host(&eta).unwrap();
    let mut device_b = DeviceSpinor::from_host(&b).unwrap().zeros_like();
    gpu.source_5d(&mut gpu.work(), &mut device_b, &device_eta);
    let d = worst(&device_b.to_host().unwrap(), &b);
    assert!(d < 1e-13, "the five-dimensional source differs by {d:e}");

    let mut psi = Psi::with_shape(&lat, 24, ls, nv, 1);
    psi.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, ls * nv, 423).as_slice());
    let device_psi = DeviceSpinor::from_host(&psi).unwrap();
    let (mut q, mut device_q) = (eta.zeros_like(), device_eta.zeros_like());
    cpu.propagator_4d(&mut q, &psi);
    gpu.propagator_4d(&mut device_q, &device_psi);
    let d = worst(&device_q.to_host().unwrap(), &q);
    assert!(
        d < 1e-14,
        "the four-dimensional quark field differs by {d:e}"
    );
    cpu.midpoint_4d(&mut q, &psi);
    gpu.midpoint_4d(&mut device_q, &device_psi);
    let d = worst(&device_q.to_host().unwrap(), &q);
    assert!(d < 1e-14, "the midpoint field differs by {d:e}");
}

/// The matrix products of the fifth dimension in TF32 are those of single precision to the
/// ten bits of mantissa of TF32, and not better: they are another kernel.
#[test]
fn tf32_products_on_the_device() {
    use ruttice::fermion::SplitParity;
    for ls in [4, 8, 12, 16] {
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 391);
        let params = MdwfParams {
            ls,
            m5: 1.8,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary: ANTIPERIODIC_T,
        };
        let mut gpu = CudaMdwf::<f32>::new(&lat, params).unwrap();
        gpu.set_gauge(&u).unwrap();
        let psi = DeviceSpinor::<f32>::from_host(&random_fifth(&lat, ls, 392)).unwrap();
        let mut inp = DeviceParities.half_like(&psi, Parity::Odd);
        DeviceParities.extract(&mut inp, &psi, Parity::Odd);
        let (mut exact, mut rounded) = (inp.zeros_like(), inp.zeros_like());
        let mut work = gpu.even_odd_work();
        for dagger in [Dagger::No, Dagger::Yes] {
            gpu.diag_inv(&mut work, &mut exact, &inp, Parity::Odd, dagger);
            gpu.set_tf32(true);
            gpu.diag_inv(&mut work, &mut rounded, &inp, Parity::Odd, dagger);
            gpu.set_tf32(false);
            let (mut norm, mut difference) = ([0.0f64], [0.0f64]);
            exact.norm_sqr(&mut norm);
            rounded.axpy(&[C64::new(-1.0, 0.0)], &exact);
            rounded.norm_sqr(&mut difference);
            let relative = (difference[0] / norm[0]).sqrt();
            assert!(
                (1e-6..2e-3).contains(&relative),
                "Ls = {ls}: C⁻¹ in TF32 differs from single precision by {relative:e}"
            );
        }
    }
}

/// Mixed precision on the device: defect correction in double precision around CG in single
/// precision reaches the residual of double precision, as the operator on the CPU sees it.
#[test]
fn mixed_precision_solve_on_the_device() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 381);
    let params = MdwfParams {
        ls: 6,
        m5: 1.8,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = mdwf_pair(&lat, &u, params);
    let mut single = CudaMdwf::<f32>::new(&lat, params).unwrap();
    single.set_gauge(&u).unwrap();
    let b = random_fifth(&lat, params.ls, 382);

    let device_b = DeviceSpinor::<f64>::from_host(&b).unwrap();
    let like = DeviceSpinor::<f32>::zeros_fifth(&lat, params.ls).unwrap();
    let mut a = Normal::new(Bound::new(&gpu, Variant::D), &device_b);
    let inner = Solve::new(
        Cg::new(Stop::simple(1e-8, 2000)),
        Normal::new(Bound::new(&single, Variant::D), &like),
    );
    let mut m = Convert::new(inner, &like);
    let mut x = device_b.zeros_like();
    let info = Refine::new(Stop::simple(1e-22, 50))
        .solve(&mut a, Some(&mut m), &mut x, &device_b)
        .unwrap_or_else(|e| panic!("the mixed precision solve failed: {e:?}"));
    let x = x.to_host().unwrap();

    let mut cpu_normal = Normal::new(Bound::new(&cpu, Variant::D), &b);
    let mut r = b.zeros_like();
    cpu_normal.mult(&mut r, &x);
    r.aypx(&[C64::new(-1.0, 0.0)], &b);
    let (mut rr, mut bb) = ([0.0], [0.0]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    assert!(
        rr[0] / bb[0] < 1e-20,
        "|b - D†D x|² / |b|² = {:e} after {} corrections",
        rr[0] / bb[0],
        info.iterations
    );
    println!(
        "mixed precision: {} corrections in double precision",
        info.iterations
    );
}

/// Neff's α on the device, and the Pauli-Villars preconditioned operator on the even sites in
/// both forms, are those on the CPU.
#[test]
fn pauli_villars_preconditioned_operator_on_the_device() {
    for alpha in [1.0, 0.6] {
        for (dims, params) in mdwf_cases().into_iter().take(3) {
            let what = format!("MDWF {dims:?} {params:?} α {alpha}");
            let lat = Lattice::single_node(dims);
            let u = random_gauge(&lat, 391);
            let psi = random_fifth(&lat, params.ls, 392);
            let (mut cpu, mut gpu) = mdwf_pair(&lat, &u, params);
            let (mut cpu_pv, mut gpu_pv) = mdwf_pair(&lat, &u, params.pauli_villars());
            cpu.set_alpha(alpha);
            cpu_pv.set_alpha(alpha);
            gpu.set_alpha(alpha);
            gpu_pv.set_alpha(alpha);
            let index = cpu.even_odd_index().unwrap().clone();
            same_operator(&cpu, &gpu, &psi, 1e-13, &what);
            same_blocks(&cpu, &gpu, &index, &psi, &what);

            let mut inp = HalfPsi::with_shape(&lat, 24, params.ls, 1, 1);
            extract_parity(&mut inp, &psi, &index, Parity::Even);
            let device_inp = DeviceSpinor::from_host_half(&inp, &index).unwrap();
            let device_odd = DeviceSpinor::zeros_half(&lat, params.ls, Parity::Odd).unwrap();
            for form in [EoForm::Right, EoForm::Left] {
                let mut cpu_a = PvEvenOdd::new(&cpu, &cpu_pv, form, &inp);
                let mut gpu_a = PvRatio::new(&gpu, &gpu_pv, form, &device_inp, &device_odd);
                let (mut expected, mut out) = (inp.zeros_like(), device_inp.zeros_like());
                cpu_a.mult(&mut expected, &inp);
                gpu_a.mult(&mut out, &device_inp);
                let d = worst(&out.to_host_half(&index).unwrap(), &expected);
                assert!(d < 1e-12, "{what}: Ã ({form:?}) differs by {d:e}");
                cpu_a.mult_dag(&mut expected, &inp);
                gpu_a.mult_dag(&mut out, &device_inp);
                let d = worst(&out.to_host_half(&index).unwrap(), &expected);
                assert!(d < 1e-12, "{what}: Ã† ({form:?}) differs by {d:e}");
            }
        }
    }
}

/// The fused operations of `Vector` on the device (one pass, the sums finished on the device)
/// against those of the CPU, for more vectors than one launch takes coefficients for (16).
#[test]
fn fused_vector_operations_on_the_device() {
    let lat = Lattice::single_node([4, 4, 2, 6]);
    let (ls, nv) = (4, 19);
    let many = |seed| {
        let flat = random_spinor(&lat, ls * nv, seed);
        let mut psi = Psi::with_shape(&lat, 24, ls, nv, 1);
        psi.as_mut_slice().copy_from_slice(flat.as_slice());
        psi
    };
    let (mut s, x, y, z) = (many(501), many(502), many(503), many(504));
    let up = |f: &Psi| DeviceSpinor::from_host(f).unwrap();
    let (mut ds, dx, dy, dz) = (up(&s), up(&x), up(&y), up(&z));
    let a: Vec<C64> = (0..nv)
        .map(|v| C64::new(0.3 - 0.1 * v as f64, 1.1 - 0.05 * v as f64))
        .collect();
    let b: Vec<C64> = (0..nv)
        .map(|v| C64::new(-0.7 + 0.04 * v as f64, 0.2 * v as f64 - 1.0))
        .collect();
    let close = |what: &str, cpu: &[f64], gpu: &[f64]| {
        for (v, (c, g)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (c - g).abs() <= 1e-12 * c.abs().max(1.0),
                "{what}, vector {v}: {c:e} on the CPU, {g:e} on the device"
            );
        }
    };
    let close_c = |what: &str, cpu: &[C64], gpu: &[C64]| {
        let re: Vec<f64> = cpu.iter().map(|z| z.re).collect();
        let im: Vec<f64> = cpu.iter().map(|z| z.im).collect();
        close(
            &format!("{what} (re)"),
            &re,
            &gpu.iter().map(|z| z.re).collect::<Vec<_>>(),
        );
        close(
            &format!("{what} (im)"),
            &im,
            &gpu.iter().map(|z| z.im).collect::<Vec<_>>(),
        );
    };
    let same = |what: &str, ds: &DeviceSpinor, s: &Psi| {
        let d = worst(&ds.to_host().unwrap(), s);
        assert!(d < 1e-13, "{what} differs by {d:e}");
    };

    let (mut n_cpu, mut n_gpu) = (vec![0.0; nv], vec![0.0; nv]);
    s.axpy_norm_sqr(&a, &x, &mut n_cpu);
    ds.axpy_norm_sqr(&a, &dx, &mut n_gpu);
    same("axpy_norm_sqr", &ds, &s);
    close("axpy_norm_sqr", &n_cpu, &n_gpu);

    let (mut d_cpu, mut d_gpu) = (vec![C64::ZERO; nv], vec![C64::ZERO; nv]);
    s.xpay_norm_sqr_dot(&x, &b, &y, &mut n_cpu, &z, &mut d_cpu);
    ds.xpay_norm_sqr_dot(&dx, &b, &dy, &mut n_gpu, &dz, &mut d_gpu);
    same("xpay_norm_sqr_dot", &ds, &s);
    close("xpay_norm_sqr_dot |s|²", &n_cpu, &n_gpu);
    close_c("xpay_norm_sqr_dot <z|s>", &d_cpu, &d_gpu);

    s.xpay(&y, &a, &z);
    ds.xpay(&dy, &a, &dz);
    same("xpay", &ds, &s);

    s.axpy2(&a, &x, &b, &y);
    ds.axpy2(&a, &dx, &b, &dy);
    same("axpy2", &ds, &s);

    s.axpy_aypx(&b, &z, &a, &x);
    ds.axpy_aypx(&b, &dz, &a, &dx);
    same("axpy_aypx", &ds, &s);

    let (mut m_cpu, mut m_gpu) = (vec![0.0; nv], vec![0.0; nv]);
    s.dot_norms(&y, &mut d_cpu, &mut n_cpu, &mut m_cpu);
    ds.dot_norms(&dy, &mut d_gpu, &mut n_gpu, &mut m_gpu);
    close_c("dot_norms <s|y>", &d_cpu, &d_gpu);
    close("dot_norms |s|²", &n_cpu, &n_gpu);
    close("dot_norms |y|²", &m_cpu, &m_gpu);

    // The plain reductions, through the same two stages.
    s.dot(&z, &mut d_cpu);
    ds.dot(&dz, &mut d_gpu);
    close_c("dot", &d_cpu, &d_gpu);
    s.norm_sqr(&mut n_cpu);
    ds.norm_sqr(&mut n_gpu);
    close("norm_sqr", &n_cpu, &n_gpu);

    // A change of basis of more vectors than the in-place kernel takes (32).
    {
        use ruttice::Subspace;
        let nv = 37;
        let flat = random_spinor(&lat, ls * nv, 505);
        let mut x = Psi::with_shape(&lat, 24, ls, nv, 1);
        x.as_mut_slice().copy_from_slice(flat.as_slice());
        let mut dx = up(&x);
        let c: Vec<C64> = (0..nv * nv)
            .map(|k| C64::new((0.37 * k as f64).sin(), (0.11 * k as f64).cos()))
            .collect();
        let mut expected = x.zeros_like();
        for i in 0..nv {
            expected.axpy(&c[i * nv..(i + 1) * nv], &x.select(&vec![i; nv]));
        }
        dx.rotate(&c);
        let d = worst(&dx.to_host().unwrap(), &expected);
        assert!(
            d < 1e-12,
            "the change of basis of {nv} vectors differs by {d:e}"
        );

        // The Gram matrix with another set (tiles of 4: neither 37 nor 6 a multiple) and of
        // the set with itself, against the pairwise inner products on the CPU.
        let ny = 6;
        let flat = random_spinor(&lat, ls * ny, 506);
        let mut y = Psi::with_shape(&lat, 24, ls, ny, 1);
        y.as_mut_slice().copy_from_slice(flat.as_slice());
        let (dx, dy) = (up(&x), up(&y));
        for (a, b, da, db) in [(&x, &y, &dx, &dy), (&y, &y, &dy, &dy)] {
            let (na, nb) = (a.num_vectors(), b.num_vectors());
            let mut expected = vec![C64::ZERO; na * nb];
            let mut d = [C64::ZERO];
            for i in 0..na {
                for j in 0..nb {
                    a.select(&[i]).dot(&b.select(&[j]), &mut d);
                    expected[i * nb + j] = d[0];
                }
            }
            close_c(&format!("gram {na} x {nb}"), &expected, &da.gram(db));
        }
    }
}

/// Several vectors in one field on the device: the linear algebra with a coefficient for each,
/// taking some out and putting them back, the operators, and a solve of all of them together.
#[test]
fn several_vectors_in_a_field_on_the_device() {
    let lat = Lattice::single_node([4, 4, 2, 6]);
    let u = random_gauge(&lat, 411);
    let (ls, nv) = (4, 3);
    let many = |seed| {
        let flat = random_spinor(&lat, ls * nv, seed);
        let mut psi = Psi::with_shape(&lat, 24, ls, nv, 1);
        psi.as_mut_slice().copy_from_slice(flat.as_slice());
        psi
    };
    let (mut x, y) = (many(412), many(413));
    let (mut dx, dy) = (
        DeviceSpinor::from_host(&x).unwrap(),
        DeviceSpinor::from_host(&y).unwrap(),
    );
    assert_eq!(dx.num_vectors(), nv);
    let same = |dx: &DeviceSpinor, x: &Psi, what: &str| {
        let d = worst(&dx.to_host().unwrap(), x);
        assert!(d < 1e-13, "{what} differs by {d:e}");
    };
    same(&dx, &x, "the field after the transfer");

    let a = [C64::new(0.3, -1.1), C64::new(-2.0, 0.4), C64::new(0.0, 1.0)];
    let (mut dot, mut device_dot) = ([C64::ZERO; 3], [C64::ZERO; 3]);
    x.dot(&y, &mut dot);
    dx.dot(&dy, &mut device_dot);
    let (mut norm, mut device_norm) = ([0.0; 3], [0.0; 3]);
    x.norm_sqr(&mut norm);
    dx.norm_sqr(&mut device_norm);
    for v in 0..nv {
        assert!((dot[v] - device_dot[v]).norm_sqr().sqrt() < 1e-11 * dot[v].norm_sqr().sqrt());
        assert!((norm[v] - device_norm[v]).abs() < 1e-12 * norm[v]);
    }
    x.axpy(&a, &y);
    dx.axpy(&a, &dy);
    same(&dx, &x, "axpy");
    x.aypx(&a, &y);
    dx.aypx(&a, &dy);
    same(&dx, &x, "aypx");
    x.scale(&[-0.7, 2.0, 0.1]);
    dx.scale(&[-0.7, 2.0, 0.1]);
    same(&dx, &x, "scale");
    let (part, device_part) = (x.select(&[2, 0]), dx.select(&[2, 0]));
    same(&device_part, &part, "select");
    x.assign(&[1, 2], &part);
    dx.assign(&[1, 2], &device_part);
    same(&dx, &x, "assign");

    // Another basis of the subspace of the vectors, in place.
    {
        use ruttice::Subspace;
        let c: Vec<C64> = (0..nv * nv)
            .map(|k| C64::new(0.3 * k as f64 - 1.0, 0.2 - 0.1 * (k * k) as f64))
            .collect();
        let mut expected = x.zeros_like();
        for i in 0..nv {
            expected.axpy(&c[i * nv..(i + 1) * nv], &x.select(&vec![i; nv]));
        }
        dx.rotate(&c);
        x.copy_from(&expected);
        same(&dx, &x, "the change of basis");
    }

    // More vectors than one launch of the linear algebra takes coefficients for.
    {
        let more = 19;
        let (mut x, y) = (
            random_spinor(&lat, more, 417),
            random_spinor(&lat, more, 418),
        );
        let (mut dx, dy) = (
            DeviceSpinor::from_host(&x).unwrap(),
            DeviceSpinor::from_host(&y).unwrap(),
        );
        let a: Vec<C64> = (0..more)
            .map(|v| C64::new(0.1 * v as f64 - 0.9, 1.0 / (1.0 + v as f64)))
            .collect();
        let scales: Vec<f64> = (0..more).map(|v| 0.5 + 0.25 * v as f64).collect();
        x.axpy(&a, &y);
        dx.axpy(&a, &dy);
        same(&dx, &x, "axpy of 19 vectors");
        x.aypx(&a, &y);
        dx.aypx(&a, &dy);
        same(&dx, &x, "aypx of 19 vectors");
        x.scale(&scales);
        dx.scale(&scales);
        same(&dx, &x, "scale of 19 vectors");
        let (mut dot, mut device_dot) = (vec![C64::ZERO; more], vec![C64::ZERO; more]);
        x.dot(&y, &mut dot);
        dx.dot(&dy, &mut device_dot);
        let (mut norm, mut device_norm) = (vec![0.0; more], vec![0.0; more]);
        x.norm_sqr(&mut norm);
        dx.norm_sqr(&mut device_norm);
        for v in 0..more {
            assert!((dot[v] - device_dot[v]).norm_sqr().sqrt() < 1e-11 * norm[v]);
            assert!((norm[v] - device_norm[v]).abs() < 1e-12 * norm[v]);
        }
    }

    let params = MdwfParams {
        ls,
        m5: 1.8,
        mass: 0.2,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (mut cpu, mut gpu) = mdwf_pair(&lat, &u, params);
    cpu.set_alpha(0.7);
    gpu.set_alpha(0.7);
    let index = cpu.even_odd_index().unwrap().clone();
    same_operator(&cpu, &gpu, &x, 1e-13, "MDWF with three vectors");
    same_blocks(&cpu, &gpu, &index, &x, "MDWF with three vectors");
    solves_even_odd(&cpu, &gpu, &y, "MDWF with three vectors");

    let wilson = WilsonParams {
        kappa: 0.11,
        boundary: MIXED,
    };
    let four = |seed| random_spinor(&lat, nv, seed);
    let (cpu, gpu) = wilson_pair(&lat, &u, wilson);
    let index = cpu.even_odd_index().unwrap().clone();
    same_operator(&cpu, &gpu, &four(414), 1e-13, "Wilson with three vectors");
    same_blocks(&cpu, &gpu, &index, &four(414), "Wilson with three vectors");
    let clover = CloverParams {
        kappa: 0.11,
        csw: 1.2,
        boundary: MIXED,
    };
    let (cpu, gpu) = clover_pair(&lat, &u, clover);
    same_operator(&cpu, &gpu, &four(415), 1e-13, "clover with three vectors");
    same_blocks(&cpu, &gpu, &index, &four(415), "clover with three vectors");
    solves_even_odd(&cpu, &gpu, &four(416), "clover with three vectors");
}

/// The numbers of the tables a run prints: of every line that starts with an integer (a time
/// slice, or the pair of quarks), the reals after it.
fn tables(output: &str) -> Vec<f64> {
    output
        .lines()
        .filter(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|first| first.parse::<usize>().is_ok())
        })
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .filter(|token| token.contains(['.', 'e']))
                .filter_map(|token| token.trim_matches(['(', ')', ',']).parse::<f64>().ok())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// What `ruttice` prints for the parameter file `text` with `Main.backend : backend`.
fn run_with_backend(text: &str, backend: &str, name: &str) -> String {
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let text = text
        .replace("BACKEND", backend)
        .replace("CONF", examples.join("conf_04040408.txt").to_str().unwrap());
    let path = std::env::temp_dir().join(format!(
        "ruttice_cuda_{name}_{backend}_{}.yaml",
        std::process::id()
    ));
    std::fs::write(&path, text).unwrap();
    let mut out = Vec::new();
    let result = ruttice::app::run_file(&path, &mut out);
    let _ = std::fs::remove_file(&path);
    let out = String::from_utf8(out).unwrap();
    result.unwrap_or_else(|e| panic!("{name} on {backend} failed: {e:?}\n{out}"));
    out
}

/// A parameter file gives the same tables with `backend : cuda_bridge` as with `cpu`.
fn same_tables(text: &str, name: &str, tolerance: f64) {
    let (cpu, gpu) = (
        run_with_backend(text, "cpu", name),
        run_with_backend(text, "cuda_bridge", name),
    );
    let (a, b) = (tables(&cpu), tables(&gpu));
    assert!(!a.is_empty() && a.len() == b.len(), "{name}:\n{cpu}\n{gpu}");
    for (x, y) in a.iter().zip(&b) {
        assert!(
            (x - y).abs() <= tolerance * x.abs().max(y.abs()) + 1e-13,
            "{name}: {x:e} on the CPU, {y:e} on the device\n{gpu}"
        );
    }
}

const WILSON_RUN: &str = "Main:
  lattice_size : [4, 4, 4, 8]
  backend      : BACKEND
Gauge:
  config_filename_input : CONF
Fopr:
  fermion_type       : FERMION
  hopping_parameter  : 0.12
  boundary_condition : [-1, -1, -1, -1]
Solver:
  solver_type                   : SOLVER
  operator_type                 : OPERATOR
  convergence_criterion_squared : 1.0e-26
  maximum_number_of_iteration   : 2000
  maximum_number_of_restart     : 3
Source:
  source_type     : Local
  source_position : [0, 0, 0, 0]
";

const MDWF_RUN: &str = "Main:
  lattice_size : [4, 4, 4, 8]
  backend      : BACKEND
Gauge:
  config_filename_input : CONF
Fopr:
  fermion_type             : Domainwall_5din
  quark_mass               : 0.1
  quark_mass_PauliVillars  : 1.0
  domain_wall_height       : 1.0
  extent_of_5th_dimension  : 4
  coefficient_b            : 1.5
  coefficient_c            : 0.5
  boundary_condition       : [1, 1, 1, -1]
  neff_alpha               : ALPHA
Solver:
  solver_type                   : SOLVER
  operator_type                 : OPERATOR
  number_of_kept_directions     : 4
  maximum_number_of_iteration   : 4000
  convergence_criterion_squared : 1.0e-22
  preconditioner_type           : PRECONDITIONER
Source:
  source_type             : Local
  source_position         : [0, 0, 0, 0]
  number_of_rhs_per_solve : RHS
";

/// `ruttice` with `backend : cuda_bridge`: the correlators of the CPU, for the Wilson and
/// clover solvers on the whole lattice and on the even sites.
#[test]
fn wilson_runs_from_a_parameter_file_on_the_device() {
    for (fermion, solver, operator) in [
        ("Wilson", "CG", "Even_odd"),
        ("Wilson", "BiCGStab", "Lexical"),
        ("Clover\n  clover_coefficient : 1.2", "BiCGStab", "Even_odd"),
        ("Clover\n  clover_coefficient : 1.2", "FGMRES", "Lexical"),
    ] {
        let text = WILSON_RUN
            .replace("FERMION", fermion)
            .replace("SOLVER", solver)
            .replace("OPERATOR", operator);
        same_tables(
            &text,
            &format!("{}_{solver}_{operator}", &fermion[..6]),
            1e-8,
        );
    }
}

/// The same for domain-wall fermions: the correlators and the residual mass, for the outer
/// solvers, both forms of the operator, with Neff's α and with the Chebyshev preconditioner.
#[test]
fn domain_wall_runs_from_a_parameter_file_on_the_device() {
    for (solver, operator, alpha, preconditioner, rhs) in [
        ("GCR", "PV_right", "1.0", "None", "1"),
        ("GCR", "PV_left", "0.6", "None", "12"),
        ("Block_GCR", "PV_left", "0.6", "None", "4"),
        ("BiCGStab", "PV_left", "0.6", "None", "6"),
        ("FGMRES", "PV_right", "1.0", "Chebyshev", "4"),
        ("CG", "Normal_left", "1.0", "None", "12"),
    ] {
        let text = MDWF_RUN
            .replace("RHS", rhs)
            .replace("SOLVER", solver)
            .replace("OPERATOR", operator)
            .replace("ALPHA", alpha)
            .replace("PRECONDITIONER", preconditioner);
        same_tables(
            &text,
            &format!("mdwf_{solver}_{operator}_{preconditioner}"),
            1e-6,
        );
    }
}

/// The P of the Wilson multigrid (one spinor a site, chirality by γ5) on the device, folded in
/// spin, restricts and prolongs as the CPU's: on the whole lattice and on either parity, for one
/// vector and several. P kept in single precision on both, so the two differ by the order of
/// their sums only.
#[test]
fn wilson_transfers_on_the_device() {
    use ruttice::backend::cuda_bridge::{DeviceBasis, DeviceCoarse};
    use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
    use ruttice::{extract_parity, Coarse};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 441);
    let mut op = Wilson::<f64>::new(
        &lat,
        WilsonParams {
            kappa: 0.12,
            boundary: ANTIPERIODIC_T,
        },
    );
    op.set_gauge(&u).unwrap();
    let n = 4;
    let x = random_spinor(&lat, n, 442);
    // γ5 = [[0, 1], [1, 0]] in the Dirac representation: spins 0, 1 and 2, 3 exchanged.
    let mut g5x = x.zeros_like();
    for (out, inp) in g5x
        .as_mut_slice()
        .chunks_exact_mut(24)
        .zip(x.as_slice().chunks_exact(24))
    {
        out[..12].copy_from_slice(&inp[12..]);
        out[12..].copy_from_slice(&inp[..12]);
    }
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = Basis::<f64, f32>::from_vectors(&agg, &x, &g5x).unwrap();
    let device = DeviceBasis::new(&basis).unwrap();
    let close = |a: &Field<f64, Coarse>, b: &DeviceCoarse, what: &str| {
        let b = b.to_host(a).unwrap();
        let size = a.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let d = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            size > 0.0 && d < 1e-12 * size,
            "{what} differs by {d:e} of {size:e}"
        );
    };
    let index = op.even_odd_index().unwrap().clone();
    for nv in [1, 3] {
        let fine = random_spinor(&lat, nv, 443 + nv as u64);
        let mut c_cpu = basis.coarse_field(nv);
        basis.restrict(&mut c_cpu, &fine);
        let mut c_gpu = device.coarse_field::<f64>(nv);
        device.restrict(&mut c_gpu, &DeviceSpinor::from_host(&fine).unwrap());
        close(&c_cpu, &c_gpu, &format!("R of {nv} vectors"));

        let mut f_cpu = Psi::with_shape(&lat, 24, 1, nv, 1);
        basis.prolong(&mut f_cpu, &c_cpu);
        let mut f_gpu = DeviceSpinor::<f64>::from_host(&f_cpu).unwrap();
        device.prolong(&mut f_gpu, &DeviceCoarse::from_host(&c_cpu).unwrap());
        let d = worst(&f_gpu.to_host().unwrap(), &f_cpu);
        assert!(d < 1e-12, "P of {nv} vectors differs by {d:e}");

        for parity in [Parity::Even, Parity::Odd] {
            let mut h = HalfPsi::with_shape(&lat, 24, 1, nv, 1);
            h.set_parity(parity);
            extract_parity(&mut h, &fine, &index, parity);
            let mut c_cpu = basis.coarse_field(nv);
            basis.restrict_half(&mut c_cpu, &h, &index);
            let mut c_gpu = device.coarse_field::<f64>(nv);
            let dh = DeviceSpinor::from_host_half(&h, &index).unwrap();
            device.restrict(&mut c_gpu, &dh);
            close(&c_cpu, &c_gpu, &format!("R on {parity:?} of {nv} vectors"));
        }
    }
}

/// The coarse operators P† D P of the Wilson and clover operators made on the device
/// (`coarse_blocks`: P of unit vectors, the part within a site and the eight hops sorted by
/// whether they leave the aggregate) are those `CoarseOp::build` makes on the CPU, on the same P.
#[test]
fn wilson_and_clover_coarse_operators_on_the_device() {
    use ruttice::backend::cuda_bridge::DeviceBasis;
    use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 451);
    let n = 4;
    let x = random_spinor(&lat, n, 452);
    let mut g5x = x.zeros_like();
    for (out, inp) in g5x
        .as_mut_slice()
        .chunks_exact_mut(24)
        .zip(x.as_slice().chunks_exact(24))
    {
        out[..12].copy_from_slice(&inp[12..]);
        out[12..].copy_from_slice(&inp[..12]);
    }
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = Basis::<f64, f32>::from_vectors(&agg, &x, &g5x).unwrap();
    let device = DeviceBasis::new(&basis).unwrap();
    let close = |cpu: &CoarseOp<f64>, gpu: &CoarseOp<f64>, what: &str| {
        let ((dc, lc), (dg, lg)) = (cpu.blocks(), gpu.blocks());
        for (a, b, part) in [(dc, dg, "blocks"), (lc, lg, "links")] {
            assert_eq!(a.len(), b.len(), "{what} {part}");
            let size = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let d = a
                .iter()
                .zip(b)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("{what} {part}: differ by {d:e} of {size:e}");
            assert!(size > 0.0 && d < 1e-12 * size, "{what} {part}");
        }
    };
    let wp = WilsonParams {
        kappa: 0.13,
        boundary: MIXED,
    };
    let mut cw = Wilson::<f64>::new(&lat, wp);
    cw.set_gauge(&u).unwrap();
    let mut gw = CudaWilson::<f64>::new(&lat, wp).unwrap();
    gw.set_gauge(&u).unwrap();
    close(
        &CoarseOp::build(&cw, &mut cw.work(), &basis),
        &gw.coarse_operator(&device).unwrap(),
        "Wilson",
    );
    let cp = CloverParams {
        kappa: 0.12,
        csw: 1.7,
        boundary: ANTIPERIODIC_T,
    };
    let (cc, gc) = clover_pair(&lat, &u, cp);
    close(
        &CoarseOp::build(&cc, &mut cc.work(), &basis),
        &gc.coarse_operator(&device).unwrap(),
        "clover",
    );
}

/// SAP (the Schwarz smoother of the Wilson and clover multigrid) on the device is that of the
/// CPU: the same algorithm (`Sap`) with the block operator, the block minimal-residual steps
/// and the colours of the device; to rounding, for one vector and several.
#[test]
fn sap_on_the_device_is_sap_on_the_cpu() {
    use ruttice::backend::cuda_bridge::device_sap;
    use ruttice::solver::preconditioner::multigrid::{Aggregation, Sap};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 461);
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let check =
        |cpu: &mut dyn LinearFn<Psi>, gpu: &mut dyn LinearFn<DeviceSpinor<f64>>, what: &str| {
            for nv in [1, 3] {
                let b = random_spinor(&lat, nv, 462 + nv as u64);
                let mut x = b.zeros_like();
                cpu.mult(&mut x, &b);
                let db = DeviceSpinor::from_host(&b).unwrap();
                let mut dx = db.zeros_like();
                gpu.mult(&mut dx, &db);
                let size = x.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let d = worst(&dx.to_host().unwrap(), &x);
                eprintln!("{what}, {nv} vectors: differ by {d:e} of {size:e}");
                assert!(size > 0.0 && d < 1e-11 * size, "{what}, {nv} vectors");
            }
        };
    let wp = WilsonParams {
        kappa: 0.13,
        boundary: ANTIPERIODIC_T,
    };
    let mut cw = Wilson::<f64>::new(&lat, wp);
    cw.set_gauge(&u).unwrap();
    let mut gw = CudaWilson::<f64>::new(&lat, wp).unwrap();
    gw.set_gauge(&u).unwrap();
    check(
        &mut Sap::new(&cw, &agg, 4, 6),
        &mut device_sap(&gw, &agg, 4, 6).unwrap(),
        "Wilson",
    );
    let cp = CloverParams {
        kappa: 0.12,
        csw: 1.7,
        boundary: MIXED,
    };
    let (cc, gc) = clover_pair(&lat, &u, cp);
    check(
        &mut Sap::new(&cc, &agg, 4, 6),
        &mut device_sap(&gc, &agg, 4, 6).unwrap(),
        "clover",
    );
}

/// The prolongation of the domain-wall multigrid and its adjoint on the device are those of
/// the CPU, on the whole lattice and on the sites of either parity, for several vectors.
#[test]
fn multigrid_transfers_on_the_device() {
    use ruttice::backend::cuda_bridge::{DeviceBasis, DeviceCoarse};
    use ruttice::solver::preconditioner::multigrid::{pv_eo_basis, Aggregation};
    use ruttice::{extract_parity, Coarse, ConvertFrom};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 431);
    let (ls, n, nv) = (4, 3, 2);
    let params = MdwfParams {
        ls,
        m5: 1.8,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, _) = mdwf_pair(&lat, &u, params);
    let (pv, _) = mdwf_pair(
        &lat,
        &u,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    );
    let index = cpu.even_odd_index().unwrap().clone();
    let half = |parity, vectors: usize, seed| {
        let mut h = HalfPsi::with_shape(&lat, 24, ls, vectors, 1);
        h.set_parity(parity);
        extract_parity(
            &mut h,
            &random_spinor(&lat, ls * vectors, seed),
            &index,
            parity,
        );
        h
    };
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = pv_eo_basis(&cpu, &pv, &agg, &half(Parity::Even, n, 432)).unwrap();
    let device = DeviceBasis::from_pv_eo(&basis).unwrap();
    let same = |a: &Field<f64, Coarse>, b: &DeviceCoarse, what: &str| {
        let b = b.to_host(a).unwrap();
        let d = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(d < 1e-10, "{what} differs by {d:e}");
    };

    // P made on the device in single precision is the P of the CPU, as far as half precision
    // keeps it there.
    {
        let (_, gpu) = mdwf_pair(&lat, &u, params);
        let (_, gpu_pv) = mdwf_pair(
            &lat,
            &u,
            MdwfParams {
                mass: 1.0,
                ..params
            },
        );
        let y = DeviceSpinor::from_host_half(&half(Parity::Even, n, 432), &index).unwrap();
        let made = DeviceBasis::pv_eo(&gpu, &gpu_pv, &agg, &y, false).unwrap();
        let ruttice::solver::preconditioner::multigrid::PvEoBasis::Half(stored) = &basis else {
            panic!("pv_eo_basis keeps P in half precision");
        };
        let a: Vec<f32> = stored.stored().iter().map(|x| x.to_f32()).collect();
        let b = made.stored().unwrap();
        assert_eq!(a.len(), b.len());
        let size = a.iter().fold(0.0f64, |m, x| m.max(f64::from(x.abs())));
        let d = a
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (a, b)| m.max(f64::from((a - b).abs())));
        assert!(
            d < 1e-3 * size,
            "P made on the device differs by {d:e} of {size:e}"
        );
    }

    // On the whole lattice.
    let mut fine = Psi::with_shape(&lat, 24, ls, nv, 1);
    fine.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, ls * nv, 433).as_slice());
    let device_fine = DeviceSpinor::from_host(&fine).unwrap();
    let mut coarse = basis.coarse_field(nv);
    basis.restrict(&mut coarse, &fine);
    let mut device_coarse = device.coarse_field::<f64>(nv);
    device.restrict(&mut device_coarse, &device_fine);
    same(&coarse, &device_coarse, "P† on the whole lattice");
    let (mut back, mut device_back) = (fine.zeros_like(), device_fine.zeros_like());
    basis.prolong(&mut back, &coarse);
    device.prolong(&mut device_back, &device_coarse);
    let d = worst(&device_back.to_host().unwrap(), &back);
    assert!(d < 1e-13, "P on the whole lattice differs by {d:e}");

    for parity in [Parity::Even, Parity::Odd] {
        let h = half(parity, nv, 434);
        let device_h = DeviceSpinor::from_host_half(&h, &index).unwrap();
        basis.restrict_half(&mut coarse, &h, &index);
        device.restrict(&mut device_coarse, &device_h);
        same(&coarse, &device_coarse, "P† ι");
        let (mut back, mut device_back) = (h.zeros_like(), device_h.zeros_like());
        basis.prolong_half(&mut back, &coarse, &index);
        device.prolong(&mut device_back, &device_coarse);
        let d = worst(&device_back.to_host_half(&index).unwrap(), &back);
        assert!(d < 1e-13, "π P differs by {d:e}");
    }

    // The coarse operator, its adjoint and its even-odd form.
    {
        use ruttice::backend::cuda_bridge::DeviceCoarseOp;
        use ruttice::solver::preconditioner::multigrid::BlockMap;
        use ruttice::{LinearFn, LinearOp};
        let mut op = basis.coarse_op(&cpu, &mut cpu.work());
        assert!(op.prepare_even_odd().unwrap());
        let device_op = DeviceCoarseOp::<f64>::new(&op, true).unwrap();
        // The coarse operator built on the device is the one built on the CPU.
        let (_, gpu) = mdwf_pair(&lat, &u, params);
        let built = gpu.coarse_operator(&device).unwrap();
        for (what, a, b) in [
            ("blocks", op.blocks().0, built.blocks().0),
            ("links", op.blocks().1, built.blocks().1),
        ] {
            let d = a
                .iter()
                .zip(b)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                d < 1e-13,
                "the {what} of the coarse operator differ by {d:e}"
            );
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = || {
            let mut f = basis.coarse_field(nv);
            for v in f.as_mut_slice() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *v = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            }
            f
        };
        let (x, g) = (random(), random());
        let (dx, dg) = (
            DeviceCoarse::<f64>::from_host(&x).unwrap(),
            DeviceCoarse::<f64>::from_host(&g).unwrap(),
        );
        same(&x, &dx, "a coarse field after the transfer");
        let (mut out, mut dout) = (x.zeros_like(), dx.zeros_like());
        op.mult(&mut out, &x);
        device_op.apply(&mut dout, &dx, false);
        same(&out, &dout, "the coarse operator");
        op.mult_dag(&mut out, &x);
        device_op.apply(&mut dout, &dx, true);
        same(&out, &dout, "the adjoint of the coarse operator");

        let (mut odd, mut dodd) = (x.zeros_like(), dx.zeros_like());
        let (mut even, mut deven) = (x.zeros_like(), dx.zeros_like());
        op.even_block_inverse(&mut even, &x);
        device_op.even_block_inverse(&mut deven, &dx);
        same(
            &even,
            &deven,
            "the inverses of the blocks on the even sites",
        );
        op.schur(&mut out, &even, &mut odd);
        device_op.schur(&mut dout, &deven, &mut dodd);
        same(&out, &dout, "the Schur complement");
        op.schur_source(&mut out, &g, &mut odd);
        device_op.schur_source(&mut dout, &dg, &mut dodd);
        same(&out, &dout, "the source of the Schur system");
        op.schur_solution(&mut even, &g, &mut odd);
        device_op.schur_solution(&mut deven, &dg, &mut dodd);
        same(&even, &deven, "the solution from that of the Schur system");

        let mut blocks = BlockMap::abs_inverse_of_a_gamma5(&op);
        let device_blocks = device_op.blocks(blocks.site_blocks()).unwrap();
        blocks.mult(&mut out, &x);
        device_op.apply_blocks(&mut dout, &dx, &device_blocks);
        same(&out, &dout, "matrices on the coarse sites");
        // The same matrices made on the device, by its Jacobi method.
        let made = device_op.abs_inverse_of_a_gamma5().unwrap();
        device_op.apply_blocks(&mut dout, &dx, &made);
        same(&out, &dout, "|H|⁻¹ made on the device");
        device_op.apply_blocks(&mut dout, &dx, &device_blocks);
        device_op.gamma5(&mut dout);
        for site in out.as_mut_slice().chunks_exact_mut(4 * basis.coarse_dim()) {
            for vector in site.chunks_exact_mut(2 * basis.coarse_dim()) {
                for r in &mut vector[basis.coarse_dim()..] {
                    *r = -*r;
                }
            }
        }
        same(&out, &dout, "Γ5 on the coarse lattice");
    }

    // Between the precisions, and the linear algebra of coarse fields.
    let mut single = device.coarse_field::<f32>(nv);
    single.convert_from(&device_coarse);
    let mut norms = [[0.0; 2]; 3];
    device_coarse.norm_sqr(&mut norms[0]);
    single.norm_sqr(&mut norms[1]);
    coarse.norm_sqr(&mut norms[2]);
    for v in 0..nv {
        assert!((norms[0][v] - norms[2][v]).abs() < 1e-12 * norms[2][v]);
        assert!((norms[1][v] - norms[2][v]).abs() < 1e-5 * norms[2][v]);
    }
}

/// The coarse correction of the domain-wall multigrid on the device is that of the CPU, for
/// both forms of the operator and the second factor by MINRES on its Hermitian form or by
/// BiCGStab on its exact adjoint: with the coarse systems solved to rounding in double
/// precision the two agree to that.
#[test]
fn multigrid_correction_on_the_device() {
    use ruttice::backend::cuda_bridge::{DeviceBasis, DeviceCorrection};
    use ruttice::fermion::EoForm;
    use ruttice::kernel::Dirac;
    use ruttice::solver::preconditioner::multigrid::{
        pv_eo_basis, Aggregation, CoarseKrylov, PvEoCycle, PvEoParams,
    };
    use ruttice::{extract_parity, Aos};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 441);
    let (ls, n, nv) = (4, 4, 2);
    let params = MdwfParams {
        ls,
        m5: 1.8,
        mass: 0.1,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = mdwf_pair(&lat, &u, params);
    let (cpu_pv, gpu_pv) = mdwf_pair(
        &lat,
        &u,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    );
    let (mut cpu_a, mut gpu_a) = mdwf_pair(&lat, &u, params);
    cpu_a.set_alpha(0.6);
    gpu_a.set_alpha(0.6);
    let index = cpu.even_odd_index().unwrap().clone();
    let half = |vectors: usize, seed| {
        let mut h = HalfPsi::with_shape(&lat, 24, ls, vectors, 1);
        h.set_parity(Parity::Even);
        extract_parity(
            &mut h,
            &random_spinor(&lat, ls * vectors, seed),
            &index,
            Parity::Even,
        );
        h
    };
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = pv_eo_basis(&cpu, &cpu_pv, &agg, &half(n, 442)).unwrap();
    let minres = PvEoParams {
        coarse_tolerance_sqr: 1e-26,
        pv_tolerance_sqr: 1e-26,
        coarse_max_iterations: 400,
        pv_max_iterations: 400,
        ..PvEoParams::default()
    };
    let adjoint = PvEoParams {
        pv_solver: CoarseKrylov::BiCgStab,
        ..minres
    };
    let r = half(nv, 443);
    let device_r = DeviceSpinor::from_host_half(&r, &index).unwrap();
    let even = device_r.zeros_like();
    for (form, cpu_outer, gpu_outer, settings) in [
        (EoForm::Right, (&cpu, &cpu_pv), (&gpu, &gpu_pv), minres),
        (EoForm::Left, (&cpu_a, &cpu_pv), (&gpu_a, &gpu_pv), minres),
        (EoForm::Right, (&cpu, &cpu_pv), (&gpu, &gpu_pv), adjoint),
        (EoForm::Left, (&cpu_a, &cpu_pv), (&gpu_a, &gpu_pv), adjoint),
    ] {
        let mut cycle = PvEoCycle::<Dirac, f64, Aos>::with_coarse_storage(
            &cpu,
            &cpu_pv,
            (cpu_outer.0, cpu_outer.1, form),
            &basis,
            settings,
        )
        .unwrap();
        let mut expected = r.zeros_like();
        cycle.coarse_correction(&mut expected, &r);
        let mut correction = DeviceCorrection::<f64, f64>::new(
            DeviceBasis::from_pv_eo(&basis).unwrap(),
            (&gpu, &gpu_pv),
            gpu_outer,
            form,
            settings,
            &even,
        )
        .unwrap();
        let mut e = even.zeros_like();
        correction.correct(&mut e, &device_r);
        let d = worst(&e.to_host_half(&index).unwrap(), &expected);
        let size = expected
            .as_slice()
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(
            d < 1e-9 * size,
            "{form:?}, {:?}: the correction differs by {d:e} of {size:e}",
            settings.pv_solver
        );
    }
}

/// The levels below the coarse lattice on the device are those of the CPU: their test vectors
/// (P† of the vectors P is made of), the V-cycle of the quark factor built from them and the
/// absolute-value preconditioner of the Pauli-Villars factor, with one level and with two, the
/// first of them refined by the bootstrap, and with a last level larger than a block of threads
/// (1536 complex numbers); and the setups take as many applications of the coarse operators.
#[test]
fn multigrid_levels_on_the_device() {
    use ruttice::backend::cuda_bridge::{
        DeviceAbsMultilevel, DeviceBasis, DeviceCoarse, DeviceCoarseOp, DeviceMultilevel,
    };
    use ruttice::solver::preconditioner::multigrid::{
        pv_eo_basis, pv_eo_coarse_vectors, AbsMultilevel, Aggregation, LevelBootstrap,
        SignedMultilevel,
    };
    use ruttice::{extract_parity, Coarse};
    let ls = 4;
    // The largest difference of two coarse fields, and the size of the first.
    let compare = |a: &Field<f64, Coarse>, b: &Field<f64, Coarse>| -> (f64, f64) {
        let size = a.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let diff = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        (diff, size)
    };
    for (dims, n, super_blocks, bootstrap, pv) in [
        ([4, 4, 4, 8], 4, vec![2], vec![], true),
        (
            [8, 8, 8, 8],
            4,
            vec![2, 2],
            vec![LevelBootstrap {
                rounds: 1,
                iterations: 3,
            }],
            true,
        ),
        ([4, 4, 4, 8], 12, vec![1], vec![], false),
    ] {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 451);
        let params = MdwfParams {
            ls,
            m5: 1.8,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary: ANTIPERIODIC_T,
        };
        let (cpu, gpu) = mdwf_pair(&lat, &u, params);
        let (cpu_pv, gpu_pv) = mdwf_pair(
            &lat,
            &u,
            MdwfParams {
                mass: 1.0,
                ..params
            },
        );
        let index = cpu.even_odd_index().unwrap().clone();
        let mut y = HalfPsi::with_shape(&lat, 24, ls, n, 1);
        y.set_parity(Parity::Even);
        extract_parity(
            &mut y,
            &random_spinor(&lat, ls * n, 452),
            &index,
            Parity::Even,
        );
        let agg = Aggregation::new(&lat, [2, 2, 2, 2]);
        let basis = pv_eo_basis(&cpu, &cpu_pv, &agg, &y).unwrap();
        let device_basis = DeviceBasis::from_pv_eo(&basis).unwrap();

        // The test vectors.
        let vectors = pv_eo_coarse_vectors(&cpu, &cpu_pv, &basis, &y);
        let device_y = DeviceSpinor::from_host_half(&y, &index).unwrap();
        let device_vectors = device_basis
            .pv_eo_coarse_vectors(&gpu, &gpu_pv, &device_y)
            .unwrap();
        for (j, v) in vectors.iter().enumerate() {
            let d = device_vectors.select(&[j]).to_host(v).unwrap();
            let size = v.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let diff = v
                .as_slice()
                .iter()
                .zip(d.as_slice())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(diff < 1e-12 * size, "test vector {j}: {diff:e} of {size:e}");
        }

        // The V-cycle, from the same vectors.
        let mut coarse = basis.coarse_op(&cpu, &mut cpu.work()).to_storage::<f64>();
        let coarse_lat = coarse.lattice().clone();
        let device_op = DeviceCoarseOp::<f64>::new(&coarse, false).unwrap();
        let levels =
            SignedMultilevel::build_with(&mut coarse, &vectors, &super_blocks, &bootstrap).unwrap();
        let mut on_device =
            DeviceCoarse::<f64>::zeros(coarse_lat.num_sites(), coarse.dim(), vectors.len())
                .unwrap();
        for (j, v) in vectors.iter().enumerate() {
            on_device.assign(&[j], &DeviceCoarse::from_host(v).unwrap());
        }
        let device_levels = DeviceMultilevel::build(
            &device_op,
            &coarse_lat,
            &on_device,
            &super_blocks,
            &bootstrap,
        )
        .unwrap();
        assert_eq!(levels.setup_applications, device_levels.setup_applications);
        assert_eq!(levels.coarsest_dim(), device_levels.coarsest_dim());
        let mut r = Field::<f64, Coarse>::with_shape(&coarse_lat, 2 * coarse.dim(), 1, 2, 1);
        for (k, x) in r.as_mut_slice().iter_mut().enumerate() {
            *x = (0.37 * k as f64).sin() + 0.1 * (1.3 * k as f64).cos();
        }
        let mut expected = r.zeros_like();
        levels.apply(&mut expected, &r);
        let device_r = DeviceCoarse::from_host(&r).unwrap();
        let mut out = device_r.zeros_like();
        device_levels.apply(&mut out, &device_r);
        let (diff, size) = compare(&expected, &out.to_host(&r).unwrap());
        eprintln!(
            "{dims:?}, super-blocks {super_blocks:?}: the V-cycles differ by {diff:e} of {size:e} \
             (coarsest {} complex numbers)",
            levels.coarsest_dim()
        );
        assert!(size > 0.0 && diff < 1e-12 * size);
        if !pv {
            continue;
        }

        // The levels of the Pauli-Villars factor, added to a field that is not zero.
        let mut coarse_pv = basis
            .coarse_op(&cpu_pv, &mut cpu_pv.work())
            .to_storage::<f64>();
        let device_pv = DeviceCoarseOp::<f64>::new(&coarse_pv, false).unwrap();
        let abs = AbsMultilevel::build(&mut coarse_pv, &vectors, &super_blocks).unwrap();
        let device_abs =
            DeviceAbsMultilevel::build(&device_pv, &coarse_lat, &on_device, &super_blocks).unwrap();
        assert_eq!(abs.setup_applications, device_abs.setup_applications);
        assert_eq!(abs.coarsest_dim(), device_abs.coarsest_dim());
        let mut expected = r.zeros_like();
        expected.copy_from(&r);
        abs.add(&mut expected, &r);
        let mut out = DeviceCoarse::from_host(&r).unwrap();
        device_abs.add(&mut out, &device_r);
        let (diff, size) = compare(&expected, &out.to_host(&r).unwrap());
        eprintln!(
            "  the Pauli-Villars levels differ by {diff:e} of {size:e} (coarsest {} complex numbers)",
            abs.coarsest_dim()
        );
        assert!(size > 0.0 && diff < 1e-12 * size);
    }
}

/// Stout smearing on the device is that of the CPU: the smeared links of three steps of ρ = 0.1
/// and one of weights only in space, the chain rule of the force through them, and the domain-
/// wall operator on the links smeared there.
#[test]
fn stout_smearing_on_the_device() {
    use ruttice::backend::cuda_bridge::DeviceLinks;
    use ruttice::gauge::smearing::{spatial, uniform, Stout};
    use ruttice::gauge::{GaugeSmearing, Smear};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 461);
    let largest = |a: &Field<f64, Gauge>, b: &Field<f64, Gauge>| -> (f64, f64) {
        let size = a.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let diff = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        (diff, size)
    };
    let device_u = DeviceLinks::from_host(&u).unwrap();
    for (rho, steps) in [(uniform(0.1), 3), (spatial(0.15), 1)] {
        let smearing = GaugeSmearing {
            smear: Smear::Stout(Stout { rho }),
            steps,
        };
        let levels = smearing.levels(&u);
        let device_levels = smearing.levels_on_device(&device_u).unwrap();
        for (k, (a, b)) in levels.iter().zip(&device_levels).enumerate() {
            let (diff, size) = largest(a, &b.to_host().unwrap());
            assert!(
                diff < 1e-13 * size,
                "step {k}: the links differ by {diff:e}"
            );
        }
        // Σ' arbitrary matrices.
        let sigma = random_gauge(&lat, 462);
        let expected = smearing.chain(
            {
                let mut s = Field::<f64, Gauge>::gauge(&lat);
                s.as_mut_slice().copy_from_slice(sigma.as_slice());
                s
            },
            &u,
            &levels,
        );
        let got = smearing
            .chain_on_device(
                DeviceLinks::from_host(&sigma).unwrap(),
                &device_u,
                &device_levels,
            )
            .unwrap()
            .to_host()
            .unwrap();
        let (diff, size) = largest(&expected, &got);
        eprintln!("{steps} steps: the chain rule differs by {diff:e} of {size:e}");
        assert!(diff < 1e-12 * size);
    }

    // D on links smeared on the device is D on links smeared on the CPU.
    let smearing = GaugeSmearing {
        smear: Smear::Stout(Stout { rho: uniform(0.1) }),
        steps: 3,
    };
    let v = smearing.apply(&u);
    let params = MdwfParams {
        ls: 4,
        m5: 1.0,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, _) = mdwf_pair(&lat, &v, params);
    let mut gpu = CudaMdwf::<f64>::new(&lat, params).unwrap();
    let device_v = smearing.levels_on_device(&device_u).unwrap().pop().unwrap();
    gpu.set_gauge_on_device(&device_v).unwrap();
    let x = random_fifth(&lat, 4, 463);
    let expected = {
        let mut out = x.zeros_like();
        cpu.mult(&mut cpu.work(), &mut out, &x, Variant::D);
        out
    };
    let mut out = DeviceSpinor::from_host(&x).unwrap().zeros_like();
    gpu.mult(
        &mut gpu.work(),
        &mut out,
        &DeviceSpinor::from_host(&x).unwrap(),
        Variant::D,
    );
    let d = worst(&out.to_host().unwrap(), &expected);
    assert!(d < 1e-12, "D on the smeared links differs by {d:e}");
}

/// The derivative of the pseudofermion action by the links for the domain-wall operator on the
/// device is that of the CPU (`ForceOp::derivative`), with and without Neff's α.
#[test]
fn domain_wall_force_on_the_device() {
    use ruttice::hmc::ForceOp;
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 471);
    let params = MdwfParams {
        ls: 6,
        m5: 1.4,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    for alpha in [1.0, 0.6] {
        let (mut cpu, mut gpu) = mdwf_pair(&lat, &u, params);
        cpu.set_alpha(alpha);
        gpu.set_alpha(alpha);
        let (zeta, eta) = (random_fifth(&lat, 6, 472), random_fifth(&lat, 6, 473));
        let expected = cpu.derivative(&zeta, &eta);
        let got = gpu
            .derivative(
                &DeviceSpinor::from_host(&zeta).unwrap(),
                &DeviceSpinor::from_host(&eta).unwrap(),
            )
            .to_host()
            .unwrap();
        let size = expected
            .as_slice()
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        let diff = expected
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("α = {alpha}: the derivatives differ by {diff:e} of {size:e}");
        assert!(size > 0.0 && diff < 1e-12 * size);
    }
}

/// The derivatives of the Wilson and clover actions on the device are those of the CPU: the hops
/// (`ForceOp::derivative`), the clover term of two weighted pairs walked at once, ln |det T| on
/// each parity, and the even-odd clover term with the derivative of ln |det T_o|.
#[test]
fn wilson_and_clover_forces_on_the_device() {
    use ruttice::backend::cuda_bridge::DeviceLinks;
    use ruttice::hmc::{EvenOddForceOp, ForceOp};
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 491);
    let du = DeviceLinks::from_host(&u).unwrap();
    let close = |expected: &Field<f64, Gauge>, got: &DeviceLinks, what: &str| {
        let got = got.to_host().unwrap();
        let size = expected
            .as_slice()
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        let diff = expected
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("{what}: the derivatives differ by {diff:e} of {size:e}");
        assert!(size > 0.0 && diff < 1e-12 * size, "{what}");
    };
    let fields: Vec<Psi> = (0..4).map(|k| random_spinor(&lat, 1, 492 + k)).collect();
    let dev: Vec<DeviceSpinor<f64>> = fields
        .iter()
        .map(|f| DeviceSpinor::from_host(f).unwrap())
        .collect();

    let wp = WilsonParams {
        kappa: 0.13,
        boundary: ANTIPERIODIC_T,
    };
    let mut cw = Wilson::<f64>::new(&lat, wp);
    ForceOp::set_links(&mut cw, &u).unwrap();
    let mut gw = CudaWilson::<f64>::new(&lat, wp).unwrap();
    ForceOp::set_links(&mut gw, &du).unwrap();
    close(
        &cw.derivative(&fields[0], &fields[1]),
        &gw.derivative(&dev[0], &dev[1]),
        "Wilson hops",
    );

    let params = CloverParams {
        kappa: 0.12,
        csw: 1.7,
        boundary: ANTIPERIODIC_T,
    };
    let kc = params.kappa * params.csw;
    let mut cc = Clover::<f64>::new(&lat, params);
    ForceOp::set_links(&mut cc, &u).unwrap();
    let mut gc = CudaClover::<f64>::new(&lat, params).unwrap();
    ForceOp::set_links(&mut gc, &du).unwrap();
    close(
        &cc.derivative(&fields[0], &fields[1]),
        &gc.derivative(&dev[0], &dev[1]),
        "clover hops",
    );
    close(
        &cc.clover_derivative(
            kc,
            &u,
            &[
                (1.0, &fields[0], &fields[1]),
                (-0.3, &fields[2], &fields[3]),
            ],
        ),
        &gc.clover_derivative(
            kc,
            &du,
            &[(1.0, &dev[0], &dev[1]), (-0.3, &dev[2], &dev[3])],
        ),
        "clover term",
    );
    for parity in [Parity::Even, Parity::Odd] {
        let (a, b) = (cc.log_det_diag(&lat, parity), gc.log_det_diag(&lat, parity));
        eprintln!("ln |det T| on {parity:?}: {a} and {b}");
        assert!(a != 0.0 && (a - b).abs() < 1e-12 * a.abs());
    }
    close(
        &cc.clover_derivative_eo(kc, &u, Parity::Odd, &fields[0], &fields[1]),
        &gc.clover_derivative_eo(kc, &du, Parity::Odd, &dev[0], &dev[1]),
        "even-odd clover term with ln det",
    );
}

/// D† x = b through Ã† on the device (`adjoint_right_hand_side`, `adjoint_solution` with the
/// fields of the device): the right-hand side is that of the CPU, and the x of the device
/// solves D† x = b on the CPU.
#[test]
fn d_dagger_through_the_adjoint_pv_operator_on_the_device() {
    use ruttice::fermion::{adjoint_right_hand_side, adjoint_solution, EoForm, PvRatio};
    use ruttice::solver::{Gcr, Solver};
    use ruttice::Half;
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 481);
    let params = MdwfParams {
        ls: 6,
        m5: 1.4,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let (cpu, gpu) = mdwf_pair(&lat, &u, params);
    let (_, gpu_pv) = mdwf_pair(&lat, &u, params.pauli_villars());
    let b = random_fifth(&lat, 6, 482);
    let index = cpu.even_odd_index().unwrap().clone();
    let mut even = Field::<f64, Spinor, Half>::with_shape(&lat, 24, 6, 1, 1);
    even.set_parity(Parity::Even);
    let mut cpu_rhs = even.zeros_like();
    adjoint_right_hand_side(&cpu, &*index, &mut cpu_rhs, &b);
    let b_dev = DeviceSpinor::from_host(&b).unwrap();
    let (e, o) = (
        DeviceSpinor::<f64>::shape_half(&lat, 6, Parity::Even).unwrap(),
        DeviceSpinor::<f64>::shape_half(&lat, 6, Parity::Odd).unwrap(),
    );
    let mut rhs = e.zeros_with(1);
    adjoint_right_hand_side(&gpu, &DeviceParities, &mut rhs, &b_dev);
    let host_rhs = rhs.to_host_half(&index).unwrap();
    let d = host_rhs
        .as_slice()
        .iter()
        .zip(cpu_rhs.as_slice())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(d < 1e-12, "the right-hand sides differ by {d:e}");
    let mut a_dag = PvRatio::new(&gpu_pv, &gpu, EoForm::Right, &e, &o);
    let mut y = rhs.zeros_like();
    let info = Gcr::new(Stop::simple(1e-26, 4000), 8)
        .solve(&mut a_dag, None, &mut y, &rhs)
        .unwrap();
    let mut x = b_dev.zeros_like();
    adjoint_solution(&gpu, &gpu_pv, &DeviceParities, &mut x, &y, &b_dev);
    let x = x.to_host().unwrap();
    let mut r = x.zeros_like();
    cpu.mult(&mut cpu.work(), &mut r, &x, Variant::Dagger);
    r.aypx(&[C64::new(-1.0, 0.0)], &b);
    let (mut left, mut size) = ([0.0], [0.0]);
    r.norm_sqr(&mut left);
    b.norm_sqr(&mut size);
    eprintln!(
        "Ã† on the device: GCR(8) {} iterations, |D† x - b|²/|b|² = {:e} (on the CPU)",
        info.iterations,
        left[0] / size[0]
    );
    assert!(left[0] < 1e-20 * size[0]);
}

/// HMC with the multigrid of domain-wall fermions in the solves of the Ratio action on the
/// device, on all sites and even-odd, and as a cache: the trajectories of CG on the CPU, the
/// plaquette to rounding and ΔH to the tolerance of the solves.
#[test]
fn hmc_with_the_domain_wall_multigrid_on_the_device() {
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let cg = std::fs::read_to_string(examples.join("hmc_domainwall_pv.yaml"))
        .unwrap()
        .replace(
            "  lattice_size : [4, 4, 4, 8]",
            "  lattice_size : [4, 4, 4, 8]\n  backend      : BACKEND",
        )
        .replace("conf_04040408.txt", "CONF")
        .replace("number_of_trajectories  : 1", "number_of_trajectories  : 2");
    let mg = cg.replace(
        "    solver_type                   : CG",
        "    solver_type                   : GCR\n    preconditioner_type           : MG\n    number_of_kept_directions     : 8",
    ) + "  MultiGrid:\n    Setup:\n      filter_number_of_vectors : 8\n      filter_number_of_rounds  : 1\n    MultiGrid_Level1:\n      block_size             : [2, 2, 2, 4]\n      number_of_test_vectors : 16\n";
    assert_ne!(mg, cg);
    let rows = |out: &str| -> Vec<(f64, f64)> {
        out.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                if f.len() < 6 {
                    return None;
                }
                Some((f[1].parse().ok()?, f[2].parse().ok()?))
            })
            .collect()
    };
    let even_odd = |text: &str| {
        text.replace(
            "action_type     : Ratio",
            "action_type     : Ratio\n  operator_type   : Even_odd",
        )
    };
    for (name, cg, mg) in [
        ("hmc_mg_lex", cg.clone(), mg.clone()),
        ("hmc_mg_eo", even_odd(&cg), even_odd(&mg)),
        // P and the coarse parts kept from solve to solve.
        (
            "hmc_mg_cache",
            cg.clone(),
            mg.clone() + "  MultiGrid_HMC:\n    rebuild_iteration_growth : 1.5\n",
        ),
        // The cycle in single precision, the outer GCR in double.
        (
            "hmc_mg_fine_single",
            cg.clone(),
            mg.replace(
                "  MultiGrid:\n",
                "  MultiGrid:\n    fine_precision : single\n",
            ),
        ),
    ] {
        let reference = run_with_backend(&cg, "cpu", name);
        let gpu = run_with_backend(&mg, "cuda_bridge", name);
        let (a, b) = (rows(&reference), rows(&gpu));
        assert!(a.len() == 2 && a.len() == b.len(), "\n{reference}\n{gpu}");
        for ((p, dh), (q, dk)) in a.iter().zip(&b) {
            eprintln!("{name}: plaquette {p} and {q}, ΔH {dh} and {dk}");
            assert!(
                (p - q).abs() < 1e-12,
                "{name}: plaquette {p} by CG, {q} by the multigrid"
            );
            assert!(
                (dh - dk).abs() < 1e-7,
                "{name}: ΔH {dh} by CG, {dk} by the multigrid"
            );
        }
    }
}

/// HMC with mixed precision on the device (CG in single precision corrected in double), for
/// the Ratio action on all sites and even-odd and the Standard action: the trajectories of CG
/// in double precision on the CPU.
#[test]
fn hmc_with_mixed_precision_on_the_device() {
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let cg = std::fs::read_to_string(examples.join("hmc_domainwall_pv.yaml"))
        .unwrap()
        .replace(
            "  lattice_size : [4, 4, 4, 8]",
            "  lattice_size : [4, 4, 4, 8]\n  backend      : BACKEND",
        )
        .replace("conf_04040408.txt", "CONF")
        .replace("number_of_trajectories  : 1", "number_of_trajectories  : 2");
    let mixed = |text: &str| {
        text.replace(
            "    convergence_criterion_squared : 1.0e-28",
            "    convergence_criterion_squared : 1.0e-28\n    precision                     : mixed",
        )
    };
    let rows = |out: &str| -> Vec<(f64, f64)> {
        out.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                if f.len() < 6 {
                    return None;
                }
                Some((f[1].parse().ok()?, f[2].parse().ok()?))
            })
            .collect()
    };
    let even_odd = cg.replace(
        "action_type     : Ratio",
        "action_type     : Ratio\n  operator_type   : Even_odd",
    );
    let standard = cg.replace("action_type     : Ratio", "action_type     : Standard");
    assert_ne!(mixed(&cg), cg);
    for (name, text) in [
        ("hmc_mixed_lex", cg.clone()),
        ("hmc_mixed_eo", even_odd),
        ("hmc_mixed_standard", standard),
    ] {
        let reference = run_with_backend(&text, "cpu", name);
        let gpu = run_with_backend(&mixed(&text), "cuda_bridge", name);
        let (a, b) = (rows(&reference), rows(&gpu));
        assert!(a.len() == 2 && a.len() == b.len(), "\n{reference}\n{gpu}");
        for ((p, dh), (q, dk)) in a.iter().zip(&b) {
            eprintln!("{name}: plaquette {p} and {q}, ΔH {dh} and {dk}");
            assert!(
                (p - q).abs() < 1e-12,
                "{name}: plaquette {p} in double, {q} mixed"
            );
            assert!(
                (dh - dk).abs() < 1e-7,
                "{name}: ΔH {dh} in double, {dk} mixed"
            );
        }
    }
}

/// HMC with the domain-wall Ratio action on the device (`Main.backend : cuda_bridge`): Bridge++'s
/// test_HMC_Domainwall_Nf2_PV (stout links, Iwasaki action) gives the trajectories of the CPU,
/// the plaquette to rounding and ΔH to the tolerance of the solves.
#[test]
fn hmc_with_domain_wall_fermions_on_the_device() {
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let text = std::fs::read_to_string(examples.join("hmc_domainwall_pv.yaml"))
        .unwrap()
        .replace(
            "  lattice_size : [4, 4, 4, 8]",
            "  lattice_size : [4, 4, 4, 8]\n  backend      : BACKEND",
        )
        .replace("conf_04040408.txt", "CONF")
        .replace("number_of_trajectories  : 1", "number_of_trajectories  : 2");
    let rows = |out: &str| -> Vec<(f64, f64)> {
        out.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                if f.len() < 6 {
                    return None;
                }
                Some((f[1].parse().ok()?, f[2].parse().ok()?))
            })
            .collect()
    };
    let (cpu, gpu) = (
        run_with_backend(&text, "cpu", "hmc_dw"),
        run_with_backend(&text, "cuda_bridge", "hmc_dw"),
    );
    let (a, b) = (rows(&cpu), rows(&gpu));
    assert!(a.len() == 2 && a.len() == b.len(), "\n{cpu}\n{gpu}");
    assert!((a[0].0 - 0.62754640357188).abs() < 1e-11, "{gpu}");
    for ((p, dh), (q, dk)) in a.iter().zip(&b) {
        assert!(
            (p - q).abs() < 1e-12,
            "plaquette {p} on the CPU, {q} on the device"
        );
        assert!(
            (dh - dk).abs() < 1e-7,
            "ΔH {dh} on the CPU, {dk} on the device"
        );
    }

    // Two flavours of the same operator alone (action_type Standard), with Neff's α in its
    // solves and the initial guess of the force solves.
    // The even-odd ratio (`Action_F_alt_Ratio_eo`), with three initial guesses.
    let even_odd = text
        .replace(
            "action_type     : Ratio",
            "action_type     : Ratio\n  operator_type   : Even_odd",
        )
        .replace(
            "    convergence_criterion_squared : 1.0e-28",
            "    convergence_criterion_squared : 1.0e-28\n    number_of_initial_guess       : 3",
        );
    assert_ne!(even_odd, text);
    let (cpu, gpu) = (
        run_with_backend(&even_odd, "cpu", "hmc_dw_eo"),
        run_with_backend(&even_odd, "cuda_bridge", "hmc_dw_eo"),
    );
    let (a, b) = (rows(&cpu), rows(&gpu));
    assert!(a.len() == 2 && a.len() == b.len(), "\n{cpu}\n{gpu}");
    for ((p, dh), (q, dk)) in a.iter().zip(&b) {
        assert!(
            (p - q).abs() < 1e-12,
            "even-odd: plaquette {p} on the CPU, {q} on the device"
        );
        assert!(
            (dh - dk).abs() < 1e-7,
            "even-odd: ΔH {dh} on the CPU, {dk} on the device"
        );
    }

    let standard = text
        .replace("action_type     : Ratio", "action_type     : Standard")
        .replace("    ## neff_alpha : 0.6", "    neff_alpha : 0.6")
        .replace(
            "    convergence_criterion_squared : 1.0e-28",
            "    convergence_criterion_squared : 1.0e-28\n    number_of_initial_guess       : 3",
        );
    assert_ne!(standard, text);
    let (cpu, gpu) = (
        run_with_backend(&standard, "cpu", "hmc_dw_standard"),
        run_with_backend(&standard, "cuda_bridge", "hmc_dw_standard"),
    );
    let (a, b) = (rows(&cpu), rows(&gpu));
    assert!(a.len() == 2 && a.len() == b.len(), "\n{cpu}\n{gpu}");
    for ((p, dh), (q, dk)) in a.iter().zip(&b) {
        assert!(
            (p - q).abs() < 1e-12,
            "plaquette {p} on the CPU, {q} on the device"
        );
        assert!(
            (dh - dk).abs() < 1e-7,
            "ΔH {dh} on the CPU, {dk} on the device"
        );
    }
}

/// HMC with clover fermions on the device: Bridge++'s test_HMC_Clover_RHMC_Nf2p1 (stout links,
/// Iwasaki action; the Ratio, Standard and Rational actions) gives the trajectory of the CPU and of
/// Bridge++; so does it with the Standard action even-odd (ln det T_oo and its force), and with
/// Wilson fermions in an even-odd Ratio.
#[test]
fn hmc_with_wilson_and_clover_fermions_on_the_device() {
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let text = std::fs::read_to_string(examples.join("hmc_clover_rhmc_nf2p1.yaml"))
        .unwrap()
        .replace(
            "  lattice_size : [4, 4, 4, 8]",
            "  lattice_size : [4, 4, 4, 8]\n  backend      : BACKEND",
        )
        .replace("conf_04040408.txt", "CONF");
    let rows = |out: &str| -> Vec<(f64, f64)> {
        out.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                if f.len() < 6 {
                    return None;
                }
                Some((f[1].parse().ok()?, f[2].parse().ok()?))
            })
            .collect()
    };
    let same = |text: &str, name: &str| -> Vec<(f64, f64)> {
        let (cpu, gpu) = (
            run_with_backend(text, "cpu", name),
            run_with_backend(text, "cuda_bridge", name),
        );
        let (a, b) = (rows(&cpu), rows(&gpu));
        assert!(!a.is_empty() && a.len() == b.len(), "{name}\n{cpu}\n{gpu}");
        for ((p, dh), (q, dk)) in a.iter().zip(&b) {
            eprintln!("{name}: plaquette {p} / {q}, ΔH {dh} / {dk}");
            assert!(
                (p - q).abs() < 1e-12,
                "{name}: plaquette {p} on the CPU, {q} on the device"
            );
            assert!(
                (dh - dk).abs() < 1e-7,
                "{name}: ΔH {dh} on the CPU, {dk} on the device"
            );
        }
        b
    };
    let b = same(&text, "hmc_clover");
    assert!(
        (b[0].0 - 0.627668186553629).abs() < 1e-11,
        "Bridge++'s plaquette"
    );

    let even_odd = text.replace(
        "  action_type     : Standard\n",
        "  action_type     : Standard\n  operator_type   : Even_odd\n",
    );
    assert_ne!(even_odd, text);
    same(&even_odd, "hmc_clover_eo");

    let wilson = text
        .replace(
            "    fermion_type       : Clover\n",
            "    fermion_type       : Wilson\n",
        )
        .replace("    clover_coefficient : 1.0\n", "")
        .replace(
            "  action_type     : Ratio\n",
            "  action_type     : Ratio\n  operator_type   : Even_odd\n",
        );
    same(&wilson, "hmc_wilson_eo");
}

const MULTIGRID: &str = "MultiGrid:
  coarse_precision : single
  Setup:
    filter_number_of_vectors   : 8
    filter_number_of_rounds    : 1
    filter_number_in_block     : 4
    bootstrap_number_of_rounds : 1
    bootstrap_number_of_steps  : 2
    seed                       : 21
  Smoother:
    smoother_type   : Chebyshev
    number_of_steps : 4
    interval_ratio  : 30
  MultiGrid_Level1:
    block_size             : [2, 2, 2, 4]
    number_of_test_vectors : 16
    Coarse_Solver_D:
      solver_type                   : BiCGStab
      maximum_number_of_iteration   : 200
      convergence_criterion_squared : 1.0e-2
    Coarse_Solver_PV:
      solver_type                   : MINRES
      maximum_number_of_iteration   : 200
      convergence_criterion_squared : 1.0e-2
";

/// The multigrid of domain-wall fermions with `backend : cuda_bridge`, all of it on the
/// device. In double precision above the coarse lattice the test vectors are those of the CPU
/// to rounding, and so are the cycles: as many iterations. In single precision (the default)
/// they are not; the correlators are the same either way.
#[test]
fn domain_wall_multigrid_runs_on_the_device() {
    let text = MDWF_RUN
        .replace("RHS", "4")
        .replace("SOLVER", "GCR")
        .replace("OPERATOR", "PV_left")
        .replace("ALPHA", "0.6")
        .replace("PRECONDITIONER", "MG")
        + MULTIGRID;
    same_tables(&text, "mdwf_mg", 1e-6);

    // The prolongation kept in single precision, not in half precision.
    let single = text.replace(
        "  coarse_precision : single",
        "  coarse_precision : single\n  basis_precision  : single",
    );
    same_tables(&single, "mdwf_mg_single", 1e-6);

    let text = text.replace(
        "  coarse_precision : single",
        "  coarse_precision : single\n  fine_precision   : double\n  basis_precision  : single",
    );
    let (cpu, gpu) = (
        run_with_backend(&text, "cpu", "mdwf_mg_double"),
        run_with_backend(&text, "cuda_bridge", "mdwf_mg_double"),
    );
    // As many iterations of the outer solver, for every solve.
    let iterations = |out: &str| -> Vec<String> {
        out.lines()
            .filter_map(|line| {
                line.split(" iterations")
                    .next()
                    .filter(|_| line.contains(" iterations"))
            })
            .filter_map(|head| head.split_whitespace().last().map(str::to_string))
            .collect()
    };
    assert_eq!(iterations(&cpu), iterations(&gpu), "\n{cpu}\n{gpu}");
    assert!(!iterations(&cpu).is_empty(), "{cpu}");
    same_tables(&text, "mdwf_mg_double", 1e-6);

    // A level below the coarse lattice, on the device as well.
    let levels = MDWF_RUN
        .replace("RHS", "4")
        .replace("SOLVER", "GCR")
        .replace("OPERATOR", "PV_left")
        .replace("ALPHA", "0.6")
        .replace("PRECONDITIONER", "MG")
        + MULTIGRID
        + "  MultiGrid_Level2:\n    super_block_size : 2\n";
    let gpu = run_with_backend(&levels, "cuda_bridge", "mdwf_mg_levels");
    assert!(
        gpu.lines()
            .any(|l| l.contains("levels below the coarse lattice [2]")
                && l.contains("on the device:")),
        "{gpu}"
    );
    same_tables(&levels, "mdwf_mg_levels", 1e-6);

    // The same levels for the Pauli-Villars factor too.
    let both = levels.replace(
        "  coarse_precision : single",
        "  coarse_precision : single\n  levels_for_pauli_villars : true",
    );
    let gpu = run_with_backend(&both, "cuda_bridge", "mdwf_mg_levels_pv");
    assert!(
        gpu.contains("the same levels for the Pauli-Villars factor on the device:"),
        "{gpu}"
    );
    same_tables(&both, "mdwf_mg_levels_pv", 1e-6);

    // The second factor by BiCGStab on its exact adjoint, all of it on the device.
    let exact = levels.replace(
        "    Coarse_Solver_PV:\n      solver_type                   : MINRES",
        "    Coarse_Solver_PV:\n      solver_type                   : BiCGStab",
    );
    assert_ne!(exact, levels);
    same_tables(&exact, "mdwf_mg_exact_adjoint", 1e-6);
}

/// What the domain-wall operator on the device gives for `psi` on the lattice of `lat`: D, D†,
/// Γ5 D (`FermionOp`) on the whole lattice; the even-odd Pauli-Villars operator Ã in both forms
/// on the even part of `psi`, and its adjoint; and |psi|², summed over the processes. The fields
/// come back on the host, the even ones put into a whole field.
fn mdwf_results(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    psi: &Psi,
    params: MdwfParams,
    alpha: f64,
) -> (Vec<Psi>, f64) {
    use ruttice::fermion::SplitParity;
    let mut d = CudaMdwf::<f64>::new(lat, params).unwrap();
    d.set_gauge(u).unwrap();
    d.set_alpha(alpha);
    let mut pv = CudaMdwf::<f64>::new(
        lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    )
    .unwrap();
    pv.set_gauge(u).unwrap();
    pv.set_alpha(alpha);
    let x = DeviceSpinor::from_host(psi).unwrap();
    let mut out = Vec::new();
    for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
        let mut y = x.zeros_like();
        Bound::new(&d, variant).mult(&mut y, &x);
        out.push(y.to_host().unwrap());
    }
    let mut x_e = DeviceParities.half_like(&x, Parity::Even);
    DeviceParities.extract(&mut x_e, &x, Parity::Even);
    let odd = DeviceParities.half_like(&x, Parity::Odd);
    for form in [EoForm::Right, EoForm::Left] {
        let mut a = PvRatio::new(&d, &pv, form, &x_e, &odd);
        let mut y = x_e.zeros_like();
        a.mult(&mut y, &x_e);
        let mut z = x_e.zeros_like();
        a.mult_dag(&mut z, &x_e);
        for half in [y, z] {
            let mut whole = x.zeros_like();
            DeviceParities.insert(&mut whole, &half);
            out.push(whole.to_host().unwrap());
        }
    }
    let mut n = [0.0];
    x.norm_sqr(&mut n);
    (out, n[0])
}

/// On several processes, each with its device (here threads of a `LocalComm` on one device),
/// the domain-wall operator gives what it gives on one, bit for bit: the faces carry the half
/// spinors of the legs and the surface sums every leg of a site in the order of one process.
/// Cut along t, along x and t, and along every direction (local extents 2 in x, y, z).
#[test]
fn mdwf_on_several_processes_is_mdwf_on_one() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 411);
    for (boundary, alpha) in [(ANTIPERIODIC_T, 1.0), (MIXED, 1.3)] {
        let params = MdwfParams {
            ls: 4,
            m5: 1.8,
            mass: 0.05,
            b: 1.5,
            c: 0.5,
            boundary,
        };
        let psi = random_fifth(&whole, params.ls, 412);
        let (one, norm) = mdwf_results(&whole, &u, &psi, params, alpha);
        for grid in [[1, 1, 1, 2], [2, 1, 1, 2], [2, 2, 2, 2]] {
            let parts = ruttice::LocalComm::run(global, grid, |lat| {
                let u = ruttice::local_part(&u, &lat);
                let psi = ruttice::local_part(&psi, &lat);
                let (fields, n) = mdwf_results(&lat, &u, &psi, params, alpha);
                (
                    fields
                        .iter()
                        .map(|f| f.as_slice().to_vec())
                        .collect::<Vec<_>>(),
                    n,
                    lat,
                )
            });
            for (k, single) in one.iter().enumerate() {
                let pieces: Vec<Psi> = parts
                    .iter()
                    .map(|(fields, _, lat)| {
                        let mut f = Field::<f64, Spinor>::with_shape(lat, 24, params.ls, 1, 1);
                        f.as_mut_slice().copy_from_slice(&fields[k]);
                        f
                    })
                    .collect();
                let several = ruttice::assemble(&whole, &pieces);
                assert!(
                    several.as_slice() == single.as_slice(),
                    "result {k} on the grid {grid:?} ({boundary:?}, α {alpha}): differs by {}",
                    worst(&several, single)
                );
            }
            for (_, n, _) in &parts {
                assert!(
                    (n - norm).abs() <= 1e-13 * norm,
                    "|psi|² {n} against {norm}"
                );
            }
        }
    }
}

/// The Wilson and clover operators on `lat` for the comparisons of several processes with one:
/// D, D† and γ5 D of both, the hops between the parities, T and T⁻¹, CGNE on the clover Schur
/// complement (as whole fields on the host), the derivatives of HMC (the hops, the clover term
/// of two pairs, the even-odd clover term with ln det), and the sums ln |det T| on each parity
/// and |ψ|², with the CG iterations.
fn wilson_clover_results(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    fields: &[Psi],
    boundary: [Boundary; 4],
) -> (Vec<Psi>, Vec<Field<f64, Gauge>>, Vec<f64>) {
    use ruttice::backend::cuda_bridge::DeviceLinks;
    use ruttice::fermion::SplitParity;
    use ruttice::hmc::{EvenOddForceOp, ForceOp, LinkField};
    let du = DeviceLinks::from_host(u).unwrap();
    let wp = WilsonParams {
        kappa: 0.13,
        boundary,
    };
    let cp = CloverParams {
        kappa: 0.12,
        csw: 1.7,
        boundary,
    };
    let mut w = CudaWilson::<f64>::new(lat, wp).unwrap();
    ForceOp::set_links(&mut w, &du).unwrap();
    let mut c = CudaClover::<f64>::new(lat, cp).unwrap();
    ForceOp::set_links(&mut c, &du).unwrap();
    let dev: Vec<DeviceSpinor<f64>> = fields
        .iter()
        .map(|f| DeviceSpinor::from_host(f).unwrap())
        .collect();
    let x = &dev[0];
    let mut spinors = Vec::new();
    for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
        let mut y = x.zeros_like();
        w.mult(&mut None, &mut y, x, variant);
        spinors.push(y.to_host().unwrap());
        c.mult(&mut None, &mut y, x, variant);
        spinors.push(y.to_host().unwrap());
    }
    let whole = |half: &DeviceSpinor<f64>| {
        let mut f = x.zeros_like();
        DeviceParities.insert(&mut f, half);
        f.to_host().unwrap()
    };
    let mut x_o = DeviceParities.half_like(x, Parity::Odd);
    DeviceParities.extract(&mut x_o, x, Parity::Odd);
    for dagger in [Dagger::No, Dagger::Yes] {
        let mut y = DeviceParities.half_like(x, Parity::Even);
        w.hop(&mut (), &mut y, &x_o, Parity::Even, dagger);
        spinors.push(whole(&y));
    }
    for inverse in [false, true] {
        let mut y = x.zeros_like();
        c.site(&mut y, x, inverse);
        spinors.push(y.to_host().unwrap());
    }
    let mut x_e = DeviceParities.half_like(x, Parity::Even);
    DeviceParities.extract(&mut x_e, x, Parity::Even);
    let odd = DeviceParities.half_like(x, Parity::Odd);
    let mut schur = Schur::new(&c, Parity::Even, &x_e, &odd);
    let mut sol = x_e.zeros_like();
    let info = solve_cgne(
        &mut Cg::new(Stop::simple(1e-20, 2000)),
        &mut schur,
        &mut sol,
        &x_e,
    )
    .unwrap();
    spinors.push(whole(&sol));

    let kc = cp.kappa * cp.csw;
    let pairs = [(1.0, &dev[1], &dev[2]), (-0.3, &dev[3], &dev[0])];
    let links = [
        w.derivative(&dev[1], &dev[2]),
        c.derivative(&dev[1], &dev[2]),
        c.clover_derivative(kc, &du, &pairs),
        c.clover_derivative_eo(kc, &du, Parity::Odd, &dev[1], &dev[2]),
    ]
    .iter()
    .map(LinkField::to_host)
    .collect();
    let mut n = [0.0];
    x.norm_sqr(&mut n);
    let sums = vec![
        c.log_det_diag(lat, Parity::Even),
        c.log_det_diag(lat, Parity::Odd),
        n[0],
        info.iterations as f64,
    ];
    (spinors, links, sums)
}

/// The Wilson and clover operators on several processes (threads of a `LocalComm` on one
/// device) give what they give on one bit for bit: D (the clover D in one pass from T ψ, the
/// legs added in the order of one process), the hops between the parities, T, the solution of
/// CGNE on the Schur complement, the derivatives of HMC (the leaves read the links and the
/// insertions extended); the sums the same up to their order, CG the same iterations.
#[test]
fn wilson_and_clover_on_several_processes_are_those_on_one() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 501);
    let fields: Vec<Psi> = (0..4).map(|k| random_spinor(&whole, 1, 502 + k)).collect();
    for boundary in [ANTIPERIODIC_T, MIXED] {
        let (spinors, links, sums) = wilson_clover_results(&whole, &u, &fields, boundary);
        for grid in [[1, 1, 1, 2], [2, 1, 1, 2], [2, 2, 2, 2]] {
            let parts = ruttice::LocalComm::run(global, grid, |lat| {
                let u = ruttice::local_part(&u, &lat);
                let f: Vec<Psi> = fields
                    .iter()
                    .map(|f| ruttice::local_part(f, &lat))
                    .collect();
                let (s, l, sums) = wilson_clover_results(&lat, &u, &f, boundary);
                let raw = |v: &[Psi]| v.iter().map(|f| f.as_slice().to_vec()).collect::<Vec<_>>();
                let raw_links: Vec<Vec<f64>> = l.iter().map(|f| f.as_slice().to_vec()).collect();
                (raw(&s), raw_links, sums, lat)
            });
            // The last spinor is the solution of CGNE: the sums over the processes are in another
            // order, so it agrees to the tolerance, not bit for bit.
            let solves = spinors.len() - 1;
            for (k, single) in spinors.iter().enumerate() {
                let pieces: Vec<Psi> = parts
                    .iter()
                    .map(|(s, _, _, lat)| {
                        let mut f = Field::<f64, Spinor>::with_shape(lat, 24, 1, 1, 1);
                        f.as_mut_slice().copy_from_slice(&s[k]);
                        f
                    })
                    .collect();
                let several = ruttice::assemble(&whole, &pieces);
                if k == solves {
                    let scale = single.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    assert!(
                        worst(&several, single) < 1e-10 * scale,
                        "CGNE on the grid {grid:?} ({boundary:?}): differs by {}",
                        worst(&several, single)
                    );
                    continue;
                }
                assert!(
                    several.as_slice() == single.as_slice(),
                    "spinor result {k} on the grid {grid:?} ({boundary:?}): differs by {}",
                    worst(&several, single)
                );
            }
            for (k, single) in links.iter().enumerate() {
                let pieces: Vec<Field<f64, Gauge>> = parts
                    .iter()
                    .map(|(_, l, _, lat)| {
                        let mut f = Field::<f64, Gauge>::gauge(lat);
                        f.as_mut_slice().copy_from_slice(&l[k]);
                        f
                    })
                    .collect();
                let several = ruttice::assemble(&whole, &pieces);
                assert!(
                    several.as_slice() == single.as_slice(),
                    "derivative {k} on the grid {grid:?} ({boundary:?}) differs"
                );
            }
            for (_, _, s, _) in &parts {
                for (a, b) in s[..3].iter().zip(&sums[..3]) {
                    assert!(
                        (a - b).abs() <= 1e-12 * b.abs(),
                        "sums {s:?} against {sums:?} on the grid {grid:?}"
                    );
                }
                // CG's iterations, as in `cgne_on_several_processes_is_cgne_on_one`.
                assert!(
                    (s[3] - sums[3]).abs() * 500.0 <= sums[3].max(500.0),
                    "{} iterations against {} on the grid {grid:?}",
                    s[3],
                    sums[3]
                );
            }
        }
    }
}

/// CGNE on the even-odd Pauli-Villars operator (left form, α 1.3) from the even part of `b`:
/// the iterations and the solution, put into a whole field on the host.
fn mdwf_cgne(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    b: &Psi,
    params: MdwfParams,
) -> (usize, Psi) {
    use ruttice::fermion::SplitParity;
    let mut d = CudaMdwf::<f64>::new(lat, params).unwrap();
    d.set_gauge(u).unwrap();
    d.set_alpha(1.3);
    let mut pv = CudaMdwf::<f64>::new(
        lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    )
    .unwrap();
    pv.set_gauge(u).unwrap();
    pv.set_alpha(1.3);
    let whole = DeviceSpinor::from_host(b).unwrap();
    let mut b_e = DeviceParities.half_like(&whole, Parity::Even);
    DeviceParities.extract(&mut b_e, &whole, Parity::Even);
    let odd = DeviceParities.half_like(&whole, Parity::Odd);
    let mut a = PvRatio::new(&d, &pv, EoForm::Left, &b_e, &odd);
    let mut x = b_e.zeros_like();
    let info = solve_cgne(
        &mut Cg::new(Stop::simple(1e-22, 20000)),
        &mut a,
        &mut x,
        &b_e,
    )
    .unwrap();
    let mut out = whole.zeros_like();
    DeviceParities.insert(&mut out, &x);
    (info.iterations, out.to_host().unwrap())
}

/// The solvers on several processes: CGNE takes as many iterations as on one (but one), and its
/// solution is that of one up to the order of the sums over the processes.
#[test]
fn cgne_on_several_processes_is_cgne_on_one() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 421);
    let params = MdwfParams {
        ls: 4,
        m5: 1.8,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let b = random_fifth(&whole, params.ls, 422);
    let (iterations, x) = mdwf_cgne(&whole, &u, &b, params);
    for grid in [[1, 1, 1, 2], [2, 1, 1, 2]] {
        let parts = ruttice::LocalComm::run(global, grid, |lat| {
            let u = ruttice::local_part(&u, &lat);
            let b = ruttice::local_part(&b, &lat);
            let (n, x) = mdwf_cgne(&lat, &u, &b, params);
            (n, x.as_slice().to_vec(), lat)
        });
        let pieces: Vec<Psi> = parts
            .iter()
            .map(|(_, x, lat)| {
                let mut f = Field::<f64, Spinor>::with_shape(lat, 24, params.ls, 1, 1);
                f.as_mut_slice().copy_from_slice(x);
                f
            })
            .collect();
        let several = ruttice::assemble(&whole, &pieces);
        let scale = x.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let diff = worst(&several, &x);
        eprintln!("grid {grid:?}: {} iterations (one process {iterations}), |Δx| {diff:.1e} of {scale:.1e}", parts[0].0);
        // The sums over the processes are in another order (the operator is that of one process
        // bit for bit), which moves the iteration at which |r|² first falls below the tolerance
        // by a few in a thousand: 1247 and 1248 against 1246 have been seen.
        for (n, _, _) in &parts {
            assert!(
                n.abs_diff(iterations) * 500 <= iterations,
                "{n} iterations on the grid {grid:?}"
            );
        }
        assert!(
            diff < 1e-9 * scale,
            "solution on the grid {grid:?}: differs by {diff}"
        );
    }
}

/// What HMC on the device does with the links, on the lattice of `lat`: two steps of stout
/// smearing, the chain rule of the force through them (of a field of matrices `m`), the force
/// of the rectangle action, the derivative of the domain-wall operator (α 1.3) for two spinors
/// ζ, η; and the sums: plaquette, action, kinetic energy of `m`.
fn hmc_parts(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    m: &Field<f64, Gauge>,
    zeta: &Psi,
    eta: &Psi,
    params: MdwfParams,
) -> (Vec<Field<f64, Gauge>>, [f64; 3]) {
    use ruttice::backend::cuda_bridge::DeviceLinks;
    use ruttice::hmc::LinkField;
    let smearing = ruttice::gauge::GaugeSmearing {
        smear: ruttice::gauge::Smear::Stout(ruttice::gauge::smearing::Stout::uniform(0.1)),
        steps: 2,
    };
    let du = <DeviceLinks as LinkField>::from_host(u);
    let dm = <DeviceLinks as LinkField>::from_host(m);
    let levels = du.smear_levels(&smearing);
    let force = DeviceLinks::smeared_force(&dm, &du, Some(&smearing), &levels);
    let gauge = du.gauge_force(2.3, 3.648, -0.331);
    let mut d = CudaMdwf::<f64>::new(lat, params).unwrap();
    d.set_gauge(u).unwrap();
    d.set_alpha(1.3);
    let derivative = d
        .derivative_on_device(
            &DeviceSpinor::from_host(zeta).unwrap(),
            &DeviceSpinor::from_host(eta).unwrap(),
        )
        .unwrap();
    let fields = [&levels[1], &force, &gauge, &derivative]
        .iter()
        .map(|f| LinkField::to_host(*f))
        .collect();
    let sums = [
        du.plaquette(),
        du.gauge_action(2.3, 3.648, -0.331),
        dm.kinetic_energy(),
    ];
    (fields, sums)
}

/// HMC's operations on the links on several processes, each with its device: the fields are
/// those of one process bit for bit (the halos of the extended links carry the neighbours,
/// corners included), the sums the same up to their order.
#[test]
fn hmc_on_several_processes_is_hmc_on_one() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 431);
    let m = random_gauge(&whole, 432);
    let params = MdwfParams {
        ls: 4,
        m5: 1.8,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: MIXED,
    };
    let (zeta, eta) = (random_fifth(&whole, 4, 433), random_fifth(&whole, 4, 434));
    let (one, sums) = hmc_parts(&whole, &u, &m, &zeta, &eta, params);
    for grid in [[1, 1, 1, 2], [2, 1, 1, 2]] {
        let parts = ruttice::LocalComm::run(global, grid, |lat| {
            let local = |f: &Field<f64, Gauge>| ruttice::local_part(f, &lat);
            let (fields, s) = hmc_parts(
                &lat,
                &local(&u),
                &local(&m),
                &ruttice::local_part(&zeta, &lat),
                &ruttice::local_part(&eta, &lat),
                params,
            );
            (
                fields
                    .iter()
                    .map(|f| f.as_slice().to_vec())
                    .collect::<Vec<_>>(),
                s,
                lat,
            )
        });
        for (k, single) in one.iter().enumerate() {
            let pieces: Vec<Field<f64, Gauge>> = parts
                .iter()
                .map(|(fields, _, lat)| {
                    let mut f = Field::<f64, Gauge>::gauge(lat);
                    f.as_mut_slice().copy_from_slice(&fields[k]);
                    f
                })
                .collect();
            let several = ruttice::assemble(&whole, &pieces);
            let diff = several
                .as_slice()
                .iter()
                .zip(single.as_slice())
                .fold(0.0f64, |w, (a, b)| w.max((a - b).abs()));
            assert!(
                diff == 0.0,
                "field {k} on the grid {grid:?}: differs by {diff}"
            );
        }
        for (_, s, _) in &parts {
            for (a, b) in s.iter().zip(&sums) {
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs(),
                    "sums {s:?} against {sums:?}"
                );
            }
        }
    }
}

/// The multigrid's coarse lattice on the device for the lattice of `lat`: P from four test
/// vectors (whole fields, their even parts taken), the coarse operator probed, and fine fields
/// of what it does to P† of the even part of `psi`: D_c, Γ5 D_c Γ5, its exact adjoint D_c† and
/// the Schur complement, prolonged back.
fn coarse_results(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    vectors: &[Psi],
    psi: &Psi,
    params: MdwfParams,
) -> Vec<Psi> {
    use ruttice::backend::cuda_bridge::DeviceBasis;
    use ruttice::fermion::SplitParity;
    use ruttice::solver::preconditioner::multigrid::Aggregation;
    let mut d = CudaMdwf::<f64>::new(lat, params).unwrap();
    d.set_gauge(u).unwrap();
    let mut pv = CudaMdwf::<f64>::new(
        lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    )
    .unwrap();
    pv.set_gauge(u).unwrap();
    let first = DeviceSpinor::from_host(&vectors[0]).unwrap();
    let mut y = first.zeros_with(vectors.len());
    for (k, v) in vectors.iter().enumerate() {
        y.assign(&[k], &DeviceSpinor::from_host(v).unwrap());
    }
    let mut y_e = DeviceParities.half_like(&y, Parity::Even);
    DeviceParities.extract(&mut y_e, &y, Parity::Even);
    let agg = Aggregation::new(lat, [2, 2, 2, 2]);
    let basis = DeviceBasis::pv_eo(&d, &pv, &agg, &y_e, false).unwrap();
    let coarse = d.coarse_operator_on_device::<f64>(&basis, true).unwrap();
    let x = DeviceSpinor::from_host(psi).unwrap();
    let mut x_e = DeviceParities.half_like(&x, Parity::Even);
    DeviceParities.extract(&mut x_e, &x, Parity::Even);
    let mut c = basis.coarse_field::<f64>(1);
    basis.restrict(&mut c, &x_e);
    let adjoint = coarse.adjoint().unwrap();
    let mut outs = Vec::new();
    for k in 0..4 {
        let mut out = c.zeros_like();
        match k {
            0 => coarse.apply(&mut out, &c, false),
            1 => coarse.apply(&mut out, &c, true),
            2 => adjoint.apply(&mut out, &c, false),
            _ => {
                let mut odd = c.zeros_like();
                coarse.schur(&mut out, &c, &mut odd);
            }
        }
        let mut fine = x_e.zeros_like();
        basis.prolong(&mut fine, &out);
        let mut whole = x.zeros_like();
        DeviceParities.insert(&mut whole, &fine);
        outs.push(whole.to_host().unwrap());
    }
    outs
}

/// The multigrid on several processes, each with its device: P, the coarse operator probed
/// across the cuts (the legs from the faces of the probing vectors) and its hops through the
/// ghosts of the coarse lattice give what one process gives, bit for bit.
#[test]
fn coarse_lattice_on_several_processes_is_that_on_one() {
    let global = [4, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 441);
    let params = MdwfParams {
        ls: 4,
        m5: 1.8,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let vectors: Vec<Psi> = (0..4).map(|k| random_fifth(&whole, 4, 442 + k)).collect();
    let psi = random_fifth(&whole, 4, 450);
    let one = coarse_results(&whole, &u, &vectors, &psi, params);
    for grid in [[1, 1, 1, 2], [2, 1, 1, 2]] {
        let parts = ruttice::LocalComm::run(global, grid, |lat| {
            let local: Vec<Psi> = vectors
                .iter()
                .map(|v| ruttice::local_part(v, &lat))
                .collect();
            let fields = coarse_results(
                &lat,
                &ruttice::local_part(&u, &lat),
                &local,
                &ruttice::local_part(&psi, &lat),
                params,
            );
            (
                fields
                    .iter()
                    .map(|f| f.as_slice().to_vec())
                    .collect::<Vec<_>>(),
                lat,
            )
        });
        for (k, single) in one.iter().enumerate() {
            let pieces: Vec<Psi> = parts
                .iter()
                .map(|(fields, lat)| {
                    let mut f = Field::<f64, Spinor>::with_shape(lat, 24, params.ls, 1, 1);
                    f.as_mut_slice().copy_from_slice(&fields[k]);
                    f
                })
                .collect();
            let several = ruttice::assemble(&whole, &pieces);
            assert!(
                several.as_slice() == single.as_slice(),
                "result {k} on the grid {grid:?}: differs by {}",
                worst(&several, single)
            );
        }
    }
}

/// The levels below the coarse lattice on the device for the lattice of `lat`: the V-cycle of
/// the quark factor (super-blocks of 2, then the last level dense) and the |H|⁻¹ levels of the
/// Pauli-Villars factor, applied to P† of the even part of `psi` and prolonged back.
fn level_results(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    vectors: &[Psi],
    psi: &Psi,
    params: MdwfParams,
) -> Vec<Psi> {
    use ruttice::backend::cuda_bridge::{DeviceAbsMultilevel, DeviceBasis, DeviceMultilevel};
    use ruttice::fermion::SplitParity;
    use ruttice::solver::preconditioner::multigrid::Aggregation;
    let mut d = CudaMdwf::<f64>::new(lat, params).unwrap();
    d.set_gauge(u).unwrap();
    let mut pv = CudaMdwf::<f64>::new(
        lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    )
    .unwrap();
    pv.set_gauge(u).unwrap();
    let first = DeviceSpinor::from_host(&vectors[0]).unwrap();
    let mut y = first.zeros_with(vectors.len());
    for (k, v) in vectors.iter().enumerate() {
        y.assign(&[k], &DeviceSpinor::from_host(v).unwrap());
    }
    let mut y_e = DeviceParities.half_like(&y, Parity::Even);
    DeviceParities.extract(&mut y_e, &y, Parity::Even);
    let agg = Aggregation::new(lat, [2, 2, 2, 2]);
    let basis = DeviceBasis::pv_eo(&d, &pv, &agg, &y_e, false).unwrap();
    let coarse_d = d.coarse_operator_on_device::<f64>(&basis, true).unwrap();
    let coarse_pv = pv.coarse_operator_on_device::<f64>(&basis, true).unwrap();
    let tests = basis.pv_eo_coarse_vectors(&d, &pv, &y_e).unwrap();
    let ml = DeviceMultilevel::build(&coarse_d, agg.coarse_lattice(), &tests, &[2], &[]).unwrap();
    let abs = DeviceAbsMultilevel::build(&coarse_pv, agg.coarse_lattice(), &tests, &[2]).unwrap();
    let x = DeviceSpinor::from_host(psi).unwrap();
    let mut x_e = DeviceParities.half_like(&x, Parity::Even);
    DeviceParities.extract(&mut x_e, &x, Parity::Even);
    let mut c = basis.coarse_field::<f64>(1);
    basis.restrict(&mut c, &x_e);
    let mut outs = Vec::new();
    for k in 0..2 {
        let mut out = c.zeros_like();
        match k {
            0 => ml.apply(&mut out, &c),
            _ => abs.add(&mut out, &c),
        }
        let mut fine = x_e.zeros_like();
        basis.prolong(&mut fine, &out);
        let mut whole = x.zeros_like();
        DeviceParities.insert(&mut whole, &fine);
        outs.push(whole.to_host().unwrap());
    }
    outs
}

/// The levels below the coarse lattice on several processes: super-blocks within a process,
/// their operators probed through the ghosts (colours by the global coordinates), the last level
/// dense over all of them: what one process gives, bit for bit.
#[test]
fn levels_on_several_processes_are_those_on_one() {
    let global = [8, 4, 4, 8];
    let whole = Lattice::single_node(global);
    let u = random_gauge(&whole, 461);
    let params = MdwfParams {
        ls: 4,
        m5: 1.8,
        mass: 0.05,
        b: 1.5,
        c: 0.5,
        boundary: ANTIPERIODIC_T,
    };
    let vectors: Vec<Psi> = (0..4).map(|k| random_fifth(&whole, 4, 462 + k)).collect();
    let psi = random_fifth(&whole, 4, 470);
    let one = level_results(&whole, &u, &vectors, &psi, params);
    for grid in [[1, 1, 1, 2], [2, 1, 1, 2]] {
        let parts = ruttice::LocalComm::run(global, grid, |lat| {
            let local: Vec<Psi> = vectors
                .iter()
                .map(|v| ruttice::local_part(v, &lat))
                .collect();
            let fields = level_results(
                &lat,
                &ruttice::local_part(&u, &lat),
                &local,
                &ruttice::local_part(&psi, &lat),
                params,
            );
            (
                fields
                    .iter()
                    .map(|f| f.as_slice().to_vec())
                    .collect::<Vec<_>>(),
                lat,
            )
        });
        for (k, single) in one.iter().enumerate() {
            let pieces: Vec<Psi> = parts
                .iter()
                .map(|(fields, lat)| {
                    let mut f = Field::<f64, Spinor>::with_shape(lat, 24, params.ls, 1, 1);
                    f.as_mut_slice().copy_from_slice(&fields[k]);
                    f
                })
                .collect();
            let several = ruttice::assemble(&whole, &pieces);
            assert!(
                several.as_slice() == single.as_slice(),
                "result {k} on the grid {grid:?}: differs by {}",
                worst(&several, single)
            );
        }
    }
}
