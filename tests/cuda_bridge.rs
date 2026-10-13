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
    let (lat, ls) = (psi.lattice(), psi.ls());
    let half = |parity| {
        let mut f = HalfPsi::with_shape(lat, 24, ls, 1, 1);
        f.set_parity(parity);
        f
    };
    let device = |parity| DeviceSpinor::zeros_half(lat, ls, parity).unwrap();
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

    let mut r = b.zeros_like();
    cpu.mult(&mut cpu.work(), &mut r, &x, Variant::D);
    r.aypx(&[C64::new(-1.0, 0.0)], b);
    let (mut rr, mut bb) = ([0.0], [0.0]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    assert!(
        rr[0] / bb[0] < 1e-18,
        "{what}: |b - D x|² / |b|² = {:e} after {} iterations",
        rr[0] / bb[0],
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
  number_of_rhs_per_solve : 1
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
    for (solver, operator, alpha, preconditioner) in [
        ("GCR", "PV_right", "1.0", "None"),
        ("GCR", "PV_left", "0.6", "None"),
        ("BiCGStab", "PV_left", "0.6", "None"),
        ("FGMRES", "PV_right", "1.0", "Chebyshev"),
        ("CG", "Normal_left", "1.0", "None"),
    ] {
        let text = MDWF_RUN
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
