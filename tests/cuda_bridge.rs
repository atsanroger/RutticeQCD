//! The CUDA kernels against the operators on the CPU (feature `cuda-bridge`; needs a GPU).
#![cfg(feature = "cuda-bridge")]

use ruttice::backend::cuda_bridge::{CudaMdwf, CudaWilson, DeviceSpinor};
use ruttice::fermion::{
    Dagger, EvenOdd, FermionOp, Mdwf, MdwfParams, Variant, Wilson, WilsonParams,
};
use ruttice::gauge::random_gauge;
use ruttice::{
    extract_parity, random_spinor, Boundary, Field, Half, Lattice, Parity, Spinor, Vector,
};

/// The Wilson operator on the device is the one on the CPU, for periodic, antiperiodic and
/// Dirichlet boundaries and a lattice whose sites do not fill the blocks of 32.
#[test]
fn wilson_on_the_device_is_wilson_on_the_cpu() {
    for (dims, boundary) in [
        ([4, 4, 4, 8], [Boundary::Periodic; 4]),
        (
            [4, 4, 4, 8],
            [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        ),
        (
            [6, 4, 2, 6],
            [
                Boundary::Antiperiodic,
                Boundary::Periodic,
                Boundary::Dirichlet,
                Boundary::Antiperiodic,
            ],
        ),
    ] {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 301);
        let psi = random_spinor(&lat, 1, 302);
        let params = WilsonParams {
            kappa: 0.123,
            boundary,
        };
        let mut cpu = Wilson::<f64>::new(&lat, params);
        cpu.set_gauge(&u).unwrap();
        let mut expected = psi.zeros_like();
        cpu.mult(&mut cpu.work(), &mut expected, &psi, Variant::D);

        let gpu = CudaWilson::new(&u, params).unwrap();
        let inp = DeviceSpinor::from_host(&psi).unwrap();
        let mut out = DeviceSpinor::zeros(&lat).unwrap();
        gpu.mult(&mut out, &inp).unwrap();
        let got = out.to_host().unwrap();

        let worst = got
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-13, "{dims:?} {boundary:?}: differs by {worst:e}");
    }
}

/// The domain-wall operator on the device is the one on the CPU: Shamir and Möbius, with a
/// mass, an Ls that does not fill the blocks of four of the kernel, and the three boundaries.
#[test]
fn mdwf_on_the_device_is_mdwf_on_the_cpu() {
    let antiperiodic_t = [
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Antiperiodic,
    ];
    let mixed = [
        Boundary::Antiperiodic,
        Boundary::Periodic,
        Boundary::Dirichlet,
        Boundary::Antiperiodic,
    ];
    for (dims, ls, b, c, m5, mass, boundary) in [
        (
            [4, 4, 4, 8],
            8,
            1.0,
            0.0,
            1.0,
            0.05,
            [Boundary::Periodic; 4],
        ),
        ([4, 4, 4, 8], 6, 1.5, 0.5, 1.8, 0.02, antiperiodic_t),
        ([6, 4, 2, 6], 5, 2.0, 1.0, 1.4, 0.3, mixed),
        ([4, 4, 4, 4], 2, 1.5, 0.5, 1.8, 1.0, antiperiodic_t),
    ] {
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 311);
        // A field with a fifth dimension from one with `ls` vectors: the same reals.
        let flat = random_spinor(&lat, ls, 312);
        let mut psi = Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
        psi.as_mut_slice().copy_from_slice(flat.as_slice());
        let params = MdwfParams {
            ls,
            m5,
            mass,
            b,
            c,
            boundary,
        };
        let mut cpu = Mdwf::<f64>::new(&lat, params);
        cpu.set_gauge(&u).unwrap();
        let mut expected = psi.zeros_like();
        cpu.mult(&mut cpu.work(), &mut expected, &psi, Variant::D);

        let gpu = CudaMdwf::new(&u, params).unwrap();
        let inp = DeviceSpinor::from_host(&psi).unwrap();
        let mut out = DeviceSpinor::zeros_fifth(&lat, ls).unwrap();
        gpu.mult(&mut out, &inp).unwrap();
        let got = out.to_host().unwrap();

        let worst = got
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst < 1e-13,
            "{dims:?} Ls {ls} b {b} c {c} {boundary:?}: differs by {worst:e}"
        );
    }
}

/// The even-odd blocks of the domain-wall operator on the device are those on the CPU: the
/// hop to each parity, C⁻¹ and C†⁻¹, and the Schur factor made of them.
#[test]
fn mdwf_even_odd_on_the_device_is_that_on_the_cpu() {
    let antiperiodic_t = [
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Antiperiodic,
    ];
    let mixed = [
        Boundary::Antiperiodic,
        Boundary::Periodic,
        Boundary::Dirichlet,
        Boundary::Antiperiodic,
    ];
    let worst = |a: &Field<f64, Spinor, Half>, b: &Field<f64, Spinor, Half>| {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
    };
    for (dims, ls, b, c, m5, mass, boundary) in [
        (
            [4, 4, 4, 8],
            8,
            1.0,
            0.0,
            1.0,
            0.05,
            [Boundary::Periodic; 4],
        ),
        ([4, 4, 4, 8], 6, 1.5, 0.5, 1.8, 0.02, antiperiodic_t),
        ([6, 4, 2, 6], 5, 2.0, 1.0, 1.4, 0.3, mixed),
        ([2, 4, 4, 4], 2, 1.5, 0.5, 1.8, 1.0, antiperiodic_t),
    ] {
        let what = format!("{dims:?} Ls {ls} b {b} c {c}");
        let lat = Lattice::single_node(dims);
        let u = random_gauge(&lat, 321);
        let flat = random_spinor(&lat, ls, 322);
        let mut psi = Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
        psi.as_mut_slice().copy_from_slice(flat.as_slice());
        let params = MdwfParams {
            ls,
            m5,
            mass,
            b,
            c,
            boundary,
        };
        let mut cpu = Mdwf::<f64>::new(&lat, params);
        cpu.set_gauge(&u).unwrap();
        let index = cpu.even_odd_index().unwrap().clone();
        let mut work = cpu.even_odd_work();
        let gpu = CudaMdwf::new(&u, params).unwrap();
        let half = |parity| {
            let mut f = Field::<f64, Spinor, Half>::with_shape(&lat, 24, ls, 1, 1);
            f.set_parity(parity);
            f
        };

        for to in [Parity::Even, Parity::Odd] {
            let from = to.flip();
            let mut inp = half(from);
            extract_parity(&mut inp, &psi, &index, from);
            let device_inp = DeviceSpinor::from_host_half(&inp, &index).unwrap();

            let mut expected = half(to);
            cpu.hop(&mut work, &mut expected, &inp, to, Dagger::No);
            let mut out = DeviceSpinor::zeros_half(&lat, ls, to).unwrap();
            gpu.hop(&mut out, &device_inp, to).unwrap();
            let d = worst(&out.to_host_half(&index).unwrap(), &expected);
            assert!(d < 1e-13, "{what}: the hop to {to:?} differs by {d:e}");

            for dagger in [Dagger::No, Dagger::Yes] {
                let mut expected = half(from);
                cpu.diag_inv(&mut work, &mut expected, &inp, from, dagger);
                let mut out = DeviceSpinor::zeros_half(&lat, ls, from).unwrap();
                gpu.site_inv(&mut out, &device_inp, dagger == Dagger::Yes)
                    .unwrap();
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
        for (x, y) in expected.as_mut_slice().iter_mut().zip(e.as_slice()) {
            *x -= y;
        }
        let device_inp = DeviceSpinor::from_host_half(&inp, &index).unwrap();
        let mut out = DeviceSpinor::zeros_half(&lat, ls, even).unwrap();
        let mut scratch = [
            DeviceSpinor::zeros_half(&lat, ls, even).unwrap(),
            DeviceSpinor::zeros_half(&lat, ls, odd).unwrap(),
        ];
        gpu.schur_factor(&mut out, &device_inp, &mut scratch)
            .unwrap();
        let d = worst(&out.to_host_half(&index).unwrap(), &expected);
        assert!(d < 1e-12, "{what}: the Schur factor differs by {d:e}");
    }
}
