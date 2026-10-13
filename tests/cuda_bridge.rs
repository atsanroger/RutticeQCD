//! The CUDA kernels against the operators on the CPU (feature `cuda-bridge`; needs a GPU).
#![cfg(feature = "cuda-bridge")]

use ruttice::backend::cuda_bridge::{CudaWilson, DeviceSpinor};
use ruttice::fermion::{FermionOp, Variant, Wilson, WilsonParams};
use ruttice::gauge::random_gauge;
use ruttice::{random_spinor, Boundary, Lattice, Vector};

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
