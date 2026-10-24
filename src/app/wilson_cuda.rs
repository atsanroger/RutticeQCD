//! Wilson and clover propagators on the GPU (`Main.backend : cuda_bridge`): the same solve as
//! on the CPU, the right-hand sides of a quark together on the device.

use super::wilson::{solve_as_asked, worst_residual};
use super::*;
use crate::backend::cuda_bridge::{
    CudaClover, CudaError, CudaWilson, DeviceParities, DeviceSpinor,
};

fn device(e: CudaError) -> CliError {
    CliError::Run(format!("Main.backend cuda_bridge: {e}"))
}

/// The boundaries the kernels take; on several processes, even local extents (the faces of
/// the hops are cut on the sites of one parity).
fn check_boundary(lat: &Lattice, boundary: &[Boundary; ND]) -> Result<(), CliError> {
    if lat.comm.size() > 1 && lat.local.iter().any(|l| l % 2 != 0) {
        return invalid(format!(
            "Main.backend cuda_bridge on several processes: the local lattice {:?} must have even extents",
            lat.local
        ));
    }
    if boundary.iter().any(|b| matches!(b, Boundary::Phase(..))) {
        return invalid(
            "Main.backend cuda_bridge takes periodic, antiperiodic and Dirichlet boundaries"
                .to_string(),
        );
    }
    Ok(())
}

/// The propagator of `gpu` from the sources `b`, and its residual as `cpu`, the same operator
/// on the CPU, sees it.
fn solve_quark<C, G>(
    cpu: &C,
    gpu: &G,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError>
where
    C: FermionOp<V>,
    G: FermionOp<DeviceSpinor> + EvenOdd<DeviceSpinor>,
{
    let rhs = DeviceSpinor::from_host(b).map_err(device)?;
    let (x, info) = solve_as_asked(gpu, Some(&DeviceParities), &rhs, solver, None, None)
        .map_err(|e| CliError::Solve { quark, source: e })?;
    let s = x.to_host().map_err(device)?;
    Ok((worst_residual(cpu, &s, b), s, info))
}

pub(super) fn wilson(
    lat: &Arc<Lattice>,
    params: WilsonParams,
    u: &Field<f64, Gauge>,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError> {
    check_boundary(lat, &params.boundary)?;
    let operator = |e| CliError::Operator { quark, source: e };
    let mut cpu = Wilson::<f64>::new(lat, params);
    cpu.set_gauge(u).map_err(operator)?;
    let mut gpu = CudaWilson::new(lat, params).map_err(device)?;
    gpu.set_gauge(u).map_err(operator)?;
    solve_quark(&cpu, &gpu, b, solver, quark)
}

pub(super) fn clover(
    lat: &Arc<Lattice>,
    params: CloverParams,
    u: &Field<f64, Gauge>,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError> {
    check_boundary(lat, &params.boundary)?;
    let operator = |e| CliError::Operator { quark, source: e };
    let mut cpu = Clover::<f64>::new(lat, params);
    cpu.set_gauge(u).map_err(operator)?;
    let mut gpu = CudaClover::new(lat, params).map_err(device)?;
    gpu.set_gauge(u).map_err(operator)?;
    solve_quark(&cpu, &gpu, b, solver, quark)
}
