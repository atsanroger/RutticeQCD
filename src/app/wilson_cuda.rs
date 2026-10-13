//! Wilson and clover propagators on the GPU (`Main.backend : cuda_bridge`): the same solve as
//! on the CPU, one right-hand side at a time on the device.

use super::wilson::{solve_as_asked, worst_residual};
use super::*;
use crate::backend::cuda_bridge::{
    CudaClover, CudaError, CudaWilson, DeviceParities, DeviceSpinor,
};

fn device(e: CudaError) -> CliError {
    CliError::Run(format!("Main.backend cuda_bridge: {e}"))
}

/// The boundaries the kernels take.
fn check_boundary(boundary: &[Boundary; ND]) -> Result<(), CliError> {
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
    let mut s = b.zeros_like();
    let mut total = SolveInfo {
        iterations: 0,
        restarts: 0,
        residual_sqr: Vec::new(),
    };
    for column in 0..b.num_vectors() {
        let rhs = DeviceSpinor::from_host(&b.select(&[column])).map_err(device)?;
        let (x, info) = solve_as_asked(gpu, Some(&DeviceParities), &rhs, solver, None, None)
            .map_err(|e| CliError::Solve { quark, source: e })?;
        s.assign(&[column], &x.to_host().map_err(device)?);
        total.iterations = total.iterations.max(info.iterations);
        total.restarts = total.restarts.max(info.restarts);
        total.residual_sqr.extend(info.residual_sqr);
    }
    Ok((worst_residual(cpu, &s, b), s, total))
}

pub(super) fn wilson(
    lat: &Arc<Lattice>,
    params: WilsonParams,
    u: &Field<f64, Gauge>,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError> {
    check_boundary(&params.boundary)?;
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
    check_boundary(&params.boundary)?;
    let operator = |e| CliError::Operator { quark, source: e };
    let mut cpu = Clover::<f64>::new(lat, params);
    cpu.set_gauge(u).map_err(operator)?;
    let mut gpu = CudaClover::new(lat, params).map_err(device)?;
    gpu.set_gauge(u).map_err(operator)?;
    solve_quark(&cpu, &gpu, b, solver, quark)
}
