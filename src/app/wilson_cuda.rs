//! Wilson and clover propagators on the GPU (`Main.backend : cuda_bridge`): the same solve as
//! on the CPU, the right-hand sides of a quark together on the device.

use super::wilson::{solve_as_asked, worst_residual};
use super::*;
use crate::backend::cuda_bridge::{
    device_multigrid, CudaClover, CudaError, CudaWilson, DeviceParities, DeviceSpinor,
    MultigridOperator,
};
use crate::fermion::OnParity;
use crate::solver::Convert;

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

/// The propagator of `gpu` from the sources `b`, preconditioned by the multigrid on the device
/// when the file asks for it (in single precision on the operator `single` makes), and its
/// residual as `cpu`, the same operator on the CPU, sees it.
fn solve_quark<C, G, G32>(
    (cpu, gpu): (&C, &G),
    single: impl FnOnce() -> Result<G32, CliError>,
    lat: &Arc<Lattice>,
    b: &V,
    solver: &SolverConfig,
    quark: usize,
) -> Result<(f64, V, SolveInfo), CliError>
where
    C: FermionOp<V>,
    G: FermionOp<DeviceSpinor> + EvenOdd<DeviceSpinor> + MultigridOperator<f64>,
    G32: MultigridOperator<f32>,
{
    let rhs = DeviceSpinor::from_host(b).map_err(device)?;
    let setup_failed = |e| CliError::Multigrid { quark, source: e };
    let solved = match &solver.multigrid {
        None => solve_as_asked(gpu, Some(&DeviceParities), &rhs, solver, None, None),
        Some(mg) => {
            let params = mg.params(solver.even_odd);
            match mg.precision {
                Precision::Double => {
                    let mut m = device_multigrid(gpu, lat, &params).map_err(setup_failed)?;
                    propagator(gpu, &rhs, solver, &mut m)
                }
                Precision::Single => {
                    let op32 = single()?;
                    let m32 = device_multigrid(&op32, lat, &params).map_err(setup_failed)?;
                    let like = DeviceSpinor::<f32>::zeros(lat)
                        .map_err(device)?
                        .zeros_with(rhs.num_vectors());
                    let mut m = Convert::new(m32, &like);
                    // Bound to a name so that the preconditioner, which borrows op32, is
                    // dropped before op32 is.
                    #[allow(clippy::let_and_return)]
                    let solved = propagator(gpu, &rhs, solver, &mut m);
                    solved
                }
            }
        }
    };
    let (x, info) = solved.map_err(|e| CliError::Solve { quark, source: e })?;
    let s = x.to_host().map_err(device)?;
    Ok((worst_residual(cpu, &s, b), s, info))
}

/// `solve_as_asked` on the device with a preconditioner `m` of D: for the even-odd solve it is
/// used on the even sites (`OnParity`).
fn propagator<G: FermionOp<DeviceSpinor> + EvenOdd<DeviceSpinor>>(
    gpu: &G,
    rhs: &DeviceSpinor,
    solver: &SolverConfig,
    m: &mut dyn LinearFn<DeviceSpinor>,
) -> Result<(DeviceSpinor, SolveInfo), SolveError> {
    if !solver.even_odd {
        return solve_as_asked(gpu, Some(&DeviceParities), rhs, solver, Some(m), None);
    }
    let mut on_even = OnParity::new(m, &DeviceParities, rhs);
    solve_as_asked(
        gpu,
        Some(&DeviceParities),
        rhs,
        solver,
        None,
        Some(&mut on_even),
    )
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
    let single = || {
        let mut op = CudaWilson::<f32>::new(lat, params).map_err(device)?;
        op.set_gauge(u).map_err(operator)?;
        Ok(op)
    };
    solve_quark((&cpu, &gpu), single, lat, b, solver, quark)
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
    let single = || {
        let mut op = CudaClover::<f32>::new(lat, params).map_err(device)?;
        op.set_gauge(u).map_err(operator)?;
        Ok(op)
    };
    solve_quark((&cpu, &gpu), single, lat, b, solver, quark)
}
