//! The multigrid of domain-wall fermions in the solves of HMC on the device: the driver of every
//! backend (`hmc::dwf_multigrid`) with the device as its backend (`mdwf_backend::OnDevice`).

use super::mdwf_backend::OnDevice;
use super::{CudaMdwf, DeviceSpinor};
use crate::hmc::dwf_multigrid::{two_solves, two_solves_half, with_solves, DwfMultigridSetup};
use crate::solver::{SolveError, SolveInfo, Stop};

type V = DeviceSpinor<f64>;

/// x = (D†D)⁻¹ b for `d` over the Pauli-Villars operator `pv` with the multigrid of `setup`,
/// on the device; `x` holds the initial guess.
pub fn normal_solve(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    x: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lat.clone();
    with_solves(&OnDevice, (d, pv), setup, flags, &lat, |solves| {
        two_solves(d, solves, x, b, stop)
    })
}

/// z = (K†K)⁻¹ b on the even sites for K = T_ee⁻¹ Ŝ of `d`, on the device.
pub(crate) fn normal_solve_half(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    z: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lat.clone();
    with_solves(&OnDevice, (d, pv), setup, flags, &lat, |solves| {
        two_solves_half(d, &lat, solves, z, b, stop)
    })
}
