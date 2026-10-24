//! Mixed precision on the device: the domain-wall operator in single precision on the links of
//! the device, as `fermion::SingleOperator`. The solves themselves are those of every backend
//! (`fermion::SingleAll`, `fermion::SingleEven`): the corrections of the defect correction of
//! HMC, and the CGNE of the propagators with reliable updates.

use super::{CudaError, CudaMdwf, DeviceLinks, DeviceSpinor};
use crate::fermion::{MdwfParams, SingleOperator};
use crate::{Lattice, Parity, Vector};
use std::sync::Arc;

impl SingleOperator for CudaMdwf<f32> {
    type Whole = DeviceSpinor<f32>;
    type Half = DeviceSpinor<f32>;
    type Links = DeviceLinks;

    fn set_links(&mut self, links: &DeviceLinks) {
        self.set_gauge_on_device(links)
            .expect("the links suit the operator");
    }

    fn whole(&self, lat: &Arc<Lattice>, n: usize) -> DeviceSpinor<f32> {
        DeviceSpinor::<f32>::zeros_fifth(lat, self.params().ls)
            .expect("fields on the device")
            .zeros_with(n)
    }

    fn half(&self, lat: &Arc<Lattice>, parity: Parity, n: usize) -> DeviceSpinor<f32> {
        DeviceSpinor::<f32>::zeros_half(lat, self.params().ls, parity)
            .expect("fields on the device")
            .zeros_with(n)
    }

    fn mult_q(&self, out: &mut DeviceSpinor<f32>, inp: &DeviceSpinor<f32>, inverse: bool) {
        CudaMdwf::mult_q(self, out, inp, inverse);
    }
}

/// The domain-wall operator `params` in single precision on the device, with Neff's `alpha`.
pub(crate) fn single_mdwf(
    lat: &Arc<Lattice>,
    params: MdwfParams,
    alpha: f64,
) -> Result<CudaMdwf<f32>, CudaError> {
    let mut op = CudaMdwf::<f32>::new(lat, params)?;
    op.set_alpha(alpha);
    Ok(op)
}
