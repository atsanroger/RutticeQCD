//! Mixed precision in the solves of HMC on the device (`hmc::mixed`): the domain-wall operator
//! in single precision on the links of the device, CG on D†D or on K†K (K = T_ee⁻¹ Ŝ, the
//! even-odd Ratio action) there, the corrections of the defect correction in double precision.

use super::{CudaError, CudaMdwf, DeviceLinks, DeviceSpinor};
use crate::fermion::{Bound, EoForm, MdwfParams, SchurFactor, Variant};
use crate::hmc::SingleNormal;
use crate::solver::{Cg, SolveError, Solver, Stop};
use crate::{ConvertFrom, Lattice, Normal, Parity, Vector};
use std::sync::Arc;

/// The domain-wall operator in single precision on the device: CG on D†D (`even_odd`: on K†K
/// on the even sites) to |r|²/|b|² < `tolerance_sqr`.
pub struct SingleCuda {
    op: CudaMdwf<f32>,
    pub tolerance_sqr: f64,
    even_odd: bool,
}

impl SingleCuda {
    /// With Neff's `alpha` (1 for the plain operator; the even-odd action takes none).
    pub fn new(
        lat: &Arc<Lattice>,
        params: MdwfParams,
        alpha: f64,
        tolerance_sqr: f64,
        even_odd: bool,
    ) -> Result<Self, CudaError> {
        let mut op = CudaMdwf::<f32>::new(lat, params)?;
        op.set_alpha(alpha);
        Ok(Self {
            op,
            tolerance_sqr,
            even_odd,
        })
    }
}

impl SingleCuda {
    /// On an operator in single precision made elsewhere (`CudaMdwf::converted_from`).
    pub fn from_operator(op: CudaMdwf<f32>, tolerance_sqr: f64, even_odd: bool) -> Self {
        Self {
            op,
            tolerance_sqr,
            even_odd,
        }
    }
}

impl SingleNormal<DeviceSpinor<f64>, DeviceLinks> for SingleCuda {
    fn set_links(&mut self, links: &DeviceLinks) {
        self.op
            .set_gauge_on_device(links)
            .expect("the links suit the operator");
    }

    fn normal(
        &mut self,
        e: &mut DeviceSpinor<f64>,
        r: &DeviceSpinor<f64>,
        max_iterations: usize,
    ) -> usize {
        let (lat, ls) = (r.lat.clone(), self.op.params().ls);
        let made = "fields on the device";
        let like = match self.even_odd {
            false => DeviceSpinor::<f32>::zeros_fifth(&lat, ls),
            true => DeviceSpinor::<f32>::zeros_half(&lat, ls, Parity::Even),
        }
        .expect(made);
        let mut r32 = like.zeros_with(r.num_vectors());
        r32.convert_from(r);
        let mut e32 = r32.zeros_like();
        let stop = Stop::simple(self.tolerance_sqr, max_iterations);
        let info = match self.even_odd {
            false => {
                let mut d = Bound::new(&self.op, Variant::D);
                Cg::new(stop).solve(&mut Normal::new(&mut d, &r32), None, &mut e32, &r32)
            }
            true => {
                let odd = DeviceSpinor::<f32>::zeros_half(&lat, ls, Parity::Odd).expect(made);
                let mut k = SchurFactor::new(&self.op, EoForm::Left, &r32, &odd);
                Cg::new(stop).solve(&mut Normal::new(&mut k, &r32), None, &mut e32, &r32)
            }
        };
        e.convert_from(&e32);
        match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(err) => panic!("the single-precision solve on the device broke down: {err:?}"),
        }
    }
}
