//! Mixed precision on the device: the domain-wall operator in single precision on the links of
//! the device, CG on D†D or on K†K (K = T_ee⁻¹ Ŝ, the even-odd Ratio action) there. In the solves
//! of HMC (`hmc::mixed`) the corrections of a defect correction in double precision; in the
//! CGNE of the propagators one CG with reliable updates (`SingleCuda::reliable_normal`).

use super::{CudaError, CudaMdwf, DeviceLinks, DeviceSpinor};
use crate::fermion::{AfterQ, Bound, EoForm, MdwfParams, SchurFactor, Variant};
use crate::hmc::SingleNormal;
use crate::solver::{reliable_cg, Cg, SolveError, SolveInfo, Solver, Stop};
use crate::{ConvertFrom, Lattice, LinearFn, Normal, Parity, Vector};
use std::sync::Arc;

/// The domain-wall operator in single precision on the device: CG on D†D (`even_odd`: on K†K
/// on the even sites) to |r|²/|b|² < `tolerance_sqr`.
pub struct SingleCuda {
    op: CudaMdwf<f32>,
    pub tolerance_sqr: f64,
    even_odd: bool,
    /// The operator with Neff's α whose Q the even-odd solves take (`with_neff`).
    neff: Option<CudaMdwf<f32>>,
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
            neff: None,
        })
    }

    /// The even-odd solves on K Q_e (Neff's `alpha`), as `TwoFlavorRatioEo::with_neff` asks.
    pub fn with_neff(mut self, alpha: f64) -> Result<Self, CudaError> {
        assert!(self.even_odd, "Neff's Q of the even-odd solves");
        let mut q = CudaMdwf::<f32>::on_the_links_of(&self.op, *self.op.params())?;
        q.set_alpha(alpha);
        self.neff = Some(q);
        Ok(self)
    }
}

impl SingleCuda {
    /// On an operator in single precision made elsewhere (`CudaMdwf::converted_from`).
    pub fn from_operator(op: CudaMdwf<f32>, tolerance_sqr: f64, even_odd: bool) -> Self {
        Self {
            op,
            tolerance_sqr,
            even_odd,
            neff: None,
        }
    }
}

impl SingleCuda {
    /// `a x = b`, `a` the normal operator in double precision of which this one is the single
    /// precision (D†D, or K†K on the even sites): CG in single precision with reliable updates
    /// (`solver::reliable_cg`), each when |r|² has fallen by `drop_sqr`. `x` holds the initial
    /// guess.
    pub fn reliable_normal(
        &mut self,
        a: &mut dyn LinearFn<DeviceSpinor<f64>>,
        x: &mut DeviceSpinor<f64>,
        b: &DeviceSpinor<f64>,
        stop: Stop,
        drop_sqr: f64,
    ) -> Result<SolveInfo, SolveError> {
        let (lat, ls) = (b.lat.clone(), self.op.params().ls);
        let made = "fields on the device";
        let like = match self.even_odd {
            false => DeviceSpinor::<f32>::zeros_fifth(&lat, ls),
            true => DeviceSpinor::<f32>::zeros_half(&lat, ls, Parity::Even),
        }
        .expect(made)
        .zeros_with(b.num_vectors());
        match self.even_odd {
            false => {
                let mut d = Bound::new(&self.op, Variant::D);
                let mut low = Normal::new(&mut d, &like);
                reliable_cg(a, &mut low, &like, x, b, stop, drop_sqr)
            }
            true => {
                let odd = DeviceSpinor::<f32>::zeros_half(&lat, ls, Parity::Odd).expect(made);
                let mut k = SchurFactor::new(&self.op, EoForm::Left, &like, &odd);
                let mut low = Normal::new(&mut k, &like);
                reliable_cg(a, &mut low, &like, x, b, stop, drop_sqr)
            }
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
        tolerance_sqr: f64,
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
        let stop = Stop::simple(self.tolerance_sqr.max(tolerance_sqr), max_iterations);
        let info = match self.even_odd {
            false => {
                let mut d = Bound::new(&self.op, Variant::D);
                Cg::new(stop).solve(&mut Normal::new(&mut d, &r32), None, &mut e32, &r32)
            }
            true => {
                let odd = DeviceSpinor::<f32>::zeros_half(&lat, ls, Parity::Odd).expect(made);
                let mut k = SchurFactor::new(&self.op, EoForm::Left, &r32, &odd);
                match &self.neff {
                    Some(n) => {
                        let q = |out: &mut DeviceSpinor<f32>, inp: &DeviceSpinor<f32>, inverse| {
                            n.mult_q(out, inp, inverse)
                        };
                        let mut a = AfterQ::new(&mut k, &q, &r32);
                        Cg::new(stop).solve(&mut Normal::new(&mut a, &r32), None, &mut e32, &r32)
                    }
                    None => {
                        Cg::new(stop).solve(&mut Normal::new(&mut k, &r32), None, &mut e32, &r32)
                    }
                }
            }
        };
        e.convert_from(&e32);
        match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(err) => panic!("the single-precision solve on the device broke down: {err:?}"),
        }
    }
}
