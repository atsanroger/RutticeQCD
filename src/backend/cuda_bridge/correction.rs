//! The coarse correction of the multigrid of domain-wall fermions on the device: the
//! `Correction` of the multigrid with the fields, P and the coarse operators of the device,
//! which are made here.

use super::*;
use crate::fermion::EoForm;
use crate::solver::preconditioner::multigrid::{CoarseKrylov, Correction, FnMap, PvEoParams};
use crate::{Parity, Vector};
use std::error::Error;
use std::rc::Rc;

/// `Correction` with what is on the device.
type OnDevice<'a, F, CT> = Correction<
    'a,
    DeviceSpinor<F>,
    DeviceCoarse<F>,
    DeviceCoarse<CT>,
    CudaMdwf<F>,
    DeviceBasis,
    Rc<DeviceCoarseOp<CT>>,
>;

/// The correction for the operator `PvRatio::new(outer.0, outer.1, form, ..)` on fields in the
/// precision `F`, the coarse solves in the precision `CT`.
pub struct DeviceCorrection<'a, F: DeviceReal = f64, CT: DeviceReal = f32>(OnDevice<'a, F, CT>);

impl<'a, F: DeviceReal, CT: DeviceReal> DeviceCorrection<'a, F, CT> {
    /// Whether the correction with these settings is on the device: the second factor by
    /// MINRES (the exact adjoint of a coarse operator is not there).
    pub fn supports(params: &PvEoParams) -> bool {
        params.pv_solver == CoarseKrylov::Minres && params.d_solver != CoarseKrylov::Minres
    }

    /// With the prolongation `device`. `plain`: D and the second factor (D_PV, or D for the
    /// normal equations), which the coarse operators are built from; `outer`: the same as they
    /// are in the operator that is solved (with Neff's α); `like`: a field on the even sites.
    pub fn new(
        device: DeviceBasis,
        plain: (&'a CudaMdwf<F>, &'a CudaMdwf<F>),
        outer: (&'a CudaMdwf<F>, &'a CudaMdwf<F>),
        form: EoForm,
        params: PvEoParams,
        like: &DeviceSpinor<F>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(Self::supports(&params));
        assert_eq!(like.parity(), Some(Parity::Even));
        let even_odd = params.coarse_even_odd;
        let build = |op: &CudaMdwf<F>| -> Result<_, Box<dyn Error>> {
            Ok(Rc::new(DeviceCoarseOp::<CT>::new(
                &op.coarse_operator(&device)?,
                even_odd,
            )?))
        };
        let (coarse_d, coarse_pv) = (build(plain.0)?, build(plain.1)?);
        // The matrices on the coarse sites: the inverses of the blocks of P†DP, and |H_xx|⁻¹
        // of the second factor.
        let (d, pv, abs) = (
            coarse_d.clone(),
            coarse_pv.clone(),
            coarse_pv.abs_inverse_of_a_gamma5()?,
        );
        let blocks_d = FnMap(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            d.apply_blocks(out, inp, d.block_inverses());
        });
        let blocks_pv = FnMap(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            pv.apply_blocks(out, inp, &abs);
        });
        let coarse_like = device.coarse_field::<CT>(1);
        Ok(Self(Correction::new(
            plain,
            outer,
            form,
            device,
            (coarse_d, coarse_pv),
            (Box::new(blocks_d), Box::new(blocks_pv)),
            params,
            &like.zeros_with(1),
            &coarse_like,
        )))
    }

    /// Coarse-operator applications so far (both factors).
    pub fn coarse_applications(&self) -> usize {
        self.0.coarse_applications()
    }

    /// `e` = the coarse correction of `r`, fields on the even sites.
    pub fn correct(&mut self, e: &mut DeviceSpinor<F>, r: &DeviceSpinor<F>) {
        self.0.correct(e, r);
    }
}
