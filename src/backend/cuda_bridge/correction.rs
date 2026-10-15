//! The coarse correction of the multigrid of domain-wall fermions on the device: the
//! `Correction` of the multigrid with the fields, P and the coarse operators of the device,
//! which are made here.

use super::*;
use crate::fermion::EoForm;
use crate::solver::preconditioner::multigrid::{
    CoarseKrylov, Correction, FnMap, LevelBootstrap, PvEoParams,
};
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
pub struct DeviceCorrection<'a, F: DeviceReal = f64, CT: DeviceReal = f32> {
    correction: OnDevice<'a, F, CT>,
    /// P†DP and the coarse lattice, for levels below it.
    coarse_d: Rc<DeviceCoarseOp<CT>>,
    coarse: Arc<Lattice>,
}

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
        for (name, op) in [("P†DP", &coarse_d), ("P†D_PV P", &coarse_pv)] {
            trace_checksum(name, || op.checksums().map(|c| c.0));
            trace_checksum(&format!("the block inverses of {name}"), || {
                op.checksums().map(|c| c.1)
            });
        }
        // The matrices on the coarse sites: the inverses of the blocks of P†DP, and |H_xx|⁻¹
        // of the second factor.
        let (d, pv, abs) = (
            coarse_d.clone(),
            coarse_pv.clone(),
            coarse_pv.abs_inverse_of_a_gamma5()?,
        );
        trace_checksum("|H_xx|⁻¹", || abs.checksum());
        let blocks_d = FnMap(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            d.apply_blocks(out, inp, d.block_inverses());
        });
        let blocks_pv = FnMap(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            pv.apply_blocks(out, inp, &abs);
        });
        let coarse_like = device.coarse_field::<CT>(1);
        let coarse = device.coarse.clone();
        Ok(Self {
            correction: Correction::new(
                plain,
                outer,
                form,
                device,
                (coarse_d.clone(), coarse_pv),
                (Box::new(blocks_d), Box::new(blocks_pv)),
                params,
                &like.zeros_with(1),
                &coarse_like,
            ),
            coarse_d,
            coarse,
        })
    }

    /// Coarse-operator applications so far (both factors).
    pub fn coarse_applications(&self) -> usize {
        self.correction.coarse_applications()
    }

    /// `e` = the coarse correction of `r`, fields on the even sites.
    pub fn correct(&mut self, e: &mut DeviceSpinor<F>, r: &DeviceSpinor<F>) {
        self.correction.correct(e, r);
    }
}

impl<F: DeviceReal, CT: LevelReal> DeviceCorrection<'_, F, CT> {
    /// Add levels below the coarse lattice to the solves of P†DP, on the device as on the CPU
    /// (`PvEoCycle::add_levels_with`): from the test vectors `vectors` on the coarse lattice
    /// (`DeviceBasis::pv_eo_coarse_vectors`), with super-blocks of `super_blocks[ℓ]` sites per
    /// direction on level ℓ + 2 refined by `bootstrap[ℓ]`. Returns the applications of P†DP the
    /// setup took.
    pub fn add_levels<U: DeviceReal>(
        &mut self,
        vectors: &DeviceCoarse<U>,
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<usize, Box<dyn Error>> {
        let mut v = must(DeviceCoarse::<CT>::zeros(
            vectors.sites,
            vectors.dim,
            vectors.vectors,
        ));
        crate::ConvertFrom::convert_from(&mut v, vectors);
        let levels =
            DeviceMultilevel::build(&self.coarse_d, &self.coarse, &v, super_blocks, bootstrap)?;
        let applications = levels.setup_applications;
        self.correction.set_levels_d(Box::new(
            move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| levels.apply(out, inp),
        ));
        Ok(applications)
    }
}
