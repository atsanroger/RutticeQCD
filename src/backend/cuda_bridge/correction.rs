//! The coarse correction of the multigrid of domain-wall fermions on the device: the
//! `Correction` of the multigrid with the fields, P and the coarse operators of the device,
//! which are made here.

use super::*;
use crate::fermion::EoForm;
use crate::solver::preconditioner::multigrid::{
    CoarseKrylov, Correction, FnMap, LevelBootstrap, PvEoParams,
};
use crate::{LinearFn, Parity, Vector};
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
    /// The two coarse operators and the coarse lattice, for levels below it.
    coarse_d: Rc<DeviceCoarseOp<CT>>,
    coarse_pv: Rc<DeviceCoarseOp<CT>>,
    coarse: Arc<Lattice>,
}

impl<'a, F: DeviceReal, CT: LevelReal> DeviceCorrection<'a, F, CT> {
    /// Whether the correction with these settings is on the device: the quark factor not by
    /// MINRES, which is only for the Hermitian form of the second; that by MINRES on its
    /// Hermitian form or by another solver on its exact adjoint.
    pub fn supports(params: &PvEoParams) -> bool {
        params.d_solver != CoarseKrylov::Minres
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
            DeviceCoarseOp::<CT>::new(&op.coarse_operator(&device)?, even_odd)
        };
        let minres = params.pv_solver == CoarseKrylov::Minres;
        let coarse_d = Rc::new(build(plain.0)?);
        // The second factor: P†D_PV P for MINRES, else its exact adjoint P†D_PV†P.
        let pv_op = build(plain.1)?;
        let coarse_pv = Rc::new(match minres {
            true => pv_op,
            false => pv_op.adjoint()?,
        });
        for (name, op) in [("P†DP", &coarse_d), ("P†D_PV P", &coarse_pv)] {
            trace_checksum(name, || op.checksums().map(|c| c.0));
            trace_checksum(&format!("the block inverses of {name}"), || {
                op.checksums().map(|c| c.1)
            });
        }
        // The matrices on the coarse sites: the inverses of the blocks of P†DP, and of the
        // second factor |H_xx|⁻¹ for MINRES, else the inverses of the blocks of the adjoint.
        let (d, pv) = (coarse_d.clone(), coarse_pv.clone());
        let blocks_d = FnMap(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            d.apply_blocks(out, inp, d.block_inverses());
        });
        let blocks_pv: Box<dyn LinearFn<DeviceCoarse<CT>> + 'a> = if minres {
            let abs = coarse_pv.abs_inverse_of_a_gamma5()?;
            trace_checksum("|H_xx|⁻¹", || abs.checksum());
            Box::new(FnMap(
                move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                    pv.apply_blocks(out, inp, &abs);
                },
            ))
        } else {
            Box::new(FnMap(
                move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                    pv.apply_blocks(out, inp, pv.block_inverses());
                },
            ))
        };
        let coarse_like = device.coarse_field::<CT>(1);
        let coarse = device.coarse.clone();
        Ok(Self {
            correction: Correction::new(
                plain,
                outer,
                form,
                device,
                (coarse_d.clone(), coarse_pv.clone()),
                (Box::new(blocks_d), blocks_pv),
                params,
                &like.zeros_with(1),
                &coarse_like,
            ),
            coarse_d,
            coarse_pv,
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
        let v = vectors.to_precision::<CT>()?;
        let levels =
            DeviceMultilevel::build(&self.coarse_d, &self.coarse, &v, super_blocks, bootstrap)?;
        let applications = levels.setup_applications;
        self.correction.set_levels_d(Box::new(
            move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| levels.apply(out, inp),
        ));
        Ok(applications)
    }

    /// The same levels for the solves of the Pauli-Villars factor, by MINRES on its Hermitian
    /// form, as the CPU has them (`PvEoCycle::add_pv_levels`, `AbsMultilevel`), from the same
    /// test vectors. Returns the applications of the coarse operator the setup took.
    pub fn add_pv_levels<U: DeviceReal>(
        &mut self,
        vectors: &DeviceCoarse<U>,
        super_blocks: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        if self.correction.params().pv_solver != CoarseKrylov::Minres {
            return Err("MultiGrid.levels_for_pauli_villars needs MINRES for the Pauli-Villars factor (Coarse_Solver_PV)".into());
        }
        let v = vectors.to_precision::<CT>()?;
        let levels = DeviceAbsMultilevel::build(&self.coarse_pv, &self.coarse, &v, super_blocks)?;
        let applications = levels.setup_applications;
        self.correction.set_levels_pv(Box::new(
            move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| levels.add(out, inp),
        ));
        Ok(applications)
    }
}
