//! The coarse correction of the multigrid of domain-wall fermions on the device: what
//! `PvEoCycle::coarse_correction` does on the CPU, step by step (see the top of
//! `multigrid::mdwf_even_odd`), with the second factor solved by MINRES and no levels below
//! the coarse lattice.

use super::*;
use crate::fermion::{Dagger, EoForm, EvenOdd};
use crate::solver::preconditioner::multigrid::{coarse_solve, CoarseKrylov, PvEoParams};
use crate::solver::Stop;
use crate::{ConvertFrom, LinearFn, Parity, Vector};
use std::cell::Cell;
use std::error::Error;

/// A closure as a linear map.
struct Map<F>(F);

impl<V: Vector, F: FnMut(&mut V, &V)> LinearFn<V> for Map<F> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        (self.0)(out, inp);
    }
}

/// e = O_L π P (P†DP)⁻¹ P† ι M π P (P†D_PV†P)⁻¹ P† ι O_R r for the operator
/// `PvRatio::new(outer.0, outer.1, form, ..)`, the coarse solves in the precision `CT`.
pub struct DeviceCorrection<'a, CT: DeviceReal = f32> {
    /// D and the second factor (D_PV, or D for the normal equations), plain ...
    plain: (&'a CudaMdwf, &'a CudaMdwf),
    /// ... and as they are in the operator that is solved (with Neff's α).
    outer: (&'a CudaMdwf, &'a CudaMdwf),
    form: EoForm,
    basis: DeviceBasis,
    coarse_d: DeviceCoarseOp<CT>,
    coarse_pv: DeviceCoarseOp<CT>,
    /// |H_xx|⁻¹ of the second factor.
    blocks_pv: DeviceBuffer<CT>,
    params: PvEoParams,
    h: DeviceSpinor,
    g: DeviceCoarse<f64>,
    gc: DeviceCoarse<CT>,
    w: DeviceCoarse<CT>,
    c: DeviceCoarse<CT>,
    coarse_applications: Cell<usize>,
}

impl<'a, CT: DeviceReal> DeviceCorrection<'a, CT> {
    /// Whether the correction with these settings is on the device.
    pub fn supports(params: &PvEoParams) -> bool {
        params.pv_solver == CoarseKrylov::Minres && params.d_solver != CoarseKrylov::Minres
    }

    /// The coarse operators are those of `plain`; `like`: a field on the even sites.
    pub fn new(
        device: DeviceBasis,
        plain: (&'a CudaMdwf, &'a CudaMdwf),
        outer: (&'a CudaMdwf, &'a CudaMdwf),
        form: EoForm,
        params: PvEoParams,
        like: &DeviceSpinor,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(Self::supports(&params));
        assert_eq!(like.parity(), Some(Parity::Even));
        let even_odd = params.coarse_even_odd;
        let coarse_d = DeviceCoarseOp::<CT>::new(&plain.0.coarse_operator(&device)?, even_odd)?;
        let coarse_pv = DeviceCoarseOp::<CT>::new(&plain.1.coarse_operator(&device)?, even_odd)?;
        Ok(Self {
            plain,
            outer,
            form,
            blocks_pv: coarse_pv.abs_inverse_of_a_gamma5()?,
            coarse_d,
            coarse_pv,
            params,
            h: like.zeros_like(),
            g: device.coarse_field(1),
            gc: device.coarse_field(1),
            w: device.coarse_field(1),
            c: device.coarse_field(1),
            basis: device,
            coarse_applications: Cell::new(0),
        })
    }

    /// Coarse-operator applications so far (both factors).
    pub fn coarse_applications(&self) -> usize {
        self.coarse_applications.get()
    }

    /// `e` = the coarse correction of `r`, fields on the even sites.
    pub fn correct(&mut self, e: &mut DeviceSpinor, r: &DeviceSpinor) {
        let nv = r.num_vectors();
        crate::fit(&mut self.h, r);
        if self.g.num_vectors() != nv {
            self.g = self.g.zeros_with(nv);
            for f in [&mut self.gc, &mut self.w, &mut self.c] {
                *f = f.zeros_with(nv);
            }
        }
        let ((op, pv), (op_a, pv_a)) = (self.plain, self.outer);
        let mut work = op.even_odd_work();
        let (even, n) = (Parity::Even, &self.coarse_applications);
        let (d, second) = (&self.coarse_d, &self.coarse_pv);
        let stop = Stop::simple(
            self.params.coarse_tolerance_sqr,
            self.params.coarse_max_iterations,
        );
        let pv_stop = Stop::simple(self.params.pv_tolerance_sqr, self.params.pv_max_iterations);

        // h = O_R r, g = P† ι h.
        match self.form {
            EoForm::Right => pv.diag(&mut work, &mut self.h, r, even, Dagger::Yes),
            EoForm::Left => pv_a.mult_q(&mut self.h, r, true),
        }
        self.basis.restrict(&mut self.g, &self.h);
        self.gc.convert_from(&self.g);

        // w = (P†D_PV†P)⁻¹ g, by MINRES on H = D_c Γ5 with H w = Γ5 g, preconditioned by the
        // blocks |H_xx|⁻¹.
        self.w.set_zero();
        second.gamma5(&mut self.gc);
        let blocks_pv = &self.blocks_pv;
        let mut blocks = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            second.apply_blocks(out, inp, blocks_pv);
        });
        if second.has_even_odd() {
            let (mut source, mut odd) = (self.gc.zeros_like(), self.gc.zeros_like());
            second.schur_source(&mut source, &self.gc, &mut odd);
            let (mut turned, mut scratch) = (self.gc.zeros_like(), self.gc.zeros_like());
            // The Schur complement of H on the even sites: S Γ5.
            let mut a = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                n.set(n.get() + inp.num_vectors());
                turned.copy_from(inp);
                second.gamma5(&mut turned);
                second.schur(out, &turned, &mut scratch);
            });
            let kind = self.params.pv_solver;
            coarse_solve(
                kind,
                pv_stop,
                &mut a,
                Some(&mut blocks),
                &mut self.w,
                &source,
            );
            second.gamma5(&mut self.w);
            second.schur_solution(&mut self.w, &self.gc, &mut odd);
            second.gamma5(&mut self.w);
            n.set(n.get() + nv);
        } else {
            let mut a = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                n.set(n.get() + inp.num_vectors());
                second.apply(out, inp, true);
                second.gamma5(out);
            });
            let kind = self.params.pv_solver;
            coarse_solve(
                kind,
                pv_stop,
                &mut a,
                Some(&mut blocks),
                &mut self.w,
                &self.gc,
            );
        }

        // g = P† ι M π P w; M = 1 (right form) or C_e^(α) C_PV,e^(α)† (left form).
        self.g.convert_from(&self.w);
        self.basis.prolong(&mut self.h, &self.g);
        if self.form == EoForm::Left {
            pv_a.diag(&mut work, e, &self.h, even, Dagger::Yes);
            op_a.diag(&mut work, &mut self.h, e, even, Dagger::No);
        }
        self.basis.restrict(&mut self.g, &self.h);
        self.gc.convert_from(&self.g);

        // c = (P†DP)⁻¹ g: on the even coarse sites where the lattice allows, preconditioned
        // by the inverses of the blocks.
        self.c.set_zero();
        let kind = self.params.d_solver;
        if d.has_even_odd() {
            let (mut source, mut odd) = (self.gc.zeros_like(), self.gc.zeros_like());
            d.schur_source(&mut source, &self.gc, &mut odd);
            let mut scratch = self.gc.zeros_like();
            let mut a = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                n.set(n.get() + inp.num_vectors());
                d.schur(out, inp, &mut scratch);
            });
            let mut m = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                d.even_block_inverse(out, inp);
            });
            coarse_solve(kind, stop, &mut a, Some(&mut m), &mut self.c, &source);
            d.schur_solution(&mut self.c, &self.gc, &mut odd);
            n.set(n.get() + nv);
        } else {
            let mut a = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                n.set(n.get() + inp.num_vectors());
                d.apply(out, inp, false);
            });
            let mut m = Map(|out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                d.apply_blocks(out, inp, d.block_inverses());
            });
            coarse_solve(kind, stop, &mut a, Some(&mut m), &mut self.c, &self.gc);
        }

        // e = O_L π P c: C_e (right form) or Q_e⁻¹ (left form).
        self.g.convert_from(&self.c);
        self.basis.prolong(&mut self.h, &self.g);
        match self.form {
            EoForm::Right => op.diag(&mut work, e, &self.h, even, Dagger::No),
            EoForm::Left => op_a.mult_q(e, &self.h, true),
        }
    }
}
