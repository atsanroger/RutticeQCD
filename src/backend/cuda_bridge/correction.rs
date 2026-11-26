//! The coarse correction of the multigrid of domain-wall fermions on the device: the
//! `Correction` of the multigrid with the fields, P and the coarse operators of the device,
//! which are made here.

use super::levels::AdjointPass;
use super::*;
use crate::fermion::EoForm;
use crate::solver::preconditioner::multigrid::{
    Adjoint, ApplyAdjoint, CoarseKrylov, Correction, FnMap, LevelBootstrap, Levels, PvEoParams,
};
use crate::{LinearFn, Parity, Vector};
use std::error::Error;
use std::rc::Rc;

/// exp/ls16-diag: a pair (v, w) of fields on the whole lattice, on the host.
pub(crate) type RepairPair = (
    crate::Field<f64, crate::Spinor>,
    crate::Field<f64, crate::Spinor>,
);

thread_local! {
    // exp/ls16-diag (RUTTICE_REPAIR): v = P c and w = D v of the last `diag_detail`, and the
    // pairs (v, w) put in place of the last columns of P when it is made (`DeviceBasis::pv_eo`).
    pub(crate) static REPAIR_STASH: std::cell::RefCell<Option<RepairPair>> =
        const { std::cell::RefCell::new(None) };
    pub(crate) static REPAIR_COLUMNS: std::cell::RefCell<Vec<RepairPair>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `Correction` with what is on the device.
type OnDevice<'a, F, CT> = Correction<
    'a,
    DeviceSpinor<F>,
    DeviceCoarse<F>,
    DeviceCoarse<CT>,
    CudaMdwf<F>,
    Rc<DeviceBasis>,
    Rc<DeviceCoarseOp<CT>>,
>;

/// The correction for the operator `PvRatio::new(outer.0, outer.1, form, ..)` on fields in the
/// precision `F`, the coarse solves in the precision `CT`.
pub struct DeviceCorrection<'a, F: DeviceReal = f64, CT: LevelReal = f32> {
    correction: OnDevice<'a, F, CT>,
    /// The two coarse operators and the coarse lattice, for levels below it.
    coarse_d: Rc<DeviceCoarseOp<CT>>,
    coarse_pv: Rc<DeviceCoarseOp<CT>>,
    coarse: Arc<Lattice>,
    parts: DeviceParts<CT>,
}

/// What a correction is made of on the device besides its operators: P, the coarse operators
/// and the matrices on their sites. A clone (it is all shared) makes the correction for other
/// links again without building any of it (`DeviceCorrection::from_parts`): the coarse
/// correction is then that of the links it was built for, still a fixed map.
pub struct DeviceParts<CT: LevelReal> {
    basis: Rc<DeviceBasis>,
    coarse_d: Rc<DeviceCoarseOp<CT>>,
    coarse_pv: Rc<DeviceCoarseOp<CT>>,
    /// The adjoints of the factors solved by `FactorSolve::Regularized`: (P†DP)† for the first,
    /// P†D_PV P for the second.
    coarse_pv_adj: Option<DeviceAdjoint<CT>>,
    coarse_d_adj: Option<DeviceAdjoint<CT>>,
    /// |H_xx|⁻¹ of the second factor, for MINRES.
    abs: Option<Rc<DeviceBuffer<CT>>>,
    /// The levels below the coarse lattice of either factor (`add_levels`, `add_pv_levels`).
    levels_d: Option<Rc<Levels<'static, DeviceCoarse<CT>>>>,
    levels_pv: Option<Rc<Levels<'static, DeviceCoarse<CT>>>>,
}

impl<CT: LevelReal> Clone for DeviceParts<CT> {
    fn clone(&self) -> Self {
        Self {
            basis: self.basis.clone(),
            coarse_d: self.coarse_d.clone(),
            coarse_pv: self.coarse_pv.clone(),
            coarse_pv_adj: self.coarse_pv_adj.clone(),
            coarse_d_adj: self.coarse_d_adj.clone(),
            abs: self.abs.clone(),
            levels_d: self.levels_d.clone(),
            levels_pv: self.levels_pv.clone(),
        }
    }
}

impl<'a, F: DeviceReal, CT: LevelReal> DeviceCorrection<'a, F, CT> {
    /// Whether the correction with these settings is on the device: the quark factor not by
    /// MINRES, which is only for the Hermitian form of the second; that by MINRES on its
    /// Hermitian form or by another solver on its exact adjoint.
    pub(crate) fn supports(params: &PvEoParams) -> bool {
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
        Self::shared(Rc::new(device), plain, outer, form, params, like)
    }

    /// `new` with a prolongation that other corrections share (that of Ã and of Ã†).
    pub fn shared(
        device: Rc<DeviceBasis>,
        plain: (&'a CudaMdwf<F>, &'a CudaMdwf<F>),
        outer: (&'a CudaMdwf<F>, &'a CudaMdwf<F>),
        form: EoForm,
        params: PvEoParams,
        like: &DeviceSpinor<F>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(Self::supports(&params));
        let parts = Self::build(device, plain, params)?;
        Self::from_parts(parts, plain, outer, form, params, like)
    }

    /// P†DP and the second factor on `device`, and the matrices on their sites.
    fn build(
        device: Rc<DeviceBasis>,
        plain: (&CudaMdwf<F>, &CudaMdwf<F>),
        params: PvEoParams,
    ) -> Result<DeviceParts<CT>, Box<dyn Error>> {
        let even_odd = params.coarse_even_odd;
        // Made on the device and kept there.
        let build = |op: &CudaMdwf<F>| -> Result<_, Box<dyn Error>> {
            op.coarse_operator_on_device::<CT>(&device, even_odd)
        };
        let minres = params.pv_solver == CoarseKrylov::Minres;
        let coarse_d = Rc::new(build(plain.0)?);
        let coarse_d_adj = match params.d_solver.regularized() {
            Some(_) => Some(DeviceAdjoint::of(&coarse_d, None)?),
            None => None,
        };
        // The second factor: P†D_PV P for MINRES, else its exact adjoint P†D_PV†P.
        let pv_op = build(plain.1)?;
        let dsym_pv = plain.1.dsym_pv();
        assert!(
            !dsym_pv || (!minres && params.pv_solver.regularized().is_some()),
            "RUTTICE_DSYM_PV goes with the regularized solve of the second factor"
        );
        let (coarse_pv, coarse_pv_adj) = match (minres, params.pv_solver.regularized()) {
            // exp/ls16-diag (RUTTICE_DSYM_PV): B = P† D_PV R P itself and its adjoint; the
            // correction solves σ3 B σ3.
            (false, Some(_)) if dsym_pv => {
                let b = Rc::new(pv_op);
                let adjoint = DeviceAdjoint::of(&b, None)?;
                (b, Some(adjoint))
            }
            (true, _) => (Rc::new(pv_op), None),
            (false, None) => (Rc::new(pv_op.adjoint()?), None),
            (false, Some(_)) => {
                let adjoint = Rc::new(pv_op.adjoint()?);
                let of_it = DeviceAdjoint::of(&adjoint, Some(pv_op))?;
                (adjoint, Some(of_it))
            }
        };
        for (name, op) in [("P†DP", &coarse_d), ("P†D_PV P", &coarse_pv)] {
            trace_checksum(name, || op.checksums().map(|c| c.0));
            trace_checksum(&format!("the block inverses of {name}"), || {
                op.checksums().map(|c| c.1)
            });
        }
        let abs = match minres {
            true => {
                let abs = coarse_pv.abs_inverse_of_a_gamma5()?;
                trace_checksum("|H_xx|⁻¹", || abs.checksum());
                Some(Rc::new(abs))
            }
            false => None,
        };
        Ok(DeviceParts {
            basis: device,
            coarse_d,
            coarse_pv,
            coarse_pv_adj,
            coarse_d_adj,
            abs,
            levels_d: None,
            levels_pv: None,
        })
    }

    /// The correction of `parts` (`parts()` of a correction with the same settings) for the
    /// operators given: nothing is built.
    pub fn from_parts(
        parts: DeviceParts<CT>,
        plain: (&'a CudaMdwf<F>, &'a CudaMdwf<F>),
        outer: (&'a CudaMdwf<F>, &'a CudaMdwf<F>),
        form: EoForm,
        params: PvEoParams,
        like: &DeviceSpinor<F>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(Self::supports(&params));
        assert_eq!(like.parity(), Some(Parity::Even));
        assert_eq!(
            parts.abs.is_some(),
            params.pv_solver == CoarseKrylov::Minres,
            "parts made for another solver of the second factor"
        );
        let (coarse_d, coarse_pv) = (parts.coarse_d.clone(), parts.coarse_pv.clone());
        // The matrices on the coarse sites: the inverses of the blocks of P†DP, and of the
        // second factor |H_xx|⁻¹ for MINRES, else the inverses of the blocks of the adjoint.
        let (d, pv) = (coarse_d.clone(), coarse_pv.clone());
        let blocks_d = FnMap(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
            d.apply_blocks(out, inp, d.block_inverses());
        });
        let blocks_pv: Box<dyn LinearFn<DeviceCoarse<CT>> + 'a> = match parts.abs.clone() {
            Some(abs) => Box::new(FnMap(
                move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                    pv.apply_blocks(out, inp, &abs);
                },
            )),
            None => Box::new(FnMap(
                move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                    pv.apply_blocks(out, inp, pv.block_inverses());
                },
            )),
        };
        let device = parts.basis.clone();
        let coarse_like = device.coarse_field::<CT>(1);
        // exp/ls16-diag (RUTTICE_SPLIT): the coarse fields of the second factor on P_v.
        let pv_like = device.pv.as_ref().map(|b| b.coarse_field::<CT>(1));
        let coarse = device.coarse.clone();
        let (levels_d, levels_pv) = (parts.levels_d.clone(), parts.levels_pv.clone());
        let mut c = Self {
            correction: Correction::new(
                plain,
                outer,
                form,
                device,
                (coarse_d.clone(), coarse_pv.clone()),
                (Box::new(blocks_d), blocks_pv),
                (
                    parts.coarse_d_adj.clone().map(DeviceAdjoint::boxed),
                    parts.coarse_pv_adj.clone().map(DeviceAdjoint::boxed),
                ),
                params,
                &like.zeros_with(1),
                &coarse_like,
            ),
            coarse_d,
            coarse_pv,
            coarse,
            parts,
        };
        if let Some(like) = &pv_like {
            c.correction.set_pv_like(like);
        }
        if let Some(l) = levels_d {
            c.correction
                .set_levels_d(Box::new(move |out, inp| l(out, inp)));
        }
        if let Some(l) = levels_pv {
            c.correction
                .set_levels_pv(Box::new(move |out, inp| l(out, inp)));
        }
        Ok(c)
    }

    /// What the correction is made of, to make it again for other links.
    pub fn parts(&self) -> DeviceParts<CT> {
        self.parts.clone()
    }

    /// Coarse-operator applications so far (both factors).
    pub fn coarse_applications(&self) -> usize {
        self.correction.coarse_applications()
    }

    /// `e` = the coarse correction of `r`, fields on the even sites.
    pub fn correct(&mut self, e: &mut DeviceSpinor<F>, r: &DeviceSpinor<F>) {
        // exp/ls16-diag (RUTTICE_DUMP_QUARK=dir): once, at the first correction of the solve.
        if crate::env::DUMP_ARMED.with(|a| a.replace(false)) {
            if let Some(dir) = std::env::var_os("RUTTICE_DUMP_QUARK") {
                if let Err(e) = self.dump_quark(std::path::Path::new(&dir)) {
                    panic!("RUTTICE_DUMP_QUARK: {e}");
                }
            }
        }
        self.correction.correct(e, r);
        if crate::env::DIAG_DETAIL.with(|f| f.get()) {
            for (name, f) in [("r (input)", r), ("e (output)", &*e)] {
                let (by_s, peak, pr) = must(f.diag_profile());
                let s: Vec<String> = by_s.iter().map(|v| format!("{v:.3}")).collect();
                eprintln!(
                    "diag profile {name}: |.|^2 by s [{}] site max/mean {peak:.3e} participation {pr:.3e}",
                    s.join(" ")
                );
            }
            self.diag_detail(e);
        }
    }

    /// exp/ls16-diag (RUTTICE_DUMP_QUARK=dir): the quark coarse operator D_c as a dense matrix,
    /// dir/dc.bin (n as a little-endian u64, then D_c e_j for j = 0..n, n complex numbers each
    /// as f32 re, im) and dir/sigma3.bin (the coarse Γ5 on each e_j, n f32); then it waits for
    /// dir/c.bin (k as u64, then k coarse vectors of n complex numbers as f64 re, im), which the
    /// analysis writes, and prints for each c: |c|, |P c|, |D_c c|, |P† D_f P c| and
    /// |D_f P c|, D_f = D B5⁻¹ under RUTTICE_DSYM (the operator D_c is the Galerkin one of),
    /// else D, and the share of |P c|² and |D_f P c|² on each s with the participation ratio
    /// of the sites. A singular vector c of D_c with |D_f P c| ≫ |D_c c| is not a low mode of D_f.
    fn dump_quark(&self, dir: &std::path::Path) -> Result<(), Box<dyn Error>> {
        use crate::solver::preconditioner::multigrid::{CoarseOperator, NeffQ};
        use crate::C64;
        use std::io::Write;
        let (_, _, basis, d, coarse_d) = self.correction.diag_parts();
        let (sites, dim) = (coarse_d.sites(), coarse_d.dim());
        let n = sites * dim;
        std::fs::create_dir_all(dir)?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("dc.bin"))?);
        out.write_all(&(n as u64).to_le_bytes())?;
        let mut sigma = Vec::with_capacity(4 * n);
        let batch = 256;
        for start in (0..n).step_by(batch) {
            let count = batch.min(n - start);
            let mut units = vec![C64::new(0.0, 0.0); count * n];
            for j in 0..count {
                units[j * n + start + j] = C64::new(1.0, 0.0);
            }
            let mut x = DeviceCoarse::<CT>::from_complex(sites, dim, &units)?;
            let mut y = x.zeros_like();
            coarse_d.mult(&mut y, &x);
            for z in y.to_complex()? {
                out.write_all(&(z.re as f32).to_le_bytes())?;
                out.write_all(&(z.im as f32).to_le_bytes())?;
            }
            coarse_d.gamma5(&mut x);
            let g = x.to_complex()?;
            for j in 0..count {
                sigma.extend_from_slice(&(g[j * n + start + j].re as f32).to_le_bytes());
            }
        }
        out.flush()?;
        drop(out);
        std::fs::write(dir.join("sigma3.bin"), &sigma)?;
        eprintln!("dump: D_c of dimension {n} in {}; waiting for c.bin", dir.display());
        let path = dir.join("c.bin");
        let start = std::time::Instant::now();
        while !path.exists() {
            if start.elapsed().as_secs() > 5400 {
                return Err("RUTTICE_DUMP_QUARK: no c.bin within 90 minutes".into());
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
        let bytes = std::fs::read(&path)?;
        let k = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        assert_eq!(bytes.len(), 8 + 16 * k * n, "c.bin: k vectors of n complex f64");
        let f = |i: usize| f64::from_le_bytes(bytes[8 + 8 * i..16 + 8 * i].try_into().unwrap());
        let cz: Vec<C64> = (0..k * n).map(|i| C64::new(f(2 * i), f(2 * i + 1))).collect();
        let norms = |v: &dyn Fn(&mut [f64])| {
            let mut s = vec![0.0; k];
            v(&mut s);
            s.iter().map(|x| x.sqrt()).collect::<Vec<f64>>()
        };
        let xc = DeviceCoarse::<CT>::from_complex(sites, dim, &cz)?;
        let mut yc = xc.zeros_like();
        coarse_d.mult(&mut yc, &xc);
        let xf = DeviceCoarse::<F>::from_complex(sites, dim, &cz)?;
        let mut v = basis.fine_field::<F>(k)?;
        basis.prolong(&mut v, &xf);
        let mut w = v.zeros_like();
        if !d.dsym_right(&mut w, &v) {
            w.copy_from(&v);
        }
        let mut dv = v.zeros_like();
        d.mult_plain(&mut dv, &w);
        let mut pdv = xf.zeros_like();
        basis.restrict(&mut pdv, &dv);
        let (nc, npc, ndc, npd, nd) = (
            norms(&|s| xf.norm_sqr(s)),
            norms(&|s| v.norm_sqr(s)),
            norms(&|s| yc.norm_sqr(s)),
            norms(&|s| pdv.norm_sqr(s)),
            norms(&|s| dv.norm_sqr(s)),
        );
        for i in 0..k {
            eprintln!(
                "dump c{i}: |c| {:.4e} |Pc| {:.4e} |D_c c| {:.4e} |P†D_f Pc| {:.4e} |D_f Pc| {:.4e} ratio {:.3e}",
                nc[i], npc[i], ndc[i], npd[i], nd[i], nd[i] / ndc[i]
            );
        }
        // The share of |P c|² and of |D_f P c|² on each s, by vector.
        for i in 0..k {
            let shape = |f: &DeviceSpinor<F>| -> Result<String, Box<dyn Error>> {
                let (by_s, _, pr) = f.select(&[i]).diag_profile()?;
                let s: Vec<String> = by_s.iter().map(|v| format!("{v:.3}")).collect();
                Ok(format!("[{}] PR {pr:.3}", s.join(" ")))
            };
            eprintln!("dump c{i} profile Pc: {}", shape(&v)?);
            eprintln!("dump c{i} profile D_f Pc: {}", shape(&dv)?);
        }
        Ok(())
    }

    /// exp/ls16-diag: for the quark solution c of the last correction, v = P c on the whole
    /// lattice, |D v|/|v|, |P†D v|/|v| and their ratio (about 1 for a low mode of D, large where
    /// D v is nearly orthogonal to range P), and |P†D v − (P†DP) c|/|P†D v| as a check.
    fn diag_detail(&self, like: &DeviceSpinor<F>) {
        use crate::field::vector::ConvertFrom;
        use crate::solver::preconditioner::multigrid::CoarseOperator;
        let (g, c, basis, d, coarse_d) = self.correction.diag_parts();
        let nv = g.num_vectors();
        let norm = |x: &dyn Fn(&mut [f64])| {
            let mut s = vec![0.0; nv];
            x(&mut s);
            s[0].sqrt()
        };
        let mut v = must(DeviceSpinor::<F>::zeros_fifth(&like.lat, like.ls)).zeros_with(nv);
        basis.prolong(&mut v, g);
        let mut dv = v.zeros_like();
        d.mult_plain(&mut dv, &v);
        let mut pdv = g.zeros_like();
        basis.restrict(&mut pdv, &dv);
        let mut dc = c.zeros_like();
        coarse_d.mult(&mut dc, c);
        let mut dc_f = g.zeros_like();
        dc_f.convert_from(&dc);
        // RUTTICE_REPAIR: the last v = P c and w = D v, on the host (the device is full).
        REPAIR_STASH.with(|s| {
            *s.borrow_mut() = Some((must(v.to_host()), must(dv.to_host())));
        });
        let n_pdv = norm(&|s| pdv.norm_sqr(s));
        dc_f.axpy(&vec![crate::C64::new(-1.0, 0.0); nv], &pdv);
        let (n_v, n_dv) = (norm(&|s| v.norm_sqr(s)), norm(&|s| dv.norm_sqr(s)));
        eprintln!(
            "diag detail: |D Pc|/|Pc| {:.4e} |P†D Pc|/|Pc| {:.4e} ratio {:.4e} check {:.2e}",
            n_dv / n_v,
            n_pdv / n_v,
            n_dv / n_pdv,
            norm(&|s| dc_f.norm_sqr(s)) / n_pdv
        );
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
        let levels: Rc<Levels<'static, DeviceCoarse<CT>>> =
            Rc::new(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| {
                levels.apply(out, inp)
            });
        let l = levels.clone();
        self.correction
            .set_levels_d(Box::new(move |out, inp| l(out, inp)));
        self.parts.levels_d = Some(levels);
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
        let levels: Rc<Levels<'static, DeviceCoarse<CT>>> =
            Rc::new(move |out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>| levels.add(out, inp));
        let l = levels.clone();
        self.correction
            .set_levels_pv(Box::new(move |out, inp| l(out, inp)));
        self.parts.levels_pv = Some(levels);
        Ok(applications)
    }
}

/// The adjoint A† of a coarse factor A of the device for its regularized solve: applied from the
/// blocks of A by `coarse_adjoint_pass_kernel` (one process, unless `RUTTICE_COARSE_ADJOINT=copy`),
/// or a copy of it, a second coarse operator.
pub(crate) enum DeviceAdjoint<CT: LevelReal> {
    Pass(Rc<AdjointPass<CT>>),
    Copy(Rc<DeviceCoarseOp<CT>>),
}

impl<CT: LevelReal> Clone for DeviceAdjoint<CT> {
    fn clone(&self) -> Self {
        match self {
            DeviceAdjoint::Pass(p) => DeviceAdjoint::Pass(p.clone()),
            DeviceAdjoint::Copy(c) => DeviceAdjoint::Copy(c.clone()),
        }
    }
}

impl<CT: LevelReal> DeviceAdjoint<CT> {
    /// The adjoint of `a`; `copy` is A† where it has been made anyway (else `a.adjoint()` is
    /// made when a copy is needed).
    fn of(
        a: &Rc<DeviceCoarseOp<CT>>,
        copy: Option<DeviceCoarseOp<CT>>,
    ) -> Result<Self, Box<dyn Error>> {
        // Read once: it changes the rounding of the coarse solves and their memory, so the run
        // says so on stderr.
        static WANTS_COPY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let wants_copy = *WANTS_COPY.get_or_init(|| {
            let copy =
                crate::env::choice("RUTTICE_COARSE_ADJOINT", &["kernel", "copy"]) == Some(1);
            if copy {
                eprintln!(
                    "RUTTICE_COARSE_ADJOINT=copy: the adjoint coarse operators are stored copies (another rounding, more memory)"
                );
            }
            copy
        });
        if !wants_copy {
            if let Some(pass) = a.adjoint_pass()? {
                return Ok(DeviceAdjoint::Pass(Rc::new(pass)));
            }
        }
        Ok(DeviceAdjoint::Copy(Rc::new(match copy {
            Some(c) => c,
            None => a.adjoint()?,
        })))
    }

    fn boxed<'a>(self) -> Adjoint<'a, DeviceCoarse<CT>>
    where
        CT: 'a,
    {
        match self {
            DeviceAdjoint::Pass(p) => Box::new(SharedPass(p)),
            DeviceAdjoint::Copy(c) => Box::new(c),
        }
    }
}

/// An `AdjointPass` shared by the corrections of the same parts.
struct SharedPass<CT: LevelReal>(Rc<AdjointPass<CT>>);

impl<CT: LevelReal> ApplyAdjoint<DeviceCoarse<CT>> for SharedPass<CT> {
    fn apply_adjoint(&self, out: &mut DeviceCoarse<CT>, inp: &DeviceCoarse<CT>) {
        self.0.apply(out, inp);
    }
}

impl<CT: LevelReal> DeviceParts<CT> {
    /// The prolongation the parts were made on.
    pub(crate) fn basis(&self) -> &DeviceBasis {
        &self.basis
    }
}
