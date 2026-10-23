//! The multigrid of domain-wall fermions in the solves of HMC: (D†D)⁻¹ b as two solves, D† w = b
//! and then D x = w, both on the even sites with the one cycle M of Ã, repeated on the residual
//! of D†D x = b until it is below the tolerance. D x = w is GCR on Ã with M. D† w = b is GCR on
//! K† (K = C_e⁻¹ S) with C_e† Γ5 M Γ5 (`fermion::adjoint_by_gamma5`): for Möbius fermions
//! D† Γ5 D⁻¹ Γ5 = Γ5 (1 − c [Y, W] D⁻¹) Γ5, so the multigrid of D serves D† without a cycle or a
//! coarse operator of its own. Along a trajectory the test vectors y are kept and P and the
//! coarse operators are made again for the links of every solve, and y itself is improved (the
//! bootstrap, from the y of before) when the pseudofermions are drawn and after every
//! `refresh_every` force solves.
//!
//! With `rebuild_growth` the multigrid is a cache instead: P and the coarse parts of M
//! (`CoarseParts`: the coarse operators, the inverses of their blocks, the top of the smoother)
//! are kept from solve to solve and the cycle made from them for the new links, the smoother and
//! GCR working with the operators of the new links; they are made again when the vectors change
//! (a bootstrap) or when the GCR iterations per solve on the even sites have grown to
//! `rebuild_growth` times those after the last rebuild.
//!
//! The even-odd Ratio action solves K†K on the even sites, K = T_ee⁻¹ Ŝ (the left
//! `SchurFactor`): K⁻† r = T_ee† Ŝ⁻† r and K⁻¹ w = Ŝ⁻¹ T_ee w, where Ŝ⁻¹ u is the even part of
//! the solution of D x = (u, 0) and Ŝ⁻† that of D† (`normal_solve_half`), the same two solves.

use super::fermion_action::Psi;
use super::HalfPsi;
use crate::fermion::{adjoint_by_gamma5, Dagger, EoForm, Mdwf, PvEvenOdd, SchurFactor};
use crate::kernel::GammaRepre;
use crate::solver::preconditioner::multigrid::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseParts,
    LevelBootstrap, LowModeParams, PvEoBasis, PvEoCycle, PvEoParams,
};
use crate::solver::{Gcr, SolveError, SolveInfo, Solver, Stop};
use crate::{Lattice, LinearFn, LinearOp, Parity, Vector, C64, ND};
use std::sync::Arc;

/// The levels below the coarse lattice (on the device only), as the propagator files give them:
/// the first `count` super-blocks and refinements, and whether the Pauli-Villars factor has
/// them too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoarseLevels {
    pub count: usize,
    pub super_blocks: [usize; 3],
    pub bootstrap: [LevelBootstrap; 3],
    pub for_pauli_villars: bool,
}

impl CoarseLevels {
    /// From the lists of the file (at most three levels below the coarse one).
    pub fn new(super_blocks: &[usize], bootstrap: &[LevelBootstrap], for_pv: bool) -> Self {
        assert!(super_blocks.len() <= 3 && bootstrap.len() == super_blocks.len());
        let mut l = Self {
            count: super_blocks.len(),
            for_pauli_villars: for_pv,
            ..Self::default()
        };
        l.super_blocks[..l.count].copy_from_slice(super_blocks);
        l.bootstrap[..l.count].copy_from_slice(bootstrap);
        l
    }
}

/// The setup of the multigrid of an action and what it keeps along a trajectory.
pub struct DwfMultigridSetup {
    /// Aggregates of the coarse lattice.
    pub block: [usize; ND],
    /// Test vectors kept (P has twice as many columns).
    pub test_vectors: usize,
    pub low_modes: LowModeParams,
    pub bootstrap: BootstrapParams,
    /// The cycle of Ã; that of Ã† is the same with the roles exchanged, its second factor (then
    /// P†D†P) by BiCGStab on the exact adjoint.
    pub cycle: PvEoParams,
    /// The coarse solves in double precision; false (the default): in single precision.
    pub coarse_double: bool,
    /// On the device: the cycle (smoother, residuals, P) in single precision, the outer GCR in
    /// double; false (the default, and always on the CPU): in double precision.
    pub fine_single: bool,
    /// GCR of each solve on the even sites: directions kept, and when it stops.
    pub directions: usize,
    pub inner: Stop,
    /// The bootstrap is run again after every this many force solves (and when the
    /// pseudofermions are drawn); 0: only then.
    pub refresh_every: usize,
    /// Keep P and the coarse parts from solve to solve and make them again when the GCR
    /// iterations per solve on the even sites have grown to this many times those after the last
    /// rebuild (module documentation); 0: make them for every solve.
    pub rebuild_growth: f64,
    /// Levels below the coarse lattice for the coarse solves (on the device); none by default.
    pub levels: CoarseLevels,
    /// Neff's α of the solves with D (on the device): D^(α) in the left form, the setup that of
    /// D; 1 (the default): D in the right form.
    pub alpha: f64,
    vectors: Option<HalfPsi>,
    since: usize,
    cache: Option<Cache>,
    /// GCR iterations per solve on the even sites after the last rebuild, and whether the next
    /// solve rebuilds.
    baseline: Option<f64>,
    rebuild_due: bool,
    /// Rebuilds of P and the coarse parts for new links, and solves that reused them.
    pub rebuilds: usize,
    pub reuses: usize,
    /// What the multigrid keeps on the device, for an operator there.
    #[cfg(feature = "cuda-bridge")]
    pub(crate) device: Option<crate::backend::cuda_bridge::hmc_multigrid::DeviceState>,
    /// Full setups, bootstraps from the vectors of before, and solves on the even sites so far.
    pub setups: usize,
    pub refreshes: usize,
    pub inner_solves: usize,
    /// Seconds so far: of the setups and bootstraps; of making P and the cycles again for new
    /// links; of the solves on the even sites (the cycles' applications included).
    pub seconds: Seconds,
}

/// Where the time of the multigrid went (`DwfMultigridSetup::seconds`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Seconds {
    pub setup: f64,
    pub rebuild: f64,
    pub solve: f64,
}

impl DwfMultigridSetup {
    pub fn new(block: [usize; ND], test_vectors: usize, cycle: PvEoParams, inner: Stop) -> Self {
        Self {
            block,
            test_vectors,
            low_modes: LowModeParams::default(),
            bootstrap: BootstrapParams {
                cycle,
                ..BootstrapParams::default()
            },
            cycle,
            coarse_double: false,
            fine_single: false,
            directions: 8,
            inner,
            refresh_every: 0,
            rebuild_growth: 0.0,
            levels: CoarseLevels::default(),
            alpha: 1.0,
            vectors: None,
            since: 0,
            cache: None,
            baseline: None,
            rebuild_due: false,
            rebuilds: 0,
            reuses: 0,
            #[cfg(feature = "cuda-bridge")]
            device: None,
            setups: 0,
            refreshes: 0,
            inner_solves: 0,
            seconds: Seconds::default(),
        }
    }

    /// P for `op` and `pv` on their current links: from the kept vectors, after a bootstrap
    /// when one is due (`fresh`: the first solve since the pseudofermions were drawn), or from
    /// a full setup when there are none.
    fn basis<G: GammaRepre>(
        &mut self,
        op: &Mdwf<f64, G>,
        pv: &Mdwf<f64, G>,
        lat: &std::sync::Arc<crate::Lattice>,
        (fresh, md): (bool, bool),
    ) -> PvEoBasis<f64> {
        let start = std::time::Instant::now();
        let agg = Aggregation::new(lat, self.block);
        let due = self.due(fresh);
        let setup = due || self.vectors.is_none();
        let ok = "the multigrid setup succeeds";
        let basis = match (self.vectors.as_mut(), due) {
            (Some(y), false) => pv_eo_basis(op, pv, &agg, y).expect(ok),
            (Some(y), true) => {
                self.refreshes += 1;
                pv_eo_bootstrap(op, pv, &agg, y, self.bootstrap).expect(ok)
            }
            (None, _) => {
                self.setups += 1;
                let (y, _) = pv_eo_low_modes(op, pv, lat, self.low_modes);
                let mut y = y.select(&(0..self.test_vectors).collect::<Vec<_>>());
                let basis = pv_eo_bootstrap(op, pv, &agg, &mut y, self.bootstrap).expect(ok);
                self.vectors = Some(y);
                basis
            }
        };
        if due {
            self.since = 0;
        }
        if md {
            self.since += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        match setup {
            true => self.seconds.setup += elapsed,
            false => self.seconds.rebuild += elapsed,
        }
        basis
    }

    /// A bootstrap is due: the first solve since the pseudofermions were drawn, or
    /// `refresh_every` force solves since the last.
    fn due(&self, fresh: bool) -> bool {
        fresh || (self.refresh_every > 0 && self.since >= self.refresh_every)
    }

    /// The bookkeeping of a setup made elsewhere (on a device), as `basis` keeps it here.
    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn bootstrap_due(&self, fresh: bool) -> bool {
        self.due(fresh)
    }

    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn count_setup(&mut self) {
        self.setups += 1;
    }

    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn count_refresh(&mut self) {
        self.refreshes += 1;
    }

    /// After the vectors of a solve: the force solves since the last bootstrap.
    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn count_solve_for_refresh(&mut self, due: bool, md: bool) {
        if due {
            self.since = 0;
        }
        if md {
            self.since += 1;
        }
    }

    /// The seconds of a solve made elsewhere, and its solves on the even sites.
    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn record(&mut self, setup: f64, rebuild: f64, solve: f64, solves: usize) {
        self.seconds.setup += setup;
        self.seconds.rebuild += rebuild;
        self.seconds.solve += solve;
        self.inner_solves += solves;
    }

    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn count_rebuild(&mut self) {
        self.rebuilds += 1;
    }

    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn count_reuse(&mut self) {
        self.reuses += 1;
    }

    /// The iterations have grown: the next solve makes P and the coarse parts again.
    #[cfg(feature = "cuda-bridge")]
    pub(crate) fn rebuild_is_due(&self) -> bool {
        self.rebuild_due
    }

    /// After a solve with the cache: the iterations per solve on the even sites set the
    /// baseline when the parts were just made, and are compared with it when they were reused.
    pub(crate) fn after_solve(&mut self, reused: bool, per_solve: f64) {
        match reused {
            false => {
                self.baseline = Some(per_solve);
                self.rebuild_due = false;
            }
            true => {
                let base = self.baseline.unwrap_or(per_solve);
                self.rebuild_due = per_solve > self.rebuild_growth * base;
            }
        }
    }
}

/// What the cache keeps: P and the coarse parts of the cycle of Ã.
struct Cache {
    basis: PvEoBasis<f64>,
    parts: Parts,
}

enum Parts {
    Single(CoarseParts<f32>),
    Double(CoarseParts<f64>),
}

/// The cycle of Ã (M) of one solve, with the coarse solves in single or in double precision; the
/// solves with D† use it too (`adjoint_by_gamma5`).
enum Cycle<'c, G: GammaRepre> {
    Single(PvEoCycle<'c, G, f32>),
    Double(PvEoCycle<'c, G, f64>),
}

impl<G: GammaRepre> Cycle<'_, G> {
    fn m(&mut self) -> &mut dyn LinearFn<HalfPsi> {
        match self {
            Cycle::Single(m) => m,
            Cycle::Double(m) => m,
        }
    }

    fn into_parts(self) -> Parts {
        match self {
            Cycle::Single(m) => Parts::Single(m.into_parts()),
            Cycle::Double(m) => Parts::Double(m.into_parts()),
        }
    }
}

/// x = (D†D)⁻¹ b for `op` with the multigrid of `setup` (module documentation); `pv` is D_PV
/// (mass 1) on the same links. `x` holds the initial guess.
#[allow(clippy::too_many_arguments)]
pub fn normal_solve<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    x: &mut Psi,
    b: &Psi,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    with_solves(op, pv, setup, flags, b.lattice(), |solves| {
        two_solves(op, solves, x, b, stop)
    })
}

/// z = (K†K)⁻¹ b on the even sites for K = T_ee⁻¹ Ŝ of `op` (the left `SchurFactor`, the
/// operator of the even-odd Ratio action) with the multigrid of `setup` (module
/// documentation). `z` holds the initial guess.
#[allow(clippy::too_many_arguments)]
pub fn normal_solve_half<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    z: &mut HalfPsi,
    b: &HalfPsi,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lattice().clone();
    with_solves(op, pv, setup, flags, &lat, |solves| {
        two_solves_half(op, &lat, solves, z, b, stop)
    })
}

/// `run` with the solves on the even sites of `setup`'s multigrid for the current links, and
/// the bookkeeping of the setup; the iterations of the result are the GCR iterations.
fn with_solves<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    setup: &mut DwfMultigridSetup,
    (fresh, md): (bool, bool),
    lat: &Arc<Lattice>,
    run: impl FnOnce(&mut Solves<'_, '_, G>) -> Result<SolveInfo, SolveError>,
) -> Result<SolveInfo, SolveError> {
    assert!(
        op.alpha() == 1.0 && pv.alpha() == 1.0,
        "the multigrid of HMC takes D without Neff's α"
    );
    let caching = setup.rebuild_growth > 0.0;
    let reuse = caching && setup.cache.is_some() && !setup.rebuild_due && !setup.due(fresh);
    let (basis, parts) = match setup.cache.take().filter(|_| reuse) {
        Some(Cache { basis, parts }) => {
            if md {
                setup.since += 1;
            }
            setup.reuses += 1;
            (basis, Some(parts))
        }
        None => {
            setup.rebuilds += 1;
            (setup.basis(op, pv, lat, (fresh, md)), None)
        }
    };
    let start = std::time::Instant::now();
    let made = "the cycle is made";
    let outer = (op, pv, EoForm::Right);
    let mut cycle = match (parts, setup.coarse_double) {
        (Some(Parts::Single(m)), _) => {
            Cycle::Single(PvEoCycle::from_parts(op, pv, outer, &basis, setup.cycle, m))
        }
        (Some(Parts::Double(m)), _) => {
            Cycle::Double(PvEoCycle::from_parts(op, pv, outer, &basis, setup.cycle, m))
        }
        (None, false) => Cycle::Single(
            PvEoCycle::with_coarse_storage(op, pv, outer, &basis, setup.cycle).expect(made),
        ),
        (None, true) => Cycle::Double(
            PvEoCycle::with_coarse_storage(op, pv, outer, &basis, setup.cycle).expect(made),
        ),
    };
    setup.seconds.rebuild += start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let index = op.even_odd_index().expect("even extents").clone();
    let mut like = HalfPsi::with_shape(lat, 24, op.params().ls, 1, 1);
    like.set_parity(Parity::Even);
    let mut solves = Solves {
        op,
        pv,
        index: &index,
        like,
        gcr: Gcr::new(setup.inner, setup.directions),
        cycle: &mut cycle,
        inner: 0,
        count: 0,
    };
    let result = run(&mut solves);
    let (inner, count) = (solves.inner, solves.count);
    drop(solves);
    setup.inner_solves += count;
    setup.seconds.solve += start.elapsed().as_secs_f64();
    if caching {
        // Rebuild for the next solve once the iterations per solve have grown.
        setup.after_solve(reuse, inner as f64 / count.max(1) as f64);
        let parts = cycle.into_parts();
        setup.cache = Some(Cache { basis, parts });
    }
    result.map(|mut info| {
        info.iterations = inner;
        info
    })
}

/// The solves on the even sites of one solve of D†D, with their cycles.
struct Solves<'a, 'c, G: GammaRepre> {
    op: &'a Mdwf<f64, G>,
    pv: &'a Mdwf<f64, G>,
    index: &'a crate::EvenOddIndex,
    like: HalfPsi,
    gcr: Gcr<HalfPsi>,
    cycle: &'a mut Cycle<'c, G>,
    /// GCR iterations and solves so far.
    inner: usize,
    count: usize,
}

impl<G: GammaRepre> Solves<'_, '_, G> {
    fn counted(&mut self, info: Result<SolveInfo, SolveError>) -> Result<(), SolveError> {
        self.inner += match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(e) => return Err(e),
        };
        self.count += 1;
        Ok(())
    }

    /// D e = r through Ã and M.
    fn forward(&mut self, e: &mut Psi, r: &Psi) -> Result<(), SolveError> {
        let mut a = PvEvenOdd::new(self.op, self.pv, EoForm::Right, &self.like);
        let (mut rhs, mut y) = (self.like.zeros_like(), self.like.zeros_like());
        a.right_hand_side(&mut rhs, r);
        let info = self.gcr.solve(&mut a, Some(self.cycle.m()), &mut y, &rhs);
        a.solution(e, &y, r);
        self.counted(info)
    }

    /// D† e = r with Γ5 M Γ5 (`adjoint_by_gamma5`).
    fn adjoint(&mut self, e: &mut Psi, r: &Psi) -> Result<(), SolveError> {
        let op = self.op;
        let gamma5 = |out: &mut HalfPsi, inp: &HalfPsi| op.mult_gamma5_on(out, inp);
        let mut odd = self.like.zeros_like();
        odd.set_parity(Parity::Odd);
        let info = adjoint_by_gamma5(
            (self.op, self.pv),
            self.index,
            &gamma5,
            self.cycle.m(),
            &mut self.gcr,
            (&self.like, &odd),
            e,
            r,
        );
        self.counted(info)
    }
}

/// The two solves on the even sites a solve of D†D is made of, for fields `V` of the whole
/// lattice: D e = r through Ã and its cycle M, D† e = r with Γ5 M Γ5.
pub(crate) trait EvenSolves<V> {
    fn forward(&mut self, e: &mut V, r: &V) -> Result<(), SolveError>;
    fn adjoint(&mut self, e: &mut V, r: &V) -> Result<(), SolveError>;
}

impl<G: GammaRepre> EvenSolves<Psi> for Solves<'_, '_, G> {
    fn forward(&mut self, e: &mut Psi, r: &Psi) -> Result<(), SolveError> {
        Solves::forward(self, e, r)
    }
    fn adjoint(&mut self, e: &mut Psi, r: &Psi) -> Result<(), SolveError> {
        Solves::adjoint(self, e, r)
    }
}

/// D†D x = b by solving D† w = r and D e = w on the residual r until |r|²/|b|² is below the
/// tolerance (at most `stop.max_restarts` + 1 times).
pub(crate) fn two_solves<O: super::ForceOp>(
    op: &O,
    solves: &mut dyn EvenSolves<O::Field>,
    x: &mut O::Field,
    b: &O::Field,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    use crate::fermion::{Bound, Variant};
    let n = b.num_vectors();
    let mut b_norm = vec![0.0; n];
    b.norm_sqr(&mut b_norm);
    let mut d = Bound::new(op, Variant::D);
    let [mut t, mut r, mut w, mut e] = std::array::from_fn(|_| b.zeros_like());
    let mut info = SolveInfo {
        iterations: 0,
        restarts: 0,
        residual_sqr: vec![0.0; n],
    };
    for sweep in 0..=stop.max_restarts {
        info.restarts = sweep;
        d.mult(&mut t, x);
        d.mult_dag(&mut r, &t);
        r.aypx(&vec![C64::new(-1.0, 0.0); n], b);
        r.norm_sqr(&mut info.residual_sqr);
        for (r, b) in info.residual_sqr.iter_mut().zip(&b_norm) {
            *r = if *b > 0.0 { *r / b } else { 0.0 };
        }
        if info.residual_sqr.iter().all(|r| *r < stop.tolerance_sqr) {
            return Ok(info);
        }
        if sweep == stop.max_restarts {
            break;
        }
        solves.adjoint(&mut w, &r)?;
        solves.forward(&mut e, &w)?;
        x.axpy(&vec![C64::new(1.0, 0.0); n], &e);
    }
    Err(SolveError::NotConverged(info))
}

/// K†K z = b on the even sites, K = T_ee⁻¹ Ŝ, by w = T_ee† Ŝ⁻† r and e = Ŝ⁻¹ T_ee w on the
/// residual r, each Ŝ⁻¹ (Ŝ⁻†) the even part of a solve of D (D†) with no source on the odd
/// sites, until |r|²/|b|² is below the tolerance (at most `stop.max_restarts` + 1 times).
pub(crate) fn two_solves_half<O: super::EvenOddHalves>(
    op: &O,
    lat: &Arc<Lattice>,
    solves: &mut dyn EvenSolves<O::Field>,
    z: &mut O::Half,
    b: &O::Half,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let n = b.num_vectors();
    assert_eq!(n, 1, "one right-hand side");
    let odd = op.new_half(lat, Parity::Odd);
    let mut k = SchurFactor::new(op, EoForm::Left, b, &odd);
    let mut work = op.even_odd_work();
    let mut b_norm = vec![0.0; n];
    b.norm_sqr(&mut b_norm);
    // t is also the work field of the two solves (the solves on 24⁴ are short of memory).
    let [mut t, mut r, mut w] = std::array::from_fn(|_| b.zeros_like());
    let (mut source, mut whole) = (op.new_field(lat), op.new_field(lat));
    let mut info = SolveInfo {
        iterations: 0,
        restarts: 0,
        residual_sqr: vec![0.0; n],
    };
    for sweep in 0..=stop.max_restarts {
        info.restarts = sweep;
        k.mult(&mut t, z);
        k.mult_dag(&mut r, &t);
        r.aypx(&vec![C64::new(-1.0, 0.0); n], b);
        r.norm_sqr(&mut info.residual_sqr);
        for (r, b) in info.residual_sqr.iter_mut().zip(&b_norm) {
            *r = if *b > 0.0 { *r / b } else { 0.0 };
        }
        if info.residual_sqr.iter().all(|r| *r < stop.tolerance_sqr) {
            return Ok(info);
        }
        if sweep == stop.max_restarts {
            break;
        }
        // w = T_ee† Ŝ⁻† r.
        source.set_zero();
        op.put_half(&mut source, &r);
        solves.adjoint(&mut whole, &source)?;
        op.half_of(&mut t, &whole, Parity::Even);
        op.diag(&mut work, &mut w, &t, Parity::Even, Dagger::Yes);
        // e = Ŝ⁻¹ T_ee w, in t.
        op.diag(&mut work, &mut t, &w, Parity::Even, Dagger::No);
        source.set_zero();
        op.put_half(&mut source, &t);
        solves.forward(&mut whole, &source)?;
        op.half_of(&mut t, &whole, Parity::Even);
        z.axpy(&vec![C64::new(1.0, 0.0); n], &t);
    }
    Err(SolveError::NotConverged(info))
}
