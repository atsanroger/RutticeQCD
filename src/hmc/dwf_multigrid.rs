//! The multigrid of domain-wall fermions in the solves of HMC, the same on every backend:
//! (D†D)⁻¹ b as two solves, D† w = b and then D x = w, both on the even sites with the one
//! cycle M of Ã, repeated on the residual of D†D x = b until it is below the tolerance. D x = w is
//! GCR on Ã with M. D† w = b is GCR on K† (K = C_e⁻¹ S) with C_e† Γ5 M Γ5
//! (`fermion::adjoint_by_gamma5`): for Möbius fermions D† Γ5 D⁻¹ Γ5 = Γ5 (1 − c [Y, W] D⁻¹) Γ5, so
//! the multigrid of D serves D† without a cycle or a coarse operator of its own. Along a
//! trajectory the test vectors y are kept and P and the coarse operators are made again for the
//! links of every solve, and y itself is improved (the bootstrap, from the y of before) when the
//! pseudofermions are drawn and after every `refresh_every` force solves.
//!
//! With Neff's α (`DwfMultigridSetup::alpha`) the solves with D are those of D^(α) = D Q in the
//! left form, Ã_L^(α), as in the propagator runs, and x = Q x^(α): the setup, P and the coarse
//! operators are those of D, and only the smoother has a range of its own. The solves with D†
//! stay as they are: D^(α)† = Q D†, so α would only scale their right-hand side.
//!
//! With `rebuild_growth` the multigrid is a cache instead: P and the coarse parts of M (the
//! coarse operators, the inverses of their blocks, the levels below the coarse lattice, the tops
//! of the smoothers) are kept from solve to solve and the cycles made from them for the new
//! links, the smoother and GCR working with the operators of the new links; they are made again
//! when the vectors change (a bootstrap) or when the GCR iterations per solve on the even sites
//! have grown to `rebuild_growth` times those after the last rebuild. Where the sweeps stall,
//! the solve is done again with P made for the links of now, then after a bootstrap.
//!
//! The even-odd Ratio action solves K†K on the even sites, K = T_ee⁻¹ Ŝ (the left
//! `SchurFactor`): K⁻† r = T_ee† Ŝ⁻† r and K⁻¹ w = Ŝ⁻¹ T_ee w, where Ŝ⁻¹ u is the even part of
//! the solution of D x = (u, 0) and Ŝ⁻† that of D† (`normal_solve_half`), the same two solves.
//!
//! All of that is written once, here (`with_solves`, `Solves`); a backend supplies its fields,
//! the test vectors, P and the coarse parts, and the cycles made of them (`Backend`: `OnCpu`
//! below, `backend::cuda_bridge::hmc_multigrid` on the device).

use super::fermion_action::Psi;
use super::HalfPsi;
use crate::fermion::{
    adjoint_by_gamma5, Dagger, EoForm, EvenOdd, Mdwf, PvEvenOdd, PvRatio, SchurFactor, SplitParity,
};
use crate::kernel::GammaRepre;
use crate::solver::preconditioner::multigrid::{
    pv_eo_basis_stored, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, random_even,
    Aggregation, BasisStorage, BootstrapParams, CoarseParts, LevelBootstrap, LowModeParams,
    PvEoBasis, PvEoCycle, PvEoParams,
};
use crate::solver::{largest_magnitude, Convert, Gcr, SolveError, SolveInfo, Solver, Stop};
use crate::{
    EvenOddIndex, Field, Half, Lattice, LinearFn, LinearOp, Parity, Spinor, Storage, Vector, C64,
    ND,
};
use std::any::Any;
use std::error::Error;
use std::sync::Arc;

/// The levels below the coarse lattice, as the propagator files give them:
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
    /// The cycle of Ã; the solves with D† use it between two Γ5.
    pub cycle: PvEoParams,
    /// The coarse solves in double precision; false (the default): in single precision.
    pub coarse_double: bool,
    /// The cycle (the setup, the smoother, the residuals, P) in single precision, the outer GCR
    /// in double; false (the default): in double precision.
    pub fine_single: bool,
    /// P kept in single precision; false (the default): in half precision.
    pub basis_single: bool,
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
    /// Levels below the coarse lattice for the coarse solves; none by default.
    pub levels: CoarseLevels,
    /// Neff's α of the solves with D: D^(α) in the left form, the setup that of D; 1 (the
    /// default): D in the right form.
    pub alpha: f64,
    /// What the backend keeps between the solves (`Kept`).
    state: Option<Box<dyn Any>>,
    since: usize,
    /// GCR iterations per solve on the even sites after the last rebuild, and whether the next
    /// solve rebuilds.
    baseline: Option<f64>,
    rebuild_due: bool,
    /// Rebuilds of P and the coarse parts for new links, and solves that reused them.
    pub rebuilds: usize,
    pub reuses: usize,
    /// Full setups, bootstraps from the vectors of before, and solves on the even sites so far.
    pub setups: usize,
    pub refreshes: usize,
    pub inner_solves: usize,
    /// Seconds so far: of the setups and bootstraps; of making P and the coarse parts again for
    /// new links; of the solves on the even sites (the cycles' applications included).
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
            basis_single: false,
            directions: 8,
            inner,
            refresh_every: 0,
            rebuild_growth: 0.0,
            levels: CoarseLevels::default(),
            alpha: 1.0,
            state: None,
            since: 0,
            baseline: None,
            rebuild_due: false,
            rebuilds: 0,
            reuses: 0,
            setups: 0,
            refreshes: 0,
            inner_solves: 0,
            seconds: Seconds::default(),
        }
    }

    /// A bootstrap is due: the first solve since the pseudofermions were drawn, or
    /// `refresh_every` force solves since the last.
    fn due(&self, fresh: bool) -> bool {
        fresh || (self.refresh_every > 0 && self.since >= self.refresh_every)
    }

    /// After a solve with the cache: the iterations per solve on the even sites set the
    /// baseline when the parts were just made, and are compared with it when they were reused.
    fn after_solve(&mut self, reused: bool, per_solve: f64) {
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

    /// The super-blocks and refinements of the levels below the coarse lattice.
    pub(crate) fn level_lists(&self) -> (&[usize], &[LevelBootstrap]) {
        let l = &self.levels;
        (&l.super_blocks[..l.count], &l.bootstrap[..l.count])
    }
}

/// What a backend supplies to the multigrid of HMC (module documentation); the rest is
/// `with_solves`.
pub(crate) trait Backend {
    /// D (and D_PV) in double precision, as the actions hold them.
    type Op: EvenOdd<Self::Half>;
    /// Fields on all sites and on the sites of one parity.
    type Field: Vector;
    type Half: Vector;
    type Split: SplitParity<Self::Field, Self::Half>;
    /// The test vectors as the backend keeps them between the solves.
    type Vectors: 'static;
    /// What the cycles are made of: P, the coarse parts with their levels, the tops of the
    /// smoothers of Ã and of Ã_L^(α). Kept from solve to solve by the cache.
    type Parts: 'static;

    fn split(&self) -> &Self::Split;
    /// Zero fields of one vector on the even and on the odd sites, for `op`.
    fn halves(&self, op: &Self::Op, lat: &Arc<Lattice>) -> (Self::Half, Self::Half);
    fn gamma5(op: &Self::Op, out: &mut Self::Half, inp: &Self::Half);
    /// `out = Q inp` on all sites, Q at the α of `op`.
    fn mult_q(op: &Self::Op, out: &mut Self::Field, inp: &Self::Field);
    /// D^(α) on the links of `op`.
    fn with_alpha(op: &Self::Op, alpha: f64) -> Result<Self::Op, Box<dyn Error>>;
    /// The test vectors for `ops` from those `kept` (a full setup without any, improved by a
    /// bootstrap when `due`, else as they are), and from them P, the coarse parts with the
    /// levels the setup asks for, and the tops of the smoothers of Ã and (with `d_alpha`) of
    /// Ã_L^(α); with the seconds the vectors took.
    #[allow(clippy::type_complexity)]
    fn rebuild(
        &self,
        ops: (&Self::Op, &Self::Op),
        d_alpha: Option<&Self::Op>,
        lat: &Arc<Lattice>,
        setup: &DwfMultigridSetup,
        kept: Option<Self::Vectors>,
        due: bool,
    ) -> Result<(Self::Vectors, Self::Parts, f64), Box<dyn Error>>;
    /// The cycles of one solve, made of `parts` for the operators on the links of now.
    fn cycles<'a>(
        &'a self,
        ops: (&'a Self::Op, &'a Self::Op),
        d_alpha: Option<&'a Self::Op>,
        lat: &Arc<Lattice>,
        setup: &DwfMultigridSetup,
        parts: &'a Self::Parts,
    ) -> Result<Box<dyn Cycles<Self::Half> + 'a>, Box<dyn Error>>;
}

/// The cycles of one solve: M of Ã, or with `neff` of Ã_L^(α).
pub(crate) trait Cycles<H> {
    fn cycle(&self, neff: bool) -> Box<dyn LinearFn<H> + '_>;
}

/// The sweeps of a solve, over the solves on the even sites (`two_solves`, `two_solves_half`).
type Run<'r, V> = dyn FnMut(&mut dyn EvenSolves<V>) -> Result<SolveInfo, SolveError> + 'r;

/// What the setup keeps for a backend between the solves.
struct Kept<B: Backend> {
    vectors: Option<B::Vectors>,
    parts: Option<B::Parts>,
}

/// `run` with the solves on the even sites of `setup`'s multigrid for the current links of
/// `ops`, and the bookkeeping of the setup; the iterations of the result are the GCR iterations.
pub(crate) fn with_solves<B: Backend + 'static>(
    backend: &B,
    ops: (&B::Op, &B::Op),
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    lat: &Arc<Lattice>,
    mut run: impl FnMut(&mut dyn EvenSolves<B::Field>) -> Result<SolveInfo, SolveError>,
) -> Result<SolveInfo, SolveError> {
    let mut attempt = |setup: &mut DwfMultigridSetup, flags| {
        solves_once(backend, ops, setup, flags, lat, &mut run)
            .unwrap_or_else(|e| panic!("the multigrid of HMC failed: {e}"))
    };
    let mut result = attempt(setup, flags);
    // When the sweeps stall: P and the coarse parts again for the links of now (those kept may be
    // of links far from them), then the test vectors improved by a bootstrap as well. Each goes
    // on from the solution so far (`run` starts from it).
    for (what, fresh) in [("P made again", flags.0), ("a bootstrap", true)] {
        let Err(SolveError::NotConverged(info)) = &result else {
            break;
        };
        eprintln!(
            "the multigrid of HMC stalled at |r|²/|b|² {:.1e}: again with {what}",
            info.residual_sqr.iter().fold(0.0_f64, |m, r| m.max(*r))
        );
        setup.rebuild_due = true;
        result = attempt(setup, (fresh, flags.1));
    }
    result
}

/// One attempt of `with_solves`.
fn solves_once<B: Backend + 'static>(
    backend: &B,
    (d, pv): (&B::Op, &B::Op),
    setup: &mut DwfMultigridSetup,
    (fresh, md): (bool, bool),
    lat: &Arc<Lattice>,
    run: &mut Run<'_, B::Field>,
) -> Result<Result<SolveInfo, SolveError>, Box<dyn Error>> {
    let start = std::time::Instant::now();
    let caching = setup.rebuild_growth > 0.0;
    let due = setup.due(fresh);
    let mut kept = setup
        .state
        .take()
        .and_then(|k| k.downcast::<Kept<B>>().ok())
        .map_or(
            Kept::<B> {
                vectors: None,
                parts: None,
            },
            |k| *k,
        );
    let reuse = caching && kept.parts.is_some() && !setup.rebuild_due && !due;
    let d_alpha = match setup.alpha != 1.0 {
        true => Some(B::with_alpha(d, setup.alpha)?),
        false => None,
    };
    let parts = match kept.parts.take().filter(|_| reuse) {
        Some(parts) => {
            setup.reuses += 1;
            parts
        }
        None => {
            match (&kept.vectors, due) {
                (None, _) => setup.setups += 1,
                (Some(_), true) => setup.refreshes += 1,
                (Some(_), false) => {}
            }
            let kept_vectors = kept.vectors.take();
            let (vectors, parts, vector_seconds) =
                backend.rebuild((d, pv), d_alpha.as_ref(), lat, setup, kept_vectors, due)?;
            // The vectors count as the setup, P and the coarse parts as the rebuild.
            setup.seconds.setup += vector_seconds;
            setup.seconds.rebuild += start.elapsed().as_secs_f64() - vector_seconds;
            if due {
                setup.since = 0;
            }
            setup.rebuilds += 1;
            kept.vectors = Some(vectors);
            parts
        }
    };
    if md {
        setup.since += 1;
    }
    let start = std::time::Instant::now();
    let (inner, count, result) = {
        let cycles = backend.cycles((d, pv), d_alpha.as_ref(), lat, setup, &parts)?;
        let (even, odd) = backend.halves(d, lat);
        let mut solves = Solves::<B> {
            backend,
            d,
            pv,
            d_alpha: d_alpha.as_ref(),
            even,
            odd,
            gcr: Gcr::new(setup.inner, setup.directions),
            cycles,
            inner: 0,
            count: 0,
        };
        let result = run(&mut solves);
        (solves.inner, solves.count, result)
    };
    setup.inner_solves += count;
    setup.seconds.solve += start.elapsed().as_secs_f64();
    if caching {
        // Rebuild for the next solve once the iterations per solve have grown.
        setup.after_solve(reuse, inner as f64 / count.max(1) as f64);
        kept.parts = Some(parts);
    }
    setup.state = Some(Box::new(kept));
    Ok(result.map(|mut info| {
        info.iterations = inner;
        info
    }))
}

/// The solves on the even sites of one solve of D†D, with the cycles of the backend.
struct Solves<'a, B: Backend> {
    backend: &'a B,
    d: &'a B::Op,
    pv: &'a B::Op,
    d_alpha: Option<&'a B::Op>,
    even: B::Half,
    odd: B::Half,
    gcr: Gcr<B::Half>,
    cycles: Box<dyn Cycles<B::Half> + 'a>,
    /// GCR iterations and solves so far.
    inner: usize,
    count: usize,
}

impl<B: Backend> Solves<'_, B> {
    fn counted(&mut self, info: Result<SolveInfo, SolveError>) -> Result<(), SolveError> {
        self.inner += match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(e) => return Err(e),
        };
        self.count += 1;
        Ok(())
    }
}

impl<B: Backend> EvenSolves<B::Field> for Solves<'_, B> {
    /// D e = r through Ã and M, or with Neff's α through Ã_L^(α) and its M: e = Q e^(α).
    fn forward(&mut self, e: &mut B::Field, r: &B::Field) -> Result<(), SolveError> {
        let (d, form) = match self.d_alpha {
            Some(d) => (d, EoForm::Left),
            None => (self.d, EoForm::Right),
        };
        let split = self.backend.split();
        let mut a = PvRatio::new(d, self.pv, form, &self.even, &self.odd);
        let mut rhs = self.even.zeros_with(r.num_vectors());
        a.right_hand_side(split, &mut rhs, r);
        let mut y = rhs.zeros_like();
        let mut m = self.cycles.cycle(self.d_alpha.is_some());
        let info = self.gcr.solve(&mut a, Some(m.as_mut()), &mut y, &rhs);
        drop(m);
        match self.d_alpha {
            Some(d) => {
                let mut x = e.zeros_like();
                a.solution(split, &mut x, &y, r);
                B::mult_q(d, e, &x);
            }
            None => a.solution(split, e, &y, r),
        }
        self.counted(info)
    }

    /// D† e = r with Γ5 M Γ5 (`adjoint_by_gamma5`).
    fn adjoint(&mut self, e: &mut B::Field, r: &B::Field) -> Result<(), SolveError> {
        let mut m = self.cycles.cycle(false);
        let d = self.d;
        let gamma5 = |out: &mut B::Half, inp: &B::Half| B::gamma5(d, out, inp);
        let info = adjoint_by_gamma5(
            (self.d, self.pv),
            self.backend.split(),
            &gamma5,
            m.as_mut(),
            &mut self.gcr,
            (&self.even, &self.odd),
            e,
            r,
        );
        drop(m);
        self.counted(info)
    }
}

/// The two solves on the even sites a solve of D†D is made of, for fields `V` of the whole
/// lattice: D e = r through Ã and its cycle M, D† e = r with Γ5 M Γ5.
pub(crate) trait EvenSolves<V> {
    fn forward(&mut self, e: &mut V, r: &V) -> Result<(), SolveError>;
    fn adjoint(&mut self, e: &mut V, r: &V) -> Result<(), SolveError>;
}

/// x = (D†D)⁻¹ b for `op` with the multigrid of `setup` (module documentation), on the CPU;
/// `pv` is D_PV (mass 1) on the same links. `x` holds the initial guess.
pub fn normal_solve<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    x: &mut Psi,
    b: &Psi,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lattice().clone();
    let backend = OnCpu::<G>::new(op);
    with_solves(&backend, (op, pv), setup, flags, &lat, |solves| {
        two_solves(op, solves, x, b, stop)
    })
}

/// z = (K†K)⁻¹ b on the even sites for K = T_ee⁻¹ Ŝ of `op` (the left `SchurFactor`, the
/// operator of the even-odd Ratio action) with the multigrid of `setup` (module
/// documentation), on the CPU. `z` holds the initial guess.
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
    let backend = OnCpu::<G>::new(op);
    with_solves(&backend, (op, pv), setup, flags, &lat, |solves| {
        two_solves_half(op, &lat, solves, z, b, stop)
    })
}

/// The CPU as a `Backend`: the cycles of `PvEoCycle`, in double precision or (`fine_single`) on
/// the operators converted to single precision.
pub(crate) struct OnCpu<G: GammaRepre> {
    index: Arc<EvenOddIndex>,
    _gamma: std::marker::PhantomData<G>,
}

impl<G: GammaRepre> OnCpu<G> {
    pub(crate) fn new(op: &Mdwf<f64, G>) -> Self {
        Self {
            index: op.even_odd_index().expect("even extents").clone(),
            _gamma: std::marker::PhantomData,
        }
    }
}

type HalfOf<F> = Field<F, Spinor, Half>;

/// The test vectors of the CPU, in the precision of the cycle.
pub(crate) enum CpuVectors {
    Double(HalfOf<f64>),
    Single(HalfOf<f32>),
}

/// The coarse parts in the precision of the coarse solves.
enum AnyParts {
    Single(CoarseParts<f32>),
    Double(CoarseParts<f64>),
}

/// What the cycles of the CPU are made of, in the precision `F` of the cycle.
pub(crate) struct PartsIn<F: Storage> {
    basis: PvEoBasis<F>,
    coarse: AnyParts,
    /// The tops of the smoothers of Ã and of Ã_L^(α).
    tops: (f64, f64),
}

pub(crate) enum CpuParts {
    Double(PartsIn<f64>),
    Single(PartsIn<f32>),
}

/// The test vectors for `ops` stored as `F` (`Backend::rebuild`).
fn vectors_in<F: Storage, G: GammaRepre>(
    ops: (&Mdwf<F, G>, &Mdwf<F, G>),
    lat: &Arc<Lattice>,
    setup: &DwfMultigridSetup,
    kept: Option<HalfOf<F>>,
    due: bool,
) -> Result<HalfOf<F>, Box<dyn Error>> {
    let agg = Aggregation::new(lat, setup.block);
    let (op, pv) = ops;
    Ok(match kept {
        None => {
            let (y, _) = pv_eo_low_modes(op, pv, lat, setup.low_modes);
            let mut y = y.select(&(0..setup.test_vectors).collect::<Vec<_>>());
            pv_eo_bootstrap(op, pv, &agg, &mut y, setup.bootstrap)?;
            y
        }
        Some(mut y) if due => {
            pv_eo_bootstrap(op, pv, &agg, &mut y, setup.bootstrap)?;
            y
        }
        Some(y) => y,
    })
}

/// 1.1 |λ_max| of Ã of `op` over `pv` in `form`: the top of a smoother's interval, as
/// `PvEoCycle` takes it.
fn top_of<F: Storage, G: GammaRepre>(
    (op, pv): (&Mdwf<F, G>, &Mdwf<F, G>),
    form: EoForm,
    lat: &Arc<Lattice>,
    index: &EvenOddIndex,
    params: PvEoParams,
) -> f64 {
    let mut start = random_even::<F>(lat, index, op.params().ls, 1, params.seed);
    let like = start.zeros_like();
    1.1 * largest_magnitude(
        &mut PvEvenOdd::new(op, pv, form, &like),
        &mut start,
        params.power_steps,
    )
}

/// P, the coarse parts with their levels and the tops for `ops` stored as `F`
/// (`Backend::rebuild`).
fn parts_in<F: Storage, G: GammaRepre>(
    (op, pv): (&Mdwf<F, G>, &Mdwf<F, G>),
    d_alpha: Option<&Mdwf<F, G>>,
    lat: &Arc<Lattice>,
    index: &EvenOddIndex,
    setup: &DwfMultigridSetup,
    y: &HalfOf<F>,
) -> Result<PartsIn<F>, Box<dyn Error>> {
    let agg = Aggregation::new(lat, setup.block);
    let storage = match setup.basis_single {
        true => BasisStorage::Single,
        false => BasisStorage::Half,
    };
    let basis = pv_eo_basis_stored(op, pv, &agg, y, storage)?;
    let (super_blocks, refine) = setup.level_lists();
    let vectors = (!super_blocks.is_empty()).then(|| pv_eo_coarse_vectors(op, pv, &basis, y));
    let outer = (op, pv, EoForm::Right);
    macro_rules! made {
        ($ct:ty, $variant:ident) => {{
            let mut c = PvEoCycle::<G, $ct, crate::Aos, F>::with_coarse_storage(
                op,
                pv,
                outer,
                &basis,
                setup.cycle,
            )?;
            if let Some(v) = &vectors {
                c.add_levels_with(v, super_blocks, refine)?;
                if setup.levels.for_pauli_villars {
                    c.add_pv_levels(v, super_blocks)?;
                }
            }
            AnyParts::$variant(c.into_parts())
        }};
    }
    let coarse = match setup.coarse_double {
        false => made!(f32, Single),
        true => made!(f64, Double),
    };
    let hi = match &coarse {
        AnyParts::Single(p) => p.top(),
        AnyParts::Double(p) => p.top(),
    };
    let hi_alpha = match d_alpha {
        Some(d) => top_of((d, pv), EoForm::Left, lat, index, setup.cycle),
        None => hi,
    };
    Ok(PartsIn {
        basis,
        coarse,
        tops: (hi, hi_alpha),
    })
}

/// The cycle of `parts` for `ops` (`d_alpha`: of Ã_L^(α)), acting on fields stored as `F`.
fn cycle_in<'a, F: Storage, G: GammaRepre>(
    (d, pv): (&'a Mdwf<F, G>, &'a Mdwf<F, G>),
    d_alpha: Option<&'a Mdwf<F, G>>,
    parts: &'a PartsIn<F>,
    params: PvEoParams,
) -> Box<dyn LinearFn<HalfOf<F>> + 'a> {
    let (outer, hi) = match d_alpha {
        Some(a) => ((a, pv, EoForm::Left), parts.tops.1),
        None => ((d, pv, EoForm::Right), parts.tops.0),
    };
    match &parts.coarse {
        AnyParts::Single(p) => Box::new(PvEoCycle::from_parts(
            d,
            pv,
            outer,
            &parts.basis,
            params,
            p.clone().with_top(hi),
        )),
        AnyParts::Double(p) => Box::new(PvEoCycle::from_parts(
            d,
            pv,
            outer,
            &parts.basis,
            params,
            p.clone().with_top(hi),
        )),
    }
}

/// D, D_PV and D^(α) in single precision.
type SingleOps<G> = (Mdwf<f32, G>, Mdwf<f32, G>, Option<Mdwf<f32, G>>);

/// The cycles of one solve on the CPU: the operators of the cycle (converted to single
/// precision with `fine_single`) and the parts.
struct CpuCycles<'a, G: GammaRepre> {
    ops: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>),
    d_alpha: Option<&'a Mdwf<f64, G>>,
    /// D, D_PV and D^(α) in single precision, on the links of now.
    single: Option<SingleOps<G>>,
    parts: &'a CpuParts,
    params: PvEoParams,
    like: HalfOf<f32>,
}

impl<G: GammaRepre> Cycles<HalfPsi> for CpuCycles<'_, G> {
    fn cycle(&self, neff: bool) -> Box<dyn LinearFn<HalfPsi> + '_> {
        match (self.parts, &self.single) {
            (CpuParts::Double(p), _) => {
                cycle_in(self.ops, self.d_alpha.filter(|_| neff), p, self.params)
            }
            (CpuParts::Single(p), Some((d, pv, d_alpha))) => {
                let m = cycle_in((d, pv), d_alpha.as_ref().filter(|_| neff), p, self.params);
                Box::new(Convert::new(m, &self.like))
            }
            (CpuParts::Single(_), None) => unreachable!("made with fine_single"),
        }
    }
}

impl<G: GammaRepre> Backend for OnCpu<G> {
    type Op = Mdwf<f64, G>;
    type Field = Psi;
    type Half = HalfPsi;
    type Split = EvenOddIndex;
    type Vectors = CpuVectors;
    type Parts = CpuParts;

    fn split(&self) -> &EvenOddIndex {
        &self.index
    }

    fn halves(&self, op: &Mdwf<f64, G>, lat: &Arc<Lattice>) -> (HalfPsi, HalfPsi) {
        let half = |parity| {
            let mut h = HalfPsi::with_shape(lat, 24, op.params().ls, 1, 1);
            h.set_parity(parity);
            h
        };
        (half(Parity::Even), half(Parity::Odd))
    }

    fn gamma5(op: &Mdwf<f64, G>, out: &mut HalfPsi, inp: &HalfPsi) {
        op.mult_gamma5_on(out, inp);
    }

    fn mult_q(op: &Mdwf<f64, G>, out: &mut Psi, inp: &Psi) {
        op.mult_q(out, inp, false);
    }

    fn with_alpha(op: &Mdwf<f64, G>, alpha: f64) -> Result<Mdwf<f64, G>, Box<dyn Error>> {
        let mut a = Mdwf::converted_from(op);
        a.set_alpha(alpha);
        Ok(a)
    }

    fn rebuild(
        &self,
        (d, pv): (&Mdwf<f64, G>, &Mdwf<f64, G>),
        d_alpha: Option<&Mdwf<f64, G>>,
        lat: &Arc<Lattice>,
        setup: &DwfMultigridSetup,
        kept: Option<CpuVectors>,
        due: bool,
    ) -> Result<(CpuVectors, CpuParts, f64), Box<dyn Error>> {
        let start = std::time::Instant::now();
        let index = &self.index;
        Ok(match setup.fine_single {
            false => {
                let kept = match kept {
                    Some(CpuVectors::Double(y)) => Some(y),
                    _ => None,
                };
                let y = vectors_in((d, pv), lat, setup, kept, due)?;
                let seconds = start.elapsed().as_secs_f64();
                let parts = parts_in((d, pv), d_alpha, lat, index, setup, &y)?;
                (CpuVectors::Double(y), CpuParts::Double(parts), seconds)
            }
            true => {
                let ops = (Mdwf::converted_from(d), Mdwf::converted_from(pv));
                let alpha = d_alpha.map(Mdwf::converted_from);
                let kept = match kept {
                    Some(CpuVectors::Single(y)) => Some(y),
                    _ => None,
                };
                let y = vectors_in((&ops.0, &ops.1), lat, setup, kept, due)?;
                let seconds = start.elapsed().as_secs_f64();
                let parts = parts_in((&ops.0, &ops.1), alpha.as_ref(), lat, index, setup, &y)?;
                (CpuVectors::Single(y), CpuParts::Single(parts), seconds)
            }
        })
    }

    fn cycles<'a>(
        &'a self,
        ops: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>),
        d_alpha: Option<&'a Mdwf<f64, G>>,
        lat: &Arc<Lattice>,
        setup: &DwfMultigridSetup,
        parts: &'a CpuParts,
    ) -> Result<Box<dyn Cycles<HalfPsi> + 'a>, Box<dyn Error>> {
        let single = match parts {
            CpuParts::Single(_) => Some((
                Mdwf::converted_from(ops.0),
                Mdwf::converted_from(ops.1),
                d_alpha.map(Mdwf::converted_from),
            )),
            CpuParts::Double(_) => None,
        };
        let mut like = HalfOf::<f32>::with_shape(lat, 24, ops.0.params().ls, 1, 1);
        like.set_parity(Parity::Even);
        Ok(Box::new(CpuCycles {
            ops,
            d_alpha,
            single,
            parts,
            params: setup.cycle,
            like,
        }))
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
