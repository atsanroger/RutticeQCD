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
//! With `through_d` (`MultiGrid_HMC.dagger_solve : Through_D`) the solves with D† are instead one solve with D each: for Möbius
//! fermions with constant b, c, Γ5 D† Γ5 = B5 D B5⁻¹ with B5 = b − cY, which acts within a site,
//! so D⁻† = Γ5 B5 D⁻¹ B5⁻¹ Γ5 on all sites and on the even sites alike (the note, Sec. IV.D).
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
//! have grown to `rebuild_growth` times those after the last rebuild, and then after a bootstrap
//! of y for the links of now. Where the sweeps stall, the solve is done again after a bootstrap,
//! twice.
//!
//! The even-odd Ratio action solves K†K on the even sites, K = T_ee⁻¹ Ŝ (the left
//! `SchurFactor`): K⁻† r = T_ee† Ŝ⁻† r and K⁻¹ w = Ŝ⁻¹ T_ee w, where Ŝ⁻¹ u is the even part of
//! the solution of D x = (u, 0) and Ŝ⁻† that of D† (`normal_solve_half`), the same two solves.
//!
//! All of that is written once, here (`with_solves`, `Solves`); a backend supplies its fields,
//! the test vectors, P and the coarse parts, and the cycles made of them
//! (`solver::preconditioner::multigrid::MdwfBackend`: `OnCpu`, and on the device `OnDevice`),
//! as it does for the propagator runs.

use super::fermion_action::Psi;
use super::HalfPsi;
use crate::fermion::{
    adjoint_by_gamma5, adjoint_by_gamma5_even, Dagger, EoForm, EvenOdd, Mdwf, PvRatio, SchurFactor,
    SplitParity,
};
use crate::kernel::GammaRepre;
use crate::solver::preconditioner::multigrid::{
    Cycles, MdwfBackend, MdwfMultigridSettings, OnCpu, SetupStage,
};
use crate::solver::{refine_normal, Fgmres, Gcr, SolveError, SolveInfo, Solver, Stop};
use crate::{Lattice, LinearFn, LinearOp, Parity, Vector};
use std::any::Any;
use std::error::Error;
use std::sync::Arc;

/// The setup of the multigrid of an action and what it keeps along a trajectory.
pub struct DwfMultigridSetup {
    /// What the parameter file asks of the multigrid, as the propagator runs take it.
    pub mg: MdwfMultigridSettings,
    /// The Krylov solver of each solve on the even sites: GCR keeping `directions` directions,
    /// or with `fgmres` FGMRES restarted after `directions` steps; and when it stops.
    pub directions: usize,
    pub inner: Stop,
    pub(crate) fgmres: bool,
    /// The bootstrap is run again after this many force solves since the last one, whatever made
    /// that (the pseudofermions drawn, a rebuild, a stall); 0: only when they are needed.
    pub(crate) refresh_every: usize,
    /// Keep P and the coarse parts from solve to solve and make them again, after a bootstrap of
    /// the vectors, when the GCR iterations per solve on the even sites have grown to this many
    /// times those after the last rebuild (module documentation); 0: make them for every solve.
    pub(crate) rebuild_growth: f64,
    /// The solves with D† by one solve with D, D⁻† = Γ5 B5 D⁻¹ B5⁻¹ Γ5 (constant b, c); false:
    /// GCR with Γ5 M Γ5.
    pub(crate) through_d: bool,
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
    /// The next rebuild follows a stalled solve: its bootstrap does at least one round, whatever
    /// the gain (`BootstrapParams::min_rounds`).
    after_stall: bool,
    /// Rebuilds of P and the coarse parts for new links, and solves that reused them.
    pub(crate) rebuilds: usize,
    pub(crate) reuses: usize,
    /// Full setups, bootstraps from the vectors of before, and solves on the even sites so far.
    pub setups: usize,
    pub(crate) refreshes: usize,
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
    pub fn new(mg: MdwfMultigridSettings, inner: Stop) -> Self {
        Self {
            mg,
            directions: 8,
            inner,
            fgmres: false,
            refresh_every: 0,
            rebuild_growth: 0.0,
            through_d: false,
            alpha: 1.0,
            state: None,
            since: 0,
            baseline: None,
            rebuild_due: false,
            after_stall: false,
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
}

/// The sweeps of a solve, over the solves on the even sites (`two_solves`, `two_solves_half`).
type Run<'r, V, H> = dyn FnMut(&mut dyn EvenSolves<V, H>) -> Result<SolveInfo, SolveError> + 'r;

/// What the setup keeps for a backend between the solves.
struct Kept<B: MdwfBackend> {
    vectors: Option<B::Vectors>,
    parts: Option<B::Parts>,
}

/// `run` with the solves on the even sites of `setup`'s multigrid for the current links of
/// `ops`, and the bookkeeping of the setup; the iterations of the result are the GCR iterations,
/// those of the attempts that stalled included.
pub(crate) fn with_solves<B: MdwfBackend + 'static>(
    backend: &B,
    ops: (&B::Op, &B::Op),
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    lat: &Arc<Lattice>,
    mut run: impl FnMut(&mut dyn EvenSolves<B::Field, B::Half>) -> Result<SolveInfo, SolveError>,
) -> Result<SolveInfo, SolveError> {
    let mut attempt = |setup: &mut DwfMultigridSetup, flags| {
        solves_once(backend, ops, setup, flags, lat, &mut run)
            .unwrap_or_else(|e| panic!("the multigrid of HMC failed: {e}"))
    };
    let mut result = attempt(setup, flags);
    let mut stalled = 0;
    // When the sweeps stall: the test vectors improved by a bootstrap for the links of now, and P
    // and the coarse parts made of them (a rebuild bootstraps first), twice. Each goes on from the
    // solution so far (`run` starts from it).
    for (what, fresh) in [("a bootstrap", flags.0), ("another bootstrap", true)] {
        let Err(SolveError::NotConverged(info)) = &result else {
            break;
        };
        eprintln!(
            "the multigrid of HMC stalled at |r|²/|b|² {:.1e}: again with {what}",
            info.residual_sqr.iter().fold(0.0_f64, |m, r| m.max(*r))
        );
        stalled += info.iterations;
        setup.rebuild_due = true;
        setup.after_stall = true;
        // The same solve: a force solve counts once towards `refresh_every`.
        result = attempt(setup, (fresh, false));
    }
    if let Ok(info) | Err(SolveError::NotConverged(info)) = &mut result {
        info.iterations += stalled;
    }
    result
}

/// One attempt of `with_solves`.
fn solves_once<B: MdwfBackend + 'static>(
    backend: &B,
    (d, pv): (&B::Op, &B::Op),
    setup: &mut DwfMultigridSetup,
    (fresh, md): (bool, bool),
    lat: &Arc<Lattice>,
    run: &mut Run<'_, B::Field, B::Half>,
) -> Result<Result<SolveInfo, SolveError>, Box<dyn Error>> {
    let start = std::time::Instant::now();
    let caching = setup.rebuild_growth > 0.0;
    // A rebuild improves the vectors first: P made of the kept y for links that have moved
    // away from those y was made for is worse than the P kept.
    let due = setup.due(fresh) || setup.rebuild_due;
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
    // The cycles of Ã and, with Neff's α, of Ã_L^(α).
    let mut outers = vec![(d, pv, EoForm::Right)];
    if let Some(a) = &d_alpha {
        outers.push((a, pv, EoForm::Left));
    }
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
            let quiet = &mut |_: SetupStage<'_>| {};
            // After a stall the probe could end the bootstrap at once on vectors that gave it.
            let mut mg = setup.mg;
            if setup.after_stall {
                mg.bootstrap.min_rounds = mg.bootstrap.min_rounds.max(1);
            }
            let (vectors, parts, vector_seconds) =
                backend.rebuild((d, pv), &outers, lat, &mg, kept_vectors, due, quiet)?;
            // The vectors count as the setup, P and the coarse parts as the rebuild.
            setup.seconds.setup += vector_seconds;
            setup.seconds.rebuild += start.elapsed().as_secs_f64() - vector_seconds;
            if due {
                setup.since = 0;
            }
            setup.rebuilds += 1;
            // Consumed; with the cache `after_solve` decides the next one.
            setup.rebuild_due = false;
            setup.after_stall = false;
            kept.vectors = Some(vectors);
            parts
        }
    };
    if md {
        setup.since += 1;
    }
    let start = std::time::Instant::now();
    let (inner, count, mut result) = {
        let cycles = backend.cycles((d, pv), &outers, lat, &setup.mg, &parts)?;
        let (even, odd) = backend.halves(d, lat);
        let mut solves = Solves::<B> {
            backend,
            d,
            pv,
            d_alpha: d_alpha.as_ref(),
            b5: match setup.through_d {
                true => Some(B::b5_of(d)?),
                false => None,
            },
            even,
            odd,
            krylov: match setup.fgmres {
                // As many steps in all as GCR, in cycles of `directions`.
                true => Box::new(Fgmres::new(Stop {
                    max_iterations: setup.directions,
                    max_restarts: setup.inner.max_iterations / setup.directions.max(1),
                    ..setup.inner
                })),
                false => Box::new(Gcr::new(setup.inner, setup.directions)),
            },
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
    // The sweeps count no iterations of their own: those of the solve are GCR's, stalled or not.
    if let Ok(info) | Err(SolveError::NotConverged(info)) = &mut result {
        info.iterations = inner;
    }
    Ok(result)
}

/// The solves on the even sites of one solve of D†D, with the cycles of the backend.
struct Solves<'a, B: MdwfBackend> {
    backend: &'a B,
    d: &'a B::Op,
    pv: &'a B::Op,
    d_alpha: Option<&'a B::Op>,
    /// With `through_d`: D† through D, Γ5 D† Γ5 = B5 D B5⁻¹.
    b5: Option<B::Op>,
    even: B::Half,
    odd: B::Half,
    krylov: Box<dyn Solver<B::Half>>,
    cycles: Box<dyn Cycles<B::Half> + 'a>,
    /// GCR iterations and solves so far.
    inner: usize,
    count: usize,
}

impl<B: MdwfBackend> Solves<'_, B> {
    /// `out = B5⁻¹ Γ5 inp` (`inverse`) or `out = Γ5 B5 inp` on fields of the whole lattice, parity
    /// by parity: Γ5 and B5 act within a site.
    fn site_b5(&self, b5: &B::Op, out: &mut B::Field, inp: &B::Field, inverse: bool) {
        let split = self.backend.split();
        let mut work = b5.even_odd_work();
        for parity in [Parity::Even, Parity::Odd] {
            let mut h = split.half_like(inp, parity);
            split.extract(&mut h, inp, parity);
            let mut g = h.zeros_like();
            match inverse {
                true => {
                    B::gamma5(self.d, &mut g, &h);
                    b5.diag_inv(&mut work, &mut h, &g, parity, Dagger::No);
                }
                false => {
                    b5.diag(&mut work, &mut g, &h, parity, Dagger::No);
                    B::gamma5(self.d, &mut h, &g);
                }
            }
            split.insert(out, &h);
        }
    }

    fn counted(&mut self, info: Result<SolveInfo, SolveError>) -> Result<(), SolveError> {
        self.inner += match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(e) => return Err(e),
        };
        self.count += 1;
        Ok(())
    }
}

impl<B: MdwfBackend> EvenSolves<B::Field, B::Half> for Solves<'_, B> {
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
        let mut m = self.cycles.cycle(usize::from(self.d_alpha.is_some()));
        let info = self.krylov.solve(&mut a, Some(m.as_mut()), &mut y, &rhs);
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

    /// D† e = r with Γ5 M Γ5 (`adjoint_by_gamma5`), or with `through_d` by one solve
    /// with D, e = Γ5 B5 D⁻¹ B5⁻¹ Γ5 r.
    fn adjoint(&mut self, e: &mut B::Field, r: &B::Field) -> Result<(), SolveError> {
        if let Some(b5) = self.b5.take() {
            let mut src = r.zeros_like();
            self.site_b5(&b5, &mut src, r, true);
            let done = self.forward(e, &src);
            if done.is_ok() {
                src.copy_from(e);
                self.site_b5(&b5, e, &src, false);
            }
            self.b5 = Some(b5);
            return done;
        }
        let mut m = self.cycles.cycle(0);
        let d = self.d;
        let gamma5 = |out: &mut B::Half, inp: &B::Half| B::gamma5(d, out, inp);
        let info = adjoint_by_gamma5(
            (self.d, self.pv),
            self.backend.split(),
            &gamma5,
            m.as_mut(),
            self.krylov.as_mut(),
            (&self.even, &self.odd),
            e,
            r,
        );
        drop(m);
        self.counted(info)
    }

    /// `forward` for an r with nothing on the odd sites, on the even sites only: e is the even
    /// part of D⁻¹ (r, 0); r is overwritten.
    fn forward_even(&mut self, e: &mut B::Half, r: &mut B::Half) -> Result<(), SolveError> {
        let (d, form) = match self.d_alpha {
            Some(d) => (d, EoForm::Left),
            None => (self.d, EoForm::Right),
        };
        let mut a = PvRatio::new(d, self.pv, form, &self.even, &self.odd);
        // The right-hand side in e; r, not needed after it, holds y.
        let (rhs, y) = (e, r);
        crate::fit(rhs, y);
        a.right_hand_side_even(rhs, y);
        y.set_zero();
        let mut m = self.cycles.cycle(usize::from(self.d_alpha.is_some()));
        let info = self.krylov.solve(&mut a, Some(m.as_mut()), y, rhs);
        drop(m);
        let e = rhs;
        match self.d_alpha {
            Some(d) => {
                let mut x = e.zeros_like();
                a.solution_even(&mut x, y);
                B::mult_q_half(d, e, &x);
            }
            None => a.solution_even(e, y),
        }
        self.counted(info)
    }

    /// `adjoint` for an r with nothing on the odd sites, on the even sites only: e is the even
    /// part of D⁻† (r, 0) (`adjoint_by_gamma5_even`, for which b′_e = r; with
    /// `through_d` one solve with D, Γ5 B5 D⁻¹ B5⁻¹ Γ5); `spare` is overwritten.
    fn adjoint_even(
        &mut self,
        e: &mut B::Half,
        r: &B::Half,
        spare: &mut B::Half,
    ) -> Result<(), SolveError> {
        // exp: D† e = (r, 0) as e = Γ5 B5 D⁻¹ B5⁻¹ Γ5 (r, 0): Γ5 and B5 act within a site, so
        // the source stays on the even sites and one solve with D on them does it.
        if let Some(b5) = self.b5.take() {
            let even = Parity::Even;
            let mut work = b5.even_odd_work();
            crate::fit(e, r);
            crate::fit(spare, r);
            B::gamma5(self.d, e, r);
            b5.diag_inv(&mut work, spare, e, even, Dagger::No);
            let done = self.forward_even(e, spare);
            if done.is_ok() {
                b5.diag(&mut work, spare, e, even, Dagger::No);
                B::gamma5(self.d, e, spare);
            }
            self.b5 = Some(b5);
            return done;
        }
        let mut m = self.cycles.cycle(0);
        let d = self.d;
        let gamma5 = |out: &mut B::Half, inp: &B::Half| B::gamma5(d, out, inp);
        let info = adjoint_by_gamma5_even(
            (self.d, self.pv),
            &gamma5,
            m.as_mut(),
            self.krylov.as_mut(),
            (&self.even, &self.odd),
            r,
            (e, spare),
        );
        drop(m);
        self.counted(info)
    }
}

/// The two solves on the even sites a solve of D†D is made of, for fields `V` of the whole
/// lattice: D e = r through Ã and its cycle M, D† e = r with Γ5 M Γ5; and the same for sources
/// with nothing on the odd sites, on fields `H` of the even sites alone.
pub(crate) trait EvenSolves<V, H> {
    fn forward(&mut self, e: &mut V, r: &V) -> Result<(), SolveError>;
    fn adjoint(&mut self, e: &mut V, r: &V) -> Result<(), SolveError>;
    /// `r` is overwritten.
    fn forward_even(&mut self, e: &mut H, r: &mut H) -> Result<(), SolveError>;
    /// `spare` is overwritten.
    fn adjoint_even(&mut self, e: &mut H, r: &H, spare: &mut H) -> Result<(), SolveError>;
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
pub(crate) fn normal_solve_half<G: GammaRepre>(
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

/// D†D x = b by solving D† w = r and D e = w on the residual r until |r|²/|b|² is below the
/// tolerance (at most `stop.max_restarts` + 1 times).
pub(crate) fn two_solves<O: super::ForceOp, H>(
    op: &O,
    solves: &mut dyn EvenSolves<O::Field, H>,
    x: &mut O::Field,
    b: &O::Field,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    use crate::fermion::{Bound, Variant};
    let mut d = Bound::new(op, Variant::D);
    refine_normal(&mut d, 0.0, x, b, stop, &mut |_, _, e, r, w, _| {
        solves.adjoint(w, r)?;
        solves.forward(e, w)?;
        Ok(0)
    })
}

/// The left Schur factor K of `op` with its fields made for each application (`SchurFactor`).
struct KOnDemand<'k, F, H> {
    op: &'k F,
    like: (&'k H, &'k H),
}

impl<F: crate::fermion::EvenOdd<H>, H: Vector> LinearFn<H> for KOnDemand<'_, F, H> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        SchurFactor::new(self.op, EoForm::Left, self.like.0, self.like.1).mult(out, inp);
    }
}

impl<F: crate::fermion::EvenOdd<H>, H: Vector> LinearOp<H> for KOnDemand<'_, F, H> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        SchurFactor::new(self.op, EoForm::Left, self.like.0, self.like.1).mult_dag(out, inp);
    }
}

/// K†K z = b on the even sites, K = T_ee⁻¹ Ŝ, by w = T_ee† Ŝ⁻† r and e = Ŝ⁻¹ T_ee w on the
/// residual r, each Ŝ⁻¹ (Ŝ⁻†) the even part of a solve of D (D†) with no source on the odd
/// sites, until |r|²/|b|² is below the tolerance (at most `stop.max_restarts` + 1 times).
pub(crate) fn two_solves_half<O: super::EvenOddHalves>(
    op: &O,
    lat: &Arc<Lattice>,
    solves: &mut dyn EvenSolves<O::Field, O::Half>,
    z: &mut O::Half,
    b: &O::Half,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    assert_eq!(b.num_vectors(), 1, "one right-hand side");
    // K for the residual between the sweeps, its fields made for each application and given
    // back after it, so that it holds nothing during the solves; a field of the odd sites for
    // its shape (on the device with no vectors, on the CPU a zero field).
    let odd = op.shape_half(lat, Parity::Odd);
    let mut k = KOnDemand {
        op,
        like: (b, &odd),
    };
    let mut work = op.even_odd_work();
    // The sources have nothing on the odd sites: the solves on the even sites alone, no field of
    // the whole lattice.
    refine_normal(&mut k, 0.0, z, b, stop, &mut |_, x, e, r, t, _| {
        // x is not needed by the two solves: set aside meanwhile (on the device, to the host).
        let back = op.set_aside(x);
        // t = Ŝ⁻† r (e as the solver's field), then T_ee T_ee† t (e as the field between), then
        // e = Ŝ⁻¹ t.
        let mut solve = || {
            solves.adjoint_even(t, r, e)?;
            op.diag(&mut work, e, t, Parity::Even, Dagger::Yes);
            op.diag(&mut work, t, e, Parity::Even, Dagger::No);
            solves.forward_even(e, t)
        };
        let done = solve();
        if let Some(back) = back {
            *x = back();
        }
        done.map(|_| 0)
    })
}
