//! The multigrid of domain-wall fermions in the solves of HMC: (D†D)⁻¹ b as two solves, D† w = b
//! through Ã† (the Pauli-Villars operator with D and D_PV exchanged, `fermion::adjoint_*`) with
//! the cycle M† and then D x = w through Ã with M, both by GCR on the even sites, repeated on
//! the residual of D†D x = b until it is below the tolerance. Ã and Ã† share the prolongation P,
//! made from the test vectors y; along a trajectory y is kept and P and the coarse operators are
//! made again for the links of every solve, and y itself is improved (the bootstrap, from the y
//! of before) when the pseudofermions are drawn and after every `refresh_every` force solves.
//!
//! The even-odd Ratio action solves K†K on the even sites, K = T_ee⁻¹ Ŝ (the left
//! `SchurFactor`): K⁻† r = T_ee† Ŝ⁻† r and K⁻¹ w = Ŝ⁻¹ T_ee w, where Ŝ⁻¹ u is the even part of
//! the solution of D x = (u, 0) and Ŝ⁻† that of D† (`normal_solve_half`), the same two solves.

use super::fermion_action::Psi;
use super::HalfPsi;
use crate::fermion::{
    adjoint_right_hand_side, adjoint_solution, Dagger, EoForm, EvenOdd, Mdwf, PvEvenOdd,
    SchurFactor,
};
use crate::kernel::GammaRepre;
use crate::solver::preconditioner::multigrid::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseKrylov,
    LowModeParams, PvEoBasis, PvEoCycle, PvEoParams,
};
use crate::solver::{Gcr, SolveError, SolveInfo, Solver, Stop};
use crate::{extract_parity, insert_parity, Lattice, LinearFn, LinearOp, Parity, Vector, C64, ND};
use std::sync::Arc;

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
    /// The coarse solves in double precision; false (the default): in single precision. The
    /// fine part of the cycle is in double precision either way.
    pub coarse_double: bool,
    /// GCR of each solve on the even sites: directions kept, and when it stops.
    pub directions: usize,
    pub inner: Stop,
    /// The bootstrap is run again after every this many force solves (and when the
    /// pseudofermions are drawn); 0: only then.
    pub refresh_every: usize,
    vectors: Option<HalfPsi>,
    since: usize,
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
            directions: 8,
            inner,
            refresh_every: 0,
            vectors: None,
            since: 0,
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
        let due = fresh || (self.refresh_every > 0 && self.since >= self.refresh_every);
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
    with_solves(op, pv, setup, flags, b.lattice(), |solves| {
        two_solves_half(op, solves, z, b, stop)
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
    let basis = setup.basis(op, pv, lat, (fresh, md));
    let start = std::time::Instant::now();
    let made = "the cycle is made";
    let double = setup.coarse_double;
    let cycle = match double {
        false => Box::new(
            PvEoCycle::<G, f32>::with_coarse_storage(
                op,
                pv,
                (op, pv, EoForm::Right),
                &basis,
                setup.cycle,
            )
            .expect(made),
        ) as Box<dyn LinearFn<HalfPsi>>,
        true => Box::new(
            PvEoCycle::<G, f64>::with_coarse_storage(
                op,
                pv,
                (op, pv, EoForm::Right),
                &basis,
                setup.cycle,
            )
            .expect(made),
        ),
    };
    // In M† the second factor is P†D†P, of the quark operator: its exact adjoint by BiCGStab
    // took fewer iterations than MINRES on its Γ5 form on the v2 16⁴ configuration (17 against
    // 18 with stout links at M5 1.0, 57 against 76 at M5 1.8, m 0.001), while in M (P†D_PV P,
    // well conditioned) MINRES is far better.
    let dual = PvEoParams {
        pv_solver: CoarseKrylov::BiCgStab,
        ..setup.cycle
    };
    let cycle_dag = match double {
        false => Box::new(
            PvEoCycle::<G, f32>::with_coarse_storage(pv, op, (pv, op, EoForm::Right), &basis, dual)
                .expect(made),
        ) as Box<dyn LinearFn<HalfPsi>>,
        true => Box::new(
            PvEoCycle::<G, f64>::with_coarse_storage(pv, op, (pv, op, EoForm::Right), &basis, dual)
                .expect(made),
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
        cycle,
        cycle_dag,
        inner: 0,
        count: 0,
    };
    let result = run(&mut solves);
    setup.inner_solves += solves.count;
    setup.seconds.solve += start.elapsed().as_secs_f64();
    result.map(|mut info| {
        info.iterations = solves.inner;
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
    cycle: Box<dyn LinearFn<HalfPsi> + 'c>,
    cycle_dag: Box<dyn LinearFn<HalfPsi> + 'c>,
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
        let info = self
            .gcr
            .solve(&mut a, Some(self.cycle.as_mut()), &mut y, &rhs);
        a.solution(e, &y, r);
        self.counted(info)
    }

    /// D† e = r through Ã† and M†.
    fn adjoint(&mut self, e: &mut Psi, r: &Psi) -> Result<(), SolveError> {
        let mut a_dag = PvEvenOdd::new(self.pv, self.op, EoForm::Right, &self.like);
        let (mut rhs, mut y) = (self.like.zeros_like(), self.like.zeros_like());
        adjoint_right_hand_side(self.op, self.index, &mut rhs, r);
        let info = self
            .gcr
            .solve(&mut a_dag, Some(self.cycle_dag.as_mut()), &mut y, &rhs);
        adjoint_solution(self.op, self.pv, self.index, e, &y, r);
        self.counted(info)
    }
}

/// D†D x = b by solving D† w = r and D e = w on the residual r until |r|²/|b|² is below the
/// tolerance (at most `stop.max_restarts` + 1 times).
fn two_solves<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    solves: &mut Solves<'_, '_, G>,
    x: &mut Psi,
    b: &Psi,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    use crate::fermion::{Bound, Variant};
    let n = b.num_vectors();
    let mut b_norm = vec![0.0; n];
    b.norm_sqr(&mut b_norm);
    let mut d = Bound::new(op, Variant::D);
    let (mut t, mut r, mut w, mut e) = (
        b.zeros_like(),
        b.zeros_like(),
        b.zeros_like(),
        b.zeros_like(),
    );
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
fn two_solves_half<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    solves: &mut Solves<'_, '_, G>,
    z: &mut HalfPsi,
    b: &HalfPsi,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    use super::ForceOp;
    let n = b.num_vectors();
    assert_eq!(n, 1, "one right-hand side");
    let lat = b.lattice().clone();
    let index = solves.index;
    let mut odd = b.zeros_like();
    odd.set_parity(Parity::Odd);
    let mut k = SchurFactor::new(op, EoForm::Left, b, &odd);
    let mut work = op.even_odd_work();
    let mut b_norm = vec![0.0; n];
    b.norm_sqr(&mut b_norm);
    let [mut t, mut r, mut s, mut w, mut e] = std::array::from_fn(|_| b.zeros_like());
    let (mut source, mut whole) = (op.new_field(&lat), op.new_field(&lat));
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
        insert_parity(&mut source, &r, index);
        solves.adjoint(&mut whole, &source)?;
        extract_parity(&mut s, &whole, index, Parity::Even);
        op.diag(&mut work, &mut w, &s, Parity::Even, Dagger::Yes);
        // e = Ŝ⁻¹ T_ee w.
        op.diag(&mut work, &mut s, &w, Parity::Even, Dagger::No);
        source.set_zero();
        insert_parity(&mut source, &s, index);
        solves.forward(&mut whole, &source)?;
        extract_parity(&mut e, &whole, index, Parity::Even);
        z.axpy(&vec![C64::new(1.0, 0.0); n], &e);
    }
    Err(SolveError::NotConverged(info))
}
