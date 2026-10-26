use super::cycle::{Corrects, Smooths, TwoGrid};
use super::levels::BlockMap;
use super::{solve_shared, CoarseOperator, Levels, Transfer};
use super::{Aggregation, Basis, CoarseOp, MultigridError, Sap, SignedMultilevel};
use crate::fermion::{Bound, HopTerms, Variant};
use crate::kernel::Number;
use crate::solver::{BiCgStab, Fgmres, Solve, Solver, Stop};
use crate::{Coarse, ConvertFrom, Field, Lattice, LinearFn, Spinor, Storage, Vector, C64, ND};
use std::cell::{Cell, Ref, RefCell};
use std::sync::Arc;

/// Parameters of the two-level multigrid preconditioner.
#[derive(Clone, Copy, Debug)]
pub struct MultigridParams {
    /// Size of the blocks: the aggregates of the coarse lattice and the blocks of the
    /// smoother.
    pub block: [usize; ND],
    /// Test vectors; the coarse lattice carries twice as many components per site.
    pub num_test_vectors: usize,
    /// Inverse iterations that improve the test vectors after the initial smoothing.
    pub setup_iterations: usize,
    /// The coarse-grid solver.
    pub coarse_solver: CoarseSolverChoice,
    /// When the coarse solve stops; for GMRES `max_iterations` is m. Not reaching the
    /// tolerance is not an error.
    pub coarse_stop: Stop,
    /// The smoother after (and, with `pre_smoothing`, before) the coarse correction.
    pub smoother: SmootherChoice,
    /// Smooth before the coarse correction too, from a zero start, instead of starting
    /// from x = b.
    pub pre_smoothing: bool,
    /// Precondition the coarse operator by its diagonal blocks.
    pub precondition_coarse_diagonal: bool,
    /// Solve the coarse system on the even coarse sites (the Schur complement, see
    /// `CoarseOp`), where the coarse lattice allows it, with or without levels below it, and
    /// nor `precondition_coarse_diagonal`.
    pub coarse_even_odd: bool,
    /// Seed of the random initial test vectors.
    pub seed: u64,
    /// Levels below the coarse lattice (`Multigrid::add_levels`): the super-block size of
    /// each, 0 for none.
    pub levels: [usize; 3],
}

/// The smoother of the cycle (the design notes, §7.6.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SmootherChoice {
    /// SAP: `cycles` cycles, `block_steps` minimal-residual steps per block.
    Sap { cycles: usize, block_steps: usize },
    /// GMRES(m) on the cycle's operator from a zero start: at most `iterations` steps, fewer
    /// if |r|²/|b|² drops below `tolerance_sqr`.
    Gmres {
        iterations: usize,
        tolerance_sqr: f64,
    },
}

/// The coarse-grid solver (the design notes, §7.6.0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseSolverChoice {
    BiCgStab,
    /// GMRES(m), restarted as `coarse_stop` allows.
    Gmres,
}

impl MultigridParams {
    /// The parameters of the Bridge++ test `test_alt_Multigrid.yaml`, with the given block
    /// size: 16 test vectors, 4 setup iterations, coarse solves to |r|²/|b|² < 2e-3 in at
    /// most 200 iterations, 4 SAP cycles with 6 minimal-residual steps per block.
    pub fn bridge(block: [usize; ND]) -> Self {
        Self {
            block,
            num_test_vectors: 16,
            setup_iterations: 4,
            coarse_solver: CoarseSolverChoice::BiCgStab,
            coarse_stop: Stop::simple(2e-3, 200),
            smoother: SmootherChoice::Sap {
                cycles: 4,
                block_steps: 6,
            },
            pre_smoothing: false,
            precondition_coarse_diagonal: false,
            coarse_even_odd: false,
            seed: 1,
            levels: [0; 3],
        }
    }
}

type Fine<T> = Field<T, Spinor>;

/// The members of the cycle (the design notes, §7.6.3); each can be exchanged. `V` the fine
/// fields, `C` the coarse ones.
pub struct CycleParts<'a, V, C> {
    /// The operator whose inverse the cycle approximates: D for clover, A for the
    /// domain-wall target operator.
    pub operator: Box<dyn LinearFn<V> + 'a>,
    /// Smoothing before the coarse correction, from a zero start. Without it the cycle
    /// starts from x = b, as Bridge++ `APrecond_MG` does.
    pub pre_smoother: Option<Box<dyn LinearFn<V> + 'a>>,
    /// Smoothing after the coarse correction. It also smooths the test vectors in the setup.
    pub post_smoother: Box<dyn LinearFn<V> + 'a>,
    /// Solves the coarse system; not reaching its tolerance is not an error.
    pub coarse_solver: Box<dyn Solver<C> + 'a>,
}

/// How the setup builds the test vectors and the coarse operator, beyond the defaults.
pub struct SetupHooks<'a, V> {
    /// Applied to every test vector after it was smoothed (domain-wall: D_PV).
    pub after_smoothing: Option<Box<dyn LinearFn<V> + 'a>>,
    /// A site-local factor R in the coarse operator P† D R P (domain-wall: C_PV^{-1}).
    pub coarse_right: Option<Box<dyn LinearFn<V> + 'a>>,
    /// Precondition the coarse operator by its diagonal blocks.
    pub precondition_coarse_diagonal: bool,
    /// Prepare the coarse operator for solves on the even coarse sites
    /// (`MultigridParams::coarse_even_odd`).
    pub coarse_even_odd: bool,
    /// Build the coarse operator as the Galerkin projection `P† A P` of this map, cut down
    /// to nearest neighbours (`CoarseOp::probe`), instead of from the parts of the fine
    /// operator; `coarse_right` is then not used.
    pub galerkin_of: Option<Box<dyn LinearFn<V> + 'a>>,
    /// Makes test vectors for a restriction of its own from the initial vectors: a smoother for the
    /// adjoint of the operator of the cycle.
    pub adjoint_smoother: Option<Box<dyn LinearFn<V> + 'a>>,
}

impl<V> Default for SetupHooks<'_, V> {
    fn default() -> Self {
        Self {
            after_smoothing: None,
            coarse_right: None,
            precondition_coarse_diagonal: false,
            coarse_even_odd: false,
            galerkin_of: None,
            adjoint_smoother: None,
        }
    }
}

/// The levels below the coarse lattice as the cycle uses them: their V-cycle, and the
/// inverses of the diagonal blocks of the coarse operator they are combined with.
pub struct CycleLevels<'a, C> {
    pub apply: Box<Levels<'a, C>>,
    pub blocks: Option<Box<dyn LinearFn<C> + 'a>>,
    /// Applications of the coarse operator to build them.
    pub applications: usize,
}

/// What the multigrid of operators of Wilson type needs of a backend: the operator's fields,
/// P from test vectors, the coarse operator on P, the levels below it, the smoother. The setup
/// and the cycle are written once (`WilsonMultigrid`); the CPU (`WilsonOnCpu`) and the device
/// (`cuda_bridge::WilsonOnDevice`) supply these.
pub trait MultigridBackend<'a> {
    type Fine: Vector + 'a;
    type Coarse: Vector + 'a;
    type Basis: Transfer<Self::Fine, Self::Coarse> + 'a;
    /// The coarse operator, as its solver and the preconditioner of that share it.
    type CoarseOp: CoarseOperator<Self::Coarse> + 'a;
    /// What the levels below the coarse lattice are made from.
    type CoarseTests;
    /// The operator of the cycle (D).
    fn operator(&self) -> Box<dyn LinearFn<Self::Fine> + 'a>;
    /// The smoother `choice` on the blocks of `agg`.
    fn smoother(
        &self,
        agg: &Arc<Aggregation>,
        choice: SmootherChoice,
    ) -> Box<dyn LinearFn<Self::Fine> + 'a>;
    /// P: the chiral halves (by the operator's Γ5) of the vectors of `x`, orthonormalised on
    /// the aggregates.
    fn basis(
        &mut self,
        agg: &Arc<Aggregation>,
        x: &Self::Fine,
    ) -> Result<Self::Basis, MultigridError>;
    /// The coarse operator on `basis` (with `restriction` for P† of a restriction of its own),
    /// as `hooks` asks, prepared for solves on the even coarse sites if `even_odd`.
    fn coarse_operator(
        &mut self,
        basis: &Self::Basis,
        restriction: Option<&Self::Basis>,
        hooks: &mut SetupHooks<'a, Self::Fine>,
        even_odd: bool,
    ) -> Result<Self::CoarseOp, MultigridError>;
    /// The current test vectors, orthonormalised on every aggregate, as one fine field: vector
    /// i is P applied to the coarse vector with ones at components i and n + i.
    fn test_vectors(&self, basis: &Self::Basis) -> Self::Fine;
    /// P† x of the smoothed test vectors `x`, from which the levels below are made.
    fn coarse_tests(&self, basis: &Self::Basis, x: &Self::Fine) -> Self::CoarseTests;
    /// Whether the coarse operator was preconditioned by its diagonal blocks, and so the
    /// right-hand side of a coarse solve must be.
    fn diagonal_preconditioned(coarse: &Self::CoarseOp) -> bool;
    fn apply_diagonal_inverse(coarse: &Self::CoarseOp, out: &mut Self::Coarse, inp: &Self::Coarse);
    /// The levels below the coarse lattice on super-blocks of `super_blocks[ℓ]` coarse sites
    /// per direction on level ℓ + 2.
    fn levels(
        coarse: &mut Self::CoarseOp,
        tests: &Self::CoarseTests,
        super_blocks: &[usize],
    ) -> Result<CycleLevels<'a, Self::Coarse>, MultigridError>;
}

/// Two-level multigrid as a preconditioner: an approximate inverse of the cycle's operator
/// A, to be passed as `m` to a flexible outer solver (FGMRES, or BiCGStab, which is the
/// flexible form with a preconditioner). Written once over its backend `B`.
pub struct WilsonMultigrid<'a, B: MultigridBackend<'a>> {
    backend: B,
    agg: Arc<Aggregation>,
    /// The test vectors of the last setup round restricted to the coarse lattice: test
    /// vectors for levels below it.
    coarse_tests: B::CoarseTests,
    hooks: SetupHooks<'a, B::Fine>,
    cycle: TwoGrid<B::Fine, Smoothing<'a, B::Fine>, WilsonCorrection<'a, B>>,
}

/// The multigrid on the CPU.
pub type Multigrid<'a, F, T> = WilsonMultigrid<'a, WilsonOnCpu<'a, F, T>>;

/// The smoothing of the cycle: before the coarse correction from x = b or by a pre-smoother,
/// after it by the post-smoother.
struct Smoothing<'a, V> {
    operator: Box<dyn LinearFn<V> + 'a>,
    pre: Option<Box<dyn LinearFn<V> + 'a>>,
    post: Box<dyn LinearFn<V> + 'a>,
}

impl<V: Vector> Smooths<V> for Smoothing<'_, V> {
    fn pre(&mut self, x: &mut V, b: &V) {
        // Without pre-smoothing the cycle starts from x = b, not from zero: Bridge++'s guess
        // |v> = |w> (`APrecond_MG::mult_single`), which its iteration counts depend on.
        match self.pre.as_mut() {
            Some(pre) => pre.mult(x, b),
            None => x.copy_from(b),
        }
    }

    fn residual(&mut self, r: &mut V, x: &V, b: &V) {
        self.operator.mult(r, x);
        r.aypx(&vec![C64::new(-1.0, 0.0); b.num_vectors()], b);
    }

    fn post(&mut self, e: &mut V, r: &V) {
        self.post.mult(e, r);
    }
}

/// The coarse correction e = P A_c⁻¹ R r: R = P† (or the adjoint of a basis of its own), A_c⁻¹
/// an inexact solve of the coarse system, with the levels below it if there are any.
struct WilsonCorrection<'a, B: MultigridBackend<'a>> {
    basis: B::Basis,
    /// The basis of the restriction when it is not that of the prolongation.
    restriction: Option<B::Basis>,
    /// Shared by the coarse solver and the levels below, which apply it too.
    coarse: B::CoarseOp,
    /// Levels below the coarse lattice.
    levels: Option<CycleLevels<'a, B::Coarse>>,
    solver: Box<dyn Solver<B::Coarse> + 'a>,
    /// R r, its diagonal-preconditioned form, and the coarse solution.
    scratch: Option<[B::Coarse; 3]>,
}

impl<'a, B: MultigridBackend<'a>> Corrects<B::Fine> for WilsonCorrection<'a, B> {
    fn correct(&mut self, e: &mut B::Fine, r: &B::Fine) {
        let nv = r.num_vectors();
        if !self
            .scratch
            .as_ref()
            .is_some_and(|s| s[0].num_vectors() == nv)
        {
            self.scratch = Some(std::array::from_fn(|_| self.basis.coarse_field(nv)));
        }
        let Some([coarse_r, coarse_rhs, coarse_e]) = self.scratch.as_mut() else {
            unreachable!("the scratch fields were just created");
        };
        self.restriction
            .as_ref()
            .unwrap_or(&self.basis)
            .restrict(coarse_r, r);
        let rhs = if B::diagonal_preconditioned(&self.coarse) {
            B::apply_diagonal_inverse(&self.coarse, coarse_rhs, coarse_r);
            &*coarse_rhs
        } else {
            &*coarse_r
        };
        coarse_e.set_zero();
        // An inexact coarse solve is expected; whatever it reached is used.
        let (levels, blocks) = match self.levels.as_mut() {
            Some(l) => (Some(&*l.apply), l.blocks.as_deref_mut()),
            None => (None, None),
        };
        let solver = &mut self.solver;
        solve_shared(
            &self.coarse,
            blocks.map(|b| b as &mut dyn LinearFn<_>),
            levels,
            &Cell::new(0),
            &mut |a, m, x, b| {
                let _ = solver.solve(a, m, x, b);
            },
            coarse_e,
            rhs,
        );
        self.basis.prolong(e, coarse_e);
    }
}

impl<'a, B: MultigridBackend<'a>> WilsonMultigrid<'a, B> {
    /// The cycle from the choices in `params` for the backend's operator, from the initial test
    /// vectors `initial` improved `setup_iterations` times, or (without `smooth`) taken as they
    /// are with only the coarse operator made. `MultigridParams::bridge` gives the clover
    /// multigrid of Bridge++ (the design notes, §7.6.2): no pre-smoothing, SAP after the coarse
    /// correction, BiCGStab on the coarse lattice.
    pub fn from_parameters(
        backend: B,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
        initial: B::Fine,
        smooth: bool,
        setup_iterations: usize,
    ) -> Result<Self, MultigridError> {
        let agg = Aggregation::new(lat, params.block);
        let coarse_solver: Box<dyn Solver<B::Coarse> + 'a> = match params.coarse_solver {
            CoarseSolverChoice::BiCgStab => Box::new(BiCgStab::new(params.coarse_stop)),
            CoarseSolverChoice::Gmres => Box::new(Fgmres::new(params.coarse_stop)),
        };
        let parts = CycleParts {
            operator: backend.operator(),
            pre_smoother: params
                .pre_smoothing
                .then(|| backend.smoother(&agg, params.smoother)),
            post_smoother: backend.smoother(&agg, params.smoother),
            coarse_solver,
        };
        let hooks = SetupHooks {
            precondition_coarse_diagonal: params.precondition_coarse_diagonal,
            coarse_even_odd: params.coarse_even_odd,
            ..SetupHooks::default()
        };
        let mut mg = Self::assemble(
            backend,
            &agg,
            initial,
            smooth,
            setup_iterations,
            parts,
            hooks,
        )?;
        let levels: Vec<usize> = params.levels.into_iter().take_while(|s| *s > 0).collect();
        if !levels.is_empty() {
            mg.add_levels(&levels)?;
        }
        Ok(mg)
    }

    /// Multigrid from any members and setup hooks. `initial` holds the initial test
    /// vectors (random, or eigenvectors); without `smooth` they are the test vectors as they are.
    pub fn assemble(
        mut backend: B,
        agg: &Arc<Aggregation>,
        initial: B::Fine,
        smooth: bool,
        setup_iterations: usize,
        parts: CycleParts<'a, B::Fine, B::Coarse>,
        mut hooks: SetupHooks<'a, B::Fine>,
    ) -> Result<Self, MultigridError> {
        let CycleParts {
            operator,
            pre_smoother,
            mut post_smoother,
            coarse_solver,
        } = parts;
        let mut x = initial.zeros_like();
        if smooth {
            post_smoother.mult(&mut x, &initial);
        } else {
            x.copy_from(&initial);
        }
        let basis = basis_after_smoothing(&mut backend, agg, &mut hooks, &x)?;
        let restriction = match hooks.adjoint_smoother.as_mut() {
            Some(smoother) => {
                assert_eq!(
                    setup_iterations, 0,
                    "improving the test vectors of a separate restriction is not implemented"
                );
                let mut y = initial.zeros_like();
                smoother.mult(&mut y, &initial);
                Some(backend.basis(agg, &y)?)
            }
            None => None,
        };
        let even_odd = hooks.coarse_even_odd;
        let coarse = backend.coarse_operator(&basis, restriction.as_ref(), &mut hooks, even_odd)?;
        let coarse_tests = backend.coarse_tests(&basis, &x);
        let smoothing = Smoothing {
            operator,
            pre: pre_smoother,
            post: post_smoother,
        };
        let correction = WilsonCorrection {
            basis,
            restriction,
            coarse,
            levels: None,
            solver: coarse_solver,
            scratch: None,
        };
        let mut mg = Self {
            backend,
            agg: agg.clone(),
            coarse_tests,
            hooks,
            cycle: TwoGrid::new(smoothing, correction, &initial),
        };
        for _ in 0..setup_iterations {
            let v = mg.backend.test_vectors(mg.basis());
            mg.cycle.mult(&mut x, &v);
            let c = mg.cycle.correction_mut();
            c.basis = basis_after_smoothing(&mut mg.backend, &mg.agg, &mut mg.hooks, &x)?;
            c.coarse = mg
                .backend
                .coarse_operator(&c.basis, None, &mut mg.hooks, even_odd)?;
        }
        // The smoothed test vectors on the coarse lattice: P† x, which varies from block to
        // block with the near-null space (P† of the orthonormalised vectors would not).
        mg.coarse_tests = mg.backend.coarse_tests(mg.basis(), &x);
        Ok(mg)
    }

    /// Add levels below the coarse lattice to the coarse solves: a V-cycle on super-blocks of
    /// `super_blocks[ℓ]` coarse sites per direction on level ℓ + 2 (the last solved exactly), from
    /// the test vectors of the setup restricted to the coarse lattice, combined multiplicatively
    /// with the inverses of the diagonal blocks of the coarse operator (`multigrid::levels`).
    pub fn add_levels(&mut self, super_blocks: &[usize]) -> Result<usize, MultigridError> {
        let c = self.cycle.correction_mut();
        let levels = B::levels(&mut c.coarse, &self.coarse_tests, super_blocks)?;
        let applications = levels.applications;
        c.levels = Some(levels);
        Ok(applications)
    }

    /// The current test vectors, orthonormalised on every aggregate, as one fine field:
    /// vector i is P applied to the coarse vector with ones at components i and n + i.
    pub fn test_vectors(&self) -> B::Fine {
        self.backend.test_vectors(self.basis())
    }

    pub fn basis(&self) -> &B::Basis {
        &self.cycle.correction().basis
    }
}

/// The basis from smoothed vectors `x`, passed through `after_smoothing` first.
fn basis_after_smoothing<'a, B: MultigridBackend<'a>>(
    backend: &mut B,
    agg: &Arc<Aggregation>,
    hooks: &mut SetupHooks<'a, B::Fine>,
    x: &B::Fine,
) -> Result<B::Basis, MultigridError> {
    match hooks.after_smoothing.as_mut() {
        Some(after) => {
            let mut y = x.zeros_like();
            after.mult(&mut y, x);
            backend.basis(agg, &y)
        }
        None => backend.basis(agg, x),
    }
}

impl<'a, B: MultigridBackend<'a>> LinearFn<B::Fine> for WilsonMultigrid<'a, B> {
    fn mult(&mut self, x: &mut B::Fine, b: &B::Fine) {
        self.cycle.mult(x, b);
    }
}

/// The multigrid on the CPU: the operator `op` of the usual layout, P (`Basis`) and the coarse
/// operator (`CoarseOp`) on the host.
pub struct WilsonOnCpu<'a, F: HopTerms<Fine<T>>, T: Storage> {
    op: &'a F,
    work: F::Work,
}

impl<'a, F: HopTerms<Fine<T>>, T: Storage> WilsonOnCpu<'a, F, T> {
    pub fn new(op: &'a F) -> Self {
        Self {
            op,
            work: op.work(),
        }
    }
}

impl<'a, F, T> MultigridBackend<'a> for WilsonOnCpu<'a, F, T>
where
    F: HopTerms<Fine<T>> + 'a,
    F::Work: 'a,
    T: Storage + 'a,
{
    type Fine = Fine<T>;
    type Coarse = Field<T, Coarse>;
    type Basis = Basis<T>;
    type CoarseOp = RefCell<CoarseOp<T>>;
    type CoarseTests = Vec<Field<f64, Coarse>>;

    fn operator(&self) -> Box<dyn LinearFn<Fine<T>> + 'a> {
        Box::new(Bound::new(self.op, Variant::D))
    }

    fn smoother(
        &self,
        agg: &Arc<Aggregation>,
        choice: SmootherChoice,
    ) -> Box<dyn LinearFn<Fine<T>> + 'a> {
        match choice {
            SmootherChoice::Sap {
                cycles,
                block_steps,
            } => Box::new(Sap::new(self.op, agg, cycles, block_steps)),
            SmootherChoice::Gmres {
                iterations,
                tolerance_sqr,
            } => Box::new(Solve::new(
                Fgmres::new(Stop::simple(tolerance_sqr, iterations)),
                Bound::new(self.op, Variant::D),
            )),
        }
    }

    fn basis(&mut self, agg: &Arc<Aggregation>, x: &Fine<T>) -> Result<Basis<T>, MultigridError> {
        basis_from(self.op, &mut self.work, agg, x)
    }

    fn coarse_operator(
        &mut self,
        basis: &Basis<T>,
        restriction: Option<&Basis<T>>,
        hooks: &mut SetupHooks<'a, Fine<T>>,
        even_odd: bool,
    ) -> Result<RefCell<CoarseOp<T>>, MultigridError> {
        let mut coarse = build_coarse(self.op, &mut self.work, basis, restriction, hooks)?;
        if even_odd {
            coarse.prepare_even_odd()?;
        }
        Ok(RefCell::new(coarse))
    }

    fn test_vectors(&self, basis: &Basis<T>) -> Fine<T> {
        let n = basis.num_test_vectors();
        let mut c = basis.coarse_field(n);
        let one = T::store(T::Compute::ONE);
        let site_len = c.site_len();
        for site in 0..c.num_sites() {
            let s = c.site_mut(0, site);
            for i in 0..n {
                let v = &mut s[i * site_len / n..(i + 1) * site_len / n];
                v[2 * i] = one;
                v[2 * (n + i)] = one;
            }
        }
        let mut fine = basis.fine_field(n);
        basis.prolong(&mut fine, &c);
        fine
    }

    fn coarse_tests(&self, basis: &Basis<T>, x: &Fine<T>) -> Vec<Field<f64, Coarse>> {
        let mut restricted = basis.coarse_field(x.num_vectors());
        basis.restrict(&mut restricted, x);
        (0..x.num_vectors())
            .map(|i| {
                let one = restricted.select(&[i]);
                let mut v =
                    Field::<f64, Coarse>::with_shape(one.lattice(), one.internal_dof(), 1, 1, 1);
                v.convert_from(&one);
                v
            })
            .collect()
    }

    fn diagonal_preconditioned(coarse: &RefCell<CoarseOp<T>>) -> bool {
        coarse.borrow().is_diagonal_preconditioned()
    }

    fn apply_diagonal_inverse(
        coarse: &RefCell<CoarseOp<T>>,
        out: &mut Field<T, Coarse>,
        inp: &Field<T, Coarse>,
    ) {
        coarse.borrow().apply_diagonal_inverse(out, inp);
    }

    fn levels(
        coarse: &mut RefCell<CoarseOp<T>>,
        tests: &Vec<Field<f64, Coarse>>,
        super_blocks: &[usize],
    ) -> Result<CycleLevels<'a, Field<T, Coarse>>, MultigridError> {
        let coarse = coarse.get_mut();
        let blocks = if coarse.is_diagonal_preconditioned() {
            None
        } else {
            Some(BlockMap::inverse_diagonal(coarse)?)
        };
        let levels = SignedMultilevel::build(coarse, tests, super_blocks)?;
        let applications = levels.setup_applications;
        Ok(CycleLevels {
            apply: Box::new(move |out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>| {
                levels.apply(out, inp)
            }),
            blocks: blocks.map(|b| Box::new(b) as Box<dyn LinearFn<Field<T, Coarse>> + 'a>),
            applications,
        })
    }
}

impl<'a, F, T> Multigrid<'a, F, T>
where
    F: HopTerms<Fine<T>> + 'a,
    F::Work: 'a,
    T: Storage + 'a,
{
    /// A cycle on D from the choices in `params`, with random initial test vectors.
    /// `MultigridParams::bridge` gives the clover multigrid of Bridge++ (the design notes, §7.6.2):
    /// no pre-smoothing, SAP after the coarse correction, BiCGStab on the coarse lattice.
    pub fn new(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
    ) -> Result<Self, MultigridError> {
        let initial = random_vectors::<T>(lat, 1, params.num_test_vectors, params.seed);
        Self::from_vectors(op, lat, params, initial, params.setup_iterations)
    }

    /// `new` from given initial test vectors, improved `setup_iterations` times: those of an
    /// earlier setup (`test_vectors`) for links nearby make a setup that costs one smoothing
    /// of the vectors and the coarse operator.
    pub fn from_vectors(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
        initial: Fine<T>,
        setup_iterations: usize,
    ) -> Result<Self, MultigridError> {
        Self::from_parameters(
            WilsonOnCpu::new(op),
            lat,
            params,
            initial,
            true,
            setup_iterations,
        )
    }

    /// The cycle for `op` from the test vectors of a cycle for links nearby (`test_vectors`),
    /// taken as they are: only the coarse operator is made again.
    pub fn rebuilt(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
        vectors: Fine<T>,
    ) -> Result<Self, MultigridError> {
        Self::from_parameters(WilsonOnCpu::new(op), lat, params, vectors, false, 0)
    }

    /// Multigrid from any members and setup hooks. `initial` holds the initial test
    /// vectors (random, or eigenvectors); `op` provides the parts the coarse operator is
    /// built from and Γ5.
    pub fn with_parts(
        op: &'a F,
        agg: &Arc<Aggregation>,
        initial: Fine<T>,
        setup_iterations: usize,
        parts: CycleParts<'a, Fine<T>, Field<T, Coarse>>,
        hooks: SetupHooks<'a, Fine<T>>,
    ) -> Result<Self, MultigridError> {
        Self::assemble(
            WilsonOnCpu::new(op),
            agg,
            initial,
            true,
            setup_iterations,
            parts,
            hooks,
        )
    }

    pub fn coarse_operator(&self) -> Ref<'_, CoarseOp<T>> {
        self.cycle.correction().coarse.borrow()
    }
}

fn build_coarse<F: HopTerms<Fine<T>>, T: Storage>(
    op: &F,
    work: &mut F::Work,
    basis: &Basis<T>,
    restriction: Option<&Basis<T>>,
    hooks: &mut SetupHooks<'_, Fine<T>>,
) -> Result<CoarseOp<T>, MultigridError> {
    if let Some(a) = hooks.galerkin_of.as_mut() {
        let restriction = restriction.unwrap_or(basis);
        let precondition = hooks.precondition_coarse_diagonal;
        return CoarseOp::probe(a.as_mut(), basis, restriction, precondition);
    }
    assert!(
        restriction.is_none(),
        "a separate restriction needs the coarse operator made by probing"
    );
    let right = hooks
        .coarse_right
        .as_mut()
        .map(|r| r.as_mut() as &mut dyn LinearFn<Fine<T>>);
    CoarseOp::build_with(op, work, basis, right, hooks.precondition_coarse_diagonal)
}

/// Orthonormalise the chiral halves of the vectors of `x` on the aggregates, with the
/// chirality given by the operator's Γ5.
fn basis_from<F: HopTerms<Field<T, Spinor>>, T: Storage>(
    op: &F,
    work: &mut F::Work,
    agg: &Arc<Aggregation>,
    x: &Field<T, Spinor>,
) -> Result<Basis<T>, MultigridError> {
    let mut g5x = x.zeros_like();
    op.mult_gamma5(work, &mut g5x, x);
    let mut x64 = Field::<f64, Spinor>::with_shape(
        agg.fine_lattice(),
        x.internal_dof(),
        x.ls(),
        x.num_vectors(),
        1,
    );
    let mut g64 = x64.zeros_like();
    x64.convert_from(x);
    g64.convert_from(&g5x);
    Basis::from_vectors(agg, &x64, &g64)
}

/// `n` vectors of `ls` spinors per site with every real uniform in (−1, 1), from SplitMix64.
pub fn random_vectors<T: Storage>(
    lat: &Arc<Lattice>,
    ls: usize,
    n: usize,
    seed: u64,
) -> Field<T, Spinor> {
    let mut v = Field::<T, Spinor>::with_shape(lat, 24, ls, n, 1);
    let mut state = seed;
    for r in v.as_mut_slice() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
        *r = T::store(Number::from_f64(u));
    }
    v
}
