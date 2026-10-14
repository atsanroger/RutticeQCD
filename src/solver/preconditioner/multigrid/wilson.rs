use super::levels::BlockMap;
use super::solve_shared;
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

/// The members of the cycle (the design notes, §7.6.3); each can be exchanged.
pub struct CycleParts<'a, T: Storage> {
    /// The operator whose inverse the cycle approximates: D for clover, A for the
    /// domain-wall target operator.
    pub operator: Box<dyn LinearFn<Fine<T>> + 'a>,
    /// Smoothing before the coarse correction, from a zero start. Without it the cycle
    /// starts from x = b, as Bridge++ `APrecond_MG` does.
    pub pre_smoother: Option<Box<dyn LinearFn<Fine<T>> + 'a>>,
    /// Smoothing after the coarse correction. It also smooths the test vectors in the setup.
    pub post_smoother: Box<dyn LinearFn<Fine<T>> + 'a>,
    /// Solves the coarse system; not reaching its tolerance is not an error.
    pub coarse_solver: Box<dyn Solver<Field<T, Coarse>> + 'a>,
}

/// How the setup builds the test vectors and the coarse operator, beyond the defaults.
pub struct SetupHooks<'a, T: Storage> {
    /// Applied to every test vector after it was smoothed (domain-wall: D_PV).
    pub after_smoothing: Option<Box<dyn LinearFn<Fine<T>> + 'a>>,
    /// A site-local factor R in the coarse operator P† D R P (domain-wall: C_PV^{-1}).
    pub coarse_right: Option<Box<dyn LinearFn<Fine<T>> + 'a>>,
    /// Precondition the coarse operator by its diagonal blocks.
    pub precondition_coarse_diagonal: bool,
    /// Prepare the coarse operator for solves on the even coarse sites
    /// (`MultigridParams::coarse_even_odd`).
    pub coarse_even_odd: bool,
    /// Build the coarse operator as the Galerkin projection `P† A P` of this map, cut down
    /// to nearest neighbours (`CoarseOp::probe`), instead of from the parts of the fine
    /// operator; `coarse_right` is then not used.
    pub galerkin_of: Option<Box<dyn LinearFn<Fine<T>> + 'a>>,
    /// Makes test vectors for a restriction of its own from the initial vectors: a smoother for the
    /// adjoint of the operator of the cycle.
    pub adjoint_smoother: Option<Box<dyn LinearFn<Fine<T>> + 'a>>,
}

impl<T: Storage> Default for SetupHooks<'_, T> {
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

/// Two-level multigrid as a preconditioner: an approximate inverse of the cycle's operator
/// A, to be passed as `m` to a flexible outer solver (FGMRES, or BiCGStab, which is the
/// flexible form with a preconditioner).
pub struct Multigrid<'a, F: HopTerms<Fine<T>>, T: Storage> {
    op: &'a F,
    work: F::Work,
    agg: Arc<Aggregation>,
    basis: Basis<T>,
    /// The basis of the restriction when it is not that of the prolongation.
    restriction: Option<Basis<T>>,
    /// Shared by the coarse solver and the levels below, which apply it too.
    coarse: RefCell<CoarseOp<T>>,
    /// The test vectors of the last setup round restricted to the coarse lattice, one per
    /// field: test vectors for levels below it.
    coarse_tests: Vec<Field<f64, Coarse>>,
    /// Levels below the coarse lattice, and the inverses of the diagonal blocks of the
    /// coarse operator they are combined with.
    levels: Option<(SignedMultilevel, Option<BlockMap>)>,
    parts: CycleParts<'a, T>,
    hooks: SetupHooks<'a, T>,
    scratch: Option<Scratch<T>>,
}

struct Scratch<T: Storage> {
    r: Fine<T>,
    e: Fine<T>,
    coarse_r: Field<T, Coarse>,
    coarse_rhs: Field<T, Coarse>,
    coarse_e: Field<T, Coarse>,
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
        Self::from_parameters(op, lat, params, initial, true, setup_iterations)
    }

    /// The cycle for `op` from the test vectors of a cycle for links nearby (`test_vectors`),
    /// taken as they are: only the coarse operator is made again.
    pub fn rebuilt(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
        vectors: Fine<T>,
    ) -> Result<Self, MultigridError> {
        Self::from_parameters(op, lat, params, vectors, false, 0)
    }

    fn from_parameters(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
        initial: Fine<T>,
        smooth: bool,
        setup_iterations: usize,
    ) -> Result<Self, MultigridError> {
        let agg = Aggregation::new(lat, params.block);
        let smoother = || -> Box<dyn LinearFn<Fine<T>> + 'a> {
            match params.smoother {
                SmootherChoice::Sap {
                    cycles,
                    block_steps,
                } => Box::new(Sap::new(op, &agg, cycles, block_steps)),
                SmootherChoice::Gmres {
                    iterations,
                    tolerance_sqr,
                } => Box::new(Solve::new(
                    Fgmres::new(Stop::simple(tolerance_sqr, iterations)),
                    Bound::new(op, Variant::D),
                )),
            }
        };
        let coarse_solver: Box<dyn Solver<Field<T, Coarse>> + 'a> = match params.coarse_solver {
            CoarseSolverChoice::BiCgStab => Box::new(BiCgStab::new(params.coarse_stop)),
            CoarseSolverChoice::Gmres => Box::new(Fgmres::new(params.coarse_stop)),
        };
        let parts = CycleParts {
            operator: Box::new(Bound::new(op, Variant::D)),
            pre_smoother: params.pre_smoothing.then(smoother),
            post_smoother: smoother(),
            coarse_solver,
        };
        let hooks = SetupHooks {
            precondition_coarse_diagonal: params.precondition_coarse_diagonal,
            coarse_even_odd: params.coarse_even_odd,
            ..SetupHooks::default()
        };
        let mut mg = Self::assemble(op, &agg, initial, smooth, setup_iterations, parts, hooks)?;
        let levels: Vec<usize> = params.levels.into_iter().take_while(|s| *s > 0).collect();
        if !levels.is_empty() {
            mg.add_levels(&levels)?;
        }
        Ok(mg)
    }

    /// Multigrid from any members and setup hooks. `initial` holds the initial test
    /// vectors (random, or eigenvectors); `op` provides the parts the coarse operator is
    /// built from and Γ5.
    pub fn with_parts(
        op: &'a F,
        agg: &Arc<Aggregation>,
        initial: Fine<T>,
        setup_iterations: usize,
        parts: CycleParts<'a, T>,
        hooks: SetupHooks<'a, T>,
    ) -> Result<Self, MultigridError> {
        Self::assemble(op, agg, initial, true, setup_iterations, parts, hooks)
    }

    /// `with_parts`; without `smooth` the initial vectors are the test vectors as they are.
    fn assemble(
        op: &'a F,
        agg: &Arc<Aggregation>,
        initial: Fine<T>,
        smooth: bool,
        setup_iterations: usize,
        mut parts: CycleParts<'a, T>,
        mut hooks: SetupHooks<'a, T>,
    ) -> Result<Self, MultigridError> {
        let mut work = op.work();
        let mut x = initial.zeros_like();
        if smooth {
            parts.post_smoother.mult(&mut x, &initial);
        } else {
            x.copy_from(&initial);
        }
        let basis = basis_after_smoothing(op, &mut work, agg, &mut hooks, &x)?;
        let restriction = match hooks.adjoint_smoother.as_mut() {
            Some(smoother) => {
                assert_eq!(
                    setup_iterations, 0,
                    "improving the test vectors of a separate restriction is not implemented"
                );
                let mut y = initial.zeros_like();
                smoother.mult(&mut y, &initial);
                Some(basis_from(op, &mut work, agg, &y)?)
            }
            None => None,
        };
        let mut coarse = build_coarse(op, &mut work, &basis, restriction.as_ref(), &mut hooks)?;
        if hooks.coarse_even_odd {
            coarse.prepare_even_odd()?;
        }
        let mut mg = Self {
            op,
            work,
            agg: agg.clone(),
            basis,
            restriction,
            coarse: RefCell::new(coarse),
            coarse_tests: Vec::new(),
            levels: None,
            parts,
            hooks,
            scratch: None,
        };
        for _ in 0..setup_iterations {
            let v = mg.test_vectors();
            mg.mult(&mut x, &v);
            mg.basis = basis_after_smoothing(mg.op, &mut mg.work, &mg.agg, &mut mg.hooks, &x)?;
            let coarse = build_coarse(mg.op, &mut mg.work, &mg.basis, None, &mut mg.hooks)?;
            mg.coarse = RefCell::new(coarse);
        }
        // The smoothed test vectors on the coarse lattice: P† x, which varies from block to
        // block with the near-null space (P† of the orthonormalised vectors would not).
        let mut restricted = mg.basis.coarse_field(x.num_vectors());
        mg.basis.restrict(&mut restricted, &x);
        mg.coarse_tests = (0..x.num_vectors())
            .map(|i| {
                let one = restricted.select(&[i]);
                let mut v =
                    Field::<f64, Coarse>::with_shape(one.lattice(), one.internal_dof(), 1, 1, 1);
                v.convert_from(&one);
                v
            })
            .collect();
        Ok(mg)
    }

    /// Add levels below the coarse lattice to the coarse solves: a V-cycle on super-blocks of
    /// `super_blocks[ℓ]` coarse sites per direction on level ℓ + 2 (the last solved exactly), from
    /// the test vectors of the setup restricted to the coarse lattice, combined multiplicatively
    /// with the inverses of the diagonal blocks of the coarse operator (`multigrid::levels`).
    pub fn add_levels(&mut self, super_blocks: &[usize]) -> Result<usize, MultigridError> {
        let coarse = self.coarse.get_mut();
        let blocks = if coarse.is_diagonal_preconditioned() {
            None
        } else {
            Some(BlockMap::inverse_diagonal(coarse)?)
        };
        let levels = SignedMultilevel::build(coarse, &self.coarse_tests, super_blocks)?;
        let applications = levels.setup_applications;
        self.levels = Some((levels, blocks));
        Ok(applications)
    }
}

impl<F: HopTerms<Fine<T>>, T: Storage> Multigrid<'_, F, T> {
    pub fn basis(&self) -> &Basis<T> {
        &self.basis
    }

    pub fn coarse_operator(&self) -> Ref<'_, CoarseOp<T>> {
        self.coarse.borrow()
    }

    /// The current test vectors, orthonormalised on every aggregate, as one fine field:
    /// vector i is P applied to the coarse vector with ones at components i and n + i.
    pub fn test_vectors(&self) -> Fine<T> {
        let n = self.basis.num_test_vectors();
        let mut c = self.basis.coarse_field(n);
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
        let mut fine = self.basis.fine_field(n);
        self.basis.prolong(&mut fine, &c);
        fine
    }
}

/// The basis from smoothed vectors `x`, passed through `after_smoothing` first.
fn basis_after_smoothing<F: HopTerms<Fine<T>>, T: Storage>(
    op: &F,
    work: &mut F::Work,
    agg: &Arc<Aggregation>,
    hooks: &mut SetupHooks<'_, T>,
    x: &Fine<T>,
) -> Result<Basis<T>, MultigridError> {
    match hooks.after_smoothing.as_mut() {
        Some(after) => {
            let mut y = x.zeros_like();
            after.mult(&mut y, x);
            basis_from(op, work, agg, &y)
        }
        None => basis_from(op, work, agg, x),
    }
}

fn build_coarse<F: HopTerms<Fine<T>>, T: Storage>(
    op: &F,
    work: &mut F::Work,
    basis: &Basis<T>,
    restriction: Option<&Basis<T>>,
    hooks: &mut SetupHooks<'_, T>,
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
pub(super) fn random_vectors<T: Storage>(
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

impl<F: HopTerms<Fine<T>>, T: Storage> LinearFn<Fine<T>> for Multigrid<'_, F, T> {
    fn mult(&mut self, x: &mut Fine<T>, b: &Fine<T>) {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.scratch.as_ref().is_some_and(|s| s.r.same_shape(b)) {
            self.scratch = Some(Scratch {
                r: b.zeros_like(),
                e: b.zeros_like(),
                coarse_r: self.basis.coarse_field(nv),
                coarse_rhs: self.basis.coarse_field(nv),
                coarse_e: self.basis.coarse_field(nv),
            });
        }
        let Some(Scratch {
            r,
            e,
            coarse_r,
            coarse_rhs,
            coarse_e,
        }) = self.scratch.as_mut()
        else {
            unreachable!("the scratch fields were just created");
        };
        let parts = &mut self.parts;
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let mut residual = |r: &mut Fine<T>, x: &Fine<T>| {
            parts.operator.mult(r, x);
            r.aypx(&minus_one, b);
        };

        match parts.pre_smoother.as_mut() {
            Some(pre) => pre.mult(x, b),
            None => x.copy_from(b),
        }
        residual(r, x);
        self.restriction
            .as_ref()
            .unwrap_or(&self.basis)
            .restrict(coarse_r, r);
        let rhs = if self.coarse.borrow().is_diagonal_preconditioned() {
            self.coarse
                .borrow()
                .apply_diagonal_inverse(coarse_rhs, coarse_r);
            &*coarse_rhs
        } else {
            &*coarse_r
        };
        coarse_e.set_zero();
        // An inexact coarse solve is expected; whatever it reached is used.
        let (levels, blocks) = match self.levels.as_mut() {
            Some((levels, blocks)) => (Some(&*levels), blocks.as_mut()),
            None => (None, None),
        };
        let coarse_solver = &mut parts.coarse_solver;
        let levels = levels.map(|l| move |out: &mut _, inp: &_| l.apply(out, inp));
        solve_shared(
            &self.coarse,
            blocks.map(|b| b as &mut dyn LinearFn<_>),
            levels.as_ref().map(|l| l as &super::Levels<'_, _>),
            &Cell::new(0),
            &mut |a, m, x, b| {
                let _ = coarse_solver.solve(a, m, x, b);
            },
            coarse_e,
            rhs,
        );
        self.basis.prolong(e, coarse_e);
        x.axpy(&one, e);

        residual(r, x);
        parts.post_smoother.mult(e, r);
        x.axpy(&one, e);
    }
}
