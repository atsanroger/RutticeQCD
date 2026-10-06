use super::{Aggregation, Basis, CoarseOp, MultigridError, Sap};
use crate::fermion::{Bound, HopTerms, Variant};
use crate::kernel::Real;
use crate::solver::{BiCgStab, Solver, Stop};
use crate::{Coarse, ConvertFrom, Field, Lattice, LinearFn, Spinor, Storage, Vector, C64, ND};
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
    /// The coarse-grid solve (BiCGStab). Not reaching the tolerance is not an error.
    pub coarse_stop: Stop,
    /// SAP cycles of the smoother.
    pub smoother_cycles: usize,
    /// Minimal-residual steps per block solve in the smoother.
    pub smoother_block_steps: usize,
    /// Seed of the random initial test vectors.
    pub seed: u64,
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
            coarse_stop: Stop::simple(2e-3, 200),
            smoother_cycles: 4,
            smoother_block_steps: 6,
            seed: 1,
        }
    }
}

type Fine<T> = Field<T, Spinor>;

/// The members of the cycle (DESIGN.md §7.6.3); each can be exchanged.
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
}

impl<T: Storage> Default for SetupHooks<'_, T> {
    fn default() -> Self {
        Self {
            after_smoothing: None,
            coarse_right: None,
            precondition_coarse_diagonal: false,
        }
    }
}

/// Two-level multigrid as a preconditioner: an approximate inverse of the cycle's operator
/// A, to be passed as `m` to a flexible outer solver (FGMRES, or BiCGStab, which is the
/// flexible form with a preconditioner).
///
/// One application, for a right-hand side b:
///
/// 1. x = S_pre(b) with a pre-smoother, otherwise x = b;
/// 2. coarse-grid correction: x += P A_c^{-1} P† (b − A x), with A_c^{-1} an inexact solve
///    on the coarse lattice (after multiplying by the inverse diagonal blocks when the
///    coarse operator is preconditioned by them);
/// 3. x += S_post(b − A x).
///
/// Without a pre-smoother this is the cycle of Bridge++ `APrecond_MG`; with one it is the
/// cycle of the CUDA domain-wall `APrecond_MG_dw`.
///
/// The test vectors are found when the preconditioner is built: the initial vectors are
/// smoothed by S_post (and passed through `after_smoothing`), then improved by
/// `setup_iterations` rounds of v_i ← M v_i with the cycle M built from the current
/// vectors. P and the coarse operator are rebuilt after each round.
///
/// Works on one process only, like the aggregation it is built on.
pub struct Multigrid<'a, F: HopTerms<Fine<T>>, T: Storage> {
    op: &'a F,
    work: F::Work,
    agg: Arc<Aggregation>,
    basis: Basis<T>,
    coarse: CoarseOp<T>,
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
    /// The clover multigrid of Bridge++ (DESIGN.md §7.6.2): cycle on D, no pre-smoothing,
    /// SAP after the coarse correction, BiCGStab on the coarse lattice, random initial
    /// test vectors.
    pub fn new(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
    ) -> Result<Self, MultigridError> {
        let agg = Aggregation::new(lat, params.block);
        let parts = CycleParts {
            operator: Box::new(Bound::new(op, Variant::D)),
            pre_smoother: None,
            post_smoother: Box::new(Sap::new(
                op,
                &agg,
                params.smoother_cycles,
                params.smoother_block_steps,
            )),
            coarse_solver: Box::new(BiCgStab::new(params.coarse_stop)),
        };
        let initial = random_vectors::<T>(lat, params.num_test_vectors, params.seed);
        Self::with_parts(
            op,
            &agg,
            initial,
            params.setup_iterations,
            parts,
            SetupHooks::default(),
        )
    }

    /// Multigrid from any members and setup hooks. `initial` holds the initial test
    /// vectors (random, or eigenvectors); `op` provides the parts the coarse operator is
    /// built from and Γ5.
    pub fn with_parts(
        op: &'a F,
        agg: &Arc<Aggregation>,
        initial: Fine<T>,
        setup_iterations: usize,
        mut parts: CycleParts<'a, T>,
        mut hooks: SetupHooks<'a, T>,
    ) -> Result<Self, MultigridError> {
        let mut work = op.work();
        let mut x = initial.zeros_like();
        parts.post_smoother.mult(&mut x, &initial);
        let basis = basis_after_smoothing(op, &mut work, agg, &mut hooks, &x)?;
        let coarse = build_coarse(op, &mut work, &basis, &mut hooks)?;
        let mut mg = Self {
            op,
            work,
            agg: agg.clone(),
            basis,
            coarse,
            parts,
            hooks,
            scratch: None,
        };
        for _ in 0..setup_iterations {
            let v = mg.test_vectors();
            mg.mult(&mut x, &v);
            mg.basis = basis_after_smoothing(mg.op, &mut mg.work, &mg.agg, &mut mg.hooks, &x)?;
            mg.coarse = build_coarse(mg.op, &mut mg.work, &mg.basis, &mut mg.hooks)?;
        }
        Ok(mg)
    }
}

impl<F: HopTerms<Fine<T>>, T: Storage> Multigrid<'_, F, T> {
    pub fn basis(&self) -> &Basis<T> {
        &self.basis
    }

    pub fn coarse_operator(&self) -> &CoarseOp<T> {
        &self.coarse
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
    hooks: &mut SetupHooks<'_, T>,
) -> Result<CoarseOp<T>, MultigridError> {
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

/// `n` vectors with every real uniform in (−1, 1), from SplitMix64.
fn random_vectors<T: Storage>(lat: &Arc<Lattice>, n: usize, seed: u64) -> Field<T, Spinor> {
    let mut v = Field::<T, Spinor>::spinor(lat, n);
    let mut state = seed;
    for r in v.as_mut_slice() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
        *r = T::store(Real::from_f64(u));
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
        self.basis.restrict(coarse_r, r);
        let rhs = if self.coarse.is_diagonal_preconditioned() {
            self.coarse.apply_diagonal_inverse(coarse_rhs, coarse_r);
            &*coarse_rhs
        } else {
            &*coarse_r
        };
        coarse_e.set_zero();
        // An inexact coarse solve is expected; whatever it reached is used.
        let _ = parts
            .coarse_solver
            .solve(&mut self.coarse, None, coarse_e, rhs);
        self.basis.prolong(e, coarse_e);
        x.axpy(&one, e);

        residual(r, x);
        parts.post_smoother.mult(e, r);
        x.axpy(&one, e);
    }
}
