use super::{Aggregation, Basis, CoarseOp, MultigridError, Sap};
use crate::fermion::{HopTerms, Variant};
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

/// Two-level multigrid as a preconditioner: an approximate inverse of D, to be passed as
/// `m` to a flexible outer solver.
///
/// One application, for a right-hand side b (the cycle of the Bridge++ class
/// `APrecond_MG`):
///
/// 1. start from x = b;
/// 2. coarse-grid correction: x += P D_c^{-1} P† (b − D x), with D_c^{-1} an inexact
///    BiCGStab solve on the coarse lattice;
/// 3. smoothing: x += S (b − D x), with S the SAP smoother.
///
/// The coarse solve stops at a tolerance, so the map is not exactly linear; use it with
/// FGMRES or another flexible method.
///
/// The test vectors are found when the preconditioner is built: random vectors are
/// smoothed by S, then improved by `setup_iterations` rounds of x_i ← M x_i with the
/// cycle M built from the current vectors. P is rebuilt after each round.
///
/// Works on one process only, like the aggregation it is built on.
pub struct Multigrid<'a, F: HopTerms<Field<T, Spinor>>, T: Storage> {
    op: &'a F,
    work: F::Work,
    basis: Basis<T>,
    coarse: CoarseOp<T>,
    coarse_solver: BiCgStab<Field<T, Coarse>>,
    smoother: Sap<'a, F, T>,
    scratch: Option<Scratch<T>>,
}

struct Scratch<T: Storage> {
    r: Field<T, Spinor>,
    e: Field<T, Spinor>,
    coarse_r: Field<T, Coarse>,
    coarse_e: Field<T, Coarse>,
}

impl<'a, F: HopTerms<Field<T, Spinor>>, T: Storage> Multigrid<'a, F, T> {
    /// Build the aggregation, find the test vectors and build the coarse operator. `lat`
    /// is the lattice `op` acts on.
    pub fn new(
        op: &'a F,
        lat: &Arc<Lattice>,
        params: &MultigridParams,
    ) -> Result<Self, MultigridError> {
        let agg = Aggregation::new(lat, params.block);
        let n = params.num_test_vectors;
        let mut smoother = Sap::new(
            op,
            &agg,
            params.smoother_cycles,
            params.smoother_block_steps,
        );
        let mut work = op.work();

        let v = random_vectors::<T>(lat, n, params.seed);
        let mut x = v.zeros_like();
        smoother.mult(&mut x, &v);
        let basis = basis_from(op, &mut work, &agg, &x)?;
        let coarse = CoarseOp::build(op, &mut work, &basis);

        let mut mg = Self {
            op,
            work,
            basis,
            coarse,
            coarse_solver: BiCgStab::new(params.coarse_stop),
            smoother,
            scratch: None,
        };
        for _ in 0..params.setup_iterations {
            let v = mg.test_vectors();
            mg.mult(&mut x, &v);
            mg.basis = basis_from(mg.op, &mut mg.work, &agg, &x)?;
            mg.coarse = CoarseOp::build(mg.op, &mut mg.work, &mg.basis);
        }
        Ok(mg)
    }

    pub fn basis(&self) -> &Basis<T> {
        &self.basis
    }

    pub fn coarse_operator(&self) -> &CoarseOp<T> {
        &self.coarse
    }

    /// The current test vectors, orthonormalised on every aggregate, as one fine field:
    /// vector i is P applied to the coarse vector with ones at components i and n + i.
    pub fn test_vectors(&self) -> Field<T, Spinor> {
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
        let mut fine = Field::spinor(self.basis.aggregation().fine_lattice(), n);
        self.basis.prolong(&mut fine, &c);
        fine
    }
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

impl<F: HopTerms<Field<T, Spinor>>, T: Storage> LinearFn<Field<T, Spinor>> for Multigrid<'_, F, T> {
    fn mult(&mut self, x: &mut Field<T, Spinor>, b: &Field<T, Spinor>) {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.scratch.as_ref().is_some_and(|s| s.r.same_shape(b)) {
            self.scratch = Some(Scratch {
                r: b.zeros_like(),
                e: b.zeros_like(),
                coarse_r: self.basis.coarse_field(nv),
                coarse_e: self.basis.coarse_field(nv),
            });
        }
        let Some(Scratch {
            r,
            e,
            coarse_r,
            coarse_e,
        }) = self.scratch.as_mut()
        else {
            unreachable!("the scratch fields were just created");
        };
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let residual =
            |op: &F, work: &mut F::Work, r: &mut Field<T, Spinor>, x: &Field<T, Spinor>| {
                op.mult(work, r, x, Variant::D);
                r.aypx(&minus_one, b);
            };

        x.copy_from(b);
        residual(self.op, &mut self.work, r, x);
        self.basis.restrict(coarse_r, r);
        coarse_e.set_zero();
        // An inexact coarse solve is expected; whatever it reached is used.
        let _ = self
            .coarse_solver
            .solve(&mut self.coarse, None, coarse_e, coarse_r);
        self.basis.prolong(e, coarse_e);
        x.axpy(&one, e);

        residual(self.op, &mut self.work, r, x);
        self.smoother.mult(e, r);
        x.axpy(&one, e);
    }
}
