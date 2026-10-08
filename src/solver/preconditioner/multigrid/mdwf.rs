//! The two-level multigrid for Möbius domain-wall fermions: the cycle of `Multigrid` with
//! the members of the Pauli-Villars preconditioned method (Kanamori, Chen, Matsufuru,
//! PoS(LATTICE2024)414; the working note `pvprec_amg` of Chen, Kanamori, Matsufuru).
//!
//! - The cycle approximates the inverse of `A = (D_PV C_PV^-1)† D C^-1`
//!   (`PvPreconditioned`), so it is the preconditioner of a flexible solver on A.
//! - Smoother, before and after the coarse correction: GMRES(m) on A from a zero start, or
//!   a polynomial of A fitted once (`MdwfSmoother`).
//! - Aggregates: blocks in space-time that keep the whole fifth dimension, split by the
//!   chirality of Γ5 = γ5 R.
//! - Test vectors: random vectors, smoothed, then multiplied by D_PV.
//! - Coarse operator: `P† D C_PV^-1 P`, built from the parts of D with the exact
//!   site-local `C_PV^-1` as a right factor, so that it reaches the eight neighbouring
//!   aggregates only; preconditioned by the inverse of its diagonal blocks. This is the
//!   reference's choice for its A = (D_PV C_PV^-1)† C^-1 D, kept for now with the paper's A:
//!   on a 12^4 configuration neither this nor the Galerkin projection of A, for either A,
//!   reduces a smooth error (DESIGN.md §7.6.5 (d)).
//! - Coarse solver: GMRES(m), restarted a few times, to a loose tolerance.

use super::cycle::random_vectors;
use super::{Aggregation, CycleParts, Multigrid, MultigridError, SetupHooks};
use crate::fermion::{Mdwf, MdwfAt, PvPreconditioned, SiteInverse, Variant};
use crate::kernel::GammaRepre;
use crate::solver::{Fgmres, Polynomial, Solve, Stop};
use crate::{Adjoint, Field, Lattice, LinearFn, Spinor, Storage, ND};
use std::sync::Arc;

/// Parameters of the domain-wall multigrid.
#[derive(Clone, Copy, Debug)]
pub struct MdwfMultigridParams {
    /// Size of the aggregates in space-time.
    pub block: [usize; ND],
    /// Test vectors; the coarse lattice carries twice as many components per site.
    pub num_test_vectors: usize,
    /// Rounds of improving the test vectors with the cycle itself after the first setup.
    pub setup_iterations: usize,
    pub smoother: MdwfSmoother,
    /// The coarse solver: GMRES(`max_iterations`) with `max_restarts` restarts.
    pub coarse: Stop,
    /// Seed of the random initial test vectors.
    pub seed: u64,
    /// How the coarse operator is made.
    pub coarse_operator: MdwfCoarse,
    /// Multiply the smoothed test vectors by D_PV before they span the aggregates.
    pub through_pauli_villars: bool,
}

/// The smoother of the domain-wall multigrid.
#[derive(Clone, Copy, Debug)]
pub enum MdwfSmoother {
    /// At most `max_iterations` steps of GMRES on A, fewer if |r|²/|b|² drops below
    /// `tolerance_sqr`. Keeps `max_iterations + 1` vectors, and is a different map for
    /// every input.
    Gmres(Stop),
    /// `Polynomial`: a polynomial of A with `degree` coefficients, fitted to a random
    /// vector in the setup and applied `passes` times as a stationary correction. Keeps
    /// three vectors, and is the same linear map every time.
    Polynomial { degree: usize, passes: usize },
}

/// The coarse operator of the domain-wall multigrid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdwfCoarse {
    /// `P† D C_PV^-1 P`, from the parts of D with the exact site-local `C_PV^-1` as a right
    /// factor: reaches the eight neighbouring aggregates exactly.
    RightPreconditioned,
    /// `P† A P` of the operator of the cycle, found by probing and cut down to the eight
    /// neighbouring aggregates.
    Galerkin,
    /// `R A P` with a restriction of its own, from test vectors smoothed with A†; found by
    /// probing like `Galerkin`. A is not Γ5-Hermitian, so what its smoother leaves is not
    /// in the span of P.
    PetrovGalerkin,
}

impl MdwfMultigridParams {
    /// The settings of the working note for its 12^4 test (blocks of 3^4 there): 8 test
    /// vectors, no setup iterations, GMRES(4) as the smoother stopping at a residual of
    /// 1e-3, GMRES(5) with at most two cycles on the coarse lattice stopping at 1e-2.
    pub fn note(block: [usize; ND]) -> Self {
        Self {
            block,
            num_test_vectors: 8,
            setup_iterations: 0,
            smoother: MdwfSmoother::Gmres(Stop::simple(1e-6, 4)),
            coarse: Stop {
                tolerance_sqr: 1e-4,
                max_iterations: 5,
                max_restarts: 1,
            },
            seed: 1,
            coarse_operator: MdwfCoarse::RightPreconditioned,
            through_pauli_villars: true,
        }
    }
}

/// The multigrid preconditioner for `PvPreconditioned::new(op, ..)`: pass it as `m` to
/// BiCGStab or FGMRES solving `A y = X† b` (then x = C^-1 y; `PvPreconditioned::solution`).
pub fn mdwf_multigrid<'a, T, G>(
    op: &'a Mdwf<T, G>,
    lat: &Arc<Lattice>,
    params: &MdwfMultigridParams,
) -> Result<Multigrid<'a, Mdwf<T, G>, T>, MultigridError>
where
    T: Storage + 'a,
    G: GammaRepre + 'a,
{
    let ls = op.params().ls;
    let like = Field::<T, Spinor>::with_shape(lat, 24, ls, 1, 1);
    let sample = random_vectors::<T>(lat, ls, 1, params.seed ^ 0x5EED);
    let smoother = |dagger: bool| -> Result<Box<dyn LinearFn<Field<T, Spinor>> + 'a>, _> {
        let a = PvPreconditioned::new(op, &like);
        Ok(match (params.smoother, dagger) {
            (MdwfSmoother::Gmres(stop), false) => Box::new(Solve::new(Fgmres::new(stop), a)),
            (MdwfSmoother::Gmres(stop), true) => {
                Box::new(Solve::new(Fgmres::new(stop), Adjoint(a)))
            }
            (MdwfSmoother::Polynomial { degree, passes }, false) => Box::new(
                Polynomial::fit(a, &sample, degree, passes)
                    .map_err(|_| MultigridError::DependentPowers { degree })?,
            ),
            (MdwfSmoother::Polynomial { degree, passes }, true) => Box::new(
                Polynomial::fit(Adjoint(a), &sample, degree, passes)
                    .map_err(|_| MultigridError::DependentPowers { degree })?,
            ),
        })
    };
    let parts = CycleParts {
        operator: Box::new(PvPreconditioned::new(op, &like)),
        pre_smoother: Some(smoother(false)?),
        post_smoother: smoother(false)?,
        coarse_solver: Box::new(Fgmres::new(params.coarse)),
    };
    let mut hooks = SetupHooks {
        precondition_coarse_diagonal: true,
        ..SetupHooks::default()
    };
    if params.through_pauli_villars {
        hooks.after_smoothing = Some(Box::new(MdwfAt::new(op, Variant::D, 1.0)));
    }
    match params.coarse_operator {
        MdwfCoarse::RightPreconditioned => {
            hooks.coarse_right = Some(Box::new(SiteInverse {
                op,
                mass: 1.0,
                dagger: false,
            }));
        }
        MdwfCoarse::Galerkin => {
            hooks.galerkin_of = Some(Box::new(PvPreconditioned::new(op, &like)));
        }
        MdwfCoarse::PetrovGalerkin => {
            hooks.galerkin_of = Some(Box::new(PvPreconditioned::new(op, &like)));
            hooks.adjoint_smoother = Some(smoother(true)?);
        }
    }
    let agg = Aggregation::new(lat, params.block);
    let initial = random_vectors::<T>(lat, ls, params.num_test_vectors, params.seed);
    Multigrid::with_parts(op, &agg, initial, params.setup_iterations, parts, hooks)
}
