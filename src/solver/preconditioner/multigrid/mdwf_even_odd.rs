//! The multigrid for the even-odd Pauli-Villars operator of Möbius domain-wall fermions
//! (`PvEvenOdd`; the working log (branch `notes`), section 0, "factorized coarse correction").
//!
//! With S the Schur complement of D on the even sites, S_PV that of D_PV, ι the injection
//! of the even sites into the whole lattice and π its adjoint, S⁻¹ = π D⁻¹ ι and
//! S_PV†⁻¹ = π D_PV†⁻¹ ι, so
//!
//! ```text
//!   Ã⁻¹   = C_e S⁻¹ S_PV†⁻¹ C_PV,e†                    (right form)
//!   Ã_L⁻¹ = Q_e⁻¹ S⁻¹ (C_e^(α) C_PV,e^(α)†) S_PV†⁻¹ O_R  (left form, Neff's α)
//! ```
//!
//! (C^(α) = C Q: C_e^(α) C_PV,e^(α)† = C_e Q_e^k C_PV,e†, k = 2 with α in D_PV too, and then
//! O_R = Q_e⁻¹; k = 1 and O_R = 1 with α in D alone). The coarse correction replaces
//! D⁻¹ and D_PV†⁻¹ by their Galerkin inverses P (P†DP)⁻¹ P† and P (P†D_PV†P)⁻¹ P†:
//!
//! ```text
//!   e = O_L π P (P†DP)⁻¹ P† ι M π P (P†D_PV†P)⁻¹ P† ι O_R r,
//! ```
//!
//! with O_L = C_e, M = 1, O_R = C_PV,e† for the right form. P and the two coarse factors
//! are those of the plain operators: one setup serves every α and both forms.
//!
//! P is built from vectors y on the even sites (approximate low modes of Ã): for each,
//! s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y), whose image D s lives on the even sites, and
//! t̂ = (π D s, −(D_PV†)_oo⁻¹ (D_PV†)_oe π D s), whose image under D_PV† does; with these
//! the correction is exact on the vectors it is built from. The aggregates keep the whole
//! fifth dimension and are split by the chirality of Γ5 = γ5 R; P is stored folded and in
//! single precision.
//!
//! The coarse solves: the quark factor by BiCGStab with the exact inverses of the diagonal
//! blocks as right preconditioner; the Pauli-Villars factor, held as P†D_PV P and applied
//! as its Γ5-adjoint, by MINRES on H = Γ5 (P†D_PV P)† with the block |H_xx|⁻¹. The cycle:
//! y = S b, y += e(b − Ã y), y += S(b − Ã y), S a Chebyshev polynomial in the operator.
//!
//! This file has P and the operator; the cycle is in `mdwf_cycle`, the test vectors that P
//! is built from (low modes, bootstrap) in `mdwf_vectors`.

use super::{Aggregation, Basis, BasisBuilder, CoarseOp, MultigridError, Reflection};
use crate::fermion::{
    Dagger, EvenOdd, FermionOp, HopTerms, Mdwf, MdwfLayout, PvEvenOdd, SplitParity, Variant,
};
use crate::kernel::{Dirac, GammaRepre};
use crate::solver::{BiCgStab, Minres, Solver, Stop};
use crate::{
    Aos, Coarse, ConvertFrom, EvenOddIndex, Field, Half, Lexical, LinearFn, LinearOp, Parity,
    Spinor, Storage, Vector, F16,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

type HalfField<L = Aos, F = f64> = Field<F, Spinor, Half, L>;
type CoarseField<F = f64> = Field<F, Coarse>;

/// P for fields stored as `F`, folded on the fifth dimension, its vectors kept in half
/// precision (as on the device by default) or in single (`basis_precision : single`).
pub enum PvEoBasis<F: Storage = f64> {
    Half(Basis<F, F16>),
    Single(Basis<F, f32>),
}

/// How P keeps its vectors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasisStorage {
    #[default]
    Half,
    Single,
}

/// `$e` with `$b` the `Basis` inside `$basis`, whichever its storage.
macro_rules! on_basis {
    ($basis:expr, $b:ident => $e:expr) => {
        match $basis {
            PvEoBasis::Half($b) => $e,
            PvEoBasis::Single($b) => $e,
        }
    };
}

impl<F: Storage> PvEoBasis<F> {
    pub fn aggregation(&self) -> &Arc<Aggregation> {
        on_basis!(self, b => b.aggregation())
    }

    /// Complex numbers per coarse site.
    pub fn coarse_dim(&self) -> usize {
        on_basis!(self, b => b.coarse_dim())
    }

    pub fn num_test_vectors(&self) -> usize {
        on_basis!(self, b => b.num_test_vectors())
    }

    /// Whether P keeps only half of the fifth dimension (always, as built here).
    pub fn is_folded(&self) -> bool {
        on_basis!(self, b => b.is_folded())
    }

    pub fn coarse_field(&self, num_vectors: usize) -> CoarseField<F> {
        on_basis!(self, b => b.coarse_field(num_vectors))
    }

    pub fn restrict(&self, coarse: &mut CoarseField<F>, fine: &Field<F, Spinor>) {
        on_basis!(self, b => b.restrict(coarse, fine))
    }

    pub fn restrict_half(
        &self,
        coarse: &mut CoarseField<F>,
        half: &HalfField<Aos, F>,
        index: &EvenOddIndex,
    ) {
        on_basis!(self, b => b.restrict_half(coarse, half, index))
    }

    pub fn prolong(&self, fine: &mut Field<F, Spinor>, coarse: &CoarseField<F>) {
        on_basis!(self, b => b.prolong(fine, coarse))
    }

    pub fn prolong_half(
        &self,
        half: &mut HalfField<Aos, F>,
        coarse: &CoarseField<F>,
        index: &EvenOddIndex,
    ) {
        on_basis!(self, b => b.prolong_half(half, coarse, index))
    }

    /// P†AP for the operator `op` (`CoarseOp::build`).
    pub fn coarse_op<O: HopTerms<Field<F, Spinor>>>(
        &self,
        op: &O,
        work: &mut O::Work,
    ) -> CoarseOp<F> {
        on_basis!(self, b => CoarseOp::build(op, work, b))
    }
}

/// The vectors s_j and t̂_j on the whole lattice of the vectors y_j on the even sites (see the
/// top of the file), on every backend: `chunk` of the j at a time, `visit(j, s)` getting the s
/// of y_j .. y_(j+k−1) and `visit(n + j, t̂)` their t̂, in fields like `whole`. With
/// `unit_sites`, s is scaled to |s|² = `unit_sites` before t̂ is made of it, as P keeps its
/// vectors (`BasisBuilder::add`): P is the same, and its parts stay far from the smallest
/// numbers of half precision.
pub(crate) fn lifted_vectors<O, V, H, S>(
    (op, pv): (&O, &O),
    split: &S,
    whole: &V,
    y: &H,
    chunk: usize,
    unit_sites: Option<f64>,
    mut visit: impl FnMut(usize, &V),
) where
    O: EvenOdd<H> + FermionOp<V>,
    V: Vector,
    H: Vector,
    S: SplitParity<V, H>,
{
    let n = y.num_vectors();
    let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
    let mut full_work = op.work();
    let all: Vec<usize> = (0..n).collect();
    for which in all.chunks(chunk.max(1)) {
        let nv = which.len();
        let minus = vec![-1.0; nv];
        let yj = y.select(which);
        let mut s = whole.zeros_with(nv);
        let mut x_e = split.half_like(&s, Parity::Even);
        let mut t = split.half_like(&s, Parity::Odd);
        let mut x_o = t.zeros_like();
        // s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y)
        op.diag_inv(&mut work, &mut x_e, &yj, Parity::Even, Dagger::No);
        op.hop(&mut work, &mut t, &x_e, Parity::Odd, Dagger::No);
        op.diag_inv(&mut work, &mut x_o, &t, Parity::Odd, Dagger::No);
        x_o.scale(&minus);
        split.insert(&mut s, &x_e);
        split.insert(&mut s, &x_o);
        if let Some(sites) = unit_sites {
            to_unit_sites(&mut s, sites);
        }
        visit(which[0], &s);
        // t̂ = (π D s, −(D_PV†)_oo⁻¹ (D_PV†)_oe π D s), in the fields of s
        let mut ds = s.zeros_like();
        op.mult(&mut full_work, &mut ds, &s, Variant::D);
        split.extract(&mut x_e, &ds, Parity::Even);
        drop(ds);
        pv.hop(&mut work_pv, &mut t, &x_e, Parity::Odd, Dagger::Yes);
        pv.diag_inv(&mut work_pv, &mut x_o, &t, Parity::Odd, Dagger::Yes);
        x_o.scale(&minus);
        split.insert(&mut s, &x_e);
        split.insert(&mut s, &x_o);
        if let Some(sites) = unit_sites {
            to_unit_sites(&mut s, sites);
        }
        visit(n + which[0], &s);
    }
}

/// The numbering of the sites of each parity of `op`.
fn index_of<F: Storage, G: GammaRepre>(op: &Mdwf<F, G>) -> &Arc<EvenOddIndex> {
    op.even_odd_index().expect("an even-odd decomposition")
}

/// Every vector of `v` scaled to |v_j|² = `sites`.
fn to_unit_sites<V: Vector>(v: &mut V, sites: f64) {
    let mut norms = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut norms);
    let scales: Vec<f64> = norms
        .iter()
        .map(|n| if *n > 0.0 { (sites / n).sqrt() } else { 1.0 })
        .collect();
    v.scale(&scales);
}

/// The prolongation from vectors `y` on the even sites (see the top of the file): 2 n
/// test vectors for n vectors in `y`, on the aggregates `agg`.
pub fn pv_eo_basis<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    agg: &Arc<Aggregation>,
    y: &HalfField<Aos, F>,
) -> Result<PvEoBasis<F>, MultigridError> {
    pv_eo_basis_stored(op, pv, agg, y, BasisStorage::Half)
}

/// `pv_eo_basis` with P kept as `storage` says.
pub fn pv_eo_basis_stored<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    agg: &Arc<Aggregation>,
    y: &HalfField<Aos, F>,
    storage: BasisStorage,
) -> Result<PvEoBasis<F>, MultigridError> {
    fn built<F: Storage, S: Storage, G: GammaRepre>(
        op: &Mdwf<F, G>,
        pv: &Mdwf<F, G>,
        agg: &Arc<Aggregation>,
        y: &HalfField<Aos, F>,
    ) -> Result<Basis<F, S>, MultigridError> {
        let lat = agg.fine_lattice().clone();
        let mut builder =
            BasisBuilder::<F, S>::new(agg, 2 * y.num_vectors()).folded(Reflection(G::GAMMA5));
        let mut work = op.work();
        let whole = Field::<F, Spinor>::with_shape(&lat, 24, op.params().ls, 1, 1);
        let mut g5 = whole.zeros_like();
        let sites = Some(lat.num_global_sites() as f64);
        lifted_vectors((op, pv), &**index_of(op), &whole, y, 1, sites, |i, v| {
            op.mult_gamma5(&mut work, &mut g5, v);
            builder.add(i, v, &g5);
        });
        builder.finish()
    }
    Ok(match storage {
        BasisStorage::Half => PvEoBasis::Half(built(op, pv, agg, y)?),
        BasisStorage::Single => PvEoBasis::Single(built(op, pv, agg, y)?),
    })
}

/// The vectors s_j and t̂_j the basis was built from, restricted to the coarse lattice
/// (P† s_j, P† t̂_j): the test vectors of the levels below it (`PvEoCycle::add_levels`).
pub fn pv_eo_coarse_vectors<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    basis: &PvEoBasis<F>,
    y: &HalfField<Aos, F>,
) -> Vec<CoarseField> {
    let lat = basis.aggregation().fine_lattice().clone();
    let mut restricted: Vec<Option<CoarseField>> = (0..2 * y.num_vectors()).map(|_| None).collect();
    let whole = Field::<F, Spinor>::with_shape(&lat, 24, op.params().ls, 1, 1);
    lifted_vectors((op, pv), &**index_of(op), &whole, y, 1, None, |i, v| {
        let mut c = basis.coarse_field(1);
        basis.restrict(&mut c, v);
        let mut d = coarse_field::<f64, F>(basis, 1);
        d.convert_from(&c);
        restricted[i] = Some(d);
    });
    restricted
        .into_iter()
        .map(|c| c.expect("every vector visited"))
        .collect()
}

/// The Krylov solver of a coarse factor, right-preconditioned by the inverses of its
/// diagonal blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseKrylov {
    BiCgStab,
    /// Flexible GCR keeping the last `k` directions.
    Gcr {
        k: usize,
    },
    /// MINRES on the Γ5-Hermitian form H = Γ5 (Γ5 A Γ5) with the block |H_xx|⁻¹: only for the
    /// second (Pauli-Villars) factor, and exact only where A† = Γ5 A Γ5, which for Möbius
    /// domain-wall fermions holds approximately.
    Minres,
}

/// x = A⁻¹ b approximately by kind, preconditioned by m, from x = 0 (inner solves: no
/// application of A for the first residual nor for a last one); not reaching the tolerance
/// is not an error.
pub fn coarse_solve<V: Vector>(
    kind: CoarseKrylov,
    stop: Stop,
    a: &mut dyn LinearFn<V>,
    m: Option<&mut dyn LinearFn<V>>,
    x: &mut V,
    b: &V,
) {
    let _ = match kind {
        CoarseKrylov::BiCgStab => BiCgStab::inner(stop).solve(a, m, x, b),
        CoarseKrylov::Gcr { k } => crate::solver::Gcr::new(stop, k).solve(a, m, x, b),
        CoarseKrylov::Minres => Minres::inner(stop).solve(a, m, x, b),
    };
}

/// Settings of the cycle.
#[derive(Clone, Copy, Debug)]
pub struct PvEoParams {
    /// Steps of the Chebyshev smoother (applications of the operator: one fewer).
    pub chebyshev_steps: usize,
    /// The smoother acts on [hi/ratio, hi], hi = 1.1 |λ_max|.
    pub chebyshev_ratio: f64,
    /// The solve of the quark factor stops at |r|²/|b|² below this ...
    pub coarse_tolerance_sqr: f64,
    /// ... or after this many iterations;
    pub coarse_max_iterations: usize,
    /// that of the second factor likewise (MINRES: in its own norm).
    pub pv_tolerance_sqr: f64,
    pub pv_max_iterations: usize,
    /// The coarse factors are solved on the even coarse sites (the Schur complement, see
    /// `CoarseOp`) where the coarse lattice allows it: the quark factor, with or without
    /// levels below it, and the second factor when it is solved by MINRES.
    pub coarse_even_odd: bool,
    /// The solver of the quark factor P†DP (not `Minres`) ...
    pub d_solver: CoarseKrylov,
    /// ... and that of the second factor, P†D_PV†P (P†D†P for the normal equations).
    pub pv_solver: CoarseKrylov,
    /// Steps of the power iteration for |λ_max|, from a random vector with this seed.
    pub power_steps: usize,
    pub seed: u64,
}

impl Default for PvEoParams {
    /// The settings of the M5 = 1.0 (stout 0.1 × 3) runs: Chebyshev 8 steps on [hi/10, hi],
    /// coarse solves to 1e-2, the quark factor by BiCGStab, the Pauli-Villars one by MINRES.
    /// M5 = 1.8 without smearing takes 12 steps on [hi/30, hi]; the smoother is not chosen
    /// from the operator: with 5 steps on [hi/30, hi] the stout 12⁴ solves at m = 0.001
    /// stagnate (|r|² about 1e-5 after 2000 GCR(32) steps, 13 with 8 on [hi/10, hi]).
    fn default() -> Self {
        Self {
            chebyshev_steps: 8,
            chebyshev_ratio: 10.0,
            coarse_tolerance_sqr: 1e-2,
            coarse_max_iterations: 2000,
            pv_tolerance_sqr: 1e-2,
            pv_max_iterations: 2000,
            coarse_even_odd: true,
            d_solver: CoarseKrylov::BiCgStab,
            pv_solver: CoarseKrylov::Minres,
            power_steps: 50,
            seed: 5,
        }
    }
}

/// The operator of a cycle (`PvEoCycle::operator`), shared with the solver the cycle preconditions,
/// so that the two do not each hold its work fields.
pub struct SharedOperator<'a, G: GammaRepre = Dirac, L: MdwfLayout<F> = Aos, F: Storage = f64>(
    Rc<RefCell<PvEvenOdd<'a, F, G, L>>>,
);

impl<G: GammaRepre, L: MdwfLayout<F>, F: Storage> Clone for SharedOperator<'_, G, L, F> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, G: GammaRepre, L: MdwfLayout<F>, F: Storage> SharedOperator<'a, G, L, F> {
    pub fn new(a: PvEvenOdd<'a, F, G, L>) -> Self {
        Self(Rc::new(RefCell::new(a)))
    }

    /// See `PvEvenOdd::right_hand_side`.
    pub fn right_hand_side(&self, rhs: &mut HalfField<L, F>, b: &Field<F, Spinor, Lexical, L>) {
        self.0.borrow_mut().right_hand_side(rhs, b);
    }

    /// See `PvEvenOdd::solution`.
    pub fn solution(
        &self,
        x: &mut Field<F, Spinor, Lexical, L>,
        y: &HalfField<L, F>,
        b: &Field<F, Spinor, Lexical, L>,
    ) {
        self.0.borrow_mut().solution(x, y, b);
    }
}

impl<G: GammaRepre, L: MdwfLayout<F>, F: Storage> LinearFn<HalfField<L, F>>
    for SharedOperator<'_, G, L, F>
{
    fn mult(&mut self, out: &mut HalfField<L, F>, inp: &HalfField<L, F>) {
        self.0.borrow_mut().mult(out, inp);
    }
}

impl<G: GammaRepre, L: MdwfLayout<F>, F: Storage> LinearOp<HalfField<L, F>>
    for SharedOperator<'_, G, L, F>
{
    fn mult_dag(&mut self, out: &mut HalfField<L, F>, inp: &HalfField<L, F>) {
        self.0.borrow_mut().mult_dag(out, inp);
    }
}

/// A zero field on the coarse lattice of `basis`, stored as `T`.
pub(super) fn coarse_field<T: Storage, F: Storage>(
    basis: &PvEoBasis<F>,
    num_vectors: usize,
) -> Field<T, Coarse> {
    Field::with_shape(
        basis.aggregation().coarse_lattice(),
        2 * basis.coarse_dim(),
        1,
        num_vectors,
        1,
    )
}
