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

use super::correction::{Correction, NeffQ, Transfer};
use super::cycle::{Corrects, TwoGrid};
use super::levels::{AbsMultilevel, BlockMap};
use super::{
    Aggregation, Basis, BasisBuilder, CoarseOp, LevelBootstrap, MultigridError, Reflection,
    SignedMultilevel,
};
use crate::fermion::{
    Dagger, EoForm, EvenOdd, FermionOp, HopTerms, Mdwf, MdwfLayout, PvEvenOdd, SplitParity, Variant,
};
use crate::kernel::{Dirac, GammaRepre, LaneOf, Number, Real};
use crate::numerics::hermitian_eigen;
use crate::solver::{largest_magnitude, BiCgStab, Chebyshev, Minres, Solver, Stop};
use crate::{
    Aos, Aosoa, Coarse, ConvertFrom, EvenOddIndex, Field, Half, Lattice, Lexical, LinearFn,
    LinearOp, Parity, ParityBlocks, Spinor, Storage, Subspace, TestRng, Vector, C64, F16,
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

/// The layout of the operator a cycle preconditions: the usual one, or blocks of W sites
/// (SIMD). The coarse correction works in the usual layout (P, the coarse operators), so in
/// blocks the residual is converted before it and the correction after it.
pub trait CycleLayout<F: Storage = f64>: MdwfLayout<F> {
    /// What the conversion needs (`ParityBlocks`; nothing for the usual layout).
    type Map;
    /// `None` if the lattice cannot be put into this layout with even-odd.
    fn map(lat: &Lattice) -> Option<Self::Map>;
    /// The field itself when it is in the usual layout.
    fn as_sites(f: &HalfField<Self, F>) -> Option<&HalfField<Aos, F>>;
    fn as_sites_mut(f: &mut HalfField<Self, F>) -> Option<&mut HalfField<Aos, F>>;
    fn to_sites(map: &Self::Map, out: &mut HalfField<Aos, F>, inp: &HalfField<Self, F>);
    fn from_sites(map: &Self::Map, out: &mut HalfField<Self, F>, inp: &HalfField<Aos, F>);
}

impl<F: Storage> CycleLayout<F> for Aos {
    type Map = ();
    fn map(_: &Lattice) -> Option<()> {
        Some(())
    }
    fn as_sites(f: &HalfField<Aos, F>) -> Option<&HalfField<Aos, F>> {
        Some(f)
    }
    fn as_sites_mut(f: &mut HalfField<Aos, F>) -> Option<&mut HalfField<Aos, F>> {
        Some(f)
    }
    fn to_sites(_: &(), out: &mut HalfField<Aos, F>, inp: &HalfField<Aos, F>) {
        out.copy_from(inp);
    }
    fn from_sites(_: &(), out: &mut HalfField<Aos, F>, inp: &HalfField<Aos, F>) {
        out.copy_from(inp);
    }
}

impl<F: Storage, const W: usize> CycleLayout<F> for Aosoa<W>
where
    F::Compute: LaneOf<W>,
{
    type Map = ParityBlocks<W>;
    fn map(lat: &Lattice) -> Option<ParityBlocks<W>> {
        ParityBlocks::new(lat)
    }
    fn as_sites(_: &HalfField<Self, F>) -> Option<&HalfField<Aos, F>> {
        None
    }
    fn as_sites_mut(_: &mut HalfField<Self, F>) -> Option<&mut HalfField<Aos, F>> {
        None
    }
    fn to_sites(map: &ParityBlocks<W>, out: &mut HalfField<Aos, F>, inp: &HalfField<Self, F>) {
        map.to_sites(out, inp);
    }
    fn from_sites(map: &ParityBlocks<W>, out: &mut HalfField<Self, F>, inp: &HalfField<Aos, F>) {
        map.to_blocks(out, inp);
    }
}

/// P between fields on the even sites and the coarse lattice, on the CPU.
pub(crate) struct HalfTransfer<'a, F: Storage = f64> {
    basis: &'a PvEoBasis<F>,
    index: Arc<EvenOddIndex>,
}

impl<F: Storage> Transfer<HalfField<Aos, F>, CoarseField<F>> for HalfTransfer<'_, F> {
    fn coarse_field(&self, vectors: usize) -> CoarseField<F> {
        self.basis.coarse_field(vectors)
    }
    fn restrict(&self, coarse: &mut CoarseField<F>, fine: &HalfField<Aos, F>) {
        self.basis.restrict_half(coarse, fine, &self.index);
    }
    fn prolong(&self, fine: &mut HalfField<Aos, F>, coarse: &CoarseField<F>) {
        self.basis.prolong_half(fine, coarse, &self.index);
    }
}

impl<F: Storage, G: GammaRepre> NeffQ<HalfField<Aos, F>> for Mdwf<F, G> {
    fn mult_q(&self, out: &mut HalfField<Aos, F>, inp: &HalfField<Aos, F>, inverse: bool) {
        Mdwf::mult_q(self, out, inp, inverse);
    }
}

/// The coarse correction on the CPU, in the usual layout, the coarse systems stored as `CT`.
type CpuCorrection<'a, G, CT, F> = Correction<
    'a,
    HalfField<Aos, F>,
    CoarseField<F>,
    Field<CT, Coarse>,
    Mdwf<F, G>,
    HalfTransfer<'a, F>,
    Rc<RefCell<CoarseOp<CT>>>,
>;

/// The coarse correction for an operator in the layout `L`: P and the coarse operators work in
/// the usual layout, so in blocks the residual is converted before it and the correction
/// after it.
pub(crate) struct InLayout<'a, G: GammaRepre, CT: Storage, L: CycleLayout<F>, F: Storage> {
    correction: CpuCorrection<'a, G, CT, F>,
    /// The residual and the correction in the usual layout, when `L` is another one.
    sites: Option<[HalfField<Aos, F>; 2]>,
    map: L::Map,
    lat: Arc<Lattice>,
}

impl<G: GammaRepre, CT: Storage, L: CycleLayout<F>, F: Storage> Corrects<HalfField<L, F>>
    for InLayout<'_, G, CT, L, F>
{
    fn correct(&mut self, e: &mut HalfField<L, F>, r: &HalfField<L, F>) {
        if let (Some(r), Some(e)) = (L::as_sites(r), L::as_sites_mut(e)) {
            self.correction.correct(e, r);
            return;
        }
        let [mut r_s, mut e_s] = match self.sites.take() {
            Some(f) if f[0].num_vectors() == r.num_vectors() => f,
            _ => {
                let mut f =
                    HalfField::<Aos, F>::with_shape(&self.lat, 24, r.ls(), r.num_vectors(), 1);
                f.set_parity(Parity::Even);
                [f.zeros_like(), f]
            }
        };
        L::to_sites(&self.map, &mut r_s, r);
        self.correction.correct(&mut e_s, &r_s);
        L::from_sites(&self.map, e, &e_s);
        self.sites = Some([r_s, e_s]);
    }
}

type Cycle<'a, G, CT, L, F> =
    TwoGrid<HalfField<L, F>, SharedOperator<'a, G, L, F>, InLayout<'a, G, CT, L, F>>;

/// The multigrid cycle for `PvEvenOdd::new(op_a, pv_a, form, ..)`, as the preconditioner of a
/// flexible solver (`Gcr`, `Fgmres`, flexible `BiCgStab`) on that operator: `TwoGrid` with the
/// Chebyshev smoother and `Correction`, built here on the CPU.
pub struct PvEoCycle<
    'a,
    G: GammaRepre = Dirac,
    CT: Storage = f32,
    L: CycleLayout<F> = Aos,
    F: Storage = f64,
> {
    cycle: Cycle<'a, G, CT, L, F>,
    /// What the cycle is built from, shared with its correction.
    parts: CoarseParts<CT>,
}

/// What a cycle is built from besides its operators and its prolongation: the coarse
/// operators as the coarse systems are solved, the inverses of their blocks, and the top of the
/// smoother's interval. Taken back from a cycle (`PvEoCycle::into_parts`), they make the cycle
/// for other links again without building any of it (`PvEoCycle::from_parts`): the coarse
/// correction is then that of the links it was built for, still a fixed map, and the solver it
/// preconditions works with the operator of the new links.
pub struct CoarseParts<CT: Storage> {
    coarse: (SharedCoarse<CT>, SharedCoarse<CT>),
    blocks: (Rc<RefCell<BlockMap>>, Rc<RefCell<BlockMap>>),
    /// The levels below the coarse lattice of either factor, built on these coarse operators.
    levels: (Option<SharedLevels<CT>>, Option<SharedLevels<CT>>),
    hi: f64,
}

/// A coarse operator shared by the cycles of the same parts.
type SharedCoarse<CT> = Rc<RefCell<CoarseOp<CT>>>;

/// Levels below the coarse lattice, shared by the cycles of the same parts.
type SharedLevels<CT> = Rc<dyn Fn(&mut Field<CT, Coarse>, &Field<CT, Coarse>)>;

/// Shared: the cycles made of the same parts read the same coarse operators and levels.
impl<CT: Storage> Clone for CoarseParts<CT> {
    fn clone(&self) -> Self {
        Self {
            coarse: (self.coarse.0.clone(), self.coarse.1.clone()),
            blocks: (self.blocks.0.clone(), self.blocks.1.clone()),
            levels: (self.levels.0.clone(), self.levels.1.clone()),
            hi: self.hi,
        }
    }
}

impl<CT: Storage> CoarseParts<CT> {
    /// The top of the smoother's interval they were built with.
    pub fn top(&self) -> f64 {
        self.hi
    }

    /// The same parts with the smoother on [hi / ratio, hi] for this `hi`: the cycle of Ã† with
    /// the top of that of Ã makes the adjoint of that cycle exactly (the polynomials the same).
    pub fn with_top(mut self, hi: f64) -> Self {
        self.hi = hi;
        self
    }
}

/// A `BlockMap` the correction applies and the cycle keeps.
struct SharedBlocks(Rc<RefCell<BlockMap>>);

impl<T: Storage> LinearFn<Field<T, Coarse>> for SharedBlocks {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.0.borrow_mut().mult(out, inp);
    }
}

/// A zero field on the coarse lattice of `basis`, stored as `T`.
fn coarse_field<T: Storage, F: Storage>(
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

impl<'a, G: GammaRepre, F: Storage> PvEoCycle<'a, G, f32, Aos, F> {
    /// The cycle with single-precision coarse solves.
    pub fn new(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        outer: (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        Self::with_coarse_storage(op, pv, outer, basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage, F: Storage> PvEoCycle<'a, G, CT, Aos, F> {
    /// The cycle of `parts` (taken from a cycle of the same `basis` and settings, for other
    /// links) for the operators given: nothing is built.
    pub fn from_parts(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        outer: (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
        parts: CoarseParts<CT>,
    ) -> Self {
        Self::assemble(op, pv, outer, (outer.0, outer.1), (), basis, params, parts)
    }

    /// The cycle with the coarse solves stored as `CT`.
    pub fn with_coarse_storage(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        outer: (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        Self::with_layout(op, pv, outer, (outer.0, outer.1), (), basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage, F: Storage, const W: usize> PvEoCycle<'a, G, CT, Aosoa<W>, F>
where
    F::Compute: LaneOf<W>,
{
    /// The cycle for the operator in blocks of `W` sites: `outer` in the usual layout
    /// (for the coarse correction) and `in_blocks` the same two operators in blocks.
    pub fn in_blocks(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        outer: (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        in_blocks: (&'a Mdwf<F, G, Aosoa<W>>, &'a Mdwf<F, G, Aosoa<W>>),
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        let map = ParityBlocks::new(basis.aggregation().fine_lattice())
            .expect("the operators in blocks exist, so the lattice can be cut");
        Self::with_layout(op, pv, outer, in_blocks, map, basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage, L: CycleLayout<F>, F: Storage> PvEoCycle<'a, G, CT, L, F> {
    /// The cycle for the operator `outer_l` (the operators of `outer` in the layout
    /// `L`), with `map = L::map(lattice)`.
    pub(crate) fn with_layout(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        outer: (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        layout: (&'a Mdwf<F, G, L>, &'a Mdwf<F, G, L>),
        map: L::Map,
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        let parts = Self::build_parts(op, pv, layout, outer.2, &map, basis, params)?;
        Ok(Self::assemble(
            op, pv, outer, layout, map, basis, params, parts,
        ))
    }

    /// The cycle of `parts` (made for the same `basis` and settings, by `MdwfBackend::rebuild`
    /// or another cycle) for the operator `outer`, the smoother on its operators in the layout
    /// `L` (`layout`, with `map = L::map(lattice)`): nothing is built.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts_in_layout(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        outer: (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        layout: (&'a Mdwf<F, G, L>, &'a Mdwf<F, G, L>),
        map: L::Map,
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
        parts: CoarseParts<CT>,
    ) -> Self {
        Self::assemble(op, pv, outer, layout, map, basis, params, parts)
    }

    /// The coarse operators of `op` and `pv` on `basis`, the inverses of their blocks, and the
    /// top of the smoother's interval for the operator of `outer_l`.
    fn build_parts(
        op: &Mdwf<F, G>,
        pv: &Mdwf<F, G>,
        (op_l, pv_l): (&Mdwf<F, G, L>, &Mdwf<F, G, L>),
        form: EoForm,
        map: &L::Map,
        basis: &PvEoBasis<F>,
        params: PvEoParams,
    ) -> Result<CoarseParts<CT>, MultigridError> {
        let index = op
            .even_odd_index()
            .expect("an even-odd decomposition")
            .clone();
        let lat = basis.aggregation().fine_lattice().clone();
        let ls = op.params().ls;
        // The coarse operators, with the even-odd form where they are solved in it, and the
        // matrices on the coarse sites that precondition their solves.
        // Built as the fields are stored; the inverses on the coarse sites in double precision.
        let mut coarse_d = basis.coarse_op(op, &mut op.work()).to_storage::<f64>();
        if params.coarse_even_odd {
            coarse_d.prepare_even_odd()?;
        }
        // The second factor: P†D_PV P for MINRES on its Hermitian form, else its exact
        // adjoint P†D_PV†P, solved as the first factor is.
        let minres = params.pv_solver == CoarseKrylov::Minres;
        let mut coarse_pv = basis.coarse_op(pv, &mut pv.work()).to_storage::<f64>();
        if !minres {
            coarse_pv = coarse_pv.adjoint();
        }
        if params.coarse_even_odd {
            coarse_pv.prepare_even_odd()?;
        }
        let blocks_d = BlockMap::inverse_diagonal(&coarse_d)?;
        let blocks_pv = match minres {
            true => BlockMap::abs_inverse_of_a_gamma5(&coarse_pv),
            false => BlockMap::inverse_diagonal(&coarse_pv)?,
        };
        // The smoother's interval [hi / ratio, hi], hi = 1.1 |λ_max| of the operator.
        let start_sites = random_even::<F>(&lat, &index, ls, 1, params.seed);
        let mut like_l = HalfField::<L, F>::with_shape(&lat, 24, ls, 1, 1);
        like_l.set_parity(Parity::Even);
        let mut start = like_l.zeros_like();
        L::from_sites(map, &mut start, &start_sites);
        let hi = 1.1
            * largest_magnitude(
                &mut PvEvenOdd::new(op_l, pv_l, form, &like_l),
                &mut start,
                params.power_steps,
            );
        Ok(CoarseParts {
            coarse: (
                Rc::new(RefCell::new(coarse_d.to_storage::<CT>())),
                Rc::new(RefCell::new(coarse_pv.to_storage::<CT>())),
            ),
            blocks: (
                Rc::new(RefCell::new(blocks_d)),
                Rc::new(RefCell::new(blocks_pv)),
            ),
            levels: (None, None),
            hi,
        })
    }

    /// The cycle of `parts` for the operators given.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        op: &'a Mdwf<F, G>,
        pv: &'a Mdwf<F, G>,
        (op_a, pv_a, form): (&'a Mdwf<F, G>, &'a Mdwf<F, G>, EoForm),
        (op_l, pv_l): (&'a Mdwf<F, G, L>, &'a Mdwf<F, G, L>),
        map: L::Map,
        basis: &'a PvEoBasis<F>,
        params: PvEoParams,
        parts: CoarseParts<CT>,
    ) -> Self {
        let index = op
            .even_odd_index()
            .expect("an even-odd decomposition")
            .clone();
        let lat = basis.aggregation().fine_lattice().clone();
        let ls = op.params().ls;
        let mut like = HalfField::<Aos, F>::with_shape(&lat, 24, ls, 1, 1);
        like.set_parity(Parity::Even);
        let mut correction = Correction::new(
            (op, pv),
            (op_a, pv_a),
            form,
            HalfTransfer {
                basis,
                index: index.clone(),
            },
            (parts.coarse.0.clone(), parts.coarse.1.clone()),
            (
                Box::new(SharedBlocks(parts.blocks.0.clone())),
                Box::new(SharedBlocks(parts.blocks.1.clone())),
            ),
            params,
            &like,
            &coarse_field::<CT, F>(basis, 1),
        );
        if let Some(l) = parts.levels.0.clone() {
            correction.set_levels_d(Box::new(move |out, inp| l(out, inp)));
        }
        if let Some(l) = parts.levels.1.clone() {
            correction.set_levels_pv(Box::new(move |out, inp| l(out, inp)));
        }
        let hi = parts.hi;
        let mut like_l = HalfField::<L, F>::with_shape(&lat, 24, ls, 1, 1);
        like_l.set_parity(Parity::Even);
        let smoother = Chebyshev::new(
            SharedOperator::new(PvEvenOdd::new(op_l, pv_l, form, &like_l)),
            params.chebyshev_steps,
            hi / params.chebyshev_ratio,
            hi,
            &like_l,
        );
        let in_layout = InLayout {
            correction,
            sites: None,
            map,
            lat,
        };
        Self {
            cycle: TwoGrid::new(smoother, in_layout, &like_l),
            parts,
        }
    }

    /// `parts`, the cycle given up.
    pub fn into_parts(self) -> CoarseParts<CT> {
        self.parts
    }

    /// The operator the cycle approximates the inverse of, to be used by the solver it
    /// preconditions (one copy of its work fields for both).
    pub fn operator(&mut self) -> SharedOperator<'a, G, L, F> {
        self.cycle.smoother_mut().operator().clone()
    }

    /// Add levels below the coarse lattice to the solves of the quark factor P†DP: a V-cycle on
    /// super-blocks of `super_blocks[ℓ]` coarse sites per direction on level ℓ + 2 (the last solved
    /// exactly), from test vectors on the coarse lattice (`pv_eo_coarse_vectors`).
    pub fn add_levels(
        &mut self,
        vectors: &[CoarseField],
        super_blocks: &[usize],
    ) -> Result<usize, MultigridError> {
        self.add_levels_with(vectors, super_blocks, &[])
    }

    /// `add_levels` with the test vectors of the levels refined (`LevelBootstrap`).
    pub(crate) fn add_levels_with(
        &mut self,
        vectors: &[CoarseField],
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<usize, MultigridError> {
        let levels = SignedMultilevel::build_with(
            &mut self.parts.coarse.0.borrow_mut(),
            vectors,
            super_blocks,
            bootstrap,
        )?;
        let applications = levels.setup_applications;
        let levels: SharedLevels<CT> = Rc::new(
            move |out: &mut Field<CT, Coarse>, inp: &Field<CT, Coarse>| levels.apply(out, inp),
        );
        let l = levels.clone();
        let correction = &mut self.cycle.correction_mut().correction;
        correction.set_levels_d(Box::new(move |out, inp| l(out, inp)));
        self.parts.levels.0 = Some(levels);
        Ok(applications)
    }

    /// Add levels below the coarse lattice to the solves of the second factor P†D_PV†P as well:
    /// the absolute-value preconditioner of its Hermitian form (`AbsMultilevel`), from the same
    /// test vectors as `add_levels`, with the last level solved exactly. Only with MINRES for
    /// that factor, which the preconditioner is made for; returns the applications of the
    /// coarse operator the setup took.
    pub fn add_pv_levels(
        &mut self,
        vectors: &[CoarseField],
        super_blocks: &[usize],
    ) -> Result<usize, MultigridError> {
        assert!(
            self.cycle.correction().correction.params().pv_solver == CoarseKrylov::Minres,
            "levels for the second factor go with MINRES"
        );
        let levels =
            AbsMultilevel::build(&mut self.parts.coarse.1.borrow_mut(), vectors, super_blocks)?;
        let applications = levels.setup_applications;
        let levels: SharedLevels<CT> = Rc::new(
            move |out: &mut Field<CT, Coarse>, inp: &Field<CT, Coarse>| levels.add(out, inp),
        );
        let l = levels.clone();
        let correction = &mut self.cycle.correction_mut().correction;
        correction.set_levels_pv(Box::new(move |out, inp| l(out, inp)));
        self.parts.levels.1 = Some(levels);
        Ok(applications)
    }

    /// 1.1 |λ_max| of the operator, the top of the smoother's interval.
    pub fn top(&self) -> f64 {
        self.parts.hi
    }

    /// Coarse-operator applications so far (both factors, the preconditioners' included).
    pub fn coarse_applications(&self) -> usize {
        self.cycle.correction().correction.coarse_applications()
    }

    /// e = the coarse correction of r (see the top of the file), in the usual layout. By
    /// itself it is the coarse part of a cycle whose smoother is elsewhere.
    pub fn coarse_correction(&mut self, e: &mut HalfField<Aos, F>, r: &HalfField<Aos, F>) {
        self.cycle.correction_mut().correction.correct(e, r);
    }
}

impl<G: GammaRepre, CT: Storage, L: CycleLayout<F>, F: Storage> LinearFn<HalfField<L, F>>
    for PvEoCycle<'_, G, CT, L, F>
{
    /// y = S b; y += e(b − Ã y); y += S(b − Ã y).
    fn mult(&mut self, y: &mut HalfField<L, F>, b: &HalfField<L, F>) {
        self.cycle.mult(y, b);
    }
}

/// Settings of the Chebyshev-filtered subspace iteration for the test vectors.
#[derive(Clone, Copy, Debug)]
pub struct LowModeParams {
    /// Vectors iterated (M).
    pub vectors: usize,
    /// Rounds of filtering and Rayleigh-Ritz.
    pub rounds: usize,
    /// Vectors filtered at a time (memory: the work fields of Ã are this large).
    pub chunk: usize,
    pub seed: u64,
}

impl Default for LowModeParams {
    /// The cheap setup of the log: M = 16, two rounds (with a bootstrap after them).
    fn default() -> Self {
        Self {
            vectors: 16,
            rounds: 2,
            chunk: 4,
            seed: 21,
        }
    }
}

/// `out = Σ_i x_i c[i n + j]` for every j, in place, site by site (no copy of x).
fn rotate<F: Storage>(x: &mut HalfField<Aos, F>, c: &[C64]) {
    let n = x.num_vectors();
    let site_len = x.site_len();
    let reals = site_len / n;
    crate::for_sites(x.slab_mut(0), site_len, |_, s| {
        let old: Vec<C64> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| C64::new(z[0].load().to_f64(), z[1].load().to_f64()))
            .collect();
        for j in 0..n {
            for d in 0..reals / 2 {
                let mut z = C64::ZERO;
                for i in 0..n {
                    z += old[i * reals / 2 + d] * c[i * n + j];
                }
                s[j * reals + 2 * d] = F::store(Number::from_f64(z.re));
                s[j * reals + 2 * d + 1] = F::store(Number::from_f64(z.im));
            }
        }
    });
}

/// ⟨x_i, y_j⟩ for all i, j (row-major, x rows) in one sweep over the sites: a site holds its
/// part of every vector of a set, so each vector is read once (instead of a copy and a dot
/// product for every pair).
fn gram<F: Storage>(x: &HalfField<Aos, F>, y: &HalfField<Aos, F>) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let reals = x.site_len() / nx;
    assert!(x.num_sites() == y.num_sites() && y.site_len() / ny == reals);
    assert!(x.outer() == 1 && y.outer() == 1);
    let mut sums = crate::reduce_sites_dyn(x.num_sites(), 2 * nx * ny, |site, acc| {
        let (a, b) = (x.site(0, site), y.site(0, site));
        for (i, ai) in a.chunks_exact(reals).enumerate() {
            for (j, bj) in b.chunks_exact(reals).enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (p, q) in ai.as_chunks::<2>().0.iter().zip(bj.as_chunks::<2>().0) {
                    let (pr, pi) = (p[0].load().to_f64(), p[1].load().to_f64());
                    let (qr, qi) = (q[0].load().to_f64(), q[1].load().to_f64());
                    re += pr * qr + pi * qi;
                    im += pr * qi - pi * qr;
                }
                acc[2 * (i * ny + j)] += re;
                acc[2 * (i * ny + j) + 1] += im;
            }
        }
    });
    x.lattice().comm.sum(&mut sums);
    sums.as_chunks::<2>()
        .0
        .iter()
        .map(|[re, im]| C64::new(*re, *im))
        .collect()
}

impl<F: Storage> Subspace for HalfField<Aos, F> {
    fn rotate(&mut self, c: &[C64]) {
        rotate(self, c);
    }

    fn gram(&self, y: &Self) -> Vec<C64> {
        gram(self, y)
    }
}

/// Eigenvalues ascending, with the eigenvectors (columns) in the same order.
fn sorted_eigen(n: usize, a: &[C64]) -> (Vec<f64>, Vec<C64>) {
    let (values, v) = hermitian_eigen(n, a);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|i, j| values[*i].total_cmp(&values[*j]));
    let mut vectors = vec![C64::ZERO; n * n];
    for (j, o) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + j] = v[i * n + o];
        }
    }
    (order.iter().map(|o| values[*o]).collect(), vectors)
}

/// Orthonormalise the vectors of x through their Gram matrix (Löwdin, x ← x G^-1/2 V).
pub fn orthonormalise_all<V: Subspace>(x: &mut V) {
    let n = x.num_vectors();
    let (values, mut c) = sorted_eigen(n, &x.gram(x));
    for (j, l) in values.iter().enumerate() {
        assert!(*l > 1e-13 * values[n - 1], "the subspace lost rank");
        for i in 0..n {
            c[i * n + j] = c[i * n + j].scale(1.0 / l.sqrt());
        }
    }
    x.rotate(&c);
}

/// `vectors` random vectors with `ls` spinors per site on the even sites, from `seed`: the
/// even sites of `random_spinor(lat, ls * vectors, seed)`, drawn site by site in the same
/// order, without that field on the whole lattice.
pub fn random_even<F: Storage>(
    lat: &Arc<Lattice>,
    index: &EvenOddIndex,
    ls: usize,
    vectors: usize,
    seed: u64,
) -> HalfField<Aos, F> {
    let mut x = HalfField::<Aos, F>::with_shape(lat, 24, ls, vectors, 1);
    x.set_parity(Parity::Even);
    let mut rng = TestRng::new(seed);
    let mut reals = vec![0.0; x.site_len()];
    for global in 0..lat.num_global_sites() {
        for r in reals.iter_mut() {
            *r = rng.uniform();
        }
        let Some(site) = lat.local_site(lat.global_site_coords(global)) else {
            continue;
        };
        let (parity, half) = index.half_site(site);
        if parity == Parity::Even {
            for (d, r) in x.site_mut(0, half).iter_mut().zip(&reals) {
                *d = F::store(Number::from_f64(*r));
            }
        }
    }
    x
}

/// The random vectors on the even sites that `pv_eo_low_modes` starts from.
pub fn pv_eo_start<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> HalfField<Aos, F> {
    let index = op.even_odd_index().expect("an even-odd decomposition");
    random_even(lat, index, op.params().ls, params.vectors, params.seed)
}

/// Approximate eigenvectors of the smallest eigenvalues of a Hermitian positive operator, from
/// the vectors `x`: Chebyshev-filtered subspace iteration with Rayleigh-Ritz, the degree
/// following √(hi/lo) (at least 40), lo the largest Ritz value of the round before (hi/1000 at
/// first). `normal(like)` is the operator for sets of as many vectors as `like`. With the
/// vectors, their Ritz values.
pub fn low_modes<V: Subspace, N: LinearFn<V>>(
    mut normal: impl FnMut(&V) -> N,
    mut x: V,
    params: LowModeParams,
) -> (V, Vec<f64>) {
    let m = x.num_vectors();
    let like = x.select(&[0]);
    let mut start = x.select(&[0]);
    let hi = 1.1 * largest_magnitude(&mut normal(&like), &mut start, 50);
    let mut lo = hi / 1000.0;
    let mut theta = vec![0.0; m];
    let chunks: Vec<Vec<usize>> = (0..m)
        .collect::<Vec<_>>()
        .chunks(params.chunk.max(1))
        .map(<[usize]>::to_vec)
        .collect();
    for _ in 0..params.rounds {
        let degree = ((1.5 * (hi / lo).sqrt()).ceil() as usize).clamp(40, 1000);
        for c in &chunks {
            let xc = x.select(c);
            let mut y = xc.zeros_like();
            crate::solver::ChebyshevFilter::new(normal(&xc), degree, lo, hi, &xc).mult(&mut y, &xc);
            x.assign(c, &y);
        }
        orthonormalise_all(&mut x);
        // G = X† A X, a chunk of columns at a time.
        let mut g = vec![C64::ZERO; m * m];
        for c in &chunks {
            let xc = x.select(c);
            let mut w = xc.zeros_like();
            normal(&xc).mult(&mut w, &xc);
            let part = x.gram(&w);
            for i in 0..m {
                for (k, j) in c.iter().enumerate() {
                    g[i * m + j] = part[i * c.len() + k];
                }
            }
        }
        let (values, v) = sorted_eigen(m, &g);
        x.rotate(&v);
        theta = values;
        lo = theta[m - 1];
    }
    (x, theta)
}

/// Approximate low modes of Ã (the right singular vectors of its smallest singular values,
/// eigenvectors of Ã†Ã) for the basis: `low_modes` on `params.vectors` random vectors on the
/// even sites.
pub fn pv_eo_low_modes<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> (HalfField<Aos, F>, Vec<f64>) {
    low_modes(
        |c: &HalfField<Aos, F>| crate::Normal::new(PvEvenOdd::new(op, pv, EoForm::Right, c), c),
        pv_eo_start(op, lat, params),
        params,
    )
}

/// Settings of the multigrid-accelerated improvement of the test vectors.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapParams {
    pub rounds: usize,
    /// Steps of FGMRES on Ã x = y with the cycle, per vector and round.
    pub steps: usize,
    /// Settings of the cycle used (its coarse tolerance, smoother).
    pub cycle: PvEoParams,
}

impl Default for BootstrapParams {
    /// Two rounds of four steps (the log: as good as two more filter rounds, cheaper).
    fn default() -> Self {
        Self {
            rounds: 2,
            steps: 4,
            cycle: PvEoParams::default(),
        }
    }
}

/// A test vector improved by the multigrid built on the test vectors, `cycle`: `steps` steps of
/// FGMRES on A x = y_j from x = 0, preconditioned by the cycle.
pub fn bootstrap_vector<V: Vector>(
    a: &mut dyn LinearFn<V>,
    cycle: &mut dyn LinearFn<V>,
    yj: &V,
    steps: usize,
) -> V {
    let mut fgmres = crate::solver::Fgmres::new(Stop {
        tolerance_sqr: 1e-30,
        max_iterations: steps,
        max_restarts: 0,
    });
    let mut x = yj.zeros_like();
    let _ = fgmres.solve(a, Some(cycle), &mut x, yj);
    x
}

/// The levels below the coarse lattice of a cycle: the super-blocks of each level, how each is
/// bootstrapped, and whether the solves of the Pauli-Villars factor have them too.
pub type LevelLists<'l> = (&'l [usize], &'l [LevelBootstrap], bool);

/// Where the test vectors wait while a round of the bootstrap improves them one at a time: on
/// the CPU where they are (`InPlace`), on the device on the host (`DeviceSpinor::stash`), so that
/// the device holds only the vector worked on besides the cycle.
pub(crate) trait Waiting<H, E>: Sized {
    fn wait(y: H) -> Result<Self, E>;
    fn num_vectors(&self) -> usize;
    fn vector(&self, j: usize) -> Result<H, E>;
    fn set_vector(&mut self, j: usize, x: &H) -> Result<(), E>;
    /// All the vectors as one set again.
    fn back(self) -> Result<H, E>;
}

/// Test vectors waiting where they are.
pub(crate) struct InPlace<H>(H);

impl<H: Vector, E> Waiting<H, E> for InPlace<H> {
    fn wait(y: H) -> Result<Self, E> {
        Ok(Self(y))
    }
    fn num_vectors(&self) -> usize {
        self.0.num_vectors()
    }
    fn vector(&self, j: usize) -> Result<H, E> {
        Ok(self.0.select(&[j]))
    }
    fn set_vector(&mut self, j: usize, x: &H) -> Result<(), E> {
        self.0.assign(&[j], x);
        Ok(())
    }
    fn back(self) -> Result<H, E> {
        Ok(self.0)
    }
}

/// What a round of the bootstrap does with Ã and the cycle of the current vectors, given the
/// vectors: `bootstrap_vector` for each, the vectors waiting in `W` meanwhile.
pub(crate) type RoundWork<'r, H, W, E> =
    dyn FnMut(&mut dyn LinearFn<H>, &mut dyn LinearFn<H>, H) -> Result<W, E> + 'r;

/// How a backend makes the cycle of a round of the bootstrap: `with_cycle(y, levels, work)`
/// makes P and the coarse correction of the vectors y (with the levels below the coarse lattice
/// of its settings when `levels`) and the smoother, hands them with Ã and y to `work`, and gives
/// the vectors back once its cycle is gone.
pub(crate) type WithCycle<'w, H, W, E> =
    dyn FnMut(H, bool, &mut RoundWork<'_, H, W, E>) -> Result<H, E> + 'w;

/// The bootstrap, the same on every backend: `params.rounds` rounds, each improving every test
/// vector with the cycle made of the vectors of the round before (`with_cycle`), then the
/// vectors orthonormalised. `on_vector(round, j, x)` and `on_round(round, y)` report.
///
/// The cycles have the levels below the coarse lattice from the second round on. In the first,
/// on the filtered vectors, the coarse solves were as fast without them (24⁴, m 0.001, on the
/// device: 19.8 s with, 19.7 s without, and 7 s to make them); in the second they took 26 s
/// against 45 s.
pub(crate) fn bootstrap_rounds<H: Subspace, W: Waiting<H, E>, E>(
    mut y: H,
    params: BootstrapParams,
    with_cycle: &mut WithCycle<'_, H, W, E>,
    on_vector: &mut dyn FnMut(usize, usize, &H),
    on_round: &mut dyn FnMut(usize, &H),
) -> Result<H, E> {
    for round in 0..params.rounds {
        let mut work = |a: &mut dyn LinearFn<H>, cycle: &mut dyn LinearFn<H>, vectors: H| {
            let mut waiting = W::wait(vectors)?;
            for j in 0..waiting.num_vectors() {
                let x = bootstrap_vector(&mut *a, &mut *cycle, &waiting.vector(j)?, params.steps);
                on_vector(round, j, &x);
                waiting.set_vector(j, &x)?;
            }
            Ok(waiting)
        };
        y = with_cycle(y, round > 0, &mut work)?;
        orthonormalise_all(&mut y);
        on_round(round, &y);
    }
    Ok(y)
}

/// Improve the test vectors y with the multigrid built on them: every round, the basis and the
/// cycle (right form) from the current y, then every vector by `bootstrap_vector`.
pub fn pv_eo_bootstrap<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    agg: &Arc<Aggregation>,
    y: &mut HalfField<Aos, F>,
    params: BootstrapParams,
) -> Result<PvEoBasis<F>, MultigridError> {
    pv_eo_bootstrap_stored(
        op,
        pv,
        agg,
        y,
        params,
        BasisStorage::Half,
        (&[], &[], false),
    )
}

/// `pv_eo_bootstrap` with P kept as `storage` says, in its rounds too, and the cycles of the
/// rounds after the first with the levels below the coarse lattice `levels`.
pub fn pv_eo_bootstrap_stored<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    agg: &Arc<Aggregation>,
    y: &mut HalfField<Aos, F>,
    params: BootstrapParams,
    storage: BasisStorage,
    (super_blocks, refine, for_pv): LevelLists<'_>,
) -> Result<PvEoBasis<F>, MultigridError> {
    let like = y.select(&[0]);
    // The vectors go through the rounds by value; one vector stands in for them meanwhile.
    let taken = std::mem::replace(y, like.zeros_like());
    *y = bootstrap_rounds::<_, InPlace<_>, _>(
        taken,
        params,
        &mut |y, with_levels, work| {
            let basis = pv_eo_basis_stored(op, pv, agg, &y, storage)?;
            let mut cycle = PvEoCycle::new(op, pv, (op, pv, EoForm::Right), &basis, params.cycle)?;
            if with_levels && !super_blocks.is_empty() {
                let v = pv_eo_coarse_vectors(op, pv, &basis, &y);
                cycle.add_levels_with(&v, super_blocks, refine)?;
                if for_pv {
                    if params.cycle.pv_solver != CoarseKrylov::Minres {
                        return Err(MultigridError::Backend("MultiGrid.levels_for_pauli_villars needs MINRES for the Pauli-Villars factor (Coarse_Solver_PV)".into()));
                    }
                    cycle.add_pv_levels(&v, super_blocks)?;
                }
            }
            let mut a = PvEvenOdd::new(op, pv, EoForm::Right, &like);
            let waiting = work(&mut a, &mut cycle, y)?;
            drop(cycle);
            waiting.back()
        },
        &mut |_, _, _| {},
        &mut |_, _| {},
    )?;
    pv_eo_basis_stored(op, pv, agg, y, storage)
}
