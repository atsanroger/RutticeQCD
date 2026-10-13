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

use super::levels::{solve_shared, AbsMultilevel, BlockMap, BlocksPlusLevels, OnEven};
use super::{
    Aggregation, Basis, BasisBuilder, CoarseOp, LevelBootstrap, MultigridError, Reflection,
    SignedMultilevel,
};
use crate::fermion::{
    Dagger, EoForm, EvenOdd, FermionOp, HopTerms, Mdwf, MdwfHalfWork, MdwfLayout, PvEvenOdd,
    Variant,
};
use crate::kernel::{Dirac, GammaRepre, LaneOf};
use crate::numerics::hermitian_eigen;
use crate::solver::{largest_magnitude, BiCgStab, Chebyshev, Minres, Solver, Stop};
use crate::{
    extract_parity, insert_parity, random_spinor, Aos, Aosoa, Coarse, ConvertFrom, EvenOddIndex,
    Field, Half, Lattice, Lexical, LinearFn, LinearOp, Parity, ParityBlocks, Spinor, Storage,
    Subspace, Vector, C64,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

type HalfField<L = Aos> = Field<f64, Spinor, Half, L>;
type CoarseField = Field<f64, Coarse>;

/// P: double-precision fields, single-precision storage, folded on the fifth dimension.
pub type PvEoBasis = Basis<f64, f32>;

/// The full-lattice vectors s_j and t̂_j of the vectors y_j on the even sites (see the top
/// of the file), one j at a time: `visit(i, v)` gets s_j as i = j and t̂_j as i = n + j.
/// At most a few full-lattice fields are alive at a time.
fn lifted_vectors<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    lat: &Arc<crate::Lattice>,
    y: &HalfField,
    mut visit: impl FnMut(usize, &Field<f64, Spinor>),
) {
    let index = op
        .even_odd_index()
        .expect("an even-odd decomposition")
        .clone();
    let n = y.num_vectors();
    let ls = op.params().ls;
    let full = || Field::<f64, Spinor>::with_shape(lat, 24, ls, 1, 1);
    let half = |parity: Parity| {
        let mut h = HalfField::with_shape(lat, 24, ls, 1, 1);
        h.set_parity(parity);
        h
    };
    let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
    let mut full_work = op.work();
    for j in 0..n {
        let yj = y.select(&[j]);
        // s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y)
        let mut x_e = half(Parity::Even);
        op.diag_inv(&mut work, &mut x_e, &yj, Parity::Even, Dagger::No);
        let mut t = half(Parity::Odd);
        op.hop(&mut work, &mut t, &x_e, Parity::Odd, Dagger::No);
        let mut x_o = half(Parity::Odd);
        op.diag_inv(&mut work, &mut x_o, &t, Parity::Odd, Dagger::No);
        x_o.scale(&[-1.0]);
        let mut s = full();
        insert_parity(&mut s, &x_e, &index);
        insert_parity(&mut s, &x_o, &index);
        // t̂ = (π D s, −(D_PV†)_oo⁻¹ (D_PV†)_oe π D s)
        let mut ds = full();
        op.mult(&mut full_work, &mut ds, &s, Variant::D);
        visit(j, &s);
        drop(s);
        let mut a = half(Parity::Even);
        extract_parity(&mut a, &ds, &index, Parity::Even);
        drop(ds);
        pv.hop(&mut work_pv, &mut t, &a, Parity::Odd, Dagger::Yes);
        pv.diag_inv(&mut work_pv, &mut x_o, &t, Parity::Odd, Dagger::Yes);
        x_o.scale(&[-1.0]);
        let mut t_hat = full();
        insert_parity(&mut t_hat, &a, &index);
        insert_parity(&mut t_hat, &x_o, &index);
        visit(n + j, &t_hat);
    }
}

/// The prolongation from vectors `y` on the even sites (see the top of the file): 2 n
/// test vectors for n vectors in `y`, on the aggregates `agg`.
pub fn pv_eo_basis<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    agg: &Arc<Aggregation>,
    y: &HalfField,
) -> Result<PvEoBasis, MultigridError> {
    let lat = agg.fine_lattice().clone();
    let mut builder =
        BasisBuilder::<f64, f32>::new(agg, 2 * y.num_vectors()).folded(Reflection(G::GAMMA5));
    let mut work = op.work();
    let mut g5 = Field::<f64, Spinor>::with_shape(&lat, 24, op.params().ls, 1, 1);
    lifted_vectors(op, pv, &lat, y, |i, v| {
        op.mult_gamma5(&mut work, &mut g5, v);
        builder.add(i, v, &g5);
    });
    builder.finish()
}

/// The vectors s_j and t̂_j the basis was built from, restricted to the coarse lattice
/// (P† s_j, P† t̂_j): the test vectors of the levels below it (`PvEoCycle::add_levels`).
pub fn pv_eo_coarse_vectors<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    basis: &PvEoBasis,
    y: &HalfField,
) -> Vec<CoarseField> {
    let lat = basis.aggregation().fine_lattice().clone();
    let mut restricted: Vec<Option<CoarseField>> = (0..2 * y.num_vectors()).map(|_| None).collect();
    lifted_vectors(op, pv, &lat, y, |i, v| {
        let mut c = basis.coarse_field(1);
        basis.restrict(&mut c, v);
        restricted[i] = Some(c);
    });
    restricted
        .into_iter()
        .map(|c| c.expect("every vector visited"))
        .collect()
}

/// Γ5 on a coarse field: −1 on the second half of the components.
fn coarse_gamma5<T: Storage>(v: &mut Field<T, Coarse>) {
    let dim = v.internal_dof() / 2;
    for site in v.as_mut_slice().chunks_exact_mut(2 * dim) {
        for r in &mut site[dim..] {
            *r = T::store(-r.load());
        }
    }
}

/// How the PV factor is applied.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Apply {
    /// H = Γ5 (Γ5 D_c Γ5) = D_c Γ5, the Γ5-adjoint (for MINRES).
    Gamma5Adjoint,
    /// D_c† exactly.
    Adjoint,
}

/// A coarse factor as a linear map counting its applications.
struct Counted<'a, T: Storage> {
    a: &'a mut CoarseOp<T>,
    apply: Apply,
    n: &'a Cell<usize>,
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for Counted<'_, T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.n.set(self.n.get() + inp.num_vectors());
        match self.apply {
            Apply::Gamma5Adjoint => {
                self.a.mult_dag(out, inp);
                coarse_gamma5(out);
            }
            Apply::Adjoint => self.a.mult_adjoint(out, inp),
        }
    }
}

/// The Schur complement on the even coarse sites of H = D_c Γ5, the Γ5-Hermitian form MINRES
/// works on: it is S Γ5, with S that of D_c (`CoarseOp::schur`), since Γ5 acts within a
/// site. Counts its applications.
struct CountedSchurGamma5<'a, T: Storage> {
    a: &'a CoarseOp<T>,
    n: &'a Cell<usize>,
    odd: Field<T, Coarse>,
    turned: Field<T, Coarse>,
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for CountedSchurGamma5<'_, T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.n.set(self.n.get() + inp.num_vectors());
        crate::fit(&mut self.turned, inp);
        self.turned.copy_from(inp);
        coarse_gamma5(&mut self.turned);
        self.a.schur(out, &self.turned, &mut self.odd);
    }
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
fn coarse_solve<T: Storage>(
    kind: CoarseKrylov,
    stop: Stop,
    a: &mut dyn LinearFn<Field<T, Coarse>>,
    m: Option<&mut dyn LinearFn<Field<T, Coarse>>>,
    x: &mut Field<T, Coarse>,
    b: &Field<T, Coarse>,
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
    /// The settings of the 12⁴ and 16⁴ runs in the log: Chebyshev 5 steps on [hi/30, hi],
    /// coarse solves to 1e-2, the quark factor by BiCGStab, the Pauli-Villars one by MINRES.
    fn default() -> Self {
        Self {
            chebyshev_steps: 5,
            chebyshev_ratio: 30.0,
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
pub struct SharedOperator<'a, G: GammaRepre = Dirac, L: MdwfLayout<f64> = Aos>(
    Rc<RefCell<PvEvenOdd<'a, f64, G, L>>>,
);

impl<G: GammaRepre, L: MdwfLayout<f64>> Clone for SharedOperator<'_, G, L> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, G: GammaRepre, L: MdwfLayout<f64>> SharedOperator<'a, G, L> {
    pub fn new(a: PvEvenOdd<'a, f64, G, L>) -> Self {
        Self(Rc::new(RefCell::new(a)))
    }

    /// See `PvEvenOdd::right_hand_side`.
    pub fn right_hand_side(&self, rhs: &mut HalfField<L>, b: &Field<f64, Spinor, Lexical, L>) {
        self.0.borrow_mut().right_hand_side(rhs, b);
    }

    /// See `PvEvenOdd::solution`.
    pub fn solution(
        &self,
        x: &mut Field<f64, Spinor, Lexical, L>,
        y: &HalfField<L>,
        b: &Field<f64, Spinor, Lexical, L>,
    ) {
        self.0.borrow_mut().solution(x, y, b);
    }
}

impl<G: GammaRepre, L: MdwfLayout<f64>> LinearFn<HalfField<L>> for SharedOperator<'_, G, L> {
    fn mult(&mut self, out: &mut HalfField<L>, inp: &HalfField<L>) {
        self.0.borrow_mut().mult(out, inp);
    }
}

impl<G: GammaRepre, L: MdwfLayout<f64>> LinearOp<HalfField<L>> for SharedOperator<'_, G, L> {
    fn mult_dag(&mut self, out: &mut HalfField<L>, inp: &HalfField<L>) {
        self.0.borrow_mut().mult_dag(out, inp);
    }
}

/// The layout of the operator a cycle preconditions: the usual one, or blocks of W sites
/// (SIMD). The coarse correction works in the usual layout (P, the coarse operators), so in
/// blocks the residual is converted before it and the correction after it.
pub trait CycleLayout: MdwfLayout<f64> {
    /// What the conversion needs (`ParityBlocks`; nothing for the usual layout).
    type Map;
    /// `None` if the lattice cannot be put into this layout with even-odd.
    fn map(lat: &Lattice) -> Option<Self::Map>;
    /// The field itself when it is in the usual layout.
    fn as_sites(f: &HalfField<Self>) -> Option<&HalfField>;
    fn as_sites_mut(f: &mut HalfField<Self>) -> Option<&mut HalfField>;
    fn to_sites(map: &Self::Map, out: &mut HalfField, inp: &HalfField<Self>);
    fn from_sites(map: &Self::Map, out: &mut HalfField<Self>, inp: &HalfField);
}

impl CycleLayout for Aos {
    type Map = ();
    fn map(_: &Lattice) -> Option<()> {
        Some(())
    }
    fn as_sites(f: &HalfField) -> Option<&HalfField> {
        Some(f)
    }
    fn as_sites_mut(f: &mut HalfField) -> Option<&mut HalfField> {
        Some(f)
    }
    fn to_sites(_: &(), out: &mut HalfField, inp: &HalfField) {
        out.copy_from(inp);
    }
    fn from_sites(_: &(), out: &mut HalfField, inp: &HalfField) {
        out.copy_from(inp);
    }
}

impl<const W: usize> CycleLayout for Aosoa<W>
where
    f64: LaneOf<W>,
{
    type Map = ParityBlocks<W>;
    fn map(lat: &Lattice) -> Option<ParityBlocks<W>> {
        ParityBlocks::new(lat)
    }
    fn as_sites(_: &HalfField<Self>) -> Option<&HalfField> {
        None
    }
    fn as_sites_mut(_: &mut HalfField<Self>) -> Option<&mut HalfField> {
        None
    }
    fn to_sites(map: &ParityBlocks<W>, out: &mut HalfField, inp: &HalfField<Self>) {
        map.to_sites(out, inp);
    }
    fn from_sites(map: &ParityBlocks<W>, out: &mut HalfField<Self>, inp: &HalfField) {
        map.to_blocks(out, inp);
    }
}

/// The multigrid cycle for `PvEvenOdd::new(op_a, pv_a, form, ..)`, as the preconditioner of a
/// flexible solver (`Gcr`, `Fgmres`, flexible `BiCgStab`) on that operator.
pub struct PvEoCycle<'a, G: GammaRepre = Dirac, CT: Storage = f32, L: CycleLayout = Aos> {
    op: &'a Mdwf<f64, G>,
    pv: &'a Mdwf<f64, G>,
    op_a: &'a Mdwf<f64, G>,
    pv_a: &'a Mdwf<f64, G>,
    form: EoForm,
    index: Arc<EvenOddIndex>,
    basis: &'a PvEoBasis,
    /// Shared by the coarse solver and its preconditioner (which applies it with levels).
    coarse_d: RefCell<CoarseOp<CT>>,
    coarse_pv: CoarseOp<CT>,
    blocks_d: BlockMap,
    /// Levels below the coarse lattice for the quark factor (`add_levels`).
    levels_d: Option<SignedMultilevel>,
    blocks_pv: BlockMap,
    /// Levels below the coarse lattice for the second factor (`add_pv_levels`).
    levels_pv: Option<AbsMultilevel>,
    smoother: Chebyshev<SharedOperator<'a, G, L>, HalfField<L>>,
    params: PvEoParams,
    hi: f64,
    work: [MdwfHalfWork<f64>; 2],
    /// Scratch on the even sites (taken out while the cycle runs) and on the coarse lattice:
    /// `g` in the precision of P, the others in that of the coarse solves.
    r: Option<HalfField<L>>,
    e: Option<HalfField<L>>,
    /// The residual and the correction in the usual layout, when `L` is another one.
    sites: Option<[HalfField; 2]>,
    map: L::Map,
    h: HalfField,
    g: CoarseField,
    gc: Field<CT, Coarse>,
    w: Field<CT, Coarse>,
    c: Field<CT, Coarse>,
    coarse_applications: Cell<usize>,
}

/// A zero field on the coarse lattice of `basis`, stored as `T`.
fn coarse_field<T: Storage>(basis: &PvEoBasis, num_vectors: usize) -> Field<T, Coarse> {
    Field::with_shape(
        basis.aggregation().coarse_lattice(),
        2 * basis.coarse_dim(),
        1,
        num_vectors,
        1,
    )
}

impl<'a, G: GammaRepre> PvEoCycle<'a, G, f32, Aos> {
    /// The cycle with single-precision coarse solves.
    pub fn new(
        op: &'a Mdwf<f64, G>,
        pv: &'a Mdwf<f64, G>,
        outer: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>, EoForm),
        basis: &'a PvEoBasis,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        Self::with_coarse_storage(op, pv, outer, basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage> PvEoCycle<'a, G, CT, Aos> {
    /// The cycle with the coarse solves stored as `CT`.
    pub fn with_coarse_storage(
        op: &'a Mdwf<f64, G>,
        pv: &'a Mdwf<f64, G>,
        outer: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>, EoForm),
        basis: &'a PvEoBasis,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        Self::with_layout(op, pv, outer, (outer.0, outer.1), (), basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage, const W: usize> PvEoCycle<'a, G, CT, Aosoa<W>>
where
    f64: LaneOf<W>,
{
    /// The cycle for the operator in blocks of `W` sites: `outer` in the usual layout
    /// (for the coarse correction) and `in_blocks` the same two operators in blocks.
    pub fn in_blocks(
        op: &'a Mdwf<f64, G>,
        pv: &'a Mdwf<f64, G>,
        outer: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>, EoForm),
        in_blocks: (&'a Mdwf<f64, G, Aosoa<W>>, &'a Mdwf<f64, G, Aosoa<W>>),
        basis: &'a PvEoBasis,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        let map = ParityBlocks::new(basis.aggregation().fine_lattice())
            .expect("the operators in blocks exist, so the lattice can be cut");
        Self::with_layout(op, pv, outer, in_blocks, map, basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage, L: CycleLayout> PvEoCycle<'a, G, CT, L> {
    /// The cycle for the operator `outer_l` (the operators of `outer` in the layout
    /// `L`), with `map = L::map(lattice)`.
    pub fn with_layout(
        op: &'a Mdwf<f64, G>,
        pv: &'a Mdwf<f64, G>,
        (op_a, pv_a, form): (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>, EoForm),
        (op_l, pv_l): (&'a Mdwf<f64, G, L>, &'a Mdwf<f64, G, L>),
        map: L::Map,
        basis: &'a PvEoBasis,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        let index = op
            .even_odd_index()
            .expect("an even-odd decomposition")
            .clone();
        let lat = basis.aggregation().fine_lattice().clone();
        let ls = op.params().ls;
        let mut like = HalfField::with_shape(&lat, 24, ls, 1, 1);
        like.set_parity(Parity::Even);
        let mut coarse_d = CoarseOp::build(op, &mut op.work(), basis);
        if params.coarse_even_odd {
            coarse_d.prepare_even_odd()?;
        }
        let mut coarse_pv = CoarseOp::build(pv, &mut pv.work(), basis);
        if params.coarse_even_odd && params.pv_solver == CoarseKrylov::Minres {
            coarse_pv.prepare_even_odd()?;
        }
        let blocks_d = BlockMap::inverse_diagonal(&coarse_d)?;
        assert!(
            params.d_solver != CoarseKrylov::Minres,
            "MINRES needs a Hermitian operator: not for the quark factor"
        );
        let blocks_pv = match params.pv_solver {
            CoarseKrylov::Minres => BlockMap::abs_inverse_of_a_gamma5(&coarse_pv),
            _ => BlockMap::inverse_diagonal_adjoint(&coarse_pv)?,
        };
        let (coarse_d, coarse_pv) = (coarse_d.to_storage(), coarse_pv.to_storage());
        let mut start_sites = like.zeros_like();
        extract_parity(
            &mut start_sites,
            &random_spinor(&lat, ls, params.seed),
            &index,
            Parity::Even,
        );
        let mut like_l = HalfField::<L>::with_shape(&lat, 24, ls, 1, 1);
        like_l.set_parity(Parity::Even);
        let mut start = like_l.zeros_like();
        L::from_sites(&map, &mut start, &start_sites);
        let hi = 1.1
            * largest_magnitude(
                &mut PvEvenOdd::new(op_l, pv_l, form, &like_l),
                &mut start,
                params.power_steps,
            );
        let smoother = Chebyshev::new(
            SharedOperator::new(PvEvenOdd::new(op_l, pv_l, form, &like_l)),
            params.chebyshev_steps,
            hi / params.chebyshev_ratio,
            hi,
            &like_l,
        );
        Ok(Self {
            op,
            pv,
            op_a,
            pv_a,
            form,
            index,
            basis,
            g: basis.coarse_field(1),
            gc: coarse_field(basis, 1),
            w: coarse_field(basis, 1),
            c: coarse_field(basis, 1),
            coarse_d: RefCell::new(coarse_d),
            coarse_pv,
            blocks_d,
            levels_d: None,
            blocks_pv,
            levels_pv: None,
            smoother,
            params,
            hi,
            work: [op.even_odd_work(), op_a.even_odd_work()],
            r: Some(like_l.zeros_like()),
            e: Some(like_l.zeros_like()),
            sites: None,
            map,
            h: like,
            coarse_applications: Cell::new(0),
        })
    }

    /// The operator the cycle approximates the inverse of, to be used by the solver it
    /// preconditions (one copy of its work fields for both).
    pub fn operator(&mut self) -> SharedOperator<'a, G, L> {
        self.smoother.operator().clone()
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
    pub fn add_levels_with(
        &mut self,
        vectors: &[CoarseField],
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<usize, MultigridError> {
        let levels = SignedMultilevel::build_with(
            self.coarse_d.get_mut(),
            vectors,
            super_blocks,
            bootstrap,
        )?;
        let applications = levels.setup_applications;
        self.levels_d = Some(levels);
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
            self.params.pv_solver == CoarseKrylov::Minres,
            "levels for the second factor go with MINRES"
        );
        let levels = AbsMultilevel::build(&mut self.coarse_pv, vectors, super_blocks)?;
        let applications = levels.setup_applications;
        self.levels_pv = Some(levels);
        Ok(applications)
    }

    /// 1.1 |λ_max| of the operator, the top of the smoother's interval.
    pub fn top(&self) -> f64 {
        self.hi
    }

    /// Coarse-operator applications so far (both factors, the preconditioners' included).
    pub fn coarse_applications(&self) -> usize {
        self.coarse_applications.get()
    }

    /// `coarse_correction` in the layout of the operator.
    fn correct(&mut self, e: &mut HalfField<L>, r: &HalfField<L>) {
        if let (Some(r), Some(e)) = (L::as_sites(r), L::as_sites_mut(e)) {
            self.coarse_correction(e, r);
            return;
        }
        let [mut r_s, mut e_s] = match self.sites.take() {
            Some(f) if f[0].num_vectors() == r.num_vectors() => f,
            _ => {
                let lat = self.basis.aggregation().fine_lattice();
                let mut f = HalfField::with_shape(lat, 24, r.ls(), r.num_vectors(), 1);
                f.set_parity(Parity::Even);
                [f.zeros_like(), f]
            }
        };
        L::to_sites(&self.map, &mut r_s, r);
        self.coarse_correction(&mut e_s, &r_s);
        L::from_sites(&self.map, e, &e_s);
        self.sites = Some([r_s, e_s]);
    }

    /// e = the coarse correction of r (see the top of the file). By itself it is the coarse
    /// part of a cycle whose smoother is elsewhere (`TwoGrid`, with the operator on a device).
    pub fn coarse_correction(&mut self, e: &mut HalfField, r: &HalfField) {
        let nv = r.num_vectors();
        if self.g.num_vectors() != nv {
            self.g = self.basis.coarse_field(nv);
        }
        for f in [&mut self.gc, &mut self.w, &mut self.c] {
            if f.num_vectors() != nv {
                *f = coarse_field(self.basis, nv);
            }
        }
        crate::fit(&mut self.h, r);
        let [work, work_a] = &mut self.work;
        // h = O_R r, g = P† ι h.
        match self.form {
            EoForm::Right => self
                .pv
                .diag(work, &mut self.h, r, Parity::Even, Dagger::Yes),
            EoForm::Left => self.pv_a.mult_q(&mut self.h, r, true),
        }
        self.basis.restrict_half(&mut self.g, &self.h, &self.index);
        self.gc.convert_from(&self.g);
        let stop = Stop::simple(
            self.params.coarse_tolerance_sqr,
            self.params.coarse_max_iterations,
        );
        let pv_stop = Stop::simple(self.params.pv_tolerance_sqr, self.params.pv_max_iterations);
        let n = &self.coarse_applications;
        // w = (P†D_PV†P)⁻¹ g: on the exact adjoint, or by MINRES on
        // H = Γ5 (Γ5 P†D_PV P Γ5) with H w = Γ5 g.
        self.w.set_zero();
        let apply = if self.params.pv_solver == CoarseKrylov::Minres {
            coarse_gamma5(&mut self.gc);
            Apply::Gamma5Adjoint
        } else {
            Apply::Adjoint
        };
        if apply == Apply::Gamma5Adjoint && self.coarse_pv.has_even_odd() {
            // H w = gc on the even coarse sites: H = D_c Γ5 for the stored D_c (taken as its
            // own Γ5-adjoint), so u = Γ5 w solves D_c u = gc. The blocks |H_xx|⁻¹ act within
            // a site and keep a field on the even sites there.
            let (mut source, mut odd) = (self.gc.zeros_like(), self.gc.zeros_like());
            self.coarse_pv.schur_source(&mut source, &self.gc, &mut odd);
            // Preconditioned by the blocks, or with levels below by the even sites of blocks
            // plus levels on the whole coarse lattice (Hermitian positive either way).
            let mut with_levels;
            let m: &mut dyn LinearFn<Field<CT, Coarse>> = match self.levels_pv.as_ref() {
                None => &mut self.blocks_pv,
                Some(levels) => {
                    let coarse_pv = &self.coarse_pv;
                    with_levels = OnEven {
                        inner: BlocksPlusLevels {
                            blocks: &mut self.blocks_pv,
                            levels: Some(levels),
                        },
                        even_part: move |out: &mut Field<CT, Coarse>, inp: &Field<CT, Coarse>| {
                            coarse_pv.even_part(out, inp)
                        },
                        whole: self.gc.zeros_like(),
                    };
                    &mut with_levels
                }
            };
            coarse_solve(
                self.params.pv_solver,
                pv_stop,
                &mut CountedSchurGamma5 {
                    a: &self.coarse_pv,
                    n,
                    odd: self.gc.zeros_like(),
                    turned: self.gc.zeros_like(),
                },
                Some(m),
                &mut self.w,
                &source,
            );
            coarse_gamma5(&mut self.w);
            self.coarse_pv
                .schur_solution(&mut self.w, &self.gc, &mut odd);
            coarse_gamma5(&mut self.w);
            // The source and the odd sites of the solution: one application's work.
            n.set(n.get() + nv);
        } else {
            coarse_solve(
                self.params.pv_solver,
                pv_stop,
                &mut Counted {
                    a: &mut self.coarse_pv,
                    apply,
                    n,
                },
                Some(&mut BlocksPlusLevels {
                    blocks: &mut self.blocks_pv,
                    levels: self.levels_pv.as_ref(),
                }),
                &mut self.w,
                &self.gc,
            );
        }
        // g = P† ι M π P w; M = 1 (right form) or C_e^(α) C_PV,e^(α)† (left form).
        self.g.convert_from(&self.w);
        self.basis.prolong_half(&mut self.h, &self.g, &self.index);
        if self.form == EoForm::Left {
            self.pv_a
                .diag(work_a, e, &self.h, Parity::Even, Dagger::Yes);
            self.op_a
                .diag(work_a, &mut self.h, e, Parity::Even, Dagger::No);
        }
        self.basis.restrict_half(&mut self.g, &self.h, &self.index);
        self.gc.convert_from(&self.g);
        // c = (P†DP)⁻¹ g.
        self.c.set_zero();
        let d_solver = self.params.d_solver;
        solve_shared(
            &self.coarse_d,
            Some(&mut self.blocks_d),
            self.levels_d.as_ref(),
            n,
            &mut |a, m, x, b| coarse_solve(d_solver, stop, a, m, x, b),
            &mut self.c,
            &self.gc,
        );
        // e = O_L π P c: C_e (right form) or Q_e⁻¹ (left form).
        self.g.convert_from(&self.c);
        self.basis.prolong_half(&mut self.h, &self.g, &self.index);
        match self.form {
            EoForm::Right => self.op.diag(work, e, &self.h, Parity::Even, Dagger::No),
            EoForm::Left => self.op_a.mult_q(e, &self.h, true),
        }
    }
}

impl<G: GammaRepre, CT: Storage, L: CycleLayout> LinearFn<HalfField<L>>
    for PvEoCycle<'_, G, CT, L>
{
    /// y = S b; y += e(b − Ã y); y += S(b − Ã y).
    fn mult(&mut self, y: &mut HalfField<L>, b: &HalfField<L>) {
        let nv = b.num_vectors();
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus = vec![C64::new(-1.0, 0.0); nv];
        let (mut r, mut e) = (
            self.r.take().expect("returned after every cycle"),
            self.e.take().expect("returned after every cycle"),
        );
        crate::fit(&mut r, b);
        crate::fit(&mut e, b);
        self.smoother.mult(y, b);
        self.smoother.operator().mult_xpay(&mut r, y, &minus, b);
        self.correct(&mut e, &r);
        y.axpy(&one, &e);
        self.smoother.operator().mult_xpay(&mut r, y, &minus, b);
        self.smoother.mult(&mut e, &r);
        y.axpy(&one, &e);
        self.r = Some(r);
        self.e = Some(e);
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
fn rotate(x: &mut HalfField, c: &[C64]) {
    let n = x.num_vectors();
    let site_len = x.site_len();
    let reals = site_len / n;
    crate::for_sites(x.slab_mut(0), site_len, |_, s| {
        let old: Vec<C64> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| C64::new(z[0], z[1]))
            .collect();
        for j in 0..n {
            for d in 0..reals / 2 {
                let mut z = C64::ZERO;
                for i in 0..n {
                    z += old[i * reals / 2 + d] * c[i * n + j];
                }
                s[j * reals + 2 * d] = z.re;
                s[j * reals + 2 * d + 1] = z.im;
            }
        }
    });
}

impl Subspace for HalfField {
    fn rotate(&mut self, c: &[C64]) {
        rotate(self, c);
    }
}

/// ⟨x_i, y_j⟩ for all i, j (row-major, x rows), one pair of vectors at a time.
fn gram<V: Vector>(x: &V, y: &V) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let mut g = vec![C64::ZERO; nx * ny];
    let mut d = [C64::ZERO];
    for j in 0..ny {
        let yj = y.select(&[j]);
        for i in 0..nx {
            x.select(&[i]).dot(&yj, &mut d);
            g[i * ny + j] = d[0];
        }
    }
    g
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
fn orthonormalise_all<V: Subspace>(x: &mut V) {
    let n = x.num_vectors();
    let (values, mut c) = sorted_eigen(n, &gram(x, x));
    for (j, l) in values.iter().enumerate() {
        assert!(*l > 1e-13 * values[n - 1], "the subspace lost rank");
        for i in 0..n {
            c[i * n + j] = c[i * n + j].scale(1.0 / l.sqrt());
        }
    }
    x.rotate(&c);
}

/// The random vectors on the even sites that `pv_eo_low_modes` starts from.
pub fn pv_eo_start<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> HalfField {
    let index = op
        .even_odd_index()
        .expect("an even-odd decomposition")
        .clone();
    let (ls, m) = (op.params().ls, params.vectors);
    let mut x = HalfField::with_shape(lat, 24, ls, m, 1);
    x.set_parity(Parity::Even);
    extract_parity(
        &mut x,
        &random_spinor(lat, ls * m, params.seed),
        &index,
        Parity::Even,
    );
    x
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
            let part = gram(&x, &w);
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
pub fn pv_eo_low_modes<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> (HalfField, Vec<f64>) {
    low_modes(
        |c: &HalfField| crate::Normal::new(PvEvenOdd::new(op, pv, EoForm::Right, c), c),
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

/// One round of the improvement of the test vectors y by the multigrid built on them, `cycle`:
/// y_j ← (`steps` steps of FGMRES on A x = y_j preconditioned by the cycle), one vector at a
/// time, and y orthonormalised.
pub fn bootstrap_round<V: Subspace, A: LinearFn<V>>(
    a: &mut A,
    cycle: &mut dyn LinearFn<V>,
    y: &mut V,
    steps: usize,
) {
    let mut fgmres = crate::solver::Fgmres::new(Stop {
        tolerance_sqr: 1e-30,
        max_iterations: steps,
        max_restarts: 0,
    });
    for j in 0..y.num_vectors() {
        let yj = y.select(&[j]);
        let mut x = yj.zeros_like();
        let _ = fgmres.solve(a, Some(&mut *cycle), &mut x, &yj);
        y.assign(&[j], &x);
    }
    orthonormalise_all(y);
}

/// Improve the test vectors y with the multigrid built on them: every round, the basis and the
/// cycle (right form) from the current y, then `bootstrap_round`.
pub fn pv_eo_bootstrap<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    agg: &Arc<Aggregation>,
    y: &mut HalfField,
    params: BootstrapParams,
) -> Result<PvEoBasis, MultigridError> {
    let like = y.select(&[0]);
    for _ in 0..params.rounds {
        let basis = pv_eo_basis(op, pv, agg, y)?;
        let mut cycle = PvEoCycle::new(op, pv, (op, pv, EoForm::Right), &basis, params.cycle)?;
        let mut a = PvEvenOdd::new(op, pv, EoForm::Right, &like);
        bootstrap_round(&mut a, &mut cycle, y, params.steps);
    }
    pv_eo_basis(op, pv, agg, y)
}
