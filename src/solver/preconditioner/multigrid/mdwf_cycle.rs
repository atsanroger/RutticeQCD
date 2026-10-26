//! The cycle of the even-odd domain-wall multigrid (`mdwf_even_odd`): y = S b,
//! y += e(b − Ã y), y += S(b − Ã y), in the usual layout or in blocks of W sites.

use super::correction::{Correction, NeffQ, Transfer};
use super::cycle::{Corrects, TwoGrid};
use super::levels::{AbsMultilevel, BlockMap};
use super::mdwf_even_odd::{coarse_field, CoarseKrylov, PvEoBasis, PvEoParams, SharedOperator};
use super::mdwf_vectors::random_even;
use super::{CoarseOp, LevelBootstrap, MultigridError, SignedMultilevel};
use crate::fermion::{EoForm, FermionOp, Mdwf, MdwfLayout, PvEvenOdd};
use crate::kernel::{Dirac, GammaRepre, LaneOf};
use crate::solver::{largest_magnitude, Chebyshev};
use crate::{
    Aos, Aosoa, Coarse, EvenOddIndex, Field, Half, Lattice, LinearFn, Parity, ParityBlocks, Spinor,
    Storage, Vector,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

type HalfField<L = Aos, F = f64> = Field<F, Spinor, Half, L>;
type CoarseField<F = f64> = Field<F, Coarse>;

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

type Cycle<'a, G, CT, L, F> = TwoGrid<
    HalfField<L, F>,
    Chebyshev<SharedOperator<'a, G, L, F>, HalfField<L, F>>,
    InLayout<'a, G, CT, L, F>,
>;

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
