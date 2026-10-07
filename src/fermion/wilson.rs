//! Wilson fermions.
//!
//! D ψ(x) = ψ(x) - κ Σ_mu [ (1 - γ_mu) U_mu(x) ψ(x + mu) + (1 + γ_mu) U_mu(x - mu)^† ψ(x - mu) ]
//!
//! Provides: `mult` (fused, one sweep) for D, D† and γ5 D; the hopping terms one direction
//! at a time; the even-odd blocks (the site-diagonal blocks are the identity).
//!
//! The sweep is generic over the site-diagonal block (`DiagTerm`), so fermions that differ
//! from Wilson only within a site (clover) reuse everything here.
//! With several processes, a sweep exchanges the faces of the input: projected half
//! spinors, with the link already applied on the side that holds it. While the exchange is
//! under way the sweep computes the sites that need nothing from it, and the others after
//! it has finished.
//!
//! For fields in blocks of W sites (`Aosoa<W>`, the SIMD layout) there is a second sweep,
//! over blocks, at the end of this file: the same kernel functions with a lane type in place
//! of a real number. It covers what the sweep over sites covers except the single hopping
//! terms and the two-row links. On several processes it fills and reads the same face
//! buffers, one site at a time.
//!
//! Solvers: CG on the normal equations, BiCGStab on D, either with even-odd
//! preconditioning. Tools needed: none.
//!
//! The normalisation (hopping parameter κ, unit diagonal) and the placement of the boundary
//! phase follow Bridge++ so that results can be compared. Written from the formula above.

use super::{Dagger, EvenOdd, FermionOp, GaugeError, HopTerms, Variant};
use crate::kernel::{
    project, sink_spinor, transport, transport_half, ColorMatrix, Cplx, Dirac, GammaRepre,
    HalfSpinor, Lane, LaneOf, LinkUse, Number, Sign, Spinor as SiteSpinor, HALF_SPINOR_REALS, NC,
    NS, SPINOR_REALS,
};
use crate::{
    for_sites, Aos, Aosoa, BlockStencil, Boundary, Direction, Edge, EvenOddIndex, Field, Full,
    Gauge, Half, HaloExchange, HaloSpec, Lattice, Layout, Parity, Sites, Source, Spinor, Stencil,
    Storage, Vector, FACES, ND, NOT_RECEIVED,
};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct WilsonParams {
    /// Hopping parameter κ.
    pub kappa: f64,
    pub boundary: [Boundary; ND],
}

/// Scratch space for operators of Wilson type: the buffers of the halo exchanges, one set
/// per geometry and number of spinors per site, created on first use.
pub struct WilsonWork<T: Storage> {
    exchanges: Vec<(usize, usize, Box<dyn HaloExchange<T>>)>,
}

impl<T: Storage> WilsonWork<T> {
    fn new() -> Self {
        Self {
            exchanges: Vec::new(),
        }
    }

    /// The buffers for geometry `id` of the operator with `copies` spinors per site;
    /// `spec` describes them if they have to be created.
    fn exchange(
        &mut self,
        id: usize,
        copies: usize,
        lat: &Lattice,
        spec: impl FnOnce() -> HaloSpec,
    ) -> &mut dyn HaloExchange<T> {
        let found = self
            .exchanges
            .iter()
            .position(|(i, c, _)| *i == id && *c == copies);
        let at = found.unwrap_or_else(|| {
            let exchange = T::halo(lat.comm.as_ref(), &spec());
            self.exchanges.push((id, copies, exchange));
            self.exchanges.len() - 1
        });
        self.exchanges[at].2.as_mut()
    }
}

/// Which sites a sweep writes, where their neighbours are, and how both map to the
/// full-lattice numbering the gauge field uses.
struct Geometry {
    /// Distinguishes the geometries of one operator.
    id: usize,
    stencil: Stencil,
    /// Full-lattice index of each output site; `None` when the output is the full lattice.
    out_full: Option<Vec<u32>>,
    /// Full-lattice index of each input site; `None` when the input is the full lattice.
    inp_full: Option<Vec<u32>>,
    /// For each output site, whether any of its legs comes from another process. Empty on
    /// a single process.
    surface: Vec<bool>,
}

/// The part of the output a pass of a sweep computes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    All,
    /// The sites (or blocks) that need nothing from another process.
    Interior,
    /// The others.
    Surface,
}

/// One pass of a sweep: a part of the output, and what arrived on the faces, which only
/// the surface reads.
#[derive(Clone, Copy)]
struct Pass<'a, T> {
    received: &'a [&'a [T]; FACES],
    part: Part,
}

impl Part {
    /// Whether a pass over this part leaves out element `i`, given the flags of a geometry.
    #[inline(always)]
    fn skips(self, surface: &[bool], i: usize) -> bool {
        match self {
            Part::All => false,
            Part::Interior => surface[i],
            Part::Surface => !surface[i],
        }
    }
}

impl Geometry {
    fn new(
        id: usize,
        stencil: Stencil,
        out_full: Option<Vec<u32>>,
        inp_full: Option<Vec<u32>>,
    ) -> Self {
        let from_buffer = |site: usize| {
            (0..ND).any(|mu| {
                [Direction::Forward, Direction::Backward]
                    .into_iter()
                    .any(|dir| matches!(stencil.source(site, mu, dir), Source::Buffer { .. }))
            })
        };
        let surface = if stencil.has_buffers() {
            (0..stencil.num_sites()).map(from_buffer).collect()
        } else {
            Vec::new()
        };
        Self {
            id,
            stencil,
            out_full,
            inp_full,
            surface,
        }
    }

    fn full(lat: &Lattice, boundary: [Boundary; ND]) -> Self {
        Self::new(0, Stencil::nearest(lat, boundary), None, None)
    }

    /// Output on the sites of parity `to`, input on the other parity.
    fn parity(lat: &Lattice, boundary: [Boundary; ND], index: &EvenOddIndex, to: Parity) -> Self {
        Self::new(
            1 + to.index(),
            Stencil::nearest_parity(lat, boundary, index, to),
            Some(index.full_sites(to).to_vec()),
            Some(index.full_sites(to.flip()).to_vec()),
        )
    }

    #[inline(always)]
    fn out_site(&self, i: usize) -> usize {
        self.out_full.as_ref().map_or(i, |m| m[i] as usize)
    }

    #[inline(always)]
    fn inp_site(&self, i: usize) -> usize {
        self.inp_full.as_ref().map_or(i, |m| m[i] as usize)
    }
}

struct EvenOddGeometry {
    index: Arc<EvenOddIndex>,
    /// Indexed by the parity of the output.
    to: [Geometry; 2],
}

/// Whether a sweep includes the site-diagonal block.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Diagonal {
    /// The operator's own diagonal block applied to the input at the same site.
    Present,
    /// No diagonal term: the sweep is a pure hopping block.
    Absent,
}

/// The site-diagonal block of an operator of Wilson type. It must be Hermitian and commute
/// with γ5, so that D† = γ5 D γ5 keeps holding.
pub trait DiagTerm<T: Storage>: Send + Sync + 'static {
    /// True when the block is the identity; lets callers skip work.
    const IS_UNIT: bool;
    /// Nominal floating-point operations of applying the block at one site, including
    /// adding the result to the hopping term (see `FermionOp::flops_per_site`).
    const FLOPS_PER_SITE: usize;
    /// The block at `full_site` applied to `phi`.
    fn apply(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute>;
    /// Its inverse applied to `phi`.
    fn apply_inv(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute>;
    /// Rebuild whatever depends on the gauge field.
    fn set_gauge(&mut self, lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Result<(), GaugeError>;
}

/// The identity: plain Wilson fermions.
pub struct UnitDiag;

impl<T: Storage> DiagTerm<T> for UnitDiag {
    const IS_UNIT: bool = true;
    const FLOPS_PER_SITE: usize = 0;
    #[inline(always)]
    fn apply(&self, _full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        *phi
    }
    #[inline(always)]
    fn apply_inv(&self, _full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        *phi
    }
    fn set_gauge(&mut self, _lat: &Arc<Lattice>, _u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        Ok(())
    }
}

/// An operator of Wilson type: the Wilson hopping term plus a site-diagonal block `D`.
/// `Wilson` and `Clover` are this with different blocks.
///
/// `L` is the layout of the fields it acts on. The geometry and the operator's copy of the
/// links depend on it and are kept in `state`; the arithmetic of a leg (`kernel::project`,
/// `transport_half`, `sink_spinor`) is the same code for every layout.
pub struct WilsonOp<T: Storage, G: GammaRepre, D: DiagTerm<T>, L: WilsonLayout<T> = Aos> {
    params: WilsonParams,
    lat: Arc<Lattice>,
    state: L::State,
    link_format: LinkFormat,
    diag: D,
    _gamma: PhantomData<G>,
}

/// A layout the Wilson operator has a sweep for.
pub trait WilsonLayout<T: Storage>: Layout {
    /// The geometry and the links of an operator for this layout.
    type State: Send + Sync;
}

impl<T: Storage> WilsonLayout<T> for Aos {
    type State = SiteState<T>;
}

impl<T: Storage, const W: usize> WilsonLayout<T> for Aosoa<W> {
    type State = BlockState<T, W>;
}

/// What the operator keeps for fields in the usual layout.
pub struct SiteState<T: Storage> {
    full: Geometry,
    /// Present when every extent of the lattice is even.
    even_odd: Option<EvenOddGeometry>,
    links: LinkStore<T>,
}

/// Wilson fermions: the diagonal block is the identity.
pub type Wilson<T, G = Dirac> = WilsonOp<T, G, UnitDiag>;

/// Wilson fermions on fields in blocks of `W` sites (the SIMD layout). One process, links
/// kept in full.
pub type WilsonSimd<T, const W: usize, G = Dirac> = WilsonOp<T, G, UnitDiag, Aosoa<W>>;

/// One leg of the stencil resolved for one output site: where the neighbour is and what
/// is still to be done to it.
///
/// Links are referred to where they are stored and converted when they are used: holding
/// eight converted matrices per site cost a copy through the stack for each.
#[derive(Clone, Copy)]
enum Hop<'a, T: Storage, R: LinkRepr> {
    /// Nothing arrives (Dirichlet boundary).
    Absent,
    /// A site of the input field: project, apply the phase and the link.
    Local {
        site: usize,
        link: &'a R::Stored<T>,
        phase: Option<Cplx<T::Compute>>,
    },
    /// A projected spinor from the neighbouring process; the link is on this side.
    Projected {
        index: usize,
        link: &'a R::Stored<T>,
        phase: Option<Cplx<T::Compute>>,
    },
    /// A spinor the neighbouring process has already transported.
    Transported { index: usize },
}

/// Reads spinor `r` of the sites of an input field, with γ5 applied first if `pre` is set.
/// A struct with methods rather than closures, so that the reads are expanded in place.
///
/// The field is held as a slice of stored spinors, `copies` of them per site, for the same
/// reason as the links in `Fixed`.
struct Input<'a, T: Storage, G: GammaRepre> {
    spinors: &'a [[T; SPINOR_REALS]],
    r: usize,
    /// Spinors per site.
    copies: usize,
    pre: bool,
    _gamma: PhantomData<G>,
}

impl<T: Storage, G: GammaRepre> Input<'_, T, G> {
    #[inline(always)]
    fn local(&self, site: usize) -> SiteSpinor<T::Compute> {
        let psi = load_spinor::<T>(&self.spinors[site * self.copies + self.r]);
        if self.pre {
            G::GAMMA5.apply(&psi)
        } else {
            psi
        }
    }

    /// Position in a face buffer of spinor `r` of received site `index`.
    #[inline(always)]
    fn slot(&self, index: usize) -> usize {
        index * self.copies + self.r
    }
}

/// How an operator keeps the links.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinkFormat {
    /// All 18 reals of each link.
    #[default]
    Full,
    /// The first two rows, 12 reals; the third row is rebuilt as the conjugate of their
    /// cross product every time the link is used. A third less gauge-field memory to read,
    /// which is what limits the speed once all cores of a processor are in use, at the
    /// price of more arithmetic per link. Only for links in SU(3): `set_gauge` checks.
    TwoRows,
}

/// Largest deviation from unitarity and from unit determinant that `set_gauge` accepts
/// for `LinkFormat::TwoRows`.
const SU3_TOLERANCE: f64 = 1e-10;

/// The links of the operator, `[mu * num_sites + site]`, in the format it was asked for.
enum LinkStore<T> {
    Unset,
    Full(Vec<[T; FULL_LINK_REALS]>),
    TwoRows(Vec<[T; TWO_ROW_REALS]>),
}

const FULL_LINK_REALS: usize = 2 * NC * NC;
/// Links converted per task when the operator's copy is filled. It is filled by the
/// threads of the pool so that its pages are spread over the memory nodes like those of
/// the fields (`Cpu::alloc`).
const LINKS_PER_TASK: usize = 1024;
const TWO_ROW_REALS: usize = 2 * 2 * NC;

/// One way of keeping a link. The sweeps are generic over it, so the choice costs nothing
/// in the site loop; it is made once per sweep from the operator's `LinkFormat`.
trait LinkRepr: Copy + Send + Sync + 'static {
    type Stored<T: Storage>: Copy + Send + Sync;
    /// The links of a store that holds this representation.
    fn of<T: Storage>(store: &LinkStore<T>) -> &[Self::Stored<T>];
    fn load<T: Storage>(s: &Self::Stored<T>) -> ColorMatrix<T::Compute>;
}

#[derive(Clone, Copy)]
struct FullLinks;
#[derive(Clone, Copy)]
struct TwoRowLinks;

impl LinkRepr for FullLinks {
    type Stored<T: Storage> = [T; FULL_LINK_REALS];

    fn of<T: Storage>(store: &LinkStore<T>) -> &[Self::Stored<T>] {
        match store {
            LinkStore::Full(links) => links,
            _ => unreachable!("the sweep is chosen from the format of the store"),
        }
    }

    #[inline(always)]
    fn load<T: Storage>(s: &Self::Stored<T>) -> ColorMatrix<T::Compute> {
        let mut m = ColorMatrix::zero();
        for a in 0..NC {
            for b in 0..NC {
                let k = 2 * (a * NC + b);
                m.0[a][b] = Cplx::new(s[k].load(), s[k + 1].load());
            }
        }
        m
    }
}

impl LinkRepr for TwoRowLinks {
    type Stored<T: Storage> = [T; TWO_ROW_REALS];

    fn of<T: Storage>(store: &LinkStore<T>) -> &[Self::Stored<T>] {
        match store {
            LinkStore::TwoRows(links) => links,
            _ => unreachable!("the sweep is chosen from the format of the store"),
        }
    }

    /// For U in SU(3) the third row is the conjugate of the cross product of the first two.
    #[inline(always)]
    fn load<T: Storage>(s: &Self::Stored<T>) -> ColorMatrix<T::Compute> {
        let mut m = ColorMatrix::zero();
        for a in 0..2 {
            for b in 0..NC {
                let k = 2 * (a * NC + b);
                m.0[a][b] = Cplx::new(s[k].load(), s[k + 1].load());
            }
        }
        let (x, y) = (m.0[0], m.0[1]);
        m.0[2] = [
            (x[1] * y[2] - x[2] * y[1]).conj(),
            (x[2] * y[0] - x[0] * y[2]).conj(),
            (x[0] * y[1] - x[1] * y[0]).conj(),
        ];
        m
    }
}

/// How far a matrix is from SU(3): the largest element of `U U† - 1` and `|det U - 1|`.
fn su3_deviation(u: &ColorMatrix<f64>) -> f64 {
    let product = u.mul(&u.adj());
    let mut worst = 0.0f64;
    for a in 0..NC {
        for b in 0..NC {
            let want = if a == b { Cplx::ONE } else { Cplx::ZERO };
            worst = worst.max((product.0[a][b] - want).norm_sqr().sqrt());
        }
    }
    let m = &u.0;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    worst.max((det - Cplx::ONE).norm_sqr().sqrt())
}

/// What is the same for every site of a sweep: the gauge field and, for each leg, the
/// phase of a hop across the boundary of the global lattice.
///
/// The gauge field is held as one slice of links per direction, indexed by the full-lattice
/// site. Reading a link is then one index with one bounds check; going through
/// `Field::site` for every read recomputed the site's range from the shape of the field
/// each time.
struct Fixed<'a, T: Storage, R: LinkRepr> {
    links: [&'a [R::Stored<T>]; ND],
    phase: [Option<Cplx<T::Compute>>; FACES],
}

/// The phase a hop in direction `dir` picks up when it crosses a boundary of the global
/// lattice with this condition. A hop backward picks up the conjugate.
fn boundary_phase<C: Number>(boundary: Boundary, dir: Direction) -> Option<Cplx<C>> {
    let phase = match boundary {
        Boundary::Periodic | Boundary::Dirichlet => None,
        Boundary::Antiperiodic => Some((-1.0, 0.0)),
        Boundary::Phase(re, im) => Some(match dir {
            Direction::Forward => (re, im),
            Direction::Backward => (re, -im),
        }),
    };
    phase.map(|(re, im)| Cplx::new(Number::from_f64(re), Number::from_f64(im)))
}

/// The spin structure and link use of the legs in one direction.
#[inline(always)]
fn leg_kind(dir: Direction) -> (Sign, LinkUse) {
    match dir {
        Direction::Forward => (Sign::Minus, LinkUse::Direct),
        Direction::Backward => (Sign::Plus, LinkUse::Adjoint),
    }
}

impl<T: Storage, G: GammaRepre> WilsonOp<T, G, UnitDiag> {
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Self {
        Self::with_diag(lat, params, UnitDiag)
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> WilsonOp<T, G, D> {
    /// An operator of Wilson type with the given site-diagonal block.
    pub fn with_diag(lat: &Arc<Lattice>, params: WilsonParams, diag: D) -> Self {
        let even_odd = lat.local.iter().all(|l| l % 2 == 0).then(|| {
            let index = Arc::new(EvenOddIndex::new(lat));
            let to = [Parity::Even, Parity::Odd]
                .map(|p| Geometry::parity(lat, params.boundary, &index, p));
            EvenOddGeometry { index, to }
        });
        Self {
            params,
            lat: lat.clone(),
            state: SiteState {
                full: Geometry::full(lat, params.boundary),
                even_odd,
                links: LinkStore::Unset,
            },
            link_format: LinkFormat::Full,
            diag,
            _gamma: PhantomData,
        }
    }

    pub fn params(&self) -> &WilsonParams {
        &self.params
    }

    /// The even-odd numbering this operator uses, if the lattice allows one.
    pub fn even_odd_index(&self) -> Option<&Arc<EvenOddIndex>> {
        self.state.even_odd.as_ref().map(|eo| &eo.index)
    }

    /// Choose how the links are kept. The gauge field has to be set again afterwards.
    pub fn with_link_format(mut self, format: LinkFormat) -> Self {
        self.link_format = format;
        self.state.links = LinkStore::Unset;
        self
    }

    pub fn link_format(&self) -> LinkFormat {
        self.link_format
    }

    /// The phase a hop in direction (mu, dir) picks up when it crosses the boundary of the
    /// global lattice. A hop backward picks up the conjugate.
    fn boundary_phase(&self, mu: usize, dir: Direction) -> Option<Cplx<T::Compute>> {
        boundary_phase(self.params.boundary[mu], dir)
    }

    /// The leg in direction (mu, dir) of output site `site`.
    ///
    /// Forward legs use the link at the output site. Backward legs use the link at the
    /// neighbour; when the neighbour is on another process, that process applies it (and
    /// the phase) before sending.
    #[inline(always)]
    fn resolve<'a, R: LinkRepr>(
        geo: &Geometry,
        fixed: &Fixed<'a, T, R>,
        site: usize,
        mu: usize,
        dir: Direction,
    ) -> Hop<'a, T, R> {
        let phase = |wrapped: bool| {
            if wrapped {
                fixed.phase[Stencil::leg(mu, dir)]
            } else {
                None
            }
        };
        match (geo.stencil.source(site, mu, dir), dir) {
            (Source::Zero, _) => Hop::Absent,
            (Source::Local { site: n, wrapped }, _) => Hop::Local {
                site: n,
                link: match dir {
                    Direction::Forward => &fixed.links[mu][geo.out_site(site)],
                    Direction::Backward => &fixed.links[mu][geo.inp_site(n)],
                },
                phase: phase(wrapped),
            },
            (Source::Buffer { index, wrapped }, Direction::Forward) => Hop::Projected {
                index,
                link: &fixed.links[mu][geo.out_site(site)],
                phase: phase(wrapped),
            },
            (Source::Buffer { index, .. }, Direction::Backward) => Hop::Transported { index },
        }
    }

    /// What `resolve` needs and does not change during a sweep.
    fn fixed<R: LinkRepr>(&self) -> Fixed<'_, T, R> {
        let mut phase = [None; FACES];
        for mu in 0..ND {
            for dir in [Direction::Forward, Direction::Backward] {
                phase[Stencil::leg(mu, dir)] = self.boundary_phase(mu, dir);
            }
        }
        let (all, n) = (R::of(&self.state.links), self.lat.num_sites());
        Fixed {
            links: core::array::from_fn(|mu| &all[mu * n..(mu + 1) * n]),
            phase,
        }
    }

    /// The half spinor that arrives along one leg, ready for the sink, for spinor `r` of
    /// the site. `local(site)` reads the input field; `received` is what came in on the
    /// face of this leg.
    #[inline(always)]
    fn arrive<R: LinkRepr>(
        hop: &Hop<'_, T, R>,
        mu: usize,
        dir: Direction,
        input: &Input<'_, T, G>,
        received: &[T],
    ) -> Option<HalfSpinor<T::Compute>> {
        let (sign, link_use) = leg_kind(dir);
        match hop {
            Hop::Absent => None,
            Hop::Local { site, link, phase } => Some(transport(
                &G::GAMMA[mu],
                sign,
                &input.local(*site),
                &R::load::<T>(link),
                link_use,
                *phase,
            )),
            Hop::Projected { index, link, phase } => Some(transport_half(
                &load_half::<T>(received, input.slot(*index)),
                &R::load::<T>(link),
                link_use,
                *phase,
            )),
            Hop::Transported { index } => Some(load_half::<T>(received, input.slot(*index))),
        }
    }

    /// Add the forward and the backward leg of direction `MU` to `hops`.
    ///
    /// The direction is a compile-time constant so that the gamma matrix is one too: its
    /// table of columns and phases then folds into straight-line code. With the direction
    /// as a loop variable the compiler kept the loop and the table lookups (seen in the
    /// generated code as indirect jumps on every component).
    #[inline(always)]
    fn add_legs<const MU: usize, R: LinkRepr>(
        hops: &mut SiteSpinor<T::Compute>,
        forward: &Hop<'_, T, R>,
        backward: &Hop<'_, T, R>,
        input: &Input<'_, T, G>,
        received: &[&[T]; FACES],
    ) {
        let (fwd, bwd) = (Direction::Forward, Direction::Backward);
        let gamma = &G::GAMMA[MU];
        let face = received[Stencil::leg(MU, fwd)];
        if let Some(h) = Self::arrive(forward, MU, fwd, input, face) {
            sink_spinor(hops, &h, gamma, Sign::Minus);
        }
        let face = received[Stencil::leg(MU, bwd)];
        if let Some(h) = Self::arrive(backward, MU, bwd, input, face) {
            sink_spinor(hops, &h, gamma, Sign::Plus);
        }
    }

    /// Send the faces of `inp` that the neighbouring processes need and return what
    /// arrived, by face. Does nothing on a single process.
    fn exchange<'w, R: LinkRepr, SI: Sites>(
        &self,
        work: &'w mut WilsonWork<T>,
        geo: &Geometry,
        inp: &Field<T, Spinor, SI>,
        pre: bool,
        only: Option<(usize, Direction)>,
    ) -> [&'w [T]; FACES] {
        match self.start_exchange::<R, SI>(work, geo, inp, pre, only) {
            Some(exchange) => finish_exchange(exchange),
            None => [&[]; FACES],
        }
    }

    /// Fill the faces of `inp` that the neighbouring processes need and start sending
    /// them. `None` on a single process.
    ///
    /// Through a forward face goes what the neighbour's backward legs need: this side
    /// holds the link, so the spinor is fully transported here. Through a backward face
    /// goes the projection only; the neighbour holds the link of its forward legs.
    ///
    /// `only` restricts the packing to what the legs in one direction need.
    fn start_exchange<'w, R: LinkRepr, SI: Sites>(
        &self,
        work: &'w mut WilsonWork<T>,
        geo: &Geometry,
        inp: &Field<T, Spinor, SI>,
        pre: bool,
        only: Option<(usize, Direction)>,
    ) -> Option<&'w mut dyn HaloExchange<T>> {
        if !geo.stencil.has_buffers() {
            return None;
        }
        let copies = inp.copies();
        let exchange = work.exchange(geo.id, copies, &self.lat, || {
            geo.stencil.halo_spec(copies * HALF_SPINOR_REALS)
        });
        for mu in 0..ND {
            for send in [Direction::Forward, Direction::Backward] {
                // What leaves through a face serves the legs pointing the other way.
                let serves = match send {
                    Direction::Forward => Direction::Backward,
                    Direction::Backward => Direction::Forward,
                };
                let sites = geo.stencil.send_sites(mu, send);
                if sites.is_empty() || only.is_some_and(|leg| leg != (mu, serves)) {
                    continue;
                }
                let (sign, link_use) = leg_kind(serves);
                let phase = if geo.stencil.sends_across_boundary(mu, send) {
                    self.boundary_phase(mu, serves)
                } else {
                    None
                };
                let buffer = exchange.send_buffer(Stencil::leg(mu, send));
                for_sites(buffer, copies * HALF_SPINOR_REALS, |k, o| {
                    let site = sites[k] as usize;
                    let link = match send {
                        Direction::Forward => Some(self.link::<R>(mu, geo.inp_site(site))),
                        Direction::Backward => None,
                    };
                    let o = o.as_chunks_mut::<HALF_SPINOR_REALS>().0;
                    for (r, o) in o.iter_mut().enumerate() {
                        let psi = Self::input(inp, site, r, pre);
                        let h = match link {
                            Some(link) => {
                                let link = R::load::<T>(link);
                                transport(&G::GAMMA[mu], sign, &psi, &link, link_use, phase)
                            }
                            None => project(&G::GAMMA[mu], sign, &psi),
                        };
                        store_half::<T>(&h, o);
                    }
                });
            }
        }
        exchange.start();
        Some(exchange)
    }

    /// Spinor `r` at one site of the input, with γ5 applied if `pre` is set.
    #[inline(always)]
    fn input<SI: Sites>(
        inp: &Field<T, Spinor, SI>,
        site: usize,
        r: usize,
        pre: bool,
    ) -> SiteSpinor<T::Compute> {
        let psi = load_spinor::<T>(stored_spinor(
            &inp.site(0, site)[r * SPINOR_REALS..(r + 1) * SPINOR_REALS],
        ));
        if pre {
            G::GAMMA5.apply(&psi)
        } else {
            psi
        }
    }

    /// The link U_mu at a full-lattice site.
    #[inline(always)]
    fn link<R: LinkRepr>(&self, mu: usize, full_site: usize) -> &R::Stored<T> {
        &R::of(&self.state.links)[mu * self.lat.num_sites() + full_site]
    }

    fn check<SO: Sites, SI: Sites>(
        &self,
        geo: &Geometry,
        out: &Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
    ) {
        assert_eq!(
            out.num_sites(),
            geo.stencil.num_sites(),
            "output on the wrong sites"
        );
        assert!(out.outer() == 1 && inp.outer() == 1);
        assert_eq!(
            out.copies(),
            inp.copies(),
            "out and inp hold different numbers of spinors"
        );
    }

    /// `out = B inp` site by site, where B is the diagonal block or its inverse. `full_site`
    /// gives the full-lattice index of each site of the fields.
    fn site_block<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        inp: &Field<T, Spinor, S>,
        full_site: impl Fn(usize) -> usize + Sync,
        inverse: bool,
    ) {
        assert_eq!(out.as_slice().len(), inp.as_slice().len());
        let copies = inp.copies();
        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            let site = full_site(x);
            let o = o.as_chunks_mut::<SPINOR_REALS>().0;
            for (o, s) in o
                .iter_mut()
                .zip(inp.site(0, x).as_chunks::<SPINOR_REALS>().0)
            {
                let phi = load_spinor::<T>(s);
                let result = if inverse {
                    self.diag.apply_inv(site, &phi)
                } else {
                    self.diag.apply(site, &phi)
                };
                store_spinor::<T>(&result, o);
            }
        });
    }

    /// One fused sweep: `out = γ5^post ( diagonal φ - κ Σ hops φ )` with `φ = γ5^pre inp`.
    /// The full operator and the even-odd hopping blocks are both this, on different
    /// geometries. The last argument is (pre, post).
    fn sweep<SO: Sites, SI: Sites>(
        &self,
        work: &mut WilsonWork<T>,
        geo: &Geometry,
        out: &mut Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
        diagonal: Diagonal,
        gamma5: (bool, bool),
    ) {
        match self.state.links {
            LinkStore::Full(_) => {
                self.sweep_with::<FullLinks, SO, SI>(work, geo, out, inp, diagonal, gamma5)
            }
            LinkStore::TwoRows(_) => {
                self.sweep_with::<TwoRowLinks, SO, SI>(work, geo, out, inp, diagonal, gamma5)
            }
            LinkStore::Unset => panic!("set_gauge must be called before the operator is applied"),
        }
    }

    /// `sweep` for one representation of the links.
    fn sweep_with<R: LinkRepr, SO: Sites, SI: Sites>(
        &self,
        work: &mut WilsonWork<T>,
        geo: &Geometry,
        out: &mut Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
        diagonal: Diagonal,
        gamma5: (bool, bool),
    ) {
        self.check(geo, out, inp);
        let nothing = [&[][..]; FACES];
        let pass = |received, part| Pass { received, part };
        match self.start_exchange::<R, SI>(work, geo, inp, gamma5.0, None) {
            None => {
                let all = pass(&nothing, Part::All);
                self.sweep_part::<R, SO, SI>(geo, out, inp, diagonal, gamma5, all);
            }
            // The sites that take nothing from the faces while the faces travel.
            Some(exchange) => {
                let interior = pass(&nothing, Part::Interior);
                self.sweep_part::<R, SO, SI>(geo, out, inp, diagonal, gamma5, interior);
                let received = finish_exchange(exchange);
                let surface = pass(&received, Part::Surface);
                self.sweep_part::<R, SO, SI>(geo, out, inp, diagonal, gamma5, surface);
            }
        }
    }

    /// The sites of `part` of a sweep; the other sites of `out` are not touched.
    fn sweep_part<R: LinkRepr, SO: Sites, SI: Sites>(
        &self,
        geo: &Geometry,
        out: &mut Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
        diagonal: Diagonal,
        (pre, post): (bool, bool),
        pass: Pass<'_, T>,
    ) {
        let Pass { received, part } = pass;
        let kappa: T::Compute = Number::from_f64(self.params.kappa);
        let copies = inp.copies();
        let fixed = self.fixed::<R>();
        let spinors: &[[T; SPINOR_REALS]] = inp.slab(0).as_chunks().0;
        let (fwd, bwd) = (Direction::Forward, Direction::Backward);

        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            if part.skips(&geo.surface, x) {
                return;
            }
            // Geometry and links are resolved once per site and shared by all copies.
            let forward: [_; ND] =
                core::array::from_fn(|mu| Self::resolve(geo, &fixed, x, mu, fwd));
            let backward: [_; ND] =
                core::array::from_fn(|mu| Self::resolve(geo, &fixed, x, mu, bwd));
            let x_full = geo.out_site(x);

            for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                let input = Input {
                    spinors,
                    r,
                    copies,
                    pre,
                    _gamma: PhantomData::<G>,
                };

                let mut hops = SiteSpinor::zero();
                let (f, b) = (&forward, &backward);
                Self::add_legs::<0, R>(&mut hops, &f[0], &b[0], &input, received);
                Self::add_legs::<1, R>(&mut hops, &f[1], &b[1], &input, received);
                Self::add_legs::<2, R>(&mut hops, &f[2], &b[2], &input, received);
                Self::add_legs::<3, R>(&mut hops, &f[3], &b[3], &input, received);

                let diag = match diagonal {
                    Diagonal::Present => self.diag.apply(x_full, &input.local(x)),
                    Diagonal::Absent => SiteSpinor::zero(),
                };
                let mut result = diag.sub_scaled(kappa, &hops);
                if post {
                    result = G::GAMMA5.apply(&result);
                }
                store_spinor::<T>(&result, o);
            }
        });
    }

    /// Add one hopping term to `out`. Shared by `hop_forward` and `hop_backward`.
    fn hop_term(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        dir: Direction,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        match self.state.links {
            LinkStore::Full(_) => self.hop_term_with::<FullLinks>(work, mu, dir, out, inp),
            LinkStore::TwoRows(_) => self.hop_term_with::<TwoRowLinks>(work, mu, dir, out, inp),
            LinkStore::Unset => panic!("set_gauge must be called before the operator is applied"),
        }
    }

    /// `hop_term` for one representation of the links.
    fn hop_term_with<R: LinkRepr>(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        dir: Direction,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        let geo = &self.state.full;
        self.check(geo, out, inp);
        let (sign, _) = leg_kind(dir);
        let copies = inp.copies();
        let received = self.exchange::<R, crate::Full>(work, geo, inp, false, Some((mu, dir)));
        let fixed = self.fixed::<R>();
        let spinors: &[[T; SPINOR_REALS]] = inp.slab(0).as_chunks().0;
        let face = received[Stencil::leg(mu, dir)];
        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            let hop = Self::resolve(geo, &fixed, x, mu, dir);
            for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                let input = Input {
                    spinors,
                    r,
                    copies,
                    pre: false,
                    _gamma: PhantomData::<G>,
                };
                if let Some(h) = Self::arrive(&hop, mu, dir, &input, face) {
                    let mut acc = load_spinor::<T>(o);
                    sink_spinor(&mut acc, &h, &G::GAMMA[mu], sign);
                    store_spinor::<T>(&acc, o);
                }
            }
        });
    }
}

/// Wait for an exchange that was started and return what arrived, by face.
fn finish_exchange<T>(exchange: &mut dyn HaloExchange<T>) -> [&[T]; FACES] {
    exchange.wait();
    let exchange: &dyn HaloExchange<T> = exchange;
    core::array::from_fn(|face| exchange.recv_buffer(face))
}

/// Half spinor number `slot` of a face buffer.
#[inline(always)]
fn load_half<T: Storage>(buffer: &[T], slot: usize) -> HalfSpinor<T::Compute> {
    let s: &[T; HALF_SPINOR_REALS] = buffer
        [slot * HALF_SPINOR_REALS..(slot + 1) * HALF_SPINOR_REALS]
        .try_into()
        .expect("the range has the length of a half spinor");
    let mut h = HalfSpinor([[Cplx::ZERO; NC]; 2]);
    for a in 0..2 {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            h.0[a][c] = Cplx::new(s[k].load(), s[k + 1].load());
        }
    }
    h
}

#[inline(always)]
fn store_half<T: Storage>(h: &HalfSpinor<T::Compute>, s: &mut [T; HALF_SPINOR_REALS]) {
    for a in 0..2 {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            s[k] = T::store(h.0[a][c].re);
            s[k + 1] = T::store(h.0[a][c].im);
        }
    }
}

/// A slice that holds exactly one stored spinor, as an array.
#[inline(always)]
fn stored_spinor<T>(s: &[T]) -> &[T; SPINOR_REALS] {
    s.try_into().expect("a spinor has 24 reals")
}

#[inline(always)]
fn load_spinor<T: Storage>(s: &[T; SPINOR_REALS]) -> SiteSpinor<T::Compute> {
    let mut psi = SiteSpinor::zero();
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            psi.0[a][c] = Cplx::new(s[k].load(), s[k + 1].load());
        }
    }
    psi
}

#[inline(always)]
fn store_spinor<T: Storage>(p: &SiteSpinor<T::Compute>, s: &mut [T]) {
    let s: &mut [T; SPINOR_REALS] = s.try_into().expect("a spinor has 24 reals");
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            s[k] = T::store(p.0[a][c].re);
            s[k + 1] = T::store(p.0[a][c].im);
        }
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> FermionOp<Field<T, Spinor>> for WilsonOp<T, G, D> {
    type Work = WilsonWork<T>;

    /// The hopping term and the unit diagonal are counted as Bridge++ counts its Wilson
    /// operator in the Dirac representation: `Nc Nd (4 + 6 (4 Nc + 2) + 2 (4 Nc + 1))`,
    /// 1368 for three colours. Six of the eight legs have an off-diagonal gamma matrix
    /// and two a diagonal one, which needs no additions to project.
    fn flops_per_site(&self) -> usize {
        NC * NS * (4 + 6 * (4 * NC + 2) + 2 * (4 * NC + 1)) + D::FLOPS_PER_SITE
    }

    fn work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.num_sites() != self.lat.num_sites() || u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        // Every link in the operator's precision, `[mu][site]` as in the gauge field.
        let convert = |x: &f64| T::store(Number::from_f64(*x));
        let stored: &[[f64; FULL_LINK_REALS]] = u.as_slice().as_chunks().0;
        let links = match self.link_format {
            LinkFormat::Full => LinkStore::Full(
                stored
                    .par_iter()
                    .with_min_len(LINKS_PER_TASK)
                    .map(|link| core::array::from_fn(|k| convert(&link[k])))
                    .collect(),
            ),
            LinkFormat::TwoRows => {
                let n = self.lat.num_sites();
                for (i, link) in stored.iter().enumerate() {
                    let deviation = su3_deviation(&ColorMatrix::from_reals(link));
                    if deviation.is_nan() || deviation > SU3_TOLERANCE {
                        return Err(GaugeError::NotSpecialUnitary {
                            mu: i / n,
                            site: i % n,
                            deviation,
                        });
                    }
                }
                LinkStore::TwoRows(
                    stored
                        .par_iter()
                        .with_min_len(LINKS_PER_TASK)
                        .map(|link| core::array::from_fn(|k| convert(&link[k])))
                        .collect(),
                )
            }
        };
        self.diag.set_gauge(&self.lat, u)?;
        self.state.links = links;
        Ok(())
    }

    fn mult(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
        variant: Variant,
    ) {
        // γ5 applied to the input (pre) and to the result (post) of the same sweep.
        let (pre, post) = match variant {
            Variant::D => (false, false),
            Variant::Gamma5D => (false, true),
            Variant::Dagger => (true, true),
        };
        self.sweep(
            work,
            &self.state.full,
            out,
            inp,
            Diagonal::Present,
            (pre, post),
        );
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> HopTerms<Field<T, Spinor>> for WilsonOp<T, G, D> {
    fn hop_forward(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.hop_term(work, mu, Direction::Forward, out, inp);
    }

    fn hop_backward(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.hop_term(work, mu, Direction::Backward, out, inp);
    }

    fn mult_diag(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(&self.state.full, out, inp);
        let copies = inp.copies();
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            let term = self
                .diag
                .apply(i / copies, &load_spinor::<T>(stored_spinor(s)));
            let sum = load_spinor::<T>(stored_spinor(o)).add(&term);
            store_spinor::<T>(&sum, o);
        });
    }

    fn mult_gamma5(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(&self.state.full, out, inp);
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            store_spinor::<T>(&G::GAMMA5.apply(&load_spinor::<T>(stored_spinor(s))), o);
        });
    }

    fn hop_coefficient(&self) -> f64 {
        -self.params.kappa
    }
}

/// With M = [[T_ee, M_eo], [M_oe, T_oo]], T the site-diagonal block and
/// M_eo = -κ (hops from odd to even sites). M† = γ5 M γ5 and γ5 is site-diagonal, so the
/// blocks of M† are γ5 (block of M) γ5; T is Hermitian and commutes with γ5.
impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> EvenOdd<Field<T, Spinor, Half>>
    for WilsonOp<T, G, D>
{
    type Work = WilsonWork<T>;

    fn even_odd_work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn hop(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        to: Parity,
        dagger: Dagger,
    ) {
        let eo = self
            .state
            .even_odd
            .as_ref()
            .expect("Wilson: even-odd blocks need a lattice with even extents");
        assert_eq!(out.parity(), to, "output field has the wrong parity");
        assert_eq!(inp.parity(), to.flip(), "input field has the wrong parity");
        let g5 = dagger == Dagger::Yes;
        self.sweep(
            work,
            &eo.to[to.index()],
            out,
            inp,
            Diagonal::Absent,
            (g5, g5),
        );
    }

    fn diag(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let eo = self
                .state
                .even_odd
                .as_ref()
                .expect("Wilson: even-odd needs even extents");
            let geo = &eo.to[parity.index()];
            self.site_block(out, inp, |h| geo.out_site(h), false);
        }
    }

    fn diag_inv(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let eo = self
                .state
                .even_odd
                .as_ref()
                .expect("Wilson: even-odd needs even extents");
            let geo = &eo.to[parity.index()];
            self.site_block(out, inp, |h| geo.out_site(h), true);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Fields in blocks of W sites.

/// Which blocks a sweep writes, where their neighbours are, and how both map to the
/// numbering of all blocks, which the links use. The counterpart of `Geometry`.
struct BlockGeometry<const W: usize> {
    /// Distinguishes the geometries of one operator.
    id: usize,
    stencil: BlockStencil<W>,
    /// Number among all blocks of each output block; `None` when the output is all blocks.
    out_full: Option<Vec<u32>>,
    /// The same for the blocks of the input.
    inp_full: Option<Vec<u32>>,
    /// For each output block, whether any lane of any of its legs comes from another
    /// process. Empty on a single process.
    surface: Vec<bool>,
}

impl<const W: usize> BlockGeometry<W> {
    fn new(
        id: usize,
        stencil: BlockStencil<W>,
        out_full: Option<Vec<u32>>,
        inp_full: Option<Vec<u32>>,
    ) -> Self {
        let from_buffer =
            |block: usize| (0..Stencil::LEGS).any(|leg| stencil.received(block, leg).is_some());
        let surface = if stencil.has_buffers() {
            (0..stencil.num_blocks()).map(from_buffer).collect()
        } else {
            Vec::new()
        };
        Self {
            id,
            stencil,
            out_full,
            inp_full,
            surface,
        }
    }

    fn full(lat: &Lattice, boundary: [Boundary; ND]) -> Option<Self> {
        let stencil = BlockStencil::new(lat, boundary)?;
        Some(Self::new(0, stencil, None, None))
    }

    /// Output on the blocks of parity `to`, input on the other parity.
    fn parity(
        lat: &Lattice,
        boundary: [Boundary; ND],
        index: &EvenOddIndex,
        to: Parity,
    ) -> Option<Self> {
        Some(Self::new(
            1 + to.index(),
            BlockStencil::new_parity(lat, boundary, index, to)?,
            Some(index.full_sites(to).to_vec()),
            Some(index.full_sites(to.flip()).to_vec()),
        ))
    }

    #[inline(always)]
    fn out_block(&self, i: usize) -> usize {
        self.out_full.as_ref().map_or(i, |m| m[i] as usize)
    }

    #[inline(always)]
    fn inp_block(&self, i: usize) -> usize {
        self.inp_full.as_ref().map_or(i, |m| m[i] as usize)
    }
}

struct BlockEvenOdd<const W: usize> {
    /// The even-odd decomposition of the lattice of blocks.
    index: Arc<EvenOddIndex>,
    /// Indexed by the parity of the output.
    to: [BlockGeometry<W>; 2],
}

/// What the operator keeps for fields in blocks of `W` sites.
pub struct BlockState<T: Storage, const W: usize> {
    full: BlockGeometry<W>,
    /// Present when every extent of the lattice of blocks is even: then all lanes of a
    /// block have the same parity, and a field on one parity is a field on half the blocks.
    even_odd: Option<BlockEvenOdd<W>>,
    /// The lattice site of every lane of every block, for the site-diagonal term. Empty
    /// when that term is the identity.
    sites: Vec<[u32; W]>,
    /// `[mu * num_blocks + block]`: the link of every lane of the block. Empty until the
    /// gauge field is set.
    links: Vec<BlockLink<T, W>>,
}

/// The stored reals of one spinor, and of one link, for the `W` lanes of a block.
type BlockSpinor<T, const W: usize> = [[T; W]; SPINOR_REALS];
type BlockLink<T, const W: usize> = [[T; W]; FULL_LINK_REALS];

/// The lane type the sweep computes with.
type LaneType<T, const W: usize> = <<T as Storage>::Compute as LaneOf<W>>::Lane;

/// A field of spinors in blocks of `W` sites.
type BlockField<T, S, const W: usize> = Field<T, Spinor, S, Aosoa<W>>;

#[inline(always)]
fn load_lanes<T: Storage, const W: usize>(s: &[T; W]) -> LaneType<T, W>
where
    T::Compute: LaneOf<W>,
{
    Lane::from_array(s.map(Storage::load))
}

#[inline(always)]
fn load_block_spinor<T: Storage, const W: usize>(
    s: &BlockSpinor<T, W>,
) -> SiteSpinor<LaneType<T, W>>
where
    T::Compute: LaneOf<W>,
{
    let mut psi = SiteSpinor::zero();
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            psi.0[a][c] = Cplx::new(load_lanes::<T, W>(&s[k]), load_lanes::<T, W>(&s[k + 1]));
        }
    }
    psi
}

#[inline(always)]
fn store_block_spinor<T: Storage, const W: usize>(
    p: &SiteSpinor<LaneType<T, W>>,
    s: &mut BlockSpinor<T, W>,
) where
    T::Compute: LaneOf<W>,
{
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            s[k] = p.0[a][c].re.to_array().map(T::store);
            s[k + 1] = p.0[a][c].im.to_array().map(T::store);
        }
    }
}

#[inline(always)]
fn load_block_link<T: Storage, const W: usize>(s: &BlockLink<T, W>) -> ColorMatrix<LaneType<T, W>>
where
    T::Compute: LaneOf<W>,
{
    let mut m = ColorMatrix::zero();
    for a in 0..NC {
        for b in 0..NC {
            let k = 2 * (a * NC + b);
            m.0[a][b] = Cplx::new(load_lanes::<T, W>(&s[k]), load_lanes::<T, W>(&s[k + 1]));
        }
    }
    m
}

/// One lane of a stored spinor, as the sweep over sites would load that site.
#[inline(always)]
fn lane_spinor<T: Storage, const W: usize>(
    s: &BlockSpinor<T, W>,
    lane: usize,
) -> SiteSpinor<T::Compute> {
    let mut psi = SiteSpinor::zero();
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            psi.0[a][c] = Cplx::new(s[k][lane].load(), s[k + 1][lane].load());
        }
    }
    psi
}

/// One lane of a stored link.
#[inline(always)]
fn lane_link<T: Storage, const W: usize>(
    s: &BlockLink<T, W>,
    lane: usize,
) -> ColorMatrix<T::Compute> {
    let mut m = ColorMatrix::zero();
    for a in 0..NC {
        for b in 0..NC {
            let k = 2 * (a * NC + b);
            m.0[a][b] = Cplx::new(s[k][lane].load(), s[k + 1][lane].load());
        }
    }
    m
}

/// A field in blocks as a slice of stored spinors, `copies` of them per block.
fn block_spinors<T: Storage, S: Sites, const W: usize>(
    f: &BlockField<T, S, W>,
) -> &[BlockSpinor<T, W>] {
    f.slab(0).as_chunks::<W>().0.as_chunks().0
}

/// Reads spinor `r` of the blocks of an input field, with γ5 applied first if `pre` is set.
/// A struct with a method, not a closure, so that the read is expanded in place.
struct BlockInput<'a, T: Storage, G: GammaRepre, const W: usize> {
    spinors: &'a [BlockSpinor<T, W>],
    r: usize,
    /// Spinors per site.
    copies: usize,
    pre: bool,
    _gamma: PhantomData<G>,
}

impl<T: Storage, G: GammaRepre, const W: usize> BlockInput<'_, T, G, W>
where
    T::Compute: LaneOf<W>,
{
    #[inline(always)]
    fn at(&self, block: usize) -> SiteSpinor<LaneType<T, W>> {
        let psi = load_block_spinor::<T, W>(&self.spinors[block * self.copies + self.r]);
        if self.pre {
            G::GAMMA5.apply(&psi)
        } else {
            psi
        }
    }

    /// `h` with the lanes that come from the neighbouring process replaced by what
    /// arrived: `lanes` gives, for each lane, the site's position in `buffer`.
    #[inline(always)]
    fn arrived(
        &self,
        h: &HalfSpinor<LaneType<T, W>>,
        buffer: &[T],
        lanes: &[u32; W],
    ) -> HalfSpinor<LaneType<T, W>> {
        let mut out = *h;
        for a in 0..2 {
            for c in 0..NC {
                let k = 2 * (NC * a + c);
                let (mut re, mut im) = (h.0[a][c].re.to_array(), h.0[a][c].im.to_array());
                for lane in 0..W {
                    if lanes[lane] != NOT_RECEIVED {
                        let slot = lanes[lane] as usize * self.copies + self.r;
                        re[lane] = buffer[slot * HALF_SPINOR_REALS + k].load();
                        im[lane] = buffer[slot * HALF_SPINOR_REALS + k + 1].load();
                    }
                }
                out.0[a][c] = Cplx::new(Lane::from_array(re), Lane::from_array(im));
            }
        }
        out
    }
}

/// The treatment of the lanes of a hop that leaves the sub-lattice, in the lane type.
struct EdgeLanes<V, const W: usize> {
    from: [usize; W],
    /// The boundary factor of each lane, in the order of the lanes of the block computed.
    factor: Cplx<V>,
    /// The same factors in the order of the lanes of the neighbouring block.
    factor_at_source: Cplx<V>,
}

impl<V: Lane<W>, const W: usize> EdgeLanes<V, W> {
    fn new(edge: &Edge<W>) -> Self {
        let scalar = |x: f64| <V::Scalar as Number>::from_f64(x);
        let lanes = |values: [(f64, f64); W]| {
            Cplx::new(
                V::from_array(values.map(|(re, _)| scalar(re))),
                V::from_array(values.map(|(_, im)| scalar(im))),
            )
        };
        let mut at_source = [(1.0, 0.0); W];
        for lane in 0..W {
            at_source[edge.from[lane]] = edge.factor[lane];
        }
        Self {
            from: edge.from,
            factor: lanes(edge.factor),
            factor_at_source: lanes(at_source),
        }
    }

    /// Bring the lanes of a half spinor from the order of the neighbouring block into the
    /// order of the block computed.
    #[inline(always)]
    fn rotate(&self, h: &HalfSpinor<V>) -> HalfSpinor<V> {
        let mut out = *h;
        for a in 0..2 {
            for c in 0..NC {
                let z = h.0[a][c];
                out.0[a][c] = Cplx::new(z.re.permute(&self.from), z.im.permute(&self.from));
            }
        }
        out
    }
}

impl<T: Storage, G: GammaRepre, const W: usize> WilsonOp<T, G, UnitDiag, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    /// `None` if the lattice cannot be cut into `W` sub-lattices of equal shape.
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Option<Self> {
        Self::with_diag(lat, params, UnitDiag)
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>, const W: usize> WilsonOp<T, G, D, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    /// An operator of Wilson type with the given site-diagonal block, for fields in blocks.
    /// `None` if the lattice cannot be cut into `W` sub-lattices of equal shape.
    pub fn with_diag(lat: &Arc<Lattice>, params: WilsonParams, diag: D) -> Option<Self> {
        let full = BlockGeometry::full(lat, params.boundary)?;
        let blocks = full.stencil.blocks();
        let even_odd = match blocks.even_odd_index() {
            Some(index) => {
                let geometry = |to| BlockGeometry::parity(lat, params.boundary, &index, to);
                let to = [geometry(Parity::Even)?, geometry(Parity::Odd)?];
                Some(BlockEvenOdd {
                    index: Arc::new(index),
                    to,
                })
            }
            None => None,
        };
        let sites = if D::IS_UNIT {
            Vec::new()
        } else {
            (0..blocks.num_blocks())
                .map(|block| {
                    core::array::from_fn(|lane| {
                        u32::try_from(blocks.site(block, lane))
                            .expect("the stencil has checked that sites fit in 31 bits")
                    })
                })
                .collect()
        };
        Some(Self {
            params,
            lat: lat.clone(),
            state: BlockState {
                full,
                even_odd,
                sites,
                links: Vec::new(),
            },
            link_format: LinkFormat::Full,
            diag,
            _gamma: PhantomData,
        })
    }

    pub fn params(&self) -> &WilsonParams {
        &self.params
    }

    /// The even-odd numbering this operator uses, if the lattice allows one. It numbers
    /// blocks, not sites: it is the one to hand to `solve_even_odd`, `extract_parity` and
    /// `insert_parity` together with fields in blocks.
    pub fn even_odd_index(&self) -> Option<&Arc<EvenOddIndex>> {
        self.state.even_odd.as_ref().map(|eo| &eo.index)
    }

    /// Add the forward and the backward leg of direction `MU` of one block to `hops`.
    ///
    /// The steps are those of the sweep over sites, in the same order, so that every lane
    /// gets the bits its site gets there. Where the hop leaves the sub-lattice, the lanes of
    /// the neighbouring block are rotated into place and each lane gets its boundary factor.
    /// The backward leg multiplies by the neighbouring block's link first, in that block's
    /// order of lanes, and rotates the product: the link then never has to be rotated.
    ///
    /// Lanes whose hop leaves the local lattice towards another process are computed like
    /// the others, from the periodic image on this process, and then replaced by what
    /// arrived in `received`: a projected spinor on the forward leg, where this side holds
    /// the link, a transported one on the backward leg.
    #[inline(always)]
    fn add_block_legs<const MU: usize>(
        &self,
        hops: &mut SiteSpinor<LaneType<T, W>>,
        geo: &BlockGeometry<W>,
        block: usize,
        input: &BlockInput<'_, T, G, W>,
        edges: &[EdgeLanes<LaneType<T, W>, W>; FACES],
        received: &[&[T]; FACES],
    ) {
        let gamma = &G::GAMMA[MU];
        let (stencil, links) = (&geo.stencil, &self.state.links);
        let num_blocks = stencil.blocks().num_blocks();

        let leg = Stencil::leg(MU, Direction::Forward);
        let (from, edge) = stencil.source(block, leg);
        let mut h = project(gamma, Sign::Minus, &input.at(from));
        if edge.is_some() {
            h = edges[leg].rotate(&h);
            if let Some(lanes) = stencil.received(block, leg) {
                h = input.arrived(&h, received[leg], lanes);
            }
            h = h.scale_cplx(edges[leg].factor);
        }
        let link = load_block_link::<T, W>(&links[MU * num_blocks + geo.out_block(block)]);
        let h = transport_half(&h, &link, LinkUse::Direct, None);
        sink_spinor(hops, &h, gamma, Sign::Minus);

        let leg = Stencil::leg(MU, Direction::Backward);
        let (from, edge) = stencil.source(block, leg);
        let mut h = project(gamma, Sign::Plus, &input.at(from));
        if edge.is_some() {
            h = h.scale_cplx(edges[leg].factor_at_source);
        }
        let link = load_block_link::<T, W>(&links[MU * num_blocks + geo.inp_block(from)]);
        let mut h = transport_half(&h, &link, LinkUse::Adjoint, None);
        if edge.is_some() {
            h = edges[leg].rotate(&h);
            if let Some(lanes) = stencil.received(block, leg) {
                h = input.arrived(&h, received[leg], lanes);
            }
        }
        sink_spinor(hops, &h, gamma, Sign::Plus);
    }

    /// The site-diagonal block, or its inverse, applied to the lanes of block `full_block`
    /// one at a time. The block is written for one site, and this keeps it written once;
    /// it costs a pass through the stack per lane.
    #[inline(always)]
    fn diag_lanes(
        &self,
        full_block: usize,
        phi: &SiteSpinor<LaneType<T, W>>,
        inverse: bool,
    ) -> SiteSpinor<LaneType<T, W>> {
        let sites = &self.state.sites[full_block];
        let zero = [[[<T::Compute as Number>::from_f64(0.0); W]; NC]; NS];
        let (mut re, mut im) = (zero, zero);
        for a in 0..NS {
            for c in 0..NC {
                re[a][c] = phi.0[a][c].re.to_array();
                im[a][c] = phi.0[a][c].im.to_array();
            }
        }
        for lane in 0..W {
            let mut one = SiteSpinor::zero();
            for a in 0..NS {
                for c in 0..NC {
                    one.0[a][c] = Cplx::new(re[a][c][lane], im[a][c][lane]);
                }
            }
            let site = sites[lane] as usize;
            let result = if inverse {
                self.diag.apply_inv(site, &one)
            } else {
                self.diag.apply(site, &one)
            };
            for a in 0..NS {
                for c in 0..NC {
                    re[a][c][lane] = result.0[a][c].re;
                    im[a][c][lane] = result.0[a][c].im;
                }
            }
        }
        let mut out = SiteSpinor::zero();
        for a in 0..NS {
            for c in 0..NC {
                out.0[a][c] = Cplx::new(Lane::from_array(re[a][c]), Lane::from_array(im[a][c]));
            }
        }
        out
    }

    /// `start_exchange` for fields in blocks: the same buffers with the same contents,
    /// filled one site at a time from the lanes `BlockStencil::send_lanes` names.
    fn start_block_exchange<'w, SI: Sites>(
        &self,
        work: &'w mut WilsonWork<T>,
        geo: &BlockGeometry<W>,
        inp: &BlockField<T, SI, W>,
        pre: bool,
    ) -> Option<&'w mut dyn HaloExchange<T>> {
        let stencil = &geo.stencil;
        if !stencil.has_buffers() {
            return None;
        }
        let copies = inp.copies();
        let num_blocks = stencil.blocks().num_blocks();
        let spinors = block_spinors(inp);
        let exchange = work.exchange(geo.id, copies, &self.lat, || {
            stencil.halo_spec(copies * HALF_SPINOR_REALS)
        });
        for mu in 0..ND {
            for send in [Direction::Forward, Direction::Backward] {
                // What leaves through a face serves the legs pointing the other way.
                let serves = match send {
                    Direction::Forward => Direction::Backward,
                    Direction::Backward => Direction::Forward,
                };
                let face = Stencil::leg(mu, send);
                let lanes = stencil.send_lanes(face);
                if lanes.is_empty() {
                    continue;
                }
                let (sign, link_use) = leg_kind(serves);
                let phase = if stencil.sends_across_boundary(face) {
                    boundary_phase::<T::Compute>(self.params.boundary[mu], serves)
                } else {
                    None
                };
                let buffer = exchange.send_buffer(face);
                for_sites(buffer, copies * HALF_SPINOR_REALS, |k, o| {
                    let (block, lane) = (lanes[k].0 as usize, lanes[k].1 as usize);
                    let link = match send {
                        Direction::Forward => Some(lane_link::<T, W>(
                            &self.state.links[mu * num_blocks + geo.inp_block(block)],
                            lane,
                        )),
                        Direction::Backward => None,
                    };
                    let o = o.as_chunks_mut::<HALF_SPINOR_REALS>().0;
                    for (r, o) in o.iter_mut().enumerate() {
                        let mut psi = lane_spinor::<T, W>(&spinors[block * copies + r], lane);
                        if pre {
                            psi = G::GAMMA5.apply(&psi);
                        }
                        let h = match &link {
                            Some(link) => {
                                transport(&G::GAMMA[mu], sign, &psi, link, link_use, phase)
                            }
                            None => project(&G::GAMMA[mu], sign, &psi),
                        };
                        store_half::<T>(&h, o);
                    }
                });
            }
        }
        exchange.start();
        Some(exchange)
    }

    /// `out = γ5^post ( diagonal φ - κ Σ hops φ )` with `φ = γ5^pre inp`, block by block:
    /// `sweep` for fields in blocks.
    fn block_sweep<SO: Sites, SI: Sites>(
        &self,
        work: &mut WilsonWork<T>,
        geo: &BlockGeometry<W>,
        out: &mut BlockField<T, SO, W>,
        inp: &BlockField<T, SI, W>,
        diagonal: Diagonal,
        gamma5: (bool, bool),
    ) {
        let nothing = [&[][..]; FACES];
        let pass = |received, part| Pass { received, part };
        match self.start_block_exchange(work, geo, inp, gamma5.0) {
            None => {
                let all = pass(&nothing, Part::All);
                self.block_sweep_part(geo, out, inp, diagonal, gamma5, all);
            }
            // The blocks that take nothing from the faces while the faces travel.
            Some(exchange) => {
                let interior = pass(&nothing, Part::Interior);
                self.block_sweep_part(geo, out, inp, diagonal, gamma5, interior);
                let received = finish_exchange(exchange);
                let surface = pass(&received, Part::Surface);
                self.block_sweep_part(geo, out, inp, diagonal, gamma5, surface);
            }
        }
    }

    /// The blocks of `part` of a sweep; the other blocks of `out` are not touched.
    fn block_sweep_part<SO: Sites, SI: Sites>(
        &self,
        geo: &BlockGeometry<W>,
        out: &mut BlockField<T, SO, W>,
        inp: &BlockField<T, SI, W>,
        diagonal: Diagonal,
        (pre, post): (bool, bool),
        pass: Pass<'_, T>,
    ) {
        let Pass { received, part } = pass;
        let stencil = &geo.stencil;
        assert_eq!(
            self.state.links.len(),
            ND * stencil.blocks().num_blocks(),
            "set_gauge must be called before the operator is applied"
        );
        assert_eq!(
            out.num_blocks(),
            stencil.num_blocks(),
            "output on the wrong sites"
        );
        assert_eq!(inp.num_blocks(), out.num_blocks());
        assert!(out.outer() == 1 && inp.outer() == 1);
        assert_eq!(out.copies(), inp.copies());
        let copies = inp.copies();
        let kappa = <LaneType<T, W> as Number>::from_f64(self.params.kappa);
        let edges: [EdgeLanes<LaneType<T, W>, W>; FACES] =
            core::array::from_fn(|leg| EdgeLanes::new(stencil.edge(leg)));
        let spinors = block_spinors(inp);

        for_sites(out.slab_mut(0), copies * SPINOR_REALS * W, |block, o| {
            if part.skips(&geo.surface, block) {
                return;
            }
            let o: &mut [BlockSpinor<T, W>] = o.as_chunks_mut::<W>().0.as_chunks_mut().0;
            for (r, o) in o.iter_mut().enumerate() {
                let input = BlockInput {
                    spinors,
                    r,
                    copies,
                    pre,
                    _gamma: PhantomData::<G>,
                };
                let mut hops = SiteSpinor::zero();
                self.add_block_legs::<0>(&mut hops, geo, block, &input, &edges, received);
                self.add_block_legs::<1>(&mut hops, geo, block, &input, &edges, received);
                self.add_block_legs::<2>(&mut hops, geo, block, &input, &edges, received);
                self.add_block_legs::<3>(&mut hops, geo, block, &input, &edges, received);
                let diag = match diagonal {
                    Diagonal::Present if D::IS_UNIT => input.at(block),
                    Diagonal::Present => {
                        self.diag_lanes(geo.out_block(block), &input.at(block), false)
                    }
                    Diagonal::Absent => SiteSpinor::zero(),
                };
                let mut result = diag.sub_scaled(kappa, &hops);
                if post {
                    result = G::GAMMA5.apply(&result);
                }
                store_block_spinor::<T, W>(&result, o);
            }
        });
    }

    /// `out = B inp` block by block, where B is the diagonal term or its inverse.
    /// `full_block` gives the number among all blocks of each block of the fields.
    fn block_diag<S: Sites>(
        &self,
        out: &mut BlockField<T, S, W>,
        inp: &BlockField<T, S, W>,
        full_block: impl Fn(usize) -> usize + Sync,
        inverse: bool,
    ) {
        assert_eq!(out.as_slice().len(), inp.as_slice().len());
        let copies = inp.copies();
        let spinors = block_spinors(inp);
        for_sites(out.slab_mut(0), copies * SPINOR_REALS * W, |block, o| {
            let o: &mut [BlockSpinor<T, W>] = o.as_chunks_mut::<W>().0.as_chunks_mut().0;
            for (r, o) in o.iter_mut().enumerate() {
                let phi = load_block_spinor::<T, W>(&spinors[block * copies + r]);
                store_block_spinor::<T, W>(&self.diag_lanes(full_block(block), &phi, inverse), o);
            }
        });
    }

    fn parity_geometry(&self, parity: Parity) -> &BlockGeometry<W> {
        let eo = self.state.even_odd.as_ref().expect(
            "Wilson: even-odd blocks need a lattice of blocks with even extents; \
             check `even_odd_index` first",
        );
        &eo.to[parity.index()]
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>, const W: usize> FermionOp<BlockField<T, Full, W>>
    for WilsonOp<T, G, D, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    type Work = WilsonWork<T>;

    fn work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn flops_per_site(&self) -> usize {
        NC * NS * (4 + 6 * (4 * NC + 2) + 2 * (4 * NC + 1)) + D::FLOPS_PER_SITE
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.num_sites() != self.lat.num_sites() || u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let blocks = self.state.full.stencil.blocks();
        let num_blocks = blocks.num_blocks();
        // Filled by the threads of the pool, like the fields (`Cpu::alloc`).
        let links = (0..ND * num_blocks)
            .into_par_iter()
            .with_min_len(LINKS_PER_TASK)
            .map(|i| {
                let (mu, block) = (i / num_blocks, i % num_blocks);
                let mut link = [[T::default(); W]; FULL_LINK_REALS];
                for lane in 0..W {
                    let from = u.site(mu, blocks.site(block, lane));
                    for c in 0..FULL_LINK_REALS {
                        link[c][lane] = T::store(Number::from_f64(from[c]));
                    }
                }
                link
            })
            .collect();
        self.diag.set_gauge(&self.lat, u)?;
        self.state.links = links;
        Ok(())
    }

    fn mult(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Full, W>,
        inp: &BlockField<T, Full, W>,
        variant: Variant,
    ) {
        let gamma5 = match variant {
            Variant::D => (false, false),
            Variant::Gamma5D => (false, true),
            Variant::Dagger => (true, true),
        };
        self.block_sweep(work, &self.state.full, out, inp, Diagonal::Present, gamma5);
    }
}

/// The even-odd blocks for fields in blocks of sites; see the implementation for the usual
/// layout above. A half field holds the blocks of one parity of the lattice of blocks.
impl<T: Storage, G: GammaRepre, D: DiagTerm<T>, const W: usize> EvenOdd<BlockField<T, Half, W>>
    for WilsonOp<T, G, D, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    type Work = WilsonWork<T>;

    fn even_odd_work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn hop(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        to: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(out.parity(), to, "output field has the wrong parity");
        assert_eq!(inp.parity(), to.flip(), "input field has the wrong parity");
        let g5 = dagger == Dagger::Yes;
        self.block_sweep(
            work,
            self.parity_geometry(to),
            out,
            inp,
            Diagonal::Absent,
            (g5, g5),
        );
    }

    fn diag(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let geo = self.parity_geometry(parity);
            self.block_diag(out, inp, |h| geo.out_block(h), false);
        }
    }

    fn diag_inv(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let geo = self.parity_geometry(parity);
            self.block_diag(out, inp, |h| geo.out_block(h), true);
        }
    }
}
