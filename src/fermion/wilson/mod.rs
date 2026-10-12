//! Wilson fermions.
//!
//! D ψ(x) = ψ(x) - κ Σ_mu [ (1 - γ_mu) U_mu(x) ψ(x + mu) + (1 + γ_mu) U_mu(x - mu)^† ψ(x - mu) ]
//!
//! Provides: `mult` (fused, one sweep) for D, D† and γ5 D; the hopping terms one direction
//! at a time; the even-odd blocks (the site-diagonal blocks are the identity).
//!
//! The sweep is generic over the site-diagonal block (`DiagTerm`), so fermions that differ
//! from Wilson only within a site (clover) reuse everything here.
//! With several processes, a sweep first exchanges the faces of the input: projected half
//! spinors, with the link already applied on the side that holds it. While the exchange
//! is under way the sweep computes the sites that need nothing from it, and the others
//! after it has finished (`overlapped`). Two things make that worth doing: a thread of its
//! own waits for the exchange, since nothing travels unless somebody is inside the
//! communication library; and the two parts are lists of runs of sites (`SiteRuns`), not a
//! flag tested in two passes over all sites, which was slower than not overlapping.
//!
//! For fields in blocks of W sites (`Aosoa<W>`, the SIMD layout) there is a second sweep,
//! over blocks (`blocks.rs`; the sweep over sites is in `sites.rs`): the same kernel functions with a lane type in place
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
    for_site_runs, for_sites, for_sites_as, for_sites_then, stream_fence, streams, Aos, Aosoa,
    BlockStencil, Boundary, Direction, Edge, EvenOddIndex, Field, Full, Gauge, Half, HaloExchange,
    HaloSpec, Lattice, Layout, Parity, SiteRuns, Sites, Source, Spinor, Stencil, Storage, Vector,
    FACES, ND, NOT_RECEIVED,
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
    /// The output sites in two parts; `None` on a single process.
    parts: Option<Parts>,
}

/// The output of a sweep on several processes in two parts: the sites (or blocks) none of
/// whose legs comes from another process, and the others.
struct Parts {
    interior: SiteRuns,
    surface: SiteRuns,
}

impl Parts {
    /// `on_surface(i)`: whether element `i` of the output takes anything from another
    /// process.
    fn new(len: usize, on_surface: impl Fn(usize) -> bool) -> Self {
        Self {
            interior: SiteRuns::collect(len, |i| !on_surface(i)),
            surface: SiteRuns::collect(len, &on_surface),
        }
    }
}

/// Run the passes of a sweep around an exchange that has been started: the interior while
/// a thread of its own waits for the exchange, then the surface with what arrived. Without
/// an exchange there is one pass over everything. `pass(received, part)` computes one part
/// of the output, all of it for `None`.
///
/// The thread is started for every sweep. It spends its time inside the communication
/// library, which is where received data are copied into place.
fn overlapped<'a, T: Storage>(
    exchange: Option<&'a mut dyn HaloExchange<T>>,
    parts: Option<&Parts>,
    mut pass: impl FnMut(&[&[T]; FACES], Option<&SiteRuns>),
) {
    let nothing = [&[][..]; FACES];
    let (Some(exchange), Some(parts)) = (exchange, parts) else {
        return pass(&nothing, None);
    };
    let received: [&'a [T]; FACES] = std::thread::scope(|scope| {
        let waiting = scope.spawn(move || finish_exchange(exchange));
        pass(&nothing, Some(&parts.interior));
        waiting
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    });
    pass(&received, Some(&parts.surface));
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
        let parts = stencil
            .has_buffers()
            .then(|| Parts::new(stencil.num_sites(), from_buffer));
        Self {
            id,
            stencil,
            out_full,
            inp_full,
            parts,
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

/// A factor that acts within a site and tridiagonally in the fifth dimension:
/// `a φ_s + y (Y φ)_s`, or with Y† (`dagger`), at mass `mass`, on fields with `ls` spinors per
/// site and vector. (Y φ)_s = −P− φ_(s+1) − P+ φ_(s−1), and a term that crosses a wall comes
/// back times −m (see `fermion::mdwf`). Domain-wall fermions put (4 − M5)(b − c Y) before the
/// Wilson kernel, and its adjoint after the kernel of D†; a sweep can apply it on the way
/// (`even_odd_hop_fifth`) instead of in a pass over the field of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fifth {
    pub ls: usize,
    pub a: f64,
    pub y: f64,
    pub mass: f64,
    pub dagger: bool,
}

/// (Y φ)_s, or (Y† φ)_s, at mass `mass`, from φ_(s+1) (`above`) and φ_(s−1) (`below`), the
/// indices taken around the fifth dimension of extent `ls`.
#[inline(always)]
pub fn fifth_hop<N: Number, G: GammaRepre>(
    s: usize,
    ls: usize,
    dagger: bool,
    mass: f64,
    above: &SiteSpinor<N>,
    below: &SiteSpinor<N>,
) -> SiteSpinor<N> {
    let number = <N as Number>::from_f64;
    // The coefficient of the projected neighbour: -1, or +m across the walls.
    let across = |wall: bool| number(0.5 * if wall { mass } else { -1.0 });
    // Y takes P- from above and P+ from below; Y† the other way round.
    let (up, down) = if dagger { (1.0, -1.0) } else { (-1.0, 1.0) };
    let projected = |p: &SiteSpinor<N>, sign: f64, k: N| {
        // k (1 + sign γ5) φ
        let both = p.sub_scaled(number(-sign), &G::GAMMA5.apply(p));
        SiteSpinor::zero().sub_scaled(-k, &both)
    };
    projected(above, up, across(s == ls - 1)).add(&projected(below, down, across(s == 0)))
}

impl Fifth {
    /// The factor at s from the spinors at s − 1, s and s + 1 (around the fifth dimension).
    #[inline(always)]
    pub fn apply<N: Number, G: GammaRepre>(
        &self,
        s: usize,
        below: &SiteSpinor<N>,
        here: &SiteSpinor<N>,
        above: &SiteSpinor<N>,
    ) -> SiteSpinor<N> {
        let number = <N as Number>::from_f64;
        let result = SiteSpinor::zero().sub_scaled(number(-self.a), here);
        if self.y == 0.0 {
            return result;
        }
        let hop = fifth_hop::<N, G>(s, self.ls, self.dagger, self.mass, above, below);
        result.sub_scaled(number(-self.y), &hop)
    }

    /// The factor applied in place to the spinors of one site, `ls` per vector, each read
    /// and written through `load` and `store`. A spinor is read before it is overwritten:
    /// the first and the last of a vector are kept for the wrap-around.
    #[inline(always)]
    fn apply_in_place<N: Number, G: GammaRepre, X>(
        &self,
        spinors: &mut [X],
        load: impl Fn(&X) -> SiteSpinor<N>,
        store: impl Fn(&SiteSpinor<N>, &mut X),
    ) {
        let ls = self.ls;
        for v in spinors.chunks_exact_mut(ls) {
            let (first, mut below) = (load(&v[0]), load(&v[ls - 1]));
            for s in 0..ls {
                let here = load(&v[s]);
                let above = if s + 1 < ls { load(&v[s + 1]) } else { first };
                store(&self.apply::<N, G>(s, &below, &here, &above), &mut v[s]);
                below = here;
            }
        }
    }
}

/// What a sweep does to every spinor of its input before the hops.
#[derive(Clone, Copy)]
enum Pre {
    Plain,
    Gamma5,
    /// The factor, from the spinors s − 1, s, s + 1 of the same site.
    Fifth(Fifth),
}

impl Pre {
    fn gamma5(on: bool) -> Self {
        if on {
            Pre::Gamma5
        } else {
            Pre::Plain
        }
    }

    /// Spinor `r` of a site, `at(k)` reading spinor k of that site.
    #[inline(always)]
    fn spinor<N: Number, G: GammaRepre>(
        self,
        r: usize,
        at: impl Fn(usize) -> SiteSpinor<N>,
    ) -> SiteSpinor<N> {
        match self {
            Pre::Plain => at(r),
            Pre::Gamma5 => G::GAMMA5.apply(&at(r)),
            Pre::Fifth(f) => {
                let ls = f.ls;
                let s = r % ls;
                let base = r - s;
                f.apply::<N, G>(
                    s,
                    &at(base + (s + ls - 1) % ls),
                    &at(r),
                    &at(base + (s + 1) % ls),
                )
            }
        }
    }
}

/// What a sweep does to its result.
#[derive(Clone, Copy)]
enum Post {
    Plain,
    Gamma5,
    /// γ5, then the factor over the spinors of each site (once all of them are there).
    Gamma5Fifth(Fifth),
}

impl Post {
    fn gamma5(on: bool) -> Self {
        if on {
            Post::Gamma5
        } else {
            Post::Plain
        }
    }

    fn gamma5_first(self) -> bool {
        !matches!(self, Post::Plain)
    }
}

/// The last step of a sweep at each output site (or block): called with its number among
/// those of the output and its values, all spinors of the site, once they are written and
/// still in the cache. For what follows a hop within a site, in the same pass over the field
/// (domain-wall fermions: C⁻¹, and the difference with another field).
pub type After<'a, T> = Option<&'a (dyn Fn(usize, &mut [T]) + Sync)>;

/// Reads spinor `r` of the sites of an input field, transformed by `pre`. A struct with
/// methods rather than closures, so that the reads are expanded in place.
///
/// The field is held as a slice of stored spinors, `copies` of them per site, for the same
/// reason as the links in `Fixed`.
struct Input<'a, T: Storage, G: GammaRepre> {
    spinors: &'a [[T; SPINOR_REALS]],
    r: usize,
    /// Spinors per site.
    copies: usize,
    pre: Pre,
    _gamma: PhantomData<G>,
}

impl<T: Storage, G: GammaRepre> Input<'_, T, G> {
    #[inline(always)]
    fn local(&self, site: usize) -> SiteSpinor<T::Compute> {
        let base = site * self.copies;
        self.pre
            .spinor::<T::Compute, G>(self.r, |k| load_spinor::<T>(&self.spinors[base + k]))
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

mod blocks;
mod sites;

pub use blocks::BlockState;
