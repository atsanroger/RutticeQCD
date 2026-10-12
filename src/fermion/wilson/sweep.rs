//! What the two sweeps of the hopping term share (`sweep_sites.rs` for the usual layout,
//! `sweep_blocks.rs` for blocks of W sites): the geometry of a sweep and its two parts on
//! several processes, the legs of the stencil, what a sweep does to its input and to its
//! result (γ5, a factor in the fifth dimension, `After`), how the links are kept, and
//! loading and storing spinors.

use super::*;

/// Which sites a sweep writes, where their neighbours are, and how both map to the
/// full-lattice numbering the gauge field uses.
pub(super) struct Geometry {
    /// Distinguishes the geometries of one operator.
    pub(super) id: usize,
    pub(super) stencil: Stencil,
    /// Full-lattice index of each output site; `None` when the output is the full lattice.
    pub(super) out_full: Option<Vec<u32>>,
    /// Full-lattice index of each input site; `None` when the input is the full lattice.
    pub(super) inp_full: Option<Vec<u32>>,
    /// The output sites in two parts; `None` on a single process.
    pub(super) parts: Option<Parts>,
}

/// The output of a sweep on several processes in two parts: the sites (or blocks) none of
/// whose legs comes from another process, and the others.
pub(super) struct Parts {
    pub(super) interior: SiteRuns,
    pub(super) surface: SiteRuns,
}

impl Parts {
    /// `on_surface(i)`: whether element `i` of the output takes anything from another
    /// process.
    pub(super) fn new(len: usize, on_surface: impl Fn(usize) -> bool) -> Self {
        Self {
            interior: SiteRuns::collect(len, |i| !on_surface(i)),
            surface: SiteRuns::collect(len, &on_surface),
        }
    }
}

/// Run the passes of a sweep around an exchange that has been started: the interior while a thread
/// of its own waits for the exchange, then the surface with what arrived. `pass(received, part)`
/// computes one part of the output, all of it for `None`.
pub(super) fn overlapped<'a, T: Storage>(
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
    pub(super) fn new(
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

    pub(super) fn full(lat: &Lattice, boundary: [Boundary; ND]) -> Self {
        Self::new(0, Stencil::nearest(lat, boundary), None, None)
    }

    /// Output on the sites of parity `to`, input on the other parity.
    pub(super) fn parity(
        lat: &Lattice,
        boundary: [Boundary; ND],
        index: &EvenOddIndex,
        to: Parity,
    ) -> Self {
        Self::new(
            1 + to.index(),
            Stencil::nearest_parity(lat, boundary, index, to),
            Some(index.full_sites(to).to_vec()),
            Some(index.full_sites(to.flip()).to_vec()),
        )
    }

    #[inline(always)]
    pub(super) fn out_site(&self, i: usize) -> usize {
        self.out_full.as_ref().map_or(i, |m| m[i] as usize)
    }

    #[inline(always)]
    pub(super) fn inp_site(&self, i: usize) -> usize {
        self.inp_full.as_ref().map_or(i, |m| m[i] as usize)
    }
}

pub(super) struct EvenOddGeometry {
    pub(super) index: Arc<EvenOddIndex>,
    /// Indexed by the parity of the output.
    pub(super) to: [Geometry; 2],
}

/// Whether a sweep includes the site-diagonal block.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Diagonal {
    /// The operator's own diagonal block applied to the input at the same site.
    Present,
    /// No diagonal term: the sweep is a pure hopping block.
    Absent,
}

/// One leg of the stencil resolved for one output site: where the neighbour is and what
/// is still to be done to it.
#[derive(Clone, Copy)]
pub(super) enum Hop<'a, T: Storage, R: LinkRepr> {
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

/// A factor that acts within a site and tridiagonally in the fifth dimension: `a φ_s + y (Y φ)_s`,
/// or with Y† (`dagger`), at mass `mass`, on fields with `ls` spinors per site and vector.
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
}

/// What a sweep does to every spinor of its input before the hops.
#[derive(Clone, Copy)]
pub(super) enum Pre {
    Plain,
    Gamma5,
    /// The factor, from the spinors s − 1, s, s + 1 of the same site.
    Fifth(Fifth),
}

impl Pre {
    pub(super) fn gamma5(on: bool) -> Self {
        if on {
            Pre::Gamma5
        } else {
            Pre::Plain
        }
    }

    /// Spinor `r` of a site, `at(k)` reading spinor k of that site.
    #[inline(always)]
    pub(super) fn spinor<N: Number, G: GammaRepre>(
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
pub(super) enum Post {
    Plain,
    Gamma5,
}

impl Post {
    pub(super) fn gamma5(on: bool) -> Self {
        if on {
            Post::Gamma5
        } else {
            Post::Plain
        }
    }
}

/// The last step of a sweep at each output site (or block): called with its number among those of
/// the output and its values, all spinors of the site, once they are written and still in the
/// cache.
pub type After<'a, T> = Option<&'a (dyn Fn(usize, &mut [T]) + Sync)>;

/// Largest site (or block), in reals, that a sweep with an `After` computes in a buffer on the
/// stack before it writes it past the cache (`Storage::stream`): 64 spinors.
pub(super) const SITE_BUFFER: usize = 64 * SPINOR_REALS;

/// Reads spinor `r` of the sites of an input field, transformed by `pre`. A struct with
/// methods rather than closures, so that the reads are expanded in place.
pub(super) struct Input<'a, T: Storage, G: GammaRepre> {
    pub(super) spinors: &'a [[T; SPINOR_REALS]],
    pub(super) r: usize,
    /// Spinors per site.
    pub(super) copies: usize,
    pub(super) pre: Pre,
    pub(super) _gamma: PhantomData<G>,
}

impl<T: Storage, G: GammaRepre> Input<'_, T, G> {
    #[inline(always)]
    pub(super) fn local(&self, site: usize) -> SiteSpinor<T::Compute> {
        let base = site * self.copies;
        self.pre
            .spinor::<T::Compute, G>(self.r, |k| load_spinor::<T>(&self.spinors[base + k]))
    }

    /// Position in a face buffer of spinor `r` of received site `index`.
    #[inline(always)]
    pub(super) fn slot(&self, index: usize) -> usize {
        index * self.copies + self.r
    }
}

/// How an operator keeps the links.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinkFormat {
    /// All 18 reals of each link.
    #[default]
    Full,
    /// The first two rows, 12 reals; the third row is rebuilt as the conjugate of their cross
    /// product every time the link is used. Only for links in SU(3): `set_gauge` checks.
    TwoRows,
}

/// Largest deviation from unitarity and from unit determinant that `set_gauge` accepts
/// for `LinkFormat::TwoRows`.
pub(super) const SU3_TOLERANCE: f64 = 1e-10;

/// The links of the operator, `[mu * num_sites + site]`, in the format it was asked for.
pub(super) enum LinkStore<T> {
    Unset,
    Full(Vec<[T; FULL_LINK_REALS]>),
    TwoRows(Vec<[T; TWO_ROW_REALS]>),
}

pub(super) const FULL_LINK_REALS: usize = 2 * NC * NC;
/// Links converted per task when the operator's copy is filled. It is filled by the
/// threads of the pool so that its pages are spread over the memory nodes like those of
/// the fields (`Cpu::alloc`).
pub(super) const LINKS_PER_TASK: usize = 1024;
pub(super) const TWO_ROW_REALS: usize = 2 * 2 * NC;

/// One way of keeping a link. The sweeps are generic over it, so the choice costs nothing
/// in the site loop; it is made once per sweep from the operator's `LinkFormat`.
pub(super) trait LinkRepr: Copy + Send + Sync + 'static {
    type Stored<T: Storage>: Copy + Send + Sync;
    /// The links of a store that holds this representation.
    fn of<T: Storage>(store: &LinkStore<T>) -> &[Self::Stored<T>];
    fn load<T: Storage>(s: &Self::Stored<T>) -> ColorMatrix<T::Compute>;
}

#[derive(Clone, Copy)]
pub(super) struct FullLinks;
#[derive(Clone, Copy)]
pub(super) struct TwoRowLinks;

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
pub(super) fn su3_deviation(u: &ColorMatrix<f64>) -> f64 {
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
pub(super) struct Fixed<'a, T: Storage, R: LinkRepr> {
    pub(super) links: [&'a [R::Stored<T>]; ND],
    pub(super) phase: [Option<Cplx<T::Compute>>; FACES],
}

/// The phase a hop in direction `dir` picks up when it crosses a boundary of the global
/// lattice with this condition. A hop backward picks up the conjugate.
pub(super) fn boundary_phase<C: Number>(boundary: Boundary, dir: Direction) -> Option<Cplx<C>> {
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
pub(super) fn leg_kind(dir: Direction) -> (Sign, LinkUse) {
    match dir {
        Direction::Forward => (Sign::Minus, LinkUse::Direct),
        Direction::Backward => (Sign::Plus, LinkUse::Adjoint),
    }
}

/// Wait for an exchange that was started and return what arrived, by face.
pub(super) fn finish_exchange<T>(exchange: &mut dyn HaloExchange<T>) -> [&[T]; FACES] {
    exchange.wait();
    let exchange: &dyn HaloExchange<T> = exchange;
    core::array::from_fn(|face| exchange.recv_buffer(face))
}

/// Half spinor number `slot` of a face buffer.
#[inline(always)]
pub(super) fn load_half<T: Storage>(buffer: &[T], slot: usize) -> HalfSpinor<T::Compute> {
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
pub(super) fn store_half<T: Storage>(h: &HalfSpinor<T::Compute>, s: &mut [T; HALF_SPINOR_REALS]) {
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
pub(super) fn stored_spinor<T>(s: &[T]) -> &[T; SPINOR_REALS] {
    s.try_into().expect("a spinor has 24 reals")
}

#[inline(always)]
pub(super) fn load_spinor<T: Storage>(s: &[T; SPINOR_REALS]) -> SiteSpinor<T::Compute> {
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
pub(super) fn store_spinor<T: Storage>(p: &SiteSpinor<T::Compute>, s: &mut [T]) {
    let s: &mut [T; SPINOR_REALS] = s.try_into().expect("a spinor has 24 reals");
    for a in 0..NS {
        for c in 0..NC {
            let k = 2 * (NC * a + c);
            s[k] = T::store(p.0[a][c].re);
            s[k + 1] = T::store(p.0[a][c].im);
        }
    }
}
