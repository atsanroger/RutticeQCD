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
//! spinors, with the link already applied on the side that holds it. The exchange is
//! finished before the sweep starts; computing the interior while it is under way is not
//! done yet.
//!
//! Solvers: CG on the normal equations, BiCGStab on D, either with even-odd
//! preconditioning. Tools needed: none.
//!
//! The normalisation (hopping parameter κ, unit diagonal) and the placement of the boundary
//! phase follow Bridge++ so that results can be compared. Written from the formula above.

use super::{Dagger, EvenOdd, FermionOp, GaugeError, HopTerms, Variant};
use crate::kernel::{
    project, sink_spinor, transport, transport_half, ColorMatrix, Cplx, Dirac, GammaRepre,
    HalfSpinor, LinkUse, Real, Sign, Spinor as SiteSpinor, HALF_SPINOR_REALS, SPINOR_REALS,
};
use crate::{
    for_sites, Boundary, Direction, EvenOddIndex, Field, Gauge, Half, HaloExchange, Lattice,
    Parity, Sites, Source, Spinor, Stencil, Storage, Vector, FACES, ND,
};
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

    fn exchange(
        &mut self,
        geo: &Geometry,
        copies: usize,
        lat: &Lattice,
    ) -> &mut dyn HaloExchange<T> {
        let found = self
            .exchanges
            .iter()
            .position(|(id, c, _)| *id == geo.id && *c == copies);
        let at = found.unwrap_or_else(|| {
            let spec = geo.stencil.halo_spec(copies * HALF_SPINOR_REALS);
            let exchange = T::halo(lat.comm.as_ref(), &spec);
            self.exchanges.push((geo.id, copies, exchange));
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
}

impl Geometry {
    fn full(lat: &Lattice, boundary: [Boundary; ND]) -> Self {
        Self {
            id: 0,
            stencil: Stencil::nearest(lat, boundary),
            out_full: None,
            inp_full: None,
        }
    }

    /// Output on the sites of parity `to`, input on the other parity.
    fn parity(lat: &Lattice, boundary: [Boundary; ND], index: &EvenOddIndex, to: Parity) -> Self {
        Self {
            id: 1 + to.index(),
            stencil: Stencil::nearest_parity(lat, boundary, index, to),
            out_full: Some(index.full_sites(to).to_vec()),
            inp_full: Some(index.full_sites(to.flip()).to_vec()),
        }
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
pub struct WilsonOp<T: Storage, G: GammaRepre, D: DiagTerm<T>> {
    params: WilsonParams,
    lat: Arc<Lattice>,
    full: Geometry,
    /// Present when every extent of the lattice is even.
    even_odd: Option<EvenOddGeometry>,
    u: Option<Field<T, Gauge>>,
    diag: D,
    _gamma: PhantomData<G>,
}

/// Wilson fermions: the diagonal block is the identity.
pub type Wilson<T, G = Dirac> = WilsonOp<T, G, UnitDiag>;

/// One leg of the stencil resolved for one output site: where the neighbour is and what
/// is still to be done to it.
#[derive(Clone, Copy)]
enum Hop<R> {
    /// Nothing arrives (Dirichlet boundary).
    Absent,
    /// A site of the input field: project, apply the phase and the link.
    Local {
        site: usize,
        link: ColorMatrix<R>,
        phase: Option<Cplx<R>>,
    },
    /// A projected spinor from the neighbouring process; the link is on this side.
    Projected {
        index: usize,
        link: ColorMatrix<R>,
        phase: Option<Cplx<R>>,
    },
    /// A spinor the neighbouring process has already transported.
    Transported { index: usize },
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
            full: Geometry::full(lat, params.boundary),
            even_odd,
            u: None,
            diag,
            _gamma: PhantomData,
        }
    }

    pub fn params(&self) -> &WilsonParams {
        &self.params
    }

    /// The even-odd numbering this operator uses, if the lattice allows one.
    pub fn even_odd_index(&self) -> Option<&Arc<EvenOddIndex>> {
        self.even_odd.as_ref().map(|eo| &eo.index)
    }

    fn gauge(&self) -> &Field<T, Gauge> {
        self.u
            .as_ref()
            .expect("Wilson: set_gauge must be called before the operator is applied")
    }

    /// The phase a hop in direction (mu, dir) picks up when it crosses the boundary of the
    /// global lattice. A hop backward picks up the conjugate.
    fn boundary_phase(&self, mu: usize, dir: Direction) -> Option<Cplx<T::Compute>> {
        let phase = match self.params.boundary[mu] {
            Boundary::Periodic | Boundary::Dirichlet => None,
            Boundary::Antiperiodic => Some((-1.0, 0.0)),
            Boundary::Phase(re, im) => Some(match dir {
                Direction::Forward => (re, im),
                Direction::Backward => (re, -im),
            }),
        };
        phase.map(|(re, im)| Cplx::new(Real::from_f64(re), Real::from_f64(im)))
    }

    /// The leg in direction (mu, dir) of output site `site`.
    ///
    /// Forward legs use the link at the output site. Backward legs use the link at the
    /// neighbour; when the neighbour is on another process, that process applies it (and
    /// the phase) before sending.
    #[inline(always)]
    fn resolve(&self, geo: &Geometry, site: usize, mu: usize, dir: Direction) -> Hop<T::Compute> {
        let phase = |wrapped: bool| {
            if wrapped {
                self.boundary_phase(mu, dir)
            } else {
                None
            }
        };
        match (geo.stencil.source(site, mu, dir), dir) {
            (Source::Zero, _) => Hop::Absent,
            (Source::Local { site: n, wrapped }, _) => Hop::Local {
                site: n,
                link: match dir {
                    Direction::Forward => self.link(mu, geo.out_site(site)),
                    Direction::Backward => self.link(mu, geo.inp_site(n)),
                },
                phase: phase(wrapped),
            },
            (Source::Buffer { index, wrapped }, Direction::Forward) => Hop::Projected {
                index,
                link: self.link(mu, geo.out_site(site)),
                phase: phase(wrapped),
            },
            (Source::Buffer { index, .. }, Direction::Backward) => Hop::Transported { index },
        }
    }

    /// The half spinor that arrives along one leg, ready for the sink, for spinor `r` of
    /// the site. `local(site)` reads the input field; `received` is what came in on the
    /// face of this leg.
    #[inline(always)]
    fn arrive(
        hop: &Hop<T::Compute>,
        mu: usize,
        dir: Direction,
        local: impl Fn(usize) -> SiteSpinor<T::Compute>,
        received: &[T],
        slot: impl Fn(usize) -> usize,
    ) -> Option<HalfSpinor<T::Compute>> {
        let (sign, link_use) = leg_kind(dir);
        match hop {
            Hop::Absent => None,
            Hop::Local { site, link, phase } => Some(transport(
                &G::GAMMA[mu],
                sign,
                &local(*site),
                link,
                link_use,
                *phase,
            )),
            Hop::Projected { index, link, phase } => Some(transport_half(
                &load_half::<T>(received, slot(*index)),
                link,
                link_use,
                *phase,
            )),
            Hop::Transported { index } => Some(load_half::<T>(received, slot(*index))),
        }
    }

    /// Send the faces of `inp` that the neighbouring processes need and return what
    /// arrived, by face. Does nothing on a single process.
    ///
    /// Through a forward face goes what the neighbour's backward legs need: this side
    /// holds the link, so the spinor is fully transported here. Through a backward face
    /// goes the projection only; the neighbour holds the link of its forward legs.
    ///
    /// `only` restricts the packing to what the legs in one direction need.
    fn exchange<'w, SI: Sites>(
        &self,
        work: &'w mut WilsonWork<T>,
        geo: &Geometry,
        inp: &Field<T, Spinor, SI>,
        pre: bool,
        only: Option<(usize, Direction)>,
    ) -> [&'w [T]; FACES] {
        if !geo.stencil.has_buffers() {
            return [&[]; FACES];
        }
        let copies = inp.copies();
        let exchange = work.exchange(geo, copies, &self.lat);
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
                        Direction::Forward => Some(self.link(mu, geo.inp_site(site))),
                        Direction::Backward => None,
                    };
                    let o = o.as_chunks_mut::<HALF_SPINOR_REALS>().0;
                    for (r, o) in o.iter_mut().enumerate() {
                        let psi = Self::input(inp, site, r, pre);
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
        exchange.wait();
        let exchange: &'w dyn HaloExchange<T> = exchange;
        core::array::from_fn(|face| exchange.recv_buffer(face))
    }

    /// Spinor `r` at one site of the input, with γ5 applied if `pre` is set.
    #[inline(always)]
    fn input<SI: Sites>(
        inp: &Field<T, Spinor, SI>,
        site: usize,
        r: usize,
        pre: bool,
    ) -> SiteSpinor<T::Compute> {
        let psi = load_spinor::<T>(&inp.site(0, site)[r * SPINOR_REALS..(r + 1) * SPINOR_REALS]);
        if pre {
            G::GAMMA5.apply(&psi)
        } else {
            psi
        }
    }

    /// The link U_mu at a full-lattice site.
    #[inline(always)]
    fn link(&self, mu: usize, full_site: usize) -> ColorMatrix<T::Compute> {
        let s = self.gauge().site(mu, full_site);
        let v: [T::Compute; 18] = core::array::from_fn(|k| s[k].load());
        ColorMatrix::from_reals(&v)
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
        (pre, post): (bool, bool),
    ) {
        self.check(geo, out, inp);
        let kappa: T::Compute = Real::from_f64(self.params.kappa);
        let copies = inp.copies();
        let received = self.exchange(work, geo, inp, pre, None);
        let (fwd, bwd) = (Direction::Forward, Direction::Backward);

        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            // Geometry and links are resolved once per site and shared by all copies.
            let forward: [_; ND] = core::array::from_fn(|mu| self.resolve(geo, x, mu, fwd));
            let backward: [_; ND] = core::array::from_fn(|mu| self.resolve(geo, x, mu, bwd));
            let x_full = geo.out_site(x);

            for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                let at = |site: usize| Self::input(inp, site, r, pre);
                let slot = |index: usize| index * copies + r;

                let mut hops = SiteSpinor::zero();
                for mu in 0..ND {
                    let gamma = &G::GAMMA[mu];
                    let face = received[Stencil::leg(mu, fwd)];
                    if let Some(h) = Self::arrive(&forward[mu], mu, fwd, at, face, slot) {
                        sink_spinor(&mut hops, &h, gamma, Sign::Minus);
                    }
                    let face = received[Stencil::leg(mu, bwd)];
                    if let Some(h) = Self::arrive(&backward[mu], mu, bwd, at, face, slot) {
                        sink_spinor(&mut hops, &h, gamma, Sign::Plus);
                    }
                }

                let diag = match diagonal {
                    Diagonal::Present => self.diag.apply(x_full, &at(x)),
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
        let geo = &self.full;
        self.check(geo, out, inp);
        let (sign, _) = leg_kind(dir);
        let copies = inp.copies();
        let received = self.exchange(work, geo, inp, false, Some((mu, dir)));
        let face = received[Stencil::leg(mu, dir)];
        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            let hop = self.resolve(geo, x, mu, dir);
            for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                let at = |site: usize| Self::input(inp, site, r, false);
                let slot = |index: usize| index * copies + r;
                if let Some(h) = Self::arrive(&hop, mu, dir, at, face, slot) {
                    let mut acc = load_spinor::<T>(o);
                    sink_spinor(&mut acc, &h, &G::GAMMA[mu], sign);
                    store_spinor::<T>(&acc, o);
                }
            }
        });
    }
}

/// Half spinor number `slot` of a face buffer.
#[inline(always)]
fn load_half<T: Storage>(buffer: &[T], slot: usize) -> HalfSpinor<T::Compute> {
    let s = &buffer[slot * HALF_SPINOR_REALS..(slot + 1) * HALF_SPINOR_REALS];
    let v: [T::Compute; HALF_SPINOR_REALS] = core::array::from_fn(|k| s[k].load());
    HalfSpinor::from_reals(&v)
}

#[inline(always)]
fn store_half<T: Storage>(h: &HalfSpinor<T::Compute>, s: &mut [T; HALF_SPINOR_REALS]) {
    let mut v = [T::Compute::ZERO; HALF_SPINOR_REALS];
    h.write_reals(&mut v);
    for (dst, src) in s.iter_mut().zip(v) {
        *dst = T::store(src);
    }
}

#[inline(always)]
fn load_spinor<T: Storage>(s: &[T]) -> SiteSpinor<T::Compute> {
    let v: [T::Compute; SPINOR_REALS] = core::array::from_fn(|k| s[k].load());
    SiteSpinor::from_reals(&v)
}

#[inline(always)]
fn store_spinor<T: Storage>(p: &SiteSpinor<T::Compute>, s: &mut [T]) {
    let mut v = [T::Compute::ZERO; SPINOR_REALS];
    p.write_reals(&mut v);
    for (dst, src) in s.iter_mut().zip(v) {
        *dst = T::store(src);
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> FermionOp<Field<T, Spinor>> for WilsonOp<T, G, D> {
    type Work = WilsonWork<T>;

    fn work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.num_sites() != self.lat.num_sites() || u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let mut own = Field::<T, Gauge>::gauge(&self.lat);
        for (dst, src) in own.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *dst = T::store(Real::from_f64(*src));
        }
        self.diag.set_gauge(&self.lat, u)?;
        self.u = Some(own);
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
        self.sweep(work, &self.full, out, inp, Diagonal::Present, (pre, post));
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
        self.check(&self.full, out, inp);
        let copies = inp.copies();
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            let term = self.diag.apply(i / copies, &load_spinor::<T>(s));
            let sum = load_spinor::<T>(o).add(&term);
            store_spinor::<T>(&sum, o);
        });
    }

    fn mult_gamma5(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(&self.full, out, inp);
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            store_spinor::<T>(&G::GAMMA5.apply(&load_spinor::<T>(s)), o);
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
                .even_odd
                .as_ref()
                .expect("Wilson: even-odd needs even extents");
            let geo = &eo.to[parity.index()];
            self.site_block(out, inp, |h| geo.out_site(h), true);
        }
    }
}
