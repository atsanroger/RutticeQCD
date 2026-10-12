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
//! over blocks (`sweep_blocks.rs`; the sweep over sites is in `sweep_sites.rs`, what they
//! share in `sweep.rs`): the same kernel functions with a lane type in place of a real number. It covers what the sweep over sites covers except the single hopping
//! terms and the two-row links. On several processes it fills and reads the same face
//! buffers, one site at a time.
//!
//! Files: this one has the operator and its layouts; `site.rs` the site-diagonal block;
//! `lexical.rs` the operator on the whole lattice; `even_odd.rs` its even-odd blocks;
//! `sweep*.rs` the sweep of the hopping term, which those are made of and which clover and
//! domain-wall fermions are built on.
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

mod even_odd;
mod lexical;
mod site;
mod sweep;
mod sweep_blocks;
mod sweep_sites;

pub use site::{DiagTerm, UnitDiag};
pub use sweep::{fifth_hop, After, Fifth, LinkFormat};
pub use sweep_blocks::BlockState;

use sweep::*;
