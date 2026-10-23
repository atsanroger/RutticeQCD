//! Wilson fermions.
//!
//! Two things make that worth doing: a thread of its own waits for the exchange, since nothing
//! travels unless somebody is inside the communication library; and the two parts are lists of runs
//! of sites (`SiteRuns`), not a flag tested in two passes over all sites, which was slower than not
//! overlapping.

use super::{Dagger, EvenOdd, FermionOp, GaugeError, HopTerms, Variant};
use crate::kernel::{
    project, sink_spinor, transport, transport_half, ColorMatrix, Cplx, Dirac, GammaRepre,
    HalfSpinor, Lane, LaneOf, LinkUse, Number, Sign, Spinor as SiteSpinor, HALF_SPINOR_REALS, NC,
    NS, SPINOR_REALS,
};
use crate::{
    for_site_runs, for_sites, for_sites_as, for_sites_then, stream_fence, streams, Aos, Aosoa,
    BlockStencil, Boundary, Direction, Edge, EvenOddIndex, Field, Gauge, Half, HaloExchange,
    HaloSpec, Lattice, Layout, Lexical, Parity, SiteRuns, Sites, Source, Spinor, Stencil, Storage,
    Vector, FACES, ND, NOT_RECEIVED,
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
pub struct WilsonOp<T: Storage, G: GammaRepre, D: DiagTerm<T>, L: WilsonLayout<T> = Aos> {
    params: WilsonParams,
    lat: Arc<Lattice>,
    state: L::State,
    link_format: LinkFormat,
    diag: D,
    _gamma: PhantomData<G>,
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>, L: WilsonLayout<T>> WilsonOp<T, G, D, L> {
    /// The lattice the operator is on.
    pub fn lattice(&self) -> &Arc<Lattice> {
        &self.lat
    }

    /// The block of the operator within a site.
    pub fn diag_term(&self) -> &D {
        &self.diag
    }
}

impl<T: Storage, G: GammaRepre> WilsonOp<T, G, UnitDiag, Aos> {
    /// The links of `other`, an operator on the same lattice, converted to `T`: the operator in
    /// another precision without the gauge field it was set from.
    pub(crate) fn take_links_of<U: Storage>(&mut self, other: &WilsonOp<U, G, UnitDiag, Aos>) {
        assert!(Arc::ptr_eq(&self.lat, &other.lat) || self.lat.local == other.lat.local);
        self.state.links = other.state.links.converted();
        self.link_format = other.link_format;
    }
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
pub use sweep::links_in_su3;
mod sweep_blocks;
mod sweep_sites;

pub use site::{DiagTerm, UnitDiag};
pub use sweep::{fifth_hop, neff_q, After, Fifth, LinkFormat};
pub use sweep_blocks::BlockState;

use sweep::*;
