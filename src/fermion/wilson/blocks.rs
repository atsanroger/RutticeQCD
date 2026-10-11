//! Fields in blocks of W sites (the SIMD layout): the same kernel functions with a lane
//! type in place of a real number.

use super::*;

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
    /// The output blocks in two parts; `None` on a single process. A block is on the
    /// surface if any lane of any of its legs comes from another process.
    parts: Option<Parts>,
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
        let parts = stencil
            .has_buffers()
            .then(|| Parts::new(stencil.num_blocks(), from_buffer));
        Self {
            id,
            stencil,
            out_full,
            inp_full,
            parts,
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

/// Reads spinor `r` of the blocks of an input field, transformed by `pre`. A struct with a
/// method, not a closure, so that the read is expanded in place.
struct BlockInput<'a, T: Storage, G: GammaRepre, const W: usize> {
    spinors: &'a [BlockSpinor<T, W>],
    r: usize,
    /// Spinors per site.
    copies: usize,
    pre: Pre,
    _gamma: PhantomData<G>,
}

impl<T: Storage, G: GammaRepre, const W: usize> BlockInput<'_, T, G, W>
where
    T::Compute: LaneOf<W>,
{
    #[inline(always)]
    fn at(&self, block: usize) -> SiteSpinor<LaneType<T, W>> {
        let base = block * self.copies;
        self.pre.spinor::<LaneType<T, W>, G>(self.r, |k| {
            load_block_spinor::<T, W>(&self.spinors[base + k])
        })
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

    /// `even_odd_hop_fifth` for fields in blocks.
    pub fn even_odd_hop_fifth(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        to: Parity,
        dagger: Dagger,
        fifth: Fifth,
    ) {
        assert_eq!(out.parity(), to, "output field has the wrong parity");
        assert_eq!(inp.parity(), to.flip(), "input field has the wrong parity");
        assert_eq!(
            inp.copies() % fifth.ls,
            0,
            "the fields hold ls spinors per vector"
        );
        let ends = match dagger {
            Dagger::No => (Pre::Fifth(fifth), Post::Plain),
            Dagger::Yes => (Pre::Gamma5, Post::Gamma5Fifth(fifth)),
        };
        let geo = self.parity_geometry(to);
        self.block_sweep(work, geo, out, inp, Diagonal::Absent, ends);
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
        pre: Pre,
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
                let (len, work) = (copies * HALF_SPINOR_REALS, copies * SPINOR_REALS);
                for_sites_as(buffer, len, work, |k, o| {
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
                        let psi = pre.spinor::<T::Compute, G>(r, |k| {
                            lane_spinor::<T, W>(&spinors[block * copies + k], lane)
                        });
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

    /// `out = post( diagonal φ - κ Σ hops φ )` with `φ = pre(inp)`, block by block: `sweep`
    /// for fields in blocks.
    fn block_sweep<SO: Sites, SI: Sites>(
        &self,
        work: &mut WilsonWork<T>,
        geo: &BlockGeometry<W>,
        out: &mut BlockField<T, SO, W>,
        inp: &BlockField<T, SI, W>,
        diagonal: Diagonal,
        (pre, post): (Pre, Post),
    ) {
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
        let exchange = self.start_block_exchange(work, geo, inp, pre);
        let (out, block_len) = (out.slab_mut(0), copies * SPINOR_REALS * W);
        // A factor over the spinors of a block reads them back: they stay in the cache.
        let stream = streams(out, block_len) && !matches!(post, Post::Gamma5Fifth(_));
        let fence = || {
            if stream {
                stream_fence();
            }
        };

        overlapped(exchange, geo.parts.as_ref(), |received, part| {
            let block = |block: usize, o: &mut [T]| {
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
                    if post.gamma5_first() {
                        result = G::GAMMA5.apply(&result);
                    }
                    if stream {
                        let mut stored = [[T::default(); W]; SPINOR_REALS];
                        store_block_spinor::<T, W>(&result, &mut stored);
                        T::stream(o.as_flattened_mut(), stored.as_flattened());
                    } else {
                        store_block_spinor::<T, W>(&result, o);
                    }
                }
                if let Post::Gamma5Fifth(f) = post {
                    f.apply_in_place::<LaneType<T, W>, G, BlockSpinor<T, W>>(
                        o,
                        |x| load_block_spinor::<T, W>(x),
                        |p, x| store_block_spinor::<T, W>(p, x),
                    );
                }
            };
            match part {
                None => for_sites_then(out, block_len, block, fence),
                Some(runs) => for_site_runs(out, block_len, runs, block, fence),
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
            Variant::D => (Pre::Plain, Post::Plain),
            Variant::Gamma5D => (Pre::Plain, Post::Gamma5),
            Variant::Dagger => (Pre::Gamma5, Post::Gamma5),
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
            (Pre::gamma5(g5), Post::gamma5(g5)),
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
