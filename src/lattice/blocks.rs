//! Blocks of sites for the SIMD layout (`Aosoa<W>`).

use super::even_odd::{EvenOddIndex, Parity};
use super::stencil::{Boundary, Direction, Stencil};
use super::{Lattice, ND};
use crate::comm::{HaloSpec, FACES};

/// The division of a lattice into blocks of `lanes` sites.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blocks {
    local: [usize; ND],
    /// Sub-lattices per direction; the product is the number of lanes.
    split: [usize; ND],
    /// Extent of one sub-lattice, which is also the lattice of the blocks.
    reduced: [usize; ND],
    lanes: usize,
}

fn lex(x: [usize; ND], dims: [usize; ND]) -> usize {
    x[0] + dims[0] * (x[1] + dims[1] * (x[2] + dims[2] * x[3]))
}

fn unlex(mut i: usize, dims: [usize; ND]) -> [usize; ND] {
    core::array::from_fn(|d| {
        let x = i % dims[d];
        i /= dims[d];
        x
    })
}

impl Blocks {
    /// `lanes` must be a power of two. The directions are halved in turn, the last one
    /// first, skipping those whose extent is no longer even; `None` if the lattice cannot
    /// be cut into that many equal sub-lattices.
    pub fn new(local: [usize; ND], lanes: usize) -> Option<Self> {
        if !lanes.is_power_of_two() {
            return None;
        }
        let (mut split, mut reduced, mut have) = ([1; ND], local, 1);
        while have < lanes {
            let before = have;
            for mu in (0..ND).rev() {
                if have < lanes && reduced[mu].is_multiple_of(2) {
                    split[mu] *= 2;
                    reduced[mu] /= 2;
                    have *= 2;
                }
            }
            if have == before {
                return None;
            }
        }
        Some(Self {
            local,
            split,
            reduced,
            lanes,
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn num_blocks(&self) -> usize {
        self.reduced.iter().product()
    }

    /// Sub-lattices per direction.
    pub fn split(&self) -> [usize; ND] {
        self.split
    }

    /// Extent of the lattice of blocks.
    pub(crate) fn reduced(&self) -> [usize; ND] {
        self.reduced
    }

    /// The lattice site (in the usual numbering) in one lane of one block.
    pub fn site(&self, block: usize, lane: usize) -> usize {
        let (o, s) = (unlex(block, self.reduced), unlex(lane, self.split));
        lex(
            core::array::from_fn(|d| o[d] + self.reduced[d] * s[d]),
            self.local,
        )
    }

    /// The even-odd decomposition of the lattice of blocks, if every one of its extents is
    /// even. Then every lane of a block has the parity of the block's position, so a field
    /// on one parity is a field on the blocks of that parity.
    pub fn even_odd_index(&self) -> Option<EvenOddIndex> {
        self.reduced
            .iter()
            .all(|l| l.is_multiple_of(2))
            .then(|| EvenOddIndex::of_extents(self.reduced))
    }

    /// The block and the lane of a lattice site.
    pub(crate) fn block_lane(&self, site: usize) -> (usize, usize) {
        let x = unlex(site, self.local);
        let o = core::array::from_fn(|d| x[d] % self.reduced[d]);
        let s = core::array::from_fn(|d| x[d] / self.reduced[d]);
        (lex(o, self.reduced), lex(s, self.split))
    }
}

/// What the lanes of a neighbouring block need when the hop to it leaves the sub-lattice.
#[derive(Clone, Debug)]
pub(crate) struct Edge<const W: usize> {
    /// Lane `i` of the block being computed reads lane `from[i]` of the neighbouring block.
    pub from: [usize; W],
    /// For each lane of the block being computed: the factor (re, im) of the boundary
    /// condition if its hop crosses the boundary of the global lattice, (1, 0) otherwise.
    /// Zero for a Dirichlet boundary.
    pub factor: [(f64, f64); W],
}

/// In `BlockStencil::received`: the lane does not come from another process.
pub(crate) const NOT_RECEIVED: u32 = u32::MAX;

/// Nearest neighbours of blocks: the counterpart of `Stencil` for the SIMD layout.
pub(crate) struct BlockStencil<const W: usize> {
    blocks: Blocks,
    /// Blocks covered: all of them, or those of one parity.
    num_blocks: usize,
    /// `[block * LEGS + leg]`: the neighbouring block, with the top bit set if the hop
    /// leaves the sub-lattice.
    neighbor: Vec<u32>,
    /// Per leg, what to do to the lanes of such a hop.
    edge: [Edge<W>; Stencil::LEGS],
    /// Rank on the other side of each face.
    neighbour_rank: [usize; FACES],
    /// Whether leaving through each face crosses the boundary of the global lattice.
    at_global_boundary: [bool; FACES],
    /// For each face, the (block, lane) of the input sites the neighbour on that side
    /// needs, in the order of `Stencil::send_sites`.
    send: [Vec<(u32, u32)>; FACES],
    /// `[block * LEGS + leg]`: the entry of `received` for that hop, or `NOT_RECEIVED`.
    /// Empty on a single process.
    received_at: Vec<u32>,
    /// For each lane of such a hop, its position in what arrived on the face of the leg,
    /// or `NOT_RECEIVED`.
    received: Vec<[u32; W]>,
}

const EDGE: u32 = 1 << 31;

impl<const W: usize> BlockStencil<W> {
    /// All blocks; neighbours are numbered among all blocks. `None` if the lattice cannot
    /// be cut into W sub-lattices.
    pub fn new(lat: &Lattice, boundary: [Boundary; ND]) -> Option<Self> {
        let blocks = Blocks::new(lat.local, W)?;
        let covered = 0..blocks.num_blocks();
        Some(Self::build(lat, boundary, blocks, covered, Some))
    }

    /// The blocks of one parity; neighbours are numbered within the other parity. `index`
    /// is the even-odd decomposition of the lattice of blocks (`Blocks::even_odd_index`).
    pub(crate) fn new_parity(
        lat: &Lattice,
        boundary: [Boundary; ND],
        index: &EvenOddIndex,
        parity: Parity,
    ) -> Option<Self> {
        let blocks = Blocks::new(lat.local, W)?;
        let covered = index.full_sites(parity).iter().map(|b| *b as usize);
        Some(Self::build(lat, boundary, blocks, covered, |block| {
            let (p, half) = index.half_site(block);
            (p != parity).then_some(half)
        }))
    }

    /// `covered` are the blocks of this stencil, in order. `input_index` gives the index of
    /// a block in the field the neighbours are read from, or `None` if that field does not
    /// live on the block.
    fn build(
        lat: &Lattice,
        boundary: [Boundary; ND],
        blocks: Blocks,
        covered: impl ExactSizeIterator<Item = usize>,
        input_index: impl Fn(usize) -> Option<usize>,
    ) -> Self {
        let (reduced, split) = (blocks.reduced, blocks.split);
        let n = covered.len();
        assert!(blocks.num_blocks() < EDGE as usize);
        let legs = |leg: usize| {
            let (mu, dir) = (
                leg % ND,
                if leg < ND {
                    Direction::Forward
                } else {
                    Direction::Backward
                },
            );
            debug_assert_eq!(Stencil::leg(mu, dir), leg);
            (mu, dir)
        };
        // One step in a periodic box, and whether it wrapped around.
        let step = |x: [usize; ND], mu: usize, dir: Direction, dims: [usize; ND]| {
            let mut y = x;
            let wrapped = match dir {
                Direction::Forward => {
                    y[mu] = (x[mu] + 1) % dims[mu];
                    x[mu] + 1 == dims[mu]
                }
                Direction::Backward => {
                    y[mu] = (x[mu] + dims[mu] - 1) % dims[mu];
                    x[mu] == 0
                }
            };
            (y, wrapped)
        };

        let small = |i: usize| u32::try_from(i).expect("checked against EDGE above");
        let mut neighbor = vec![0u32; n * Stencil::LEGS];
        // Position of each block among the covered ones.
        let mut covered_at = vec![NOT_RECEIVED; blocks.num_blocks()];
        for (i, block) in covered.enumerate() {
            covered_at[block] = small(i);
            let o = unlex(block, reduced);
            for leg in 0..Stencil::LEGS {
                let (mu, dir) = legs(leg);
                let (to, leaves) = step(o, mu, dir, reduced);
                let to = input_index(lex(to, reduced))
                    .expect("the neighbour of a covered block holds no input");
                neighbor[i * Stencil::LEGS + leg] = small(to) | if leaves { EDGE } else { 0 };
            }
        }
        let at_global_boundary: [bool; FACES] = core::array::from_fn(|leg| {
            let (mu, dir) = legs(leg);
            lat.at_global_boundary(mu, dir)
        });
        let edge = core::array::from_fn(|leg| {
            let (mu, dir) = legs(leg);
            let mut from = [0; W];
            let mut factor = [(1.0, 0.0); W];
            for lane in 0..W {
                // Leaving the sub-lattice, the hop arrives in the next sub-lattice along mu;
                // from the last one it leaves the local lattice.
                let (to, crosses) = step(unlex(lane, split), mu, dir, split);
                from[lane] = lex(to, split);
                if crosses && at_global_boundary[leg] {
                    factor[lane] = match (boundary[mu], dir) {
                        (Boundary::Periodic, _) => (1.0, 0.0),
                        (Boundary::Antiperiodic, _) => (-1.0, 0.0),
                        (Boundary::Phase(re, im), Direction::Forward) => (re, im),
                        (Boundary::Phase(re, im), Direction::Backward) => (re, -im),
                        (Boundary::Dirichlet, _) => (0.0, 0.0),
                    };
                }
            }
            Edge { from, factor }
        });

        // What is exchanged, site by site in the order of `Stencil`: both the sites sent
        // through a face and the sites that receive on it are taken by increasing site
        // number, and only the latter are counted where a Dirichlet boundary cuts the hop.
        let exchanged = |mu: usize| lat.grid[mu] > 1;
        let mut send: [Vec<(u32, u32)>; FACES] = Default::default();
        let (mut received_at, mut received) = (Vec::new(), Vec::<[u32; W]>::new());
        if (0..ND).any(exchanged) {
            received_at = vec![NOT_RECEIVED; n * Stencil::LEGS];
            let mut arrived = [0u32; FACES];
            for site in 0..lat.num_sites() {
                let x = unlex(site, blocks.local);
                let (block, lane) = blocks.block_lane(site);
                for leg in 0..Stencil::LEGS {
                    let (mu, dir) = legs(leg);
                    if !exchanged(mu) || !step(x, mu, dir, blocks.local).1 {
                        continue;
                    }
                    if let Some(i) = input_index(block) {
                        send[leg].push((small(i), small(lane)));
                    }
                    let cut = at_global_boundary[leg] && boundary[mu] == Boundary::Dirichlet;
                    if covered_at[block] != NOT_RECEIVED && !cut {
                        let k = covered_at[block] as usize * Stencil::LEGS + leg;
                        if received_at[k] == NOT_RECEIVED {
                            received_at[k] = small(received.len());
                            received.push([NOT_RECEIVED; W]);
                        }
                        received[received_at[k] as usize][lane] = arrived[leg];
                        arrived[leg] += 1;
                    }
                }
            }
        }
        Self {
            blocks,
            num_blocks: n,
            neighbor,
            edge,
            neighbour_rank: core::array::from_fn(|leg| {
                let (mu, dir) = legs(leg);
                lat.neighbour_rank(mu, dir)
            }),
            at_global_boundary,
            send,
            received_at,
            received,
        }
    }

    /// Whether any neighbour comes from another process.
    pub(crate) fn has_buffers(&self) -> bool {
        self.send.iter().any(|s| !s.is_empty())
    }

    /// The (block, lane) in the input field of the sites to send through face `leg`, in
    /// the order the neighbour indexes them. Empty if that direction has a single process.
    pub(crate) fn send_lanes(&self, leg: usize) -> &[(u32, u32)] {
        &self.send[leg]
    }

    /// Whether what is sent through face `leg` crosses the boundary of the global lattice.
    pub(crate) fn sends_across_boundary(&self, leg: usize) -> bool {
        self.at_global_boundary[leg]
    }

    /// The exchange that carries `reals_per_site` reals for every site of every face.
    pub(crate) fn halo_spec(&self, reals_per_site: usize) -> HaloSpec {
        HaloSpec {
            neighbour: self.neighbour_rank,
            len: core::array::from_fn(|face| reals_per_site * self.send[face].len()),
        }
    }

    /// For a hop of `block` along `leg` that takes lanes from the neighbouring process:
    /// the position of each lane in what arrived on the face of the leg, `NOT_RECEIVED` for
    /// the lanes found on this process.
    #[inline(always)]
    pub(crate) fn received(&self, block: usize, leg: usize) -> Option<&[u32; W]> {
        match self.received_at.get(block * Stencil::LEGS + leg) {
            None | Some(&NOT_RECEIVED) => None,
            Some(at) => Some(&self.received[*at as usize]),
        }
    }

    pub fn blocks(&self) -> &Blocks {
        &self.blocks
    }

    /// Number of blocks this stencil covers.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// What a hop along `leg` that leaves the sub-lattice does to the lanes.
    pub(crate) fn edge(&self, leg: usize) -> &Edge<W> {
        &self.edge[leg]
    }

    /// The neighbouring block of `block` along `leg` (numbered as `Stencil::leg`), and what
    /// to do to its lanes if the hop leaves the sub-lattice.
    #[inline(always)]
    pub fn source(&self, block: usize, leg: usize) -> (usize, Option<&Edge<W>>) {
        let n = self.neighbor[block * Stencil::LEGS + leg];
        let to = (n & !EDGE) as usize;
        if n & EDGE != 0 {
            (to, Some(&self.edge[leg]))
        } else {
            (to, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_site_is_one_lane_of_one_block() {
        for (local, lanes, split) in [
            ([4, 4, 4, 8], 4, [1, 1, 2, 2]),
            ([4, 4, 4, 8], 8, [1, 2, 2, 2]),
            ([4, 4, 4, 8], 16, [2, 2, 2, 2]),
            ([4, 6, 2, 8], 2, [1, 1, 1, 2]),
            ([6, 6, 6, 8], 4, [1, 1, 2, 2]),
            ([6, 6, 6, 8], 32, [2, 2, 2, 4]),
            ([4, 4, 4, 8], 1, [1, 1, 1, 1]),
        ] {
            let blocks = Blocks::new(local, lanes).unwrap();
            assert_eq!(blocks.split(), split, "{local:?} into {lanes}");
            let sites: usize = local.iter().product();
            assert_eq!(blocks.num_blocks() * lanes, sites);
            let mut seen = vec![false; sites];
            for block in 0..blocks.num_blocks() {
                for lane in 0..lanes {
                    let site = blocks.site(block, lane);
                    assert!(!std::mem::replace(&mut seen[site], true));
                    assert_eq!(blocks.block_lane(site), (block, lane));
                }
            }
        }
        assert!(Blocks::new([3, 3, 3, 3], 2).is_none());
        assert!(Blocks::new([4, 4, 4, 8], 3).is_none());
        assert!(Blocks::new([2, 2, 2, 2], 32).is_none());
    }

    /// The block stencil names, lane by lane, the same neighbour sites and the same
    /// boundary crossings as the site stencil.
    #[test]
    fn block_neighbours_are_the_site_neighbours() {
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let boundary = [
            Boundary::Phase(0.6, 0.8),
            Boundary::Periodic,
            Boundary::Dirichlet,
            Boundary::Antiperiodic,
        ];
        let sites = Stencil::nearest(&lat, boundary);
        fn check<const W: usize>(lat: &Lattice, boundary: [Boundary; ND], sites: &Stencil) {
            use crate::lattice::stencil::Source;
            let st = BlockStencil::<W>::new(lat, boundary).unwrap();
            let blocks = st.blocks();
            for block in 0..blocks.num_blocks() {
                for leg in 0..Stencil::LEGS {
                    let (mu, dir) = (
                        leg % ND,
                        [Direction::Forward, Direction::Backward][leg / ND],
                    );
                    let (to, edge) = st.source(block, leg);
                    for lane in 0..W {
                        let (from, factor) = match edge {
                            Some(e) => (e.from[lane], e.factor[lane]),
                            None => (lane, (1.0, 0.0)),
                        };
                        let got = blocks.site(to, from);
                        match sites.source(blocks.site(block, lane), mu, dir) {
                            Source::Local { site, wrapped } => {
                                assert_eq!(got, site);
                                let want = match (wrapped, boundary[mu], dir) {
                                    (false, _, _) | (true, Boundary::Periodic, _) => (1.0, 0.0),
                                    (true, Boundary::Antiperiodic, _) => (-1.0, 0.0),
                                    (true, Boundary::Phase(re, im), Direction::Forward) => (re, im),
                                    (true, Boundary::Phase(re, im), Direction::Backward) => {
                                        (re, -im)
                                    }
                                    (true, Boundary::Dirichlet, _) => unreachable!(),
                                };
                                assert_eq!(factor, want);
                            }
                            Source::Zero => assert_eq!(factor, (0.0, 0.0)),
                            Source::Buffer { .. } => unreachable!("one process"),
                        }
                    }
                }
            }
        }
        check::<1>(&lat, boundary, &sites);
        check::<2>(&lat, boundary, &sites);
        check::<4>(&lat, boundary, &sites);
        check::<8>(&lat, boundary, &sites);
        check::<16>(&lat, boundary, &sites);
    }

    /// On several processes, for all blocks and for the blocks of each parity: every lane
    /// of every hop is the neighbour `Stencil` names, on this process or at the same
    /// position of what arrives, with the same factor; and the same sites are sent.
    #[test]
    fn exchanged_lanes_are_the_exchanged_sites() {
        use crate::lattice::stencil::Source;
        use crate::LocalComm;
        let boundary = [
            Boundary::Phase(0.6, 0.8),
            Boundary::Periodic,
            Boundary::Dirichlet,
            Boundary::Antiperiodic,
        ];
        fn phase(boundary: Boundary, dir: Direction) -> (f64, f64) {
            match (boundary, dir) {
                (Boundary::Periodic, _) => (1.0, 0.0),
                (Boundary::Antiperiodic, _) => (-1.0, 0.0),
                (Boundary::Phase(re, im), Direction::Forward) => (re, im),
                (Boundary::Phase(re, im), Direction::Backward) => (re, -im),
                (Boundary::Dirichlet, _) => (0.0, 0.0),
            }
        }
        /// `site_of(i)`: the lattice site of output site `i` of `sites`; `input(block)`:
        /// the lattice sites of the lanes of input block `block`.
        fn compare<const W: usize>(
            st: &BlockStencil<W>,
            sites: &Stencil,
            boundary: [Boundary; ND],
            out_site: &dyn Fn(usize, usize) -> usize,
            inp_site: &dyn Fn(usize, usize) -> usize,
            site_index: &dyn Fn(usize) -> usize,
        ) {
            assert_eq!(st.num_blocks() * W, sites.num_sites());
            assert_eq!(st.has_buffers(), sites.has_buffers());
            assert_eq!(st.halo_spec(12), sites.halo_spec(12));
            for leg in 0..Stencil::LEGS {
                let (mu, dir) = (
                    leg % ND,
                    [Direction::Forward, Direction::Backward][leg / ND],
                );
                assert_eq!(
                    st.sends_across_boundary(leg),
                    sites.sends_across_boundary(mu, dir)
                );
                let sent: Vec<usize> = st
                    .send_lanes(leg)
                    .iter()
                    .map(|(block, lane)| site_index(inp_site(*block as usize, *lane as usize)))
                    .collect();
                let want: Vec<usize> = sites
                    .send_sites(mu, dir)
                    .iter()
                    .map(|s| *s as usize)
                    .collect();
                assert_eq!(sent, want, "W = {W}, face {leg}");

                for block in 0..st.num_blocks() {
                    let (to, edge) = st.source(block, leg);
                    let received = st.received(block, leg);
                    assert!(received.is_none() || edge.is_some());
                    for lane in 0..W {
                        let (from, factor) = match edge {
                            Some(e) => (e.from[lane], e.factor[lane]),
                            None => (lane, (1.0, 0.0)),
                        };
                        let arrives = received.map_or(NOT_RECEIVED, |r| r[lane]);
                        let site = site_index(out_site(block, lane));
                        match sites.source(site, mu, dir) {
                            Source::Local { site, wrapped } => {
                                assert_eq!(arrives, NOT_RECEIVED);
                                assert_eq!(site_index(inp_site(to, from)), site);
                                let want = if wrapped {
                                    phase(boundary[mu], dir)
                                } else {
                                    (1.0, 0.0)
                                };
                                assert_eq!(factor, want);
                            }
                            Source::Buffer { index, wrapped } => {
                                assert_eq!(arrives as usize, index, "W = {W}, leg {leg}");
                                let want = if wrapped {
                                    phase(boundary[mu], dir)
                                } else {
                                    (1.0, 0.0)
                                };
                                assert_eq!(factor, want);
                            }
                            Source::Zero => {
                                assert_eq!(arrives, NOT_RECEIVED);
                                assert_eq!(factor, (0.0, 0.0));
                            }
                        }
                    }
                }
            }
        }
        fn check<const W: usize>(lat: &Lattice, boundary: [Boundary; ND]) {
            let blocks = Blocks::new(lat.local, W).unwrap();
            let st = BlockStencil::<W>::new(lat, boundary).unwrap();
            let sites = Stencil::nearest(lat, boundary);
            let site = |block: usize, lane: usize| blocks.site(block, lane);
            compare(&st, &sites, boundary, &site, &site, &|s| s);

            let Some(block_index) = blocks.even_odd_index() else {
                return;
            };
            let site_index = EvenOddIndex::new(lat);
            for to in [Parity::Even, Parity::Odd] {
                let st = BlockStencil::<W>::new_parity(lat, boundary, &block_index, to).unwrap();
                let sites = Stencil::nearest_parity(lat, boundary, &site_index, to);
                let out =
                    |block: usize, lane: usize| blocks.site(block_index.full_site(to, block), lane);
                let inp = |block: usize, lane: usize| {
                    blocks.site(block_index.full_site(to.flip(), block), lane)
                };
                compare(&st, &sites, boundary, &out, &inp, &|s| {
                    site_index.half_site(s).1
                });
            }
        }
        for grid in [
            [2, 1, 1, 1],
            [1, 2, 1, 2],
            [2, 2, 2, 2],
            [1, 1, 2, 4],
            [1, 1, 1, 1],
        ] {
            LocalComm::run([8, 8, 8, 16], grid, |lat| {
                check::<1>(&lat, boundary);
                check::<2>(&lat, boundary);
                check::<4>(&lat, boundary);
                check::<8>(&lat, boundary);
                check::<16>(&lat, boundary);
            });
        }
    }
}
