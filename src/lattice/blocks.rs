//! Blocks of sites for the SIMD layout (`Aosoa<W>`).
//!
//! The local lattice is cut into W sub-lattices of equal shape, by halving it along some
//! directions. A block holds one site from each sub-lattice, the ones at the same position
//! inside their sub-lattice; they are the W lanes of the block. This is the arrangement
//! Grid uses. Its point: the neighbour of a block in any direction is a whole block with
//! the lanes in the same order, so a stencil loads it as one vector. Only when the hop
//! leaves the sub-lattice do the lanes have to be rotated (`BlockStencil`).

use super::stencil::{Boundary, Direction, Stencil};
use super::{Lattice, ND};

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
    pub fn reduced(&self) -> [usize; ND] {
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

    /// The block and the lane of a lattice site.
    pub fn block_lane(&self, site: usize) -> (usize, usize) {
        let x = unlex(site, self.local);
        let o = core::array::from_fn(|d| x[d] % self.reduced[d]);
        let s = core::array::from_fn(|d| x[d] / self.reduced[d]);
        (lex(o, self.reduced), lex(s, self.split))
    }
}

/// What the lanes of a neighbouring block need when the hop to it leaves the sub-lattice.
#[derive(Clone, Debug)]
pub struct Edge<const W: usize> {
    /// Lane `i` of the block being computed reads lane `from[i]` of the neighbouring block.
    pub from: [usize; W],
    /// For each lane of the block being computed: the factor (re, im) of the boundary
    /// condition if its hop crosses the boundary of the lattice, (1, 0) otherwise.
    /// Zero for a Dirichlet boundary.
    pub factor: [(f64, f64); W],
}

/// Nearest neighbours of blocks: the counterpart of `Stencil` for the SIMD layout.
///
/// One process only so far: neighbours beyond the lattice are the periodic images.
pub struct BlockStencil<const W: usize> {
    blocks: Blocks,
    /// `[block * LEGS + leg]`: the neighbouring block, with the top bit set if the hop
    /// leaves the sub-lattice.
    neighbor: Vec<u32>,
    /// Per leg, what to do to the lanes of such a hop.
    edge: [Edge<W>; Stencil::LEGS],
}

const EDGE: u32 = 1 << 31;

impl<const W: usize> BlockStencil<W> {
    /// `None` if the lattice cannot be cut into W sub-lattices.
    pub fn new(lat: &Lattice, boundary: [Boundary; ND]) -> Option<Self> {
        assert_eq!(
            lat.grid, [1; ND],
            "the SIMD layout on several processes is not implemented yet"
        );
        let blocks = Blocks::new(lat.local, W)?;
        let (reduced, split) = (blocks.reduced, blocks.split);
        let n = blocks.num_blocks();
        assert!(n < EDGE as usize);
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

        let mut neighbor = vec![0u32; n * Stencil::LEGS];
        for block in 0..n {
            let o = unlex(block, reduced);
            for leg in 0..Stencil::LEGS {
                let (mu, dir) = legs(leg);
                let (to, leaves) = step(o, mu, dir, reduced);
                let to = u32::try_from(lex(to, reduced)).expect("checked against EDGE above");
                neighbor[block * Stencil::LEGS + leg] = to | if leaves { EDGE } else { 0 };
            }
        }
        let edge = core::array::from_fn(|leg| {
            let (mu, dir) = legs(leg);
            let mut from = [0; W];
            let mut factor = [(1.0, 0.0); W];
            for lane in 0..W {
                // Leaving the sub-lattice, the hop arrives in the next sub-lattice along mu;
                // from the last one it crosses the boundary of the lattice.
                let (to, crosses) = step(unlex(lane, split), mu, dir, split);
                from[lane] = lex(to, split);
                if crosses {
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
        Some(Self {
            blocks,
            neighbor,
            edge,
        })
    }

    pub fn blocks(&self) -> &Blocks {
        &self.blocks
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
}
