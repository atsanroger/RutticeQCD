use crate::{Direction, Lattice, ND};
use std::sync::Arc;

/// The decomposition of a lattice into rectangular blocks. Each block is one site of the
/// coarse lattice.
pub struct Aggregation {
    fine: Arc<Lattice>,
    coarse: Arc<Lattice>,
    block: [usize; ND],
    block_volume: usize,
    /// Fine sites of coarse site c: `sites[c * block_volume..(c + 1) * block_volume]`.
    sites: Vec<u32>,
    /// Coarse site of each fine site.
    coarse_of: Vec<u32>,
}

impl Aggregation {
    /// Every extent of the lattice on one process must be a multiple of the block size in
    /// that direction, so that no block is split between processes. The coarse lattice is
    /// split over the same process grid as the fine one.
    pub fn new(fine: &Arc<Lattice>, block: [usize; ND]) -> Arc<Self> {
        for mu in 0..ND {
            assert!(
                block[mu] > 0 && fine.local[mu].is_multiple_of(block[mu]),
                "block size {:?} does not divide the lattice {:?} on one process",
                block,
                fine.local
            );
        }
        let coarse = Lattice::new(
            core::array::from_fn(|mu| fine.global[mu] / block[mu]),
            fine.grid,
            fine.comm.clone(),
        );
        let block_volume: usize = block.iter().product();
        let n = fine.num_sites();
        assert!(n <= u32::MAX as usize);

        let mut coarse_of = vec![0u32; n];
        let mut filled = vec![0usize; coarse.num_sites()];
        let mut sites = vec![0u32; n];
        for site in 0..n {
            let x = fine.coords(site);
            let c = coarse.site(core::array::from_fn(|mu| x[mu] / block[mu]));
            coarse_of[site] = c as u32;
            sites[c * block_volume + filled[c]] = site as u32;
            filled[c] += 1;
        }
        Arc::new(Self {
            fine: fine.clone(),
            coarse,
            block,
            block_volume,
            sites,
            coarse_of,
        })
    }

    pub fn fine_lattice(&self) -> &Arc<Lattice> {
        &self.fine
    }

    pub fn coarse_lattice(&self) -> &Arc<Lattice> {
        &self.coarse
    }

    pub fn block(&self) -> [usize; ND] {
        self.block
    }

    /// Number of fine sites in one block.
    pub fn block_volume(&self) -> usize {
        self.block_volume
    }

    /// The fine sites of one coarse site, in increasing order.
    #[inline(always)]
    pub fn fine_sites(&self, coarse_site: usize) -> &[u32] {
        &self.sites[coarse_site * self.block_volume..(coarse_site + 1) * self.block_volume]
    }

    #[inline(always)]
    pub fn coarse_site(&self, fine_site: usize) -> usize {
        self.coarse_of[fine_site] as usize
    }

    /// Whether the neighbour of `fine_site` in direction (mu, dir) lies in another block.
    /// With a single block along mu the neighbour across the lattice boundary is counted as
    /// another block too (the block's own periodic image).
    pub fn crosses(&self, fine_site: usize, mu: usize, dir: Direction) -> bool {
        let x = self.fine.coords(fine_site)[mu] % self.block[mu];
        match dir {
            Direction::Forward => x == self.block[mu] - 1,
            Direction::Backward => x == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fine_site_is_in_exactly_one_block() {
        let fine = Lattice::single_node([4, 4, 6, 8]);
        let agg = Aggregation::new(&fine, [2, 4, 3, 2]);
        assert_eq!(agg.coarse_lattice().local, [2, 1, 2, 4]);
        assert_eq!(agg.block_volume(), 48);
        let mut seen = vec![false; fine.num_sites()];
        for c in 0..agg.coarse_lattice().num_sites() {
            let sites = agg.fine_sites(c);
            assert!(sites.windows(2).all(|w| w[0] < w[1]));
            for s in sites {
                assert!(!seen[*s as usize]);
                seen[*s as usize] = true;
                assert_eq!(agg.coarse_site(*s as usize), c);
            }
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn crossing_is_decided_by_the_position_inside_the_block() {
        let fine = Lattice::single_node([4, 4, 4, 8]);
        let agg = Aggregation::new(&fine, [2, 4, 1, 4]);
        let at = |x| fine.site(x);
        assert!(agg.crosses(at([1, 0, 0, 0]), 0, Direction::Forward));
        assert!(!agg.crosses(at([1, 0, 0, 0]), 0, Direction::Backward));
        // A block as long as the lattice: only the wrap-around crosses.
        assert!(agg.crosses(at([0, 3, 0, 0]), 1, Direction::Forward));
        assert!(!agg.crosses(at([0, 2, 0, 0]), 1, Direction::Forward));
        // Blocks of one site: every hop crosses.
        assert!(agg.crosses(at([0, 0, 2, 0]), 2, Direction::Forward));
        assert!(agg.crosses(at([0, 0, 2, 0]), 2, Direction::Backward));
        assert!(agg.crosses(at([0, 0, 0, 4]), 3, Direction::Backward));
    }

    #[test]
    #[should_panic]
    fn blocks_must_divide_the_lattice() {
        let _ = Aggregation::new(&Lattice::single_node([4, 4, 4, 8]), [2, 2, 2, 3]);
    }
}
