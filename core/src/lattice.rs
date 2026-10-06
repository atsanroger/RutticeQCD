use crate::comm::{Comm, SingleComm};
use crate::stencil::Direction;
use std::sync::Arc;

/// Number of space-time dimensions.
pub const ND: usize = 4;

/// A structured lattice. Sites are numbered lexicographically with x fastest:
/// `site = x + Lx * (y + Ly * (z + Lz * t))`.
///
/// With several processes, each holds one equal box of the global lattice. Processes are
/// numbered the same way as sites, by their position in the process grid.
pub struct Lattice {
    /// Size on this process.
    pub local: [usize; ND],
    pub global: [usize; ND],
    /// Process grid.
    pub grid: [usize; ND],
    /// Global coordinates of local site 0.
    pub origin: [usize; ND],
    /// Sub-lattices per direction for the SIMD layout; product is the lane count.
    pub split: [usize; ND],
    pub comm: Arc<dyn Comm>,
}

impl Lattice {
    pub fn single_node(dims: [usize; ND]) -> Arc<Self> {
        Self::new(dims, [1; ND], Arc::new(SingleComm))
    }

    /// This process's part of a lattice of size `global` split over a process grid. Every
    /// extent must be a multiple of the number of processes in that direction.
    pub fn new(global: [usize; ND], grid: [usize; ND], comm: Arc<dyn Comm>) -> Arc<Self> {
        assert_eq!(
            comm.size(),
            grid.iter().product::<usize>(),
            "the process grid {grid:?} does not match the number of processes"
        );
        for mu in 0..ND {
            assert!(
                grid[mu] > 0 && global[mu] > 0 && global[mu].is_multiple_of(grid[mu]),
                "the process grid {grid:?} does not divide the lattice {global:?}"
            );
        }
        let local: [usize; ND] = core::array::from_fn(|mu| global[mu] / grid[mu]);
        let position = grid_coords(grid, comm.rank());
        Arc::new(Self {
            local,
            global,
            grid,
            origin: core::array::from_fn(|mu| position[mu] * local[mu]),
            split: [1; ND],
            comm,
        })
    }

    pub fn num_sites(&self) -> usize {
        self.local.iter().product()
    }

    #[inline]
    pub fn site(&self, x: [usize; ND]) -> usize {
        x[0] + self.local[0] * (x[1] + self.local[1] * (x[2] + self.local[2] * x[3]))
    }

    #[inline]
    pub fn coords(&self, mut site: usize) -> [usize; ND] {
        let mut x = [0; ND];
        for (xd, l) in x.iter_mut().zip(self.local) {
            *xd = site % l;
            site /= l;
        }
        x
    }

    /// Global coordinates of a local site.
    #[inline]
    pub fn global_coords(&self, site: usize) -> [usize; ND] {
        let x = self.coords(site);
        core::array::from_fn(|mu| self.origin[mu] + x[mu])
    }

    /// Position of this process in the process grid.
    pub fn grid_position(&self) -> [usize; ND] {
        core::array::from_fn(|mu| self.origin[mu] / self.local[mu])
    }

    /// Rank of the process next to this one in direction (mu, dir), periodically.
    pub fn neighbour_rank(&self, mu: usize, dir: Direction) -> usize {
        let mut p = self.grid_position();
        let n = self.grid[mu];
        p[mu] = match dir {
            Direction::Forward => (p[mu] + 1) % n,
            Direction::Backward => (p[mu] + n - 1) % n,
        };
        p[0] + self.grid[0] * (p[1] + self.grid[1] * (p[2] + self.grid[2] * p[3]))
    }

    /// Whether a step from this process in direction (mu, dir) crosses the boundary of the
    /// global lattice.
    pub fn at_global_boundary(&self, mu: usize, dir: Direction) -> bool {
        let p = self.grid_position()[mu];
        match dir {
            Direction::Forward => p + 1 == self.grid[mu],
            Direction::Backward => p == 0,
        }
    }
}

fn grid_coords(grid: [usize; ND], mut rank: usize) -> [usize; ND] {
    let mut p = [0; ND];
    for (pd, n) in p.iter_mut().zip(grid) {
        *pd = rank % n;
        rank /= n;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::LocalComm;

    #[test]
    fn site_and_coords_are_inverse() {
        let lat = Lattice::single_node([4, 6, 2, 8]);
        for site in 0..lat.num_sites() {
            assert_eq!(lat.site(lat.coords(site)), site);
        }
        assert_eq!(lat.coords(1), [1, 0, 0, 0]);
        assert_eq!(lat.coords(4), [0, 1, 0, 0]);
    }

    #[test]
    fn the_boxes_of_all_ranks_tile_the_global_lattice() {
        let global = [4, 6, 4, 8];
        let boxes = LocalComm::run(global, [2, 3, 1, 2], |lat| {
            assert_eq!(lat.local, [2, 2, 4, 4]);
            (0..lat.num_sites())
                .map(|s| lat.global_coords(s))
                .collect::<Vec<_>>()
        });
        assert_eq!(boxes.len(), 12);
        let whole = Lattice::single_node(global);
        let mut seen = vec![false; whole.num_sites()];
        for x in boxes.into_iter().flatten() {
            assert!(!std::mem::replace(&mut seen[whole.site(x)], true));
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn neighbours_wrap_around_the_process_grid() {
        let got = LocalComm::run([4, 6, 4, 8], [2, 3, 1, 1], |lat| {
            let (f, b) = (Direction::Forward, Direction::Backward);
            (
                lat.grid_position(),
                [lat.neighbour_rank(0, f), lat.neighbour_rank(0, b)],
                [lat.neighbour_rank(1, f), lat.neighbour_rank(1, b)],
                lat.neighbour_rank(2, f),
                [lat.at_global_boundary(1, f), lat.at_global_boundary(1, b)],
            )
        });
        // rank = px + 2 * py
        assert_eq!(got[0], ([0, 0, 0, 0], [1, 1], [2, 4], 0, [false, true]));
        assert_eq!(got[3], ([1, 1, 0, 0], [2, 2], [5, 1], 3, [false, false]));
        assert_eq!(got[5], ([1, 2, 0, 0], [4, 4], [1, 3], 5, [true, false]));
    }
}
