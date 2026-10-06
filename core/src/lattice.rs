use crate::comm::{Comm, SingleComm};
use std::sync::Arc;

/// Number of space-time dimensions.
pub const ND: usize = 4;

/// A structured lattice. Sites are numbered lexicographically with x fastest:
/// `site = x + Lx * (y + Ly * (z + Lz * t))`.
pub struct Lattice {
    /// Size on this process.
    pub local: [usize; ND],
    pub global: [usize; ND],
    /// Process grid.
    pub grid: [usize; ND],
    /// Sub-lattices per direction for the SIMD layout; product is the lane count.
    pub split: [usize; ND],
    pub comm: Arc<dyn Comm>,
}

impl Lattice {
    pub fn single_node(dims: [usize; ND]) -> Arc<Self> {
        Arc::new(Self {
            local: dims,
            global: dims,
            grid: [1; ND],
            split: [1; ND],
            comm: Arc::new(SingleComm),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_and_coords_are_inverse() {
        let lat = Lattice::single_node([4, 2, 6, 8]);
        for s in 0..lat.num_sites() {
            assert_eq!(lat.site(lat.coords(s)), s);
        }
        assert_eq!(lat.site([1, 0, 0, 0]), 1);
        assert_eq!(lat.site([0, 0, 0, 1]), 4 * 2 * 6);
    }
}
