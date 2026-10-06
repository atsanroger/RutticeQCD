//! Even-odd (checkerboard) decomposition of a lattice.

use crate::field::{Aos, Cpu, Field, Full, Half, Kind};
use crate::lattice::Lattice;
use crate::storage::Storage;

/// Parity of a site: whether x + y + z + t is even or odd.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    pub fn flip(self) -> Self {
        match self {
            Parity::Even => Parity::Odd,
            Parity::Odd => Parity::Even,
        }
    }
    pub fn index(self) -> usize {
        match self {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
    pub fn of(x: [usize; 4]) -> Self {
        if (x[0] + x[1] + x[2] + x[3]).is_multiple_of(2) {
            Parity::Even
        } else {
            Parity::Odd
        }
    }
}

/// The correspondence between full-lattice sites and the sites of each parity.
///
/// Sites of one parity are numbered in the order they appear in the full lattice, which
/// gives `x / 2 + (Lx / 2) * (y + Ly * (z + Lz * t))`.
pub struct EvenOddIndex {
    full_of: [Vec<u32>; 2],
    half_of: Vec<u32>,
    parity_of: Vec<Parity>,
}

impl EvenOddIndex {
    /// Every extent must be even, so that the checkerboard is consistent across the
    /// periodic boundaries.
    pub fn new(lat: &Lattice) -> Self {
        assert!(
            lat.local.iter().all(|l| l % 2 == 0),
            "even-odd decomposition needs even extents, got {:?}",
            lat.local
        );
        let n = lat.num_sites();
        assert!(n <= u32::MAX as usize);
        let mut full_of = [Vec::with_capacity(n / 2), Vec::with_capacity(n / 2)];
        let mut half_of = vec![0u32; n];
        let mut parity_of = vec![Parity::Even; n];
        for site in 0..n {
            let p = Parity::of(lat.coords(site));
            half_of[site] = full_of[p.index()].len() as u32;
            parity_of[site] = p;
            full_of[p.index()].push(site as u32);
        }
        Self {
            full_of,
            half_of,
            parity_of,
        }
    }

    pub fn num_half_sites(&self) -> usize {
        self.full_of[0].len()
    }

    /// Full-lattice index of site `half` of the given parity.
    #[inline(always)]
    pub fn full_site(&self, parity: Parity, half: usize) -> usize {
        self.full_of[parity.index()][half] as usize
    }

    /// All full-lattice indices of one parity, in half-lattice order.
    pub fn full_sites(&self, parity: Parity) -> &[u32] {
        &self.full_of[parity.index()]
    }

    /// Parity of a full-lattice site and its index within that parity.
    #[inline(always)]
    pub fn half_site(&self, full: usize) -> (Parity, usize) {
        (self.parity_of[full], self.half_of[full] as usize)
    }
}

/// Copy the sites of one parity out of a full field. The half field takes that parity.
pub fn extract_parity<T: Storage, K: Kind>(
    half: &mut Field<T, K, Half, Aos, Cpu>,
    full: &Field<T, K, Full, Aos, Cpu>,
    index: &EvenOddIndex,
    parity: Parity,
) {
    assert_eq!(half.site_len(), full.site_len());
    assert_eq!(half.outer(), full.outer());
    assert_eq!(half.num_sites(), index.num_half_sites());
    half.set_parity(parity);
    for o in 0..full.outer() {
        for (h, f) in index.full_sites(parity).iter().enumerate() {
            half.site_mut(o, h)
                .copy_from_slice(full.site(o, *f as usize));
        }
    }
}

/// Copy a half field into the sites of its parity in a full field; other sites are untouched.
pub fn insert_parity<T: Storage, K: Kind>(
    full: &mut Field<T, K, Full, Aos, Cpu>,
    half: &Field<T, K, Half, Aos, Cpu>,
    index: &EvenOddIndex,
) {
    assert_eq!(half.site_len(), full.site_len());
    assert_eq!(half.outer(), full.outer());
    assert_eq!(half.num_sites(), index.num_half_sites());
    for o in 0..full.outer() {
        for (h, f) in index.full_sites(half.parity()).iter().enumerate() {
            full.site_mut(o, *f as usize)
                .copy_from_slice(half.site(o, h));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Spinor;

    #[test]
    fn numbering_matches_the_formula_and_is_a_bijection() {
        let lat = Lattice::single_node([4, 2, 6, 4]);
        let idx = EvenOddIndex::new(&lat);
        assert_eq!(idx.num_half_sites(), lat.num_sites() / 2);
        for site in 0..lat.num_sites() {
            let x = lat.coords(site);
            let (p, h) = idx.half_site(site);
            assert_eq!(p, Parity::of(x));
            assert_eq!(h, x[0] / 2 + 2 * (x[1] + 2 * (x[2] + 6 * x[3])));
            assert_eq!(idx.full_site(p, h), site);
        }
    }

    #[test]
    fn extract_then_insert_restores_the_field() {
        let lat = Lattice::single_node([4, 4, 2, 2]);
        let idx = EvenOddIndex::new(&lat);
        let mut full = Field::<f64, Spinor>::spinor(&lat, 2);
        for (i, x) in full.as_mut_slice().iter_mut().enumerate() {
            *x = i as f64;
        }
        let mut even = Field::<f64, Spinor, Half>::spinor_half(&lat, 2, Parity::Even);
        let mut odd = Field::<f64, Spinor, Half>::spinor_half(&lat, 2, Parity::Even);
        extract_parity(&mut even, &full, &idx, Parity::Even);
        extract_parity(&mut odd, &full, &idx, Parity::Odd);
        assert_eq!(odd.parity(), Parity::Odd);

        let mut back = Field::<f64, Spinor>::spinor(&lat, 2);
        insert_parity(&mut back, &even, &idx);
        assert_ne!(back.as_slice(), full.as_slice());
        insert_parity(&mut back, &odd, &idx);
        assert_eq!(back.as_slice(), full.as_slice());
    }

    #[test]
    #[should_panic]
    fn odd_extent_is_rejected() {
        let _ = EvenOddIndex::new(&Lattice::single_node([4, 3, 4, 4]));
    }
}
