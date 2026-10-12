//! Even-odd (checkerboard) decomposition of a lattice.

use crate::field::storage::Storage;
use crate::field::{Aos, Aosoa, Cpu, Field, Half, Kind, Layout, Lexical};
use crate::kernel::{Number, Real};
use crate::lattice::blocks::Blocks;
use crate::lattice::Lattice;

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
pub struct EvenOddIndex {
    full_of: [Vec<u32>; 2],
    half_of: Vec<u32>,
    parity_of: Vec<Parity>,
}

impl EvenOddIndex {
    /// Every extent must be even, so that the checkerboard is consistent across the
    /// periodic boundaries.
    pub fn new(lat: &Lattice) -> Self {
        Self::of_extents(lat.local)
    }

    /// The decomposition of a box with these extents, numbered like a lattice (x fastest).
    /// Used for the lattice itself and for the lattice of blocks of the SIMD layout.
    pub fn of_extents(extents: [usize; 4]) -> Self {
        assert!(
            extents.iter().all(|l| l % 2 == 0),
            "even-odd decomposition needs even extents, got {extents:?}"
        );
        let n: usize = extents.iter().product();
        assert!(n <= u32::MAX as usize);
        let coords = |mut site: usize| {
            extents.map(|l| {
                let x = site % l;
                site /= l;
                x
            })
        };
        let mut full_of = [Vec::with_capacity(n / 2), Vec::with_capacity(n / 2)];
        let mut half_of = vec![0u32; n];
        let mut parity_of = vec![Parity::Even; n];
        for site in 0..n {
            let p = Parity::of(coords(site));
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
pub fn extract_parity<T: Storage, K: Kind, L: Layout>(
    half: &mut Field<T, K, Half, L, Cpu>,
    full: &Field<T, K, Lexical, L, Cpu>,
    index: &EvenOddIndex,
    parity: Parity,
) {
    assert_eq!(half.site_len(), full.site_len());
    assert_eq!(half.outer(), full.outer());
    assert_eq!(half.num_blocks(), index.num_half_sites());
    half.set_parity(parity);
    let n = full.block_len();
    for o in 0..full.outer() {
        let (to, from) = (half.slab_mut(o), full.slab(o));
        for (h, f) in index.full_sites(parity).iter().enumerate() {
            let f = *f as usize;
            to[h * n..(h + 1) * n].copy_from_slice(&from[f * n..(f + 1) * n]);
        }
    }
}

/// Copy a half field into the sites of its parity in a full field; other sites are untouched.
pub fn insert_parity<T: Storage, K: Kind, L: Layout>(
    full: &mut Field<T, K, Lexical, L, Cpu>,
    half: &Field<T, K, Half, L, Cpu>,
    index: &EvenOddIndex,
) {
    assert_eq!(half.site_len(), full.site_len());
    assert_eq!(half.outer(), full.outer());
    assert_eq!(half.num_blocks(), index.num_half_sites());
    let n = full.block_len();
    for o in 0..full.outer() {
        let (to, from) = (full.slab_mut(o), half.slab(o));
        for (h, f) in index.full_sites(half.parity()).iter().enumerate() {
            let f = *f as usize;
            to[f * n..(f + 1) * n].copy_from_slice(&from[h * n..(h + 1) * n]);
        }
    }
}

/// Fields on one parity in the usual layout and in blocks of `W` sites (the SIMD layout) number
/// their sites differently: the usual one by `EvenOddIndex::new`, the blocks by the even-odd
/// decomposition of the lattice of blocks, in which every lane of a block has the parity of the
/// block.
pub struct ParityBlocks<const W: usize> {
    /// `[parity][block * W + lane]`: the site on that parity in the usual numbering.
    site_of: [Vec<u32>; 2],
}

impl<const W: usize> ParityBlocks<W> {
    /// `None` if the lattice cannot be cut into `W` sub-lattices whose lattice of blocks has
    /// even extents (what the operators in blocks need for even-odd too).
    pub fn new(lat: &Lattice) -> Option<Self> {
        let blocks = Blocks::new(lat.local, W)?;
        let block_index = blocks.even_odd_index()?;
        let index = EvenOddIndex::new(lat);
        let site_of = [Parity::Even, Parity::Odd].map(|parity| {
            let mut v = Vec::with_capacity(block_index.num_half_sites() * W);
            for block in block_index.full_sites(parity) {
                for lane in 0..W {
                    let (p, half) = index.half_site(blocks.site(*block as usize, lane));
                    assert_eq!(p, parity, "the lanes of a block have its parity");
                    v.push(half as u32);
                }
            }
            v
        });
        Some(Self { site_of })
    }

    /// `out` (in blocks) = `inp` (usual layout), on the parity of `inp`, converting the
    /// precision if they differ.
    pub fn to_blocks<T: Storage, U: Storage, K: Kind>(
        &self,
        out: &mut Field<T, K, Half, Aosoa<W>, Cpu>,
        inp: &Field<U, K, Half, Aos, Cpu>,
    ) {
        assert_eq!((out.site_len(), out.outer()), (inp.site_len(), inp.outer()));
        out.set_parity(inp.parity());
        let site_of = &self.site_of[inp.parity().index()];
        let (n, block_len) = (inp.site_len(), out.block_len());
        for o in 0..inp.outer() {
            let from = inp.slab(o);
            crate::for_light_sites(out.slab_mut(o), block_len, |block, d| {
                for lane in 0..W {
                    let s = site_of[block * W + lane] as usize;
                    for (c, x) in from[s * n..(s + 1) * n].iter().enumerate() {
                        d[lane + W * c] = T::store(Number::from_f64(x.load().to_f64()));
                    }
                }
            });
        }
    }

    /// `out` (usual layout) = `inp` (in blocks), on the parity of `inp`.
    pub fn to_sites<T: Storage, U: Storage, K: Kind>(
        &self,
        out: &mut Field<T, K, Half, Aos, Cpu>,
        inp: &Field<U, K, Half, Aosoa<W>, Cpu>,
    ) {
        assert_eq!((out.site_len(), out.outer()), (inp.site_len(), inp.outer()));
        out.set_parity(inp.parity());
        let site_of = &self.site_of[inp.parity().index()];
        let (n, block_len) = (out.site_len(), inp.block_len());
        for o in 0..inp.outer() {
            let (to, from) = (out.slab_mut(o), inp.slab(o));
            for (block, b) in from.chunks_exact(block_len).enumerate() {
                for lane in 0..W {
                    let s = site_of[block * W + lane] as usize;
                    for (c, x) in to[s * n..(s + 1) * n].iter_mut().enumerate() {
                        *x = T::store(Number::from_f64(b[lane + W * c].load().to_f64()));
                    }
                }
            }
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
