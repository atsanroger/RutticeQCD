//! Random fields for checks and examples.
//!
//! These random numbers only build inputs for tests and checks. They are not the
//! generators of the physics code and make no attempt to match any other program.

use crate::{Field, Lattice, Lexical, Spinor};
use rayon::prelude::*;
use std::sync::Arc;

/// The increment of the state of SplitMix64 per draw.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64.
pub struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// The generator of `seed` after `draws` draws: the state grows by `GAMMA` per draw, so a
    /// loop can start each site at its own draw and fill the sites in any order.
    pub(crate) fn skipped(seed: u64, draws: u64) -> Self {
        Self(seed.wrapping_add(draws.wrapping_mul(GAMMA)))
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GAMMA);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (-1, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// A five-dimensional spinor field, `ls` spinors per site, every real uniform in (-1, 1): the
/// reals of `random_spinor(lat, ls, seed)` in the shape of one field with `ls` slices.
pub fn random_fifth(lat: &Arc<Lattice>, ls: usize, seed: u64) -> Field<f64, Spinor, Lexical> {
    let mut b = Field::<f64, Spinor, Lexical>::with_shape(lat, 24, ls, 1, 1);
    b.as_mut_slice()
        .copy_from_slice(random_spinor(lat, ls, seed).as_slice());
    b
}

/// `num_vectors` spinor fields with every real uniform in (-1, 1). The numbers are drawn in the
/// order of the whole lattice and every process keeps its own: the field does not depend on
/// how the lattice is split.
pub fn random_spinor(lat: &Arc<Lattice>, num_vectors: usize, seed: u64) -> Field<f64, Spinor> {
    let mut psi = Field::<f64, Spinor>::spinor(lat, num_vectors);
    let n = psi.site_len();
    let globals: Vec<usize> = (0..psi.num_sites())
        .map(|site| lat.global_site_index(lat.global_coords(site)))
        .collect();
    psi.as_mut_slice()
        .par_chunks_mut(n)
        .zip(&globals)
        .for_each(|(reals, &global)| {
            let mut rng = TestRng::skipped(seed, (global * n) as u64);
            for x in reals {
                *x = rng.uniform();
            }
        });
    psi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::multigrid::random_even;
    use crate::{EvenOddIndex, LocalComm, Parity};

    const GLOBAL: [usize; 4] = [4, 4, 2, 8];

    /// The reals of `per_site` per site drawn one after the other in the order of the whole
    /// lattice: the definition both fields follow.
    fn sequential(lat: &Lattice, per_site: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = TestRng::new(seed);
        (0..lat.num_global_sites())
            .map(|_| (0..per_site).map(|_| rng.uniform()).collect())
            .collect()
    }

    fn check(lat: &Arc<Lattice>) {
        let (ls, vectors, seed) = (3, 2, 77);
        let psi = random_spinor(lat, vectors, seed);
        let reference = sequential(lat, psi.site_len(), seed);
        assert_eq!(psi.as_slice().len(), psi.num_sites() * psi.site_len());
        for site in 0..psi.num_sites() {
            let global = lat.global_site_index(lat.global_coords(site));
            assert_eq!(lat.global_site_coords(global), lat.global_coords(site));
            assert_eq!(psi.site(0, site), &reference[global][..], "site {site}");
        }

        let index = EvenOddIndex::new(lat);
        let even = random_even::<f64>(lat, &index, ls, vectors, seed);
        let reference = sequential(lat, even.site_len(), seed);
        assert_eq!(even.as_slice().len(), even.num_sites() * even.site_len());
        for half in 0..even.num_sites() {
            let site = index.full_site(Parity::Even, half);
            let global = lat.global_site_index(lat.global_coords(site));
            assert_eq!(
                even.site(0, half),
                &reference[global][..],
                "even site {half}"
            );
        }
    }

    /// Each site jumps to its own draws, in parallel; the fields are those of the draws one
    /// after the other, bit for bit, on one process and on a split lattice.
    #[test]
    fn site_jumps_give_the_sequential_draws() {
        check(&Lattice::single_node(GLOBAL));
        LocalComm::run(GLOBAL, [2, 1, 1, 2], |lat| check(&lat));
    }
}
