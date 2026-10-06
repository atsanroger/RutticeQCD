//! Random fields for checks and examples.
//!
//! These random numbers only build inputs for tests and checks. They are not the
//! generators of the physics code and make no attempt to match any other program.

use crate::{Field, Lattice, Spinor};
use std::sync::Arc;

/// SplitMix64.
pub struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
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

/// `num_vectors` spinor fields with every real uniform in (-1, 1).
pub fn random_spinor(lat: &Arc<Lattice>, num_vectors: usize, seed: u64) -> Field<f64, Spinor> {
    let mut rng = TestRng::new(seed);
    let mut psi = Field::<f64, Spinor>::spinor(lat, num_vectors);
    for x in psi.as_mut_slice() {
        *x = rng.uniform();
    }
    psi
}
