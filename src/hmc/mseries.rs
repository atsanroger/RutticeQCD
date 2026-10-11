//! The M-series random numbers of Bridge++ (`RandomNumbers_Mseries`): a lagged-Fibonacci
//! generator x_n = x_(n−521) XOR x_(n−32) of 31-bit words (J. Makino, Parallel Computing 20
//! (1994) 1357). Bridge++'s class is a C++ version of the Fortran code of Makino and Miyamura,
//! whose public version came with S. Hioki's QCDMPI; Bridge++ does not use QCDMPI itself.
//! The same numbers as Bridge++ for the same seed, so that its runs can be repeated here;
//! written from the description of the algorithm, not from either code.

use crate::{Lattice, ND};

const NP: usize = 521;
const NQ: usize = 32;
/// 2^−31.
const NORM: f64 = 4.656612870908988e-10;

/// The generator and its state.
#[derive(Clone)]
pub struct Mseries {
    w: [u32; NP],
    jr: usize,
    kr: usize,
}

impl Mseries {
    /// The generator Bridge++ makes from the integer `ndelay` (its seed, or the trajectory
    /// number of the HMC): the initial words from a linear congruential sequence of bits,
    /// then the sequence moved on by `ndelay` · 2^40 steps.
    pub fn new(ndelay: u32) -> Self {
        // One bit from each of NP numbers of x ← 69069 x (mod 2^32), the top bit.
        let mut bits = [0u32; NP];
        let mut x: u32 = 1;
        for b in bits.iter_mut() {
            x = x.wrapping_mul(69069);
            *b = x >> 31;
        }
        // Each word collects 32 successive bits of the bit sequence b_n = b_(n−NP) ^ b_(n−NQ),
        // run in place; the lowest of them is dropped.
        let mut w = [0u32; NP];
        let (mut jr, mut kr) = (0, NP - NQ);
        for word in w.iter_mut() {
            let mut collected: u32 = 0;
            for _ in 0..32 {
                collected = collected.wrapping_mul(2).wrapping_add(bits[jr]);
                bits[jr] ^= bits[kr];
                jr = (jr + 1) % NP;
                kr = (kr + 1) % NP;
            }
            *word = collected >> 1;
        }
        let mut m = Self { w, jr, kr };
        m.delay(ndelay);
        m
    }

    /// Move the sequence on: the words become those `ndelay` · 2^40 steps later, by the
    /// polynomial x^(ndelay · 2^40) modulo the characteristic polynomial x^NP + x^(NP−NQ) + 1
    /// (over GF(2)), applied to the words.
    fn delay(&mut self, ndelay: u32) {
        // The words and the next NP − 1 of the sequence.
        let mut seq = [0u32; 2 * NP - 1];
        seq[..NP].copy_from_slice(&self.w);
        for i in NP..2 * NP - 1 {
            seq[i] = seq[i - NP] ^ seq[i - NQ];
        }
        // The binary digits of ndelay above the part below NP, after 40 zeros: the exponent
        // is reached by squaring (and multiplying by x where a digit is 1), from x^m.
        const SQUARINGS: usize = 40;
        let mut digits = vec![0u32; SQUARINGS];
        let mut m = ndelay as usize;
        while m > NP - 1 {
            digits.push((m % 2) as u32);
            m /= 2;
        }
        let mut c = [0u32; 2 * NP];
        c[m] = 1;
        for &digit in digits.iter().rev() {
            let digit = digit as usize;
            // c(x)² · x^digit: the coefficient of x^i goes to x^(2i + digit).
            for i in (0..NP).rev() {
                c[2 * i + digit] = c[i];
                c[2 * i + 1 - digit] = 0;
            }
            // Reduced modulo the characteristic polynomial.
            for i in (NP..2 * NP).rev() {
                c[i - NP] ^= c[i];
                c[i - NQ] ^= c[i];
            }
        }
        for (j, word) in self.w.iter_mut().enumerate() {
            let mut sum = 0u32;
            for i in 0..NP {
                sum ^= c[i].wrapping_mul(seq[j + i]);
            }
            *word = sum;
        }
    }

    /// A uniform number in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        self.w[self.jr] ^= self.w[self.kr];
        let r = f64::from(self.w[self.jr]) * NORM;
        self.jr = (self.jr + 1) % NP;
        self.kr = (self.kr + 1) % NP;
        r
    }

    /// Two Gaussian numbers of variance 1/2 (Box–Muller), as Bridge++ draws them.
    #[inline]
    pub fn gauss_pair(&mut self) -> (f64, f64) {
        let (r1, r2) = (self.uniform(), self.uniform());
        let radius = (-(1.0 - r1).ln() * 2.0).sqrt() * (1.0 / 2.0_f64.sqrt());
        let angle = 2.0 * (4.0 * 1.0_f64.atan()) * r2;
        (radius * angle.cos(), radius * angle.sin())
    }

    /// `nin` Gaussian numbers of variance 1/2 for every `ex` < `nex` and every site of the
    /// global lattice, in the order Bridge++ draws them (`gauss_lex_global`): `ex`
    /// outermost, then the global sites with x fastest, then the numbers of a site, which
    /// come in pairs (for an odd `nin` the second number of the last pair is dropped).
    /// `put(ex, local site, numbers)` gets those of the sites this process holds; every
    /// process draws all of them, so all get the same numbers on any process grid.
    pub fn gauss_lex_global(
        &mut self,
        lat: &Lattice,
        nin: usize,
        nex: usize,
        mut put: impl FnMut(usize, usize, &[f64]),
    ) {
        let mut numbers = vec![0.0; nin + 1];
        let global = lat.global;
        let total: usize = global.iter().product();
        for ex in 0..nex {
            for g in 0..total {
                for pair in numbers
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .take(nin.div_ceil(2))
                {
                    let (a, b) = self.gauss_pair();
                    pair[0] = a;
                    pair[1] = b;
                }
                let mut rest = g;
                let x: [usize; ND] = core::array::from_fn(|mu| {
                    let c = rest % global[mu];
                    rest /= global[mu];
                    c
                });
                if let Some(site) = lat.local_site(x) {
                    put(ex, site, &numbers[..nin]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_uniform_and_repeatable() {
        let (mut a, mut b) = (Mseries::new(0), Mseries::new(0));
        let mut c = Mseries::new(12345);
        let n = 100_000;
        let (mut sum, mut differ) = (0.0, false);
        for _ in 0..n {
            let x = a.uniform();
            assert!((0.0..1.0).contains(&x));
            assert_eq!(x, b.uniform());
            differ |= x != c.uniform();
            sum += x;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.01);
        assert!(differ, "another seed gives other numbers");
    }
}
