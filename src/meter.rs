//! Floating-point operations and bytes moved, counted where the work is done: every application
//! of an operator and every operation of a `Vector` adds its own, computed from the shapes (QUDA
//! counts its kernels the same way, `Tunable::flops`, `bytes`). The bytes are a model, the least
//! a kernel must move (each value read once and written once), not a measurement; caches make
//! the real traffic smaller, so the arithmetic intensity is an estimate for a roofline before
//! Nsight Compute.
//!
//! The SU(3) building blocks follow QUDA's counts: a product of two 3 × 3 complex matrices is
//! 198 flops (`MATRIX_PRODUCT`), a sum 18.
//!
//! A count is two additions to numbers of the calling thread: no lock, no atomics, nothing on the
//! device. Each process (each thread of `LocalComm`) counts its own work. `Counts::now` before
//! and after a piece of a run gives what it did.

use std::cell::Cell;

/// What kind of work a count is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Fermion operators on the fine lattice (hops, the site blocks, the fifth dimension).
    Fine = 0,
    /// Coarse operators and the levels below them.
    Coarse = 1,
    /// Restriction and prolongation.
    Transfer = 2,
    /// The linear algebra of vectors, and copies and casts of fields.
    Vector = 3,
    /// What is made of the links alone: plaquettes, gauge actions and forces, smearing and its
    /// chain rule, the algebra of links and momenta in HMC, the clover blocks.
    Gauge = 4,
    /// The fermion forces: the derivatives of fermion actions by the links.
    Force = 5,
}

const KINDS: usize = 6;
const WORK: [Work; KINDS] = [
    Work::Fine,
    Work::Coarse,
    Work::Transfer,
    Work::Vector,
    Work::Gauge,
    Work::Force,
];

/// Slot of a count: the kind of work, then single (0) or double (1) precision.
fn slot(work: Work, real_bytes: usize) -> usize {
    2 * work as usize + usize::from(real_bytes > 4)
}

thread_local! {
    static FLOPS: [Cell<f64>; 2 * KINDS] = const { [const { Cell::new(0.0) }; 2 * KINDS] };
    static BYTES: [Cell<f64>; 2 * KINDS] = const { [const { Cell::new(0.0) }; 2 * KINDS] };
    static CALLS: Cell<u64> = const { Cell::new(0) };
}

/// Flops of a product of two 3 × 3 complex matrices: 27 complex products (6 each) and 18
/// complex sums (2 each).
pub(crate) const MATRIX_PRODUCT: f64 = 198.0;
/// Flops of a sum of two 3 × 3 complex matrices, or of one times a real.
pub(crate) const MATRIX_SUM: f64 = 18.0;
/// Reals of a 3 × 3 complex matrix.
pub(crate) const MATRIX_REALS: f64 = 18.0;

/// Flops of Gauss–Jordan on a complex n × n matrix with the identity beside it: for every
/// column, the pivot row times 1/pivot (2n complex products, 6 each) and the other n − 1 rows
/// cleared (2n multiply-adds each, 8).
pub(crate) fn gauss_jordan(n: usize) -> f64 {
    let n = n as f64;
    n * (2.0 * n * 6.0 + (n - 1.0) * 2.0 * n * 8.0)
}

/// Adds `flops` and `bytes` of `work` done in reals of `real_bytes` bytes.
#[inline]
pub(crate) fn add(work: Work, real_bytes: usize, flops: f64, bytes: f64) {
    let k = slot(work, real_bytes);
    FLOPS.with(|f| f[k].set(f[k].get() + flops));
    BYTES.with(|b| b[k].set(b[k].get() + bytes));
    CALLS.with(|c| c.set(c.get() + 1));
}

/// Runs `f` and takes back what it counted: for an operation that both backends count by one
/// formula, whose pieces would count themselves differently on each. Counts made by other
/// threads are not seen (counting is done by the thread that launches the work).
pub(crate) fn uncounted<R>(f: impl FnOnce() -> R) -> R {
    let before = Counts::now();
    let r = f();
    FLOPS.with(|fl| {
        BYTES.with(|b| {
            for k in 0..2 * KINDS {
                fl[k].set(before.flops[k / 2][k % 2]);
                b[k].set(before.bytes[k / 2][k % 2]);
            }
        })
    });
    CALLS.with(|c| c.set(before.calls));
    r
}

/// A vector operation on `reals` reals of `real_bytes` bytes each: `flops` and `moved` (reals
/// read plus written) for every one of them.
#[inline]
pub(crate) fn vector(reals: usize, real_bytes: usize, flops: f64, moved: f64) {
    let r = reals as f64;
    add(
        Work::Vector,
        real_bytes,
        flops * r,
        moved * r * real_bytes as f64,
    );
}

/// The counts of this thread so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counts {
    /// By kind of work (`Work as usize`), then single and double precision.
    pub flops: [[f64; 2]; KINDS],
    pub bytes: [[f64; 2]; KINDS],
    /// How many counts were added: what counting itself cost, in additions.
    pub calls: u64,
}

impl Counts {
    pub fn now() -> Self {
        let mut c = Self::default();
        FLOPS.with(|f| {
            BYTES.with(|b| {
                for k in 0..2 * KINDS {
                    c.flops[k / 2][k % 2] = f[k].get();
                    c.bytes[k / 2][k % 2] = b[k].get();
                }
            })
        });
        c.calls = CALLS.with(Cell::get);
        c
    }

    /// What was counted since `before`.
    pub fn since(&self, before: &Self) -> Self {
        let mut c = *self;
        for w in 0..KINDS {
            for p in 0..2 {
                c.flops[w][p] -= before.flops[w][p];
                c.bytes[w][p] -= before.bytes[w][p];
            }
        }
        c.calls -= before.calls;
        c
    }

    pub fn total_flops(&self) -> f64 {
        self.flops.iter().flatten().sum()
    }

    pub fn total_bytes(&self) -> f64 {
        self.bytes.iter().flatten().sum()
    }

    /// One line: Flop, GFlop/s and FLOP/byte over `seconds`, then the kinds of work that have
    /// any, with their share of the flops and their own intensity, and how many counts were
    /// made.
    pub fn line(&self, seconds: f64) -> String {
        let (f, b) = (self.total_flops(), self.total_bytes());
        let mut s = format!(
            "{:.3e} Flop, {:.1} GFlop/s, {:.2} Flop/byte",
            f,
            f / seconds * 1e-9,
            if b > 0.0 { f / b } else { 0.0 }
        );
        for (w, work) in WORK.iter().enumerate() {
            for (p, precision) in ["single", "double"].iter().enumerate() {
                let (fw, bw) = (self.flops[w][p], self.bytes[w][p]);
                if fw == 0.0 && bw == 0.0 {
                    continue;
                }
                s.push_str(&format!(
                    "; {work:?} {precision} {:.0}% ({:.2} Flop/byte)",
                    100.0 * fw / f.max(f64::MIN_POSITIVE),
                    if bw > 0.0 { fw / bw } else { 0.0 }
                ));
            }
        }
        // What counting cost: a few ns a count (`cost_of_a_count`).
        s.push_str(&format!("; {} counts", self.calls));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_add_up_by_kind_and_precision() {
        let before = Counts::now();
        add(Work::Fine, 8, 1320.0, 1000.0);
        add(Work::Fine, 4, 10.0, 20.0);
        vector(100, 8, 4.0, 3.0);
        let c = Counts::now().since(&before);
        assert_eq!(c.flops[0], [10.0, 1320.0]);
        assert_eq!(c.bytes[3][1], 100.0 * 3.0 * 8.0);
        assert_eq!(c.total_flops(), 1330.0 + 400.0);
        assert_eq!(c.calls, 3);
    }

    /// What one count costs (run by hand: `cargo test --release --lib meter -- --ignored
    /// --nocapture`).
    #[test]
    #[ignore]
    fn cost_of_a_count() {
        let n = 100_000_000u64;
        let start = std::time::Instant::now();
        for k in 0..n {
            add(std::hint::black_box(Work::Vector), 8, k as f64, 1.0);
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / n as f64;
        std::hint::black_box(Counts::now());
        eprintln!("one count: {ns:.2} ns");
    }
}
