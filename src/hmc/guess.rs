//! Initial guesses for the solves of the molecular dynamics from the solutions of the solves
//! before them (the chronological inverter of Brower, Ivanenko, Levi and Orginos,
//! hep-lat/9509012).
//!
//! Along a trajectory the links move by O(ε) per step, so the solution x = A⁻¹ b of a force
//! solve differs little from the previous ones. The guess is the best combination of the last
//! `depth` solutions v_1 … v_n for the present A: x₀ = Σ c_j v_j with c minimising the A-norm
//! of the error, (V†AV) c = V†b, which takes n products with A. With depth 1 this is the last
//! solution rescaled.
//!
//! The guess changes the rounding of the solution, so with a guess the run is no longer the
//! same number by number as without; that is why it is off by default (Bridge++'s tests start
//! from zero). It also makes the molecular dynamics reversible only to the precision of the
//! solves, which at |r|²/|b|² ≤ 1e-20 or below is far below what the Metropolis test sees.

use crate::{LinearFn, Vector, C64};
use std::collections::VecDeque;

/// The last solutions of a sequence of solves with the same right-hand side.
pub struct Chrono<V> {
    /// How many solutions are kept; 0: no guess (start from zero).
    pub depth: usize,
    past: VecDeque<V>,
}

impl<V: Vector> Chrono<V> {
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            past: VecDeque::new(),
        }
    }

    /// Forget the solutions (a new right-hand side: the pseudofermions were drawn again).
    pub fn clear(&mut self) {
        self.past.clear();
    }

    /// Solutions kept now.
    pub fn len(&self) -> usize {
        self.past.len()
    }

    pub fn is_empty(&self) -> bool {
        self.past.is_empty()
    }

    /// Keep the solution `x` of the last solve.
    pub fn record(&mut self, x: &V) {
        if self.depth == 0 {
            return;
        }
        let mut v = x.zeros_like();
        v.copy_from(x);
        self.past.push_front(v);
        self.past.truncate(self.depth);
    }

    /// `x` = the guess for A x = b (zero without solutions kept).
    pub fn guess(&self, a: &mut dyn LinearFn<V>, b: &V, x: &mut V) {
        x.set_zero();
        if self.past.is_empty() {
            return;
        }
        // An orthonormal basis of the solutions (Gram-Schmidt twice, which is enough in
        // floating point); a solution already in the span of the newer ones is dropped.
        let mut q: Vec<V> = Vec::new();
        for v in &self.past {
            let mut w = v.zeros_like();
            w.copy_from(v);
            let mut n0 = [0.0];
            w.norm_sqr(&mut n0);
            for _ in 0..2 {
                for qj in &q {
                    let mut c = [C64::ZERO];
                    qj.dot(&w, &mut c);
                    w.axpy(&[-c[0]], qj);
                }
            }
            let mut n = [0.0];
            w.norm_sqr(&mut n);
            if n[0] > 1e-20 * n0[0] {
                w.scale(&[1.0 / n[0].sqrt()]);
                q.push(w);
            }
        }
        // (Q†AQ) c = Q†b.
        let k = q.len();
        let mut g = vec![C64::ZERO; k * k];
        let mut d = vec![C64::ZERO; k];
        let mut aq = b.zeros_like();
        for j in 0..k {
            a.mult(&mut aq, &q[j]);
            for i in 0..k {
                let mut c = [C64::ZERO];
                q[i].dot(&aq, &mut c);
                g[k * i + j] = c[0];
            }
            let mut c = [C64::ZERO];
            q[j].dot(b, &mut c);
            d[j] = c[0];
        }
        if let Some(c) = solve_small(k, &mut g, &mut d) {
            for (qj, cj) in q.iter().zip(c) {
                x.axpy(&[cj], qj);
            }
        }
    }
}

/// Solve the k × k system G c = d (row-major) by Gaussian elimination with partial pivoting;
/// `None` if it is singular.
fn solve_small(k: usize, g: &mut [C64], d: &mut [C64]) -> Option<Vec<C64>> {
    for col in 0..k {
        let p = (col..k).max_by(|&a, &b| {
            g[k * a + col]
                .norm_sqr()
                .total_cmp(&g[k * b + col].norm_sqr())
        })?;
        if g[k * p + col].norm_sqr() == 0.0 {
            return None;
        }
        if p != col {
            for j in 0..k {
                g.swap(k * p + j, k * col + j);
            }
            d.swap(p, col);
        }
        let pivot = g[k * col + col];
        for r in col + 1..k {
            let f = g[k * r + col] / pivot;
            for j in col..k {
                let t = g[k * col + j];
                g[k * r + j] = g[k * r + j] - f * t;
            }
            let t = d[col];
            d[r] = d[r] - f * t;
        }
    }
    let mut c = vec![C64::ZERO; k];
    for r in (0..k).rev() {
        let mut s = d[r];
        for j in r + 1..k {
            s = s - g[k * r + j] * c[j];
        }
        c[r] = s / g[k * r + r];
    }
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_systems_are_solved() {
        let m = |re: f64, im: f64| C64::new(re, im);
        let g0 = [m(0.0, 1.0), m(2.0, 0.0), m(1.0, -1.0), m(3.0, 0.5)];
        let x = [m(1.0, 2.0), m(-0.5, 0.25)];
        let mut d = [g0[0] * x[0] + g0[1] * x[1], g0[2] * x[0] + g0[3] * x[1]];
        let mut g = g0;
        let c = solve_small(2, &mut g, &mut d).unwrap();
        for (a, b) in c.iter().zip(x) {
            assert!((*a - b).norm_sqr() < 1e-28);
        }
    }
}
