//! Flexible GCR keeping the last k directions (truncated GCR, Orthomin(k)).

use super::{not_finite, true_residual, SolveError, SolveInfo, Solver, Stop};
use crate::{LinearFn, Vector, C64};

/// Flexible GCR(k): every step applies the preconditioner once, z = M r, makes q = A z orthogonal
/// to the q of the last `k` steps (and z alike), and minimises the residual along q.
pub struct Gcr<V> {
    pub stop: Stop,
    pub k: usize,
    /// The pairs (z, q), q = A z normalised, oldest first; one more than `k`.
    pairs: Vec<(V, V)>,
    r: Option<V>,
    /// With `recording`: |r|²/|b|² of every vector before every step (the residual of the
    /// recurrence), and after the last one, of every solve.
    pub(crate) history: Option<Vec<Vec<f64>>>,
}

impl<V: Vector> Gcr<V> {
    pub fn new(stop: Stop, k: usize) -> Self {
        Self {
            stop,
            k,
            pairs: Vec::new(),
            r: None,
            history: None,
        }
    }

    /// Keep the residual history (`history`).
    pub(crate) fn recording(mut self) -> Self {
        self.history = Some(Vec::new());
        self
    }
}

impl<V: Vector> Solver<V> for Gcr<V> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        mut m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.r.as_ref().is_some_and(|r| r.same_shape(b)) {
            self.r = Some(b.zeros_like());
            self.pairs = (0..=self.k)
                .map(|_| (b.zeros_like(), b.zeros_like()))
                .collect();
        }
        let r = self.r.as_mut().expect("made above");
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let tol = self.stop.tolerance_sqr;
        let limit = self.stop.max_iterations * (self.stop.max_restarts + 1);
        let (mut rr, mut nq, mut residual) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
        let mut dots = vec![C64::ZERO; nv];
        let (mut iterations, mut cycles) = (0, 0);
        loop {
            a.mult_xpay(r, x, &minus_one, b);
            // `next`: the slot of the pair made in the coming step; the kept ones are the
            // `kept` before it, cyclically.
            let (mut next, mut kept) = (0, 0);
            let mut in_cycle = 0;
            while in_cycle < self.stop.max_iterations && iterations < limit {
                r.norm_sqr(&mut rr);
                if not_finite(&rr) {
                    return Err(SolveError::Breakdown {
                        iterations,
                        what: "the residual is not finite",
                    });
                }
                if let Some(h) = self.history.as_mut() {
                    h.push(
                        (0..nv)
                            .map(|j| rr[j] / b_norm[j].max(f64::MIN_POSITIVE))
                            .collect(),
                    );
                }
                let active: Vec<bool> = (0..nv).map(|j| rr[j] > tol * b_norm[j]).collect();
                if !active.iter().any(|a| *a) {
                    break;
                }
                let slots = self.pairs.len();
                let z = next;
                {
                    let (zv, qv) = &mut self.pairs[z];
                    match m.as_mut() {
                        Some(m) => m.mult(zv, r),
                        None => zv.copy_from(r),
                    }
                    a.mult(qv, zv);
                }
                // Orthogonalise against the kept pairs.
                for back in 1..=kept {
                    let old = (next + slots - back) % slots;
                    let (left, right) = if old < z {
                        let (l, r) = self.pairs.split_at_mut(z);
                        (&l[old], &mut r[0])
                    } else {
                        let (l, r) = self.pairs.split_at_mut(old);
                        (&r[0], &mut l[z])
                    };
                    left.1.dot(&right.1, &mut dots);
                    let minus: Vec<C64> = dots.iter().map(|d| -*d).collect();
                    right.1.axpy(&minus, &left.1);
                    right.0.axpy(&minus, &left.0);
                }
                let (zv, qv) = &mut self.pairs[z];
                qv.norm_sqr(&mut nq);
                if (0..nv).any(|j| active[j] && !(nq[j] > 0.0 && nq[j].is_finite())) {
                    return Err(SolveError::Breakdown {
                        iterations,
                        what: "A z vanished",
                    });
                }
                let scale: Vec<f64> = (0..nv)
                    .map(|j| if active[j] { 1.0 / nq[j].sqrt() } else { 0.0 })
                    .collect();
                qv.scale(&scale);
                zv.scale(&scale);
                qv.dot(r, &mut dots);
                x.axpy(&dots, zv);
                let minus: Vec<C64> = dots.iter().map(|d| -*d).collect();
                r.axpy(&minus, qv);
                next = (next + 1) % slots;
                kept = (kept + 1).min(self.k);
                iterations += 1;
                in_cycle += 1;
            }
            true_residual(a, x, b, &b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: cycles,
                residual_sqr: residual.clone(),
            };
            if residual.iter().all(|r| *r <= tol) {
                return Ok(info);
            }
            if residual.iter().any(|r| !r.is_finite()) {
                return Err(SolveError::Breakdown {
                    iterations,
                    what: "the solution is not finite",
                });
            }
            if cycles >= self.stop.max_restarts || iterations >= limit {
                return Err(SolveError::NotConverged(info));
            }
            cycles += 1;
        }
    }
}
