//! Preconditioned MINRES for Hermitian, possibly indefinite, operators.

use super::{true_residual, SolveError, SolveInfo, Solver, Stop};
use crate::{LinearFn, Vector, C64};

/// MINRES (Elman, Silvester and Wathen, *Finite Elements and Fast Iterative Solvers*,
/// algorithm 2.4): Lanczos in the M-inner product with Givens rotations. A must be
/// Hermitian, M (optional) Hermitian positive definite and fixed. Unlike conjugate
/// residuals it cannot break down on an indefinite A, which divides by ⟨z, A z⟩.
///
/// It minimises the M-weighted norm of the residual (√⟨r, M r⟩), and stops when that is
/// below √`tolerance_sqr` times the same norm of b, or when the true |b − A x|² / |b|² is
/// below `tolerance_sqr`; after `max_iterations` steps it restarts from the current
/// solution. Every vector of a multi-vector has its own recurrence.
pub struct Minres<V> {
    pub stop: Stop,
    work: Option<[V; 9]>,
}

impl<V: Vector> Minres<V> {
    pub fn new(stop: Stop) -> Self {
        Self { stop, work: None }
    }
}

fn real(c: &[f64]) -> Vec<C64> {
    c.iter().map(|v| C64::new(*v, 0.0)).collect()
}

impl<V: Vector> Solver<V> for Minres<V> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        mut m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.work.as_ref().is_some_and(|w| w[0].same_shape(b)) {
            self.work = Some(std::array::from_fn(|_| b.zeros_like()));
        }
        let [r, v, v_prev, z, az, w, w_prev, v_new, z_new] =
            self.work.as_mut().expect("made above");
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let tol = self.stop.tolerance_sqr;
        let mut residual = vec![0.0; nv];
        let mut dots = vec![C64::ZERO; nv];
        let mut re_dot = |u: &V, v: &V| -> Vec<f64> {
            u.dot(v, &mut dots);
            dots.iter().map(|z| z.re).collect()
        };
        // The M⁻¹-norm of b: the stopping rule is relative to it in every cycle.
        match m.as_mut() {
            Some(m) => m.mult(r, b),
            None => r.copy_from(b),
        }
        let reference: Vec<f64> = re_dot(r, b).iter().map(|g| g.max(0.0)).collect();
        let (mut iterations, mut cycles) = (0, 0);
        loop {
            // The correction e with A e = r = b − A x, from zero.
            a.mult_xpay(r, x, &minus_one, b);
            v.copy_from(r);
            v_prev.scale(&vec![0.0; nv]);
            match m.as_mut() {
                Some(m) => m.mult(z, v),
                None => z.copy_from(v),
            }
            let mut gamma: Vec<f64> = re_dot(z, v).iter().map(|g| g.max(0.0).sqrt()).collect();
            let mut gamma_prev = vec![1.0_f64; nv];
            let mut eta = gamma.clone();
            let (mut c, mut c_prev, mut s, mut s_prev) =
                (vec![1.0; nv], vec![1.0; nv], vec![0.0; nv], vec![0.0; nv]);
            w.scale(&vec![0.0; nv]);
            w_prev.scale(&vec![0.0; nv]);
            let done = |eta: &[f64]| (0..nv).all(|j| eta[j] * eta[j] <= tol * reference[j]);
            let mut in_cycle = 0;
            while in_cycle < self.stop.max_iterations && !done(&eta) {
                let inv_gamma: Vec<f64> = gamma.iter().map(|g| 1.0 / g.max(1e-300)).collect();
                z.scale(&inv_gamma);
                a.mult(az, z);
                let delta = re_dot(az, z);
                // v_new = A z − (δ/γ) v − (γ/γ_prev) v_prev
                v_new.copy_from(az);
                let k1: Vec<f64> = (0..nv).map(|j| -delta[j] / gamma[j].max(1e-300)).collect();
                // γ is zero for a vector that is already solved (a zero right-hand side).
                let k2: Vec<f64> = (0..nv)
                    .map(|j| -gamma[j] / gamma_prev[j].max(1e-300))
                    .collect();
                v_new.axpy(&real(&k1), v);
                v_new.axpy(&real(&k2), v_prev);
                match m.as_mut() {
                    Some(m) => m.mult(z_new, v_new),
                    None => z_new.copy_from(v_new),
                }
                let gamma_new: Vec<f64> = re_dot(z_new, v_new)
                    .iter()
                    .map(|g| g.max(0.0).sqrt())
                    .collect();
                let (mut a1, mut a2, mut a3) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
                let (mut c_new, mut s_new) = (vec![0.0; nv], vec![0.0; nv]);
                for j in 0..nv {
                    let a0 = c[j] * delta[j] - c_prev[j] * s[j] * gamma[j];
                    a1[j] = (a0 * a0 + gamma_new[j] * gamma_new[j]).sqrt().max(1e-300);
                    a2[j] = s[j] * delta[j] + c_prev[j] * c[j] * gamma[j];
                    a3[j] = s_prev[j] * gamma[j];
                    c_new[j] = a0 / a1[j];
                    s_new[j] = gamma_new[j] / a1[j];
                }
                // w_new = (z − a3 w_prev − a2 w) / a1, kept in w_prev; then x += c_new η w_new.
                let inv_a1: Vec<f64> = a1.iter().map(|v| 1.0 / v).collect();
                let m2: Vec<f64> = (0..nv).map(|j| -a2[j] * inv_a1[j]).collect();
                let m3: Vec<f64> = (0..nv).map(|j| -a3[j] * inv_a1[j]).collect();
                w_prev.scale(&m3);
                w_prev.axpy(&real(&inv_a1), z);
                w_prev.axpy(&real(&m2), w);
                std::mem::swap(w, w_prev);
                let step: Vec<f64> = (0..nv).map(|j| c_new[j] * eta[j]).collect();
                x.axpy(&real(&step), w);
                for j in 0..nv {
                    eta[j] *= -s_new[j];
                }
                std::mem::swap(v_prev, v);
                std::mem::swap(v, v_new);
                std::mem::swap(z, z_new);
                gamma_prev = gamma;
                gamma = gamma_new;
                c_prev = c;
                c = c_new;
                s_prev = s;
                s = s_new;
                iterations += 1;
                in_cycle += 1;
            }
            true_residual(a, x, b, &b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: cycles,
                residual_sqr: residual.clone(),
            };
            if done(&eta) || residual.iter().all(|r| *r <= tol) {
                return Ok(info);
            }
            if residual.iter().any(|r| !r.is_finite()) {
                return Err(SolveError::Breakdown {
                    iterations,
                    what: "the solution is not finite",
                });
            }
            if cycles >= self.stop.max_restarts {
                return Err(SolveError::NotConverged(info));
            }
            cycles += 1;
        }
    }
}
