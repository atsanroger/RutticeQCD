//! BiCGStab for general (non-Hermitian) operators.

use crate::{true_residual, SolveError, SolveInfo, Solver, Stop};
use ruttice_core::{LinearFn, Vector, C64};

/// BiCGStab with an optional right preconditioner. Every vector of a multi-vector has its
/// own coefficients; a converged vector stops being updated.
///
/// The shadow residual is the residual at the start of a cycle. BiCGStab breaks down when
/// the shadow residual becomes orthogonal to the vectors it is tested against; this is not
/// exotic: a point source on a Wilson-type operator does it on the second step, because
/// hopping out and straight back is projected to zero. A breakdown after at least one
/// step therefore starts a new cycle from the current solution, with a new shadow
/// residual. Only a breakdown on the very first step of a cycle is an error.
pub struct BiCgStab<V> {
    pub stop: Stop,
    work: Option<Work<V>>,
}

struct Work<V> {
    r: V,
    r0: V,
    p: V,
    v: V,
    s: V,
    t: V,
    p_hat: V,
    s_hat: V,
}

impl<V: Vector> BiCgStab<V> {
    pub fn new(stop: Stop) -> Self {
        Self { stop, work: None }
    }
}

fn usable(z: C64) -> bool {
    z.norm_sqr() > 0.0 && z.re.is_finite() && z.im.is_finite()
}

impl<V: Vector> Solver<V> for BiCgStab<V> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        mut m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.work.as_ref().is_some_and(|w| w.r.same_shape(b)) {
            let z = || b.zeros_like();
            self.work = Some(Work {
                r: z(),
                r0: z(),
                p: z(),
                v: z(),
                s: z(),
                t: z(),
                p_hat: z(),
                s_hat: z(),
            });
        }
        let Some(Work {
            r,
            r0,
            p,
            v,
            s,
            t,
            p_hat,
            s_hat,
        }) = self.work.as_mut()
        else {
            unreachable!("the work vectors were just created");
        };

        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let tol = self.stop.tolerance_sqr;
        let limit = self.stop.max_iterations * (self.stop.max_restarts + 1);

        let mut rr = vec![0.0; nv];
        let mut rho = vec![C64::ZERO; nv];
        let mut rho_new = vec![C64::ZERO; nv];
        let mut alpha = vec![C64::ZERO; nv];
        let mut omega = vec![C64::ZERO; nv];
        let mut coef = vec![C64::ZERO; nv];
        let mut dots = vec![C64::ZERO; nv];
        let mut tt = vec![0.0; nv];
        let mut active = vec![true; nv];
        let mut residual = vec![0.0; nv];
        let mut iterations = 0;
        // Cycles that ended at the iteration limit; cycles ended by a breakdown are counted
        // in `cycles` only.
        let mut exhausted = 0;
        let mut cycles = 0;

        loop {
            a.mult_xpay(r, x, &minus_one, b);
            r0.copy_from(r);
            p.copy_from(r);
            r.norm_sqr(&mut rr);
            r0.dot(r, &mut rho);
            let mut in_cycle = 0;
            let mut broke_down: Option<&'static str> = None;

            while in_cycle < self.stop.max_iterations && iterations < limit {
                for j in 0..nv {
                    active[j] = rr[j] > tol * b_norm[j];
                }
                if !active.iter().any(|a| *a) {
                    break;
                }

                // v = A M p,  alpha = rho / <r0, v>
                match m.as_mut() {
                    Some(m) => m.mult(p_hat, p),
                    None => p_hat.copy_from(p),
                }
                a.mult(v, p_hat);
                r0.dot(v, &mut dots);
                if (0..nv).any(|j| active[j] && !usable(dots[j])) {
                    broke_down = Some("<r0, A p> vanished");
                    break;
                }
                for j in 0..nv {
                    alpha[j] = if active[j] {
                        rho[j] / dots[j]
                    } else {
                        C64::ZERO
                    };
                    coef[j] = -alpha[j];
                }
                // s = r - alpha v
                s.copy_from(r);
                s.axpy(&coef, v);

                // t = A M s,  omega = <t, s> / <t, t>
                match m.as_mut() {
                    Some(m) => m.mult(s_hat, s),
                    None => s_hat.copy_from(s),
                }
                a.mult(t, s_hat);
                t.dot(s, &mut dots);
                t.norm_sqr(&mut tt);
                for j in 0..nv {
                    omega[j] = if active[j] && tt[j] > 0.0 {
                        C64::new(dots[j].re / tt[j], dots[j].im / tt[j])
                    } else {
                        C64::ZERO
                    };
                    coef[j] = -omega[j];
                }

                // x += alpha M p + omega M s,  r = s - omega t
                x.axpy(&alpha, p_hat);
                x.axpy(&omega, s_hat);
                r.copy_from(s);
                r.axpy_norm_sqr(&coef, t, &mut rr);
                iterations += 1;
                in_cycle += 1;

                // p = r + beta (p - omega v),  beta = (rho_new / rho) (alpha / omega)
                r0.dot(r, &mut rho_new);
                let still_active = |j: usize| active[j] && rr[j] > tol * b_norm[j];
                if (0..nv).any(|j| still_active(j) && !(usable(rho_new[j]) && usable(omega[j]))) {
                    broke_down = Some("the new rho or omega vanished");
                    break;
                }
                for j in 0..nv {
                    coef[j] = -omega[j];
                    dots[j] = if still_active(j) {
                        (rho_new[j] / rho[j]) * (alpha[j] / omega[j])
                    } else {
                        C64::ZERO
                    };
                }
                p.axpy(&coef, v);
                p.aypx(&dots, r);
                rho.copy_from_slice(&rho_new);
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
            match broke_down {
                // No step was taken in this cycle: a new cycle would start from the same
                // state and break down the same way.
                Some(what) if in_cycle == 0 => {
                    return Err(SolveError::Breakdown { iterations, what });
                }
                Some(_) => {}
                None => exhausted += 1,
            }
            if exhausted > self.stop.max_restarts || iterations >= limit {
                return Err(SolveError::NotConverged(info));
            }
            cycles += 1;
        }
    }
}
