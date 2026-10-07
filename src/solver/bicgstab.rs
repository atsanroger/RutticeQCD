//! BiCGStab for general (non-Hermitian) operators.

use super::{fewer, kept, true_residual, Narrowed, SolveError, SolveInfo, Solver, Stop};
use crate::{LinearFn, Vector, C64};

/// BiCGStab with an optional right preconditioner. Every vector of a multi-vector has its
/// own coefficients; a converged vector stops being updated. Without a preconditioner,
/// once a quarter of the vectors have converged the others are gathered into a smaller set
/// and the iteration goes on with that: the operator and the updates no longer run over
/// vectors that are finished. The recurrences are per vector, so the iterates of the
/// vectors that go on are unchanged.
///
/// With a preconditioner M the solution is updated with the preconditioned vectors
/// themselves (x += α M p + ω M s), so M may change from one application to the next: this
/// is the flexible BiCGStab of Bridge++ (`ASolver_FBiCGStab`), usable with a multigrid
/// preconditioner.
///
/// ω is stabilised as in Bridge++ (`Omega_tolerance`): when |⟨t, s⟩| / (|t| |s|) is below
/// `omega_tolerance`, ω is scaled up by `omega_tolerance` divided by that ratio. Bridge++
/// uses 0.7, the default here; 0 turns it off.
///
/// The shadow residual is the residual at the start of a cycle. BiCGStab breaks down when
/// the shadow residual becomes orthogonal to the vectors it is tested against; this is not
/// exotic: a point source on a Wilson-type operator does it on the second step, because
/// hopping out and straight back is projected to zero. A breakdown after at least one
/// step therefore starts a new cycle from the current solution, with a new shadow
/// residual. Only a breakdown on the very first step of a cycle is an error.
pub struct BiCgStab<V> {
    pub stop: Stop,
    pub omega_tolerance: f64,
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
        Self {
            stop,
            omega_tolerance: 0.7,
            work: None,
        }
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
        let all_b_norm = b_norm.clone();
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
        let mut ss = vec![0.0; nv];
        let mut active = vec![true; nv];
        let mut residual = vec![0.0; nv];
        let mut iterations = 0;
        // Cycles that ended at the iteration limit; cycles ended by a breakdown are counted
        // in `cycles` only.
        let mut exhausted = 0;
        let mut cycles = 0;

        loop {
            // A cycle starts with all vectors; the previous one may have ended with fewer.
            for w in [&mut *r, &mut *r0, &mut *p, &mut *v, &mut *s, &mut *t] {
                crate::fit(w, b);
            }
            for list in [&mut rr, &mut tt, &mut ss] {
                list.resize(nv, 0.0);
            }
            for list in [
                &mut rho,
                &mut rho_new,
                &mut alpha,
                &mut omega,
                &mut coef,
                &mut dots,
            ] {
                list.resize(nv, C64::ZERO);
            }
            active.resize(nv, true);
            // |b|^2 of the vectors the cycle is working on, and their solution once they
            // are fewer than all.
            let mut b_norm = b_norm.clone();
            let mut narrowed: Option<Narrowed<V>> = None;

            a.mult_xpay(r, x, &minus_one, b);
            r0.copy_from(r);
            p.copy_from(r);
            r.norm_sqr(&mut rr);
            r0.dot(r, &mut rho);
            let mut in_cycle = 0;
            let mut broke_down: Option<&'static str> = None;

            while in_cycle < self.stop.max_iterations && iterations < limit {
                for j in 0..active.len() {
                    active[j] = rr[j] > tol * b_norm[j];
                }
                if !active.iter().any(|a| *a) {
                    break;
                }
                if let Some(keep) = fewer(&active).filter(|_| m.is_none()) {
                    narrowed = Some(Narrowed::narrow(narrowed, x, &keep));
                    for w in [&mut *r, &mut *r0, &mut *p] {
                        *w = w.select(&keep);
                    }
                    for w in [&mut *v, &mut *s, &mut *t] {
                        *w = w.zeros_with(keep.len());
                    }
                    for list in [&mut rr, &mut tt, &mut ss, &mut b_norm] {
                        kept(list, &keep);
                    }
                    for list in [
                        &mut rho,
                        &mut rho_new,
                        &mut alpha,
                        &mut omega,
                        &mut coef,
                        &mut dots,
                    ] {
                        kept(list, &keep);
                    }
                    kept(&mut active, &keep);
                }
                let nv = active.len();

                // v = A M p,  alpha = rho / <r0, v>. Without a preconditioner M p is p
                // itself, not a copy of it.
                let preconditioned = m.is_some();
                if let Some(m) = m.as_mut() {
                    m.mult(p_hat, p);
                }
                a.mult(v, if preconditioned { &*p_hat } else { &*p });
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
                s.xpay(r, &coef, v);

                // t = A M s,  omega = <t, s> / <t, t>
                if let Some(m) = m.as_mut() {
                    m.mult(s_hat, s);
                }
                a.mult(t, if preconditioned { &*s_hat } else { &*s });
                t.dot_norms(s, &mut dots, &mut tt, &mut ss);
                for j in 0..nv {
                    omega[j] = if active[j] && tt[j] > 0.0 {
                        let mut w = C64::new(dots[j].re / tt[j], dots[j].im / tt[j]);
                        let cosine = dots[j].norm_sqr().sqrt() / (tt[j] * ss[j]).sqrt();
                        if cosine > 0.0 && cosine < self.omega_tolerance {
                            w = w.scale(self.omega_tolerance / cosine);
                        }
                        w
                    } else {
                        C64::ZERO
                    };
                    coef[j] = -omega[j];
                }

                // x += alpha M p + omega M s,  r = s - omega t,  rho_new = <r0, r>
                let x = narrowed.as_mut().map_or(&mut *x, |n| &mut n.x);
                if preconditioned {
                    x.axpy2(&alpha, p_hat, &omega, s_hat);
                } else {
                    x.axpy2(&alpha, p, &omega, s);
                }
                r.xpay_norm_sqr_dot(s, &coef, t, &mut rr, r0, &mut rho_new);
                iterations += 1;
                in_cycle += 1;

                // p = r + beta (p - omega v),  beta = (rho_new / rho) (alpha / omega)
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
                p.axpy_aypx(&coef, v, &dots, r);
                rho.copy_from_slice(&rho_new);
            }

            // Back to all vectors for the residual of the solution as it now is.
            if let Some(last) = narrowed {
                x.assign(&last.of, &last.x);
            }
            crate::fit(r, b);
            true_residual(a, x, b, &all_b_norm, r, &minus_one, &mut residual);
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
