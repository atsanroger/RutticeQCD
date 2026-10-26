//! Conjugate gradient for Hermitian positive-definite operators.

use super::{
    fewer, kept, not_finite, true_residual, Narrowed, SolveError, SolveInfo, Solver, Stop,
};
use crate::{LinearFn, Vector, C64};

/// Conjugate gradient, optionally preconditioned (the preconditioner must then be Hermitian
/// positive definite and the same at every application).
pub struct Cg<V> {
    pub stop: Stop,
    work: Option<Work<V>>,
}

struct Work<V> {
    r: V,
    p: V,
    ap: V,
    z: V,
}

impl<V: Vector> Cg<V> {
    pub fn new(stop: Stop) -> Self {
        Self { stop, work: None }
    }
}

fn real(x: f64) -> C64 {
    C64::new(x, 0.0)
}

impl<V: Vector> Solver<V> for Cg<V> {
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
            self.work = Some(Work {
                r: b.zeros_like(),
                p: b.zeros_like(),
                ap: b.zeros_like(),
                z: b.zeros_like(),
            });
        }
        let Some(Work { r, p, ap, z }) = self.work.as_mut() else {
            unreachable!("the work vectors were just created");
        };

        let minus_one = vec![real(-1.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let tol = self.stop.tolerance_sqr;
        let converged =
            |rr: &[f64], b_norm: &[f64]| rr.iter().zip(b_norm).all(|(r, n)| *r <= tol * n);

        let mut rr = vec![0.0; nv];
        let mut rz = vec![C64::ZERO; nv];
        let mut rz_new = vec![C64::ZERO; nv];
        let mut pap = vec![C64::ZERO; nv];
        let mut alpha = vec![C64::ZERO; nv];
        let mut beta = vec![C64::ZERO; nv];
        let mut residual = vec![0.0; nv];
        let mut iterations = 0;

        let all_b_norm = b_norm.clone();
        for restart in 0..=self.stop.max_restarts {
            // A cycle starts with all vectors; the previous one may have ended with fewer.
            for w in [&mut *r, &mut *p, &mut *ap] {
                crate::fit(w, b);
            }
            rr.resize(nv, 0.0);
            for list in [&mut rz, &mut rz_new, &mut pap, &mut alpha, &mut beta] {
                list.resize(nv, C64::ZERO);
            }
            let mut b_norm = all_b_norm.clone();
            let mut narrowed: Option<Narrowed<V>> = None;

            // r = b - A x
            a.mult_xpay(r, x, &minus_one, b);
            r.norm_sqr(&mut rr);
            // A NaN residual counts as neither converged nor active: without this the loop
            // would apply A to the end of its iterations and report `NotConverged`.
            if not_finite(&rr) {
                return Err(SolveError::Breakdown {
                    iterations,
                    what: "the residual is not finite",
                });
            }
            // Without a preconditioner z is r: it is not copied, and <r, z> is |r|^2.
            match m.as_mut() {
                Some(m) => {
                    m.mult(z, r);
                    p.copy_from(z);
                    r.dot(z, &mut rz);
                    if (0..nv).any(|j| {
                        rr[j] > tol * b_norm[j] && !(rz[j].re > 0.0 && rz[j].re.is_finite())
                    }) {
                        return Err(SolveError::Breakdown {
                            iterations,
                            what: "<r, M r> is not positive",
                        });
                    }
                }
                None => {
                    p.copy_from(r);
                    for (rz, rr) in rz.iter_mut().zip(&rr) {
                        *rz = real(*rr);
                    }
                }
            }

            for _ in 0..self.stop.max_iterations {
                if converged(&rr, &b_norm) {
                    break;
                }
                let active: Vec<bool> = (0..rr.len()).map(|j| rr[j] > tol * b_norm[j]).collect();
                if let Some(keep) = fewer(&active).filter(|_| m.is_none()) {
                    narrowed = Some(Narrowed::narrow(narrowed, x, &keep));
                    for w in [&mut *r, &mut *p] {
                        *w = w.select(&keep);
                    }
                    *ap = ap.zeros_with(keep.len());
                    for list in [&mut rr, &mut b_norm] {
                        kept(list, &keep);
                    }
                    for list in [&mut rz, &mut rz_new, &mut pap, &mut alpha, &mut beta] {
                        kept(list, &keep);
                    }
                }
                let nv = rr.len();
                a.mult_dot(ap, p, &mut pap);
                for j in 0..nv {
                    let active = rr[j] > tol * b_norm[j];
                    if active && !(pap[j].re > 0.0 && pap[j].re.is_finite()) {
                        return Err(SolveError::Breakdown {
                            iterations,
                            what: "<p, A p> is not positive",
                        });
                    }
                    alpha[j] = if active {
                        real(rz[j].re / pap[j].re)
                    } else {
                        C64::ZERO
                    };
                }
                narrowed
                    .as_mut()
                    .map_or(&mut *x, |n| &mut n.x)
                    .axpy(&alpha, p);
                for a in alpha.iter_mut() {
                    *a = -*a;
                }
                r.axpy_norm_sqr(&alpha, ap, &mut rr);
                if not_finite(&rr) {
                    return Err(SolveError::Breakdown {
                        iterations,
                        what: "the residual is not finite",
                    });
                }

                match m.as_mut() {
                    Some(m) => {
                        m.mult(z, r);
                        r.dot(z, &mut rz_new);
                        if (0..nv).any(|j| {
                            rr[j] > tol * b_norm[j]
                                && !(rz_new[j].re > 0.0 && rz_new[j].re.is_finite())
                        }) {
                            return Err(SolveError::Breakdown {
                                iterations,
                                what: "<r, M r> is not positive",
                            });
                        }
                    }
                    None => {
                        for (rz, rr) in rz_new.iter_mut().zip(&rr) {
                            *rz = real(*rr);
                        }
                    }
                }
                for j in 0..nv {
                    // A vector whose alpha was zero keeps its old r and z, so beta is one for
                    // it and p stays where it was; force zero to keep p = z exactly.
                    beta[j] = if alpha[j].re != 0.0 {
                        real(rz_new[j].re / rz[j].re)
                    } else {
                        C64::ZERO
                    };
                }
                rz.copy_from_slice(&rz_new);
                p.aypx(&beta, if m.is_some() { &*z } else { &*r });
                iterations += 1;
            }

            // Back to all vectors for the residual of the solution as it now is.
            if let Some(last) = narrowed {
                x.assign(&last.of, &last.x);
            }
            crate::fit(r, b);
            true_residual(a, x, b, &all_b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: restart,
                residual_sqr: residual.clone(),
            };
            if residual.iter().all(|r| *r <= tol) {
                return Ok(info);
            }
            if not_finite(&residual) {
                return Err(SolveError::Breakdown {
                    iterations,
                    what: "the solution is not finite",
                });
            }
            if restart == self.stop.max_restarts {
                return Err(SolveError::NotConverged(info));
            }
        }
        unreachable!("the loop returns on its last pass");
    }
}
