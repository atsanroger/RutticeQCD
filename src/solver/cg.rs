//! Conjugate gradient for Hermitian positive-definite operators.

use super::{true_residual, SolveError, SolveInfo, Solver, Stop};
use crate::{LinearFn, Vector, C64};

/// Conjugate gradient, optionally preconditioned (the preconditioner must then be Hermitian
/// positive definite and the same at every application).
///
/// Every vector of a multi-vector has its own coefficients and its own convergence test;
/// a converged vector stops being updated while the others continue.
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

        for restart in 0..=self.stop.max_restarts {
            // r = b - A x
            a.mult_xpay(r, x, &minus_one, b);
            r.norm_sqr(&mut rr);
            match m.as_mut() {
                Some(m) => m.mult(z, r),
                None => z.copy_from(r),
            }
            p.copy_from(z);
            r.dot(z, &mut rz);

            for _ in 0..self.stop.max_iterations {
                if converged(&rr, &b_norm) {
                    break;
                }
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
                x.axpy(&alpha, p);
                for a in alpha.iter_mut() {
                    *a = -*a;
                }
                r.axpy_norm_sqr(&alpha, ap, &mut rr);

                match m.as_mut() {
                    Some(m) => m.mult(z, r),
                    None => z.copy_from(r),
                }
                r.dot(z, &mut rz_new);
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
                p.aypx(&beta, z);
                iterations += 1;
            }

            true_residual(a, x, b, &b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: restart,
                residual_sqr: residual.clone(),
            };
            if residual.iter().all(|r| *r <= tol) {
                return Ok(info);
            }
            if restart == self.stop.max_restarts {
                return Err(SolveError::NotConverged(info));
            }
        }
        unreachable!("the loop returns on its last pass");
    }
}
