//! Flexible GMRES: for general operators, with a preconditioner that may change from one
//! application to the next (an inner iterative solver, a multigrid cycle).

use super::{true_residual, SolveError, SolveInfo, Solver, Stop};
use crate::numerics::HessenbergLsq;
use crate::{LinearFn, Vector, C64};

/// Restarted flexible GMRES with right preconditioning.
pub struct Fgmres<V> {
    pub stop: Stop,
    work: Option<Work<V>>,
}

struct Work<V> {
    r: V,
    w: V,
    /// The orthonormal basis, `krylov + 1` vectors.
    v: Vec<V>,
    /// The preconditioned basis vectors; allocated when a preconditioner is first used.
    z: Vec<V>,
}

impl<V: Vector> Fgmres<V> {
    pub fn new(stop: Stop) -> Self {
        assert!(
            stop.max_iterations > 0,
            "the Krylov space needs at least one vector"
        );
        Self { stop, work: None }
    }
}

impl<V: Vector> Solver<V> for Fgmres<V> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        mut m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        let krylov = self.stop.max_iterations;
        let fits = |w: &Work<V>| w.r.same_shape(b) && w.v.len() == krylov + 1;
        if !self.work.as_ref().is_some_and(fits) {
            self.work = Some(Work {
                r: b.zeros_like(),
                w: b.zeros_like(),
                v: (0..krylov + 1).map(|_| b.zeros_like()).collect(),
                z: Vec::new(),
            });
        }
        let Some(Work { r, w, v, z }) = self.work.as_mut() else {
            unreachable!("the work vectors were just created");
        };
        if m.is_some() && z.len() != krylov {
            *z = (0..krylov).map(|_| b.zeros_like()).collect();
        }

        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let tol = self.stop.tolerance_sqr;

        let mut lsq: Vec<HessenbergLsq> = (0..nv).map(|_| HessenbergLsq::new(krylov)).collect();
        let mut h = vec![vec![C64::ZERO; krylov + 1]; nv];
        let mut y = vec![vec![C64::ZERO; krylov]; nv];
        let mut estimate = vec![0.0; nv];
        let mut coef = vec![C64::ZERO; nv];
        let mut norms = vec![0.0; nv];
        let mut scale = vec![0.0; nv];
        let mut w_norm = vec![0.0; nv];
        let mut basis_dots = vec![vec![C64::ZERO; nv]; krylov + 1];
        let mut residual = vec![0.0; nv];
        let mut iterations = 0;

        for cycle in 0..=self.stop.max_restarts + 1 {
            // r = b - A x, and the true relative residual.
            true_residual(a, x, b, &b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: cycle.saturating_sub(1),
                residual_sqr: residual.clone(),
            };
            if residual.iter().all(|r| *r <= tol) {
                return Ok(info);
            }
            if cycle > self.stop.max_restarts {
                return Err(SolveError::NotConverged(info));
            }

            // v_0 = r / |r|. A vector that has converged gets a zero basis and takes no
            // further part in this cycle.
            r.norm_sqr(&mut norms);
            for j in 0..nv {
                let active = residual[j] > tol && norms[j] > 0.0;
                let beta = norms[j].sqrt();
                scale[j] = if active { 1.0 / beta } else { 0.0 };
                lsq[j].reset(if active { beta } else { 0.0 });
            }
            v[0].copy_from(r);
            v[0].scale(&scale);

            let mut built = 0;
            for k in 0..krylov {
                // w = A M v_k
                match m.as_mut() {
                    Some(m) => {
                        m.mult(&mut z[k], &v[k]);
                        a.mult(w, &z[k]);
                    }
                    None => a.mult(w, &v[k]),
                }
                // Orthogonalise against the basis: classical Gram-Schmidt, twice, with the
                // second pass's corrections folded into H. One pass loses orthogonality on
                // strongly non-normal operators (the domain-wall multigrid target), as in
                // the GMRES(m) of the CUDA domain-wall multigrid this follows.
                w.norm_sqr(&mut w_norm);
                for pass in 0..2 {
                    for (i, vi) in v.iter().enumerate().take(k + 1) {
                        vi.dot(w, &mut basis_dots[i]);
                    }
                    for (i, vi) in v.iter().enumerate().take(k + 1) {
                        for j in 0..nv {
                            let d = basis_dots[i][j];
                            h[j][i] = if pass == 0 { d } else { h[j][i] + d };
                            coef[j] = -d;
                        }
                        w.axpy(&coef, vi);
                    }
                }
                w.norm_sqr(&mut norms);
                for j in 0..nv {
                    let n = norms[j].sqrt();
                    h[j][k + 1] = C64::new(n, 0.0);
                    // A norm that is round-off compared with |A v_k| means the Krylov space
                    // is exhausted for this vector: its solution is already exact within the
                    // space. Scaling the round-off up would only add noise.
                    let exhausted = n <= 1e-12 * w_norm[j].sqrt() || n == 0.0;
                    if exhausted {
                        h[j][k + 1] = C64::ZERO;
                    }
                    scale[j] = if exhausted { 0.0 } else { 1.0 / n };
                    estimate[j] = lsq[j].push(&mut h[j][..k + 2]);
                }
                v[k + 1].copy_from(w);
                v[k + 1].scale(&scale);
                iterations += 1;
                built = k + 1;
                if (0..nv).all(|j| estimate[j] * estimate[j] <= tol * b_norm[j]) {
                    break;
                }
            }

            // x += Σ_i y_i (M v_i)
            for j in 0..nv {
                lsq[j].solve(&mut y[j][..built]);
            }
            for i in 0..built {
                for j in 0..nv {
                    coef[j] = y[j][i];
                }
                x.axpy(&coef, if m.is_some() { &z[i] } else { &v[i] });
            }
        }
        unreachable!("the loop returns on its last pass");
    }
}
