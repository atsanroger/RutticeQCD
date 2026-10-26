//! Conjugate gradient for several shifts at once: (A + σ_i) x_i = b for a Hermitian positive
//! A and σ_i ≥ 0 (or at least A + σ_i positive), all from the Krylov space of the first
//! system, at the cost of one CG and a few vector updates per shift. The algorithm and its
//! stopping rules are those of Bridge++'s `Shiftsolver_CG` (Jegerlehner's multi-shift CG):

use super::{SolveError, SolveInfo, Stop};
use crate::{LinearFn, Vector, C64};

/// Multi-shift CG; `stop.max_iterations` bounds the iterations, there are no restarts.
pub struct MultiShiftCg {
    pub stop: Stop,
}

impl MultiShiftCg {
    pub fn new(stop: Stop) -> Self {
        Self { stop }
    }

    /// Solve (A + σ_i) x_i = b for every shift; `xs` gets one vector per shift (any content
    /// on entry is replaced: the solves start from zero). `b` holds one vector. The info has
    /// the true |b − (A + σ_i) x_i|²/|b|² per shift.
    ///
    /// The solve stops on the residual of the first shift alone, as Bridge++'s does: the
    /// first shift must be the smallest, whose system converges last (the poles of
    /// `RationalApprox` are in increasing order, and so are those of Bridge++'s files).
    pub fn solve<V: Vector>(
        &mut self,
        a: &mut dyn LinearFn<V>,
        shifts: &[f64],
        xs: &mut Vec<V>,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        assert_eq!(b.num_vectors(), 1, "one right-hand side");
        let n = shifts.len();
        assert!(n > 0, "at least one shift");
        assert!(
            shifts.iter().all(|s| *s >= shifts[0]),
            "the first shift is the smallest: the solve stops on its residual"
        );
        let real = |x: f64| [C64::new(x, 0.0)];
        let mut b_norm = [0.0];
        b.norm_sqr(&mut b_norm);
        let snorm = 1.0 / b_norm[0];
        let tol = self.stop.tolerance_sqr;
        xs.clear();
        xs.extend((0..n).map(|_| b.zeros_like()));
        let mut ps: Vec<V> = (0..n)
            .map(|_| {
                let mut p = b.zeros_like();
                p.copy_from(b);
                p
            })
            .collect();
        let mut r = b.zeros_like();
        r.copy_from(b);
        let mut s = b.zeros_like();
        let mut rr = b_norm[0];
        let mut zeta1 = vec![1.0; n];
        let mut zeta2 = vec![1.0; n];
        let mut pp = vec![0.0; n];
        let mut active = n;
        // Bridge++'s variables: `alpha` is CG's β, `beta` is −α.
        let (mut alpha_p, mut beta_p) = (0.0, 1.0);
        let mut iterations = 0;
        let mut converged = rr * snorm < tol;
        while !converged && iterations < self.stop.max_iterations {
            // s = (A + σ_0) p_0.
            a.mult(&mut s, &ps[0]);
            s.axpy(&real(shifts[0]), &ps[0]);
            let mut pap = [C64::ZERO];
            s.dot(&ps[0], &mut pap);
            let rr_p = rr;
            let beta = -rr_p / pap[0].re;
            if !(beta.is_finite() && beta < 0.0) {
                return Err(SolveError::Breakdown {
                    iterations,
                    what: "<p, (A + σ) p> is not positive",
                });
            }
            xs[0].axpy(&real(-beta), &ps[0]);
            let mut n_r = [0.0];
            r.axpy_norm_sqr(&real(beta), &s, &mut n_r);
            rr = n_r[0];
            let alpha = rr / rr_p;
            ps[0].aypx(&real(alpha), &r);
            pp[0] = rr;
            let alpha_h = 1.0 + alpha_p * beta / beta_p;
            for i in 1..active {
                let shift = shifts[i] - shifts[0];
                let zeta = 1.0 / ((alpha_h - shift * beta) / zeta1[i] + (1.0 - alpha_h) / zeta2[i]);
                let zr = zeta / zeta1[i];
                xs[i].axpy(&real(-beta * zr), &ps[i]);
                ps[i].scale(&[alpha * zr * zr]);
                ps[i].axpy(&real(zeta), &r);
                let mut n_p = [0.0];
                ps[i].norm_sqr(&mut n_p);
                pp[i] = n_p[0] * snorm;
                zeta2[i] = zeta1[i];
                zeta1[i] = zeta;
            }
            // Drop the converged shifts from the end of the list (Bridge++ compares the
            // first entry, |r|² not divided by |b|², the same way; it stops the loop anyway).
            if let Some(last) = (0..active).rev().find(|&i| pp[i] > tol) {
                active = last + 1;
            }
            alpha_p = alpha;
            beta_p = beta;
            iterations += 1;
            converged = rr * snorm < tol;
        }
        let mut residual_sqr = Vec::with_capacity(n);
        for (x, sigma) in xs.iter().zip(shifts) {
            a.mult(&mut s, x);
            s.axpy(&real(*sigma), x);
            s.axpy(&real(-1.0), b);
            let mut d = [0.0];
            s.norm_sqr(&mut d);
            residual_sqr.push(d[0] * snorm);
        }
        let info = SolveInfo {
            iterations,
            restarts: 0,
            residual_sqr,
        };
        if converged {
            Ok(info)
        } else {
            Err(SolveError::NotConverged(info))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::{Bound, FermionOp, Variant, Wilson, WilsonParams};
    use crate::gauge::random_gauge;
    use crate::solver::{Cg, Solver};
    use crate::{random_spinor, Boundary, Field, Lattice, Normal, Spinor, ND};

    /// Each shifted solution is that of CG on its own system, on D†D of the Wilson operator
    /// on a random gauge field, and the true residuals meet the tolerance.
    #[test]
    fn shifted_solutions_are_those_of_cg() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 31);
        let mut d = Wilson::<f64>::new(
            &lat,
            WilsonParams {
                kappa: 0.12,
                boundary: [Boundary::Periodic; ND],
            },
        );
        d.set_gauge(&u).unwrap();
        let b: Field<f64, Spinor> = random_spinor(&lat, 1, 5);
        let shifts = [1e-4, 3e-3, 0.05, 0.4, 2.0, 30.0];
        let stop = Stop::simple(1e-24, 2000);
        let mut bound = Bound::new(&d, Variant::D);
        let mut normal = Normal::new(&mut bound, &b);
        let mut xs = Vec::new();
        let info = MultiShiftCg::new(stop)
            .solve(&mut normal, &shifts, &mut xs, &b)
            .unwrap();
        eprintln!("{info:?}");
        assert!(info.residual_sqr.iter().all(|r| *r < 1e-22));
        for (x, sigma) in xs.iter().zip(shifts) {
            // CG on D†D + σ.
            struct Shifted<'a, A>(&'a mut A, f64);
            impl<A: LinearFn<Field<f64, Spinor>>> LinearFn<Field<f64, Spinor>> for Shifted<'_, A> {
                fn mult(&mut self, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>) {
                    self.0.mult(out, inp);
                    out.axpy(&[C64::new(self.1, 0.0)], inp);
                }
            }
            let mut y = b.zeros_like();
            Cg::new(stop)
                .solve(&mut Shifted(&mut normal, sigma), None, &mut y, &b)
                .unwrap();
            let mut diff = y.zeros_like();
            diff.copy_from(&y);
            diff.axpy(&[C64::new(-1.0, 0.0)], x);
            let (mut nd, mut ny) = ([0.0], [0.0]);
            diff.norm_sqr(&mut nd);
            y.norm_sqr(&mut ny);
            assert!(nd[0] < 1e-20 * ny[0], "shift {sigma}: {}", nd[0] / ny[0]);
        }
    }
}
