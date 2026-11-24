//! The cycle of a two-grid method on any kind of vector.

use crate::solver::Chebyshev;
use crate::{fit, LinearFn, Vector, C64};

/// A coarse correction: `e` from a residual `r`.
pub trait Corrects<V> {
    fn correct(&mut self, e: &mut V, r: &V);
}

impl<V, F: FnMut(&mut V, &V)> Corrects<V> for F {
    fn correct(&mut self, e: &mut V, r: &V) {
        self(e, r);
    }
}

/// The smoothing of a cycle on the operator A, and the residual of A.
pub trait Smooths<V> {
    /// y from b before the coarse correction.
    fn pre(&mut self, y: &mut V, b: &V);
    /// r = b − A y.
    fn residual(&mut self, r: &mut V, y: &V, b: &V);
    /// e ≈ A⁻¹ r after the coarse correction.
    fn post(&mut self, e: &mut V, r: &V);
}

/// A Chebyshev polynomial in A smooths before and after; it holds A.
impl<V: Vector, A: LinearFn<V>> Smooths<V> for Chebyshev<A, V> {
    fn pre(&mut self, y: &mut V, b: &V) {
        self.mult(y, b);
    }

    fn residual(&mut self, r: &mut V, y: &V, b: &V) {
        // The operator may have it in one sweep.
        let minus = vec![C64::new(-1.0, 0.0); b.num_vectors()];
        self.operator().mult_xpay(r, y, &minus, b);
    }

    fn post(&mut self, e: &mut V, r: &V) {
        self.mult(e, r);
    }
}

/// `y = S₁ b; y += E(b − A y); y += S₂(b − A y)`: S₁ and S₂ the smoothing, E the coarse
/// correction. A preconditioner for a flexible solver on A.
pub struct TwoGrid<V, S, E> {
    smoother: S,
    correction: E,
    r: V,
    e: V,
}

impl<V: Vector, S: Smooths<V>, E: Corrects<V>> TwoGrid<V, S, E> {
    pub fn new(smoother: S, correction: E, like: &V) -> Self {
        Self {
            smoother,
            correction,
            r: like.zeros_like(),
            e: like.zeros_like(),
        }
    }

    pub(crate) fn smoother_mut(&mut self) -> &mut S {
        &mut self.smoother
    }

    pub fn correction(&self) -> &E {
        &self.correction
    }

    pub(crate) fn correction_mut(&mut self) -> &mut E {
        &mut self.correction
    }
}

impl<V: Vector, S: Smooths<V>, E: Corrects<V>> LinearFn<V> for TwoGrid<V, S, E> {
    fn mult(&mut self, y: &mut V, b: &V) {
        assert!(y.same_shape(b));
        let one = vec![C64::new(1.0, 0.0); b.num_vectors()];
        fit(&mut self.r, b);
        fit(&mut self.e, b);
        // RUTTICE_DIAG=1 (exp/ls16-diag): |r| after each stage of the cycle, relative to the
        // stage before, and the cosine of the angle between b and Ã M b; one more residual.
        let diag = crate::env::switch("RUTTICE_DIAG");
        let nv = b.num_vectors();
        let norm = |v: &V| {
            let mut s = vec![0.0; nv];
            v.norm_sqr(&mut s);
            s[0].sqrt()
        };
        if diag && crate::env::DIAG_POWER.with(|p| p.replace(false)) {
            // Power iteration on the coarse correction E (e = E r): its dominant eigenvalue λ
            // and vector x. For an E that inverts Ã on range P, Ã x ≈ x/λ, so
            // μ = |λ| |Ã x|/|x| is 1; μ ≫ 1 means E inverts as small a direction Ã keeps O(1).
            let mut x = b.zeros_like();
            x.copy_from(b);
            let mut ex = b.zeros_like();
            let mut ax = b.zeros_like();
            let zero = b.zeros_like();
            for k in 1..=30 {
                let nx = norm(&x);
                x.scale(&vec![1.0 / nx; nv]);
                self.correction.correct(&mut ex, &x);
                let mut d = vec![C64::ZERO; nv];
                x.dot(&ex, &mut d);
                // Ã x = −(0 − Ã x)
                self.smoother.residual(&mut ax, &x, &zero);
                let gain = norm(&ex);
                if k % 5 == 0 || k == 1 {
                    eprintln!(
                        "diag power {k}: |E x|/|x| {gain:.4e} x†Ex {:.4e}{:+.4e}i |A x|/|x| {:.4e} mu {:.4e}",
                        d[0].re,
                        d[0].im,
                        norm(&ax),
                        gain * norm(&ax)
                    );
                }
                x.copy_from(&ex);
            }
        }
        let nb = if diag { norm(b) } else { 0.0 };
        self.smoother.pre(y, b);
        self.smoother.residual(&mut self.r, y, b);
        let n1 = if diag { norm(&self.r) } else { 0.0 };
        let r1 = diag.then(|| {
            let mut c = self.r.zeros_like();
            c.copy_from(&self.r);
            c
        });
        self.correction.correct(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
        self.smoother.residual(&mut self.r, y, b);
        let n2 = if diag { norm(&self.r) } else { 0.0 };
        if let Some(mut d) = r1 {
            // d = r1 − r2 = Ã e, the change the coarse correction makes to the residual; the
            // share of O_R d in range P (the rest is what range P fails to hold of Ã e), and
            // the share of O_R r1, both by the correction's own restriction.
            d.axpy(&vec![C64::new(-1.0, 0.0); nv], &self.r);
            let nd = norm(&d);
            let mut scratch = d.zeros_like();
            let share = |c: &mut E, v: &V, s: &mut V| {
                super::correction::DIAG_PROBE.with(|p| p.set(true));
                c.correct(s, v);
                super::correction::DIAG_PROBE.with(|p| p.set(false));
                super::correction::DIAG_SHARE.with(|x| x.get())
            };
            let s_d = share(&mut self.correction, &d, &mut scratch);
            let ne = norm(&self.e);
            eprintln!(
                "diag probe: |A e|/|r1| {:.3e} share of O_R A e in range P {:.4e} |e|/|r1| {:.3e} |A e|/|e| {:.3e}",
                nd / n1,
                s_d,
                ne / n1,
                nd / ne
            );
        }
        self.smoother.post(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
        if diag {
            // r3 = b − Ã y, then v = b − r3 = Ã M b.
            let mut r3 = b.zeros_like();
            self.smoother.residual(&mut r3, y, b);
            let n3 = norm(&r3);
            let mut v = b.zeros_like();
            v.copy_from(b);
            v.axpy(&vec![C64::new(-1.0, 0.0); nv], &r3);
            let mut d = vec![C64::ZERO; nv];
            b.dot(&v, &mut d);
            let cos = d[0].re / (nb * norm(&v));
            eprintln!(
                "diag cycle: pre {:.3e} coarse {:.3e} post {:.3e} whole {:.3e} cos(b, AMb) {:.4e}",
                n1 / nb,
                n2 / n1,
                n3 / n2,
                n3 / nb,
                cos
            );
        }
    }
}
