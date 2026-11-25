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

/// exp/ls16-diag: e ← ω e for ω = RUTTICE_COARSE_OMEGA (1 when unset): the coarse
/// correction damped against the smoothers.
fn damp_coarse<V: Vector>(e: &mut V) {
    let omega = std::env::var("RUTTICE_COARSE_OMEGA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    if omega != 1.0 {
        e.scale(&vec![omega; e.num_vectors()]);
    }
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

    /// exp/ls16-diag: x†Ã M x of a unit x split by the stages of the cycle, M x = y1 + e + f
    /// (y1 the pre-smoother's, e the coarse correction's, f the post-smoother's), each term
    /// x†Ã(·); the device correction prints the profiles of its input and output.
    fn diag_stages(&mut self, x: &V, y: &mut V, ay: &mut V) {
        let nv = x.num_vectors();
        let one = vec![C64::new(1.0, 0.0); nv];
        // x†Ã v = x†x − x†(x − Ã v) for a unit x, with no field besides the two given.
        let term = |s: &mut S, v: &V, out: &mut V| {
            s.residual(out, v, x);
            let mut d = vec![C64::ZERO; nv];
            x.dot(out, &mut d);
            C64::new(1.0, 0.0) - d[0]
        };
        self.smoother.pre(y, x);
        let t_pre = term(&mut self.smoother, y, ay);
        self.smoother.residual(&mut self.r, y, x);
        crate::env::DIAG_DETAIL.with(|f| f.set(true));
        self.correction.correct(&mut self.e, &self.r);
        crate::env::DIAG_DETAIL.with(|f| f.set(false));
        damp_coarse(&mut self.e);
        let t_coarse = term(&mut self.smoother, &self.e, ay);
        y.axpy(&one, &self.e);
        self.smoother.residual(&mut self.r, y, x);
        self.smoother.post(&mut self.e, &self.r);
        let t_post = term(&mut self.smoother, &self.e, ay);
        let total = t_pre + t_coarse + t_post;
        eprintln!(
            "diag stages: x†A(pre) {:.4e}{:+.4e}i x†A(coarse) {:.4e}{:+.4e}i x†A(post) {:.4e}{:+.4e}i sum {:.4e}{:+.4e}i",
            t_pre.re, t_pre.im, t_coarse.re, t_coarse.im, t_post.re, t_post.im, total.re, total.im
        );
    }

    /// exp/ls16-diag: `y = M b`, one cycle without the diagnostics.
    fn plain_cycle(&mut self, y: &mut V, b: &V) {
        let one = vec![C64::new(1.0, 0.0); b.num_vectors()];
        self.smoother.pre(y, b);
        self.smoother.residual(&mut self.r, y, b);
        self.correction.correct(&mut self.e, &self.r);
        damp_coarse(&mut self.e);
        y.axpy(&one, &self.e);
        self.smoother.residual(&mut self.r, y, b);
        self.smoother.post(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
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
            let mut ex = b.zeros_like();
            let mut ax = b.zeros_like();
            let zero = b.zeros_like();
            // The whole E, then the quark factor alone, then the Pauli-Villars factor alone.
            for (factor, label) in [(0u8, "E"), (1, "quark"), (2, "pv")] {
                super::correction::DIAG_FACTOR.with(|f| f.set(factor));
                let mut x = b.zeros_like();
                x.copy_from(b);
                for k in 1..=30 {
                    let nx = norm(&x);
                    x.scale(&vec![1.0 / nx; nv]);
                    crate::env::DIAG_DETAIL.with(|f| f.set(factor == 0 && k == 30));
                    self.correction.correct(&mut ex, &x);
                    crate::env::DIAG_DETAIL.with(|f| f.set(false));
                    let mut d = vec![C64::ZERO; nv];
                    x.dot(&ex, &mut d);
                    // Ã x = −(0 − Ã x)
                    self.smoother.residual(&mut ax, &x, &zero);
                    let gain = norm(&ex);
                    if k % 5 == 0 || k == 1 {
                        eprintln!(
                        "diag power {label} {k}: |E x|/|x| {gain:.4e} x†Ex {:.4e}{:+.4e}i |A x|/|x| {:.4e} mu {:.4e}",
                        d[0].re,
                        d[0].im,
                        norm(&ax),
                        gain * norm(&ax)
                    );
                    }
                    x.copy_from(&ex);
                }
            }
            super::correction::DIAG_FACTOR.with(|f| f.set(0));
            // Power iteration on the residual propagator K = 1 − Ã M of one cycle, then on
            // 3 − Ã M = 2 + K. |K x|/|x| above 1 means the cycle alone makes that residual
            // grow; the shifted one favors the eigenvalue of Ã M of smallest real part (the
            // largest |3 − λ| when the spectrum lies in Re λ < 3), which K may hide behind an
            // overcorrected one (λ near 3, harmless to GCR). x†Ã M x = shift − x†(shift − ÃM)x.
            for (label, shift) in [("K", 1.0), ("3-AM", 3.0)] {
                let mut x = b.zeros_like();
                x.copy_from(b);
                let mut y = b.zeros_like();
                let mut kx = b.zeros_like();
                for k in 1..=40 {
                    let nx = norm(&x);
                    x.scale(&vec![1.0 / nx; nv]);
                    y.scale(&vec![0.0; nv]);
                    self.plain_cycle(&mut y, &x);
                    self.smoother.residual(&mut kx, &y, &x);
                    kx.axpy(&vec![C64::new(shift - 1.0, 0.0); nv], &x);
                    let mut d = vec![C64::ZERO; nv];
                    x.dot(&kx, &mut d);
                    if k % 5 == 0 || k == 1 {
                        eprintln!(
                            "diag power {label} {k}: |. x|/|x| {:.4e} x†(.)x {:.4e}{:+.4e}i x†AMx {:.4e}{:+.4e}i",
                            norm(&kx),
                            d[0].re,
                            d[0].im,
                            shift - d[0].re,
                            -d[0].im
                        );
                    }
                    if k == 40 {
                        // Last step: kx is no longer needed.
                        self.diag_stages(&x, &mut y, &mut kx);
                    } else {
                        x.copy_from(&kx);
                    }
                }
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
        damp_coarse(&mut self.e);
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
                "diag cycle: pre {:.3e} coarse {:.3e} post {:.3e} whole {:.3e} cos(b, AMb) {:.4e} b†AMb/|b|^2 {:.4e}{:+.4e}i |b†AMb|/(|b||AMb|) {:.4e}",
                n1 / nb,
                n2 / n1,
                n3 / n2,
                n3 / nb,
                cos,
                d[0].re / (nb * nb),
                d[0].im / (nb * nb),
                d[0].norm_sqr().sqrt() / (nb * norm(&v))
            );
        }
    }
}
