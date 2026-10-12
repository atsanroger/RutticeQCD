//! Chebyshev polynomials of an operator: a low-pass filter for making multigrid test
//! vectors, and Chebyshev iteration as a fixed smoother (the design notes, §7.6.5 (b), (c)).

use crate::{fit, LinearFn, Vector, C64};

/// `out = T_n(ℓ(H)) inp` with `ℓ(x) = (2x - (hi + lo)) / (hi - lo)`, for a Hermitian H whose
/// spectrum lies in `[0, hi]`.
pub struct ChebyshevFilter<H, V> {
    h: H,
    degree: usize,
    lo: f64,
    hi: f64,
    previous: V,
    current: V,
    next: V,
}

impl<H: LinearFn<V>, V: Vector> ChebyshevFilter<H, V> {
    /// `like` only provides the shape of the work vectors.
    pub fn new(h: H, degree: usize, lo: f64, hi: f64, like: &V) -> Self {
        assert!(degree >= 1 && 0.0 <= lo && lo < hi);
        Self {
            h,
            degree,
            lo,
            hi,
            previous: like.zeros_like(),
            current: like.zeros_like(),
            next: like.zeros_like(),
        }
    }

    /// `out = ℓ(H) inp`.
    fn mapped(h: &mut H, lo: f64, hi: f64, out: &mut V, inp: &V) {
        let nv = inp.num_vectors();
        h.mult(out, inp);
        out.scale(&vec![2.0 / (hi - lo); nv]);
        out.axpy(&vec![C64::new(-(hi + lo) / (hi - lo), 0.0); nv], inp);
    }
}

impl<H: LinearFn<V>, V: Vector> LinearFn<V> for ChebyshevFilter<H, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let nv = inp.num_vectors();
        for w in [&mut self.previous, &mut self.current, &mut self.next] {
            fit(w, inp);
        }
        // T_0 = 1, T_1 = ℓ.
        self.previous.copy_from(inp);
        Self::mapped(&mut self.h, self.lo, self.hi, &mut self.current, inp);
        let two = vec![2.0; nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        for _ in 1..self.degree {
            Self::mapped(&mut self.h, self.lo, self.hi, &mut self.next, &self.current);
            self.next.scale(&two);
            self.next.axpy(&minus_one, &self.previous);
            std::mem::swap(&mut self.previous, &mut self.current);
            std::mem::swap(&mut self.current, &mut self.next);
        }
        out.copy_from(&self.current);
    }
}

/// `steps` steps of Chebyshev iteration for `A x = inp` from `x = 0`, for an A whose
/// spectrum lies in an ellipse with real foci `lo` and `hi`, `0 < lo < hi`: a fixed
/// polynomial approximation of `A^-1` that is best, in the minimax sense, on that ellipse.
///
/// For a non-Hermitian A with eigenvalues off the real axis the ellipse through `[lo, hi]` must
/// still enclose the part of the spectrum to be damped; Bridge++ and Grid have only the Hermitian,
/// real-interval version, so this one follows the textbook, not a reference code.
pub struct Chebyshev<A, V> {
    a: A,
    steps: usize,
    lo: f64,
    hi: f64,
    r: V,
    d: V,
    ad: V,
}

impl<A: LinearFn<V>, V: Vector> Chebyshev<A, V> {
    /// `like` only provides the shape of the work vectors.
    pub fn new(a: A, steps: usize, lo: f64, hi: f64, like: &V) -> Self {
        assert!(steps >= 1 && 0.0 < lo && lo < hi);
        Self {
            a,
            steps,
            lo,
            hi,
            r: like.zeros_like(),
            d: like.zeros_like(),
            ad: like.zeros_like(),
        }
    }

    /// The operator, for a caller that needs it besides the polynomial (a multigrid cycle
    /// computing residuals) without keeping a second copy of its work fields.
    pub fn operator(&mut self) -> &mut A {
        &mut self.a
    }
}

impl<A: LinearFn<V>, V: Vector> LinearFn<V> for Chebyshev<A, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let nv = inp.num_vectors();
        for w in [&mut self.r, &mut self.d, &mut self.ad] {
            fit(w, inp);
        }
        let (theta, delta) = ((self.hi + self.lo) / 2.0, (self.hi - self.lo) / 2.0);
        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;
        self.r.copy_from(inp);
        self.d.copy_from(inp);
        self.d.scale(&vec![1.0 / theta; nv]);
        out.set_zero();
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        for step in 0..self.steps {
            out.axpy(&one, &self.d);
            if step + 1 == self.steps {
                break;
            }
            self.a.mult(&mut self.ad, &self.d);
            self.r.axpy(&minus_one, &self.ad);
            let rho_next = 1.0 / (2.0 * sigma - rho);
            self.d.scale(&vec![rho_next * rho; nv]);
            self.d
                .axpy(&vec![C64::new(2.0 * rho_next / delta, 0.0); nv], &self.r);
            rho = rho_next;
        }
    }
}

/// An estimate of the largest |λ| of A by `steps` steps of the power iteration from
/// `start`, which is overwritten. For the upper ends `hi` of the Chebyshev polynomials;
/// multiply by a safety factor (1.1) before use.
pub fn largest_magnitude<A: LinearFn<V>, V: Vector>(a: &mut A, start: &mut V, steps: usize) -> f64 {
    let nv = start.num_vectors();
    let mut next = start.zeros_like();
    let mut norms = vec![0.0; nv];
    let mut estimate = 0.0;
    for _ in 0..steps {
        start.norm_sqr(&mut norms);
        let scale: Vec<f64> = norms.iter().map(|n| 1.0 / n.sqrt()).collect();
        start.scale(&scale);
        a.mult(&mut next, start);
        next.norm_sqr(&mut norms);
        estimate = norms.iter().map(|n| n.sqrt()).fold(0.0, f64::max);
        std::mem::swap(start, &mut next);
    }
    estimate
}
