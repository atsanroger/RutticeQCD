//! A fixed polynomial of an operator as an approximate inverse.

use crate::numerics::{invert_complex, Singular};
use crate::{fit, LinearFn, Vector, C64};

/// `out = p(A) inp` with a polynomial p whose coefficients were fitted once: the
/// minimal-residual polynomial of a sample vector, `min |s - A p(A) s|` over the
/// polynomials of degree `degree - 1`, frozen afterwards.
///
/// GMRES(m) from a zero start is the same with the polynomial fitted anew to every input.
/// Freezing it gives up some strength and gains two things: no Krylov basis is kept (three
/// work vectors whatever the degree), and the map is the same linear map every time it is
/// applied, which a solver such as BiCGStab expects of its preconditioner.
///
/// With `passes > 1` the polynomial is applied as a stationary correction,
/// `x += p(A) (b - A x)`: the error is multiplied by `(1 - p(A) A)` in every pass, so a low
/// degree, whose fit is well conditioned, can still be made strong.
///
/// One application costs `passes * degree - 1` applications of A.
pub struct Polynomial<A, V> {
    a: A,
    /// `p(z) = Σ_k coefficients[k] z^k`.
    coefficients: Vec<C64>,
    passes: usize,
    tmp: V,
    residual: V,
    correction: V,
}

impl<A: LinearFn<V>, V: Vector> Polynomial<A, V> {
    /// Fit the polynomial to `sample`; if it holds several vectors, to all of them at once.
    ///
    /// The powers `A^k s` turn towards the largest eigenvalues as k grows and become nearly
    /// dependent, so the fit fails beyond a moderate degree (`Singular`).
    pub fn fit(mut a: A, sample: &V, degree: usize, passes: usize) -> Result<Self, Singular> {
        assert!(degree >= 1 && passes >= 1);
        let nv = sample.num_vectors();
        let mut powers: Vec<V> = (0..degree).map(|_| sample.zeros_like()).collect();
        a.mult(&mut powers[0], sample);
        for k in 1..degree {
            let (done, rest) = powers.split_at_mut(k);
            a.mult(&mut rest[0], &done[k - 1]);
        }

        // Least squares for s ≈ Σ_k c_k A^{k+1} s through the normal equations, with the
        // columns scaled to unit length.
        let mut per_vector = vec![0.0; nv];
        let lengths: Vec<f64> = powers
            .iter()
            .map(|p| {
                p.norm_sqr(&mut per_vector);
                per_vector.iter().sum::<f64>().sqrt()
            })
            .collect();
        if lengths.iter().any(|l| *l == 0.0 || !l.is_finite()) {
            return Err(Singular);
        }
        let mut dots = vec![C64::ZERO; nv];
        let mut sum = |x: &V, y: &V| {
            x.dot(y, &mut dots);
            dots.iter().fold(C64::ZERO, |s, d| s + *d)
        };
        let mut gram = vec![0.0; 2 * degree * degree];
        let mut rhs = vec![C64::ZERO; degree];
        for j in 0..degree {
            rhs[j] = sum(&powers[j], sample).scale(1.0 / lengths[j]);
            for k in 0..degree {
                let g = sum(&powers[j], &powers[k]).scale(1.0 / (lengths[j] * lengths[k]));
                gram[2 * (degree * j + k)] = g.re;
                gram[2 * (degree * j + k) + 1] = g.im;
            }
        }
        let mut inverse = vec![0.0; 2 * degree * degree];
        invert_complex(degree, &mut gram, &mut inverse)?;
        let coefficients: Vec<C64> = (0..degree)
            .map(|j| {
                let c = (0..degree).fold(C64::ZERO, |c, k| {
                    let at = 2 * (degree * j + k);
                    c + C64::new(inverse[at], inverse[at + 1]) * rhs[k]
                });
                c.scale(1.0 / lengths[j])
            })
            .collect();
        if coefficients
            .iter()
            .any(|c| !c.re.is_finite() || !c.im.is_finite())
        {
            return Err(Singular);
        }

        Ok(Self {
            a,
            coefficients,
            passes,
            tmp: sample.zeros_like(),
            residual: sample.zeros_like(),
            correction: sample.zeros_like(),
        })
    }

    /// The coefficients of p, lowest power first.
    pub fn coefficients(&self) -> &[C64] {
        &self.coefficients
    }
}

/// `out = p(A) b` by Horner's rule; `tmp` is scratch.
fn horner<A: LinearFn<V>, V: Vector>(
    a: &mut A,
    coefficients: &[Vec<C64>],
    out: &mut V,
    tmp: &mut V,
    b: &V,
) {
    // The result moves between the two vectors with every application of A; start in the
    // one that makes it end in `out`.
    let steps = coefficients.len() - 1;
    let (mut now, mut next) = if steps.is_multiple_of(2) {
        (out, tmp)
    } else {
        (tmp, out)
    };
    now.set_zero();
    now.axpy(&coefficients[steps], b);
    for c in coefficients[..steps].iter().rev() {
        a.mult(next, now);
        next.axpy(c, b);
        std::mem::swap(&mut now, &mut next);
    }
}

impl<A: LinearFn<V>, V: Vector> LinearFn<V> for Polynomial<A, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let nv = inp.num_vectors();
        fit(&mut self.tmp, inp);
        let coefficients: Vec<Vec<C64>> = self.coefficients.iter().map(|c| vec![*c; nv]).collect();
        horner(&mut self.a, &coefficients, out, &mut self.tmp, inp);
        if self.passes == 1 {
            return;
        }
        fit(&mut self.residual, inp);
        fit(&mut self.correction, inp);
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        for _ in 1..self.passes {
            self.a.mult(&mut self.residual, out);
            self.residual.aypx(&minus_one, inp);
            horner(
                &mut self.a,
                &coefficients,
                &mut self.correction,
                &mut self.tmp,
                &self.residual,
            );
            out.axpy(&one, &self.correction);
        }
    }
}
