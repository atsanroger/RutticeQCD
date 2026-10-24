//! The complex Schur form of a small dense matrix, for the Krylov-Schur method (`arnoldi`):
//! A = Q T Q† with T upper triangular and Q unitary, by the reduction to Hessenberg form
//! (Householder) and the shifted QR algorithm (Wilkinson shifts, Givens rotations); the
//! reordering of T (`move_to`), and the eigenvectors of T by back substitution. Matrices are
//! row-major n × n vectors of complex numbers.

use crate::C64;

/// The principal square root.
fn csqrt(z: C64) -> C64 {
    let r = z.norm_sqr().sqrt();
    let re = ((r + z.re) / 2.0).max(0.0).sqrt();
    let im = ((r - z.re) / 2.0).max(0.0).sqrt();
    C64::new(re, if z.im < 0.0 { -im } else { im })
}

/// The rotation (c, s), c real, with [c s; −s̄ c] (a, b)ᵀ = (r, 0)ᵀ.
fn givens(a: C64, b: C64) -> (f64, C64) {
    let (na, nb) = (a.norm_sqr().sqrt(), b.norm_sqr().sqrt());
    if nb == 0.0 {
        return (1.0, C64::ZERO);
    }
    if na == 0.0 {
        return (0.0, b.conj().scale(1.0 / nb));
    }
    let r = (na * na + nb * nb).sqrt();
    let phase = a.scale(1.0 / na);
    (na / r, phase * b.conj().scale(1.0 / r))
}

/// Rows i and i+1 of `m` (n columns) times G = [c s; −s̄ c].
fn rotate_rows(m: &mut [C64], n: usize, i: usize, (c, s): (f64, C64), from: usize) {
    for k in from..n {
        let (x, y) = (m[i * n + k], m[(i + 1) * n + k]);
        m[i * n + k] = x.scale(c) + s * y;
        m[(i + 1) * n + k] = y.scale(c) - s.conj() * x;
    }
}

/// Columns i and i+1 of `m` (rows `upto`) times G†.
fn rotate_columns(m: &mut [C64], n: usize, i: usize, (c, s): (f64, C64), upto: usize) {
    for k in 0..upto {
        let (x, y) = (m[k * n + i], m[k * n + i + 1]);
        m[k * n + i] = x.scale(c) + s.conj() * y;
        m[k * n + i + 1] = y.scale(c) - s * x;
    }
}

/// A = Q H Q†, H upper Hessenberg: `a` becomes H, `q` (n × n) the accumulated Q.
fn hessenberg(a: &mut [C64], q: &mut [C64], n: usize) {
    for col in 0..n.saturating_sub(2) {
        // The Householder vector of a[col+1.., col].
        let mut v: Vec<C64> = (col + 1..n).map(|i| a[i * n + col]).collect();
        let alpha = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if alpha == 0.0 {
            continue;
        }
        let x0 = v[0];
        let phase = if x0.norm_sqr() > 0.0 {
            x0.scale(1.0 / x0.norm_sqr().sqrt())
        } else {
            C64::new(1.0, 0.0)
        };
        v[0] = x0 + phase.scale(alpha);
        let vn = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vn));
        // a = (I − 2vv†) a (I − 2vv†), q = q (I − 2vv†).
        for k in 0..n {
            let s: C64 =
                (0..v.len()).fold(C64::ZERO, |s, t| s + v[t].conj() * a[(col + 1 + t) * n + k]);
            for t in 0..v.len() {
                a[(col + 1 + t) * n + k] = a[(col + 1 + t) * n + k] - (v[t] * s).scale(2.0);
            }
        }
        for m in [&mut *a, &mut *q] {
            for k in 0..n {
                let s: C64 = (0..v.len()).fold(C64::ZERO, |s, t| s + m[k * n + col + 1 + t] * v[t]);
                for t in 0..v.len() {
                    m[k * n + col + 1 + t] = m[k * n + col + 1 + t] - (s * v[t].conj()).scale(2.0);
                }
            }
        }
        for i in col + 2..n {
            a[i * n + col] = C64::ZERO;
        }
    }
}

/// The complex Schur form of the n × n `a`: (T, Q) with a = Q T Q†. None if the QR algorithm
/// did not converge.
pub fn schur(a: &[C64], n: usize) -> Option<(Vec<C64>, Vec<C64>)> {
    let mut t = a.to_vec();
    let mut q = vec![C64::ZERO; n * n];
    for i in 0..n {
        q[i * n + i] = C64::new(1.0, 0.0);
    }
    hessenberg(&mut t, &mut q, n);
    let eps = f64::EPSILON;
    let mut hi = n;
    let mut iterations = 0;
    while hi > 1 {
        // Deflate at the bottom.
        let i = hi - 1;
        let scale = t[i * n + i].norm_sqr().sqrt() + t[(i - 1) * n + i - 1].norm_sqr().sqrt();
        if t[i * n + i - 1].norm_sqr().sqrt() <= eps * scale.max(f64::MIN_POSITIVE) {
            t[i * n + i - 1] = C64::ZERO;
            hi -= 1;
            iterations = 0;
            continue;
        }
        iterations += 1;
        if iterations > 100 * n {
            return None;
        }
        // The active window [lo, hi).
        let mut lo = hi - 1;
        while lo > 0 {
            let s = t[lo * n + lo].norm_sqr().sqrt() + t[(lo - 1) * n + lo - 1].norm_sqr().sqrt();
            if t[lo * n + lo - 1].norm_sqr().sqrt() <= eps * s.max(f64::MIN_POSITIVE) {
                t[lo * n + lo - 1] = C64::ZERO;
                break;
            }
            lo -= 1;
        }
        // The Wilkinson shift: the eigenvalue of the trailing 2 × 2 nearer its last entry, with
        // an exceptional shift now and then.
        let (a11, a12, a21, a22) = (
            t[(i - 1) * n + i - 1],
            t[(i - 1) * n + i],
            t[i * n + i - 1],
            t[i * n + i],
        );
        let half = (a11 - a22).scale(0.5);
        let root = csqrt(half * half + a12 * a21);
        let (m1, m2) = (a22 - half + root, a22 - half - root);
        // a22 − half ± root are the eigenvalues; take the one nearer a22.
        let mut mu = if (m1 - a22).norm_sqr() < (m2 - a22).norm_sqr() {
            m1
        } else {
            m2
        };
        let mu_ok = mu.re.is_finite() && mu.im.is_finite();
        if !mu_ok || iterations % 11 == 0 {
            mu = a22 + C64::new(t[i * n + i - 1].norm_sqr().sqrt(), 0.0);
        }
        // One implicit single-shift QR sweep on [lo, hi).
        let mut g = givens(t[lo * n + lo] - mu, t[(lo + 1) * n + lo]);
        for k in lo..hi - 1 {
            if k > lo {
                g = givens(t[k * n + k - 1], t[(k + 1) * n + k - 1]);
            }
            rotate_rows(&mut t, n, k, g, if k > lo { k - 1 } else { k });
            rotate_columns(&mut t, n, k, g, (k + 3).min(hi).max(k + 2).min(n));
            rotate_columns(&mut q, n, k, g, n);
            if k > lo {
                t[(k + 1) * n + k - 1] = C64::ZERO;
            }
        }
    }
    // Clear what is left below the diagonal.
    for i in 1..n {
        for j in 0..i {
            t[i * n + j] = C64::ZERO;
        }
    }
    Some((t, q))
}

/// Reorders the Schur form (t, q) of an n × n matrix so that the diagonal entry at `from`
/// moves to `to` (< from), by swaps of neighbours.
pub(crate) fn move_to(t: &mut [C64], q: &mut [C64], n: usize, from: usize, to: usize) {
    for i in (to..from).rev() {
        // Swap t[i][i] and t[i+1][i+1].
        let (a, b, x) = (t[i * n + i], t[(i + 1) * n + i + 1], t[i * n + i + 1]);
        if a == b {
            continue;
        }
        let g = givens(x, b - a);
        rotate_columns(t, n, i, g, n);
        rotate_rows(t, n, i, g, 0);
        rotate_columns(q, n, i, g, n);
        t[(i + 1) * n + i] = C64::ZERO;
        t[i * n + i] = b;
        t[(i + 1) * n + i + 1] = a;
    }
}

/// The eigenvector of the upper triangular t for its k-th diagonal entry, normalised: x_k = 1,
/// x_j = 0 for j > k, back substitution above.
pub(crate) fn triangular_eigenvector(t: &[C64], n: usize, k: usize) -> Vec<C64> {
    let lambda = t[k * n + k];
    let mut x = vec![C64::ZERO; n];
    x[k] = C64::new(1.0, 0.0);
    let small = f64::EPSILON
        * (0..n)
            .map(|i| t[i * n + i].norm_sqr().sqrt())
            .fold(0.0, f64::max);
    for i in (0..k).rev() {
        let s: C64 = (i + 1..=k).fold(C64::ZERO, |s, j| s + t[i * n + j] * x[j]);
        let mut d = t[i * n + i] - lambda;
        if d.norm_sqr().sqrt() < small {
            d = C64::new(small.max(f64::MIN_POSITIVE), 0.0);
        }
        x[i] = -(s / d);
    }
    let norm = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    x.iter_mut().for_each(|z| *z = z.scale(1.0 / norm));
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[C64], b: &[C64], n: usize) -> Vec<C64> {
        let mut c = vec![C64::ZERO; n * n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
        c
    }

    fn dagger(a: &[C64], n: usize) -> Vec<C64> {
        let mut d = vec![C64::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = a[j * n + i].conj();
            }
        }
        d
    }

    fn random(n: usize, seed: u64) -> Vec<C64> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        (0..n * n).map(|_| C64::new(next(), next())).collect()
    }

    fn worst(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0, |m, (x, y)| m.max((*x - *y).norm_sqr().sqrt()))
    }

    #[test]
    fn schur_form_of_a_random_matrix() {
        let n = 30;
        let a = random(n, 3);
        let (t, q) = schur(&a, n).expect("converges");
        // A = Q T Q†, Q unitary, T triangular.
        let back = matmul(&matmul(&q, &t, n), &dagger(&q, n), n);
        assert!(worst(&a, &back) < 1e-12, "{}", worst(&a, &back));
        let mut id = vec![C64::ZERO; n * n];
        (0..n).for_each(|i| id[i * n + i] = C64::new(1.0, 0.0));
        assert!(worst(&matmul(&dagger(&q, n), &q, n), &id) < 1e-12);
        for i in 1..n {
            for j in 0..i {
                assert_eq!(t[i * n + j], C64::ZERO);
            }
        }
        // Moving an eigenvalue keeps A = Q T Q†.
        let (mut t2, mut q2) = (t.clone(), q.clone());
        let last = t[(n - 1) * n + n - 1];
        move_to(&mut t2, &mut q2, n, n - 1, 0);
        assert!((t2[0] - last).norm_sqr().sqrt() < 1e-12);
        let back = matmul(&matmul(&q2, &t2, n), &dagger(&q2, n), n);
        assert!(worst(&a, &back) < 1e-11, "{}", worst(&a, &back));
        // An eigenvector: A (Q x) = λ (Q x).
        let k = 7;
        let x = triangular_eigenvector(&t2, n, k);
        let qx: Vec<C64> = (0..n)
            .map(|i| (0..n).fold(C64::ZERO, |s, j| s + q2[i * n + j] * x[j]))
            .collect();
        let aqx: Vec<C64> = (0..n)
            .map(|i| (0..n).fold(C64::ZERO, |s, j| s + a[i * n + j] * qx[j]))
            .collect();
        let lambda = t2[k * n + k];
        let r = aqx.iter().zip(&qx).fold(0.0f64, |m, (y, x)| {
            m.max((*y - lambda * *x).norm_sqr().sqrt())
        });
        assert!(r < 1e-11, "{r}");
    }
}
