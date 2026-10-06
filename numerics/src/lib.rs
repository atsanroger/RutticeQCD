//! Small dense numerical tools that do not involve lattice fields.

use std::fmt;

/// The matrix has no inverse to working precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Singular;

impl fmt::Display for Singular {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "matrix is singular to working precision")
    }
}

impl std::error::Error for Singular {}

/// Invert a dense complex `n × n` matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Both matrices are row-major with (re, im) pairs: element (r, c) is at `2 * (n * r + c)`.
/// `a` is used as scratch and holds no meaningful value afterwards.
pub fn invert_complex(n: usize, a: &mut [f64], inv: &mut [f64]) -> Result<(), Singular> {
    assert_eq!(a.len(), 2 * n * n);
    assert_eq!(inv.len(), 2 * n * n);
    let at = |r: usize, c: usize| 2 * (n * r + c);

    inv.fill(0.0);
    for d in 0..n {
        inv[at(d, d)] = 1.0;
    }
    let scale = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Err(Singular);
    }

    for col in 0..n {
        // Pivot: the row at or below `col` with the largest entry in this column.
        let mut pivot = col;
        let mut best = 0.0;
        for r in col..n {
            let k = at(r, col);
            let size = a[k] * a[k] + a[k + 1] * a[k + 1];
            if size > best {
                best = size;
                pivot = r;
            }
        }
        if best.sqrt() <= scale * f64::EPSILON * n as f64 {
            return Err(Singular);
        }
        if pivot != col {
            for c in 0..n {
                for m in [&mut *a, &mut *inv] {
                    m.swap(at(pivot, c), at(col, c));
                    m.swap(at(pivot, c) + 1, at(col, c) + 1);
                }
            }
        }

        // Scale the pivot row so the pivot becomes one.
        let (pr, pi) = (a[at(col, col)], a[at(col, col) + 1]);
        let (ir, ii) = (pr / best, -pi / best);
        for c in 0..n {
            for m in [&mut *a, &mut *inv] {
                let k = at(col, c);
                let (x, y) = (m[k], m[k + 1]);
                m[k] = x * ir - y * ii;
                m[k + 1] = x * ii + y * ir;
            }
        }

        // Eliminate this column from every other row.
        for r in 0..n {
            if r == col {
                continue;
            }
            let (fr, fi) = (a[at(r, col)], a[at(r, col) + 1]);
            if fr == 0.0 && fi == 0.0 {
                continue;
            }
            for c in 0..n {
                for m in [&mut *a, &mut *inv] {
                    let (x, y) = (m[at(col, c)], m[at(col, c) + 1]);
                    m[at(r, c)] -= fr * x - fi * y;
                    m[at(r, c) + 1] -= fr * y + fi * x;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiply(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; 2 * n * n];
        for r in 0..n {
            for k in 0..n {
                let (ar, ai) = (a[2 * (n * r + k)], a[2 * (n * r + k) + 1]);
                for col in 0..n {
                    let (br, bi) = (b[2 * (n * k + col)], b[2 * (n * k + col) + 1]);
                    c[2 * (n * r + col)] += ar * br - ai * bi;
                    c[2 * (n * r + col) + 1] += ar * bi + ai * br;
                }
            }
        }
        c
    }

    fn sample(n: usize) -> Vec<f64> {
        (0..2 * n * n)
            .map(|i| {
                (((i * 37 + 11) % 29) as f64 - 14.0) / 7.0
                    + if i % (2 * n + 2) == 0 { 6.0 } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn inverse_times_matrix_is_the_identity() {
        for n in [1, 3, 12] {
            let a = sample(n);
            let mut inv = vec![0.0; 2 * n * n];
            invert_complex(n, &mut a.clone(), &mut inv).unwrap();
            for product in [multiply(n, &a, &inv), multiply(n, &inv, &a)] {
                for r in 0..n {
                    for c in 0..n {
                        let want = if r == c { 1.0 } else { 0.0 };
                        assert!((product[2 * (n * r + c)] - want).abs() < 1e-12);
                        assert!(product[2 * (n * r + c) + 1].abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn needs_pivoting() {
        // [[0, 1], [1, 0]] has a zero in the first pivot position.
        let a = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut inv = vec![0.0; 8];
        invert_complex(2, &mut a.clone(), &mut inv).unwrap();
        assert_eq!(inv, a);
    }

    #[test]
    fn singular_matrices_are_reported() {
        let mut rank_one = vec![1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 0.0];
        assert_eq!(
            invert_complex(2, &mut rank_one, &mut [0.0; 8]),
            Err(Singular)
        );
        assert_eq!(
            invert_complex(2, &mut [0.0; 8], &mut [0.0; 8]),
            Err(Singular)
        );
    }
}
