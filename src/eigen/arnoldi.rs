//! Eigenpairs of a non-Hermitian operator by the Krylov-Schur method (Stewart, SIAM J. Matrix
//! Anal. Appl. 23 (2001) 601), on any [`Vector`] (on the host or on a device): Arnoldi to
//! `max_dim` vectors with every vector made orthogonal to all the others twice, the Schur form
//! of the projection reordered so that the wanted Ritz values come first, and a restart from
//! the first `keep` Schur vectors and the residual direction. The residual of a Ritz pair
//! (θ, V Q y) is |b† y| with b the residual row of the Krylov-Schur relation; at the end the
//! pairs are checked by |A x − θ x| itself.

use super::schur::{move_to, schur, triangular_eigenvector};
use crate::{LinearFn, Vector, C64};

/// Which end of the spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    SmallestReal,
    LargestReal,
    LargestMagnitude,
    SmallestMagnitude,
}

impl Which {
    /// Lower is wanted first.
    fn score(self, z: C64) -> f64 {
        match self {
            Which::SmallestReal => z.re,
            Which::LargestReal => -z.re,
            Which::LargestMagnitude => -z.norm_sqr(),
            Which::SmallestMagnitude => z.norm_sqr(),
        }
    }
}

/// The settings of [`arnoldi`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArnoldiParams {
    /// Eigenpairs wanted.
    pub wanted: usize,
    /// Schur vectors kept at a restart (at least `wanted`).
    pub keep: usize,
    /// Arnoldi vectors before a restart (more than `keep`).
    pub max_dim: usize,
    /// Converged when the residual of every wanted pair is at most tolerance |θ|.
    pub tolerance: f64,
    pub max_restarts: usize,
    pub which: Which,
}

/// The eigenpairs found, in the order of `which`.
pub struct NonHermitianPairs<V> {
    pub values: Vec<C64>,
    /// Normalised.
    pub vectors: Vec<V>,
    /// |A x − θ x| of each.
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub restarts: usize,
    /// Applications of A.
    pub applications: usize,
    /// All the Ritz values of the last projection (`max_dim` of them): a picture of the
    /// spectrum, converged or not.
    pub ritz: Vec<C64>,
}

fn dot<V: Vector>(a: &V, b: &V) -> C64 {
    let mut d = [C64::ZERO];
    a.dot(b, &mut d);
    d[0]
}

fn norm<V: Vector>(a: &V) -> f64 {
    let mut n = [0.0];
    a.norm_sqr(&mut n);
    n[0].sqrt()
}

/// Σ_j basis[j] c[j].
fn combine<V: Vector>(basis: &[V], c: impl Fn(usize) -> C64) -> V {
    let mut x = basis[0].zeros_like();
    for (j, v) in basis.iter().enumerate() {
        x.axpy(&[c(j)], v);
    }
    x
}

/// `params.wanted` eigenpairs of `a` at the end of the spectrum `params.which`, from the start
/// vector `start` (one vector).
pub fn arnoldi<V: Vector, A: LinearFn<V>>(
    a: &mut A,
    start: &V,
    params: ArnoldiParams,
) -> NonHermitianPairs<V> {
    let ArnoldiParams {
        wanted,
        keep,
        max_dim: m,
        tolerance,
        max_restarts,
        which,
    } = params;
    assert_eq!(start.num_vectors(), 1);
    assert!(wanted >= 1 && keep >= wanted && m > keep + 1);
    let mut applications = 0;
    let mut basis: Vec<V> = Vec::with_capacity(m + 1);
    let mut first = start.zeros_like();
    first.copy_from(start);
    first.scale(&[1.0 / norm(start)]);
    basis.push(first);
    // The Krylov-Schur relation A V_m = V_m H + v_m h_m, h rows 0..=m, columns 0..m.
    let mut h = vec![C64::ZERO; (m + 1) * m];
    let at = |i: usize, j: usize| i * m + j;
    let (mut kept, mut restarts) = (0, 0);
    loop {
        for j in kept..m {
            let mut w = basis[j].zeros_like();
            a.mult(&mut w, &basis[j]);
            applications += 1;
            for _ in 0..2 {
                for (i, v) in basis.iter().enumerate().take(j + 1) {
                    let c = dot(v, &w);
                    h[at(i, j)] += c;
                    w.axpy(&[-c], v);
                }
            }
            let beta = norm(&w);
            h[at(j + 1, j)] = C64::new(beta, 0.0);
            w.scale(&[1.0 / beta.max(f64::MIN_POSITIVE)]);
            basis.push(w);
        }
        // The Schur form of the square part, the wanted values first.
        let square: Vec<C64> = h[..m * m].to_vec();
        let (mut t, mut q) = schur(&square, m).expect("the Schur form of the projection");
        for p in 0..keep {
            let best = (p..m)
                .min_by(|&x, &y| {
                    which
                        .score(t[x * m + x])
                        .total_cmp(&which.score(t[y * m + y]))
                })
                .expect("a candidate");
            move_to(&mut t, &mut q, m, best, p);
        }
        // b = h_m Q, the residual row.
        let b: Vec<C64> = (0..m)
            .map(|k| (0..m).fold(C64::ZERO, |s, j| s + h[at(m, j)] * q[j * m + k]))
            .collect();
        let estimates: Vec<(C64, Vec<C64>, f64)> = (0..wanted)
            .map(|k| {
                let y = triangular_eigenvector(&t, m, k);
                let r = (0..m)
                    .fold(C64::ZERO, |s, j| s + b[j] * y[j])
                    .norm_sqr()
                    .sqrt();
                (t[k * m + k], y, r)
            })
            .collect();
        let converged = estimates
            .iter()
            .all(|(theta, _, r)| *r <= tolerance * theta.norm_sqr().sqrt().max(f64::MIN_POSITIVE));
        if converged || restarts == max_restarts {
            let mut out = NonHermitianPairs {
                values: Vec::new(),
                vectors: Vec::new(),
                residuals: Vec::new(),
                converged,
                restarts,
                applications,
                ritz: (0..m).map(|k| t[k * m + k]).collect(),
            };
            for (theta, y, _) in estimates {
                // x = V_m Q y.
                let qy: Vec<C64> = (0..m)
                    .map(|j| (0..m).fold(C64::ZERO, |s, k| s + q[j * m + k] * y[k]))
                    .collect();
                let mut x = combine(&basis[..m], |j| qy[j]);
                x.scale(&[1.0 / norm(&x)]);
                let mut ax = x.zeros_like();
                a.mult(&mut ax, &x);
                out.applications += 1;
                ax.axpy(&[-theta], &x);
                out.values.push(theta);
                out.residuals.push(norm(&ax));
                out.vectors.push(x);
            }
            return out;
        }
        // Restart from the first `keep` Schur vectors and the residual direction.
        let residual_direction = basis.pop().expect("the residual direction");
        let mut restarted: Vec<V> = (0..keep)
            .map(|i| combine(&basis[..m], |j| q[j * m + i]))
            .collect();
        restarted.push(residual_direction);
        basis = restarted;
        h.iter_mut().for_each(|z| *z = C64::ZERO);
        for i in 0..keep {
            for j in i..keep {
                h[at(i, j)] = t[i * m + j];
            }
            h[at(keep, i)] = b[i];
        }
        kept = keep;
        restarts += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{random_spinor, Field, Lattice, Spinor};

    /// (A x)_k = λ_k x_k + γ x_(k+1): non-normal, the eigenvalues λ_k.
    struct Bidiagonal {
        lambda: Vec<C64>,
        gamma: f64,
    }

    impl LinearFn<Field<f64, Spinor>> for Bidiagonal {
        fn mult(&mut self, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>) {
            let x: Vec<C64> = inp
                .as_slice()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|z| C64::new(z[0], z[1]))
                .collect();
            let n = x.len();
            for (k, o) in out
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                let mut y = self.lambda[k] * x[k];
                if k + 1 < n {
                    y += x[k + 1].scale(self.gamma);
                }
                o[0] = y.re;
                o[1] = y.im;
            }
        }
    }

    #[test]
    fn arnoldi_finds_the_ends_of_a_non_normal_spectrum() {
        let lat = Lattice::single_node([2, 2, 2, 4]);
        let start = random_spinor(&lat, 1, 11);
        let n = start.as_slice().len() / 2;
        // A cloud around 1 (radius 0.6) and three outliers: one in the left half plane.
        let mut lambda: Vec<C64> = (0..n)
            .map(|k| {
                let phi = 2.399963 * k as f64;
                let r = 0.6 * ((k as f64 + 0.5) / n as f64).sqrt();
                C64::new(1.0 + r * phi.cos(), r * phi.sin())
            })
            .collect();
        lambda[n / 10] = C64::new(-0.05, 0.2);
        lambda[n / 2] = C64::new(0.01, -0.3);
        lambda[n - 7] = C64::new(3.0, 1.0);
        let mut sorted = lambda.clone();
        sorted.sort_by(|a, b| a.re.total_cmp(&b.re));
        let mut op = Bidiagonal {
            lambda: lambda.clone(),
            gamma: 0.3,
        };
        let params = ArnoldiParams {
            wanted: 3,
            keep: 10,
            max_dim: 40,
            tolerance: 1e-9,
            max_restarts: 500,
            which: Which::SmallestReal,
        };
        let e = arnoldi(&mut op, &start, params);
        eprintln!(
            "smallest real: {:?} ({} restarts, {} applications), residuals {:?}",
            e.values, e.restarts, e.applications, e.residuals
        );
        assert!(e.converged);
        for (got, want) in e.values.iter().zip(&sorted) {
            assert!(
                (*got - *want).norm_sqr().sqrt() < 1e-7,
                "{got:?} for {want:?}"
            );
        }
        assert!(e.residuals.iter().all(|r| *r < 1e-6), "{:?}", e.residuals);
        let e = arnoldi(
            &mut op,
            &start,
            ArnoldiParams {
                which: Which::LargestMagnitude,
                wanted: 1,
                ..params
            },
        );
        assert!(e.converged && (e.values[0] - C64::new(3.0, 1.0)).norm_sqr().sqrt() < 1e-8);
    }
}
