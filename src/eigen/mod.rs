//! Eigenvectors of Hermitian operators.
//!
//! [`lowest`]: the lowest eigenpairs of a Hermitian positive operator A by the thick-restart
//! Lanczos method (Wu and Simon, SIAM J. Matrix Anal. Appl. 22 (2000) 602), on any [`Vector`]
//! (on the host or on a device). Lanczos converges first at the ends of the spectrum, and the low
//! end of a Dirac operator is crowded; with a Chebyshev polynomial ([`Chebyshev`]) Lanczos runs
//! on B = T_d(ℓ(A)) instead, ℓ mapping [lo, hi] onto [−1, 1], for which the eigenvalues of A
//! below lo are the largest by far: the same eigenvectors, found in far fewer restarts. The
//! eigenvalues of A are the Rayleigh quotients of the Ritz vectors, and convergence is judged
//! by |A x − λ x| itself.
//!
//! Every Lanczos vector is kept and made orthogonal to all the others twice (no loss of
//! orthogonality): the memory is `max_dim` + `keep` vectors at a restart.

mod arnoldi;
pub mod schur;

pub use arnoldi::{arnoldi, ArnoldiParams, NonHermitianPairs, Which};

use crate::numerics::hermitian_eigen;
use crate::{LinearFn, Vector, C64};

/// The polynomial T_d(ℓ(A)) Lanczos runs on, ℓ(x) = (2x − (hi + lo)) / (hi − lo). `degree` must
/// be even, so that the eigenvalues of A below lo (where |ℓ| > 1) become large and positive;
/// hi must bound the spectrum of A from above. The polynomial is applied by the three-term
/// recurrence with no rescaling, so T_d(ℓ(0)) must stay a representable number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chebyshev {
    pub degree: usize,
    pub lo: f64,
    pub hi: f64,
}

/// The settings of [`lowest`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LanczosParams {
    /// Eigenpairs wanted.
    pub wanted: usize,
    /// Ritz vectors kept at a restart (at least `wanted`).
    pub keep: usize,
    /// Lanczos vectors before a restart (more than `keep`).
    pub max_dim: usize,
    /// Converged when |A x − λ x| ≤ tolerance |λ| for every wanted pair.
    pub tolerance: f64,
    pub max_restarts: usize,
    /// Run on a Chebyshev polynomial of A (see the module); none: on −A.
    pub chebyshev: Option<Chebyshev>,
}

/// The eigenpairs found, the lowest eigenvalue first.
pub struct Eigenpairs<V> {
    pub values: Vec<f64>,
    /// Normalised.
    pub vectors: Vec<V>,
    /// |A x − λ x| of each.
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub restarts: usize,
    /// Applications of A.
    pub applications: usize,
}

/// B: the operator Lanczos runs on.
struct Operator<'a, V, A> {
    a: &'a mut A,
    chebyshev: Option<Chebyshev>,
    work: [V; 3],
    applications: usize,
}

impl<V: Vector, A: LinearFn<V>> Operator<'_, V, A> {
    fn apply_a(&mut self, out: &mut V, inp: &V) {
        self.a.mult(out, inp);
        self.applications += 1;
    }

    /// out = B inp, with the work vectors of the recurrence kept from one call to the next.
    fn mult(&mut self, out: &mut V, inp: &V) {
        let Self {
            a,
            chebyshev,
            work,
            applications,
        } = self;
        let Some(c) = *chebyshev else {
            a.mult(out, inp);
            *applications += 1;
            out.scale(&[-1.0]);
            return;
        };
        // ℓ(A) v = (2 A v − (hi + lo) v) / (hi − lo).
        let (scale, shift) = (2.0 / (c.hi - c.lo), -(c.hi + c.lo) / (c.hi - c.lo));
        let mut mapped = |out: &mut V, inp: &V| {
            a.mult(out, inp);
            *applications += 1;
            out.scale(&[scale]);
            out.axpy(&[C64::new(shift, 0.0)], inp);
        };
        // T_0 = 1, T_1 = ℓ, T_(k+1) = 2 ℓ T_k − T_(k−1).
        let [previous, current, next] = work;
        previous.copy_from(inp);
        mapped(current, inp);
        for _ in 1..c.degree {
            mapped(next, current);
            next.scale(&[2.0]);
            next.axpy(&[C64::new(-1.0, 0.0)], previous);
            std::mem::swap(previous, current);
            std::mem::swap(current, next);
        }
        out.copy_from(current);
    }
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

/// The lowest `params.wanted` eigenpairs of the Hermitian positive operator `a`, from the start
/// vector `start` (one vector; not orthogonal to the wanted eigenvectors).
pub fn lowest<V: Vector, A: LinearFn<V>>(
    a: &mut A,
    start: &V,
    params: LanczosParams,
) -> Eigenpairs<V> {
    let LanczosParams {
        wanted,
        keep,
        max_dim: m,
        tolerance,
        max_restarts,
        chebyshev,
    } = params;
    assert_eq!(start.num_vectors(), 1);
    assert!(wanted >= 1 && keep >= wanted && m > keep);
    if let Some(c) = chebyshev {
        assert!(c.degree >= 2 && c.degree % 2 == 0, "an even degree");
        assert!(0.0 <= c.lo && c.lo < c.hi);
    }
    let mut b = Operator {
        a,
        chebyshev,
        work: [start.zeros_like(), start.zeros_like(), start.zeros_like()],
        applications: 0,
    };
    // The Lanczos vectors and the projection H = V† B V.
    let mut basis: Vec<V> = Vec::with_capacity(m + 1);
    let mut first = start.zeros_like();
    first.copy_from(start);
    first.scale(&[1.0 / norm(start)]);
    basis.push(first);
    let mut h = vec![C64::ZERO; m * m];
    let mut kept = 0;
    let mut restarts = 0;
    loop {
        // Extend the basis to m vectors (and one more, the residual direction).
        let mut beta = 0.0;
        for j in kept..m {
            let mut w = basis[j].zeros_like();
            b.mult(&mut w, &basis[j]);
            let mut column = vec![C64::ZERO; j + 1];
            for _ in 0..2 {
                for (i, v) in basis.iter().enumerate().take(j + 1) {
                    let c = dot(v, &w);
                    column[i] += c;
                    w.axpy(&[-c], v);
                }
            }
            for (i, c) in column.into_iter().enumerate() {
                h[i * m + j] = c;
                h[j * m + i] = c.conj();
            }
            beta = norm(&w);
            w.scale(&[1.0 / beta]);
            if j + 1 < m {
                h[(j + 1) * m + j] = C64::new(beta, 0.0);
                h[j * m + j + 1] = C64::new(beta, 0.0);
            }
            basis.push(w);
        }
        // Ritz pairs of B, the largest first.
        let (theta, y) = hermitian_eigen(m, &h);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&p, &q| theta[q].total_cmp(&theta[p]));
        let ritz: Vec<V> = order[..keep]
            .iter()
            .map(|&k| {
                let mut x = basis[0].zeros_like();
                for (j, v) in basis.iter().enumerate().take(m) {
                    x.axpy(&[y[j * m + k]], v);
                }
                x
            })
            .collect();
        // The wanted pairs of A and their residuals.
        let mut values = Vec::with_capacity(wanted);
        let mut residuals = Vec::with_capacity(wanted);
        for x in &ritz[..wanted] {
            let mut ax = x.zeros_like();
            b.apply_a(&mut ax, x);
            let lambda = dot(x, &ax).re / dot(x, x).re;
            ax.axpy(&[C64::new(-lambda, 0.0)], x);
            values.push(lambda);
            residuals.push(norm(&ax) / norm(x));
        }
        let converged = values
            .iter()
            .zip(&residuals)
            .all(|(l, r)| *r <= tolerance * l.abs());
        if converged || restarts == max_restarts {
            let mut pairs: Vec<(f64, f64, V)> = values
                .into_iter()
                .zip(residuals)
                .zip(ritz.into_iter().take(wanted))
                .map(|((l, r), x)| (l, r, x))
                .collect();
            pairs.sort_by(|p, q| p.0.total_cmp(&q.0));
            let mut out = Eigenpairs {
                values: Vec::new(),
                vectors: Vec::new(),
                residuals: Vec::new(),
                converged,
                restarts,
                applications: b.applications,
            };
            for (l, r, mut x) in pairs {
                x.scale(&[1.0 / norm(&x)]);
                out.values.push(l);
                out.residuals.push(r);
                out.vectors.push(x);
            }
            return out;
        }
        // Thick restart: the kept Ritz vectors and the residual direction.
        let residual_direction = basis.pop().expect("the residual direction");
        basis = ritz;
        basis.push(residual_direction);
        h.iter_mut().for_each(|z| *z = C64::ZERO);
        for (i, &k) in order[..keep].iter().enumerate() {
            h[i * m + i] = C64::new(theta[k], 0.0);
            let s = y[(m - 1) * m + k].scale(beta);
            h[keep * m + i] = s.conj();
            h[i * m + keep] = s;
        }
        kept = keep;
        restarts += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{random_spinor, Field, Lattice, Spinor};

    /// A diagonal operator with the eigenvalue `d[k]` on complex component k.
    struct Diagonal(Vec<f64>);

    impl LinearFn<Field<f64, Spinor>> for Diagonal {
        fn mult(&mut self, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>) {
            for ((o, i), d) in out
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(inp.as_slice().as_chunks::<2>().0)
                .zip(&self.0)
            {
                o[0] = d * i[0];
                o[1] = d * i[1];
            }
        }
    }

    #[test]
    fn lanczos_finds_the_lowest_eigenvalues() {
        let lat = Lattice::single_node([2, 2, 2, 4]);
        let start = random_spinor(&lat, 1, 7);
        let n = start.as_slice().len() / 2;
        // A spectrum in [1e-6, 4] with an isolated low group, like that of a Dirac operator.
        let mut d: Vec<f64> = (0..n)
            .map(|k| 1e-2 + 4.0 * (k as f64 / n as f64).powi(2))
            .collect();
        d[17] = 3e-6;
        d[101] = 5e-6;
        d[202] = 4e-6;
        let mut sorted = d.clone();
        sorted.sort_by(f64::total_cmp);
        for chebyshev in [
            None,
            Some(Chebyshev {
                degree: 40,
                lo: 2e-2,
                hi: 4.1,
            }),
        ] {
            let params = LanczosParams {
                wanted: 4,
                keep: 8,
                max_dim: 24,
                tolerance: 1e-8,
                max_restarts: 300,
                chebyshev,
            };
            let e = lowest(&mut Diagonal(d.clone()), &start, params);
            assert!(
                e.converged,
                "{chebyshev:?}: not converged in {} restarts",
                e.restarts
            );
            for (got, want) in e.values.iter().zip(&sorted) {
                assert!((got - want).abs() <= 1e-8 * want, "{got:e} for {want:e}");
            }
            eprintln!(
                "{chebyshev:?}: {} restarts, {} applications",
                e.restarts, e.applications
            );
        }
    }
}
