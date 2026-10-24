//! Eigenpairs of a Hermitian operator at an end of its spectrum, or of its magnitude.
//!
//! [`lanczos`]: the thick-restart Lanczos method (Wu and Simon, SIAM J. Matrix Anal. Appl. 22
//! (2000) 602) on any [`Vector`] (on the host or on a device), at the end of the spectrum that
//! [`Order`] names (Bridge++'s `eigensolver_mode`). The lowest or highest eigenvalues come from
//! Lanczos on A; those of smallest or largest magnitude of an indefinite A (γ5 D) from Lanczos on
//! A², whose Ritz vectors mix the eigenvectors of ±λ, followed by the Rayleigh-Ritz procedure of
//! A on them. Lanczos converges first at the ends of the spectrum, and the low end of a Dirac
//! operator is crowded; with a Chebyshev polynomial ([`Chebyshev`]) Lanczos runs on
//! B = T_d(ℓ(A)) (or T_d(ℓ(A²))) instead, ℓ mapping [lo, hi] onto [−1, 1], for which the
//! eigenvalues below lo are the largest by far: the same eigenvectors, found in far fewer
//! restarts. The eigenvalues of A are Rayleigh quotients, and convergence is judged by
//! |A x − λ x| itself ([`Tolerance`]); with a threshold the run stops once the eigenvalues up to
//! it have converged (Bridge++'s `threshold_value`).
//!
//! Every Lanczos vector is kept and made orthogonal to all the others twice (no loss of
//! orthogonality): the memory is `max_dim` + `keep` vectors at a restart.

use super::{combine, dot, norm};
use crate::numerics::hermitian_eigen;
use crate::{LinearFn, Vector, C64};

/// Which eigenvalues of a Hermitian operator, in the order they are wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// The lowest first.
    Ascending,
    /// The highest first.
    Descending,
    /// The smallest in magnitude first (by Lanczos on A²).
    AbsAscending,
    /// The largest in magnitude first (by Lanczos on A²).
    AbsDescending,
}

impl Order {
    /// Whether `x` comes before `y`.
    pub fn before(self, x: f64, y: f64) -> bool {
        match self {
            Order::Ascending => x < y,
            Order::Descending => x > y,
            Order::AbsAscending => x.abs() < y.abs(),
            Order::AbsDescending => x.abs() > y.abs(),
        }
    }

    /// Whether Lanczos runs on A² (and on its low end).
    fn squared(self) -> bool {
        matches!(self, Order::AbsAscending | Order::AbsDescending)
    }

    fn low_end(self) -> bool {
        matches!(self, Order::Ascending | Order::AbsAscending)
    }
}

/// When an eigenpair (λ, x), |x| = 1, has converged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tolerance {
    /// |A x − λ x| ≤ ε |λ|.
    Relative(f64),
    /// |A x − λ x|² < ε (Bridge++'s `convergence_criterion_squared`).
    AbsoluteSquared(f64),
}

impl Tolerance {
    pub(crate) fn met(self, residual: f64, magnitude: f64) -> bool {
        match self {
            Tolerance::Relative(e) => residual <= e * magnitude,
            Tolerance::AbsoluteSquared(e) => residual * residual < e,
        }
    }
}

impl From<f64> for Tolerance {
    /// Relative.
    fn from(e: f64) -> Self {
        Tolerance::Relative(e)
    }
}

/// The polynomial T_d(ℓ(A)) Lanczos runs on (T_d(ℓ(A²)) for the orders by magnitude),
/// ℓ(x) = (2x − (hi + lo)) / (hi − lo). `degree` must be even, so that the eigenvalues below lo
/// (where |ℓ| > 1) become large and positive; hi must bound the spectrum from above, and lo, hi
/// are on the scale of A² for the orders by magnitude. The polynomial is applied by the
/// three-term recurrence with no rescaling, so T_d(ℓ(0)) must stay a representable number.
/// For the low end only (`Ascending`, `AbsAscending`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chebyshev {
    pub degree: usize,
    pub lo: f64,
    pub hi: f64,
}

/// The settings of [`lanczos`] and [`lowest`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LanczosParams {
    /// Eigenpairs wanted.
    pub wanted: usize,
    /// Ritz vectors kept at a restart (at least `wanted`).
    pub keep: usize,
    /// Lanczos vectors before a restart (more than `keep`).
    pub max_dim: usize,
    /// When a wanted pair has converged.
    pub tolerance: Tolerance,
    pub max_restarts: usize,
    /// Run on a Chebyshev polynomial (see the module); none: on ∓A (∓A²).
    pub chebyshev: Option<Chebyshev>,
    /// [`lanczos`]: which end; [`lowest`] takes `Ascending` whatever this says.
    pub order: Order,
    /// Stop as soon as the wanted pairs up to the first one at or beyond this value (in
    /// `order`) have converged; without it, when all have.
    pub threshold: Option<f64>,
}

/// The eigenpairs found, in the order asked for.
pub struct Eigenpairs<V> {
    pub values: Vec<f64>,
    /// Normalised.
    pub vectors: Vec<V>,
    /// |A x − λ x| of each.
    pub residuals: Vec<f64>,
    /// The pairs before the threshold (all of them without one).
    pub(crate) within_threshold: usize,
    pub converged: bool,
    pub restarts: usize,
    /// Applications of A.
    pub applications: usize,
}

/// B: the operator Lanczos runs on.
struct Operator<'a, V, A> {
    a: &'a mut A,
    squared: bool,
    low_end: bool,
    chebyshev: Option<Chebyshev>,
    work: [V; 3],
    /// A x, for A² x.
    half: V,
    applications: usize,
}

/// A, or A² through `half`.
fn base<V: Vector, A: LinearFn<V>>(
    a: &mut A,
    squared: bool,
    half: &mut V,
    applications: &mut usize,
    out: &mut V,
    inp: &V,
) {
    if squared {
        a.mult(half, inp);
        a.mult(out, half);
        *applications += 2;
    } else {
        a.mult(out, inp);
        *applications += 1;
    }
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
            squared,
            low_end,
            chebyshev,
            work,
            half,
            applications,
        } = self;
        let Some(c) = *chebyshev else {
            base(*a, *squared, half, applications, out, inp);
            if *low_end {
                out.scale(&[-1.0]);
            }
            return;
        };
        // ℓ(A) v = (2 A v − (hi + lo) v) / (hi − lo).
        let (scale, shift) = (2.0 / (c.hi - c.lo), -(c.hi + c.lo) / (c.hi - c.lo));
        let mut mapped = |out: &mut V, inp: &V| {
            base(*a, *squared, half, applications, out, inp);
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

/// The lowest `params.wanted` eigenpairs of the Hermitian operator `a` (`lanczos` with
/// `Order::Ascending`).
pub fn lowest<V: Vector, A: LinearFn<V>>(
    a: &mut A,
    start: &V,
    params: LanczosParams,
) -> Eigenpairs<V> {
    lanczos(
        a,
        start,
        LanczosParams {
            order: Order::Ascending,
            ..params
        },
    )
}

/// `params.wanted` eigenpairs of the Hermitian operator `a` at the end `params.order`, from
/// the start vector `start` (one vector; not orthogonal to the wanted eigenvectors).
pub fn lanczos<V: Vector, A: LinearFn<V>>(
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
        order,
        threshold,
    } = params;
    assert_eq!(start.num_vectors(), 1);
    assert!(wanted >= 1 && keep >= wanted && m > keep);
    if let Some(c) = chebyshev {
        assert!(c.degree >= 2 && c.degree % 2 == 0, "an even degree");
        assert!(0.0 <= c.lo && c.lo < c.hi);
        assert!(
            order.low_end(),
            "the Chebyshev polynomial is for the low end"
        );
    }
    let mut b = Operator {
        a,
        squared: order.squared(),
        low_end: order.low_end(),
        chebyshev,
        work: [start.zeros_like(), start.zeros_like(), start.zeros_like()],
        half: start.zeros_like(),
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
        let mut order_b: Vec<usize> = (0..m).collect();
        order_b.sort_by(|&p, &q| theta[q].total_cmp(&theta[p]));
        let ritz: Vec<V> = order_b[..keep]
            .iter()
            .map(|&k| combine(&basis[..m], |j| y[j * m + k]))
            .collect();
        // The wanted pairs of A and their residuals: the Ritz vectors of B themselves (kept for
        // the restart; `None`), or those of the Rayleigh-Ritz procedure.
        let mut pairs: Vec<(f64, f64, usize, Option<V>)> = if order.squared() {
            rayleigh_ritz(&mut b, &ritz[..wanted])
                .into_iter()
                .enumerate()
                .map(|(k, (l, r, z))| (l, r, k, Some(z)))
                .collect()
        } else {
            ritz[..wanted]
                .iter()
                .enumerate()
                .map(|(k, x)| {
                    let mut ax = x.zeros_like();
                    b.apply_a(&mut ax, x);
                    let lambda = dot(x, &ax).re / dot(x, x).re;
                    ax.axpy(&[C64::new(-lambda, 0.0)], x);
                    (lambda, norm(&ax) / norm(x), k, None)
                })
                .collect()
        };
        pairs.sort_by(|p, q| {
            if order.before(p.0, q.0) {
                std::cmp::Ordering::Less
            } else if order.before(q.0, p.0) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let met: Vec<bool> = pairs
            .iter()
            .map(|(l, r, _, _)| tolerance.met(*r, l.abs()))
            .collect();
        let within = threshold.map_or(wanted, |t| {
            pairs
                .iter()
                .position(|(l, _, _, _)| !order.before(*l, t))
                .unwrap_or(wanted)
        });
        let converged = met[..(within + 1).min(wanted)].iter().all(|&c| c);
        if converged || restarts == max_restarts {
            let mut out = Eigenpairs {
                values: Vec::new(),
                vectors: Vec::new(),
                residuals: Vec::new(),
                within_threshold: within,
                converged,
                restarts,
                applications: b.applications,
            };
            let mut ritz: Vec<Option<V>> = ritz.into_iter().map(Some).collect();
            for (l, r, k, z) in pairs {
                let mut x = z.or_else(|| ritz[k].take()).expect("each Ritz vector once");
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
        for (i, &k) in order_b[..keep].iter().enumerate() {
            h[i * m + i] = C64::new(theta[k], 0.0);
            let s = y[(m - 1) * m + k].scale(beta);
            h[keep * m + i] = s.conj();
            h[i * m + keep] = s;
        }
        kept = keep;
        restarts += 1;
    }
}

/// The Rayleigh-Ritz procedure of A on the orthonormal vectors `x`: (λ, |A z − λ z|, z) for
/// each eigenpair of x† A x, z normalised. A is applied twice per vector, one vector at a time.
fn rayleigh_ritz<V: Vector, A: LinearFn<V>>(
    b: &mut Operator<'_, V, A>,
    x: &[V],
) -> Vec<(f64, f64, V)> {
    let w = x.len();
    let mut g = vec![C64::ZERO; w * w];
    let mut ax = x[0].zeros_like();
    for (j, xj) in x.iter().enumerate() {
        b.apply_a(&mut ax, xj);
        for (i, xi) in x.iter().enumerate() {
            g[i * w + j] = dot(xi, &ax);
        }
    }
    let (lambda, y) = hermitian_eigen(w, &g);
    (0..w)
        .map(|k| {
            let mut z = combine(x, |j| y[j * w + k]);
            z.scale(&[1.0 / norm(&z)]);
            b.apply_a(&mut ax, &z);
            ax.axpy(&[C64::new(-lambda[k], 0.0)], &z);
            (lambda[k], norm(&ax), z)
        })
        .collect()
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

    fn params(chebyshev: Option<Chebyshev>, order: Order) -> LanczosParams {
        LanczosParams {
            wanted: 4,
            keep: 8,
            max_dim: 24,
            tolerance: Tolerance::Relative(1e-8),
            max_restarts: 300,
            chebyshev,
            order,
            threshold: None,
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
            let e = lowest(
                &mut Diagonal(d.clone()),
                &start,
                params(chebyshev, Order::Descending),
            );
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

    /// The four orders on an indefinite spectrum, with and without the polynomial, the absolute
    /// criterion, and a threshold.
    #[test]
    fn every_order_of_an_indefinite_spectrum() {
        let lat = Lattice::single_node([2, 2, 2, 4]);
        let start = random_spinor(&lat, 1, 9);
        let n = start.as_slice().len() / 2;
        // ±[0.05, 2] with a few small ones of either sign, like γ5 D.
        let mut d: Vec<f64> = (0..n)
            .map(|k| {
                let s = if k % 3 == 0 { -1.0 } else { 1.0 };
                s * (0.05 + 1.95 * (k as f64 / n as f64))
            })
            .collect();
        d[11] = 2e-3;
        d[40] = -3e-3;
        d[77] = 1e-2;
        let sorted_by = |order: Order| {
            let mut s = d.clone();
            s.sort_by(|x, y| {
                if order.before(*x, *y) {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });
            s
        };
        let cheb = Some(Chebyshev {
            degree: 40,
            lo: 0.07f64.powi(2),
            hi: 4.1,
        });
        for (order, chebyshev) in [
            (Order::Ascending, None),
            (Order::Descending, None),
            (Order::AbsAscending, None),
            (Order::AbsAscending, cheb),
            (Order::AbsDescending, None),
        ] {
            let p = LanczosParams {
                tolerance: Tolerance::AbsoluteSquared(1e-22),
                max_restarts: 2000,
                ..params(chebyshev, order)
            };
            let e = lanczos(&mut Diagonal(d.clone()), &start, p);
            let want = sorted_by(order);
            eprintln!(
                "{order:?} {}: {:?}, {} restarts, {} applications",
                chebyshev.is_some(),
                e.values,
                e.restarts,
                e.applications
            );
            assert!(e.converged, "{order:?}: not converged");
            for ((got, want), r) in e.values.iter().zip(&want).zip(&e.residuals) {
                assert!(
                    (got - want).abs() <= 1e-10,
                    "{order:?}: {got:e} for {want:e}"
                );
                assert!(r * r < 1e-22, "{order:?}: residual {r:e}");
            }
        }
        // The magnitudes up to 0.02: three of them, and the run stops once the fourth (0.05)
        // has converged.
        let e = lanczos(
            &mut Diagonal(d.clone()),
            &start,
            LanczosParams {
                wanted: 6,
                keep: 10,
                threshold: Some(0.02),
                tolerance: Tolerance::AbsoluteSquared(1e-22),
                max_restarts: 2000,
                ..params(cheb, Order::AbsAscending)
            },
        );
        assert!(e.converged);
        assert_eq!(e.within_threshold, 3, "{:?}", e.values);
    }
}
