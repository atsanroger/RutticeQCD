//! The smallest singular value of an operator, and the norm of its resolvent: σ_min(z − A) =
//! 1/‖(z − A)⁻¹‖, by Lanczos on the Hermitian positive (z − A)†(z − A). The ε-pseudospectrum
//! of A is where σ_min(z − A) < ε; a lower bound of σ_min on a contour around the spectrum is
//! what the pseudospectral bound of GMRES needs (‖r_k‖/‖r_0‖ ≤ L/(2π ε) max over the contour
//! of |p_k|, L its length), with no eigenvector condition number in it.

use super::transform::Shifted;
use super::{lowest, LanczosParams};
use crate::{LinearOp, Normal, Vector, C64};

/// The smallest singular value and its right singular vector.
pub struct Singular<V> {
    pub value: f64,
    pub vector: V,
    pub converged: bool,
    pub restarts: usize,
    /// Applications of A†A.
    pub applications: usize,
}

/// σ_min(A) by Lanczos on A†A from `start` (`params.wanted` is taken as 1).
pub fn smallest_singular<V: Vector, A: LinearOp<V>>(
    a: &mut A,
    start: &V,
    params: LanczosParams,
) -> Singular<V> {
    let params = LanczosParams {
        wanted: 1,
        ..params
    };
    let mut e = lowest(&mut Normal::new(a, start), start, params);
    Singular {
        value: e.values[0].max(0.0).sqrt(),
        vector: e.vectors.swap_remove(0),
        converged: e.converged,
        restarts: e.restarts,
        applications: e.applications,
    }
}

/// σ_min(z − A) at a point.
#[derive(Clone, Copy, Debug)]
pub struct Resolvent {
    pub z: C64,
    pub sigma_min: f64,
    pub converged: bool,
    pub applications: usize,
}

/// σ_min(z − A) at every point of `points`.
pub fn resolvent_norms<V: Vector, A: LinearOp<V>>(
    a: &mut A,
    points: &[C64],
    start: &V,
    params: LanczosParams,
) -> Vec<Resolvent> {
    points
        .iter()
        .map(|&z| {
            let s = smallest_singular(&mut Shifted { a: &mut *a, z }, start, params);
            Resolvent {
                z,
                sigma_min: s.value,
                converged: s.converged,
                applications: s.applications,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{random_spinor, Field, Lattice, LinearFn, Spinor};

    /// A normal operator: the eigenvalue λ_k on complex component k.
    struct Diagonal(Vec<C64>);

    impl Diagonal {
        fn apply(&self, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>, conj: bool) {
            for ((o, i), d) in out
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(inp.as_slice().as_chunks::<2>().0)
                .zip(&self.0)
            {
                let d = if conj { d.conj() } else { *d };
                let y = d * C64::new(i[0], i[1]);
                o[0] = y.re;
                o[1] = y.im;
            }
        }
    }

    impl LinearFn<Field<f64, Spinor>> for Diagonal {
        fn mult(&mut self, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>) {
            self.apply(out, inp, false);
        }
    }

    impl LinearOp<Field<f64, Spinor>> for Diagonal {
        fn mult_dag(&mut self, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>) {
            self.apply(out, inp, true);
        }
    }

    /// For a normal operator σ_min(z − A) is the distance from z to its spectrum.
    #[test]
    fn resolvent_norm_of_a_normal_operator() {
        let lat = Lattice::single_node([2, 2, 2, 4]);
        let start = random_spinor(&lat, 1, 5);
        let n = start.as_slice().len() / 2;
        let lambda: Vec<C64> = (0..n)
            .map(|k| {
                let phi = 2.399963 * k as f64;
                C64::new(1.0 + 0.5 * phi.cos(), 0.5 * phi.sin())
            })
            .collect();
        let points = [C64::new(0.0, 0.0), C64::new(0.3, 0.2), C64::new(1.0, 0.0)];
        let params = LanczosParams {
            wanted: 1,
            keep: 6,
            max_dim: 30,
            tolerance: (1e-10).into(),
            order: crate::eigen::Order::Ascending,
            threshold: None,
            max_restarts: 400,
            chebyshev: None,
        };
        let got = resolvent_norms(&mut Diagonal(lambda.clone()), &points, &start, params);
        for r in got {
            let want = lambda
                .iter()
                .map(|l| (*l - r.z).norm_sqr().sqrt())
                .fold(f64::INFINITY, f64::min);
            eprintln!("z {:?}: sigma_min {} (exact {want})", r.z, r.sigma_min);
            assert!(r.converged && (r.sigma_min - want).abs() < 1e-6 * want.max(1e-3));
        }
    }
}
