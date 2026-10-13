//! Block GCR: flexible GCR keeping the last k blocks of directions, for several right-hand
//! sides that share them.

use super::{true_residual, SolveError, SolveInfo, Solver, Stop};
use crate::{LinearFn, Vector, C64};

/// Flexible block GCR(k). Every step applies the preconditioner to all residuals at once,
/// Z = M R, makes the columns of Q = A Z orthogonal to those of the last `k` blocks and to one
/// another (and Z alike), and minimises every residual over the whole block: each right-hand
/// side uses the directions made for all of them. With one right-hand side it is `Gcr`.
pub struct BlockGcr<V> {
    pub stop: Stop,
    pub k: usize,
    /// The blocks (Z, Q), the columns of Q = A Z orthonormal or zero, oldest first; one more
    /// than `k`.
    blocks: Vec<(V, V)>,
    r: Option<V>,
}

impl<V: Vector> BlockGcr<V> {
    pub fn new(stop: Stop, k: usize) -> Self {
        Self {
            stop,
            k,
            blocks: Vec::new(),
            r: None,
        }
    }
}

/// `c[j] = <a_i, b_j>` for every column j of `b`.
fn column_dots<V: Vector>(a: &V, i: usize, b: &V, c: &mut [C64]) {
    a.select(&vec![i; b.num_vectors()]).dot(b, c);
}

/// `b_j += c[j] a_i` for every column j of `b`.
fn add_column<V: Vector>(b: &mut V, c: &[C64], a: &V, i: usize) {
    b.axpy(c, &a.select(&vec![i; c.len()]));
}

impl<V: Vector> Solver<V> for BlockGcr<V> {
    fn solve(
        &mut self,
        a: &mut dyn LinearFn<V>,
        mut m: Option<&mut dyn LinearFn<V>>,
        x: &mut V,
        b: &V,
    ) -> Result<SolveInfo, SolveError> {
        assert!(x.same_shape(b));
        let nv = b.num_vectors();
        if !self.r.as_ref().is_some_and(|r| r.same_shape(b)) {
            self.r = Some(b.zeros_like());
            self.blocks = (0..=self.k)
                .map(|_| (b.zeros_like(), b.zeros_like()))
                .collect();
        }
        let r = self.r.as_mut().expect("made above");
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let mut b_norm = vec![0.0; nv];
        b.norm_sqr(&mut b_norm);
        let tol = self.stop.tolerance_sqr;
        let limit = self.stop.max_iterations * (self.stop.max_restarts + 1);
        let (mut rr, mut nq, mut residual) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
        let mut dots = vec![C64::ZERO; nv];
        let (mut iterations, mut cycles) = (0, 0);
        loop {
            a.mult_xpay(r, x, &minus_one, b);
            let (mut next, mut kept) = (0, 0);
            let mut in_cycle = 0;
            while in_cycle < self.stop.max_iterations && iterations < limit {
                r.norm_sqr(&mut rr);
                let active: Vec<bool> = (0..nv).map(|j| rr[j] > tol * b_norm[j]).collect();
                if !active.iter().any(|a| *a) {
                    break;
                }
                let slots = self.blocks.len();
                let z = next;
                {
                    let (zv, qv) = &mut self.blocks[z];
                    match m.as_mut() {
                        Some(m) => m.mult(zv, r),
                        None => zv.copy_from(r),
                    }
                    a.mult(qv, zv);
                }
                // Orthogonal to every column of the kept blocks.
                for back in 1..=kept {
                    let old = (next + slots - back) % slots;
                    let (left, right) = if old < z {
                        let (l, r) = self.blocks.split_at_mut(z);
                        (&l[old], &mut r[0])
                    } else {
                        let (l, r) = self.blocks.split_at_mut(old);
                        (&r[0], &mut l[z])
                    };
                    for i in 0..nv {
                        column_dots(&left.1, i, &right.1, &mut dots);
                        let minus: Vec<C64> = dots.iter().map(|d| -*d).collect();
                        add_column(&mut right.1, &minus, &left.1, i);
                        add_column(&mut right.0, &minus, &left.0, i);
                    }
                }
                // Orthonormal within the block, one column after the other; the direction of a
                // converged right-hand side, or one that depends on those before it, is dropped.
                let (zv, qv) = &mut self.blocks[z];
                let mut first = vec![0.0; nv];
                qv.norm_sqr(&mut first);
                let mut any = false;
                for i in 0..nv {
                    qv.norm_sqr(&mut nq);
                    let keep = active[i] && nq[i].is_finite() && nq[i] > 1e-24 * first[i];
                    any |= keep;
                    let mut scale = vec![1.0; nv];
                    scale[i] = if keep { 1.0 / nq[i].sqrt() } else { 0.0 };
                    qv.scale(&scale);
                    zv.scale(&scale);
                    if !keep || i + 1 == nv {
                        continue;
                    }
                    column_dots(&*qv, i, &*qv, &mut dots);
                    let minus: Vec<C64> = (0..nv)
                        .map(|j| if j > i { -dots[j] } else { C64::ZERO })
                        .collect();
                    let (q_i, z_i) = (qv.select(&vec![i; nv]), zv.select(&vec![i; nv]));
                    qv.axpy(&minus, &q_i);
                    zv.axpy(&minus, &z_i);
                }
                if !any {
                    return Err(SolveError::Breakdown {
                        iterations,
                        what: "A Z vanished",
                    });
                }
                // Every residual minimised over the block.
                for i in 0..nv {
                    column_dots(&*qv, i, &*r, &mut dots);
                    add_column(x, &dots, &*zv, i);
                    let minus: Vec<C64> = dots.iter().map(|d| -*d).collect();
                    add_column(r, &minus, &*qv, i);
                }
                next = (next + 1) % slots;
                kept = (kept + 1).min(self.k);
                iterations += 1;
                in_cycle += 1;
            }
            true_residual(a, x, b, &b_norm, r, &minus_one, &mut residual);
            let info = SolveInfo {
                iterations,
                restarts: cycles,
                residual_sqr: residual.clone(),
            };
            if residual.iter().all(|r| *r <= tol) {
                return Ok(info);
            }
            if residual.iter().any(|r| !r.is_finite()) {
                return Err(SolveError::Breakdown {
                    iterations,
                    what: "the solution is not finite",
                });
            }
            if cycles >= self.stop.max_restarts || iterations >= limit {
                return Err(SolveError::NotConverged(info));
            }
            cycles += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::{Bound, FermionOp, Variant, Wilson, WilsonParams};
    use crate::gauge::random_gauge;
    use crate::solver::Gcr;
    use crate::{random_spinor, Boundary, Lattice};

    /// 1 + Σ_k c_k v_k v_k† for orthonormal v_k: `outliers` eigenvalues away from 1.
    struct Outliers {
        v: crate::Field<f64, crate::Spinor>,
        c: Vec<f64>,
    }

    impl LinearFn<crate::Field<f64, crate::Spinor>> for Outliers {
        fn mult(
            &mut self,
            out: &mut crate::Field<f64, crate::Spinor>,
            inp: &crate::Field<f64, crate::Spinor>,
        ) {
            let nv = inp.num_vectors();
            out.copy_from(inp);
            let mut d = vec![C64::ZERO; nv];
            for (k, c) in self.c.iter().enumerate() {
                let v = self.v.select(&vec![k; nv]);
                v.dot(inp, &mut d);
                let scaled: Vec<C64> = d.iter().map(|z| z.scale(*c)).collect();
                out.axpy(&scaled, &v);
            }
        }
    }

    /// With six eigenvalues away from the rest, one right-hand side needs a step for each;
    /// six right-hand sides find them together in the first steps.
    #[test]
    fn a_block_finds_the_outlying_modes_together() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let mut v = random_spinor(&lat, 6, 21);
        for i in 0..6 {
            let mut vi = v.select(&[i]);
            for j in 0..i {
                let q = v.select(&[j]);
                let mut d = [C64::ZERO];
                q.dot(&vi, &mut d);
                vi.axpy(&[-d[0]], &q);
            }
            let mut n = [0.0];
            vi.norm_sqr(&mut n);
            vi.scale(&[1.0 / n[0].sqrt()]);
            v.assign(&[i], &vi);
        }
        let mut a = Outliers {
            v,
            c: vec![9.0, 19.0, 49.0, 99.0, 499.0, 999.0],
        };
        let b = random_spinor(&lat, 6, 23);
        let stop = Stop::simple(1e-20, 100);
        let (mut plain, mut block) = (b.zeros_like(), b.zeros_like());
        let plain_info = Gcr::new(stop, 8)
            .solve(&mut a, None, &mut plain, &b)
            .unwrap();
        let block_info = BlockGcr::new(stop, 8)
            .solve(&mut a, None, &mut block, &b)
            .unwrap();
        assert!(plain_info.iterations >= 6, "{}", plain_info.iterations);
        assert!(block_info.iterations <= 3, "{}", block_info.iterations);
    }

    /// Several right-hand sides share their directions: the same solutions as GCR, in no more
    /// steps and in fewer where the operator has low modes; with one right-hand side the two
    /// are the same method.
    #[test]
    fn block_gcr_shares_the_directions() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 11);
        let stop = Stop::simple(1e-20, 4000);
        for (kappa, nv) in [(0.11, 1), (0.11, 6), (0.14, 6), (0.16, 12)] {
            let mut op = Wilson::<f64>::new(
                &lat,
                WilsonParams {
                    kappa,
                    boundary: [Boundary::Periodic; 4],
                },
            );
            op.set_gauge(&u).unwrap();
            let b = random_spinor(&lat, nv, 13);
            let mut a = Bound::new(&op, Variant::D);
            let (mut plain, mut block) = (b.zeros_like(), b.zeros_like());
            let plain_info = Gcr::new(stop, 8)
                .solve(&mut a, None, &mut plain, &b)
                .unwrap();
            let block_info = BlockGcr::new(stop, 8)
                .solve(&mut a, None, &mut block, &b)
                .unwrap();
            eprintln!(
                "kappa {kappa}, {nv} right-hand sides: GCR(8) {} steps, block GCR(8) {}",
                plain_info.iterations, block_info.iterations
            );
            assert!(block_info.residual_sqr.iter().all(|r| *r <= 1e-20));
            let mut d = plain.zeros_like();
            d.copy_from(&plain);
            d.axpy(&vec![C64::new(-1.0, 0.0); nv], &block);
            let (mut diff, mut size) = (vec![0.0; nv], vec![0.0; nv]);
            d.norm_sqr(&mut diff);
            plain.norm_sqr(&mut size);
            assert!(diff.iter().zip(&size).all(|(d, s)| *d < 1e-16 * s));
            if nv == 1 {
                assert_eq!(block_info.iterations, plain_info.iterations);
            } else {
                assert!(block_info.iterations <= plain_info.iterations);
            }
        }
    }
}
