//! The control for `mdwf_boundary`: the same measurements on the Wilson operator, whose
//! aggregation multigrid works. Is the residual of the blockwise projection of a low mode
//! (75 times that of the mode for the domain-wall A) particular to domain-wall fermions?
//!
//! ```text
//! cargo run --release --example wilson_boundary -- CONFIG L KAPPA [K] [M] [ITERS]
//! ```
//!
//! The lowest K right singular vectors of D (eigenvectors of D†D) by Chebyshev-filtered
//! subspace iteration with Rayleigh-Ritz on M vectors, the degree following √(hi/lo), as
//! `mdwf_coherence_eig`. KAPPA near its critical value makes σ_0 / |D| as small as for A.
//! Then, P split by γ5 as the Wilson multigrid does:
//! - Lüscher's test: |P P† v_j| for j = N..K−1 with P from the lowest N (blocks 2, 3, 4);
//! - for N = 8 and t the next four vectors: |D PP†t| / |D t| (blocks 2, 3, 4, 6), and the
//!   share of |D PP†t|² at distance d from the nearest block face against the share of sites.

use rayon::prelude::*;
use ruttice::fermion::{Bound, FermionOp, HopTerms, Variant, Wilson, WilsonParams};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{largest_magnitude, ChebyshevFilter, Fgmres, Solve, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, Normal, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: wilson_boundary CONFIG L KAPPA [K] [M] [ITERS]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

trait CoarseNorm {
    fn norm(&self) -> f64;
}

impl CoarseNorm for C {
    fn norm(&self) -> f64 {
        let mut n = vec![0.0; self.num_vectors()];
        self.norm_sqr(&mut n);
        n.iter().sum::<f64>().sqrt()
    }
}

fn norms(x: &V) -> Vec<f64> {
    let mut n = vec![0.0; x.num_vectors()];
    x.norm_sqr(&mut n);
    n.iter().map(|v| v.sqrt()).collect()
}

fn norm(x: &V) -> f64 {
    norms(x).iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `G[i * ny + j] = ⟨x_i, y_j⟩` (fields of the same site layout, `[site][vector][reals]`).
fn gram(x: &V, y: &V) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let reals = x.site_len() / nx;
    (0..x.num_sites())
        .into_par_iter()
        .fold(
            || vec![C64::ZERO; nx * ny],
            |mut g, site| {
                let (xs, ys) = (x.site(0, site), y.site(0, site));
                for i in 0..nx {
                    let xi = &xs[i * reals..(i + 1) * reals];
                    for j in 0..ny {
                        let yj = &ys[j * reals..(j + 1) * reals];
                        let (mut re, mut im) = (0.0, 0.0);
                        for (a, b) in xi.as_chunks::<2>().0.iter().zip(yj.as_chunks::<2>().0) {
                            re += a[0] * b[0] + a[1] * b[1];
                            im += a[0] * b[1] - a[1] * b[0];
                        }
                        g[i * ny + j] += C64::new(re, im);
                    }
                }
                g
            },
        )
        .reduce(
            || vec![C64::ZERO; nx * ny],
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += b;
                }
                a
            },
        )
}

/// `x_j ← Σ_i x_i c[i * n + j]`, site by site.
fn rotate(x: &mut V, c: &[C64]) {
    let n = x.num_vectors();
    let reals = x.site_len() / n;
    for site in 0..x.num_sites() {
        let s = x.site_mut(0, site);
        let old: Vec<C64> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| C64::new(z[0], z[1]))
            .collect();
        for j in 0..n {
            for d in 0..reals / 2 {
                let mut z = C64::ZERO;
                for i in 0..n {
                    z += old[i * reals / 2 + d] * c[i * n + j];
                }
                s[j * reals + 2 * d] = z.re;
                s[j * reals + 2 * d + 1] = z.im;
            }
        }
    }
}

/// Eigenvalues (ascending) and eigenvectors (columns) of a Hermitian matrix, cyclic Jacobi
/// as in `mdwf_coherence_eig` (which has its test).
fn eigh(mut a: Vec<C64>, n: usize) -> (Vec<f64>, Vec<C64>) {
    let mut v = vec![C64::ZERO; n * n];
    for i in 0..n {
        v[i * n + i] = C64::new(1.0, 0.0);
    }
    let total: f64 = a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q].norm_sqr();
            }
        }
        if off.sqrt() <= 1e-15 * total {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let g = a[p * n + q].norm_sqr().sqrt();
                if g <= 1e-300 {
                    continue;
                }
                let d = a[p * n + q].conj().scale(1.0 / g);
                for k in 0..n {
                    a[k * n + q] = a[k * n + q] * d;
                    v[k * n + q] = v[k * n + q] * d;
                }
                for k in 0..n {
                    a[q * n + k] = a[q * n + k] * d.conj();
                }
                let theta = (a[q * n + q].re - a[p * n + p].re) / (2.0 * g);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = kp.scale(c) - kq.scale(s);
                    a[k * n + q] = kp.scale(s) + kq.scale(c);
                    let (kp, kq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = kp.scale(c) - kq.scale(s);
                    v[k * n + q] = kp.scale(s) + kq.scale(c);
                }
                for k in 0..n {
                    let (pk, qk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = pk.scale(c) - qk.scale(s);
                    a[q * n + k] = pk.scale(s) + qk.scale(c);
                }
                a[p * n + q] = C64::ZERO;
                a[q * n + p] = C64::ZERO;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|i, j| a[i * n + i].re.total_cmp(&a[j * n + j].re));
    let values = order.iter().map(|i| a[i * n + i].re).collect();
    let mut vectors = vec![C64::ZERO; n * n];
    for (j, o) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + j] = v[i * n + o];
        }
    }
    (values, vectors)
}

fn orthonormalise(y: &mut V) {
    let n = y.num_vectors();
    let (values, mut c) = eigh(gram(y, y), n);
    for (j, l) in values.iter().enumerate() {
        assert!(*l > 1e-13 * values[n - 1], "the subspace lost rank");
        for i in 0..n {
            c[i * n + j] = c[i * n + j].scale(1.0 / l.sqrt());
        }
    }
    rotate(y, &c);
}

/// |P P† t_j|² / |t_j|² for every vector of `t`, P spanned on each (block, chirality) by the
/// first `n` vectors of `halves[chirality]`.
fn captured(agg: &Aggregation, halves: &[V; 2], n: usize, t: &V) -> Vec<f64> {
    let (nb, nt) = (halves[0].num_vectors(), t.num_vectors());
    let reals = t.site_len() / nt;
    assert_eq!(reals, halves[0].site_len() / nb);
    let sums = (0..agg.coarse_lattice().num_sites())
        .into_par_iter()
        .fold(
            || vec![0.0; nt],
            |mut acc, coarse_site| {
                let fine = agg.fine_sites(coarse_site);
                let dim = fine.len() * reals / 2;
                for half in halves {
                    let mut w = vec![C64::ZERO; n * dim];
                    for (k, site) in fine.iter().enumerate() {
                        let hs = half.site(0, *site as usize);
                        for i in 0..n {
                            for c in 0..reals / 2 {
                                let r = i * reals + 2 * c;
                                w[i * dim + k * reals / 2 + c] = C64::new(hs[r], hs[r + 1]);
                            }
                        }
                    }
                    // Gram-Schmidt; dependent vectors are zeroed.
                    for i in 0..n {
                        let (done, rest) = w.split_at_mut(i * dim);
                        let wi = &mut rest[..dim];
                        let before: f64 = wi.iter().map(|z| z.norm_sqr()).sum();
                        for j in 0..i {
                            let wj = &done[j * dim..(j + 1) * dim];
                            let mut o = C64::ZERO;
                            for (a, b) in wj.iter().zip(wi.iter()) {
                                o += a.conj() * *b;
                            }
                            for (a, b) in wj.iter().zip(wi.iter_mut()) {
                                *b = *b - o * *a;
                            }
                        }
                        let after: f64 = wi.iter().map(|z| z.norm_sqr()).sum();
                        let s = if after > 1e-24 * before {
                            1.0 / after.sqrt()
                        } else {
                            0.0
                        };
                        for z in wi.iter_mut() {
                            *z = z.scale(s);
                        }
                    }
                    for j in 0..nt {
                        for i in 0..n {
                            let mut z = C64::ZERO;
                            for (k, site) in fine.iter().enumerate() {
                                let ts = &t.site(0, *site as usize)[j * reals..(j + 1) * reals];
                                let wi = &w[i * dim + k * reals / 2..i * dim + (k + 1) * reals / 2];
                                for (a, b) in wi.iter().zip(ts.as_chunks::<2>().0) {
                                    z += a.conj() * C64::new(b[0], b[1]);
                                }
                            }
                            acc[j] += z.norm_sqr();
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0; nt],
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += b;
                }
                a
            },
        );
    let tn = norms(t);
    sums.iter().zip(tn).map(|(s, n)| s / (n * n)).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(3..=6).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let kappa: f64 = args[2].parse().map_err(|_| USAGE)?;
    let k: usize = args
        .get(3)
        .map_or(Ok(40), |s| s.parse())
        .map_err(|_| USAGE)?;
    let m: usize = args
        .get(4)
        .map_or(Ok(56), |s| s.parse())
        .map_err(|_| USAGE)?;
    let iters: usize = args
        .get(5)
        .map_or(Ok(30), |s| s.parse())
        .map_err(|_| USAGE)?;

    let lat = Lattice::single_node([l; ND]);
    let u = read_bridge_text(&path, &lat)?;
    let mut op = Wilson::<f64>::new(
        &lat,
        WilsonParams {
            kappa,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        },
    );
    op.set_gauge(&u)?;
    println!("{l}^4 Wilson, κ = {kappa}; K = {k} of M = {m}");
    let start = Instant::now();
    let random = |n: usize, seed: u64| random_spinor(&lat, n, seed);
    let like = random(1, 1);
    let normal = |x: &V| Normal::new(Bound::new(&op, Variant::D), x);
    let hi = 1.1 * largest_magnitude(&mut normal(&like), &mut random(1, 5), 50);
    println!("λ_max(D†D) ≈ {:.4}", hi / 1.1);

    let mut x = random(m, 21);
    let mut lo = hi / 1000.0;
    let mut theta = vec![0.0; m];
    for iteration in 0..iters {
        let d = ((1.5 * (hi / lo).sqrt()).ceil() as usize).clamp(40, 2000);
        let mut y = x.zeros_like();
        ChebyshevFilter::new(normal(&x), d, lo, hi, &x).mult(&mut y, &x);
        x = y;
        orthonormalise(&mut x);
        let mut w = x.zeros_like();
        normal(&x).mult(&mut w, &x);
        let (values, c) = eigh(gram(&x, &w), m);
        rotate(&mut x, &c);
        rotate(&mut w, &c);
        let change = (0..k)
            .map(|j| ((values[j] - theta[j]) / values[j]).abs())
            .fold(0.0, f64::max);
        theta = values;
        let scale: Vec<C64> = theta.iter().map(|t| C64::new(-t, 0.0)).collect();
        w.axpy(&scale, &x);
        let worst = norms(&w)[..k]
            .iter()
            .zip(&theta)
            .map(|(r, t)| r / t)
            .fold(0.0, f64::max);
        lo = theta[m - 1];
        println!(
            "iteration {iteration:2}, degree {d}: σ_0 {:.4e}, σ_{} {:.4e}, σ_{} {:.4e}; worst relative residual {worst:.1e}, change {change:.1e} ({:.0} s)",
            theta[0].sqrt(),
            k - 1,
            theta[k - 1].sqrt(),
            m - 1,
            theta[m - 1].sqrt(),
            start.elapsed().as_secs_f64()
        );
        if worst < 1e-2 || change < 1e-4 {
            break;
        }
    }
    println!(
        "σ_0 / |D| = {:.2e} (A of the domain-wall run: 2.6e-3 / 4.5 = 5.8e-4)",
        theta[0].sqrt() / (hi / 1.1).sqrt()
    );

    let v = x.select(&(0..k).collect::<Vec<_>>());
    let first: Vec<usize> = (0..16.min(k)).collect();
    let halves = {
        let x = v.select(&first);
        let mut g = x.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g, &x);
        let mut plus = x.zeros_like();
        plus.copy_from(&x);
        for (a, b) in plus.as_mut_slice().iter_mut().zip(g.as_mut_slice()) {
            (*a, *b) = (0.5 * (*a + *b), 0.5 * (*a - *b));
        }
        [plus, g]
    };
    println!("  Lüscher's test, right singular vectors: mean |PP†v_j| over j = N..K−1");
    for b in [2, 3, 4].into_iter().filter(|b| l.is_multiple_of(*b)) {
        let agg = Aggregation::new(&lat, [b; ND]);
        let mut line = format!("    {b}^4:");
        for n in [4, 8, 16].into_iter().filter(|n| *n < k) {
            let c = captured(&agg, &halves, n, &v);
            let mean = c[n..].iter().map(|x| x.sqrt()).sum::<f64>() / (k - n) as f64;
            line += &format!("  N = {n}: {mean:.3}");
        }
        println!("{line}");
    }

    let n = 8;
    let vectors = v.select(&(0..n).collect::<Vec<_>>());
    let mut g5 = vectors.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
    let mut d_op = Bound::new(&op, Variant::D);
    println!("  N = {n}: |D PP†t| / |D t| for t = v_8..v_11, and the share of |D PP†t|² at distance d (share of sites):");
    for b in [2, 3, 4, 6].into_iter().filter(|b| l.is_multiple_of(*b)) {
        let agg = Aggregation::new(&lat, [b; ND]);
        let p = Basis::from_vectors(&agg, &vectors, &g5)?;
        let dist = |site: usize| {
            lat.coords(site)
                .iter()
                .map(|x| (x % b).min(b - 1 - x % b))
                .min()
                .unwrap_or(0)
        };
        let dmax = (b - 1) / 2;
        let mut sites_at = vec![0usize; dmax + 1];
        for site in 0..lat.num_sites() {
            sites_at[dist(site)] += 1;
        }
        let mut shares = vec![0.0; dmax + 1];
        let mut line = format!("    {b}^4:");
        for j in n..n + 4 {
            let t = v.select(&[j]);
            let mut c = p.coarse_field(1);
            p.restrict(&mut c, &t);
            let mut held = t.zeros_like();
            p.prolong(&mut held, &c);
            let (mut dt, mut dh) = (t.zeros_like(), t.zeros_like());
            d_op.mult(&mut dt, &t);
            d_op.mult(&mut dh, &held);
            line += &format!(" {:.3}/{:.1}", norm(&held), norm(&dh) / norm(&dt));
            let total = norm(&dh).powi(2);
            for site in 0..lat.num_sites() {
                let s: f64 = dh.site(0, site).iter().map(|x| x * x).sum();
                shares[dist(site)] += s / total / 4.0;
            }
        }
        line += ";  d:";
        for d in 0..=dmax {
            line += &format!(
                "  {d}: {:.2} ({:.2})",
                shares[d],
                sites_at[d] as f64 / lat.num_sites() as f64
            );
        }
        println!("{line}");
    }
    // The coarse correction itself, as `mdwf_eigmg_diag` measures it for A: on 3^4 blocks,
    // the exact Galerkin correction P (P†DP)⁻¹ P† D t, then post-smoothing with GMRES(4) on
    // D; the low-mode content |X† e| (X: all M Ritz vectors). And the coarse residual
    // |P†D PP†t| / |P†D t|: how far P†DP is from having P†t as a near-null vector.
    if l.is_multiple_of(3) {
        struct Galerkin<'a, 'b> {
            d: Bound<'a, Wilson<f64>, V>,
            p: &'b Basis<f64>,
            f: V,
            g: V,
        }
        impl LinearFn<C> for Galerkin<'_, '_> {
            fn mult(&mut self, out: &mut C, inp: &C) {
                self.p.prolong(&mut self.f, inp);
                self.d.mult(&mut self.g, &self.f);
                self.p.restrict(out, &self.g);
            }
        }
        let agg = Aggregation::new(&lat, [3; ND]);
        let p = Basis::from_vectors(&agg, &vectors, &g5)?;
        let mut smoother = Solve::new(
            Fgmres::new(Stop::simple(1e-6, 4)),
            Bound::new(&op, Variant::D),
        );
        let low = |e: &V| -> f64 {
            let mut s = 0.0;
            for j in 0..m {
                let mut dd = vec![C64::ZERO; 1];
                x.select(&[j]).dot(e, &mut dd);
                s += dd[0].norm_sqr();
            }
            s.sqrt()
        };
        let mut smooth = |d: &mut Bound<'_, Wilson<f64>, V>, e: &V| -> V {
            let mut de = e.zeros_like();
            d.mult(&mut de, e);
            let mut se = e.zeros_like();
            smoother.mult(&mut se, &de);
            let mut out = e.zeros_like();
            out.copy_from(e);
            out.axpy(&[C64::new(-1.0, 0.0)], &se);
            out
        };
        println!("  3^4, N = {n}: low-mode content left (t = v_8..v_11):");
        for j in n..n + 4 {
            let t = v.select(&[j]);
            let mut c = p.coarse_field(1);
            p.restrict(&mut c, &t);
            let mut held = t.zeros_like();
            p.prolong(&mut held, &c);
            let floor = (1.0 - norm(&held).powi(2)).max(0.0).sqrt();
            let (mut dt, mut dh) = (t.zeros_like(), t.zeros_like());
            d_op.mult(&mut dt, &t);
            d_op.mult(&mut dh, &held);
            let (mut cdt, mut cdh) = (p.coarse_field(1), p.coarse_field(1));
            p.restrict(&mut cdt, &dt);
            p.restrict(&mut cdh, &dh);
            let coarse_ratio = cdh.norm() / cdt.norm();
            let mut y = cdt.zeros_like();
            let mut gal = Galerkin {
                d: Bound::new(&op, Variant::D),
                p: &p,
                f: t.zeros_like(),
                g: t.zeros_like(),
            };
            let solved = Fgmres::new(Stop {
                tolerance_sqr: 1e-16,
                max_iterations: 100,
                max_restarts: 20,
            })
            .solve(&mut gal, None, &mut y, &cdt);
            let mut e = t.zeros_like();
            p.prolong(&mut e, &y);
            e.aypx(&[C64::new(-1.0, 0.0)], &t);
            let smoothed_only = smooth(&mut d_op, &t);
            let after = smooth(&mut d_op, &e);
            // Rayleigh quotients of D and of the Hermitian γ5 D on t, PP†t and w = PP†t − t.
            let mut w = held.zeros_like();
            w.copy_from(&held);
            w.axpy(&[C64::new(-1.0, 0.0)], &t);
            let mut rq = |v: &V| -> (C64, C64) {
                let mut dv = v.zeros_like();
                d_op.mult(&mut dv, v);
                let mut g = dv.zeros_like();
                op.mult_gamma5(&mut op.work(), &mut g, &dv);
                let (mut a, mut b) = (vec![C64::ZERO; 1], vec![C64::ZERO; 1]);
                v.dot(&dv, &mut a);
                v.dot(&g, &mut b);
                let s = 1.0 / norm(v).powi(2);
                (a[0].scale(s), b[0].scale(s))
            };
            let f = |z: C64| format!("{:+.2e}{:+.2e}i", z.re, z.im);
            let ((dt, gt), (dh, gh), (dw, gw)) = (rq(&t), rq(&held), rq(&w));
            println!(
                "    j = {j}: q_D t {} PP†t {} w {};  q_γ5D t {:+.2e} PP†t {:+.2e} w {:+.2e}",
                f(dt),
                f(dh),
                f(dw),
                gt.re,
                gh.re,
                gw.re
            );
            println!(
                "    j = {j}: floor {floor:.3}; coarse residual ratio {coarse_ratio:.1}; smoothing alone {:.3}; Galerkin {:.3}{} → with smoothing {:.3}",
                low(&smoothed_only),
                low(&e),
                if solved.is_err() { "*" } else { "" },
                low(&after)
            );
        }
    }
    println!("done ({:.0} s)", start.elapsed().as_secs_f64());
    Ok(())
}
