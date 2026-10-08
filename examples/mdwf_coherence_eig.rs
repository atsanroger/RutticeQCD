//! Local coherence of the true low modes of A, the test of Lüscher (JHEP 0707:081): build
//! P from the lowest N singular vectors and see how much of the next ones it holds.
//!
//! ```text
//! cargo run --release --example mdwf_coherence_eig -- CONFIG L M5 MASS [LS] [K] [M] [DEGREE] [ITERS]
//! ```
//!
//! `mdwf_coherence` measures with filtered random vectors on both sides, which mixes the
//! quality of the test vectors with the locality of the low modes. Here the lowest K right
//! singular vectors v_j of A (eigenvectors of H = A†A, σ_j² its eigenvalues) are computed by
//! Chebyshev-filtered subspace iteration with Rayleigh-Ritz on M ≥ K vectors (DEGREE per
//! filter, at most ITERS iterations), and the left ones as u_j = A v_j / σ_j.
//!
//! Printed, for blocks b^4 with P split by Γ5 = γ5 R over aggregates that keep every s:
//! - right: P from v_0..v_{N-1}; |P P† v_j| for j = N..K-1 (mean, min, at j = N);
//! - left: the same with u;
//! - cross: P from v, |P P† u_j| for all j < K: what the Galerkin coarse operator P†AP
//!   needs, since A maps the near-null space of the right vectors onto that of the left;
//! - random: |P P† t| a random t would give, √(2N / (b^4 Ls 12)).
//!
//! Locally coherent low modes give right and left near 1 already for N much smaller than K.

use rayon::prelude::*;
use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::Aggregation;
use ruttice::solver::{largest_magnitude, ChebyshevFilter};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Normal, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_coherence_eig CONFIG L M5 MASS [LS] [K] [M] [DEGREE] [ITERS]";

type V = Field<f64, Spinor>;

/// Vectors handed to the operator at a time: its work fields are of the size of its input.
const CHUNK: usize = 8;

/// The reals of every site, site after site, `[site][vector][reals]`.
fn sites(x: &V) -> &[f64] {
    let s = x.as_slice();
    assert_eq!(s.len(), x.num_sites() * x.site_len());
    assert!(x.num_sites() < 2 || std::ptr::eq(&s[x.site_len()], &x.site(0, 1)[0]));
    s
}

fn sites_mut(x: &mut V) -> &mut [f64] {
    let (n, len) = (x.num_sites(), x.site_len());
    let p = sites(x).as_ptr();
    let s = x.as_mut_slice();
    assert!(std::ptr::eq(p, s.as_ptr()) && s.len() == n * len);
    s
}

/// `G[i * ny + j] = ⟨x_i, y_j⟩`.
fn gram(x: &V, y: &V) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let reals = x.site_len() / nx;
    assert_eq!(reals, y.site_len() / ny);
    let (xs, ys) = (sites(x), sites(y));
    xs.par_chunks(x.site_len())
        .zip(ys.par_chunks(y.site_len()))
        .fold(
            || vec![C64::ZERO; nx * ny],
            |mut g, (xs, ys)| {
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

/// `x_j ← Σ_i x_i c[i * n + j]`, site by site, in place.
fn rotate(x: &mut V, c: &[C64]) {
    let n = x.num_vectors();
    assert_eq!(c.len(), n * n);
    let len = x.site_len();
    let reals = len / n;
    sites_mut(x).par_chunks_mut(len).for_each(|s| {
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
    });
}

/// Eigenvalues (ascending) and eigenvectors (columns, `v[i * n + j]` is component i of
/// vector j) of a Hermitian n × n matrix, by cyclic complex Jacobi.
fn eigh(mut a: Vec<C64>, n: usize) -> (Vec<f64>, Vec<C64>) {
    let mut v = vec![C64::ZERO; n * n];
    for i in 0..n {
        v[i * n + i] = C64::new(1.0, 0.0);
    }
    let total: f64 = a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    for _sweep in 0..100 {
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
                // Make a_pq real: column q times d, row q times conj(d), d = conj(a_pq)/|a_pq|.
                let d = a[p * n + q].conj().scale(1.0 / g);
                for k in 0..n {
                    a[k * n + q] = a[k * n + q] * d;
                    v[k * n + q] = v[k * n + q] * d;
                }
                for k in 0..n {
                    a[q * n + k] = a[q * n + k] * d.conj();
                }
                // Then a real rotation, as in Numerical Recipes.
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

/// Apply `f` to `x` CHUNK vectors at a time, the result in place.
fn in_chunks(x: &mut V, mut f: impl FnMut(&V) -> V) {
    let n = x.num_vectors();
    for chunk in (0..n).collect::<Vec<_>>().chunks(CHUNK) {
        let y = f(&x.select(chunk));
        x.assign(chunk, &y);
    }
}

/// Orthonormalise the vectors of `y` through the eigen-decomposition of their Gram matrix.
fn orthonormalise(y: &mut V) {
    let n = y.num_vectors();
    let (values, u) = eigh(gram(y, y), n);
    let top = values[n - 1];
    let mut c = u;
    for (j, l) in values.iter().enumerate() {
        assert!(
            *l > 1e-13 * top,
            "the subspace lost rank: Gram eigenvalue {l:e} of {top:e}"
        );
        for i in 0..n {
            c[i * n + j] = c[i * n + j].scale(1.0 / l.sqrt());
        }
    }
    rotate(y, &c);
}

fn norms(x: &V) -> Vec<f64> {
    let mut n = vec![0.0; x.num_vectors()];
    x.norm_sqr(&mut n);
    n.iter().map(|v| v.sqrt()).collect()
}

/// For P spanned on each (block, chirality) by the first `n` vectors of `halves[chirality]`
/// restricted there, |P P† t_j|² / |t_j|² for every vector t_j of `t`.
fn captured(agg: &Aggregation, halves: &[V; 2], n: usize, t: &V) -> Vec<f64> {
    let nb = halves[0].num_vectors();
    let nt = t.num_vectors();
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
                    gram_schmidt(&mut w, n, dim);
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

/// Modified Gram-Schmidt on `n` vectors of length `dim`; dependent ones are zeroed.
fn gram_schmidt(w: &mut [C64], n: usize, dim: usize) {
    for i in 0..n {
        let (done, rest) = w.split_at_mut(i * dim);
        let wi = &mut rest[..dim];
        let before: f64 = wi.iter().map(|z| z.norm_sqr()).sum();
        for j in 0..i {
            let wj = &done[j * dim..(j + 1) * dim];
            let mut overlap = C64::ZERO;
            for (a, b) in wj.iter().zip(wi.iter()) {
                overlap += a.conj() * *b;
            }
            for (a, b) in wj.iter().zip(wi.iter_mut()) {
                *b = *b - overlap * *a;
            }
        }
        let after: f64 = wi.iter().map(|z| z.norm_sqr()).sum();
        let scale = if after > 1e-24 * before {
            1.0 / after.sqrt()
        } else {
            0.0
        };
        for z in wi.iter_mut() {
            *z = z.scale(scale);
        }
    }
}

fn copy(x: &V) -> V {
    let mut y = x.zeros_like();
    y.copy_from(x);
    y
}

/// The chiral halves (1 ± Γ5) x / 2.
fn chiral_halves(op: &Mdwf<f64>, x: &V) -> [V; 2] {
    let mut minus = x.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut minus, x);
    let mut plus = copy(x);
    for (v, g) in plus.as_mut_slice().iter_mut().zip(minus.as_mut_slice()) {
        (*v, *g) = (0.5 * (*v + *g), 0.5 * (*v - *g));
    }
    [plus, minus]
}

fn summary(c: &[f64]) -> (f64, f64) {
    let r: Vec<f64> = c.iter().map(|x| x.sqrt()).collect();
    let mean = r.iter().sum::<f64>() / r.len() as f64;
    let min = r.iter().cloned().fold(f64::INFINITY, f64::min);
    (mean, min)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A V = V Λ and V† V = 1 for a dense complex Hermitian matrix, and the right
    /// eigenvalues for one with a six-fold degeneracy.
    #[test]
    fn eigh_diagonalises_a_hermitian_matrix() {
        let n = 7;
        let mut a = vec![C64::ZERO; n * n];
        for i in 0..n {
            for j in i..n {
                let z = if i == j {
                    C64::new(((i * 3 + 1) as f64 * 0.7).sin(), 0.0)
                } else {
                    C64::new(
                        ((i * 5 + j * 7) as f64 * 0.31).cos(),
                        ((i * 2 + j * 11) as f64 * 0.43).sin(),
                    )
                };
                a[i * n + j] = z;
                a[j * n + i] = z.conj();
            }
        }
        let (values, v) = eigh(a.clone(), n);
        assert!(values.windows(2).all(|w| w[0] <= w[1]));
        for j in 0..n {
            for i in 0..n {
                let mut av = C64::ZERO;
                for k in 0..n {
                    av += a[i * n + k] * v[k * n + j];
                }
                let diff = av - v[i * n + j].scale(values[j]);
                assert!(diff.norm_sqr().sqrt() < 1e-12, "A v ≠ λ v at ({i}, {j})");
            }
            for jj in 0..n {
                let mut s = C64::ZERO;
                for i in 0..n {
                    s += v[i * n + j].conj() * v[i * n + jj];
                }
                let want = if j == jj { 1.0 } else { 0.0 };
                assert!((s.re - want).abs() < 1e-12 && s.im.abs() < 1e-12);
            }
        }
        // A multiple of the identity plus a rank-one term: a five-fold degeneracy.
        let mut b = vec![C64::ZERO; n * n];
        for i in 0..n {
            b[i * n + i] = C64::new(2.0, 0.0);
        }
        let u: Vec<C64> = (0..n).map(|i| C64::new(1.0, i as f64 * 0.5)).collect();
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] += u[i] * u[j].conj();
            }
        }
        let (values, _) = eigh(b, n);
        let unorm: f64 = u.iter().map(|z| z.norm_sqr()).sum();
        for l in &values[..n - 1] {
            assert!((l - 2.0).abs() < 1e-12);
        }
        assert!((values[n - 1] - 2.0 - unorm).abs() < 1e-10);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=9).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or(default.to_string());
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = arg(4, "8").parse().map_err(|_| USAGE)?;
    let k: usize = arg(5, "64").parse().map_err(|_| USAGE)?;
    let m: usize = arg(6, "80").parse().map_err(|_| USAGE)?;
    let degree: usize = arg(7, "40").parse().map_err(|_| USAGE)?;
    let iters: usize = arg(8, "40").parse().map_err(|_| USAGE)?;
    if k > m {
        return Err("K must not exceed M".into());
    }

    let lat = Lattice::single_node([l; ND]);
    let u = read_bridge_text(&path, &lat)?;
    let params = MdwfParams {
        ls,
        m5,
        mass,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut op = Mdwf::<f64>::new(&lat, params);
    op.set_gauge(&u)?;
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; K = {k} of M = {m}, filter degree {degree}, at most {iters} iterations"
    );
    let start = Instant::now();
    let elapsed = || start.elapsed().as_secs_f64();

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let like = random(1, 1);
    let hi = 1.1
        * largest_magnitude(
            &mut Normal::new(PvPreconditioned::new(&op, &like), &like),
            &mut random(1, 5),
            50,
        );
    println!("λ_max(A†A) ≈ {:.4} ({:.0} s)", hi / 1.1, elapsed());
    let apply_h = |x: &V| {
        let mut y = x.zeros_like();
        Normal::new(PvPreconditioned::new(&op, x), x).mult(&mut y, x);
        y
    };

    // Subspace iteration: filter, orthonormalise, Rayleigh-Ritz. Resumed from the checkpoint
    // of an earlier run with the same parameters, if there is one.
    let checkpoint = PathBuf::from(format!("run/eig_ckpt_{l}_{m5}_{mass}_{ls}_{k}_{m}.bin"));
    let mut x = random(m, 21);
    let mut lo = hi / 1000.0;
    let mut theta = vec![0.0; m];
    let mut first_iteration = 0;
    if let Some((done, t)) = load(&checkpoint, &mut x)? {
        first_iteration = done;
        lo = t[m - 1];
        theta = t;
        println!(
            "resumed from {} after {done} iterations: σ_0 {:.4e}, σ_{} {:.4e}",
            checkpoint.display(),
            theta[0].sqrt(),
            m - 1,
            theta[m - 1].sqrt()
        );
    }
    for iteration in first_iteration..iters {
        // The filter separates λ below lo from [lo, hi] by T_d(1 + 2 (lo − λ)/(hi − lo)),
        // which grows like exp(2 d √((lo − λ)/hi)): with lo near the low modes a fixed
        // degree stops separating them, so the degree follows √(hi/lo).
        let d = ((1.5 * (hi / lo).sqrt()).ceil() as usize).clamp(degree, 1000);
        in_chunks(&mut x, |c| {
            let mut f =
                ChebyshevFilter::new(Normal::new(PvPreconditioned::new(&op, c), c), d, lo, hi, c);
            let mut y = c.zeros_like();
            f.mult(&mut y, c);
            y
        });
        orthonormalise(&mut x);
        let mut w = copy(&x);
        in_chunks(&mut w, apply_h);
        let (values, c) = eigh(gram(&x, &w), m);
        rotate(&mut x, &c);
        rotate(&mut w, &c);
        let change = (0..k)
            .map(|j| ((values[j] - theta[j]) / values[j]).abs())
            .fold(0.0, f64::max);
        theta = values;
        // |H v_j − θ_j v_j| / θ_j.
        let scale: Vec<C64> = theta.iter().map(|t| C64::new(-t, 0.0)).collect();
        w.axpy(&scale, &x);
        let res: Vec<f64> = norms(&w).iter().zip(&theta).map(|(r, t)| r / t).collect();
        let worst = res[..k].iter().cloned().fold(0.0, f64::max);
        drop(w);
        lo = theta[m - 1];
        save(&checkpoint, iteration + 1, &theta, &x)?;
        println!(
            "iteration {iteration:2}, degree {d}: σ_0 {:.4e}, σ_{} {:.4e}, σ_{} {:.4e}; lowest {k}: worst relative residual {worst:.1e}, largest change of θ {change:.1e} ({:.0} s)",
            theta[0].sqrt(),
            k - 1,
            theta[k - 1].sqrt(),
            m - 1,
            theta[m - 1].sqrt(),
            elapsed()
        );
        // Converged vectors, or a subspace that no longer moves: the test averages over the
        // span of the lowest K, which matters more than each vector by itself.
        let last = worst < 1e-2 || change < 1e-3 || iteration + 1 == iters;
        // The test on the subspace as it is, from the fourth iteration on: numbers that no
        // longer change between iterations are the answer even before the vectors converge.
        if iteration >= 3 || last {
            report(&op, &lat, [l, ls, k], &theta, &x);
            println!("({:.0} s)", elapsed());
        }
        if last {
            break;
        }
    }
    println!("done ({:.0} s)", elapsed());
    Ok(())
}

/// `x` (its first `m` vectors) and θ after `done` iterations, to `path`.
fn save(path: &PathBuf, done: usize, theta: &[f64], x: &V) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    for h in [done as u64, theta.len() as u64, x.as_slice().len() as u64] {
        f.write_all(&h.to_le_bytes())?;
    }
    for v in theta.iter().chain(x.as_slice()) {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(tmp, path)
}

/// The iterations done and θ, with `x` filled, from a checkpoint of the same shape.
/// Iterations done and θ.
type Progress = (usize, Vec<f64>);

fn load(path: &PathBuf, x: &mut V) -> Result<Option<Progress>, Box<dyn Error>> {
    use std::io::Read;
    let Ok(file) = std::fs::File::open(path) else {
        return Ok(None);
    };
    let mut f = std::io::BufReader::new(file);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<[u8; 8]> {
        f.read_exact(&mut word)?;
        Ok(word)
    };
    let done = u64::from_le_bytes(next(&mut f)?) as usize;
    let m = u64::from_le_bytes(next(&mut f)?) as usize;
    let len = u64::from_le_bytes(next(&mut f)?) as usize;
    if m != x.num_vectors() || len != x.as_slice().len() {
        return Err(format!("{} is for another shape", path.display()).into());
    }
    let mut theta = vec![0.0; m];
    for t in theta.iter_mut() {
        *t = f64::from_le_bytes(next(&mut f)?);
    }
    for v in x.as_mut_slice() {
        *v = f64::from_le_bytes(next(&mut f)?);
    }
    Ok(Some((done, theta)))
}

/// Lüscher's test on the lowest `k` vectors of `x` (right singular vectors of A, θ = σ²).
fn report(
    op: &Mdwf<f64>,
    lat: &std::sync::Arc<Lattice>,
    [l, ls, k]: [usize; 3],
    theta: &[f64],
    x: &V,
) {
    let v = x.select(&(0..k).collect::<Vec<_>>());
    println!(
        "σ_j: {}",
        theta[..k]
            .iter()
            .step_by(8)
            .map(|t| format!("{:.3e}", t.sqrt()))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Left vectors u_j = A v_j / σ_j.
    let mut lv = copy(&v);
    in_chunks(&mut lv, |c| {
        let mut y = c.zeros_like();
        PvPreconditioned::new(op, c).mult(&mut y, c);
        y
    });
    let inv: Vec<f64> = norms(&lv).iter().map(|n| 1.0 / n).collect();
    lv.scale(&inv);

    let max_n = [4, 8, 16, 32]
        .into_iter()
        .filter(|n| *n < k)
        .max()
        .unwrap_or(1);
    let first: Vec<usize> = (0..max_n).collect();
    let right = chiral_halves(op, &v.select(&first));
    let left = chiral_halves(op, &lv.select(&first));

    let blocks: Vec<usize> = [2, 3, 4]
        .into_iter()
        .filter(|b| l.is_multiple_of(*b))
        .collect();
    println!("  block   N   right mean/min/at N   left mean/min/at N   cross mean   random");
    for b in blocks {
        let agg = Aggregation::new(lat, [b; ND]);
        for n in [4, 8, 16, 32].into_iter().filter(|n| *n < k) {
            let r = captured(&agg, &right, n, &v);
            let le = captured(&agg, &left, n, &lv);
            let cross = captured(&agg, &right, n, &lv);
            let (rm, rmin) = summary(&r[n..]);
            let (lm, lmin) = summary(&le[n..]);
            let (cm, _) = summary(&cross);
            let random = (2.0 * n as f64 / (b.pow(4) * ls * 12) as f64).sqrt();
            println!(
                "  {b}^4  {n:3}   {rm:.3} {rmin:.3} {:.3}     {lm:.3} {lmin:.3} {:.3}     {cm:.3}       {random:.3}",
                r[n].sqrt(),
                le[n].sqrt()
            );
        }
    }
}
