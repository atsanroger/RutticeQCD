//! The low modes of the even-odd operator Ã = S_X† S_Y (`mdwf_evenodd_a`), on which the
//! multigrid is to be built: Lüscher's local coherence test, the residual of the blockwise
//! projection, and whether the exact Galerkin coarse correction does anything, as
//! `mdwf_coherence_eig`, `mdwf_boundary` and `mdwf_eigmg_diag` measure them for A′.
//!
//! ```text
//! cargo run --release --example mdwf_evenodd_coherence -- CONFIG L M5 MASS [LS] [K] [M] [ITERS]
//! ```
//!
//! The lowest K right singular vectors of Ã (eigenvectors of Ã†Ã) by Chebyshev-filtered
//! subspace iteration with Rayleigh-Ritz on M vectors, the degree following √(hi/lo), four
//! vectors at a time and with a checkpoint after every iteration (`run/eoeig_ckpt_*.bin`,
//! resumed when present). Ã lives on the even sites; for the measurements its vectors are
//! put on the full lattice with zeros on the odd sites, so P is the library's `Basis` over
//! b^4 aggregates (each holds b^4/2 even sites), split by Γ5 = γ5 R, keeping every s.
//! From the fourth iteration on, after each:
//! - Lüscher's test: mean |PP†v_j| over j = N..K−1, P from the lowest N (N = 4, 8, 16);
//! - for N = 8 and t = v_8..v_11: |Ã PP†t| / |Ã t|; the coarse residual |P†Ã PP†t| / |P†Ã t|
//!   (P† here is the restriction to the even sites); and the low-mode content |X† e| (X:
//!   all M Ritz vectors) left by smoothing alone (GMRES(4) on Ã) and by the exact Galerkin
//!   correction P (P†ÃP)⁻¹ P† Ã followed by the same smoothing, against the floor.

use rayon::prelude::*;
use ruttice::fermion::{Dagger, EvenOdd, FermionOp, HopTerms, Mdwf, MdwfParams};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{largest_magnitude, ChebyshevFilter, Fgmres, Solve, Solver, Stop};
use ruttice::{
    extract_parity, insert_parity, random_spinor, Boundary, Coarse, EvenOddIndex, Field, Half,
    Lattice, LinearFn, LinearOp, Normal, Parity, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_evenodd_coherence CONFIG L M5 MASS [LS] [K] [M] [ITERS]";

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;
type C = Field<f64, Coarse>;

/// Vectors handed to Ã at a time.
const CHUNK: usize = 4;

// --- Ã = S_X† S_Y, as in `mdwf_evenodd_a`. ---

fn schur(
    op: &Mdwf<f64>,
    work: &mut <Mdwf<f64> as EvenOdd<H>>::Work,
    odd: (&mut H, &mut H),
    even: &mut H,
    out: &mut H,
    inp: &H,
    dagger: Dagger,
) {
    let (o1, o2) = odd;
    match dagger {
        Dagger::No => {
            op.diag_inv(work, even, inp, Parity::Even, Dagger::No);
            op.hop(work, o1, even, Parity::Odd, Dagger::No);
            op.diag_inv(work, o2, o1, Parity::Odd, Dagger::No);
            op.hop(work, out, o2, Parity::Even, Dagger::No);
        }
        Dagger::Yes => {
            op.hop(work, o1, inp, Parity::Odd, Dagger::Yes);
            op.diag_inv(work, o2, o1, Parity::Odd, Dagger::Yes);
            op.hop(work, even, o2, Parity::Even, Dagger::Yes);
            op.diag_inv(work, out, even, Parity::Even, Dagger::Yes);
        }
    }
    out.aypx(&vec![C64::new(-1.0, 0.0); inp.num_vectors()], inp);
}

struct HalfA<'a> {
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    work: <Mdwf<f64> as EvenOdd<H>>::Work,
    work_pv: <Mdwf<f64> as EvenOdd<H>>::Work,
    o1: H,
    o2: H,
    e1: H,
    e2: H,
}

impl<'a> HalfA<'a> {
    fn new(op: &'a Mdwf<f64>, pv: &'a Mdwf<f64>, like: &H) -> Self {
        let mut odd = like.zeros_like();
        odd.set_parity(Parity::Odd);
        Self {
            op,
            pv,
            work: op.even_odd_work(),
            work_pv: pv.even_odd_work(),
            o1: odd.zeros_like(),
            o2: odd,
            e1: like.zeros_like(),
            e2: like.zeros_like(),
        }
    }
}

fn fit(w: &mut H, like: &H, parity: Parity) {
    if w.num_vectors() != like.num_vectors() {
        *w = like.zeros_like();
        w.set_parity(parity);
    }
}

impl LinearFn<H> for HalfA<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        for w in [&mut self.o1, &mut self.o2] {
            fit(w, inp, Parity::Odd);
        }
        for w in [&mut self.e1, &mut self.e2] {
            fit(w, inp, Parity::Even);
        }
        let mut y = std::mem::replace(&mut self.e2, inp.zeros_like());
        schur(
            self.op,
            &mut self.work,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            &mut y,
            inp,
            Dagger::No,
        );
        schur(
            self.pv,
            &mut self.work_pv,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            out,
            &y,
            Dagger::Yes,
        );
        self.e2 = y;
    }
}

impl LinearOp<H> for HalfA<'_> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        for w in [&mut self.o1, &mut self.o2] {
            fit(w, inp, Parity::Odd);
        }
        for w in [&mut self.e1, &mut self.e2] {
            fit(w, inp, Parity::Even);
        }
        let mut y = std::mem::replace(&mut self.e2, inp.zeros_like());
        schur(
            self.pv,
            &mut self.work_pv,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            &mut y,
            inp,
            Dagger::No,
        );
        schur(
            self.op,
            &mut self.work,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            out,
            &y,
            Dagger::Yes,
        );
        self.e2 = y;
    }
}

// --- Dense helpers on half fields, as in `mdwf_coherence_eig`. ---

fn norms(x: &H) -> Vec<f64> {
    let mut n = vec![0.0; x.num_vectors()];
    x.norm_sqr(&mut n);
    n.iter().map(|v| v.sqrt()).collect()
}

fn norm(x: &H) -> f64 {
    norms(x).iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn site_len(x: &H) -> usize {
    let len = x.as_slice().len();
    let per_vector = 24 * x.ls();
    assert_eq!(len % (per_vector * x.num_vectors()), 0);
    per_vector * x.num_vectors()
}

fn gram(x: &H, y: &H) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let (lx, ly) = (site_len(x), site_len(y));
    let reals = lx / nx;
    x.as_slice()
        .par_chunks(lx)
        .zip(y.as_slice().par_chunks(ly))
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

fn rotate(x: &mut H, c: &[C64]) {
    let n = x.num_vectors();
    let len = site_len(x);
    let reals = len / n;
    x.as_mut_slice().par_chunks_mut(len).for_each(|s| {
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

/// Hermitian eigen-decomposition by cyclic complex Jacobi (tested in `mdwf_coherence_eig`).
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

fn orthonormalise(y: &mut H) {
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

fn in_chunks(x: &mut H, mut f: impl FnMut(&H) -> H) {
    let n = x.num_vectors();
    for chunk in (0..n).collect::<Vec<_>>().chunks(CHUNK) {
        let y = f(&x.select(chunk));
        x.assign(chunk, &y);
    }
}

fn save(path: &PathBuf, done: usize, theta: &[f64], x: &H) -> std::io::Result<()> {
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

/// Iterations done and θ.
type Progress = (usize, Vec<f64>);

fn load(path: &PathBuf, x: &mut H) -> Result<Option<Progress>, Box<dyn Error>> {
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

// --- The measurements, on the full lattice. ---

/// |P P† t_j|² / |t_j|² for every vector of `t` (full fields), P spanned on each
/// (block, chirality) by the first `n` vectors of `halves[chirality]`.
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
                        let s = if after > 1e-24 * before.max(1e-300) {
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
    let mut tn = vec![0.0; nt];
    t.norm_sqr(&mut tn);
    sums.iter().zip(tn).map(|(s, n)| s / n).collect()
}

struct Measure<'a> {
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    lat: &'a Arc<Lattice>,
    index: &'a EvenOddIndex,
    l: usize,
    ls: usize,
    k: usize,
}

impl Measure<'_> {
    fn full(&self, h: &H) -> V {
        let mut f = V::with_shape(self.lat, 24, self.ls, h.num_vectors(), 1);
        insert_parity(&mut f, h, self.index);
        f
    }

    fn half(&self, f: &V) -> H {
        let mut h = H::with_shape(self.lat, 24, self.ls, f.num_vectors(), 1);
        h.set_parity(Parity::Even);
        extract_parity(&mut h, f, self.index, Parity::Even);
        h
    }

    fn report(&self, x: &H) {
        let (k, l) = (self.k, self.l);
        let m = x.num_vectors();
        let v = self.full(&x.select(&(0..k).collect::<Vec<_>>()));
        let first: Vec<usize> = (0..16.min(k)).collect();
        let halves = {
            let x = v.select(&first);
            let mut g = x.zeros_like();
            self.op.mult_gamma5(&mut self.op.work(), &mut g, &x);
            let mut plus = x.zeros_like();
            plus.copy_from(&x);
            for (a, b) in plus.as_mut_slice().iter_mut().zip(g.as_mut_slice()) {
                (*a, *b) = (0.5 * (*a + *b), 0.5 * (*a - *b));
            }
            [plus, g]
        };
        let blocks: Vec<usize> = [2, 3, 4]
            .into_iter()
            .filter(|b| l.is_multiple_of(*b))
            .collect();
        println!("  Lüscher's test: mean |PP†v_j| over j = N..K−1");
        for &b in &blocks {
            let agg = Aggregation::new(self.lat, [b; ND]);
            let mut line = format!("    {b}^4:");
            for n in [4, 8, 16].into_iter().filter(|n| *n < k) {
                let c = captured(&agg, &halves, n, &v);
                let mean = c[n..].iter().map(|x| x.sqrt()).sum::<f64>() / (k - n) as f64;
                line += &format!("  N = {n}: {mean:.3}");
            }
            println!("{line}");
        }

        let n = 8;
        if k < n + 4 {
            return;
        }
        let vectors = v.select(&(0..n).collect::<Vec<_>>());
        let mut g5 = vectors.zeros_like();
        self.op.mult_gamma5(&mut self.op.work(), &mut g5, &vectors);
        let like = x.select(&[0]);
        let mut a = HalfA::new(self.op, self.pv, &like);
        let mut smoother = Solve::new(
            Fgmres::new(Stop::simple(1e-6, 4)),
            HalfA::new(self.op, self.pv, &like),
        );
        let low = |e: &H| -> f64 {
            let mut s = 0.0;
            for j in 0..m {
                let mut d = vec![C64::ZERO; 1];
                x.select(&[j]).dot(e, &mut d);
                s += d[0].norm_sqr();
            }
            s.sqrt()
        };
        for &b in &blocks {
            let agg = Aggregation::new(self.lat, [b; ND]);
            let p = Basis::<f64>::from_vectors(&agg, &vectors, &g5).expect("the basis");
            println!("  N = {n}, {b}^4, t = v_8..v_11:");
            for j in n..n + 4 {
                let th = x.select(&[j]);
                let t = self.full(&th);
                let mut c = p.coarse_field(1);
                p.restrict(&mut c, &t);
                let mut heldf = t.zeros_like();
                p.prolong(&mut heldf, &c);
                let held = self.half(&heldf);
                let floor = (1.0 - norm(&held).powi(2) / norm(&th).powi(2))
                    .max(0.0)
                    .sqrt();
                let (mut at, mut ah) = (th.zeros_like(), th.zeros_like());
                a.mult(&mut at, &th);
                a.mult(&mut ah, &held);
                let (mut cat, mut cah) = (p.coarse_field(1), p.coarse_field(1));
                p.restrict(&mut cat, &self.full(&at));
                p.restrict(&mut cah, &self.full(&ah));
                let cn = |c: &C| {
                    let mut v = vec![0.0; 1];
                    c.norm_sqr(&mut v);
                    v[0].sqrt()
                };
                // The exact Galerkin correction, P†ÃP through the fine lattice.
                struct Galerkin<'a, 'b, 'c> {
                    a: HalfA<'a>,
                    p: &'b Basis<f64>,
                    m: &'c Measure<'a>,
                }
                impl LinearFn<C> for Galerkin<'_, '_, '_> {
                    fn mult(&mut self, out: &mut C, inp: &C) {
                        let mut f = V::with_shape(self.m.lat, 24, self.m.ls, inp.num_vectors(), 1);
                        self.p.prolong(&mut f, inp);
                        let h = self.m.half(&f);
                        let mut ah = h.zeros_like();
                        self.a.mult(&mut ah, &h);
                        self.p.restrict(out, &self.m.full(&ah));
                    }
                }
                let mut gal = Galerkin {
                    a: HalfA::new(self.op, self.pv, &like),
                    p: &p,
                    m: self,
                };
                let mut y = cat.zeros_like();
                let solved = Fgmres::new(Stop {
                    tolerance_sqr: 1e-14,
                    max_iterations: 100,
                    max_restarts: 10,
                })
                .solve(&mut gal, None, &mut y, &cat);
                let mut corr = t.zeros_like();
                p.prolong(&mut corr, &y);
                let mut e = th.zeros_like();
                e.copy_from(&th);
                e.axpy(&[C64::new(-1.0, 0.0)], &self.half(&corr));
                let mut smooth = |e: &H| -> H {
                    let mut ae = e.zeros_like();
                    a.mult(&mut ae, e);
                    let mut se = e.zeros_like();
                    smoother.mult(&mut se, &ae);
                    let mut out = e.zeros_like();
                    out.copy_from(e);
                    out.axpy(&[C64::new(-1.0, 0.0)], &se);
                    out
                };
                let alone = smooth(&th);
                let after = smooth(&e);
                println!(
                    "    j = {j}: |PP†t| {:.3}, |Ã PP†t|/|Ã t| {:.1}, coarse {:.1}; low-mode content: smoothing alone {:.3}, Galerkin {:.3}{} → with smoothing {:.3} (floor {floor:.3})",
                    norm(&held) / norm(&th),
                    norm(&ah) / norm(&at),
                    cn(&cah) / cn(&cat),
                    low(&alone),
                    low(&e),
                    if solved.is_err() { "*" } else { "" },
                    low(&after)
                );
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `noreport`: only the setup (the vectors and their checkpoint), without the coherence
    // measurements, which build bases for several block sizes and dominate the memory.
    let report = !args.iter().any(|s| s == "noreport");
    args.retain(|s| s != "noreport");
    if !(4..=8).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args
        .get(4)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let k: usize = args
        .get(5)
        .map_or(Ok(32), |s| s.parse())
        .map_err(|_| USAGE)?;
    let m: usize = args
        .get(6)
        .map_or(Ok(44), |s| s.parse())
        .map_err(|_| USAGE)?;
    let iters: usize = args
        .get(7)
        .map_or(Ok(12), |s| s.parse())
        .map_err(|_| USAGE)?;

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
    let mut pv = Mdwf::<f64>::new(
        &lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    );
    pv.set_gauge(&u)?;
    let index: &EvenOddIndex = op.even_odd_index().ok_or("no even-odd decomposition")?;
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; Ã on the even sites; K = {k} of M = {m}");
    let start = Instant::now();
    let measure = Measure {
        op: &op,
        pv: &pv,
        lat: &lat,
        index,
        l,
        ls,
        k,
    };
    let random = |n: usize, seed: u64| {
        let mut f = V::with_shape(&lat, 24, ls, n, 1);
        f.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        measure.half(&f)
    };
    let like = random(1, 1);
    let normal = |x: &H| Normal::new(HalfA::new(&op, &pv, x), x);
    let hi = 1.1 * largest_magnitude(&mut normal(&like), &mut random(1, 5), 50);
    println!(
        "λ_max(Ã†Ã) ≈ {:.4} ({:.0} s)",
        hi / 1.1,
        start.elapsed().as_secs_f64()
    );

    let checkpoint = PathBuf::from(format!("run/eoeig_ckpt_{l}_{m5}_{mass}_{ls}_{k}_{m}.bin"));
    let mut x = random(m, 21);
    let mut lo = hi / 1000.0;
    let mut theta = vec![0.0; m];
    let mut first = 0;
    if let Some((done, t)) = load(&checkpoint, &mut x)? {
        first = done;
        lo = t[m - 1];
        theta = t;
        println!(
            "resumed after {done} iterations: σ_0 {:.4e}",
            theta[0].sqrt()
        );
    }
    for iteration in first..iters {
        let d = ((1.5 * (hi / lo).sqrt()).ceil() as usize).clamp(40, 1000);
        in_chunks(&mut x, |c| {
            let mut y = c.zeros_like();
            ChebyshevFilter::new(normal(c), d, lo, hi, c).mult(&mut y, c);
            y
        });
        orthonormalise(&mut x);
        let mut w = x.zeros_like();
        w.copy_from(&x);
        in_chunks(&mut w, |c| {
            let mut y = c.zeros_like();
            normal(c).mult(&mut y, c);
            y
        });
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
        drop(w);
        lo = theta[m - 1];
        save(&checkpoint, iteration + 1, &theta, &x)?;
        println!(
            "iteration {iteration:2}, degree {d}: σ_0 {:.4e}, σ_{} {:.4e}, σ_{} {:.4e}; worst relative residual {worst:.1e}, change {change:.1e} ({:.0} s)",
            theta[0].sqrt(),
            k - 1,
            theta[k - 1].sqrt(),
            m - 1,
            theta[m - 1].sqrt(),
            start.elapsed().as_secs_f64()
        );
        let last = worst < 1e-2 || change < 1e-3 || iteration + 1 == iters;
        if report && (iteration >= 3 || last) {
            measure.report(&x);
            println!("({:.0} s)", start.elapsed().as_secs_f64());
        }
        if last {
            break;
        }
    }
    println!("done ({:.0} s)", start.elapsed().as_secs_f64());
    Ok(())
}
