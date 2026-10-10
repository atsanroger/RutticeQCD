//! BiCGStab on the even-odd operator Ã = S_X† S_Y with a two-level cycle whose coarse
//! correction is the factorized, Γ5-Hermitian one of `mdwf_g5cycle`, built on the full
//! lattice and entered and left on the even sites.
//!
//! ```text
//! cargo run --release --example mdwf_eo_cycle -- CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [STEPS] [RATIO] [COARSE_TOL] [noalone] [cr|minres|pminres] [defl:K:STEPS]
//! ```
//!
//! The coarse solver is conjugate residuals (`cr`), MINRES, or MINRES preconditioned by the
//! block |H| (`pminres`); `defl:K:STEPS` deflates MINRES by the K eigenvectors nearest zero
//! of each Γ5-Hermitian coarse factor, from STEPS Lanczos steps.
//!
//! Ã = C_PV,e†⁻¹ S_PV† S_D C_e⁻¹ (S_D = D_ee − D_eo D_oo⁻¹ D_oe, the Schur complement of D on
//! the even sites, and S_PV that of D_PV), so Ã e = r is S_PV† S_D (C_e⁻¹ e) = C_PV,e† r, and
//! S_D⁻¹ = (D⁻¹)_ee, S_PV†⁻¹ = (D_PV†⁻¹)_ee. With ι the injection of the even sites into the
//! full lattice (zero on the odd ones) and π its adjoint:
//!
//!   e = C_e π P (P†DP)⁻¹ E (P†D_PV†P)⁻¹ P† ι C_PV,e† r,   E = P† ι π P,
//!
//! the two coarse operators being the nearest-neighbour ones of the full lattice, solved by
//! conjugate residuals on Γ5 times each (both are Γ5-Hermitian) to COARSE_TOL (default
//! 1e-8). P is built from full-lattice vectors: for the lowest N/2 Ritz vectors y of Ã from a
//! `mdwf_evenodd_coherence` checkpoint, s = (C_e⁻¹y, −C_o⁻¹ D_oe C_e⁻¹y), whose image D s
//! = (S_D C_e⁻¹y, 0) lives on the even sites, together with D s; split by Γ5 on BLOCK^4
//! aggregates (default 2). The cycle: y = S r; y += e(r − Ãy); y += S(r − Ãy), with S a
//! Chebyshev polynomial in Ã of STEPS steps (default 5) on [hi/RATIO, hi] (RATIO 30,
//! hi = 1.1 |λ_max(Ã)|).
//!
//! Printed for BiCGStab on Ã y = S_X† b′_e (seed 91) to 1e-18: the iterations, the
//! applications of Ã (two Wilson sweeps each, as in `mdwf_evenodd_a`), the coarse
//! applications, |Dx − b|²/|b|² and the time; without a preconditioner, with the
//! polynomial alone and with the cycle.

use ruttice::fermion::{Dagger, EvenOdd, FermionOp, HopTerms, Mdwf, MdwfParams, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, BasisBuilder, CoarseOp};
use ruttice::solver::{largest_magnitude, BiCgStab, Chebyshev, Fgmres, SolveError, Solver, Stop};
use ruttice::{
    extract_parity, insert_parity, random_spinor, Boundary, Coarse, EvenOddIndex, Field, Half,
    Lattice, LinearFn, LinearOp, Parity, Spinor, Vector, C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_eo_cycle CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [STEPS] [RATIO] [COARSE_TOL] [noalone] [cr|minres|pminres] [defl:K:STEPS]";

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;
type C = Field<f64, Coarse>;
/// P: double-precision fields, single-precision storage (half the memory of P).
type Pb = Basis<f64, f32>;
type Count = Rc<Cell<usize>>;

fn norm<W: Vector>(v: &W) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

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
    n: Count,
}

fn fit(w: &mut H, like: &H, parity: Parity) {
    if w.num_vectors() != like.num_vectors() {
        *w = like.zeros_like();
        w.set_parity(parity);
    }
}

impl<'a> HalfA<'a> {
    fn new(op: &'a Mdwf<f64>, pv: &'a Mdwf<f64>, like: &H, n: &Count) -> Self {
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
            n: n.clone(),
        }
    }

    fn fit_all(&mut self, inp: &H) {
        for w in [&mut self.o1, &mut self.o2] {
            fit(w, inp, Parity::Odd);
        }
        for w in [&mut self.e1, &mut self.e2] {
            fit(w, inp, Parity::Even);
        }
    }

    /// `out = S_X† inp`.
    fn x_dagger(&mut self, out: &mut H, inp: &H) {
        self.fit_all(inp);
        schur(
            self.pv,
            &mut self.work_pv,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            out,
            inp,
            Dagger::Yes,
        );
    }
}

impl LinearFn<H> for HalfA<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.fit_all(inp);
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
        self.n.set(self.n.get() + inp.num_vectors());
        self.fit_all(inp);
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

// --- Coarse solves: conjugate residuals on Γ5 A_c. ---

fn coarse_gamma5(v: &mut C) {
    let nv = v.num_vectors();
    let per_vector = v.site_len() / nv;
    for x in v.as_mut_slice().chunks_mut(per_vector) {
        for r in x[per_vector / 2..].iter_mut() {
            *r = -*r;
        }
    }
}

/// Solves A_c x = b (A_c† with `adjoint`) as (Γ5 A_c) x = Γ5 b by conjugate residuals;
/// returns the coarse applications.
fn coarse_solve(a: &mut CoarseOp<f64>, adjoint: bool, x: &mut C, b: &C, tol_sqr: f64) -> usize {
    let mut apply = |out: &mut C, inp: &C| {
        if adjoint {
            a.mult_dag(out, inp);
        } else {
            a.mult(out, inp);
        }
        coarse_gamma5(out);
    };
    let nv = b.num_vectors();
    let mut rhs = b.zeros_like();
    rhs.copy_from(b);
    coarse_gamma5(&mut rhs);
    x.scale(&vec![0.0; nv]);
    let mut r = rhs.zeros_like();
    r.copy_from(&rhs);
    let mut p = r.zeros_like();
    p.copy_from(&r);
    let mut hr = r.zeros_like();
    apply(&mut hr, &r);
    let mut hp = hr.zeros_like();
    hp.copy_from(&hr);
    let mut bb = vec![0.0; nv];
    rhs.norm_sqr(&mut bb);
    let mut rhr = vec![C64::ZERO; nv];
    r.dot(&hr, &mut rhr);
    let (mut hphp, mut rr) = (vec![0.0; nv], vec![0.0; nv]);
    for it in 0..5000 {
        hp.norm_sqr(&mut hphp);
        let alpha: Vec<C64> = (0..nv)
            .map(|j| C64::new(rhr[j].re / hphp[j].max(1e-300), 0.0))
            .collect();
        x.axpy(&alpha, &p);
        let minus: Vec<C64> = alpha.iter().map(|a| C64::new(-a.re, 0.0)).collect();
        r.axpy(&minus, &hp);
        r.norm_sqr(&mut rr);
        if (0..nv).all(|j| rr[j] <= tol_sqr * bb[j]) {
            return it + 2;
        }
        apply(&mut hr, &r);
        let mut rhr_new = vec![C64::ZERO; nv];
        r.dot(&hr, &mut rhr_new);
        let beta: Vec<C64> = (0..nv)
            .map(|j| C64::new(rhr_new[j].re / rhr[j].re, 0.0))
            .collect();
        p.aypx(&beta, &r);
        hp.aypx(&beta, &hr);
        rhr = rhr_new;
    }
    5001
}

/// Eigenvalues and eigenvectors (columns) of a Hermitian n × n matrix by cyclic complex
/// Jacobi (tested in `mdwf_coherence_eig`).
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
    let values = (0..n).map(|i| a[i * n + i].re).collect();
    (values, v)
}

/// The block-diagonal preconditioner |H_xx|⁻¹ for H = Γ5 A_c (or Γ5 A_c†): on every coarse
/// site the inverse absolute value V |Λ|⁻¹ V† of the Hermitian diagonal block. It is
/// Hermitian positive definite, as preconditioned conjugate residuals need, unlike the
/// indefinite diagonal block itself.
struct BlockAbsInv {
    dim: usize,
    blocks: Vec<Vec<C64>>,
    /// The signed inverses H_xx⁻¹ of the same blocks.
    signed: Vec<Vec<C64>>,
    /// An additive second level, making the preconditioner B + Q |Q†HQ|⁻¹ Q†.
    level: Option<Box<dyn AbsAdd>>,
}

/// A second level for the absolute-value preconditioner of H (Vecharynski and Knyazev,
/// SIAM J. Sci. Comput. 35 (2013) A696, use multigrid for |A|⁻¹ of indefinite A): Q
/// aggregates test vectors of H on super-blocks of coarse sites, split by Γ5 as on the
/// levels above; H_cc = Q†HQ is small and dense, and |H_cc|⁻¹ = V |Λ|⁻¹ V† is formed
/// exactly. B + Q |H_cc|⁻¹ Q† is Hermitian positive definite, as MINRES needs.
struct AbsLevel {
    /// The coarse sites of every super-block.
    sites: Vec<Vec<usize>>,
    /// Per super-block, `nq` orthonormal local vectors, column after column.
    q: Vec<Vec<C64>>,
    nq: usize,
    /// |H_cc|⁻¹, dense, row-major.
    inv: Vec<C64>,
    /// (Q†A_cQ)⁻¹ = (Γ5 H_cc)⁻¹, dense, row-major: the signed level for BiCGStab, which
    /// needs neither Hermiticity nor a positive preconditioner.
    signed_inv: Vec<C64>,
}

impl AbsLevel {
    /// Returns the level and a line describing it.
    fn build(
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        vectors: &[C],
        super_block: usize,
    ) -> (Self, String) {
        let start = Instant::now();
        let lat = a.lattice().clone();
        let dim = a.dim();
        let ext = lat.local;
        assert!(ext.iter().all(|e| e % super_block == 0));
        let n_super: Vec<usize> = ext.iter().map(|e| e / super_block).collect();
        let total: usize = n_super.iter().product();
        let mut sites = vec![Vec::new(); total];
        for x in 0..lat.num_sites() {
            let c = lat.coords(x);
            let mut s = 0;
            for mu in (0..ND).rev() {
                s = s * n_super[mu] + c[mu] / super_block;
            }
            sites[s].push(x);
        }
        let read = |v: &C, s: &[usize]| -> Vec<C64> {
            let mut out = Vec::with_capacity(s.len() * dim);
            for &x in s {
                let f = v.site(0, x);
                for i in 0..dim {
                    out.push(C64::new(f[2 * i], f[2 * i + 1]));
                }
            }
            out
        };
        // Local vectors: both chiralities of every test vector, orthonormalised.
        let mut q = Vec::with_capacity(total);
        let mut chir: Vec<Vec<bool>> = Vec::with_capacity(total);
        let mut nq = usize::MAX;
        for s in &sites {
            let mut cols: Vec<Vec<C64>> = Vec::new();
            let mut upper_of: Vec<bool> = Vec::new();
            for v in vectors {
                let local = read(v, s);
                for upper in [false, true] {
                    let mut c: Vec<C64> = local
                        .iter()
                        .enumerate()
                        .map(|(i, z)| {
                            if ((i % dim) >= dim / 2) == upper {
                                *z
                            } else {
                                C64::ZERO
                            }
                        })
                        .collect();
                    for _ in 0..2 {
                        for b in &cols {
                            let d: C64 = b
                                .iter()
                                .zip(&c)
                                .map(|(x, y)| x.conj() * *y)
                                .fold(C64::ZERO, |s, z| s + z);
                            for (ci, bi) in c.iter_mut().zip(b) {
                                *ci = *ci - d * *bi;
                            }
                        }
                    }
                    let n = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                    if n > 1e-8 {
                        cols.push(c.iter().map(|z| z.scale(1.0 / n)).collect());
                        upper_of.push(upper);
                    }
                }
            }
            nq = nq.min(cols.len());
            q.push(cols);
            chir.push(upper_of);
        }
        let q: Vec<Vec<C64>> = q
            .into_iter()
            .map(|cols| {
                let len = cols[0].len();
                let mut m = Vec::with_capacity(len * nq);
                for c in cols.into_iter().take(nq) {
                    m.extend(c);
                }
                m
            })
            .collect();
        // H_cc, one column at a time.
        let n = total * nq;
        let mut h = vec![C64::ZERO; n * n];
        let mut e = vectors[0].zeros_like();
        let mut he = e.zeros_like();
        for (s, sv) in sites.iter().enumerate() {
            let len = sv.len() * dim;
            for j in 0..nq {
                e.as_mut_slice().fill(0.0);
                for (k, &x) in sv.iter().enumerate() {
                    let f = e.site_mut(0, x);
                    for i in 0..dim {
                        let z = q[s][j * len + k * dim + i];
                        f[2 * i] = z.re;
                        f[2 * i + 1] = z.im;
                    }
                }
                h_mult(a, adjoint, &mut he, &e);
                for (t, tv) in sites.iter().enumerate() {
                    let local = read(&he, tv);
                    let lt = local.len();
                    for b in 0..nq {
                        let d: C64 = (0..lt)
                            .map(|i| q[t][b * lt + i].conj() * local[i])
                            .fold(C64::ZERO, |s, z| s + z);
                        h[(t * nq + b) * n + s * nq + j] = d;
                    }
                }
            }
        }
        // The signed level: Q†A_cQ = Γ5 Q†HQ, Γ5 = +1 on the lower chirality.
        let mut a_cc = vec![0.0; 2 * n * n];
        for i in 0..n {
            let (t, b) = (i / nq, i % nq);
            let sign = if chir[t][b] { -1.0 } else { 1.0 };
            for k in 0..n {
                a_cc[2 * (i * n + k)] = sign * h[i * n + k].re;
                a_cc[2 * (i * n + k) + 1] = sign * h[i * n + k].im;
            }
        }
        let mut a_inv = vec![0.0; 2 * n * n];
        let signed_inv: Vec<C64> = match ruttice::numerics::invert_complex(n, &mut a_cc, &mut a_inv)
        {
            Ok(()) => a_inv.chunks(2).map(|c| C64::new(c[0], c[1])).collect(),
            Err(_) => vec![C64::ZERO; n * n],
        };
        let h: Vec<C64> = (0..n * n)
            .map(|ij| (h[ij] + h[(ij % n) * n + ij / n].conj()).scale(0.5))
            .collect();
        let (lam, v) = eigh(h, n);
        let mut inv = vec![C64::ZERO; n * n];
        for i in 0..n {
            for k in 0..n {
                let mut z = C64::ZERO;
                for j in 0..n {
                    z += v[i * n + j].scale(1.0 / lam[j].abs().max(1e-300)) * v[k * n + j].conj();
                }
                inv[i * n + k] = z;
            }
        }
        let mut abs: Vec<f64> = lam.iter().map(|l| l.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let line = format!(
            "{total} super-blocks of {super_block}^4 × {nq} = {n}; |λ(H_cc)| {:.2e} … {:.2e} ({:.0} s, {} applications)",
            abs[0],
            abs[n - 1],
            start.elapsed().as_secs_f64(),
            total * nq
        );
        (
            Self {
                sites,
                q,
                nq,
                inv,
                signed_inv,
            },
            line,
        )
    }
}

/// What the levels below the block |H|⁻¹ add to the absolute-value preconditioner.
trait AbsAdd {
    /// out += (the levels below) inp.
    fn add(&self, out: &mut C, inp: &C);
    /// out += the signed version, for BiCGStab on A_c; false if there is none.
    fn add_signed(&self, _out: &mut C, _inp: &C) -> bool {
        false
    }
}

impl AbsAdd for AbsLevel {
    /// out += Q |H_cc|⁻¹ Q† inp.
    fn add(&self, out: &mut C, inp: &C) {
        self.add_with(&self.inv, out, inp);
    }

    /// out += Q (Q†A_cQ)⁻¹ Q† inp.
    fn add_signed(&self, out: &mut C, inp: &C) -> bool {
        self.add_with(&self.signed_inv, out, inp);
        true
    }
}

impl AbsLevel {
    /// out += Q m Q† inp for a dense m on the level.
    fn add_with(&self, m: &[C64], out: &mut C, inp: &C) {
        let nq = self.nq;
        let n = self.sites.len() * nq;
        let mut y = vec![C64::ZERO; n];
        for (s, sv) in self.sites.iter().enumerate() {
            let mut local = Vec::new();
            for &x in sv {
                let f = inp.site(0, x);
                local.extend(f.chunks(2).map(|c| C64::new(c[0], c[1])));
            }
            let lt = local.len();
            for b in 0..nq {
                y[s * nq + b] = (0..lt)
                    .map(|i| self.q[s][b * lt + i].conj() * local[i])
                    .fold(C64::ZERO, |s, z| s + z);
            }
        }
        let z: Vec<C64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|k| m[i * n + k] * y[k])
                    .fold(C64::ZERO, |s, z| s + z)
            })
            .collect();
        for (s, sv) in self.sites.iter().enumerate() {
            let per = self.q[s].len() / nq / sv.len();
            let lt = per * sv.len();
            for (k, &x) in sv.iter().enumerate() {
                let f = out.site_mut(0, x);
                for i in 0..per {
                    let mut add = C64::ZERO;
                    for b in 0..nq {
                        add += self.q[s][b * lt + k * per + i] * z[s * nq + b];
                    }
                    f[2 * i] += add.re;
                    f[2 * i + 1] += add.im;
                }
            }
        }
    }
}

// --- The multilevel absolute-value preconditioner. ---

/// The pseudo-inverse absolute value V |Λ|⁺ V† of a Hermitian n × n matrix (row-major,
/// symmetrised here): eigenvalues below 1e-10 of the largest in magnitude, which come from
/// degenerate local vectors, are left out.
fn abs_pinv(h: &[C64], n: usize) -> Vec<C64> {
    let h: Vec<C64> = (0..n * n)
        .map(|ij| (h[ij] + h[(ij % n) * n + ij / n].conj()).scale(0.5))
        .collect();
    let (lam, v) = eigh(h, n);
    let top = lam.iter().fold(0.0f64, |m, l| m.max(l.abs()));
    let inv: Vec<f64> = lam
        .iter()
        .map(|l| {
            if l.abs() > 1e-10 * top {
                1.0 / l.abs()
            } else {
                0.0
            }
        })
        .collect();
    let mut m = vec![C64::ZERO; n * n];
    for i in 0..n {
        for k in 0..n {
            let mut z = C64::ZERO;
            for j in 0..n {
                z += v[i * n + j].scale(inv[j]) * v[k * n + j].conj();
            }
            m[i * n + k] = z;
        }
    }
    m
}

fn mat_vec(m: &[C64], n: usize, x: &[C64]) -> Vec<C64> {
    (0..n)
        .map(|i| (0..n).fold(C64::ZERO, |s, k| s + m[i * n + k] * x[k]))
        .collect()
}

/// A coarse-lattice field as a plain vector, site after site, `dim` components each.
fn to_vec(f: &C) -> Vec<C64> {
    f.as_slice()
        .chunks(2)
        .map(|c| C64::new(c[0], c[1]))
        .collect()
}

fn from_vec(f: &mut C, v: &[C64]) {
    for (c, z) in f.as_mut_slice().chunks_mut(2).zip(v) {
        c[0] = z.re;
        c[1] = z.im;
    }
}

/// One level ℓ ≥ 2 of the multilevel absolute-value preconditioner: the sites of the level
/// above are grouped into super-blocks; on each, `nq` local vectors (half of them of
/// either chirality, so that Γ5 stays diagonal: components a < nq/2 are +1).
struct AbsMLevel {
    parent_dim: usize,
    sites: Vec<Vec<usize>>,
    q: Vec<Vec<C64>>,
    nq: usize,
    /// |H_ℓ,xx|⁺ per site, or, at the coarsest level, |H_ℓ|⁺ dense in `blocks[0]`.
    blocks: Vec<Vec<C64>>,
    dense: bool,
}

impl AbsMLevel {
    fn restrict(&self, v: &[C64]) -> Vec<C64> {
        let (nq, pd) = (self.nq, self.parent_dim);
        let mut out = vec![C64::ZERO; self.sites.len() * nq];
        for (s, sv) in self.sites.iter().enumerate() {
            let lt = sv.len() * pd;
            for b in 0..nq {
                let col = &self.q[s][b * lt..(b + 1) * lt];
                let mut z = C64::ZERO;
                for (k, &x) in sv.iter().enumerate() {
                    for i in 0..pd {
                        z += col[k * pd + i].conj() * v[x * pd + i];
                    }
                }
                out[s * nq + b] = z;
            }
        }
        out
    }

    fn prolong_add(&self, y: &[C64], v: &mut [C64]) {
        let (nq, pd) = (self.nq, self.parent_dim);
        for (s, sv) in self.sites.iter().enumerate() {
            let lt = sv.len() * pd;
            for b in 0..nq {
                let c = y[s * nq + b];
                let col = &self.q[s][b * lt..(b + 1) * lt];
                for (k, &x) in sv.iter().enumerate() {
                    for i in 0..pd {
                        v[x * pd + i] += col[k * pd + i] * c;
                    }
                }
            }
        }
    }

    /// This level's preconditioner on its own vector, without the levels below.
    fn own(&self, y: &[C64]) -> Vec<C64> {
        let nq = self.nq;
        if self.dense {
            return mat_vec(&self.blocks[0], y.len(), y);
        }
        let mut out = vec![C64::ZERO; y.len()];
        for (s, b) in self.blocks.iter().enumerate() {
            let z = mat_vec(b, nq, &y[s * nq..(s + 1) * nq]);
            out[s * nq..(s + 1) * nq].copy_from_slice(&z);
        }
        out
    }
}

/// The levels ℓ = 2 … L below the block |H|⁻¹ of the coarse lattice:
/// M_ℓ = B_ℓ + Q_ℓ+1 M_ℓ+1 Q_ℓ+1†, with B_ℓ the block |H_ℓ,xx|⁺ of H_ℓ = Q_ℓ† H_ℓ−1 Q_ℓ and
/// the coarsest M_L = |H_L|⁺ dense. Additive, so Hermitian positive (semi)definite, and
/// applying it needs no application of H at all; H is used only to set it up.
struct AbsMulti {
    levels: Vec<AbsMLevel>,
}

impl AbsMulti {
    /// `vectors` on the coarse lattice (restricted further for every level) and the
    /// super-block size of every level; the last level is dense.
    fn build(
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        vectors: &[C],
        super_blocks: &[usize],
    ) -> (Self, String) {
        Self::build_inner(a, adjoint, vectors, super_blocks, true)
    }

    /// Only the local vectors of every level (for `SignedMulti`), no blocks of |H|.
    fn build_vectors_only(
        a: &mut CoarseOp<f64>,
        vectors: &[C],
        super_blocks: &[usize],
    ) -> (Self, String) {
        Self::build_inner(a, false, vectors, super_blocks, false)
    }

    fn build_inner(
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        vectors: &[C],
        super_blocks: &[usize],
        with_blocks: bool,
    ) -> (Self, String) {
        let start = Instant::now();
        let lat = a.lattice().clone();
        let dim = a.dim();
        let like = vectors[0].zeros_like();
        let mut applications = 0usize;
        // H_1 on plain vectors.
        let mut h1 = |v: &[C64]| -> Vec<C64> {
            let mut f = like.zeros_like();
            from_vec(&mut f, v);
            let mut out = like.zeros_like();
            h_mult(a, adjoint, &mut out, &f);
            to_vec(&out)
        };
        let mut levels: Vec<AbsMLevel> = Vec::new();
        let mut parent_ext = lat.local;
        let mut parent_dim = dim;
        let mut parent_coords: Box<dyn Fn(usize) -> [usize; ND]> = {
            let lat = lat.clone();
            Box::new(move |x| lat.coords(x))
        };
        let mut parent_upper: Box<dyn Fn(usize) -> bool> = Box::new(move |i| i >= dim / 2);
        let mut tests: Vec<Vec<C64>> = vectors.iter().map(to_vec).collect();
        let k = tests.len();
        let mut lines = Vec::new();
        for (depth, &sb) in super_blocks.iter().enumerate() {
            assert!(parent_ext.iter().all(|e| e % sb == 0));
            let ext: [usize; ND] = parent_ext.map(|e| e / sb);
            let total: usize = ext.iter().product();
            let n_parent: usize = parent_ext.iter().product();
            let mut sites = vec![Vec::new(); total];
            for x in 0..n_parent {
                let c = parent_coords(x);
                let mut s = 0;
                for mu in (0..ND).rev() {
                    s = s * ext[mu] + c[mu] / sb;
                }
                sites[s].push(x);
            }
            // Local vectors: the lower chirality of every test vector, orthonormalised,
            // then the upper; a dependent one is left as zero.
            let nq = 2 * k;
            let q: Vec<Vec<C64>> = sites
                .iter()
                .map(|sv| {
                    let lt = sv.len() * parent_dim;
                    let mut cols: Vec<Vec<C64>> = Vec::with_capacity(nq);
                    for upper in [false, true] {
                        let first = cols.len();
                        for t in &tests {
                            let mut c: Vec<C64> = (0..lt)
                                .map(|j| {
                                    let (kk, i) = (j / parent_dim, j % parent_dim);
                                    if parent_upper(i) == upper {
                                        t[sv[kk] * parent_dim + i]
                                    } else {
                                        C64::ZERO
                                    }
                                })
                                .collect();
                            for _ in 0..2 {
                                for b in &cols[first..] {
                                    let d = b
                                        .iter()
                                        .zip(&c)
                                        .fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y);
                                    for (ci, bi) in c.iter_mut().zip(b) {
                                        *ci = *ci - d * *bi;
                                    }
                                }
                            }
                            let n = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                            let scale = if n > 1e-8 { 1.0 / n } else { 0.0 };
                            cols.push(c.iter().map(|z| z.scale(scale)).collect());
                        }
                    }
                    cols.concat()
                })
                .collect();
            let mut level = AbsMLevel {
                parent_dim,
                sites,
                q,
                nq,
                blocks: Vec::new(),
                dense: depth + 1 == super_blocks.len(),
            };
            // H_ℓ v = Q_ℓ† H_ℓ−1 Q_ℓ v, down through the levels above.
            let h_level = |v: &[C64],
                           h1: &mut dyn FnMut(&[C64]) -> Vec<C64>,
                           levels: &[AbsMLevel],
                           this: &AbsMLevel|
             -> Vec<C64> {
                let mut chain: Vec<Vec<C64>> = vec![v.to_vec()];
                for l in std::iter::once(this).chain(levels.iter().rev()) {
                    let parent_len: usize =
                        l.sites.iter().map(|s| s.len()).sum::<usize>() * l.parent_dim;
                    let mut p = vec![C64::ZERO; parent_len];
                    l.prolong_add(chain.last().unwrap(), &mut p);
                    chain.push(p);
                }
                let mut w = h1(chain.last().unwrap());
                for l in levels.iter().chain(std::iter::once(this)) {
                    w = l.restrict(&w);
                }
                w
            };
            let n = total * nq;
            if !with_blocks {
                // Only the local vectors are wanted.
            } else if level.dense {
                let mut hd = vec![C64::ZERO; n * n];
                for j in 0..n {
                    let mut e = vec![C64::ZERO; n];
                    e[j] = C64::new(1.0, 0.0);
                    let col = h_level(&e, &mut h1, &levels, &level);
                    applications += 1;
                    for i in 0..n {
                        hd[i * n + j] = col[i];
                    }
                }
                level.blocks = vec![abs_pinv(&hd, n)];
            } else {
                // Diagonal blocks by probing: sites whose coordinates agree modulo c_μ
                // (2 for an even extent, the extent itself for an odd one) are never
                // neighbours.
                let colours: [usize; ND] = ext.map(|e| if e % 2 == 0 { 2 } else { e });
                let n_colours: usize = colours.iter().product();
                let coords = |s: usize| -> [usize; ND] {
                    let mut c = [0; ND];
                    let mut r = s;
                    for mu in 0..ND {
                        c[mu] = r % ext[mu];
                        r /= ext[mu];
                    }
                    c
                };
                let colour_of = |s: usize| -> usize {
                    let c = coords(s);
                    let mut col = 0;
                    for mu in (0..ND).rev() {
                        col = col * colours[mu] + c[mu] % colours[mu];
                    }
                    col
                };
                let mut diag = vec![vec![C64::ZERO; nq * nq]; total];
                for colour in 0..n_colours {
                    for j in 0..nq {
                        let mut e = vec![C64::ZERO; n];
                        for s in 0..total {
                            if colour_of(s) == colour {
                                e[s * nq + j] = C64::new(1.0, 0.0);
                            }
                        }
                        let col = h_level(&e, &mut h1, &levels, &level);
                        applications += 1;
                        for s in 0..total {
                            if colour_of(s) == colour {
                                for i in 0..nq {
                                    diag[s][i * nq + j] = col[s * nq + i];
                                }
                            }
                        }
                    }
                }
                level.blocks = diag.iter().map(|b| abs_pinv(b, nq)).collect();
            }
            lines.push(format!(
                "{}^4 sites × {nq}{}",
                ext[0],
                if level.dense { " (dense)" } else { "" }
            ));
            // The test vectors for the next level.
            tests = tests.iter().map(|t| level.restrict(t)).collect();
            parent_ext = ext;
            parent_dim = nq;
            parent_coords = Box::new(move |s| {
                let mut c = [0; ND];
                let mut r = s;
                for mu in 0..ND {
                    c[mu] = r % ext[mu];
                    r /= ext[mu];
                }
                c
            });
            parent_upper = Box::new(move |i| i >= nq / 2);
            levels.push(level);
        }
        let line = format!(
            "levels {} ({applications} coarse applications, {:.0} s)",
            lines.join(" → "),
            start.elapsed().as_secs_f64()
        );
        (Self { levels }, line)
    }

    /// M_ℓ y for level ℓ = 2 + i.
    fn apply_level(&self, i: usize, y: &[C64]) -> Vec<C64> {
        let l = &self.levels[i];
        let mut out = l.own(y);
        if i + 1 < self.levels.len() {
            let below = &self.levels[i + 1];
            let z = self.apply_level(i + 1, &below.restrict(y));
            below.prolong_add(&z, &mut out);
        }
        out
    }
}

impl AbsAdd for AbsMulti {
    fn add(&self, out: &mut C, inp: &C) {
        let first = &self.levels[0];
        let z = self.apply_level(0, &first.restrict(&to_vec(inp)));
        let mut v = to_vec(out);
        first.prolong_add(&z, &mut v);
        from_vec(out, &v);
    }
}

// --- The signed multilevel preconditioner for BiCGStab on A_c. ---

fn site_coords(s: usize, ext: &[usize; ND]) -> [usize; ND] {
    let mut c = [0; ND];
    let mut r = s;
    for mu in 0..ND {
        c[mu] = r % ext[mu];
        r /= ext[mu];
    }
    c
}

fn site_index(c: &[usize; ND], ext: &[usize; ND]) -> usize {
    let mut s = 0;
    for mu in (0..ND).rev() {
        s = s * ext[mu] + c[mu];
    }
    s
}

/// A nearest-neighbour block operator on a level's lattice: for every site the diagonal
/// block and one block per direction (0..4 forward, 4..8 backward), nq × nq row-major.
/// With an extent of 2 both neighbours are one site and its coupling is in the forward
/// block; with an extent of 1 everything is in the diagonal block.
struct LevelOp {
    nq: usize,
    nbr: Vec<[usize; 2 * ND]>,
    blocks: Vec<Vec<Vec<C64>>>,
    diag_inv: Vec<Vec<C64>>,
}

impl LevelOp {
    fn mult(&self, v: &[C64]) -> Vec<C64> {
        let nq = self.nq;
        let mut out = vec![C64::ZERO; v.len()];
        for (t, b) in self.blocks.iter().enumerate() {
            let mut acc = mat_vec(&b[0], nq, &v[t * nq..(t + 1) * nq]);
            for d in 0..2 * ND {
                if b[1 + d].is_empty() {
                    continue;
                }
                let s = self.nbr[t][d];
                let z = mat_vec(&b[1 + d], nq, &v[s * nq..(s + 1) * nq]);
                for (a, z) in acc.iter_mut().zip(z) {
                    *a += z;
                }
            }
            out[t * nq..(t + 1) * nq].copy_from_slice(&acc);
        }
        out
    }

    /// The level operator of `level`'s lattice from the operator above, by probing: sources
    /// c_μ apart (the smallest divisor of the extent from 3 up, or the extent) never share
    /// a neighbour.
    fn probe(
        level: &AbsMLevel,
        ext: [usize; ND],
        parent_len: usize,
        apply: &mut dyn FnMut(&[C64]) -> Vec<C64>,
    ) -> (Self, usize) {
        let nq = level.nq;
        let total: usize = ext.iter().product();
        let nbr: Vec<[usize; 2 * ND]> = (0..total)
            .map(|s| {
                let c = site_coords(s, &ext);
                let mut n = [0; 2 * ND];
                for mu in 0..ND {
                    let mut f = c;
                    f[mu] = (c[mu] + 1) % ext[mu];
                    let mut b = c;
                    b[mu] = (c[mu] + ext[mu] - 1) % ext[mu];
                    n[mu] = site_index(&f, &ext);
                    n[ND + mu] = site_index(&b, &ext);
                }
                n
            })
            .collect();
        let used = |d: usize| -> bool {
            let e = ext[d % ND];
            !(e == 1 || (e == 2 && d >= ND))
        };
        let mut blocks: Vec<Vec<Vec<C64>>> = (0..total)
            .map(|_| {
                (0..=2 * ND)
                    .map(|k| {
                        if k == 0 || used(k - 1) {
                            vec![C64::ZERO; nq * nq]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect()
            })
            .collect();
        let colours: [usize; ND] = ext.map(|e| (3..=e).find(|k| e % k == 0).unwrap_or(e).max(1));
        let n_colours: usize = colours.iter().product();
        let mut applications = 0;
        for colour in 0..n_colours {
            let mut rest = colour;
            let want = colours.map(|k| {
                let w = rest % k;
                rest /= k;
                w
            });
            let sources: Vec<usize> = (0..total)
                .filter(|&s| {
                    let c = site_coords(s, &ext);
                    (0..ND).all(|mu| c[mu] % colours[mu] == want[mu])
                })
                .collect();
            for j in 0..nq {
                let mut e = vec![C64::ZERO; total * nq];
                for &s in &sources {
                    e[s * nq + j] = C64::new(1.0, 0.0);
                }
                let mut p = vec![C64::ZERO; parent_len];
                level.prolong_add(&e, &mut p);
                let r = level.restrict(&apply(&p));
                applications += 1;
                for &s in &sources {
                    for i in 0..nq {
                        blocks[s][0][i * nq + j] = r[s * nq + i];
                    }
                    // A(t, s) with s = t + dir: stored at t = s − dir in direction dir.
                    for d in 0..2 * ND {
                        if !used(d) {
                            continue;
                        }
                        let opposite = if d < ND { d + ND } else { d - ND };
                        let t = nbr[s][opposite];
                        for i in 0..nq {
                            blocks[t][1 + d][i * nq + j] = r[t * nq + i];
                        }
                    }
                }
            }
        }
        let diag_inv = blocks
            .iter()
            .map(|b| {
                let mut a: Vec<f64> = b[0].iter().flat_map(|z| [z.re, z.im]).collect();
                let mut inv = vec![0.0; 2 * nq * nq];
                match ruttice::numerics::invert_complex(nq, &mut a, &mut inv) {
                    Ok(()) => inv.chunks(2).map(|c| C64::new(c[0], c[1])).collect(),
                    Err(_) => vec![C64::ZERO; nq * nq],
                }
            })
            .collect();
        (
            Self {
                nq,
                nbr,
                blocks,
                diag_inv,
            },
            applications,
        )
    }
}

/// The levels ℓ = 2 … L below the coarse lattice as a V-cycle for A_c, without Hermiticity:
/// M_ℓ r = z₁ + D_ℓ⁻¹ (r − A_ℓ z₁), z₁ = Q_ℓ+1 M_ℓ+1 Q_ℓ+1† r, A_ℓ = Q_ℓ† A_ℓ−1 Q_ℓ kept as a
/// sparse block operator, D_ℓ its block diagonal, and the coarsest M_L = A_L⁻¹ dense. The
/// local vectors and test vectors are those of `AbsMulti`.
struct SignedMulti {
    levels: Vec<AbsMLevel>,
    ops: Vec<LevelOp>,
    coarsest_inv: Vec<C64>,
}

impl SignedMulti {
    fn build(
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        vectors: &[C],
        super_blocks: &[usize],
    ) -> (Self, String) {
        let start = Instant::now();
        // The local vectors of every level, as for the absolute-value version.
        let (multi, _) = AbsMulti::build_vectors_only(a, vectors, super_blocks);
        let lat = a.lattice().clone();
        let like = vectors[0].zeros_like();
        let mut ops: Vec<LevelOp> = Vec::new();
        let mut applications = 0;
        let mut ext = lat.local;
        let mut parent_len = like.as_slice().len() / 2;
        for (l, level) in multi.levels.iter().enumerate() {
            ext = ext.map(|e| e / super_blocks[l]);
            let (op, n) = if l == 0 {
                let mut apply = |v: &[C64]| -> Vec<C64> {
                    let mut f = like.zeros_like();
                    from_vec(&mut f, v);
                    let mut out = like.zeros_like();
                    if adjoint {
                        a.mult_dag(&mut out, &f);
                    } else {
                        a.mult(&mut out, &f);
                    }
                    to_vec(&out)
                };
                LevelOp::probe(level, ext, parent_len, &mut apply)
            } else {
                let above = &ops[l - 1];
                let mut apply = |v: &[C64]| above.mult(v);
                LevelOp::probe(level, ext, parent_len, &mut apply)
            };
            applications += if l == 0 { n } else { 0 };
            parent_len = level.sites.len() * level.nq;
            ops.push(op);
        }
        // The coarsest level dense, from its sparse operator.
        let last = ops.last().expect("at least one level");
        let n = parent_len;
        let mut dense = vec![0.0; 2 * n * n];
        for j in 0..n {
            let mut e = vec![C64::ZERO; n];
            e[j] = C64::new(1.0, 0.0);
            let col = last.mult(&e);
            for i in 0..n {
                dense[2 * (i * n + j)] = col[i].re;
                dense[2 * (i * n + j) + 1] = col[i].im;
            }
        }
        let mut inv = vec![0.0; 2 * n * n];
        let coarsest_inv = match ruttice::numerics::invert_complex(n, &mut dense, &mut inv) {
            Ok(()) => inv.chunks(2).map(|c| C64::new(c[0], c[1])).collect(),
            Err(_) => vec![C64::ZERO; n * n],
        };
        let sizes: Vec<String> = multi
            .levels
            .iter()
            .map(|l| format!("{} × {}", l.sites.len(), l.nq))
            .collect();
        let line = format!(
            "signed levels {} (coarsest dense {n}; {applications} coarse applications, {:.0} s)",
            sizes.join(" → "),
            start.elapsed().as_secs_f64()
        );
        (
            Self {
                levels: multi.levels,
                ops,
                coarsest_inv,
            },
            line,
        )
    }

    fn vcycle(&self, l: usize, r: &[C64]) -> Vec<C64> {
        if l + 1 == self.levels.len() {
            return mat_vec(&self.coarsest_inv, r.len(), r);
        }
        let below = &self.levels[l + 1];
        let zc = self.vcycle(l + 1, &below.restrict(r));
        let mut z = vec![C64::ZERO; r.len()];
        below.prolong_add(&zc, &mut z);
        let az = self.ops[l].mult(&z);
        let nq = self.ops[l].nq;
        for (t, dinv) in self.ops[l].diag_inv.iter().enumerate() {
            let r2: Vec<C64> = (0..nq).map(|i| r[t * nq + i] - az[t * nq + i]).collect();
            let c = mat_vec(dinv, nq, &r2);
            for i in 0..nq {
                z[t * nq + i] += c[i];
            }
        }
        z
    }
}

impl AbsAdd for SignedMulti {
    /// Not an absolute-value preconditioner: nothing for MINRES.
    fn add(&self, _out: &mut C, _inp: &C) {}

    fn add_signed(&self, out: &mut C, inp: &C) -> bool {
        let first = &self.levels[0];
        let z = self.vcycle(0, &first.restrict(&to_vec(inp)));
        let mut v = to_vec(out);
        first.prolong_add(&z, &mut v);
        from_vec(out, &v);
        true
    }
}

impl BlockAbsInv {
    /// The diagonal blocks by probing: the coarse operator connects nearest neighbours
    /// only, so with the sites coloured by parity, unit vectors on one colour return the
    /// diagonal blocks of that colour. 2 × dim coarse applications in all.
    fn build(a: &mut CoarseOp<f64>, adjoint: bool, like: &C) -> Self {
        let lat = a.lattice().clone();
        let dim = a.dim();
        let ns = lat.num_sites();
        let colour: Vec<usize> = (0..ns)
            .map(|x| lat.coords(x).iter().sum::<usize>() % 2)
            .collect();
        let mut blocks = vec![vec![C64::ZERO; dim * dim]; ns];
        let mut signed = Vec::with_capacity(ns);
        for c in 0..2 {
            let mut e = like.zeros_with(dim);
            for x in 0..ns {
                if colour[x] != c {
                    continue;
                }
                let s = e.site_mut(0, x);
                for k in 0..dim {
                    s[k * 2 * dim + 2 * k] = 1.0;
                }
            }
            let mut out = e.zeros_like();
            if adjoint {
                a.mult_dag(&mut out, &e);
            } else {
                a.mult(&mut out, &e);
            }
            coarse_gamma5(&mut out);
            for x in 0..ns {
                if colour[x] != c {
                    continue;
                }
                let s = out.site(0, x);
                for k in 0..dim {
                    for i in 0..dim {
                        blocks[x][i * dim + k] =
                            C64::new(s[k * 2 * dim + 2 * i], s[k * 2 * dim + 2 * i + 1]);
                    }
                }
            }
        }
        for b in blocks.iter_mut() {
            // Symmetrise against rounding, then V |Λ|⁻¹ V†.
            let h: Vec<C64> = (0..dim * dim)
                .map(|ik| {
                    let (i, k) = (ik / dim, ik % dim);
                    (b[i * dim + k] + b[k * dim + i].conj()).scale(0.5)
                })
                .collect();
            let (lam, v) = eigh(h, dim);
            let compose = |f: &dyn Fn(f64) -> f64| {
                let mut m = vec![C64::ZERO; dim * dim];
                for i in 0..dim {
                    for k in 0..dim {
                        let mut z = C64::ZERO;
                        for j in 0..dim {
                            z += v[i * dim + j].scale(f(lam[j])) * v[k * dim + j].conj();
                        }
                        m[i * dim + k] = z;
                    }
                }
                m
            };
            signed.push(compose(&|l| {
                1.0 / if l.abs() > 1e-300 { l } else { 1e-300 }
            }));
            *b = compose(&|l| 1.0 / l.abs().max(1e-300));
        }
        Self {
            dim,
            blocks,
            signed,
            level: None,
        }
    }

    /// out = A_xx⁻¹ inp (A_xx†⁻¹ for an adjoint build), block by block: H_xx⁻¹ Γ5 inp.
    fn apply_block_solve(&self, out: &mut C, inp: &C) {
        let mut g = inp.zeros_like();
        g.copy_from(inp);
        coarse_gamma5(&mut g);
        apply_blocks(&self.signed, self.dim, out, &g);
    }

    /// Test vectors for the second level: `passes` steps of inverse iteration x ← H⁻¹ x
    /// from random x, by this preconditioner's MINRES to `tol_sqr`. Returns them, the
    /// coarse applications and the median of |Hx|/|x|.
    fn test_vectors(
        &self,
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        like: &C,
        k: usize,
        tol_sqr: f64,
        passes: usize,
    ) -> (Vec<C>, usize, f64) {
        let mut seed: u64 = 0x5851_f42d_4c95_7f2d;
        let mut applications = 0;
        let mut quotients = Vec::new();
        let vectors = (0..k)
            .map(|_| {
                let mut x = like.zeros_with(1);
                for v in x.as_mut_slice() {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *v = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                }
                for _ in 0..passes {
                    // A_c y = Γ5 x is H y = x.
                    let mut r = x.zeros_like();
                    r.copy_from(&x);
                    coarse_gamma5(&mut r);
                    applications +=
                        coarse_solve_minres(a, adjoint, Some(self), None, &mut x, &r, tol_sqr);
                    x.scale(&[1.0 / norm(&x)]);
                }
                let mut hx = x.zeros_like();
                h_mult(a, adjoint, &mut hx, &x);
                quotients.push(norm(&hx) / norm(&x));
                x
            })
            .collect();
        quotients.sort_by(f64::total_cmp);
        (vectors, applications, quotients[k / 2])
    }

    fn apply(&self, out: &mut C, inp: &C) {
        self.apply_block(out, inp);
        if let Some(level) = &self.level {
            assert_eq!(inp.num_vectors(), 1);
            level.add(out, inp);
        }
    }

    fn apply_block(&self, out: &mut C, inp: &C) {
        apply_blocks(&self.blocks, self.dim, out, inp);
    }
}

/// out = M_x inp on every coarse site x, M_x dense dim × dim row-major.
fn apply_blocks(blocks: &[Vec<C64>], dim: usize, out: &mut C, inp: &C) {
    let nv = inp.num_vectors();
    for (x, m) in blocks.iter().enumerate() {
        let src: Vec<f64> = inp.site(0, x).to_vec();
        let dst = out.site_mut(0, x);
        for v in 0..nv {
            for i in 0..dim {
                let mut z = C64::ZERO;
                for k in 0..dim {
                    let r = v * 2 * dim + 2 * k;
                    z += m[i * dim + k] * C64::new(src[r], src[r + 1]);
                }
                dst[v * 2 * dim + 2 * i] = z.re;
                dst[v * 2 * dim + 2 * i + 1] = z.im;
            }
        }
    }
}

/// E = P† ι π P, block-diagonal on the coarse lattice (ιπ keeps the even sites), and its
/// inverse, by probing every column on all sites at once.
struct EvenGram {
    dim: usize,
    e: Vec<Vec<C64>>,
    inv: Vec<Vec<C64>>,
}

impl EvenGram {
    fn build(p: &Pb, lift: &Lift) -> (Self, String) {
        let dim = p.coarse_dim();
        let ns = p.coarse_field(1).as_slice().len() / (2 * dim);
        let mut e = vec![vec![C64::ZERO; dim * dim]; ns];
        let mut unit = p.coarse_field(1);
        let mut f = p.fine_field(1);
        let mut out = p.coarse_field(1);
        for k in 0..dim {
            unit.as_mut_slice().fill(0.0);
            for x in 0..ns {
                unit.site_mut(0, x)[2 * k] = 1.0;
            }
            p.prolong(&mut f, &unit);
            p.restrict(&mut out, &lift.inject(&lift.even(&f)));
            for (x, ex) in e.iter_mut().enumerate() {
                let s = out.site(0, x);
                for i in 0..dim {
                    ex[i * dim + k] = C64::new(s[2 * i], s[2 * i + 1]);
                }
            }
        }
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        let inv = e
            .iter()
            .map(|b| {
                let h: Vec<C64> = (0..dim * dim)
                    .map(|ik| {
                        let (i, k) = (ik / dim, ik % dim);
                        (b[i * dim + k] + b[k * dim + i].conj()).scale(0.5)
                    })
                    .collect();
                let (lam, v) = eigh(h, dim);
                for l in &lam {
                    lo = lo.min(*l);
                    hi = hi.max(*l);
                }
                let mut m = vec![C64::ZERO; dim * dim];
                for i in 0..dim {
                    for k in 0..dim {
                        let mut z = C64::ZERO;
                        for j in 0..dim {
                            z += v[i * dim + j].scale(1.0 / lam[j].max(1e-12))
                                * v[k * dim + j].conj();
                        }
                        m[i * dim + k] = z;
                    }
                }
                m
            })
            .collect();
        (
            Self { dim, e, inv },
            format!("E = P†ιπP: eigenvalues {lo:.2e} … {hi:.2e}"),
        )
    }
}

/// The factorized coarse correction as one operator: c = (P†DP)⁻¹ E (P†D_PV†P)⁻¹ g is
/// K c = g with K = (P†D_PV†P) E⁻¹ (P†DP), two nearest-neighbour coarse operators and a
/// block-diagonal one. K keeps the conditioning of the product, which the separate factor
/// solves lose; it is neither Hermitian nor Γ5-Hermitian, so BiCGStab, with the block
/// inverses (A_d,xx)⁻¹ E (A_pv,xx†)⁻¹ as right preconditioner.
struct CoarseK<'a> {
    cd: &'a mut CoarseOp<f64>,
    cpv: &'a mut CoarseOp<f64>,
    gram: &'a EvenGram,
    n: Count,
}

impl LinearFn<C> for CoarseK<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.n.set(self.n.get() + 2 * inp.num_vectors());
        let mut t = inp.zeros_like();
        self.cd.mult(&mut t, inp);
        let mut u = inp.zeros_like();
        apply_blocks(&self.gram.inv, self.gram.dim, &mut u, &t);
        self.cpv.mult_dag(out, &u);
    }
}

struct CoarseKPrec<'a> {
    md: &'a BlockAbsInv,
    mpv: &'a BlockAbsInv,
    gram: &'a EvenGram,
}

impl LinearFn<C> for CoarseKPrec<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        let mut t = inp.zeros_like();
        self.mpv.apply_block_solve(&mut t, inp);
        let mut u = inp.zeros_like();
        apply_blocks(&self.gram.e, self.gram.dim, &mut u, &t);
        self.md.apply_block_solve(out, &u);
    }
}

/// Eigenvalues and eigenvectors (columns of the row-major n × n `z`) of the real symmetric
/// tridiagonal matrix with diagonal `d` and off-diagonal `e` (e[i] couples i and i + 1;
/// e[n − 1] is ignored), by the implicit QL method (`tqli` of Numerical Recipes).
fn tridiagonal_eigh(mut d: Vec<f64>, mut e: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = d.len();
    e.resize(n, 0.0);
    e[n - 1] = 0.0;
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(iter < 100, "tridiagonal QL did not converge");
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut deflated = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    deflated = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..n {
                    let f = z[k * n + i + 1];
                    z[k * n + i + 1] = s * z[k * n + i] + c * f;
                    z[k * n + i] = c * z[k * n + i] - s * f;
                }
            }
            if deflated {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    (d, z)
}

/// out = H inp, H = Γ5 A_c (Γ5 A_c† with `adjoint`).
fn h_mult(a: &mut CoarseOp<f64>, adjoint: bool, out: &mut C, inp: &C) {
    if adjoint {
        a.mult_dag(out, inp);
    } else {
        a.mult(out, inp);
    }
    coarse_gamma5(out);
}

/// Modified Gram–Schmidt, twice.
fn orthonormalise(u: &mut [C]) {
    let mut d = [C64::ZERO];
    for i in 0..u.len() {
        let (done, rest) = u.split_at_mut(i);
        let ui = &mut rest[0];
        for _ in 0..2 {
            for q in done.iter() {
                q.dot(ui, &mut d);
                ui.axpy(&[-d[0]], q);
            }
        }
        let n = norm(ui);
        ui.scale(&[1.0 / n]);
    }
}

/// The eigenvectors of H = Γ5 A_c (Γ5 A_c† with `adjoint`) nearest zero, from Lanczos with
/// full reorthogonalisation, refined by Rayleigh–Ritz on their span.
struct Deflation {
    u: Vec<C>,
    /// H u_i.
    w: Vec<C>,
    theta: Vec<f64>,
}

impl Deflation {
    /// Returns the deflation and a line describing it.
    fn build(
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        like: &C,
        steps: usize,
        k: usize,
    ) -> (Self, String) {
        let mut apply = |out: &mut C, inp: &C| {
            if adjoint {
                a.mult_dag(out, inp);
            } else {
                a.mult(out, inp);
            }
            coarse_gamma5(out);
        };
        let start = Instant::now();
        let mut q0 = like.zeros_with(1);
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        for v in q0.as_mut_slice() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        }
        q0.scale(&[1.0 / norm(&q0)]);
        let mut q = vec![q0];
        let (mut alpha, mut beta) = (Vec::new(), Vec::new());
        let mut w = like.zeros_with(1);
        for j in 0..steps {
            apply(&mut w, &q[j]);
            let mut d = [C64::ZERO];
            q[j].dot(&w, &mut d);
            alpha.push(d[0].re);
            // Full reorthogonalisation, twice.
            for _ in 0..2 {
                for qi in q.iter() {
                    qi.dot(&w, &mut d);
                    w.axpy(&[-d[0]], qi);
                }
            }
            let b = norm(&w);
            if j + 1 == steps || b < 1e-12 {
                break;
            }
            beta.push(b);
            let mut next = w.zeros_like();
            next.copy_from(&w);
            next.scale(&[1.0 / b]);
            q.push(next);
        }
        let m = alpha.len();
        let (ritz, z) = tridiagonal_eigh(alpha, beta);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&i, &j| ritz[i].abs().total_cmp(&ritz[j].abs()));
        let k = k.min(m);
        let u: Vec<C> = order[..k]
            .iter()
            .map(|&i| {
                let mut v = like.zeros_with(1);
                for (j, qj) in q.iter().enumerate() {
                    v.axpy(&[C64::new(z[j * m + i], 0.0)], qj);
                }
                v
            })
            .collect();
        drop(q);
        drop(apply);
        let d = Self::rayleigh_ritz(a, adjoint, u);
        let line = format!(
            "{m} Lanczos steps, {} ({:.0} s)",
            d.describe(),
            start.elapsed().as_secs_f64()
        );
        (d, line)
    }

    /// Rayleigh–Ritz on the span of the orthonormal u: T = U†HU = V Θ V†, U ← UV.
    fn rayleigh_ritz(a: &mut CoarseOp<f64>, adjoint: bool, u: Vec<C>) -> Self {
        let k = u.len();
        let hu: Vec<C> = u
            .iter()
            .map(|v| {
                let mut o = v.zeros_like();
                h_mult(a, adjoint, &mut o, v);
                o
            })
            .collect();
        let mut t = vec![C64::ZERO; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut d = [C64::ZERO];
                u[i].dot(&hu[j], &mut d);
                t[i * k + j] = d[0];
            }
        }
        let t: Vec<C64> = (0..k * k)
            .map(|ij| (t[ij] + t[(ij % k) * k + ij / k].conj()).scale(0.5))
            .collect();
        let (theta, v) = eigh(t, k);
        let rotate = |x: &[C]| -> Vec<C> {
            (0..k)
                .map(|j| {
                    let mut o = x[0].zeros_like();
                    for i in 0..k {
                        o.axpy(&[v[i * k + j]], &x[i]);
                    }
                    o
                })
                .collect()
        };
        Self {
            u: rotate(&u),
            w: rotate(&hu),
            theta,
        }
    }

    /// The range of |θ| and the relative residuals |Hu − θu|/|θ|.
    fn describe(&self) -> String {
        let k = self.u.len();
        let mut residual: Vec<f64> = (0..k)
            .map(|i| {
                let mut r = self.w[i].zeros_like();
                r.copy_from(&self.w[i]);
                r.axpy(&[C64::new(-self.theta[i], 0.0)], &self.u[i]);
                norm(&r) / self.theta[i].abs()
            })
            .collect();
        residual.sort_by(f64::total_cmp);
        let mut sorted = self.theta.clone();
        sorted.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
        format!(
            "{k} Ritz pairs: |θ| {:.2e} … {:.2e}, |Hu − θu|/|θ| median {:.1e}, worst {:.1e}",
            sorted[0].abs(),
            sorted[k - 1].abs(),
            residual[k / 2],
            residual[k - 1]
        )
    }

    /// The eigenvectors nearest zero by subspace iteration with H⁻¹ (block-|H| MINRES to
    /// 1e-5) and Rayleigh–Ritz, from random vectors. Returns the deflation and a line.
    fn build_inverse(
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        m: &BlockAbsInv,
        like: &C,
        k: usize,
        iterations: usize,
    ) -> (Self, String) {
        let start = Instant::now();
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut u: Vec<C> = (0..k)
            .map(|_| {
                let mut v = like.zeros_with(1);
                for x in v.as_mut_slice() {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *x = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                }
                v
            })
            .collect();
        let mut applications = 0;
        let mut d = None;
        for it in 0..iterations {
            // u_i ← H⁻¹ u_i: A_c x = Γ5 u_i.
            for ui in u.iter_mut() {
                let mut g = ui.zeros_like();
                g.copy_from(ui);
                coarse_gamma5(&mut g);
                applications += coarse_solve_minres(a, adjoint, Some(m), None, ui, &g, 1e-10);
            }
            orthonormalise(&mut u);
            let r = Self::rayleigh_ritz(a, adjoint, u);
            applications += k;
            println!(
                "    inverse iteration {it}: {} ({applications} applications, {:.0} s)",
                r.describe(),
                start.elapsed().as_secs_f64()
            );
            u =
                r.u.iter()
                    .map(|v| {
                        let mut c = v.zeros_like();
                        c.copy_from(v);
                        c
                    })
                    .collect();
            d = Some(r);
        }
        let d = d.expect("at least one iteration");
        let line = format!(
            "{iterations} inverse iterations, {} ({applications} coarse applications, {:.0} s)",
            d.describe(),
            start.elapsed().as_secs_f64()
        );
        (d, line)
    }

    /// v ← (1 − U U†) v.
    fn project(&self, v: &mut C) {
        let mut d = [C64::ZERO];
        for ui in &self.u {
            ui.dot(v, &mut d);
            v.axpy(&[-d[0]], ui);
        }
    }

    /// For the solution y of the projected system, x = y + Σ_i u_i (u_i†h − w_i†y)/θ_i
    /// solves H x = h with U†(h − Hx) = 0.
    fn correct(&self, x: &mut C, h: &C) {
        let (mut uh, mut wy) = ([C64::ZERO], [C64::ZERO]);
        let coefficients: Vec<C64> = (0..self.u.len())
            .map(|i| {
                self.u[i].dot(h, &mut uh);
                self.w[i].dot(x, &mut wy);
                (uh[0] - wy[0]).scale(1.0 / self.theta[i])
            })
            .collect();
        for (c, ui) in coefficients.iter().zip(&self.u) {
            x.axpy(&[*c], ui);
        }
    }
}

/// Solves A_c x = b (A_c† with `adjoint`) by MINRES on H = Γ5 A_c, optionally preconditioned
/// by the Hermitian positive definite M and deflated by eigenvectors of H nearest zero;
/// returns the coarse applications. Conjugate residuals with the preconditioner stalled on
/// P†DP (5000 steps, residual 1.6e-2), as CR can on an indefinite matrix: it divides by
/// ⟨z, H z⟩. With deflation, MINRES solves (1 − UU†) H (1 − UU†) y = (1 − UU†) Γ5 b and
/// `Deflation::correct` adds the part in the span of U.
fn coarse_solve_minres(
    a: &mut CoarseOp<f64>,
    adjoint: bool,
    m: Option<&BlockAbsInv>,
    deflation: Option<&Deflation>,
    x: &mut C,
    b: &C,
    tol_sqr: f64,
) -> usize {
    let mut h = b.zeros_like();
    h.copy_from(b);
    coarse_gamma5(&mut h);
    let mut apply = |out: &mut C, inp: &C| {
        let projected;
        let inp = match deflation {
            Some(d) => {
                let mut t = inp.zeros_like();
                t.copy_from(inp);
                d.project(&mut t);
                projected = t;
                &projected
            }
            None => inp,
        };
        if adjoint {
            a.mult_dag(out, inp);
        } else {
            a.mult(out, inp);
        }
        coarse_gamma5(out);
        if let Some(d) = deflation {
            d.project(out);
        }
    };
    let precondition = |out: &mut C, inp: &C| match m {
        Some(m) => m.apply(out, inp),
        None => out.copy_from(inp),
    };
    match deflation {
        None => minres(&mut apply, &precondition, x, &h, tol_sqr),
        Some(d) => {
            let mut rhs = h.zeros_like();
            rhs.copy_from(&h);
            d.project(&mut rhs);
            let n = minres(&mut apply, &precondition, x, &rhs, tol_sqr);
            d.correct(x, &h);
            n
        }
    }
}

/// Preconditioned MINRES for the Hermitian (indefinite) `apply`: x with apply(x) = rhs.
fn minres(
    apply: &mut dyn FnMut(&mut C, &C),
    precondition: &dyn Fn(&mut C, &C),
    x: &mut C,
    rhs: &C,
    tol_sqr: f64,
) -> usize {
    // Elman, Silvester and Wathen, algorithm 2.4: Lanczos with the M-inner product and
    // Givens rotations, so it cannot break down on an indefinite matrix as conjugate
    // residuals can. M must be Hermitian positive definite. Stops when the M⁻¹-norm of the
    // residual, which MINRES minimises, has fallen by √tol_sqr.
    let b = rhs;
    let nv = b.num_vectors();
    let re_dot = |u: &C, v: &C| -> Vec<f64> {
        let mut d = vec![C64::ZERO; nv];
        u.dot(v, &mut d);
        d.iter().map(|z| z.re).collect()
    };
    let lin = |out: &mut C, terms: &[(&[f64], &C)]| {
        out.scale(&vec![0.0; nv]);
        for (coef, v) in terms {
            let c: Vec<C64> = coef.iter().map(|c| C64::new(*c, 0.0)).collect();
            out.axpy(&c, v);
        }
    };
    x.scale(&vec![0.0; nv]);
    let mut v = b.zeros_like();
    v.copy_from(b);
    let mut v_prev = v.zeros_like();
    let mut z = v.zeros_like();
    precondition(&mut z, &v);
    let mut gamma: Vec<f64> = re_dot(&z, &v).iter().map(|g| g.max(0.0).sqrt()).collect();
    let mut gamma_prev = vec![1.0; nv];
    let eta0 = gamma.clone();
    let mut eta = gamma.clone();
    let (mut c, mut c_prev, mut s, mut s_prev) =
        (vec![1.0; nv], vec![1.0; nv], vec![0.0; nv], vec![0.0; nv]);
    let (mut w, mut w_prev) = (v.zeros_like(), v.zeros_like());
    let (mut az, mut v_new, mut z_new, mut w_new) = (
        v.zeros_like(),
        v.zeros_like(),
        v.zeros_like(),
        v.zeros_like(),
    );
    for it in 0..5000 {
        z.scale(
            &gamma
                .iter()
                .map(|g| 1.0 / g.max(1e-300))
                .collect::<Vec<_>>(),
        );
        apply(&mut az, &z);
        let delta = re_dot(&az, &z);
        let k1: Vec<f64> = (0..nv).map(|j| -delta[j] / gamma[j]).collect();
        let k2: Vec<f64> = (0..nv).map(|j| -gamma[j] / gamma_prev[j]).collect();
        let one = vec![1.0; nv];
        lin(&mut v_new, &[(&one, &az), (&k1, &v), (&k2, &v_prev)]);
        precondition(&mut z_new, &v_new);
        let gamma_new: Vec<f64> = re_dot(&z_new, &v_new)
            .iter()
            .map(|g| g.max(0.0).sqrt())
            .collect();
        let (mut a2, mut a3, mut a1) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; nv]);
        let (mut c_new, mut s_new) = (vec![0.0; nv], vec![0.0; nv]);
        for j in 0..nv {
            let a0 = c[j] * delta[j] - c_prev[j] * s[j] * gamma[j];
            a1[j] = (a0 * a0 + gamma_new[j] * gamma_new[j]).sqrt().max(1e-300);
            a2[j] = s[j] * delta[j] + c_prev[j] * c[j] * gamma[j];
            a3[j] = s_prev[j] * gamma[j];
            c_new[j] = a0 / a1[j];
            s_new[j] = gamma_new[j] / a1[j];
        }
        let inv_a1: Vec<f64> = a1.iter().map(|v| 1.0 / v).collect();
        let m3: Vec<f64> = (0..nv).map(|j| -a3[j] * inv_a1[j]).collect();
        let m2: Vec<f64> = (0..nv).map(|j| -a2[j] * inv_a1[j]).collect();
        lin(&mut w_new, &[(&inv_a1, &z), (&m3, &w_prev), (&m2, &w)]);
        let step: Vec<f64> = (0..nv).map(|j| c_new[j] * eta[j]).collect();
        let step_c: Vec<C64> = step.iter().map(|v| C64::new(*v, 0.0)).collect();
        x.axpy(&step_c, &w_new);
        for j in 0..nv {
            eta[j] *= -s_new[j];
        }
        if (0..nv).all(|j| eta[j] * eta[j] <= tol_sqr * eta0[j] * eta0[j]) {
            return it + 1;
        }
        std::mem::swap(&mut v_prev, &mut v);
        std::mem::swap(&mut v, &mut v_new);
        std::mem::swap(&mut z, &mut z_new);
        std::mem::swap(&mut w_prev, &mut w);
        std::mem::swap(&mut w, &mut w_new);
        gamma_prev = gamma;
        gamma = gamma_new;
        c_prev = c;
        c = c_new;
        s_prev = s;
        s = s_new;
    }
    5000
}

/// Conjugate residuals, or MINRES with optional block-|H| preconditioners and deflations of
/// (P†DP, P†D_PV†P).
#[derive(Clone, Copy)]
enum CoarseSolver<'a> {
    Cr,
    Minres(
        Option<(&'a BlockAbsInv, &'a BlockAbsInv)>,
        Option<(&'a Deflation, &'a Deflation)>,
    ),
    /// The two factors as one operator K (see `CoarseK`), by BiCGStab.
    K(&'a EvenGram, &'a BlockAbsInv, &'a BlockAbsInv),
    /// BiCGStab on A_c itself, right-preconditioned by the exact block inverses A_c,xx⁻¹:
    /// no Hermiticity needed (for Möbius, c ≠ 0, Γ5 A_c is not Hermitian). `.1`: BiCGStab
    /// for the D factor only, block-|H| MINRES for the PV factor.
    BiCg((&'a BlockAbsInv, &'a BlockAbsInv), bool),
}

/// A coarse operator shared by the Krylov solver and its preconditioner.
type SharedOp<'a> = Rc<std::cell::RefCell<&'a mut CoarseOp<f64>>>;

fn shared_apply(a: &SharedOp, adjoint: bool, out: &mut C, inp: &C) {
    let mut a = a.borrow_mut();
    if adjoint {
        a.mult_dag(out, inp);
    } else {
        a.mult(out, inp);
    }
}

/// A_c (A_c† with `adjoint`) as a linear map, counting its applications.
struct CoarseLin<'a> {
    a: SharedOp<'a>,
    adjoint: bool,
    n: Count,
}

impl LinearFn<C> for CoarseLin<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.n.set(self.n.get() + inp.num_vectors());
        shared_apply(&self.a, self.adjoint, out, inp);
    }
}

/// The exact block inverses A_c,xx⁻¹ as a linear map. With a signed second level it is
/// multiplicative: z₁ = Q (Q†A_cQ)⁻¹ Q† r, then z = z₁ + B⁻¹ (r − A_c z₁) (one more
/// application of A_c, counted); an additive combination made BiCGStab worse (9301
/// coarse applications against 3612 without the level).
struct BlockSolve<'a> {
    m: &'a BlockAbsInv,
    a: SharedOp<'a>,
    adjoint: bool,
    n: Count,
}

impl LinearFn<C> for BlockSolve<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        let Some(level) = &self.m.level else {
            self.m.apply_block_solve(out, inp);
            return;
        };
        let mut z1 = inp.zeros_like();
        if !level.add_signed(&mut z1, inp) {
            self.m.apply_block_solve(out, inp);
            return;
        }
        let mut r2 = inp.zeros_like();
        shared_apply(&self.a, self.adjoint, &mut r2, &z1);
        self.n.set(self.n.get() + inp.num_vectors());
        r2.aypx(&[C64::new(-1.0, 0.0)], inp);
        self.m.apply_block_solve(out, &r2);
        out.axpy(&[C64::new(1.0, 0.0)], &z1);
    }
}

impl CoarseSolver<'_> {
    /// Solves the coarse D factor (`pv` false) or the coarse D_PV† factor (`pv` true);
    /// `adjoint`: the operator holds P†D_PV P and is applied as its adjoint.
    fn solve(
        self,
        a: &mut CoarseOp<f64>,
        adjoint: bool,
        pv: bool,
        x: &mut C,
        b: &C,
        tol: f64,
    ) -> usize {
        match self {
            CoarseSolver::K(..) => unreachable!("K solves both factors at once"),
            CoarseSolver::Cr => coarse_solve(a, adjoint, x, b, tol),
            CoarseSolver::Minres(m, d) => coarse_solve_minres(
                a,
                adjoint,
                m.map(|(md, mp)| if pv { mp } else { md }),
                d.map(|(dd, dp)| if pv { dp } else { dd }),
                x,
                b,
                tol,
            ),
            CoarseSolver::BiCg((md, mp), d_only) => {
                let m = if pv { mp } else { md };
                if pv && d_only {
                    return coarse_solve_minres(a, adjoint, Some(m), None, x, b, tol);
                }
                let n = Count::default();
                let shared: SharedOp = Rc::new(std::cell::RefCell::new(a));
                let mut lin = CoarseLin {
                    a: shared.clone(),
                    adjoint,
                    n: n.clone(),
                };
                let mut prec = BlockSolve {
                    m,
                    a: shared,
                    adjoint,
                    n: n.clone(),
                };
                x.scale(&vec![0.0; b.num_vectors()]);
                let _ = BiCgStab::new(Stop {
                    tolerance_sqr: tol,
                    max_iterations: 5000,
                    max_restarts: 0,
                })
                .solve(&mut lin, Some(&mut prec), x, b);
                n.get()
            }
        }
    }
}

/// What moves between the even sites and the full lattice.
struct Lift<'a> {
    lat: &'a Arc<Lattice>,
    index: &'a EvenOddIndex,
    ls: usize,
}

impl Lift<'_> {
    fn half(&self, parity: Parity, nv: usize) -> H {
        let mut h = H::with_shape(self.lat, 24, self.ls, nv, 1);
        h.set_parity(parity);
        h
    }
    /// ι: the even sites into a full field, zero on the odd ones.
    fn inject(&self, h: &H) -> V {
        let mut f = V::with_shape(self.lat, 24, self.ls, h.num_vectors(), 1);
        insert_parity(&mut f, h, self.index);
        f
    }
    /// π: the even sites of a full field.
    fn even(&self, f: &V) -> H {
        let mut h = self.half(Parity::Even, f.num_vectors());
        extract_parity(&mut h, f, self.index, Parity::Even);
        h
    }
}

/// The Galerkin coarse operator of the even-site product F = S_PV† S_D = C_PV,e† Ã C_e,
/// G = (πP)† F (πP), applied matrix-free (one Ã per application, counted by its HalfA):
/// the product Galerkin of Boyle's PV†M-MG, here on the even sites, to compare with the
/// factorized correction on the same P.
struct ProductGalerkin<'a> {
    a: HalfA<'a>,
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    p: &'a Pb,
    lift: &'a Lift<'a>,
}

impl LinearFn<C> for ProductGalerkin<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        let nv = inp.num_vectors();
        let mut f = V::with_shape(self.lift.lat, 24, self.lift.ls, nv, 1);
        self.p.prolong(&mut f, inp);
        let u = self.lift.even(&f);
        let mut t = u.zeros_like();
        self.op.diag(
            &mut self.op.even_odd_work(),
            &mut t,
            &u,
            Parity::Even,
            Dagger::No,
        );
        let mut s = u.zeros_like();
        self.a.mult(&mut s, &t);
        self.pv.diag(
            &mut self.pv.even_odd_work(),
            &mut t,
            &s,
            Parity::Even,
            Dagger::Yes,
        );
        self.p.restrict(out, &self.lift.inject(&t));
    }
}

/// D_PV† on the fine lattice, for probing P†D_PV†P.
struct PvDagger<'a> {
    pv: &'a Mdwf<f64>,
}

impl LinearFn<V> for PvDagger<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.pv.mult(&mut self.pv.work(), out, inp, Variant::Dagger);
    }
}

/// The cycle.
struct Cycle<'a> {
    /// Instead of the factorized coarse correction: the product Galerkin one, solved by
    /// GMRES(40) to `coarse_tol`.
    product: Option<ProductGalerkin<'a>>,
    a: HalfA<'a>,
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    smoother: Chebyshev<HalfA<'a>, H>,
    coarse: Option<(&'a Pb, &'a mut CoarseOp<f64>, &'a mut CoarseOp<f64>)>,
    lift: &'a Lift<'a>,
    coarse_tol: f64,
    /// The coarse solver: conjugate residuals, MINRES, or MINRES preconditioned by the block
    /// |H| of (P†DP, P†D_PV†P).
    solver: CoarseSolver<'a>,
    /// Smooth before the coarse correction as well as after it.
    pre: bool,
    /// Smooth after it (off only to look at the coarse correction alone).
    post: bool,
    /// The PV coarse operator holds P†D_PV P and is applied as its adjoint (true), or holds
    /// P†D_PV†P itself (false, `exactpv`).
    pv_adj: bool,
    nc: Count,
}

impl LinearFn<H> for Cycle<'_> {
    fn mult(&mut self, y: &mut H, b: &H) {
        let nv = b.num_vectors();
        let minus = vec![C64::new(-1.0, 0.0); nv];
        let one = vec![C64::new(1.0, 0.0); nv];
        let mut r = b.zeros_like();
        let mut e = b.zeros_like();
        if self.pre {
            self.smoother.mult(y, b);
        } else {
            y.scale(&vec![0.0; nv]);
        }
        if let Some((p, cd, cpv)) = self.coarse.as_mut() {
            if self.pre {
                self.a.mult(&mut r, y);
                r.aypx(&minus, b);
            } else {
                r.copy_from(b);
            }
            // g = P† ι C_PV,e† r.
            let mut z = r.zeros_like();
            self.pv.diag(
                &mut self.pv.even_odd_work(),
                &mut z,
                &r,
                Parity::Even,
                Dagger::Yes,
            );
            let mut g = p.coarse_field(nv);
            p.restrict(&mut g, &self.lift.inject(&z));
            let mut f = V::with_shape(self.lift.lat, 24, self.lift.ls, nv, 1);
            let mut c = g.zeros_like();
            match self.product.as_mut() {
                Some(pg) => {
                    // c = ((πP)† F (πP))⁻¹ g; the Ã inside are counted by pg.a.
                    let _ = Fgmres::new(Stop {
                        tolerance_sqr: self.coarse_tol,
                        max_iterations: 40,
                        max_restarts: 200,
                    })
                    .solve(pg, None, &mut c, &g);
                }
                None if matches!(self.solver, CoarseSolver::K(..)) => {
                    let CoarseSolver::K(gram, md, mpv) = self.solver else {
                        unreachable!()
                    };
                    // K c = g in one BiCGStab, K = (P†D_PV†P) E⁻¹ (P†DP).
                    let mut k = CoarseK {
                        cd,
                        cpv,
                        gram,
                        n: self.nc.clone(),
                    };
                    let mut prec = CoarseKPrec { md, mpv, gram };
                    let _ = BiCgStab::new(Stop {
                        tolerance_sqr: self.coarse_tol,
                        max_iterations: 5000,
                        max_restarts: 0,
                    })
                    .solve(&mut k, Some(&mut prec), &mut c, &g);
                }
                None => {
                    // w = (P†D_PV†P)⁻¹ g; w ← E w; c = (P†DP)⁻¹ w.
                    let mut w = g.zeros_like();
                    let mut n =
                        self.solver
                            .solve(cpv, self.pv_adj, true, &mut w, &g, self.coarse_tol);
                    p.prolong(&mut f, &w);
                    p.restrict(&mut w, &self.lift.inject(&self.lift.even(&f)));
                    n += self
                        .solver
                        .solve(cd, false, false, &mut c, &w, self.coarse_tol);
                    self.nc.set(self.nc.get() + n);
                }
            }
            // e = C_e π P c.
            p.prolong(&mut f, &c);
            let u = self.lift.even(&f);
            self.op.diag(
                &mut self.op.even_odd_work(),
                &mut e,
                &u,
                Parity::Even,
                Dagger::No,
            );
            y.axpy(&one, &e);
        }
        if self.post {
            self.a.mult(&mut r, y);
            r.aypx(&minus, b);
            self.smoother.mult(&mut e, &r);
            y.axpy(&one, &e);
        }
    }
}

/// The prolongation, the restricted basis vectors and the odd-part check.
type Built = (Pb, Vec<C>, f64);

/// P from s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y) and D s for the vectors y on the even sites,
/// split by Γ5 on the aggregates; also P† s and P† D s (test vectors for the levels below
/// the coarse lattice, free) and |odd part of D s|/|D s| (zero: a check).
///
/// With `pv_lift` the second half is t̂ = (πDs, −(D_PV†)_oo⁻¹ (D_PV†)_oe πDs) instead of D s:
/// D_PV† t̂ = ι S_PV† πDs lives on the even sites, so both coarse factor solves are exact
/// on the basis vectors (the PV factor's Galerkin inverse returns t̂ from D_PV† t̂, π t̂ =
/// πDs, and the D factor's returns s from D s); with D s itself the PV step is not. The
/// check returned is then |odd part of D_PV† t̂| / |D_PV† t̂|.
fn basis_from(
    op: &Mdwf<f64>,
    pv: &Mdwf<f64>,
    pv_lift: bool,
    lift: &Lift,
    index: &EvenOddIndex,
    agg: &Arc<Aggregation>,
    y: &H,
) -> Result<Built, Box<dyn Error>> {
    // One y at a time, so that at most a few full-lattice vectors are alive besides P:
    // s_j and its partner (D s_j, or t̂_j), with the odd-part check of D s or D_PV† t̂.
    let half_n = y.num_vectors();
    let pair = |j: usize| -> (V, V, f64) {
        let mut work = op.even_odd_work();
        let yj = y.select(&[j]);
        let mut xe = yj.zeros_like();
        op.diag_inv(&mut work, &mut xe, &yj, Parity::Even, Dagger::No);
        let mut t = lift.half(Parity::Odd, 1);
        op.hop(&mut work, &mut t, &xe, Parity::Odd, Dagger::No);
        let mut xo = lift.half(Parity::Odd, 1);
        op.diag_inv(&mut work, &mut xo, &t, Parity::Odd, Dagger::No);
        xo.scale(&[-1.0]);
        let mut s = lift.inject(&xe);
        insert_parity(&mut s, &xo, index);
        let mut ds = s.zeros_like();
        op.mult(&mut op.work(), &mut ds, &s, Variant::D);
        let odd = |f: &V| {
            let mut o = lift.half(Parity::Odd, 1);
            extract_parity(&mut o, f, index, Parity::Odd);
            norm(&o) / norm(f)
        };
        if !pv_lift {
            let check = odd(&ds);
            return (s, ds, check);
        }
        let mut wpv = pv.even_odd_work();
        let a = lift.even(&ds);
        let mut to = lift.half(Parity::Odd, 1);
        pv.hop(&mut wpv, &mut to, &a, Parity::Odd, Dagger::Yes);
        let mut tod = lift.half(Parity::Odd, 1);
        pv.diag_inv(&mut wpv, &mut tod, &to, Parity::Odd, Dagger::Yes);
        tod.scale(&[-1.0]);
        let mut t_hat = lift.inject(&a);
        insert_parity(&mut t_hat, &tod, index);
        let mut dt = t_hat.zeros_like();
        pv.mult(&mut pv.work(), &mut dt, &t_hat, Variant::Dagger);
        let check = odd(&dt);
        (s, t_hat, check)
    };
    let mut builder = BasisBuilder::<f64, f32>::new(agg, 2 * half_n);
    let mut odd_part: f64 = 0.0;
    for j in 0..half_n {
        let (s, partner, check) = pair(j);
        odd_part = odd_part.max(check);
        for (i, v) in [(j, &s), (half_n + j, &partner)] {
            let mut g5 = v.zeros_like();
            op.mult_gamma5(&mut op.work(), &mut g5, v);
            builder.add(i, v, &g5);
        }
    }
    let p = builder.finish()?;
    // The vectors on the coarse lattice (recomputed: cheap), and whether they are in the
    // range of P: |P P† v − v| / |v| for the first of either half.
    let mut restricted: Vec<C> = (0..2 * half_n).map(|_| p.coarse_field(1)).collect();
    let mut in_range = [0.0; 2];
    for j in 0..half_n {
        let (s, partner, _) = pair(j);
        for (k, (i, v)) in [(j, &s), (half_n + j, &partner)].into_iter().enumerate() {
            p.restrict(&mut restricted[i], v);
            if j == 0 {
                let mut back = v.zeros_like();
                p.prolong(&mut back, &restricted[i]);
                back.axpy(&[C64::new(-1.0, 0.0)], v);
                in_range[k] = norm(&back) / norm(v);
            }
        }
    }
    println!(
        "  check: |PP†v − v|/|v| = {:.1e} (first s), {:.1e} (first of the second half)",
        in_range[0], in_range[1]
    );
    Ok((p, restricted, odd_part))
}

/// Gram–Schmidt (twice) over the vectors of a multi-vector field.
fn orthonormalise_h(y: &mut H) {
    let nv = y.num_vectors();
    let mut vs: Vec<H> = (0..nv).map(|j| y.select(&[j])).collect();
    let mut d = [C64::ZERO];
    for i in 0..nv {
        let (done, rest) = vs.split_at_mut(i);
        let v = &mut rest[0];
        for _ in 0..2 {
            for q in done.iter() {
                q.dot(v, &mut d);
                v.axpy(&[-d[0]], q);
            }
        }
        let n = norm(v);
        v.scale(&[1.0 / n]);
    }
    for (j, v) in vs.iter().enumerate() {
        y.assign(&[j], v);
    }
}

fn load(path: &PathBuf, like: &H, count: usize) -> Result<(H, Vec<f64>), Box<dyn Error>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<[u8; 8]> {
        f.read_exact(&mut word)?;
        Ok(word)
    };
    let _ = next(&mut f)?;
    let m = u64::from_le_bytes(next(&mut f)?) as usize;
    let len = u64::from_le_bytes(next(&mut f)?) as usize;
    // Only the first `count` vectors are kept: the file is read site by site (all m vectors
    // of a site are stored together), so the other vectors are never held.
    let c = count.min(m);
    let mut x = like.zeros_with(c);
    let sites = x.num_sites();
    let reals = like.site_len() / like.num_vectors();
    if sites * m * reals != len {
        return Err("checkpoint for another shape".into());
    }
    let mut theta = vec![0.0; m];
    for t in theta.iter_mut() {
        *t = f64::from_le_bytes(next(&mut f)?);
    }
    let mut buf = vec![0u8; 8 * m * reals];
    for site in 0..sites {
        f.read_exact(&mut buf)?;
        let dst = x.site_mut(0, site);
        for (k, v) in dst.iter_mut().enumerate() {
            *v = f64::from_le_bytes(buf[8 * k..8 * k + 8].try_into().expect("8 bytes"));
        }
    }
    Ok((x, theta[..c].to_vec()))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err(USAGE.into());
    }
    let arg = |i: usize, d: &str| args.get(i).cloned().unwrap_or(d.to_string());
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let checkpoint = PathBuf::from(&args[4]);
    let ls: usize = arg(5, "8").parse().map_err(|_| USAGE)?;
    let n: usize = arg(6, "32").parse().map_err(|_| USAGE)?;
    let block: usize = arg(7, "2").parse().map_err(|_| USAGE)?;
    let steps: usize = arg(8, "5").parse().map_err(|_| USAGE)?;
    let ratio: f64 = arg(9, "30").parse().map_err(|_| USAGE)?;
    let coarse_tol: f64 = arg(10, "1e-8").parse().map_err(|_| USAGE)?;
    let skip_alone = args.iter().skip(11).any(|s| s == "noalone");
    // The coarse solver: `cr` (default), `minres`, `pminres` (block-|H| preconditioned).
    let solver_name = args
        .iter()
        .skip(11)
        .find(|s| ["cr", "minres", "pminres", "kbicg", "bicg", "dbicg"].contains(&s.as_str()))
        .cloned()
        .unwrap_or_else(|| "cr".to_string());
    // `defl:K:STEPS`: deflate the MINRES coarse solves by the K Ritz vectors nearest zero
    // from STEPS Lanczos steps on each coarse factor.
    let deflate: Option<(usize, usize)> = args.iter().skip(11).find_map(|s| {
        let mut it = s.strip_prefix("defl:")?.split(':');
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    });
    // `idefl:K:ITERS`: the deflation from ITERS steps of subspace iteration with H⁻¹.
    let inverse_deflate: Option<(usize, usize)> = args.iter().skip(11).find_map(|s| {
        let mut it = s.strip_prefix("idefl:")?.split(':');
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    });
    // `post`: no smoothing before the coarse correction. `fgmres:K`: the cycle as the
    // preconditioner of FGMRES(K) (on the fine level) instead of BiCGStab.
    let post_only = args.iter().skip(11).any(|s| s == "post");
    // `product`: the product Galerkin coarse correction (matrix-free, GMRES), its Ã
    // applications reported as the coarse count.
    let product = args.iter().skip(11).any(|s| s == "product");
    let outer: Option<usize> = args
        .iter()
        .skip(11)
        .find_map(|s| s.strip_prefix("fgmres:")?.parse().ok());

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
    let lift = Lift {
        lat: &lat,
        index,
        ls,
    };
    let full = |nv: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, nv, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * nv, seed).as_slice());
        v
    };
    let like = lift.even(&full(1, 1));
    let nothing = Count::default();
    let hi = 1.1
        * largest_magnitude(
            &mut HalfA::new(&op, &pv, &like, &nothing),
            &mut lift.even(&full(1, 5)),
            50,
        );

    // The basis: s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y) for the lowest N/2 Ritz vectors y of Ã,
    // and D s.
    let (mut y, theta) = load(&checkpoint, &like, n / 2)?;
    let agg = Aggregation::new(&lat, [block; ND]);
    // `boot:R:K:TOL`: R rounds of bootstrap setup, y ← M^K y with the current cycle M
    // (coarse solves to TOL), orthonormalised, and P rebuilt: inverse subspace iteration
    // with the multigrid as the approximate inverse.
    let boot: Option<(usize, usize, f64)> = args.iter().skip(11).find_map(|s| {
        let mut it = s.strip_prefix("boot:")?.split(':');
        Some((
            it.next()?.parse().ok()?,
            it.next()?.parse().ok()?,
            it.next()?.parse().ok()?,
        ))
    });
    // `st`: the basis s and t̂ (see `basis_from`) instead of s and D s.
    let pv_lift = args.iter().skip(11).any(|s| s == "st");
    let quotient = |y: &H| -> (f64, f64) {
        let mut a = HalfA::new(&op, &pv, &like, &nothing);
        let mut ay = y.zeros_like();
        a.mult(&mut ay, y);
        let nv = y.num_vectors();
        let (mut num, mut den) = (vec![0.0; nv], vec![0.0; nv]);
        ay.norm_sqr(&mut num);
        y.norm_sqr(&mut den);
        let mut q: Vec<f64> = (0..nv).map(|j| (num[j] / den[j]).sqrt()).collect();
        q.sort_by(f64::total_cmp);
        (q[0], q[nv / 2])
    };
    if let Some((rounds, k, tol)) = boot {
        let (q0, qm) = quotient(&y);
        println!("  bootstrap: start |Ãy|/|y| lowest {q0:.2e}, median {qm:.2e}");
        for round in 0..rounds {
            let start = Instant::now();
            let (pb, _, _) = basis_from(&op, &pv, pv_lift, &lift, index, &agg, &y)?;
            let mut cdb = CoarseOp::build(&op, &mut op.work(), &pb);
            let mut cpvb = CoarseOp::build(&pv, &mut pv.work(), &pb);
            let like_cb = pb.coarse_field(1);
            let mdb = BlockAbsInv::build(&mut cdb, false, &like_cb);
            let mpvb = BlockAbsInv::build(&mut cpvb, true, &like_cb);
            let (na, nc) = (Count::default(), Count::default());
            let mut cycle = Cycle {
                product: None,
                a: HalfA::new(&op, &pv, &like, &na),
                op: &op,
                pv: &pv,
                smoother: Chebyshev::new(
                    HalfA::new(&op, &pv, &like, &na),
                    steps,
                    hi / ratio,
                    hi,
                    &like,
                ),
                coarse: Some((&pb, &mut cdb, &mut cpvb)),
                lift: &lift,
                coarse_tol: tol,
                solver: CoarseSolver::Minres(Some((&mdb, &mpvb)), None),
                pre: true,
                post: true,
                pv_adj: true,
                nc: nc.clone(),
            };
            for _ in 0..k {
                let mut z = y.zeros_like();
                cycle.mult(&mut z, &y);
                y = z;
            }
            orthonormalise_h(&mut y);
            let (q0, qm) = quotient(&y);
            println!(
                "  bootstrap round {round}: |Ãy|/|y| lowest {q0:.2e}, median {qm:.2e}; {} Ã, {} coarse ({:.0} s)",
                na.get(),
                nc.get(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    let half_n = y.num_vectors();
    let mut work = op.even_odd_work();
    let (p, restricted, odd_part) = basis_from(&op, &pv, pv_lift, &lift, index, &agg, &y)?;
    let mut cd = CoarseOp::build(&op, &mut op.work(), &p);
    // `exactpv`: P†D_PV†P probed from D_PV† on the fine lattice. Otherwise it is the adjoint
    // of P†D_PV P by Γ5 (`mult_dag`), which for Möbius (c ≠ 0) is not exact: Γ5 D is not
    // Hermitian there.
    let pv_adj = !args.iter().skip(11).any(|s| s == "exactpv");
    let mut cpv = if pv_adj {
        CoarseOp::build(&pv, &mut pv.work(), &p)
    } else {
        let start = Instant::now();
        let mut d = PvDagger { pv: &pv };
        let c = CoarseOp::probe(&mut d, &p, &p, false)?;
        println!(
            "  P†D_PV†P probed on the fine lattice ({:.0} s)",
            start.elapsed().as_secs_f64()
        );
        c
    };
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; Ã on the even sites, |λ_max(Ã)| ≈ {:.3}; P from s and D s for the lowest {half_n} Ritz vectors of Ã (σ {:.2e} … {:.2e}) on {block}^4, coarse dimension {}; Chebyshev {steps}/{ratio}; coarse CR to {coarse_tol:e}",
        hi / 1.1,
        theta[0].sqrt(),
        theta[half_n - 1].sqrt(),
        cd.dim()
    );
    if pv_lift {
        println!("  basis s and t̂ = PV lift of πDs; check: |odd part of D_PV† t̂| / |D_PV† t̂| = {odd_part:.1e}");
    } else {
        println!(
            "  check: |odd part of D s| / |D s| = {odd_part:.1e} (D s lives on the even sites)"
        );
    }
    // The block |H| preconditioners, and a check of the coarse solves with and without them.
    let like_c = p.coarse_field(1);
    let (gram, gram_line) = EvenGram::build(&p, &lift);
    println!("  {gram_line}");
    let start = Instant::now();
    let mut m_d = BlockAbsInv::build(&mut cd, false, &like_c);
    let mut m_pv = BlockAbsInv::build(&mut cpv, pv_adj, &like_c);
    let built = start.elapsed().as_secs_f64();
    // Are the coarse operators the Galerkin ones? |A_c x − P† A P x| / |P† A P x| for x the
    // first restricted basis vector, A = D (P†DP) and A = D_PV† (P†D_PV†P).
    {
        let x = &restricted[0];
        let mut px = p.fine_field(1);
        p.prolong(&mut px, x);
        let mut errs = Vec::new();
        for (a, adjoint) in [(&op, false), (&pv, true)] {
            let mut apx = px.zeros_like();
            a.mult(
                &mut a.work(),
                &mut apx,
                &px,
                if adjoint { Variant::Dagger } else { Variant::D },
            );
            let mut want = p.coarse_field(1);
            p.restrict(&mut want, &apx);
            let mut got = want.zeros_like();
            if !adjoint {
                cd.mult(&mut got, x);
            } else if pv_adj {
                cpv.mult_dag(&mut got, x);
            } else {
                cpv.mult(&mut got, x);
            }
            got.axpy(&[C64::new(-1.0, 0.0)], &want);
            errs.push(norm(&got) / norm(&want));
        }
        println!(
            "  check: coarse operators against P†AP: P†DP {:.1e}, P†D_PV†P {:.1e}",
            errs[0], errs[1]
        );
    }
    // How far from Hermitian are H = Γ5 P†DP and H = Γ5 P†D_PV†P (exactly Hermitian only for
    // c = 0)? |Hv − H†v| / |Hv| with H† v = P† A† P Γ5 v from the fine lattice, for a random
    // v and for the first restricted basis vector (a low mode).
    if args.iter().skip(11).any(|s| s == "herm") {
        let mut rnd = p.coarse_field(1);
        p.restrict(&mut rnd, &full(1, 77));
        for (name, a, dagger) in [("Γ5 P†DP", &op, false), ("Γ5 P†D_PV†P", &pv, true)] {
            let mut line = format!("  check: non-Hermiticity of {name}:");
            for (label, v) in [("random", &rnd), ("low mode", &restricted[0])] {
                // H v = Γ5 P† A P v, A = D or D_PV†.
                let mut f = p.fine_field(1);
                p.prolong(&mut f, v);
                let mut af = f.zeros_like();
                let variant = if dagger { Variant::Dagger } else { Variant::D };
                a.mult(&mut a.work(), &mut af, &f, variant);
                let mut hv = p.coarse_field(1);
                p.restrict(&mut hv, &af);
                coarse_gamma5(&mut hv);
                // H† v = P† A† P Γ5 v.
                let mut g = v.zeros_like();
                g.copy_from(v);
                coarse_gamma5(&mut g);
                p.prolong(&mut f, &g);
                let variant = if dagger { Variant::D } else { Variant::Dagger };
                a.mult(&mut a.work(), &mut af, &f, variant);
                let mut hdv = p.coarse_field(1);
                p.restrict(&mut hdv, &af);
                hdv.axpy(&[C64::new(-1.0, 0.0)], &hv);
                line += &format!(" {label} {:.1e}", norm(&hdv) / norm(&hv));
            }
            println!("{line}");
        }
    }
    // Is the coarse correction alone exact on the vectors P was built from? r = Ã y_j for
    // the first and the last y, e = (coarse correction) r, |e − y| / |y|.
    {
        let nc = Count::default();
        let mut alone = Cycle {
            product: None,
            a: HalfA::new(&op, &pv, &like, &nothing),
            op: &op,
            pv: &pv,
            smoother: Chebyshev::new(
                HalfA::new(&op, &pv, &like, &nothing),
                1,
                hi / ratio,
                hi,
                &like,
            ),
            coarse: Some((&p, &mut cd, &mut cpv)),
            lift: &lift,
            coarse_tol: args
                .iter()
                .skip(11)
                .find_map(|s| s.strip_prefix("exacttol:")?.parse().ok())
                .unwrap_or(1e-20),
            solver: CoarseSolver::Minres(Some((&m_d, &m_pv)), None),
            pre: false,
            post: false,
            pv_adj,
            nc: nc.clone(),
        };
        let mut errors = Vec::new();
        for j in [0, half_n - 1] {
            let yj = y.select(&[j]);
            let mut r = yj.zeros_like();
            HalfA::new(&op, &pv, &like, &nothing).mult(&mut r, &yj);
            let mut e = yj.zeros_like();
            alone.mult(&mut e, &r);
            e.axpy(&[C64::new(-1.0, 0.0)], &yj);
            errors.push(norm(&e) / norm(&yj));
        }
        println!(
            "  check: coarse correction alone on r = Ã y_j: |e − y_j|/|y_j| = {:.1e} (first), {:.1e} (last); {} coarse applications",
            errors[0],
            errors[1],
            nc.get()
        );
    }
    // `abs2:K:SB:TOL:PASSES`: a second level of the absolute-value preconditioner, from K
    // test vectors (PASSES steps of inverse iteration, MINRES to TOL) on SB^4 super-blocks.
    let abs2 = args.iter().skip(11).find_map(|s| {
        let mut it = s.strip_prefix("abs2:")?.split(':');
        let k: usize = it.next()?.parse().ok()?;
        let sb: usize = it.next()?.parse().ok()?;
        let tol: f64 = it.next()?.parse().ok()?;
        let passes: usize = it.next().and_then(|p| p.parse().ok()).unwrap_or(1);
        Some((k, sb, tol, passes))
    });
    // `abs2f:SB`: the second level from the restricted basis vectors instead.
    let abs2f: Option<usize> = args
        .iter()
        .skip(11)
        .find_map(|s| s.strip_prefix("abs2f:")?.parse().ok());
    if let Some(sb) = abs2f {
        for (name, is_pv) in [("P†DP", false), ("P†D_PV†P", true)] {
            let adjoint = is_pv && pv_adj;
            let (a, m) = if is_pv {
                (&mut cpv, &mut m_pv)
            } else {
                (&mut cd, &mut m_d)
            };
            let mut quotients: Vec<f64> = restricted
                .iter()
                .map(|x| {
                    let mut hx = x.zeros_like();
                    h_mult(a, adjoint, &mut hx, x);
                    norm(&hx) / norm(x)
                })
                .collect();
            quotients.sort_by(f64::total_cmp);
            let (level, line) = AbsLevel::build(a, adjoint, &restricted, sb);
            println!(
                "  second level for {name}: the {} restricted basis vectors (|Hx|/|x| {:.2e} … {:.2e}), {line}",
                restricted.len(),
                quotients[0],
                quotients[quotients.len() - 1]
            );
            m.level = Some(Box::new(level));
        }
    }
    // `sm:SB2,SB3,…`: the signed multilevel V-cycle for BiCGStab (`bicg`, `dbicg`), from
    // the restricted basis vectors, with these super-block sizes; the last level is dense.
    let sm: Option<Vec<usize>> = args.iter().skip(11).find_map(|s| {
        s.strip_prefix("sm:")?
            .split(',')
            .map(|x| x.parse().ok())
            .collect()
    });
    if let Some(sbs) = &sm {
        for (name, is_pv) in [("P†DP", false), ("P†D_PV†P", true)] {
            let adjoint = is_pv && pv_adj;
            let (a, m) = if is_pv {
                (&mut cpv, &mut m_pv)
            } else {
                (&mut cd, &mut m_d)
            };
            let (multi, line) = SignedMulti::build(a, adjoint, &restricted, sbs);
            println!("  signed multilevel for {name}: {line}");
            m.level = Some(Box::new(multi));
        }
    }
    // `absm:SB2,SB3,…`: the multilevel version from the restricted basis vectors, with
    // these super-block sizes; the last level is dense.
    let absm: Option<Vec<usize>> = args.iter().skip(11).find_map(|s| {
        s.strip_prefix("absm:")?
            .split(',')
            .map(|x| x.parse().ok())
            .collect()
    });
    if let Some(sbs) = &absm {
        for (name, is_pv) in [("P†DP", false), ("P†D_PV†P", true)] {
            let adjoint = is_pv && pv_adj;
            let (a, m) = if is_pv {
                (&mut cpv, &mut m_pv)
            } else {
                (&mut cd, &mut m_d)
            };
            let (multi, line) = AbsMulti::build(a, adjoint, &restricted, sbs);
            println!("  multilevel for {name}: {line}");
            m.level = Some(Box::new(multi));
        }
    }
    if let Some((k, sb, tol, passes)) = abs2 {
        for (name, is_pv) in [("P†DP", false), ("P†D_PV†P", true)] {
            let adjoint = is_pv && pv_adj;
            let (a, m) = if is_pv {
                (&mut cpv, &mut m_pv)
            } else {
                (&mut cd, &mut m_d)
            };
            let (vectors, n, q) = m.test_vectors(a, adjoint, &like_c, k, tol, passes);
            let (level, line) = AbsLevel::build(a, adjoint, &vectors, sb);
            println!(
                "  second level for {name}: {k} test vectors ({passes} passes, {n} applications, median |Hx|/|x| {q:.2e}), {line}"
            );
            m.level = Some(Box::new(level));
        }
    }
    // The cost of a coarse application against one of Ã, timed here.
    {
        let mut x = p.coarse_field(1);
        p.restrict(&mut x, &full(1, 41));
        let mut y = x.zeros_like();
        let t0 = Instant::now();
        for _ in 0..20 {
            cd.mult(&mut y, &x);
        }
        let coarse_time = t0.elapsed().as_secs_f64() / 20.0;
        let xe = lift.even(&full(1, 43));
        let mut ye = xe.zeros_like();
        let mut a = HalfA::new(&op, &pv, &like, &nothing);
        let t0 = Instant::now();
        for _ in 0..10 {
            a.mult(&mut ye, &xe);
        }
        let fine_time = t0.elapsed().as_secs_f64() / 10.0;
        println!(
            "  timing: coarse application {:.1} ms, Ã {:.1} ms: one coarse application = {:.3} Ã",
            1e3 * coarse_time,
            1e3 * fine_time,
            coarse_time / fine_time
        );
    }
    let deflations = match (deflate, inverse_deflate) {
        (_, Some((k, iterations))) => {
            let (d_d, line_d) =
                Deflation::build_inverse(&mut cd, false, &m_d, &like_c, k, iterations);
            println!("  deflation of P†DP: {line_d}");
            let (d_pv, line_pv) =
                Deflation::build_inverse(&mut cpv, pv_adj, &m_pv, &like_c, k, iterations);
            println!("  deflation of P†D_PV†P: {line_pv}");
            Some((d_d, d_pv))
        }
        (Some((k, steps)), None) => {
            let (d_d, line_d) = Deflation::build(&mut cd, false, &like_c, steps, k);
            println!("  deflation of P†DP: {line_d}");
            let (d_pv, line_pv) = Deflation::build(&mut cpv, pv_adj, &like_c, steps, k);
            println!("  deflation of P†D_PV†P: {line_pv}");
            Some((d_d, d_pv))
        }
        (None, None) => None,
    };
    if !args.iter().skip(11).any(|s| s == "nocheck") {
        let mut g = p.coarse_field(1);
        p.restrict(&mut g, &full(1, 31));
        for (name, is_pv) in [("P†DP", false), ("P†D_PV†P", true)] {
            let adjoint = is_pv && pv_adj;
            let (a, m): (&mut CoarseOp<f64>, &BlockAbsInv) = if is_pv {
                (&mut cpv, &m_pv)
            } else {
                (&mut cd, &m_d)
            };
            let d = deflations
                .as_ref()
                .map(|(dd, dp)| if adjoint { dp } else { dd });
            let mut line = format!("  check: {name} to 1e-12:");
            let mut variants = vec![("CR", 0), ("MINRES", 1), ("block-|H| MINRES", 2)];
            if d.is_some() {
                variants.extend([("deflated MINRES", 3), ("deflated block-|H| MINRES", 4)]);
            }
            for (label, which) in variants {
                let mut x = g.zeros_like();
                let n = match which {
                    0 => coarse_solve(a, adjoint, &mut x, &g, 1e-12),
                    1 => coarse_solve_minres(a, adjoint, None, None, &mut x, &g, 1e-12),
                    2 => coarse_solve_minres(a, adjoint, Some(m), None, &mut x, &g, 1e-12),
                    3 => coarse_solve_minres(a, adjoint, None, d, &mut x, &g, 1e-12),
                    _ => coarse_solve_minres(a, adjoint, Some(m), d, &mut x, &g, 1e-12),
                };
                let mut ax = g.zeros_like();
                if adjoint {
                    a.mult_dag(&mut ax, &x);
                } else {
                    a.mult(&mut ax, &x);
                }
                ax.axpy(&[C64::new(-1.0, 0.0)], &g);
                line += &format!("  {label} {n} (|Ax−b|/|b| {:.0e})", norm(&ax) / norm(&g));
            }
            println!("{line}");
        }
        // The two factors as one operator K, to the same tolerance.
        let nk = Count::default();
        let mut k = CoarseK {
            cd: &mut cd,
            cpv: &mut cpv,
            gram: &gram,
            n: nk.clone(),
        };
        let mut c = g.zeros_like();
        let mut prec = CoarseKPrec {
            md: &m_d,
            mpv: &m_pv,
            gram: &gram,
        };
        let result = BiCgStab::new(Stop {
            tolerance_sqr: 1e-12,
            max_iterations: 5000,
            max_restarts: 0,
        })
        .solve(&mut k, Some(&mut prec), &mut c, &g);
        let mut kc = g.zeros_like();
        k.mult(&mut kc, &c);
        kc.axpy(&[C64::new(-1.0, 0.0)], &g);
        println!(
            "  check: K = (P†D_PV†P) E⁻¹ (P†DP) to 1e-12 by BiCGStab: {} iterations, {} coarse applications (|Kc−g|/|g| {:.0e})",
            match result {
                Ok(i) => format!("{}", i.iterations),
                Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
                Err(e) => format!("failed: {e}"),
            },
            nk.get(),
            norm(&kc) / norm(&g)
        );
        println!("  (block |H| preconditioners built in {built:.1} s)");
    }

    if args.iter().skip(11).any(|s| s == "checkonly") {
        return Ok(());
    }

    // `seeds:A-B`: right-hand sides from the seeds A..=B, one after the other on the same
    // setup (default 91). `withproduct`: also the product Galerkin cycle, as mode `product`.
    let (seed_lo, seed_hi): (u64, u64) = args
        .iter()
        .skip(11)
        .find_map(|s| {
            let (a, b) = s.strip_prefix("seeds:")?.split_once('-')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((91, 91));
    let with_product = args.iter().skip(11).any(|s| s == "withproduct");
    for seed in seed_lo..=seed_hi {
        println!("  right-hand side seed {seed}");
        // The right-hand side and the solution, as in `mdwf_evenodd_a`.
        let b = full(1, seed);
        let (mut b_e, mut b_o) = (lift.half(Parity::Even, 1), lift.half(Parity::Odd, 1));
        extract_parity(&mut b_e, &b, index, Parity::Even);
        extract_parity(&mut b_o, &b, index, Parity::Odd);
        let (mut t_o, mut t_e) = (lift.half(Parity::Odd, 1), lift.half(Parity::Even, 1));
        op.diag_inv(&mut work, &mut t_o, &b_o, Parity::Odd, Dagger::No);
        op.hop(&mut work, &mut t_e, &t_o, Parity::Even, Dagger::No);
        t_e.aypx(&[C64::new(-1.0, 0.0)], &b_e);
        let mut rhs = lift.half(Parity::Even, 1);
        HalfA::new(&op, &pv, &like, &nothing).x_dagger(&mut rhs, &t_e);
        let physical = |y: &H| {
            let mut work = op.even_odd_work();
            let mut x_e = lift.half(Parity::Even, 1);
            op.diag_inv(&mut work, &mut x_e, y, Parity::Even, Dagger::No);
            let mut t_o = lift.half(Parity::Odd, 1);
            op.hop(&mut work, &mut t_o, &x_e, Parity::Odd, Dagger::No);
            t_o.aypx(&[C64::new(-1.0, 0.0)], &b_o);
            let mut x_o = lift.half(Parity::Odd, 1);
            op.diag_inv(&mut work, &mut x_o, &t_o, Parity::Odd, Dagger::No);
            let mut x = lift.inject(&x_e);
            insert_parity(&mut x, &x_o, index);
            let mut r = b.zeros_like();
            op.mult(&mut op.work(), &mut r, &x, Variant::D);
            r.aypx(&[C64::new(-1.0, 0.0)], &b);
            (norm(&r) / norm(&b)).powi(2)
        };
        let stop = Stop {
            tolerance_sqr: 1e-18,
            max_iterations: 2000,
            max_restarts: 1,
        };

        let defl = deflations.as_ref().map(|(dd, dp)| (dd, dp));
        let mut modes: Vec<&str> = if skip_alone {
            vec!["cycle"]
        } else {
            vec!["none", "polynomial", "cycle"]
        };
        if with_product {
            modes.push("product");
        }
        for mode in modes {
            let (na, nc) = (Count::default(), Count::default());
            let mut a = HalfA::new(&op, &pv, &like, &na);
            let mut y_sol = like.zeros_like();
            let start = Instant::now();
            let result = match mode {
                "none" => BiCgStab::new(stop).solve(&mut a, None, &mut y_sol, &rhs),
                "polynomial" => {
                    let mut cheb = Chebyshev::new(
                        HalfA::new(&op, &pv, &like, &na),
                        steps,
                        hi / ratio,
                        hi,
                        &like,
                    );
                    BiCgStab::new(stop).solve(&mut a, Some(&mut cheb), &mut y_sol, &rhs)
                }
                _ => {
                    let mut cycle = Cycle {
                        product: (product || mode == "product").then(|| ProductGalerkin {
                            a: HalfA::new(&op, &pv, &like, &nc),
                            op: &op,
                            pv: &pv,
                            p: &p,
                            lift: &lift,
                        }),
                        a: HalfA::new(&op, &pv, &like, &na),
                        op: &op,
                        pv: &pv,
                        smoother: Chebyshev::new(
                            HalfA::new(&op, &pv, &like, &na),
                            steps,
                            hi / ratio,
                            hi,
                            &like,
                        ),
                        coarse: Some((&p, &mut cd, &mut cpv)),
                        lift: &lift,
                        coarse_tol,
                        solver: match solver_name.as_str() {
                            "minres" => CoarseSolver::Minres(None, defl),
                            "pminres" => CoarseSolver::Minres(Some((&m_d, &m_pv)), defl),
                            "kbicg" => CoarseSolver::K(&gram, &m_d, &m_pv),
                            "bicg" => CoarseSolver::BiCg((&m_d, &m_pv), false),
                            "dbicg" => CoarseSolver::BiCg((&m_d, &m_pv), true),
                            _ => CoarseSolver::Cr,
                        },
                        pre: !post_only,
                        post: true,
                        pv_adj,
                        nc: nc.clone(),
                    };
                    match outer {
                        Some(k) => Fgmres::new(Stop {
                            max_iterations: k,
                            max_restarts: 100,
                            ..stop
                        })
                        .solve(&mut a, Some(&mut cycle), &mut y_sol, &rhs),
                        None => {
                            BiCgStab::new(stop).solve(&mut a, Some(&mut cycle), &mut y_sol, &rhs)
                        }
                    }
                }
            };
            let it = match result {
                Ok(i) => format!("{}", i.iterations),
                Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
                Err(e) => format!("failed: {e}"),
            };
            println!(
            "  BiCGStab, {mode:<11} {it:>5} iterations: {:>6} Ã = {:>6} sweeps, {:>6} coarse, |Dx−b|² {:.0e}, {:.0} s",
            na.get(),
            2 * na.get(),
            nc.get(),
            physical(&y_sol),
            start.elapsed().as_secs_f64()
        );
        }
    }
    Ok(())
}
