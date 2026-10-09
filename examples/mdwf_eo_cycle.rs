//! BiCGStab on the even-odd operator Ã = S_X† S_Y with a two-level cycle whose coarse
//! correction is the factorized, Γ5-Hermitian one of `mdwf_g5cycle`, built on the full
//! lattice and entered and left on the even sites.
//!
//! ```text
//! cargo run --release --example mdwf_eo_cycle -- CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [STEPS] [RATIO] [COARSE_TOL] [noalone]
//! ```
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
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp};
use ruttice::solver::{largest_magnitude, BiCgStab, Chebyshev, SolveError, Solver, Stop};
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

const USAGE: &str = "usage: mdwf_eo_cycle CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [STEPS] [RATIO] [COARSE_TOL] [noalone]";

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;
type C = Field<f64, Coarse>;
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
            let mut m = vec![C64::ZERO; dim * dim];
            for i in 0..dim {
                for k in 0..dim {
                    let mut z = C64::ZERO;
                    for j in 0..dim {
                        z += v[i * dim + j].scale(1.0 / lam[j].abs().max(1e-300))
                            * v[k * dim + j].conj();
                    }
                    m[i * dim + k] = z;
                }
            }
            *b = m;
        }
        Self { dim, blocks }
    }

    fn apply(&self, out: &mut C, inp: &C) {
        let (dim, nv) = (self.dim, inp.num_vectors());
        for (x, m) in self.blocks.iter().enumerate() {
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
}

/// Preconditioned conjugate residuals for H = Γ5 A_c (Γ5 A_c† with `adjoint`) with the
/// Hermitian positive definite M: solves A_c x = b; returns the coarse applications.
fn coarse_solve_pcr(
    a: &mut CoarseOp<f64>,
    adjoint: bool,
    m: &BlockAbsInv,
    x: &mut C,
    b: &C,
    tol_sqr: f64,
) -> usize {
    let mut apply = |out: &mut C, inp: &C| {
        if adjoint {
            a.mult_dag(out, inp);
        } else {
            a.mult(out, inp);
        }
        coarse_gamma5(out);
    };
    let nv = b.num_vectors();
    let mut r = b.zeros_like();
    r.copy_from(b);
    coarse_gamma5(&mut r);
    let mut bb = vec![0.0; nv];
    r.norm_sqr(&mut bb);
    x.scale(&vec![0.0; nv]);
    let mut z = r.zeros_like();
    m.apply(&mut z, &r);
    let mut p = z.zeros_like();
    p.copy_from(&z);
    let mut w = z.zeros_like();
    apply(&mut w, &z);
    let mut q = w.zeros_like();
    q.copy_from(&w);
    let mut mq = q.zeros_like();
    let mut zw = vec![C64::ZERO; nv];
    z.dot(&w, &mut zw);
    let mut rr = vec![0.0; nv];
    for it in 0..5000 {
        m.apply(&mut mq, &q);
        let mut qmq = vec![C64::ZERO; nv];
        q.dot(&mq, &mut qmq);
        let alpha: Vec<C64> = (0..nv)
            .map(|j| C64::new(zw[j].re / qmq[j].re.max(1e-300), 0.0))
            .collect();
        let minus: Vec<C64> = alpha.iter().map(|a| C64::new(-a.re, 0.0)).collect();
        x.axpy(&alpha, &p);
        r.axpy(&minus, &q);
        z.axpy(&minus, &mq);
        r.norm_sqr(&mut rr);
        if (0..nv).all(|j| rr[j] <= tol_sqr * bb[j]) {
            return it + 2;
        }
        apply(&mut w, &z);
        let mut zw_new = vec![C64::ZERO; nv];
        z.dot(&w, &mut zw_new);
        let beta: Vec<C64> = (0..nv)
            .map(|j| C64::new(zw_new[j].re / zw[j].re, 0.0))
            .collect();
        p.aypx(&beta, &z);
        q.aypx(&beta, &w);
        zw = zw_new;
    }
    5001
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

/// The cycle.
struct Cycle<'a> {
    a: HalfA<'a>,
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    smoother: Chebyshev<HalfA<'a>, H>,
    coarse: Option<(&'a Basis<f64>, &'a mut CoarseOp<f64>, &'a mut CoarseOp<f64>)>,
    lift: &'a Lift<'a>,
    coarse_tol: f64,
    /// Block |H| preconditioners for (P†DP, P†D_PV†P): preconditioned conjugate residuals.
    prec: Option<(&'a BlockAbsInv, &'a BlockAbsInv)>,
    nc: Count,
}

impl LinearFn<H> for Cycle<'_> {
    fn mult(&mut self, y: &mut H, b: &H) {
        let nv = b.num_vectors();
        let minus = vec![C64::new(-1.0, 0.0); nv];
        let one = vec![C64::new(1.0, 0.0); nv];
        self.smoother.mult(y, b);
        let mut r = b.zeros_like();
        let mut e = b.zeros_like();
        if let Some((p, cd, cpv)) = self.coarse.as_mut() {
            self.a.mult(&mut r, y);
            r.aypx(&minus, b);
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
            // w = (P†D_PV†P)⁻¹ g; w ← E w; c = (P†DP)⁻¹ w.
            let mut w = g.zeros_like();
            let mut n = match self.prec {
                Some((_, mpv)) => coarse_solve_pcr(cpv, true, mpv, &mut w, &g, self.coarse_tol),
                None => coarse_solve(cpv, true, &mut w, &g, self.coarse_tol),
            };
            let mut f = V::with_shape(self.lift.lat, 24, self.lift.ls, nv, 1);
            p.prolong(&mut f, &w);
            p.restrict(&mut w, &self.lift.inject(&self.lift.even(&f)));
            let mut c = w.zeros_like();
            n += match self.prec {
                Some((md, _)) => coarse_solve_pcr(cd, false, md, &mut c, &w, self.coarse_tol),
                None => coarse_solve(cd, false, &mut c, &w, self.coarse_tol),
            };
            self.nc.set(self.nc.get() + n);
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
        self.a.mult(&mut r, y);
        r.aypx(&minus, b);
        self.smoother.mult(&mut e, &r);
        y.axpy(&one, &e);
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
    let mut x = like.zeros_with(m);
    if x.as_slice().len() != len {
        return Err("checkpoint for another shape".into());
    }
    let mut theta = vec![0.0; m];
    for t in theta.iter_mut() {
        *t = f64::from_le_bytes(next(&mut f)?);
    }
    for v in x.as_mut_slice() {
        *v = f64::from_le_bytes(next(&mut f)?);
    }
    let c = count.min(m);
    Ok((x.select(&(0..c).collect::<Vec<_>>()), theta[..c].to_vec()))
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
    let use_pcr = args.iter().skip(11).any(|s| s == "pcr");

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
    let (y, theta) = load(&checkpoint, &like, n / 2)?;
    let half_n = y.num_vectors();
    let mut work = op.even_odd_work();
    let mut xe = y.zeros_like();
    op.diag_inv(&mut work, &mut xe, &y, Parity::Even, Dagger::No);
    let mut t = lift.half(Parity::Odd, half_n);
    op.hop(&mut work, &mut t, &xe, Parity::Odd, Dagger::No);
    let mut xo = lift.half(Parity::Odd, half_n);
    op.diag_inv(&mut work, &mut xo, &t, Parity::Odd, Dagger::No);
    xo.scale(&vec![-1.0; half_n]);
    let mut s = lift.inject(&xe);
    insert_parity(&mut s, &xo, index);
    let mut ds = s.zeros_like();
    op.mult(&mut op.work(), &mut ds, &s, Variant::D);
    // D s lives on the even sites: a check of the reconstruction.
    let odd_part = {
        let mut o = lift.half(Parity::Odd, half_n);
        extract_parity(&mut o, &ds, index, Parity::Odd);
        norm(&o) / norm(&ds)
    };
    let mut vectors = s.zeros_with(2 * half_n);
    vectors.assign(&(0..half_n).collect::<Vec<_>>(), &s);
    vectors.assign(&(half_n..2 * half_n).collect::<Vec<_>>(), &ds);
    drop((s, ds, xe, xo, t));
    let mut g5 = vectors.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
    let agg = Aggregation::new(&lat, [block; ND]);
    let p = Basis::from_vectors(&agg, &vectors, &g5)?;
    drop((vectors, g5));
    let mut cd = CoarseOp::build(&op, &mut op.work(), &p);
    let mut cpv = CoarseOp::build(&pv, &mut pv.work(), &p);
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; Ã on the even sites, |λ_max(Ã)| ≈ {:.3}; P from s and D s for the lowest {half_n} Ritz vectors of Ã (σ {:.2e} … {:.2e}) on {block}^4, coarse dimension {}; Chebyshev {steps}/{ratio}; coarse CR to {coarse_tol:e}",
        hi / 1.1,
        theta[0].sqrt(),
        theta[half_n - 1].sqrt(),
        cd.dim()
    );
    println!("  check: |odd part of D s| / |D s| = {odd_part:.1e} (D s lives on the even sites)");
    // The block |H| preconditioners, and a check of the coarse solves with and without them.
    let like_c = p.coarse_field(1);
    let start = Instant::now();
    let m_d = BlockAbsInv::build(&mut cd, false, &like_c);
    let m_pv = BlockAbsInv::build(&mut cpv, true, &like_c);
    let built = start.elapsed().as_secs_f64();
    {
        let mut g = p.coarse_field(1);
        p.restrict(&mut g, &full(1, 31));
        for (name, adjoint) in [("P†DP", false), ("P†D_PV†P", true)] {
            let (a, m): (&mut CoarseOp<f64>, &BlockAbsInv) = if adjoint {
                (&mut cpv, &m_pv)
            } else {
                (&mut cd, &m_d)
            };
            let mut x1 = g.zeros_like();
            let n1 = coarse_solve(a, adjoint, &mut x1, &g, 1e-12);
            let mut x2 = g.zeros_like();
            let n2 = coarse_solve_pcr(a, adjoint, m, &mut x2, &g, 1e-12);
            let mut ax = g.zeros_like();
            if adjoint {
                a.mult_dag(&mut ax, &x2);
            } else {
                a.mult(&mut ax, &x2);
            }
            ax.axpy(&[C64::new(-1.0, 0.0)], &g);
            println!(
                "  check: {name} to 1e-12: CR {n1} applications, block-|H| PCR {n2}; PCR |A x − b|/|b| = {:.1e}",
                norm(&ax) / norm(&g)
            );
        }
        println!("  (block |H| preconditioners built in {built:.1} s)");
    }

    // The right-hand side and the solution, as in `mdwf_evenodd_a`.
    let b = full(1, 91);
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

    let modes: Vec<&str> = if skip_alone {
        vec!["cycle"]
    } else {
        vec!["none", "polynomial", "cycle"]
    };
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
                    prec: if use_pcr { Some((&m_d, &m_pv)) } else { None },
                    nc: nc.clone(),
                };
                BiCgStab::new(stop).solve(&mut a, Some(&mut cycle), &mut y_sol, &rhs)
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
    Ok(())
}
