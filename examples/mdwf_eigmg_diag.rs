//! Why the coarse correction of the domain-wall multigrid fails with genuine low modes in
//! P, and whether a smooth prolongation fixes it.
//!
//! ```text
//! cargo run --release --example mdwf_eigmg_diag -- CONFIG L M5 MASS CHECKPOINT [LS] [BLOCK] [N] [DEGREES...]
//! ```
//!
//! P from the lowest N (default 8) Ritz vectors of a `mdwf_coherence_eig` checkpoint, split
//! by Γ5 over BLOCK^4 aggregates (default 3). For the next four Ritz vectors t:
//! - |A P P† t| / |A t|: the residual of the part of t that P holds. Block by block, P P† t
//!   jumps at the block boundaries; for a mode with σ ≈ 3e-3 those jumps can dominate.
//! - the two-grid error (I − S A)(I − P_s (P_s† A P_s)⁻¹ P_s† A) t with the post-smoother
//!   S = GMRES(4) on A, measured by its low-mode content |X† e| (X: all Ritz vectors of the
//!   checkpoint, orthonormal), against smoothing alone, (I − S A) t. This is what decides
//!   whether the coarse grid helps.
//! - P_s = p(H) P with p the Chebyshev filter T_d(ℓ(H)) on [lo, hi], H = A†A, lo = 1e-2,
//!   for each DEGREE (default 0 16 64; 0 is P itself): a low-pass smoothing of the
//!   prolongation that removes what the block boundaries add, keeping the low modes.
//!   P_s† A P_s is applied exactly through the fine lattice.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{largest_magnitude, ChebyshevFilter, Fgmres, Solve, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, Normal, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str =
    "usage: mdwf_eigmg_diag CONFIG L M5 MASS CHECKPOINT [LS] [BLOCK] [N] [DEGREES...]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

/// The Ritz vectors of a `mdwf_coherence_eig` checkpoint (ascending) and their θ = σ².
fn load(path: &PathBuf, like: &V) -> Result<(V, Vec<f64>), Box<dyn Error>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<[u8; 8]> {
        f.read_exact(&mut word)?;
        Ok(word)
    };
    let _done = u64::from_le_bytes(next(&mut f)?);
    let m = u64::from_le_bytes(next(&mut f)?) as usize;
    let len = u64::from_le_bytes(next(&mut f)?) as usize;
    let mut x = like.zeros_with(m);
    if x.as_slice().len() != len {
        return Err(format!("{} is for another lattice or Ls", path.display()).into());
    }
    let mut theta = vec![0.0; m];
    for t in theta.iter_mut() {
        *t = f64::from_le_bytes(next(&mut f)?);
    }
    for v in x.as_mut_slice() {
        *v = f64::from_le_bytes(next(&mut f)?);
    }
    Ok((x, theta))
}

/// The smoothed prolongation P_s = p(H) P and its adjoint P_s† = P† p(H) (p(H) Hermitian),
/// and P_s† A P_s.
struct Smoothed<'a, 'b> {
    op: &'a Mdwf<f64>,
    p: &'b Basis<f64>,
    degree: usize,
    lo: f64,
    hi: f64,
    like: V,
}

impl Smoothed<'_, '_> {
    fn filter(&self, out: &mut V, inp: &V) {
        if self.degree == 0 {
            out.copy_from(inp);
            return;
        }
        let mut f = ChebyshevFilter::new(
            Normal::new(PvPreconditioned::new(self.op, inp), inp),
            self.degree,
            self.lo,
            self.hi,
            inp,
        );
        f.mult(out, inp);
    }

    fn prolong(&self, out: &mut V, c: &C) {
        let mut pc = self.like.zeros_with(c.num_vectors());
        self.p.prolong(&mut pc, c);
        self.filter(out, &pc);
    }

    fn restrict(&self, out: &mut C, r: &V) {
        let mut pr = r.zeros_like();
        self.filter(&mut pr, r);
        self.p.restrict(out, &pr);
    }
}

impl LinearFn<C> for Smoothed<'_, '_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        let mut f = self.like.zeros_with(inp.num_vectors());
        self.prolong(&mut f, inp);
        let mut af = f.zeros_like();
        PvPreconditioned::new(self.op, &f).mult(&mut af, &f);
        self.restrict(out, &af);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let checkpoint = PathBuf::from(&args[4]);
    let ls: usize = args
        .get(5)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let block: usize = args
        .get(6)
        .map_or(Ok(3), |s| s.parse())
        .map_err(|_| USAGE)?;
    let n: usize = args
        .get(7)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let degrees: Vec<usize> = if args.len() > 8 {
        args[8..]
            .iter()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()
            .map_err(|_| USAGE)?
    } else {
        vec![0, 16, 64]
    };

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
    let like = {
        let mut v = V::with_shape(&lat, 24, ls, 1, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, 1).as_slice());
        v
    };
    let (ritz, theta) = load(&checkpoint, &like)?;
    let m = ritz.num_vectors();
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4, N = {n}; {m} Ritz vectors, σ {:.2e} … {:.2e}",
        theta[0].sqrt(),
        theta[m - 1].sqrt()
    );
    let mut start = like.zeros_like();
    start.copy_from(&like);
    let hi = 1.1
        * largest_magnitude(
            &mut Normal::new(PvPreconditioned::new(&op, &like), &like),
            &mut start,
            50,
        );

    let agg = Aggregation::new(&lat, [block; ND]);
    let vectors = ritz.select(&(0..n).collect::<Vec<_>>());
    let mut g5 = vectors.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
    let p = Basis::from_vectors(&agg, &vectors, &g5)?;
    let mut a = PvPreconditioned::new(&op, &like);
    let mut smoother = Solve::new(
        Fgmres::new(Stop::simple(1e-6, 4)),
        PvPreconditioned::new(&op, &like),
    );

    // Low-mode content |X† e| / |e|, X the Ritz vectors (orthonormal).
    let low = |e: &V| -> f64 {
        let mut s = 0.0;
        for j in 0..m {
            let mut d = vec![C64::ZERO; 1];
            ritz.select(&[j]).dot(e, &mut d);
            s += d[0].norm_sqr();
        }
        s.sqrt()
    };
    // e − S A e.
    let mut smooth = |e: &V| -> V {
        let mut ae = e.zeros_like();
        a.mult(&mut ae, e);
        let mut se = e.zeros_like();
        smoother.mult(&mut se, &ae);
        let mut out = e.zeros_like();
        out.copy_from(e);
        out.axpy(&[C64::new(-1.0, 0.0)], &se);
        out
    };

    let ts: Vec<V> = (n..(n + 4).min(m)).map(|j| ritz.select(&[j])).collect();
    let mut a2 = PvPreconditioned::new(&op, &like);
    println!("  t = Ritz vector j; all numbers relative to |t| = 1 or |A t|:");
    for (k, t) in ts.iter().enumerate() {
        let mut c = p.coarse_field(1);
        p.restrict(&mut c, t);
        let mut held = like.zeros_like();
        p.prolong(&mut held, &c);
        let (mut at, mut ah) = (t.zeros_like(), t.zeros_like());
        a2.mult(&mut at, t);
        a2.mult(&mut ah, &held);
        let s = smooth(t);
        println!(
            "  j = {:2}: |PP†t| {:.3}, |A PP†t|/|A t| {:.1}; smoothing alone leaves |e| {:.3}, low-mode content {:.3}",
            n + k,
            norm(&held),
            norm(&ah) / norm(&at),
            norm(&s),
            low(&s)
        );
    }
    for degree in degrees {
        let mut ps = Smoothed {
            op: &op,
            p: &p,
            degree,
            lo: 1e-2,
            hi,
            like: like.zeros_like(),
        };
        let mut line = format!("  P_s = T_{degree}(ℓ(H)) P, Galerkin on P_s, then S:");
        for t in &ts {
            let mut at = t.zeros_like();
            a2.mult(&mut at, t);
            let mut rc = p.coarse_field(1);
            ps.restrict(&mut rc, &at);
            let mut c = rc.zeros_like();
            let solved = Fgmres::new(Stop {
                tolerance_sqr: 1e-12,
                max_iterations: 60,
                max_restarts: 5,
            })
            .solve(&mut ps, None, &mut c, &rc);
            let mut corr = t.zeros_like();
            ps.prolong(&mut corr, &c);
            let mut e = t.zeros_like();
            e.copy_from(t);
            e.axpy(&[C64::new(-1.0, 0.0)], &corr);
            let after = smooth(&e);
            line += &format!(
                "  |e| {:.3}→{:.3} low {:.3}→{:.3}{}",
                norm(&e),
                norm(&after),
                low(&e),
                low(&after),
                if solved.is_err() { "*" } else { "" }
            );
        }
        println!("{line}");
    }
    Ok(())
}
