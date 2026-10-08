//! Where the residual of the part of a low mode that P holds comes from: block boundaries
//! in four dimensions, or the fifth dimension. And a check of the exact coarse corrections
//! of `mdwf_eigmg` on vectors inside span(P).
//!
//! ```text
//! cargo run --release --example mdwf_boundary -- CONFIG L M5 MASS CHECKPOINT [LS] [N]
//! ```
//!
//! P from the lowest N (default 8) Ritz vectors of a `mdwf_coherence_eig` checkpoint, split
//! by Γ5; t the next four Ritz vectors. Printed:
//! 1. Check: for t_in = P P† t, which lies in span(P), each exact correction
//!    P (R A P)⁻¹ R A (R = P†, Q†, P†A†) must leave nothing. The projection written out here
//!    must equal the one of `Basis`.
//! 2. For blocks b^4 (b = 2, 3, 4, 6 dividing L): |A PP†t| / |A t|, and how |A PP†t|² is
//!    spread over the distance d of a site to the nearest block face (d = 0: on a face),
//!    against the share of sites at that distance.
//! 3. For b = 3 (or the smallest b > 2): the same ratio with P also split along s into
//!    groups closed under s ↔ Ls−1−s (as `mdwf_coherence_s`), and how |t|², |A t|² and
//!    |A PP†t|² are spread over s.

use rayon::prelude::*;
use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{Fgmres, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, LinearOp, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_boundary CONFIG L M5 MASS CHECKPOINT [LS] [N]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

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

/// P P† t for P spanned on each (block, chirality, group of s) by the `n` vectors of
/// `halves[chirality]` restricted there.
fn project(
    agg: &Aggregation,
    halves: &[V; 2],
    n: usize,
    ls: usize,
    groups: &[Vec<usize>],
    t: &V,
) -> V {
    let mut out = t.zeros_like();
    let site_len = t.site_len();
    let pieces: Vec<Vec<(usize, Vec<f64>)>> = (0..agg.coarse_lattice().num_sites())
        .into_par_iter()
        .map(|coarse_site| {
            let sites = agg.fine_sites(coarse_site);
            let mut acc: Vec<Vec<f64>> = vec![vec![0.0; site_len]; sites.len()];
            for half in halves {
                for group in groups {
                    let per_site = group.len() * 12;
                    let dim = sites.len() * per_site;
                    let mut w = vec![C64::ZERO; n * dim];
                    let mut tl = vec![C64::ZERO; dim];
                    for (k, site) in sites.iter().enumerate() {
                        let (hs, ts) = (half.site(0, *site as usize), t.site(0, *site as usize));
                        for (gi, s) in group.iter().enumerate() {
                            let at = k * per_site + gi * 12;
                            for c in 0..12 {
                                tl[at + c] = C64::new(ts[s * 24 + 2 * c], ts[s * 24 + 2 * c + 1]);
                                for i in 0..n {
                                    let r = (i * ls + s) * 24 + 2 * c;
                                    w[i * dim + at + c] = C64::new(hs[r], hs[r + 1]);
                                }
                            }
                        }
                    }
                    gram_schmidt(&mut w, n, dim);
                    for i in 0..n {
                        let wi = &w[i * dim..(i + 1) * dim];
                        let mut z = C64::ZERO;
                        for (a, b) in wi.iter().zip(&tl) {
                            z += a.conj() * *b;
                        }
                        for (k, _) in sites.iter().enumerate() {
                            for (gi, s) in group.iter().enumerate() {
                                let at = k * per_site + gi * 12;
                                for c in 0..12 {
                                    let v = z * wi[at + c];
                                    acc[k][s * 24 + 2 * c] += v.re;
                                    acc[k][s * 24 + 2 * c + 1] += v.im;
                                }
                            }
                        }
                    }
                }
            }
            sites.iter().map(|s| *s as usize).zip(acc).collect()
        })
        .collect();
    for block in pieces {
        for (site, values) in block {
            out.site_mut(0, site).copy_from_slice(&values);
        }
    }
    out
}

/// Partitions of s closed under s ↔ Ls−1−s, as in `mdwf_coherence_s`.
fn partitions(ls: usize) -> Vec<(String, Vec<Vec<usize>>)> {
    let nd = ls.div_ceil(2);
    let of = |ds: &[usize]| -> Vec<usize> {
        (0..ls)
            .filter(|s| ds.contains(&(*s).min(ls - 1 - s)))
            .collect()
    };
    let all: Vec<usize> = (0..nd).collect();
    let mut out = vec![("whole".to_string(), vec![of(&all)])];
    if nd > 1 {
        out.push(("d0|rest".into(), vec![of(&[0]), of(&all[1..])]));
    }
    if nd > 2 {
        out.push(("d01|rest".into(), vec![of(&[0, 1]), of(&all[2..])]));
        out.push(("each d".into(), all.iter().map(|d| of(&[*d])).collect()));
    }
    out
}

/// The exact correction P (R A P)⁻¹ R A t for R = P† (0), Q† (1), P†A† (2).
fn corrected(op: &Mdwf<f64>, [p, q]: [&Basis<f64>; 2], kind: usize, t: &V) -> V {
    struct Rap<'a, 'b> {
        a: PvPreconditioned<'a, f64>,
        p: &'b Basis<f64>,
        q: &'b Basis<f64>,
        kind: usize,
        f: V,
        g: V,
    }
    impl Rap<'_, '_> {
        fn r(&mut self, out: &mut C, r: &V) {
            match self.kind {
                0 => self.p.restrict(out, r),
                1 => self.q.restrict(out, r),
                _ => {
                    self.a.mult_dag(&mut self.g, r);
                    self.p.restrict(out, &self.g);
                }
            }
        }
    }
    impl LinearFn<C> for Rap<'_, '_> {
        fn mult(&mut self, out: &mut C, inp: &C) {
            self.p.prolong(&mut self.f, inp);
            let mut af = self.f.zeros_like();
            self.a.mult(&mut af, &self.f);
            self.r(out, &af);
        }
    }
    let mut rap = Rap {
        a: PvPreconditioned::new(op, t),
        p,
        q,
        kind,
        f: t.zeros_like(),
        g: t.zeros_like(),
    };
    let mut at = t.zeros_like();
    rap.a.mult(&mut at, t);
    let mut rc = p.coarse_field(1);
    rap.r(&mut rc, &at);
    let mut c = rc.zeros_like();
    let _ = Fgmres::new(Stop {
        tolerance_sqr: 1e-20,
        max_iterations: 100,
        max_restarts: 10,
    })
    .solve(&mut rap, None, &mut c, &rc);
    let mut left = t.zeros_like();
    p.prolong(&mut left, &c);
    left.aypx(&[C64::new(-1.0, 0.0)], t);
    left
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
    let n: usize = args
        .get(6)
        .map_or(Ok(8), |s| s.parse())
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
    let like = {
        let mut v = V::with_shape(&lat, 24, ls, 1, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, 1).as_slice());
        v
    };
    let (ritz, theta) = load(&checkpoint, &like)?;
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; P from the lowest {n} Ritz vectors (σ {:.2e} … {:.2e}), t = the next four",
        theta[0].sqrt(),
        theta[n - 1].sqrt()
    );
    let vectors = ritz.select(&(0..n).collect::<Vec<_>>());
    let ts: Vec<V> = (n..n + 4).map(|j| ritz.select(&[j])).collect();
    let mut a = PvPreconditioned::new(&op, &like);
    let apply = |a: &mut PvPreconditioned<'_, f64>, v: &V| {
        let mut w = v.zeros_like();
        a.mult(&mut w, v);
        w
    };
    let halves = |x: &V| -> [V; 2] {
        let mut g = x.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g, x);
        let mut plus = x.zeros_like();
        plus.copy_from(x);
        for (v, g) in plus.as_mut_slice().iter_mut().zip(g.as_mut_slice()) {
            (*v, *g) = (0.5 * (*v + *g), 0.5 * (*v - *g));
        }
        [plus, g]
    };
    let right = halves(&vectors);
    let whole = vec![(0..ls).collect::<Vec<_>>()];

    // 1. The check.
    let b0 = [2, 3, 4, 6]
        .into_iter()
        .find(|b| *b > 2 && l.is_multiple_of(*b))
        .unwrap_or(2);
    {
        let agg = Aggregation::new(&lat, [b0; ND]);
        let mut g5 = vectors.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
        let p = Basis::from_vectors(&agg, &vectors, &g5)?;
        let mut u = apply(&mut a, &vectors);
        let mut un = vec![0.0; n];
        u.norm_sqr(&mut un);
        u.scale(&un.iter().map(|x| 1.0 / x.sqrt()).collect::<Vec<_>>());
        let mut g5u = u.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5u, &u);
        let q = Basis::from_vectors(&agg, &u, &g5u)?;
        let t = &ts[0];
        let mut c = p.coarse_field(1);
        p.restrict(&mut c, t);
        let mut by_basis = like.zeros_like();
        p.prolong(&mut by_basis, &c);
        let mut mine = project(&agg, &right, n, ls, &whole, t);
        let held = norm(&mine);
        mine.axpy(&[C64::new(-1.0, 0.0)], &by_basis);
        println!(
            "check on {b0}^4: |PP†t| {held:.4}; written-out projection − Basis {:.1e}",
            norm(&mine)
        );
        // How far P†AP is from having P†t as a near-null vector.
        let mut line = String::from("  coarse residual ratio |P†A PP†t| / |P†A t| for the four t:");
        for t in &ts {
            let mut c = p.coarse_field(1);
            p.restrict(&mut c, t);
            let mut held = like.zeros_like();
            p.prolong(&mut held, &c);
            let (at, ah) = (apply(&mut a, t), apply(&mut a, &held));
            let (mut cat, mut cah) = (p.coarse_field(1), p.coarse_field(1));
            p.restrict(&mut cat, &at);
            p.restrict(&mut cah, &ah);
            let coarse_norm = |c: &C| {
                let mut n = vec![0.0; 1];
                c.norm_sqr(&mut n);
                n[0].sqrt()
            };
            line += &format!(" {:.1}", coarse_norm(&cah) / coarse_norm(&cat));
        }
        println!("{line}");
        let mut line = String::from("  t_in = PP†t: what each exact correction leaves of it:");
        for (kind, name) in ["R = P†", "R = Q†", "R = P†A†"].iter().enumerate() {
            let left = corrected(&op, [&p, &q], kind, &by_basis);
            line += &format!("  {name} {:.1e}", norm(&left) / norm(&by_basis));
        }
        println!("{line}");
    }

    // 2. Block boundaries.
    println!("  |A PP†t| / |A t| and the share of |A PP†t|² on sites at distance d from the nearest block face (share of sites):");
    for b in [2, 3, 4, 6].into_iter().filter(|b| l.is_multiple_of(*b)) {
        let agg = Aggregation::new(&lat, [b; ND]);
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
        let mut line = format!("    {b}^4:");
        let mut shares = vec![0.0; dmax + 1];
        for t in &ts {
            let held = project(&agg, &right, n, ls, &whole, t);
            let (at, ah) = (apply(&mut a, t), apply(&mut a, &held));
            line += &format!(" {:.1}", norm(&ah) / norm(&at));
            let total = norm(&ah).powi(2);
            for site in 0..lat.num_sites() {
                let s: f64 = ah.site(0, site).iter().map(|x| x * x).sum();
                shares[dist(site)] += s / total / ts.len() as f64;
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

    // 3. The fifth dimension.
    let agg = Aggregation::new(&lat, [b0; ND]);
    println!("  on {b0}^4, P split along s: |PP†t|, |A PP†t| / |A t| (four t)");
    for (name, groups) in partitions(ls) {
        let mut line = format!("    {name:<9} ({:2} per aggregate):", 2 * n * groups.len());
        for t in &ts {
            let held = project(&agg, &right, n, ls, &groups, t);
            let (at, ah) = (apply(&mut a, t), apply(&mut a, &held));
            line += &format!("  {:.3}, {:.1}", norm(&held), norm(&ah) / norm(&at));
        }
        println!("{line}");
    }
    println!("  share over s of |t|², |A t|², |A PP†t|² (whole, mean of the four t):");
    let mut share = vec![[0.0f64; 3]; ls];
    for t in &ts {
        let held = project(&agg, &right, n, ls, &whole, t);
        let (at, ah) = (apply(&mut a, t), apply(&mut a, &held));
        for (k, v) in [t, &at, &ah].into_iter().enumerate() {
            let total = norm(v).powi(2);
            for site in 0..lat.num_sites() {
                let x = v.site(0, site);
                for s in 0..ls {
                    let e: f64 = x[s * 24..(s + 1) * 24].iter().map(|y| y * y).sum();
                    share[s][k] += e / total / ts.len() as f64;
                }
            }
        }
    }
    for (s, [a, b, c]) in share.iter().enumerate() {
        println!("    s = {s}: {a:.3} {b:.3} {c:.3}");
    }
    Ok(())
}
