//! Local coherence of the low modes of A when the aggregates are also split along the
//! fifth dimension: does cutting s help more than adding test vectors?
//!
//! ```text
//! cargo run --release --example mdwf_coherence_s -- CONFIG L M5 MASS [LS] [DEGREE] [MAX_N]
//! ```
//!
//! MAX_N (default 32) is the largest number of test vectors; on 16^4 with Ls 8 one vector
//! is 100 MB and 7 GB hold at most 16.
//!
//! Same vectors as `mdwf_coherence` (same seeds): test vectors filtered by
//! T_DEGREE(ℓ(A†A)), low-mode errors t1, t2 filtered with twice and three times the degree.
//! The multigrid's aggregates keep every s; here each block and chirality is further cut
//! into groups of s. Γ5 = γ5 R maps s to Ls−1−s, so a group must be closed under that
//! reflection for the chirality split to stay exact: the groups are made of distances to
//! the nearer wall, d = min(s, Ls−1−s). Each (block, chirality, group) gets its own copy of
//! the n test vectors restricted to it, orthonormalised there.
//!
//! Printed: |P P† t| / |t| for each block, n and partition of d, with the coarse dimension
//! per block (2 n × number of groups). The partition `whole` is the multigrid's own and
//! reproduces `mdwf_coherence`. Rows with the same coarse dimension per block compare
//! cutting s with adding vectors at the same cost.

use rayon::prelude::*;
use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::Aggregation;
use ruttice::solver::{largest_magnitude, ChebyshevFilter};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Normal, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_coherence_s CONFIG L M5 MASS [LS] [DEGREE] [MAX_N]";

type V = Field<f64, Spinor>;

fn normalise(v: &mut V) {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    let s: Vec<f64> = n.iter().map(|x| 1.0 / x.sqrt()).collect();
    v.scale(&s);
}

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

/// Partitions of s into groups closed under s ↔ Ls−1−s, named by the distances to the
/// wall each group holds.
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

/// Modified Gram-Schmidt on `n` vectors of length `dim` stored one after another; vectors
/// that are (numerically) in the span of the earlier ones are zeroed, not kept.
fn orthonormalise(w: &mut [C64], n: usize, dim: usize) {
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

/// |P P† t|² for P spanned, on each (block, chirality, group of s), by the first `n`
/// vectors of `halves[chirality]` restricted there.
fn captured_sqr(
    agg: &Aggregation,
    halves: &[V; 2],
    n: usize,
    ls: usize,
    groups: &[Vec<usize>],
    t: &V,
) -> f64 {
    let total = halves[0].num_vectors();
    (0..agg.coarse_lattice().num_sites())
        .into_par_iter()
        .map(|coarse_site| {
            let sites = agg.fine_sites(coarse_site);
            let mut sum = 0.0;
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
                    debug_assert!(n <= total);
                    orthonormalise(&mut w, n, dim);
                    for i in 0..n {
                        let mut z = C64::ZERO;
                        for (a, b) in w[i * dim..(i + 1) * dim].iter().zip(&tl) {
                            z += a.conj() * *b;
                        }
                        sum += z.norm_sqr();
                    }
                }
            }
            sum
        })
        .sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=7).contains(&args.len()) {
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
    let degree: usize = args
        .get(5)
        .map_or(Ok(100), |s| s.parse())
        .map_err(|_| USAGE)?;
    let max_n: usize = args
        .get(6)
        .map_or(Ok(32), |s| s.parse())
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
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, filter degree {degree}");

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
    let lo = hi / 1000.0;
    let filter = |x: &V, degree: usize| {
        let mut f = ChebyshevFilter::new(
            Normal::new(PvPreconditioned::new(&op, x), x),
            degree,
            lo,
            hi,
            x,
        );
        let mut y = x.zeros_like();
        f.mult(&mut y, x);
        normalise(&mut y);
        y
    };
    let mut a = PvPreconditioned::new(&op, &like);
    let low = |a: &mut PvPreconditioned<'_, f64>, v: &V| {
        let mut w = v.zeros_like();
        a.mult(&mut w, v);
        norm(&w) / norm(v)
    };

    // Filtered a few vectors at a time and in place: the filter keeps several fields of the
    // size of its input, and 32 vectors at once do not fit into 7 GB on 12^4. Each vector is
    // filtered by itself either way, so the result is that of `mdwf_coherence`.
    let mut all = random(max_n, 11);
    for chunk in (0..max_n).collect::<Vec<_>>().chunks(4) {
        let filtered = filter(&all.select(chunk), degree);
        all.assign(chunk, &filtered);
    }
    let errors = [
        ("t1", filter(&random(1, 302), 2 * degree)),
        ("t2", filter(&random(1, 303), 3 * degree)),
    ];
    for (name, t) in &errors {
        println!("{name}: |A t|/|t| {:.4}", low(&mut a, t));
    }

    // The chiral halves (1 ± Γ5) v / 2 of the test vectors, in place of v and Γ5 v: on
    // 12^4 with Ls 8 each field of 32 vectors is 1 GB.
    let mut g5 = all.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &all);
    let mut plus = all;
    for (v, g) in plus.as_mut_slice().iter_mut().zip(g5.as_mut_slice()) {
        (*v, *g) = (0.5 * (*v + *g), 0.5 * (*v - *g));
    }
    let halves = [plus, g5];

    let blocks: Vec<usize> = [2, 3, 4]
        .into_iter()
        .filter(|b| l.is_multiple_of(*b))
        .collect();
    let parts = partitions(ls);
    println!("  block   n  partition   coarse/block   |PP†t1|/|t1|  |PP†t2|/|t2|");
    for b in blocks {
        let agg = Aggregation::new(&lat, [b; ND]);
        for n in [4, 8, 16, 32].into_iter().filter(|n| *n <= max_n) {
            for (name, groups) in &parts {
                let smallest = groups.iter().map(|g| g.len()).min().unwrap_or(0);
                // Per chirality an aggregate has b^4 |group| 6 complex dimensions.
                if n > b.pow(4) * smallest * 6 {
                    continue;
                }
                let mut line = format!("  {b}^4  {n:3}  {name:10}  {:6}", 2 * n * groups.len());
                for (_, t) in &errors {
                    let c = captured_sqr(&agg, &halves, n, ls, groups, t).sqrt() / norm(t);
                    line += &format!("         {c:.3}");
                }
                println!("{line}");
            }
        }
    }
    Ok(())
}
