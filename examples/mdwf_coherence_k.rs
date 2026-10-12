//! Local coherence of the low modes the coarse space of Kanamori's domain-wall multigrid
//! (notes of 2023-11-20, §3.2) has to hold, measured as `mdwf_coherence` does for A.
//!
//! ```text
//! cargo run --release --example mdwf_coherence_k -- CONFIG L M5 MASS [LS] [DEGREE] [PV_TOL]
//! ```
//!
//! Kanamori's coarse operator is right preconditioned, R D (M_PV)_SAP P with the correction
//! x += M_PV P x_c (eqs. 4 and 12), M_PV ≃ D_PV⁻¹. For the error, D e = r with e = D_PV⁻¹ y
//! is K y = r with K = D D_PV⁻¹, and the coarse correction approximates y by P x_c: P has to
//! hold the low modes of K. Here D_PV⁻¹ is exact (BiCGStab to |r|²/|b|² < PV_TOL, default
//! 1e-20), not SAP, so this measures the ideal of that structure.
//!
//! Test vectors: random vectors filtered by T_DEGREE(ℓ(K†K)) on [hi/1000, hi],
//! hi = 1.1 λ_max(K†K); low-mode errors t1, t2 filtered with twice and three times the
//! degree; the same seeds as `mdwf_coherence`. P is split by Γ5 = γ5 R over aggregates that
//! keep every s, as the multigrid does and as Kanamori's notes choose first (§4). Printed:
//! |P P† t| / |t| for each block and n, to be compared with A in `mdwf_coherence`.

use rayon::prelude::*;
use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfAt, MdwfParams, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::Aggregation;
use ruttice::solver::{largest_magnitude, BiCgStab, ChebyshevFilter, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_coherence_k CONFIG L M5 MASS [LS] [DEGREE] [PV_TOL]";

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

/// K = D D_PV⁻¹ (`normal` false) or K†K = D_PV†⁻¹ D† D D_PV⁻¹ (`normal` true), with D_PV⁻¹
/// and D_PV†⁻¹ solved by BiCGStab. Counts the solves and their iterations.
struct KOp<'a> {
    d: MdwfAt<'a, f64>,
    d_dag: MdwfAt<'a, f64>,
    pv: MdwfAt<'a, f64>,
    pv_dag: MdwfAt<'a, f64>,
    solver: BiCgStab<V>,
    normal: bool,
    u: Option<V>,
    w: Option<V>,
    solves: usize,
    iterations: usize,
}

impl<'a> KOp<'a> {
    fn new(op: &'a Mdwf<f64>, tol: f64, normal: bool) -> Self {
        let mass = op.params().mass;
        Self {
            d: MdwfAt::new(op, Variant::D, mass),
            d_dag: MdwfAt::new(op, Variant::Dagger, mass),
            pv: MdwfAt::new(op, Variant::D, 1.0),
            pv_dag: MdwfAt::new(op, Variant::Dagger, 1.0),
            solver: BiCgStab::new(Stop::simple(tol, 2000)),
            normal,
            u: None,
            w: None,
            solves: 0,
            iterations: 0,
        }
    }
}

impl LinearFn<V> for KOp<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let fresh = |f: &mut Option<V>| {
            if !f.as_ref().is_some_and(|f| f.same_shape(inp)) {
                *f = Some(inp.zeros_like());
            }
        };
        fresh(&mut self.u);
        fresh(&mut self.w);
        let (u, w) = (self.u.as_mut().unwrap(), self.w.as_mut().unwrap());
        let mut solve = |a: &mut MdwfAt<'_, f64>, x: &mut V, b: &V| {
            x.scale(&vec![0.0; b.num_vectors()]);
            let info = self
                .solver
                .solve(a, None, x, b)
                .unwrap_or_else(|e| panic!("D_PV solve: {e}"));
            self.solves += 1;
            self.iterations += info.iterations;
        };
        // u = D_PV⁻¹ inp, then D u.
        solve(&mut self.pv, u, inp);
        if !self.normal {
            self.d.mult(out, u);
            return;
        }
        self.d.mult(w, u);
        self.d_dag.mult(u, w);
        solve(&mut self.pv_dag, out, u);
    }
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

/// |P P† t|² for P spanned, on each (block, chirality), by the first `n` vectors of
/// `halves[chirality]` restricted there.
fn captured_sqr(agg: &Aggregation, halves: &[V; 2], n: usize, t: &V) -> f64 {
    (0..agg.coarse_lattice().num_sites())
        .into_par_iter()
        .map(|coarse_site| {
            let sites = agg.fine_sites(coarse_site);
            let reals = t.site_len();
            let dim = sites.len() * reals / 2;
            let mut sum = 0.0;
            for half in halves {
                let mut w = vec![C64::ZERO; n * dim];
                let mut tl = vec![C64::ZERO; dim];
                for (k, site) in sites.iter().enumerate() {
                    let (hs, ts) = (half.site(0, *site as usize), t.site(0, *site as usize));
                    for c in 0..reals / 2 {
                        tl[k * reals / 2 + c] = C64::new(ts[2 * c], ts[2 * c + 1]);
                        for i in 0..n {
                            let r = i * reals + 2 * c;
                            w[i * dim + k * reals / 2 + c] = C64::new(hs[r], hs[r + 1]);
                        }
                    }
                }
                orthonormalise(&mut w, n, dim);
                for i in 0..n {
                    let mut z = C64::ZERO;
                    for (a, b) in w[i * dim..(i + 1) * dim].iter().zip(&tl) {
                        z += a.conj() * *b;
                    }
                    sum += z.norm_sqr();
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
    let pv_tol: f64 = args
        .get(6)
        .map_or(Ok(1e-20), |s| s.parse())
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
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, filter degree {degree}, D_PV solves to {pv_tol:e}"
    );
    let start = Instant::now();

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let mut ktk = KOp::new(&op, pv_tol, true);
    let hi = 1.1 * largest_magnitude(&mut ktk, &mut random(1, 5), 50);
    let lo = hi / 1000.0;
    println!(
        "λ_max(K†K) ≈ {:.4}; D_PV solves: {:.1} iterations each ({:.0} s)",
        hi / 1.1,
        ktk.iterations as f64 / ktk.solves as f64,
        start.elapsed().as_secs_f64()
    );
    let filter = |x: &V, degree: usize| {
        let mut f = ChebyshevFilter::new(KOp::new(&op, pv_tol, true), degree, lo, hi, x);
        let mut y = x.zeros_like();
        f.mult(&mut y, x);
        normalise(&mut y);
        y
    };
    let mut k = KOp::new(&op, pv_tol, false);
    let mut low = |v: &V| {
        let mut w = v.zeros_like();
        k.mult(&mut w, v);
        norm(&w) / norm(v)
    };

    let max_n = 32;
    let errors = [
        ("t1", filter(&random(1, 302), 2 * degree)),
        ("t2", filter(&random(1, 303), 3 * degree)),
    ];
    for (name, t) in &errors {
        println!(
            "{name}: |K t|/|t| {:.4} ({:.0} s)",
            low(t),
            start.elapsed().as_secs_f64()
        );
    }
    // A few vectors at a time: the filter and the solver each keep several fields of the
    // size of their input.
    let all = {
        let begin = random(max_n, 11);
        let mut all = begin.zeros_like();
        for chunk in (0..max_n).collect::<Vec<_>>().chunks(4) {
            all.assign(chunk, &filter(&begin.select(chunk), degree));
            println!(
                "test vectors {}..{} ({:.0} s)",
                chunk[0],
                chunk[chunk.len() - 1],
                start.elapsed().as_secs_f64()
            );
        }
        all
    };
    let tv = all.select(&[0]);
    println!("test vector 0: |K v|/|v| {:.4}", low(&tv));

    // The chiral halves (1 ± Γ5) v / 2, in place of v and Γ5 v.
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
    println!("  block   n   |PP†t1|/|t1|  |PP†t2|/|t2|");
    for b in blocks {
        let agg = Aggregation::new(&lat, [b; ND]);
        for n in [4, 8, 16, 32] {
            let mut line = format!("  {b}^4  {n:3}");
            for (_, t) in &errors {
                let c = captured_sqr(&agg, &halves, n, t).sqrt() / norm(t);
                line += &format!("         {c:.3}");
            }
            println!("{line}");
        }
    }
    println!("done ({:.0} s)", start.elapsed().as_secs_f64());
    Ok(())
}
