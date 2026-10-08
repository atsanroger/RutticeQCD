//! Local coherence of the low modes of A: how much of a low-mode error the aggregation
//! coarse space can hold, as the blocks and the number of test vectors change.
//!
//! ```text
//! cargo run --release --example mdwf_coherence -- CONFIG L M5 MASS [LS] [DEGREE]
//! ```
//!
//! Test vectors: random vectors filtered by T_DEGREE(ℓ(A†A)) on [hi/1000, hi], hi = 1.1
//! λ_max(A†A), normalised; P built from the first n of them for blocks b^4 (b dividing L),
//! split by Γ5 = γ5 R as the multigrid does. Low-mode errors: t1, t2, filtered with twice
//! the degree from other random vectors (t2 also lower, three times the degree).
//! Printed: |P P† t| / |t| for each block size and n, with the coarse dimension per
//! aggregate and chirality (n) against the aggregate's dimension (b^4 Ls 12 / 2).
//!
//! A coarse space that holds most of t with few vectors per aggregate is what makes
//! aggregation multigrid work (local coherence); a fraction that only grows like n over
//! the aggregate's dimension means the low modes are not locally coherent in this
//! aggregation.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{largest_magnitude, ChebyshevFilter};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Normal, Spinor, Vector, ND};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_coherence CONFIG L M5 MASS [LS] [DEGREE]";

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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
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

    let max_n = 32;
    let all = filter(&random(max_n, 11), degree);
    let errors = [
        ("t1", filter(&random(1, 302), 2 * degree)),
        ("t2", filter(&random(1, 303), 3 * degree)),
    ];
    for (name, t) in &errors {
        println!("{name}: |A t|/|t| {:.4}", low(&mut a, t));
    }

    let blocks: Vec<usize> = [2, 3, 4, 6].into_iter().filter(|b| l.is_multiple_of(*b)).collect();
    println!("  block  n  aggregate dim   |PP†t1|/|t1|  |PP†t2|/|t2|");
    for b in blocks {
        let agg = Aggregation::new(&lat, [b; ND]);
        let dim = b.pow(4) * ls * 12 / 2;
        for n in [8, 16, 32] {
            if n > dim / 2 {
                continue;
            }
            let vectors = all.select(&(0..n).collect::<Vec<_>>());
            let mut g5 = vectors.zeros_like();
            op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
            let p = Basis::from_vectors(&agg, &vectors, &g5)?;
            let mut line = format!("  {b}^4  {n:2}  {n:3} / {dim:6}");
            for (_, t) in &errors {
                let mut c = p.coarse_field(1);
                p.restrict(&mut c, t);
                let mut back = t.zeros_like();
                p.prolong(&mut back, &c);
                line += &format!("     {:.3}", norm(&back));
            }
            println!("{line}");
        }
    }
    Ok(())
}
