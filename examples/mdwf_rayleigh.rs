//! Does the junk of the blockwise projection lift the coarse operator's view of a low
//! mode? Rayleigh quotients of A′ on a low mode t, on its projection PP†t and on the junk
//! w = PP†t − t.
//!
//! ```text
//! cargo run --release --example mdwf_rayleigh -- CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK]
//! ```
//!
//! Hypothesis: A′ is accretive (Re⟨v, A′v⟩ > 0), so the junk w contributes a positive real
//! part to ⟨PP†t, A′ PP†t⟩ and P†A′P sees the low mode as much less low; for the Wilson
//! operator the same quotient of the Hermitian indefinite γ5 D has both signs and cancels.
//! Printed for t = Ritz vectors N..N+3 of a `mdwf_coherence_eig` checkpoint, P from the
//! lowest N (default 8) on BLOCK^4 (default 3): q(v) = ⟨v, A′v⟩/|v|² for v = t, PP†t, w, with
//! |A′t| and |w|.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, C64, ND};
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_rayleigh CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK]";

type V = Field<f64, Spinor>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

fn load(path: &PathBuf, like: &V) -> Result<V, Box<dyn Error>> {
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
    for _ in 0..m {
        let _ = next(&mut f)?;
    }
    for v in x.as_mut_slice() {
        *v = f64::from_le_bytes(next(&mut f)?);
    }
    Ok(x)
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
    let block: usize = args
        .get(7)
        .map_or(Ok(3), |s| s.parse())
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
    let ritz = load(&checkpoint, &like)?;
    let agg = Aggregation::new(&lat, [block; ND]);
    let vectors = ritz.select(&(0..n).collect::<Vec<_>>());
    let mut g5 = vectors.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
    let p = Basis::<f64>::from_vectors(&agg, &vectors, &g5)?;
    let mut a = PvPreconditioned::new(&op, &like);
    let mut q = |v: &V| -> C64 {
        let mut av = v.zeros_like();
        a.mult(&mut av, v);
        let mut d = vec![C64::ZERO; 1];
        v.dot(&av, &mut d);
        d[0].scale(1.0 / norm(v).powi(2))
    };
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; P from {n} Ritz vectors on {block}^4");
    println!(
        "  q(v) = ⟨v, A′v⟩/|v|²:   t            PP†t            w = PP†t − t      |A′t|   |w|"
    );
    for j in n..n + 4 {
        let t = ritz.select(&[j]);
        let mut c = p.coarse_field(1);
        p.restrict(&mut c, &t);
        let mut h = like.zeros_like();
        p.prolong(&mut h, &c);
        let mut w = h.zeros_like();
        w.copy_from(&h);
        w.axpy(&[C64::new(-1.0, 0.0)], &t);
        let (qt, qh, qw) = (q(&t), q(&h), q(&w));
        let mut at = t.zeros_like();
        PvPreconditioned::new(&op, &like).mult(&mut at, &t);
        let f = |z: C64| format!("{:+.2e}{:+.2e}i", z.re, z.im);
        println!(
            "  j = {j}:  {}  {}  {}  {:.2e}  {:.3}",
            f(qt),
            f(qh),
            f(qw),
            norm(&at),
            norm(&w)
        );
    }
    Ok(())
}
