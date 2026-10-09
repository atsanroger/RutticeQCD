//! Where the small singular values of A′ = X† D C⁻¹ come from: D, or X = D_PV C_PV⁻¹?
//!
//! ```text
//! cargo run --release --example mdwf_xgain -- CONFIG L M5 MASS CHECKPOINT [LS] [COUNT]
//! ```
//!
//! For the lowest COUNT (default 16) Ritz vectors v of a `mdwf_coherence_eig` checkpoint:
//! |A′v|, |D s| with s = C⁻¹ v (relative to |s|), and the gain of X† on u = D s, which is
//! |A′v| / |D s|. A′v = X†(D s), so a gain far below 1 means the mode is small because of
//! X, the Pauli-Villars side, which does not depend on the quark mass. Also the lowest
//! singular value of X†, by inverse iteration with BiCGStab solves of X† (to 1e-20).

use ruttice::fermion::{FermionOp, Mdwf, MdwfAt, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::{BiCgStab, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, ND};
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_xgain CONFIG L M5 MASS CHECKPOINT [LS] [COUNT]";

type V = Field<f64, Spinor>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

fn load(path: &PathBuf, like: &V, count: usize) -> Result<V, Box<dyn Error>> {
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
    Ok(x.select(&(0..count.min(m)).collect::<Vec<_>>()))
}

/// X† = C_PV†⁻¹ D_PV†.
struct XDag<'a> {
    op: &'a Mdwf<f64>,
    pv_dag: MdwfAt<'a, f64>,
    w: V,
}

impl LinearFn<V> for XDag<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        if self.w.num_vectors() != inp.num_vectors() {
            self.w = inp.zeros_like();
        }
        self.pv_dag.mult(&mut self.w, inp);
        self.op.mult_site_inv(out, &self.w, 1.0, true);
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
    let count: usize = args
        .get(6)
        .map_or(Ok(16), |s| s.parse())
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
    let ritz = load(&checkpoint, &like, count)?;
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; {} Ritz vectors of {}",
        ritz.num_vectors(),
        checkpoint.display()
    );
    println!("  j    |A′v|      |Ds|/|s|   |s|/|v|   gain of X† = |A′v|/|Ds|");
    let mut a = PvPreconditioned::new(&op, &like);
    let mut d = MdwfAt::new(&op, Variant::D, mass);
    for j in 0..ritz.num_vectors() {
        let v = ritz.select(&[j]);
        let mut av = v.zeros_like();
        a.mult(&mut av, &v);
        let mut s = v.zeros_like();
        op.mult_site_inv(&mut s, &v, mass, false);
        let mut ds = v.zeros_like();
        d.mult(&mut ds, &s);
        println!(
            "  {j:2}  {:.3e}  {:.3e}  {:.3e}  {:.3e}",
            norm(&av) / norm(&v),
            norm(&ds) / norm(&s),
            norm(&s) / norm(&v),
            norm(&av) / norm(&ds)
        );
    }
    // The lowest singular value of X† by inverse iteration on X X†: y = X⁻† X⁻¹ x is two
    // solves; X X† and X† X have the same spectrum, so iterate with X† alone on X†-solves of
    // the normal form: x ← (X†)⁻¹ x twice per step through BiCGStab on X† and on X.
    let mut x = {
        let mut v = like.zeros_like();
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, 9).as_slice());
        v
    };
    let mut xdag = XDag {
        op: &op,
        pv_dag: MdwfAt::new(&op, Variant::Dagger, 1.0),
        w: like.zeros_like(),
    };
    let stop = Stop {
        tolerance_sqr: 1e-20,
        max_iterations: 5000,
        max_restarts: 3,
    };
    let mut sigma = 0.0;
    for step in 0..12 {
        let nx = norm(&x);
        x.scale(&[1.0 / nx]);
        let mut y = x.zeros_like();
        let info = BiCgStab::new(stop).solve(&mut xdag, None, &mut y, &x);
        let it = info.map(|i| i.iterations).unwrap_or(0);
        // |X⁻† x| grows like 1/σ_min of X† for the component along its lowest mode.
        sigma = 1.0 / norm(&y);
        x = y;
        println!("  inverse iteration on X†, step {step}: σ_min(X†) ≲ {sigma:.3e} ({it} BiCGStab iterations)");
    }
    println!("  σ_min(X†) ≈ {sigma:.3e}");
    Ok(())
}
