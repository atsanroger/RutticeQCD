//! A factorized coarse correction for A′ = X†Y: both factors projected separately, each
//! a compression of a Γ5-Hermitian operator, with no solve on the fine lattice.
//!
//! ```text
//! cargo run --release --example mdwf_factorized -- CONFIG L M5 MASS CHECKPOINT [LS] [BLOCK]
//! ```
//!
//! A′ = C_PV†⁻¹ D_PV† D C⁻¹, so A′ e = r is D_PV† D (C⁻¹ e) = C_PV† r. With both D and D_PV
//! Γ5-Hermitian (Γ5 = γ5 R) and P split by Γ5, the coarse correction is
//!
//!   (P†D_PV†P)(P†DP) c = P† C_PV† r,   e = C P c,
//!
//! two nearest-neighbour coarse operators built from the hops of D and D_PV. `mdwf_xgain`
//! showed that the low modes of A′ are small through both D (|Ds|/|s| ≈ 0.22) and X†
//! (gain ≈ 0.07), so a coarse grid for D alone needs X⁻† on the fine lattice, which is as
//! hard as the solve.
//!
//! Printed for t = Ritz vectors 8..11 of a `mdwf_coherence_eig` checkpoint, the low-mode
//! content |X†e| (X: all Ritz vectors) after smoothing alone (GMRES(4) on A′), after the
//! correction, and after the correction and smoothing; for P built from s = C⁻¹v (N = 8, 16)
//! and from s together with D s (8 + 8), on BLOCK^4 (default 3).

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfAt, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp};
use ruttice::solver::{Fgmres, Solve, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, LinearOp, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_factorized CONFIG L M5 MASS CHECKPOINT [LS] [BLOCK]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

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

/// The adjoint of a coarse operator as a linear map.
struct Adjoint<'a>(&'a mut CoarseOp<f64>);

impl LinearFn<C> for Adjoint<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.0.mult_dag(out, inp);
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
    let like = {
        let mut v = V::with_shape(&lat, 24, ls, 1, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, 1).as_slice());
        v
    };
    let ritz = load(&checkpoint, &like)?;
    let m = ritz.num_vectors();
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; {m} Ritz vectors; aggregates {block}^4");

    let mut a = PvPreconditioned::new(&op, &like);
    let mut smoother = Solve::new(
        Fgmres::new(Stop::simple(1e-6, 4)),
        PvPreconditioned::new(&op, &like),
    );
    let low = |e: &V| -> f64 {
        let mut s = 0.0;
        for j in 0..m {
            let mut d = vec![C64::ZERO; 1];
            ritz.select(&[j]).dot(e, &mut d);
            s += d[0].norm_sqr();
        }
        s.sqrt()
    };
    let agg = Aggregation::new(&lat, [block; ND]);
    let x_space = |v: &V| {
        let mut s = v.zeros_like();
        op.mult_site_inv(&mut s, v, mass, false);
        s
    };
    let first = |n: usize| ritz.select(&(0..n).collect::<Vec<_>>());

    // The bases: s = C⁻¹v for the lowest 8 and 16, and s with D s for the lowest 8.
    let mut bases: Vec<(String, V)> = vec![
        ("s, N = 8".into(), x_space(&first(8))),
        ("s, N = 16".into(), x_space(&first(16))),
    ];
    {
        let s = x_space(&first(8));
        let mut ds = s.zeros_like();
        MdwfAt::new(&op, Variant::D, mass).mult(&mut ds, &s);
        let mut both = s.zeros_with(16);
        both.assign(&(0..8).collect::<Vec<_>>(), &s);
        both.assign(&(8..16).collect::<Vec<_>>(), &ds);
        bases.push(("s and Ds, 8 + 8".into(), both));
    }

    let stop = Stop {
        tolerance_sqr: 1e-14,
        max_iterations: 100,
        max_restarts: 30,
    };
    for (name, vectors) in &bases {
        let mut g5 = vectors.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5, vectors);
        let p = Basis::from_vectors(&agg, vectors, &g5)?;
        let mut cd = CoarseOp::build(&op, &mut op.work(), &p);
        let mut cpv = CoarseOp::build(&pv, &mut pv.work(), &p);
        println!("  P from {name} (coarse dimension {}):", cd.dim());
        for j in 8..12 {
            let t = ritz.select(&[j]);
            let mut r = t.zeros_like();
            a.mult(&mut r, &t);
            // g = P† C_PV† r; w = (P†D_PV†P)⁻¹ g; c = (P†DP)⁻¹ w; e = t − C P c.
            let mut cr = r.zeros_like();
            op.mult_site(&mut cr, &r, 1.0, true);
            let mut g = p.coarse_field(1);
            p.restrict(&mut g, &cr);
            let mut w = g.zeros_like();
            let s1 = Fgmres::new(stop).solve(&mut Adjoint(&mut cpv), None, &mut w, &g);
            let mut c = w.zeros_like();
            let s2 = Fgmres::new(stop).solve(&mut cd, None, &mut c, &w);
            let mut pc = like.zeros_like();
            p.prolong(&mut pc, &c);
            let mut cpc = pc.zeros_like();
            op.mult_site(&mut cpc, &pc, mass, false);
            let mut e = t.zeros_like();
            e.copy_from(&t);
            e.axpy(&[C64::new(-1.0, 0.0)], &cpc);
            // The floor: what P holds of s = C⁻¹ t.
            let s = x_space(&t);
            let mut cs = p.coarse_field(1);
            p.restrict(&mut cs, &s);
            let mut held = s.zeros_like();
            p.prolong(&mut held, &cs);
            let floor = (1.0 - (norm(&held) / norm(&s)).powi(2)).max(0.0).sqrt();
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
            let alone = smooth(&t);
            let after = smooth(&e);
            println!(
                "    j = {j}: smoothing alone {:.3}; factorized coarse {:.3}{} → with smoothing {:.3} (floor {floor:.3})",
                low(&alone),
                low(&e),
                if s1.is_err() || s2.is_err() { "*" } else { "" },
                low(&after)
            );
        }
    }
    Ok(())
}
