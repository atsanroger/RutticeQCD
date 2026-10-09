//! A coarse correction for A′ = X†Y with a Γ5-Hermitian coarse operator.
//!
//! ```text
//! cargo run --release --example mdwf_g5galerkin -- CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK]
//! ```
//!
//! The Galerkin correction of A′ fails because the junk of the blockwise projection adds a
//! positive real part to ⟨PP†t, A′ PP†t⟩ (log: 40 times q(t)); the Wilson multigrid works
//! because P commutes with γ5 and P†DP compresses the Hermitian indefinite γ5 D, where the
//! junk has both signs. The domain-wall D is Γ5-Hermitian (Γ5 = γ5 R), and with Y = D C⁻¹,
//! X = D_PV C_PV⁻¹, the error equation A′ e = r is D (C⁻¹ e) = X⁻† r. So
//!
//!   e ≈ C P (P†DP)⁻¹ P† X⁻† r,
//!
//! with P from the x-space low modes C⁻¹ v (v the right singular vectors of A′ from a
//! `mdwf_coherence_eig` checkpoint), split by Γ5. P†DP is the compression of the Hermitian
//! Γ5 D (up to the coarse Γ5), and D has nearest-neighbour hops only. X⁻† r = D_PV⁻†(C_PV† r)
//! is solved exactly here (BiCGStab).
//!
//! Printed for t = Ritz vectors N..N+3, P from the lowest N (default 8) on BLOCK^4
//! (default 3): a check that X⁻† A′ t = D C⁻¹ t; the Rayleigh quotients of Γ5 D on s = C⁻¹t,
//! PP†s and the junk; the low-mode content |X†e| (X: all Ritz vectors) of t after smoothing
//! alone (GMRES(4) on A′), after this correction, and after this correction and smoothing,
//! against the floor √(1 − |PP†s|²/|s|²).

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfAt, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis};
use ruttice::solver::{BiCgStab, Fgmres, Solve, Solver, Stop};
use ruttice::{random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, Spinor, Vector, C64, ND};
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_g5galerkin CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK]";

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

/// P†DP through the fine lattice.
struct Galerkin<'a, 'b> {
    d: MdwfAt<'a, f64>,
    p: &'b Basis<f64>,
    f: V,
    g: V,
}

impl LinearFn<C> for Galerkin<'_, '_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.p.prolong(&mut self.f, inp);
        self.d.mult(&mut self.g, &self.f);
        self.p.restrict(out, &self.g);
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
    let m = ritz.num_vectors();
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; P from C⁻¹ of the lowest {n} of {m} Ritz vectors, {block}^4");

    let c_inv = |v: &V| {
        let mut out = v.zeros_like();
        op.mult_site_inv(&mut out, v, mass, false);
        out
    };
    let c_of = |v: &V| {
        let mut out = v.zeros_like();
        op.mult_site(&mut out, v, mass, false);
        out
    };
    let mut d = MdwfAt::new(&op, Variant::D, mass);
    let apply = |a: &mut dyn LinearFn<V>, v: &V| {
        let mut w = v.zeros_like();
        a.mult(&mut w, v);
        w
    };
    // X⁻† r = D_PV⁻† (C_PV† r).
    let x_inv_dag = |r: &V| -> V {
        let mut w = r.zeros_like();
        op.mult_site(&mut w, r, 1.0, true);
        let mut z = r.zeros_like();
        let mut pv_dag = MdwfAt::new(&op, Variant::Dagger, 1.0);
        BiCgStab::new(Stop {
            tolerance_sqr: 1e-24,
            max_iterations: 3000,
            max_restarts: 3,
        })
        .solve(&mut pv_dag, None, &mut z, &w)
        .expect("the D_PV† solve");
        z
    };
    // ⟨v, Γ5 D v⟩/|v|², real for the Γ5-Hermitian D.
    let q_g5d = |d: &mut MdwfAt<'_, f64>, v: &V| -> C64 {
        let dv = apply(d, v);
        let mut g = dv.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g, &dv);
        let mut s = vec![C64::ZERO; 1];
        v.dot(&g, &mut s);
        s[0].scale(1.0 / norm(v).powi(2))
    };

    let xs = c_inv(&ritz.select(&(0..n).collect::<Vec<_>>()));
    let mut g5 = xs.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &xs);
    let agg = Aggregation::new(&lat, [block; ND]);
    let p = Basis::from_vectors(&agg, &xs, &g5)?;

    let mut a = PvPreconditioned::new(&op, &like);
    let mut smoother = Solve::new(
        Fgmres::new(Stop::simple(1e-6, 4)),
        PvPreconditioned::new(&op, &like),
    );
    let low = |e: &V| -> f64 {
        let mut s = 0.0;
        for j in 0..m {
            let mut dd = vec![C64::ZERO; 1];
            ritz.select(&[j]).dot(e, &mut dd);
            s += dd[0].norm_sqr();
        }
        s.sqrt()
    };

    for j in n..n + 4 {
        let t = ritz.select(&[j]);
        let r = apply(&mut a, &t);
        // Check: X⁻† A′ t = D C⁻¹ t.
        let z = x_inv_dag(&r);
        let s = c_inv(&t);
        let mut check = apply(&mut d, &s);
        check.axpy(&[C64::new(-1.0, 0.0)], &z);
        // Rayleigh quotients of Γ5 D.
        let mut c = p.coarse_field(1);
        p.restrict(&mut c, &s);
        let mut held = s.zeros_like();
        p.prolong(&mut held, &c);
        let mut w = held.zeros_like();
        w.copy_from(&held);
        w.axpy(&[C64::new(-1.0, 0.0)], &s);
        let floor = (1.0 - (norm(&held) / norm(&s)).powi(2)).max(0.0).sqrt();
        let (qs, qh, qw) = (q_g5d(&mut d, &s), q_g5d(&mut d, &held), q_g5d(&mut d, &w));
        // The correction e = t − C P (P†DP)⁻¹ P† z.
        let mut rc = p.coarse_field(1);
        p.restrict(&mut rc, &z);
        let mut y = rc.zeros_like();
        let mut gal = Galerkin {
            d: MdwfAt::new(&op, Variant::D, mass),
            p: &p,
            f: like.zeros_like(),
            g: like.zeros_like(),
        };
        let solved = Fgmres::new(Stop {
            tolerance_sqr: 1e-14,
            max_iterations: 100,
            max_restarts: 20,
        })
        .solve(&mut gal, None, &mut y, &rc);
        let mut pc = like.zeros_like();
        p.prolong(&mut pc, &y);
        let mut e = t.zeros_like();
        e.copy_from(&t);
        e.axpy(&[C64::new(-1.0, 0.0)], &c_of(&pc));
        let mut smooth = |e: &V| -> V {
            let ae = apply(&mut a, e);
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
            "  j = {j}: check |X⁻†A′t − DC⁻¹t|/|DC⁻¹t| {:.1e}; q_Γ5D s {:+.2e} PP†s {:+.2e} w {:+.2e} (im {:.0e});  low-mode content: smoothing alone {:.3}, Γ5-Galerkin {:.3}{} → with smoothing {:.3} (floor {floor:.3})",
            norm(&check) / norm(&z),
            qs.re,
            qh.re,
            qw.re,
            qs.im.abs().max(qh.im.abs()).max(qw.im.abs()),
            low(&alone),
            low(&e),
            if solved.is_err() { "*" } else { "" },
            low(&after)
        );
    }
    Ok(())
}
