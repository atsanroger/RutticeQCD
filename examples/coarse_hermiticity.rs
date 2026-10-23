//! How far the coarse operators of the domain-wall multigrid are from Σ-Hermitian, Σ the coarse
//! Γ5 (+1 on the first half of the components of a coarse site, −1 on the second): for X = P†DP
//! and P†D_PV P,
//!
//!   δ(v) = |Σ X Σ v − X† v| / |X† v|,
//!
//! X† the exact adjoint P†X†P. δ = 0 for every v when Γ5 D Γ5 = D† (Shamir, c = 0); for Möbius
//! fermions with c ≠ 0 it is not. MINRES on Σ X (the Pauli-Villars factor) and the
//! absolute-value block preconditioner assume δ = 0. Printed for four random v and for the right
//! singular vectors of the four smallest singular values of X (Lanczos on X†X), where the
//! coarse solves spend their iterations.
//!
//! ```text
//! cargo run --release --example coarse_hermiticity -- L FORMAT FILE M5 MASS SMEAR [--c C]
//!     [--block B] [--vectors N] [--t T]
//! ```
//!
//! Ls 8, b = 1 + c (b − c = 1 as for the usual b 1.5, c 0.5), c 0.5 by default; SMEAR 1 for
//! stout 0.1 × 3. The basis as in the solver: 16 vectors filtered in 2 rounds, the bootstrap
//! 2 × 4, blocks of 4⁴, 32 test vectors (`--vectors` low modes, split by Γ5).

use ruttice::eigen::{lowest, LanczosParams, Order};
use ruttice::fermion::{FermionOp, Mdwf, MdwfParams};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseOp, LowModeParams,
    PvEoParams,
};
use ruttice::{Boundary, Coarse, Field, Lattice, LinearFn, Vector, ND};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

type V = Field<f64, Coarse>;

/// X†X.
struct NormalOf<'a>(&'a mut CoarseOp<f64>, V);

impl LinearFn<V> for NormalOf<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let NormalOf(x, t) = self;
        if !t.same_shape(inp) {
            *t = inp.zeros_like();
        }
        x.mult(t, inp);
        x.mult_adjoint(out, t);
    }
}

fn norm(v: &V) -> f64 {
    let mut n = [0.0];
    v.norm_sqr(&mut n);
    n[0].sqrt()
}

/// δ(v) = |Σ X Σ v − X† v| / |X† v|.
fn defect(x: &mut CoarseOp<f64>, v: &V) -> f64 {
    let mut sv = v.zeros_like();
    sv.copy_from(v);
    x.gamma5(&mut sv);
    let mut xsv = v.zeros_like();
    x.mult(&mut xsv, &sv);
    x.gamma5(&mut xsv);
    let mut adj = v.zeros_like();
    x.mult_adjoint(&mut adj, v);
    let a = norm(&adj);
    xsv.axpy(&[ruttice::C64::new(-1.0, 0.0)], &adj);
    norm(&xsv) / a
}

fn random_like(like: &V, seed: u64) -> V {
    let mut v = like.zeros_like();
    let mut rng = ruttice::TestRng::new(seed);
    for r in v.as_mut_slice() {
        *r = rng.uniform();
    }
    v
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        return Err("L FORMAT FILE M5 MASS SMEAR [--c C] [--block B] [--vectors N] [--t T]".into());
    }
    let value = |name: &str, default: f64| -> Result<f64, Box<dyn Error>> {
        match args.iter().position(|a| a == name) {
            Some(i) => Ok(args.get(i + 1).ok_or("a value")?.parse()?),
            None => Ok(default),
        }
    };
    let l: usize = args[0].parse()?;
    let (m5, mass): (f64, f64) = (args[3].parse()?, args[4].parse()?);
    let smear = args[5] == "1";
    let c = value("--c", 0.5)?;
    let block = value("--block", 4.0)? as usize;
    let vectors = value("--vectors", 16.0)? as usize;
    let nt = value("--t", l as f64)? as usize;
    let start = Instant::now();
    let lat = Lattice::single_node([l, l, l, nt]);
    let mut u = ruttice::app::read_gauge_named(&args[1], Path::new(&args[2]), &lat)?;
    if smear {
        let s = GaugeSmearing {
            smear: Smear::Stout(Stout { rho: uniform(0.1) }),
            steps: 3,
        };
        u = s.levels(&u).pop().expect("three steps");
    }
    let params = |m: f64| MdwfParams {
        ls: 8,
        m5,
        mass: m,
        b: 1.0 + c,
        c,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut op = Mdwf::<f64>::new(&lat, params(mass));
    op.set_gauge(&u)?;
    let mut pv = Mdwf::<f64>::new(&lat, params(1.0));
    pv.set_gauge(&u)?;
    let cycle = PvEoParams {
        chebyshev_steps: value("--steps", 8.0)? as usize,
        chebyshev_ratio: value("--ratio", 10.0)?,
        ..PvEoParams::default()
    };
    let (y, _) = pv_eo_low_modes(&op, &pv, &lat, LowModeParams::default());
    let mut y = y.select(&(0..vectors).collect::<Vec<_>>());
    let agg = Aggregation::new(&lat, [block; ND]);
    let basis = pv_eo_bootstrap(
        &op,
        &pv,
        &agg,
        &mut y,
        BootstrapParams {
            cycle,
            ..BootstrapParams::default()
        },
    )?;
    println!(
        "# {l}^3x{nt} {}; M5 {m5}, m {mass}, b {}, c {c}, stout 0.1x3: {smear}; blocks {block}^4, {} coarse components per site; setup {:.0} s",
        args[2],
        1.0 + c,
        basis.coarse_dim(),
        start.elapsed().as_secs_f64()
    );
    let like = basis.coarse_field(1);
    for (name, fine) in [("P^dag D P", &op), ("P^dag D_PV P", &pv)] {
        let mut x = basis.coarse_op(fine, &mut fine.work());
        let random: Vec<f64> = (1..=4)
            .map(|s| defect(&mut x, &random_like(&like, s)))
            .collect();
        println!(
            "{name}: random v: delta = {}",
            random
                .iter()
                .map(|d| format!("{d:.3e}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let start_vector = random_like(&like, 99);
        let e = lowest(
            &mut NormalOf(&mut x, like.zeros_like()),
            &start_vector,
            LanczosParams {
                wanted: 4,
                keep: 10,
                max_dim: 32,
                tolerance: 1e-8.into(),
                max_restarts: 2000,
                chebyshev: None,
                order: Order::Ascending,
                threshold: None,
            },
        );
        for (k, (s2, v)) in e.values.iter().zip(&e.vectors).enumerate() {
            println!(
                "{name}: singular vector {k}: sigma = {:.4e}, delta = {:.3e}  (Lanczos converged {}, {} restarts)",
                s2.max(0.0).sqrt(),
                defect(&mut x, v),
                e.converged,
                e.restarts
            );
        }
    }
    Ok(())
}
