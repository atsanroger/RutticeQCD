//! Times the build of the coarse operator of the domain-wall multigrid on the CPU
//! (`CoarseOp::build`, most of what `PvEoCycle::new` costs there), and prints a norm of D_c v
//! to check that a change of the build leaves the operator alone.
//!
//! ```text
//! cargo run --release --example coarse_build_bench -- [L] [VECTORS] [REPS]
//! ```
//!
//! Möbius domain-wall fermions (Ls = 8, M5 = 1.0, b = 1.5, c = 0.5) on a random gauge field of
//! L⁴ (default 8), aggregates of 4⁴, P made by `pv_eo_basis` from VECTORS (default 16) random
//! vectors on the even sites (coarse dimension 4 VECTORS); REPS (default 2) builds.

use ruttice::fermion::{FermionOp, Mdwf, MdwfParams};
use ruttice::gauge::random_gauge;
use ruttice::solver::preconditioner::multigrid::{pv_eo_basis, random_even, Aggregation};
use ruttice::{Boundary, Lattice, LinearFn, Vector, ND};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (l, vectors, reps) = (arg(0, 8), arg(1, 16), arg(2, 2));
    let lat = Lattice::single_node([l; ND]);
    let u = random_gauge(&lat, 7);
    let params = MdwfParams {
        ls: 8,
        m5: 1.0,
        mass: 0.01,
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
    op.set_gauge(&u).expect("gauge");
    let mut pv = Mdwf::<f64>::new(
        &lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    );
    pv.set_gauge(&u).expect("gauge");
    let index = op
        .even_odd_index()
        .expect("an even-odd decomposition")
        .clone();
    let y = random_even::<f64>(&lat, &index, params.ls, vectors, 11);
    let agg = Aggregation::new(&lat, [4; ND]);
    let basis = pv_eo_basis(&op, &pv, &agg, &y).expect("the basis");

    let mut seconds = Vec::new();
    let mut coarse = None;
    for _ in 0..reps {
        let start = Instant::now();
        coarse = Some(basis.coarse_op(&op, &mut op.work()));
        seconds.push(start.elapsed().as_secs_f64());
    }
    let mut c = coarse.expect("at least one build");
    let mut v = basis.coarse_field(1);
    for (k, x) in v.as_mut_slice().iter_mut().enumerate() {
        *x = ((k * 7 + 3) as f64 * 0.37).sin();
    }
    let mut w = v.zeros_like();
    c.mult(&mut w, &v);
    let mut norm = [0.0];
    w.norm_sqr(&mut norm);
    let times: Vec<String> = seconds.iter().map(|s| format!("{s:.2}")).collect();
    println!(
        "{l}^4 in 4^4 blocks, Ls {}, coarse dim {}: CoarseOp::build {} s, |D_c v|² = {:.15e}",
        params.ls,
        c.dim(),
        times.join(", "),
        norm[0]
    );
}
