//! Times the coarse operator of the multigrid (`CoarseOp`, nearest neighbour, dense blocks)
//! on its own: GFlop/s of `mult` and `mult_dag` for a given coarse lattice and dimension.
//!
//! ```text
//! cargo run --release --example coarse_bench -- [L] [BLOCK] [N] [REPS]
//! ```
//!
//! A Wilson operator on a random gauge field of L⁴ (default 16), aggregates of BLOCK⁴
//! (default 4), N random test vectors (default 32, coarse dimension 2N). One application
//! costs 9 · (2N)² complex multiply-adds per coarse site, counted as 8 flops each. The
//! printed norms check that a change of the kernel leaves the results alone.

use ruttice::fermion::{FermionOp, HopTerms, Wilson, WilsonParams};
use ruttice::gauge::random_gauge;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp};
use ruttice::{random_spinor, Boundary, Lattice, LinearFn, LinearOp, Vector, ND};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (l, block, n, reps) = (arg(0, 16), arg(1, 4), arg(2, 32), arg(3, 20));
    let lat = Lattice::single_node([l; ND]);
    let u = random_gauge(&lat, 7);
    let mut op = Wilson::<f64>::new(
        &lat,
        WilsonParams {
            kappa: 0.12,
            boundary: [Boundary::Periodic; ND],
        },
    );
    op.set_gauge(&u).expect("gauge");
    let v = random_spinor(&lat, n, 11);
    let mut g5 = v.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &v);
    let agg = Aggregation::new(&lat, [block; ND]);
    let basis = Basis::<f64>::from_vectors(&agg, &v, &g5).expect("basis");
    let mut c = CoarseOp::build(&op, &mut op.work(), &basis);
    let dim = c.dim();
    let sites = c.lattice().num_sites();
    let mut x = basis.coarse_field(1);
    for (k, a) in x.as_mut_slice().iter_mut().enumerate() {
        *a = ((k * 7 + 3) as f64 * 0.37).sin();
    }
    let mut y = x.zeros_like();
    let flops = (sites * 9 * dim * dim * 8) as f64;
    for (name, dagger) in [("mult", false), ("mult_dag", true)] {
        // One warm-up application, then the timed ones.
        if dagger {
            c.mult_dag(&mut y, &x);
        } else {
            c.mult(&mut y, &x);
        }
        let start = Instant::now();
        for _ in 0..reps {
            if dagger {
                c.mult_dag(&mut y, &x);
            } else {
                c.mult(&mut y, &x);
            }
        }
        let t = start.elapsed().as_secs_f64() / reps as f64;
        let mut norm = [0.0];
        y.norm_sqr(&mut norm);
        println!(
            "{l}^4 / {block}^4 = {sites} coarse sites, dim {dim}: {name:<8} {:8.3} ms, {:6.2} GFlop/s, |y|² = {:.12e}",
            1e3 * t,
            flops / t / 1e9,
            norm[0]
        );
    }
}
