//! Time gauge smearing: stout and APE steps, and the chain rule of stout smearing (what HMC
//! needs at every force evaluation).
//!
//!   cargo run --release --example smear_bench -- [L] [steps]

use ruttice::gauge::smearing::{Ape, Stout};
use ruttice::gauge::{random_gauge, GaugeSmearing, Smear};
use ruttice::{Field, Gauge, Lattice};
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let l = args.first().copied().unwrap_or(16);
    let steps = args.get(1).copied().unwrap_or(3);
    let lat = Lattice::single_node([l; 4]);
    let u = random_gauge(&lat, 1);
    let stout = GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(0.1)),
        steps,
    };
    let ape = GaugeSmearing {
        smear: Smear::Ape(Ape::uniform(0.1, 1e-15, 100)),
        steps,
    };
    println!(
        "{l}^4, {steps} steps, {} threads",
        rayon::current_num_threads()
    );
    {
        use ruttice::{shift, Boundary, Direction, Stencil};
        let t = Instant::now();
        let st = Stencil::nearest(&lat, [Boundary::Periodic; 4]);
        println!("stencil: {:.1} ms", 1e3 * t.elapsed().as_secs_f64());
        let mut v = Field::<f64, Gauge>::gauge(&lat);
        let t = Instant::now();
        for mu in 0..4 {
            shift(&mut v, &u, &st, mu, Direction::Forward);
        }
        println!(
            "shift of a gauge field: {:.1} ms",
            1e3 * t.elapsed().as_secs_f64() / 4.0
        );
        for depth in [1, 2] {
            let t = Instant::now();
            let _ = ruttice::gauge::paths::Extended::new(&u, depth);
            println!(
                "extended, depth {depth}: {:.1} ms",
                1e3 * t.elapsed().as_secs_f64()
            );
        }
        let t = Instant::now();
        let _ = ruttice::gauge::weighted_staples(&u, &ruttice::gauge::smearing::uniform(0.1));
        println!("staples: {:.1} ms", 1e3 * t.elapsed().as_secs_f64());
    }
    for (name, s) in [("stout", &stout), ("APE", &ape)] {
        let t = Instant::now();
        let levels = s.levels(&u);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "{name:6} smearing: {dt:.3} s ({:.1} ms per step)",
            1e3 * dt / steps as f64
        );
        drop(levels);
    }
    let levels = stout.levels(&u);
    let sigma: Field<f64, Gauge> = random_gauge(&lat, 2);
    let t = Instant::now();
    let _ = stout.chain(sigma, &u, &levels);
    let dt = t.elapsed().as_secs_f64();
    println!(
        "stout chain rule: {dt:.3} s ({:.1} ms per step)",
        1e3 * dt / steps as f64
    );
}
