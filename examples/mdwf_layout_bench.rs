//! Time the parts of the even-odd Möbius domain-wall operator in the usual layout and in
//! blocks of W sites (SIMD): the Wilson hop of the kernel alone, the even-odd hop of D
//! (the fifth-dimension combination and the kernel), C and C⁻¹ on one parity, and the
//! whole Ã = C_PV†⁻¹ S_PV† S C⁻¹.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example mdwf_layout_bench -- [L] [Ls] [vectors]
//! ```

use ruttice::fermion::{
    Dagger, EoForm, EvenOdd, FermionOp, Mdwf, MdwfLayout, MdwfParams, PvEvenOdd, Wilson,
    WilsonParams, WilsonSimd,
};
use ruttice::gauge::random_gauge;
use ruttice::kernel::{Dirac, LaneOf};
use ruttice::{
    extract_parity, random_spinor, Aos, Aosoa, Boundary, ConvertFrom, Field, Gauge, Half, Lattice,
    Layout, Lexical, LinearFn, Parity, Spinor, Vector, ND,
};
use std::sync::Arc;
use std::time::Instant;

/// Seconds per call of `f`, the best of three runs of `reps` calls.
fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    (0..3)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() / reps as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn params(mass: f64, ls: usize) -> MdwfParams {
    MdwfParams {
        ls,
        m5: 1.0,
        mass,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    }
}

/// The times of one layout: Wilson hop, D hop, C, C⁻¹, Ã (milliseconds).
fn run<L: MdwfLayout<f64>>(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    psi: &Field<f64, Spinor, Lexical, L>,
    wilson: impl FnOnce(&Field<f64, Spinor, Half, L>, &mut Field<f64, Spinor, Half, L>) -> f64,
    reps: usize,
) -> [f64; 5] {
    let ls = psi.ls();
    let make = |mass| {
        let mut op = Mdwf::<f64, Dirac, L>::in_layout(lat, params(mass, ls)).unwrap();
        op.set_gauge(u).unwrap();
        op
    };
    let (op, pv) = (make(0.01), make(1.0));
    let index = op.even_odd_index().unwrap().clone();
    let mut even = Field::<f64, Spinor, Half, L>::with_shape(lat, 24, ls, psi.num_vectors(), 1);
    extract_parity(&mut even, psi, &index, Parity::Even);
    let mut odd = even.zeros_like();
    odd.set_parity(Parity::Odd);
    let mut out = even.zeros_like();
    let mut work = op.even_odd_work();
    let w = wilson(&even, &mut odd);
    let hop = time(reps, || {
        op.hop(&mut work, &mut odd, &even, Parity::Odd, Dagger::No)
    });
    let site = time(reps, || op.mult_site(&mut out, &even, 0.01, false));
    let site_inv = time(reps, || op.mult_site_inv(&mut out, &even, 0.01, false));
    let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &even);
    let whole = time(reps, || a.mult(&mut out, &even));
    [w, hop, site, site_inv, whole].map(|t| 1e3 * t)
}

fn wilson_hop<L: Layout, F: EvenOdd<Field<f64, Spinor, Half, L>>>(
    op: &F,
    even: &Field<f64, Spinor, Half, L>,
    odd: &mut Field<f64, Spinor, Half, L>,
    reps: usize,
) -> f64 {
    let mut work = op.even_odd_work();
    time(reps, || {
        op.hop(&mut work, odd, even, Parity::Odd, Dagger::No)
    })
}

fn blocks<const W: usize>(
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    psi: &Field<f64, Spinor>,
    reps: usize,
) -> [f64; 5]
where
    f64: LaneOf<W>,
{
    let mut b = Field::<f64, Spinor, Lexical, Aosoa<W>>::with_shape(
        lat,
        24,
        psi.ls(),
        psi.num_vectors(),
        1,
    );
    b.convert_from(psi);
    let kernel = WilsonParams {
        kappa: 0.25,
        boundary: params(0.0, 1).boundary,
    };
    let mut w = WilsonSimd::<f64, W>::new(lat, kernel).unwrap();
    w.set_gauge(u).unwrap();
    run::<Aosoa<W>>(lat, u, &b, |e, o| wilson_hop(&w, e, o, reps), reps)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let l = args.first().copied().unwrap_or(12);
    let ls = args.get(1).copied().unwrap_or(8);
    let nv = args.get(2).copied().unwrap_or(1);
    let lat = Lattice::single_node([l; ND]);
    let u = random_gauge(&lat, 7);
    let flat = random_spinor(&lat, ls * nv, 8);
    let mut psi = Field::<f64, Spinor>::with_shape(&lat, 24, ls, nv, 1);
    psi.as_mut_slice().copy_from_slice(flat.as_slice());
    let reps = 20;
    let kernel = WilsonParams {
        kappa: 0.25,
        boundary: params(0.0, 1).boundary,
    };
    let mut w = Wilson::<f64>::new(&lat, kernel);
    w.set_gauge(&u).unwrap();
    println!("{l}^4, Ls = {ls}, {nv} vector(s), ms per call (best of 3 × {reps}):");
    println!("layout      Wilson hop   D hop      C          C^-1       A~");
    let show = |name: &str, t: [f64; 5]| {
        println!(
            "{name:<10} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            t[0], t[1], t[2], t[3], t[4]
        )
    };
    show(
        "sites",
        run::<Aos>(&lat, &u, &psi, |e, o| wilson_hop(&w, e, o, reps), reps),
    );
    show("blocks 2", blocks::<2>(&lat, &u, &psi, reps));
    show("blocks 4", blocks::<4>(&lat, &u, &psi, reps));
    show("blocks 8", blocks::<8>(&lat, &u, &psi, reps));
}
