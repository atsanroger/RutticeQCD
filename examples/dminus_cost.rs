//! What the inverse of D- = c W − 1 costs on the device (the note, Sec. "The exact adjoint and
//! why it cannot be used directly"): D- and D-† by BiCGStab and GCR(8) from a random right-hand
//! side, to 1e-10 and 1e-20, at most 20000 iterations, against the time of one D. D- is the
//! Wilson operator of mass −M5 − 1/c times c, its spectrum surrounds the origin, and neither
//! converges. W is taken as D(1, 0) − D(0, 0) on the device; Ls 8, c 0.5.
//!
//! ```text
//! cargo run --release --features cuda-bridge --example dminus_cost -- X Y Z T FORMAT FILE M5 SMEAR
//! ```
//!
//! On the X·Y·Z·T configuration FILE; SMEAR 1 for stout 0.1 × 3.

use ruttice::backend::cuda_bridge::{CudaMdwf, DeviceSpinor};
use ruttice::fermion::{FermionOp, MdwfParams, Variant};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::hmc::LinkField;
use ruttice::solver::{BiCgStab, Gcr, Solver, Stop};
use ruttice::{random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, C64};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

type V = DeviceSpinor<f64>;

/// D-† x = x - c (D(1,0)† x - D(0,0)† x), or D- without `dagger`.
struct DMinus<'a> {
    w1: &'a CudaMdwf<f64>,
    w0: &'a CudaMdwf<f64>,
    c: f64,
    dagger: bool,
    t1: Option<V>,
    applies: usize,
}

impl LinearFn<V> for DMinus<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let var = if self.dagger {
            Variant::Dagger
        } else {
            Variant::D
        };
        let t1 = self.t1.get_or_insert_with(|| inp.zeros_like());
        if t1.num_vectors() != inp.num_vectors() {
            *t1 = inp.zeros_like();
        }
        self.w1.mult(&mut self.w1.work(), out, inp, var);
        self.w0.mult(&mut self.w0.work(), t1, inp, var);
        // out = inp - c (out - t1)
        out.axpy(&[C64::new(-1.0, 0.0)], t1);
        out.aypx(&[C64::new(-self.c, 0.0)], inp);
        self.applies += 1;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dims: [usize; 4] = [
        args[0].parse()?,
        args[1].parse()?,
        args[2].parse()?,
        args[3].parse()?,
    ];
    let m5: f64 = args[6].parse()?;
    let lat = Lattice::single_node(dims);
    let u = ruttice::app::read_gauge_named(&args[4], Path::new(&args[5]), &lat)?;
    let v = match args[7] == "1" {
        true => GaugeSmearing {
            smear: Smear::Stout(Stout { rho: uniform(0.1) }),
            steps: 3,
        }
        .levels(&u)
        .pop()
        .unwrap(),
        false => u.duplicate(),
    };
    let mk = |b: f64, c: f64| {
        let mut op = CudaMdwf::<f64>::new(
            &lat,
            MdwfParams {
                ls: 8,
                m5,
                mass: 0.001,
                b,
                c,
                boundary: [
                    Boundary::Periodic,
                    Boundary::Periodic,
                    Boundary::Periodic,
                    Boundary::Antiperiodic,
                ],
            },
        )
        .unwrap();
        op.set_gauge(&v).unwrap();
        op
    };
    let (d, w1, w0) = (mk(1.5, 0.5), mk(1.0, 0.0), mk(0.0, 0.0));
    let mut host = Field::<f64, Spinor>::with_shape(&lat, 24, 8, 1, 1);
    host.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, 8, 77).as_slice());
    let b = V::from_host(&host)?;
    // One application of D, timed over 20.
    let mut y = b.zeros_like();
    d.mult(&mut d.work(), &mut y, &b, Variant::D);
    let mut n = [0.0];
    y.norm_sqr(&mut n);
    let t = Instant::now();
    for _ in 0..20 {
        d.mult(&mut d.work(), &mut y, &b, Variant::D);
    }
    y.norm_sqr(&mut n);
    let t_d = t.elapsed().as_secs_f64() / 20.0;
    println!(
        "# {dims:?} M5 {m5} stout {}: one D {:.2} ms",
        args[7],
        t_d * 1e3
    );
    for dagger in [true, false] {
        for tol in [1e-10, 1e-20] {
            for (name, mut solver) in [
                (
                    "BiCGStab",
                    Box::new(BiCgStab::new(Stop::simple(tol, 20000))) as Box<dyn Solver<V>>,
                ),
                ("GCR(8)", Box::new(Gcr::new(Stop::simple(tol, 20000), 8))),
            ] {
                let mut a = DMinus {
                    w1: &w1,
                    w0: &w0,
                    c: 0.5,
                    dagger,
                    t1: None,
                    applies: 0,
                };
                let mut x = b.zeros_like();
                let t = Instant::now();
                let r = solver.solve(&mut a, None, &mut x, &b);
                x.norm_sqr(&mut n);
                let s = t.elapsed().as_secs_f64();
                match r {
                    Ok(info) => println!(
                        "{} {name} to {tol:e}: {} iterations, {} applications of D-, {:.2} s (= {:.0} D), |r|² {:.1e}",
                        if dagger { "D-†" } else { "D- " },
                        info.iterations,
                        a.applies,
                        s,
                        s / t_d,
                        info.residual_sqr[0]
                    ),
                    Err(e) => println!(
                        "{} {name} to {tol:e}: failed after {} applications ({:.2} s): {e:?}",
                        if dagger { "D-†" } else { "D- " },
                        a.applies,
                        s
                    ),
                }
            }
        }
    }
    Ok(())
}
