//! Why Γ5 does not give D† for Möbius fermions (the note, Sec. "Γ5-transformation and the exact
//! adjoint"): with D = D+ − D- Y, D+ = 1 + b W, D- = c W − 1, the relative difference
//! |Γ5 D Γ5 b − D† b| / |D† b| (zero only for c = 0) and that of the exact relation
//! D- Γ5 D† = D Γ5 D-†, for a random b, at (b, c) = (1.5, 0.5), (1, 0) and (2, 1). W is taken
//! as D(1, 0) − D(0, 0), so only the public operator is used; Ls 8.
//!
//! ```text
//! cargo run --release --example mobius_gamma5_check -- L FORMAT FILE M5 MASS SMEAR
//! ```
//!
//! On the L⁴ configuration FILE; SMEAR 1 for stout 0.1 × 3.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, Variant};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::hmc::LinkField;
use ruttice::{random_spinor, Boundary, Field, Lattice, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::Path;

type F = Field<f64, Spinor>;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let l: usize = args[0].parse()?;
    let (m5, mass): (f64, f64) = (args[3].parse()?, args[4].parse()?);
    let lat = Lattice::single_node([l; ND]);
    let u = ruttice::app::read_gauge_named(&args[1], Path::new(&args[2]), &lat)?;
    let v = match args[5] == "1" {
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
        let mut op = Mdwf::<f64>::new(
            &lat,
            MdwfParams {
                ls: 8,
                m5,
                mass,
                b,
                c,
                boundary: [
                    Boundary::Periodic,
                    Boundary::Periodic,
                    Boundary::Periodic,
                    Boundary::Antiperiodic,
                ],
            },
        );
        op.set_gauge(&v).unwrap();
        op
    };
    let apply = |op: &Mdwf<f64>, x: &F, var: Variant| {
        let mut y = x.zeros_like();
        op.mult(&mut op.work(), &mut y, x, var);
        y
    };
    let g5 = |op: &Mdwf<f64>, x: &F| {
        let mut y = x.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut y, x);
        y
    };
    let lin = |a: &F, s: f64, b: &F| {
        let mut y = a.zeros_like();
        y.copy_from(a);
        y.axpy(&[C64::new(s, 0.0)], b);
        y
    };
    let norm = |x: &F| {
        let mut n = [0.0];
        x.norm_sqr(&mut n);
        n[0].sqrt()
    };
    let mut b = F::with_shape(&lat, 24, 8, 1, 1);
    b.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, 8, 77).as_slice());
    let (w1, w0) = (mk(1.0, 0.0), mk(0.0, 0.0));
    // W x = D(1,0) x - D(0,0) x; W† likewise with the daggers.
    let wx = |x: &F, var: Variant| lin(&apply(&w1, x, var), -1.0, &apply(&w0, x, var));
    for (bb, cc) in [(1.5, 0.5), (1.0, 0.0), (2.0, 1.0)] {
        let d = mk(bb, cc);
        let ddag = apply(&d, &b, Variant::Dagger);
        let g = g5(&d, &apply(&d, &g5(&d, &b), Variant::D));
        // -D- x = x - c W x and -D-† x = x - c W† x: the sign cancels in the relation.
        let dm = |x: &F| lin(x, -cc, &wx(x, Variant::D));
        let dmd = |x: &F| lin(x, -cc, &wx(x, Variant::Dagger));
        let left = dm(&g5(&d, &ddag));
        let right = apply(&d, &g5(&d, &dmd(&b)), Variant::D);
        println!(
            "b {bb} c {cc}: |G5 D G5 b - D†b|/|D†b| = {:.2e};  |D- G5 D† b - D G5 D-† b|/|D G5 D-† b| = {:.2e}",
            norm(&lin(&g, -1.0, &ddag)) / norm(&ddag),
            norm(&lin(&left, -1.0, &right)) / norm(&right)
        );
    }
    Ok(())
}
