//! The factorised coarse correction C of the even-odd Pauli-Villars multigrid with exact coarse
//! solves, on its own: C Ã maps the fine space into the range of C, of the dimension N_c of the
//! coarse space, so its non-zero eigenvalues are those of K = U† C Ã U for an orthonormal basis
//! U of that range. For a Galerkin correction of Ã itself K would be the identity; the farther
//! the eigenvalues of K are from 1, the more the correction amplifies (|1 − λ| > 1) what it
//! should remove. K is written for the second factor solved on its Γ5-Hermitian form (MINRES)
//! and on its exact adjoint (BiCGStab), both to rounding.
//!
//! ```text
//! cargo run --release --example correction_spectrum -- M5 MASS SMEAR(0|1) OUT_PREFIX
//! ```
//! on Bridge++'s 4³×8 configuration (`examples/conf_04040408.txt`), Ls = 8, b = 1.5, c = 0.5.

use ruttice::fermion::{EoForm, FermionOp, Mdwf, MdwfParams, PvEvenOdd};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseKrylov, LowModeParams,
    PvEoCycle, PvEoParams,
};
use ruttice::{
    extract_parity, random_spinor, Boundary, Field, Half, Lattice, LinearFn, Parity, Spinor,
    Vector, C64,
};
use std::error::Error;
use std::io::Write;
use std::path::Path;

type H = Field<f64, Spinor, Half>;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m5: f64 = args.first().ok_or("M5")?.parse()?;
    let mass: f64 = args.get(1).ok_or("MASS")?.parse()?;
    let smear = args.get(2).ok_or("SMEAR")? == "1";
    let prefix = args.get(3).ok_or("OUT_PREFIX")?;
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut u = read_bridge_text(Path::new("examples/conf_04040408.txt"), &lat)?;
    if smear {
        let s = GaugeSmearing {
            smear: Smear::Stout(Stout { rho: uniform(0.1) }),
            steps: 3,
        };
        u = s.levels(&u).pop().expect("three steps");
    }
    let ls = 8;
    let params = |m: f64| MdwfParams {
        ls,
        m5,
        mass: m,
        b: 1.5,
        c: 0.5,
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

    // The test vectors as the program makes them: the filter, then the bootstrap.
    let (y, theta) = pv_eo_low_modes(
        &op,
        &pv,
        &lat,
        LowModeParams {
            vectors: 16,
            rounds: 2,
            chunk: 4,
            seed: 21,
        },
    );
    eprintln!(
        "Ritz values of Ã†Ã {:.3e} .. {:.3e}",
        theta[0],
        theta[theta.len() - 1]
    );
    let mut y = y.select(&(0..8).collect::<Vec<_>>());
    let agg = Aggregation::new(&lat, [2, 2, 2, 4]);
    let basis = pv_eo_bootstrap(&op, &pv, &agg, &mut y, BootstrapParams::default())?;
    let like = y.select(&[0]);
    let index = op.even_odd_index().expect("even-odd").clone();

    for (name, pv_solver) in [
        ("minres", CoarseKrylov::Minres),
        ("adjoint", CoarseKrylov::BiCgStab),
    ] {
        let exact = PvEoParams {
            coarse_tolerance_sqr: 1e-22,
            pv_tolerance_sqr: 1e-22,
            coarse_max_iterations: 5000,
            pv_max_iterations: 5000,
            pv_solver,
            ..PvEoParams::default()
        };
        let mut cycle = PvEoCycle::new(&op, &pv, (&op, &pv, EoForm::Right), &basis, exact)?;
        let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);

        // An orthonormal basis of the range of C: C of random vectors, by Gram-Schmidt twice,
        // until a new one adds nothing.
        let mut basis_u: Vec<H> = Vec::new();
        let mut seed = 1000;
        let mut misses = 0;
        while misses < 8 {
            let mut r = like.zeros_like();
            extract_parity(&mut r, &random_spinor(&lat, ls, seed), &index, Parity::Even);
            seed += 1;
            let mut e = r.zeros_like();
            cycle.coarse_correction(&mut e, &r);
            let mut n0 = [0.0];
            e.norm_sqr(&mut n0);
            for _ in 0..2 {
                for b in &basis_u {
                    let mut d = [C64::ZERO];
                    b.dot(&e, &mut d);
                    e.axpy(&[-d[0]], b);
                }
            }
            let mut n = [0.0];
            e.norm_sqr(&mut n);
            if n[0] < 1e-20 * n0[0] {
                misses += 1;
                continue;
            }
            e.scale(&[1.0 / n[0].sqrt()]);
            basis_u.push(e);
        }
        let nc = basis_u.len();
        eprintln!("{name}: the range of C has dimension {nc}");

        // K = U† C Ã U, and how far C Ã U leaves the span of U.
        let mut k = vec![C64::ZERO; nc * nc];
        let mut worst_out = 0.0f64;
        for (col, u_k) in basis_u.iter().enumerate() {
            let mut au = u_k.zeros_like();
            a.mult(&mut au, u_k);
            let mut w = au.zeros_like();
            cycle.coarse_correction(&mut w, &au);
            let mut wn = [0.0];
            w.norm_sqr(&mut wn);
            let mut rest = w.zeros_like();
            rest.copy_from(&w);
            for (row, u_j) in basis_u.iter().enumerate() {
                let mut d = [C64::ZERO];
                u_j.dot(&w, &mut d);
                k[row * nc + col] = d[0];
                rest.axpy(&[-d[0]], u_j);
            }
            let mut rn = [0.0];
            rest.norm_sqr(&mut rn);
            worst_out = worst_out.max((rn[0] / wn[0]).sqrt());
        }
        eprintln!("{name}: C Ã U outside the span of U: {worst_out:.1e} at most");
        let mut f = std::fs::File::create(format!("{prefix}_{name}.txt"))?;
        writeln!(f, "{nc}")?;
        for z in &k {
            writeln!(f, "{:.17e} {:.17e}", z.re, z.im)?;
        }
    }
    Ok(())
}
