//! Ã† with the adjoint of the cycle of Ã. With the roles of D and D_PV exchanged, the even-odd
//! Pauli-Villars operator of the right form is (S C_e⁻¹)† (S_PV C_PV,e⁻¹) = Ã†, and the cycle
//! built the same way on the prolongation P of Ã is its exact adjoint in structure: the
//! correction C_PV,e π P (P†D_PV P)⁻¹ P† ι π P (P†D†P)⁻¹ P† ι C_e† = C†, the Chebyshev smoother
//! (real coefficients) on Ã†. Since ‖1 − M Ã‖ = ‖1 − Ã† M†‖ it should precondition Ã† as well
//! as M does Ã; Γ5 M Γ5 does not (Γ5 Ã Γ5 ≠ Ã†). The adjoint factor P†D†P is solved by MINRES on
//! the Γ5-Hermitian form of P†DP or by BiCGStab on its exact adjoint.
//!
//! ```text
//! cargo run --release --example adjoint_cycle -- L MASS M5 SMEAR(0|1) FORMAT FILE
//! ```
//! on the L⁴ configuration FILE (FORMAT: text, binary, nersc, ildg or scidac; the v2 test
//! configurations), Ls = 8, b = 1.5, c = 0.5, blocks of 4⁴, 32 test vectors, Chebyshev 8 steps on
//! [hi/10, hi], GCR(8) to 1e-18.

use ruttice::fermion::{EoForm, FermionOp, Mdwf, MdwfParams, PvEvenOdd};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::solver::preconditioner::multigrid::{
    pv_eo_bootstrap, pv_eo_low_modes, Aggregation, BootstrapParams, CoarseKrylov, LowModeParams,
    PvEoCycle, PvEoParams,
};
use ruttice::solver::{Gcr, Solver, Stop};
use ruttice::{
    extract_parity, random_spinor, Boundary, Field, Half, Lattice, LinearFn, Parity, Spinor,
    Vector, C64, ND,
};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

type H = Field<f64, Spinor, Half>;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let l: usize = args.first().ok_or("L")?.parse()?;
    let mass: f64 = args.get(1).ok_or("MASS")?.parse()?;
    let m5: f64 = args.get(2).ok_or("M5")?.parse()?;
    let smear = args.get(3).ok_or("SMEAR")? == "1";
    let format = args.get(4).ok_or("FORMAT")?;
    let file = args.get(5).ok_or("FILE")?;
    let start = Instant::now();
    let lat = Lattice::single_node([l; ND]);
    let mut u = ruttice::app::read_gauge_named(format, Path::new(file), &lat)?;
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

    // The setup of the program: 16 vectors filtered in 2 rounds, 16 of them kept, bootstrap
    // 2 x 4; P from them, for both cycles.
    let (y, _) = pv_eo_low_modes(
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
    let mut y = y.select(&(0..16).collect::<Vec<_>>());
    let agg = Aggregation::new(&lat, [4, 4, 4, 4]);
    let tuned = PvEoParams {
        chebyshev_steps: 8,
        chebyshev_ratio: 10.0,
        ..PvEoParams::default()
    };
    let basis = pv_eo_bootstrap(
        &op,
        &pv,
        &agg,
        &mut y,
        BootstrapParams {
            cycle: tuned,
            ..BootstrapParams::default()
        },
    )?;
    eprintln!("setup {:.0} s", start.elapsed().as_secs_f64());

    let like = y.select(&[0]);
    let index = op.even_odd_index().expect("even-odd").clone();
    let mut b = like.zeros_like();
    extract_parity(&mut b, &random_spinor(&lat, ls, 91), &index, Parity::Even);
    let mut bb = [0.0];
    b.norm_sqr(&mut bb);
    let stop = Stop::simple(1e-18, 4000);

    // Ã† is the adjoint of Ã: <x, Ã y> = <Ã† x, y>.
    {
        let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);
        let mut a_dag = PvEvenOdd::new(&pv, &op, EoForm::Right, &like);
        let mut x = like.zeros_like();
        extract_parity(&mut x, &random_spinor(&lat, ls, 92), &index, Parity::Even);
        let (mut ay, mut adx) = (like.zeros_like(), like.zeros_like());
        a.mult(&mut ay, &b);
        a_dag.mult(&mut adx, &x);
        let (mut l, mut r) = ([C64::ZERO], [C64::ZERO]);
        x.dot(&ay, &mut l);
        adx.dot(&b, &mut r);
        eprintln!(
            "<x, Ã y> = {:.12e} {:+.12e}i, <Ã† x, y> = {:.12e} {:+.12e}i",
            l[0].re, l[0].im, r[0].re, r[0].im
        );
    }

    let run = |label: &str, a: &mut PvEvenOdd<'_, f64>, m: Option<&mut dyn LinearFn<H>>| {
        let t = Instant::now();
        let mut x = like.zeros_like();
        let info = Gcr::new(stop, 8).solve(a, m, &mut x, &b);
        let mut r = like.zeros_like();
        a.mult(&mut r, &x);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let mut rr = [0.0];
        r.norm_sqr(&mut rr);
        let its = info.map_or_else(
            |e| format!("{e}"),
            |i| format!("{} iterations", i.iterations),
        );
        eprintln!(
            "{label:44}: GCR(8) {its}, |b - A x|^2/|b|^2 {:.1e}, {:.1} s",
            rr[0] / bb[0],
            t.elapsed().as_secs_f64()
        );
    };

    let mut a = PvEvenOdd::new(&op, &pv, EoForm::Right, &like);
    let mut cycle = PvEoCycle::new(&op, &pv, (&op, &pv, EoForm::Right), &basis, tuned)?;
    run("Ã  with M", &mut a, Some(&mut cycle));
    let mut a_dag = PvEvenOdd::new(&pv, &op, EoForm::Right, &like);
    for (name, pv_solver) in [
        ("MINRES, Γ5 form", CoarseKrylov::Minres),
        ("BiCGStab, exact adjoint", CoarseKrylov::BiCgStab),
    ] {
        let dual = PvEoParams { pv_solver, ..tuned };
        let mut cycle = PvEoCycle::new(&pv, &op, (&pv, &op, EoForm::Right), &basis, dual)?;
        run(
            &format!("Ã† with M† (P†D†P by {name})"),
            &mut a_dag,
            Some(&mut cycle),
        );
    }
    if l <= 12 {
        run("Ã† without a preconditioner", &mut a_dag, None);
    }
    Ok(())
}
