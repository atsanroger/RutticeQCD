//! Prolongations for the solves with D† (Ã† on the even sites, as the HMC multigrid makes them)
//! from the test vectors y of one setup of Ã, on the device (the note, Table "dualbasis"): GCR(8)
//! with the cycle of Ã† on P(y) shared with Ã, on P(Γ5 y), on P(y ∪ Γ5 y), with the lifts of D or
//! of Ã†, against Ã on P(y); with `--own 1` also a full setup of its own for Ã†. Also how much of
//! span(Γ5 y) lies in span(y), and |A z|²/|z|² of y and Γ5 y for Ã and Ã†.
//!
//! ```text
//! cargo run --release --features cuda-bridge --example dual_basis -- L FORMAT FILE M5 SMEAR
//!     [--steps S] [--ratio R] [--vectors N] [--tol T] [--max N] [--mass M] [--seed S]
//!     [--lean 1] [--own 1] [--k K]
//! ```
//!
//! On the L⁴ configuration FILE; SMEAR 1 for stout 0.1 × 3; m 0.001, Ls 8, b 1.5, c 0.5, blocks
//! of 4⁴, 32 test vectors (16 filtered in 2 rounds, the bootstrap 2 × 4), Chebyshev 8 steps on
//! [hi/10, hi] (setup S; B: `--steps 12 --ratio 30`), GCR(`--k`, 8) to 1e-12 with at most 1000
//! iterations; `--lean 1` runs only the variants that fit on 24⁴.

use ruttice::backend::cuda_bridge::multigrid::{
    bootstrap, device_correction, filtered_test_vectors, two_grid, Stage,
};
use ruttice::backend::cuda_bridge::{
    CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceSpinor,
};
use ruttice::fermion::{adjoint_right_hand_side, EoForm, FermionOp, MdwfParams, PvRatio};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::hmc::LinkField;
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, BootstrapParams, CoarseKrylov, LowModeParams, PvEoParams,
};
use ruttice::solver::{Gcr, SolveError, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, EvenOddIndex, Field, Lattice, LinearFn, Parity, Spinor, Vector, ND,
};
use std::error::Error;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

type V = DeviceSpinor<f64>;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str, default: f64| -> Result<f64, Box<dyn Error>> {
        match args.iter().position(|a| a == name) {
            Some(i) => Ok(args.get(i + 1).ok_or("a value")?.parse()?),
            None => Ok(default),
        }
    };
    let l: usize = args[0].parse()?;
    let m5: f64 = args[3].parse()?;
    let smear = args[4] == "1";
    let keep = value("--vectors", 32.0)? as usize;
    let tol = value("--tol", 1e-12)?;
    let max_it = value("--max", 1000.0)? as usize;
    let k = value("--k", 8.0)? as usize;
    let lean = value("--lean", 0.0)? != 0.0;
    let cycle = PvEoParams {
        chebyshev_steps: value("--steps", 8.0)? as usize,
        chebyshev_ratio: value("--ratio", 10.0)?,
        ..PvEoParams::default()
    };
    let dual = PvEoParams {
        pv_solver: CoarseKrylov::BiCgStab,
        ..cycle
    };
    let lat = Lattice::single_node([l; ND]);
    let u = ruttice::app::read_gauge_named(&args[1], Path::new(&args[2]), &lat)?;
    let v = match smear {
        true => GaugeSmearing {
            smear: Smear::Stout(Stout { rho: uniform(0.1) }),
            steps: 3,
        }
        .levels(&u)
        .pop()
        .unwrap(),
        false => u.duplicate(),
    };
    let quark = value("--mass", 0.001)?;
    let mk = |mass: f64| -> Result<CudaMdwf<f64>, Box<dyn Error>> {
        let mut op = CudaMdwf::<f64>::new(
            &lat,
            MdwfParams {
                ls: 8,
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
            },
        )?;
        op.set_gauge(&v)?;
        Ok(op)
    };
    let (d, pv) = (mk(quark)?, mk(1.0)?);
    let even = V::shape_half(&lat, 8, Parity::Even)?;
    let odd = V::shape_half(&lat, 8, Parity::Odd)?;
    let index = EvenOddIndex::new(&lat);
    let agg = Aggregation::new(&lat, [4; ND]);
    let low = LowModeParams::default();
    let quiet = &mut |_: Stage<'_, f64>| {};
    println!(
        "# {l}^4 M5 {m5} stout {smear}, m 0.001, Ls 8, b 1.5 c 0.5; {keep} vectors, blocks 4^4, Chebyshev {} on [hi/{}, hi]; GCR({k}) to {tol:e}",
        cycle.chebyshev_steps, cycle.chebyshev_ratio
    );

    // The setup of D: filter and two bootstrap rounds.
    let full_setup = |ops: (&CudaMdwf<f64>, &CudaMdwf<f64>),
                      params: PvEoParams,
                      quiet: &mut dyn FnMut(Stage<'_, f64>)|
     -> Result<V, Box<dyn Error>> {
        let mut y = filtered_test_vectors(ops, (&even, &odd), &lat, &index, low, keep, quiet)?;
        let mut with = device_correction(ops, &agg, params, &even, true);
        bootstrap(
            ops,
            (&even, &odd),
            &lat,
            &mut y,
            BootstrapParams {
                cycle: params,
                ..BootstrapParams::default()
            },
            &mut with,
            quiet,
        )?;
        Ok(y)
    };
    let t = Instant::now();
    let y = full_setup((&d, &pv), cycle, quiet)?;
    let setup_d = t.elapsed().as_secs_f64();

    let mut host = Field::<f64, Spinor>::with_shape(&lat, 24, 8, 1, 1);
    host.as_mut_slice()
        .copy_from_slice(random_spinor(&lat, 8, value("--seed", 91.0)? as u64).as_slice());
    let r = V::from_host(&host)?;
    let mut rhs = even.zeros_with(1);
    PvRatio::new(&d, &pv, EoForm::Right, &even, &odd).right_hand_side(
        &DeviceParities,
        &mut rhs,
        &r,
    );
    let mut rhs_dag = even.zeros_with(1);
    adjoint_right_hand_side(&d, &DeviceParities, &mut rhs_dag, &r);

    // GCR(8) on PvRatio(ops) with the cycle of `basis`.
    let solve = |ops: (&CudaMdwf<f64>, &CudaMdwf<f64>),
                 basis: Rc<DeviceBasis>,
                 params: PvEoParams,
                 rhs: &V|
     -> Result<(usize, f64, f64), Box<dyn Error>> {
        let t = Instant::now();
        let mut c =
            DeviceCorrection::<f64, f32>::shared(basis, ops, ops, EoForm::Right, params, &even)?;
        let mut m = two_grid(
            ops,
            EoForm::Right,
            (&even, &odd),
            &lat,
            params,
            Box::new(move |e, r| c.correct(e, r)),
            None,
        )?;
        let made = t.elapsed().as_secs_f64();
        let mut a = PvRatio::new(ops.0, ops.1, EoForm::Right, &even, &odd);
        let mut x = rhs.zeros_like();
        let info = Gcr::new(Stop::simple(tol, max_it), k).solve(
            &mut a,
            Some(&mut m as &mut dyn LinearFn<V>),
            &mut x,
            rhs,
        );
        let mut n = [0.0];
        x.norm_sqr(&mut n);
        let s = t.elapsed().as_secs_f64() - made;
        Ok(match info {
            Ok(i) => (i.iterations, i.residual_sqr[0], s),
            Err(SolveError::NotConverged(i)) => (i.iterations + 100000, i.residual_sqr[0], s),
            Err(e) => return Err(e.into()),
        })
    };
    let line = |name: &str, extra: f64, (it, res, s): (usize, f64, f64)| {
        let stalled = if it >= 100000 { " (not converged)" } else { "" };
        println!(
            "{name:52} {:5}{stalled}  |r|² {res:.1e}  solve {s:6.2} s  basis {extra:6.2} s",
            it % 100000
        );
    };

    let basis_d = Rc::new(DeviceBasis::pv_eo(&d, &pv, &agg, &y, true)?);
    line(
        "primal D: Ã, P(y)",
        setup_d,
        solve((&d, &pv), basis_d.clone(), cycle, &rhs)?,
    );
    line(
        "dual 0: Ã†, the same P (the HMC multigrid now)",
        0.0,
        solve((&pv, &d), basis_d.clone(), dual, &rhs_dag)?,
    );
    let g5 = |y: &V| {
        let mut z = y.zeros_like();
        d.mult_gamma5_on(&mut z, y);
        z
    };
    let dual_bootstrap = |mut z: V, rounds: usize| -> Result<V, Box<dyn Error>> {
        let mut with = device_correction((&pv, &d), &agg, dual, &even, true);
        bootstrap(
            (&pv, &d),
            (&even, &odd),
            &lat,
            &mut z,
            BootstrapParams {
                rounds,
                cycle: dual,
                ..BootstrapParams::default()
            },
            &mut with,
            &mut |_: Stage<'_, f64>| {},
        )?;
        Ok(z)
    };
    let n = y.num_vectors();
    let first: Vec<usize> = (0..n).collect();
    let second: Vec<usize> = (n..2 * n).collect();
    let g5y = g5(&y);
    let mut both = y.zeros_with(2 * n);
    both.assign(&first, &y);
    both.assign(&second, &g5y);
    let h = n / 2;
    let mut halves = y.zeros_with(n);
    halves.assign(
        &(0..h).collect::<Vec<_>>(),
        &y.select(&(0..h).collect::<Vec<_>>()),
    );
    halves.assign(
        &(h..n).collect::<Vec<_>>(),
        &g5y.select(&(0..h).collect::<Vec<_>>()),
    );
    let _ = &dual_bootstrap;
    // How far span(Γ5 y) lies in span(y): the mean of |proj Γ5 y_j|²/|Γ5 y_j|² onto the
    // orthonormalised y; and how near to null each set is for Ã and for the operator of D†,
    // the mean of |A z_j|²/|z_j|².
    let mut q = y.zeros_like();
    for j in 0..n {
        let mut v = y.select(&[j]);
        for i in 0..j {
            let mut c = [ruttice::C64::new(0.0, 0.0)];
            q.select(&[i]).dot(&v, &mut c);
            v.axpy(&[-c[0]], &q.select(&[i]));
        }
        let mut nn = [0.0];
        v.norm_sqr(&mut nn);
        v.scale(&[1.0 / nn[0].sqrt()]);
        q.assign(&[j], &v);
    }
    let captured = |z: &V| {
        let mut sum = 0.0;
        for j in 0..z.num_vectors() {
            let w = z.select(&[j]);
            let mut ww = [0.0];
            w.norm_sqr(&mut ww);
            let mut p = 0.0;
            for i in 0..n {
                let mut c = [ruttice::C64::new(0.0, 0.0)];
                q.select(&[i]).dot(&w, &mut c);
                p += c[0].norm_sqr();
            }
            sum += p / ww[0];
        }
        sum / z.num_vectors() as f64
    };
    let nullness = |ops: (&CudaMdwf<f64>, &CudaMdwf<f64>), z: &V| {
        let mut a = PvRatio::new(ops.0, ops.1, EoForm::Right, &even, &odd);
        let mut az = z.zeros_like();
        a.mult(&mut az, z);
        let nz = z.num_vectors();
        let (mut top, mut bottom) = (vec![0.0; nz], vec![0.0; nz]);
        az.norm_sqr(&mut top);
        z.norm_sqr(&mut bottom);
        let r: Vec<f64> = top.iter().zip(&bottom).map(|(t, b)| t / b).collect();
        let mean = r.iter().sum::<f64>() / nz as f64;
        let max = r.iter().cloned().fold(0.0, f64::max);
        (mean, max)
    };
    println!(
        "overlap: mean |proj_span(y) Γ5 y_j|²/|Γ5 y_j|² = {:.3} (1: Γ5 y in span y)",
        captured(&g5y)
    );
    for (name, z) in [("y", &y), ("Γ5 y", &g5y)] {
        let (pm, px) = nullness((&d, &pv), z);
        let (dm, dx) = nullness((&pv, &d), z);
        println!(
            "|A z|²/|z|² of {name:5}: for Ã mean {pm:.2e} max {px:.2e};  for the operator of D† mean {dm:.2e} max {dx:.2e}"
        );
    }
    let b10 = Rc::new(DeviceBasis::pv_eo(&d, &pv, &agg, &both, true)?);
    line(
        "primal 10: Ã with P(y and Γ5 y), 2n vectors",
        0.0,
        solve((&d, &pv), b10, cycle, &rhs)?,
    );
    for (name, z, ops_d) in [
        (
            "dual 11: P(half of y and half of Γ5 y) as for D, n",
            &halves,
            true,
        ),
        ("dual 2: P'(Γ5 y)", &g5y, false),
        ("dual 7: P'(y and Γ5 y), 2n vectors", &both, false),
        ("dual 8: P(Γ5 y) made as for D", &g5y, true),
        (
            "dual 9: P'(half of y and half of Γ5 y), n vectors",
            &halves,
            false,
        ),
        (
            "dual 10: P(y and Γ5 y) made as for D, 2n vectors",
            &both,
            true,
        ),
    ]
    .into_iter()
    .filter(|(name, _, _)| !lean || name.starts_with("dual 8") || name.starts_with("dual 10"))
    {
        let t = Instant::now();
        let b = Rc::new(match ops_d {
            false => DeviceBasis::pv_eo(&pv, &d, &agg, z, true)?,
            true => DeviceBasis::pv_eo(&d, &pv, &agg, z, true)?,
        });
        let made = t.elapsed().as_secs_f64();
        line(name, made, solve((&pv, &d), b, dual, &rhs_dag)?);
    }
    if value("--own", 0.0)? == 0.0 {
        return Ok(());
    }
    let t = Instant::now();
    let z = full_setup((&pv, &d), dual, &mut |_: Stage<'_, f64>| {})?;
    let b = Rc::new(DeviceBasis::pv_eo(&pv, &d, &agg, &z, true)?);
    let made = t.elapsed().as_secs_f64();
    line(
        "dual 6: a full setup of its own (filter, 2 rounds)",
        made,
        solve((&pv, &d), b, dual, &rhs_dag)?,
    );
    Ok(())
}
