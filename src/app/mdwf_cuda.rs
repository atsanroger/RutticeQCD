//! Möbius domain-wall solves on the GPU (`Main.backend : cuda_bridge`): the Pauli-Villars
//! preconditioned operator on the even sites and the outer solver on the device, without or
//! with the Chebyshev preconditioner or the multigrid. Of the multigrid the smoother and the
//! residuals of the cycle are on the device, and so are the filter and the solves of its
//! setup; the basis, the coarse operators and the coarse correction are those of the CPU
//! (`mdwf`), the residual and the correction crossing over in every cycle.
//! The five-dimensional source, the residual of the solution
//! (by the operator on the whole lattice, not the one that was solved) and its four-dimensional
//! fields are made on the device as well: the CPU has the four-dimensional source, the random
//! right-hand sides and the contractions.

use super::mdwf::{iterations_text, load, multigrid_cycle, write_measurements};
use super::memory_line;
use super::params::{Params, PreconditionerType, SolverType};
use crate::backend::cuda_bridge::{CudaMdwf, DeviceParities, DeviceSpinor};
use crate::fermion::{EoForm, FermionOp, Mdwf, PvRatio, SplitParity, Variant};
use crate::measure::{point_source, NUM_SOURCES};
use crate::solver::preconditioner::multigrid::{
    bootstrap_round, low_modes, pv_eo_basis, pv_eo_start, Aggregation, PvEoBasis, PvEoCycle,
    TwoGrid,
};
use crate::solver::{
    largest_magnitude, BiCgStab, BlockGcr, Cg, Chebyshev, Fgmres, Gcr, Solver, Stop,
};
use crate::{
    random_spinor, Aos, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn, Normal, Parity,
    Spinor, Vector, C64,
};
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type F = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;

/// A coarse correction of the CPU as one on the device: the residual to the CPU, its
/// correction back.
fn on_cpu<'c>(
    index: &'c EvenOddIndex,
    mut correction: impl FnMut(&mut H, &H) + 'c,
) -> impl FnMut(&mut DeviceSpinor, &DeviceSpinor) + 'c {
    move |e, r| {
        let r = r.to_host_half(index).expect("the residual from the device");
        let mut e_host = r.zeros_like();
        correction(&mut e_host, &r);
        let back = DeviceSpinor::from_host_half(&e_host, index);
        e.copy_from(&back.expect("the correction to the device"));
    }
}

/// The vectors on the even sites the multigrid is built from, with the basis on them
/// (`mdwf::multigrid_basis`): from a checkpoint, or filtered here, and improved by the
/// bootstrap if the file asks. The applications of the operator are on the device, with the
/// plain operators `device` of the ones on the CPU `cpu`; the bases and the coarse
/// corrections of the bootstrap are made on the CPU.
#[allow(clippy::too_many_arguments)]
fn device_basis(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    (d, pv): (&Mdwf<f64>, &Mdwf<f64>),
    (device_d, device_pv): (&CudaMdwf, &CudaMdwf),
    (even, odd): (&DeviceSpinor, &DeviceSpinor),
    start: Instant,
    out: &mut dyn Write,
) -> Result<(PvEoBasis, H), Box<dyn Error>> {
    let mg = params.multigrid.as_ref().ok_or("no MultiGrid section")?;
    let (_, low, boot) = params.multigrid().ok_or("no multigrid")?;
    let index = d
        .even_odd_index()
        .ok_or("no even-odd decomposition")?
        .clone();
    let keep = mg.level1.test_vectors(true) / 2;
    let ratio = || PvRatio::new(device_d, device_pv, EoForm::Right, even, odd);
    let mut y = match &mg.setup.checkpoint {
        Some(path) => {
            let mut like = H::with_shape(lat, 24, d.params().ls, 1, 1);
            like.set_parity(Parity::Even);
            let path = file.parent().map_or(path.clone(), |dir| dir.join(path));
            DeviceSpinor::from_host_half(&load(&path, &like, keep)?, &index)?
        }
        None => {
            let x = DeviceSpinor::from_host_half(&pv_eo_start(d, lat, low), &index)?;
            let (x, theta) = low_modes(|c: &DeviceSpinor| Normal::new(ratio(), c), x, low);
            writeln!(
                out,
                "  filter on the device: Ritz values of Ã†Ã {:.3e} .. {:.3e} ({:.0} s)",
                theta[0],
                theta[theta.len() - 1],
                start.elapsed().as_secs_f64()
            )?;
            x.select(&(0..keep).collect::<Vec<_>>())
        }
    };
    let agg = Aggregation::new(lat, mg.level1.block_size);
    for _ in 0..boot.rounds {
        let basis = pv_eo_basis(d, pv, &agg, &y.to_host_half(&index)?)?;
        let mut cycle = PvEoCycle::new(d, pv, (d, pv, EoForm::Right), &basis, boot.cycle)?;
        let hi = cycle.top();
        let smoother = Chebyshev::new(
            ratio(),
            boot.cycle.chebyshev_steps,
            hi / boot.cycle.chebyshev_ratio,
            hi,
            even,
        );
        let correction = on_cpu(&index, |e, r| cycle.coarse_correction(e, r));
        let mut two_grid = TwoGrid::new(smoother, correction, even);
        bootstrap_round(&mut ratio(), &mut two_grid, &mut y, boot.steps);
    }
    let y = y.to_host_half(&index)?;
    Ok((pv_eo_basis(d, pv, &agg, &y)?, y))
}

/// A random field with `ls` spinors per site.
fn random_fifth(lat: &Arc<Lattice>, ls: usize, seed: u64) -> F {
    let mut b = F::with_shape(lat, 24, ls, 1, 1);
    b.as_mut_slice()
        .copy_from_slice(random_spinor(lat, ls, seed).as_slice());
    b
}

pub(super) fn run(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    (seed_lo, seed_hi): (u64, u64),
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let fo = &params.fopr;
    let ls = fo.ls();
    let make = |mass: f64, alpha: f64| -> Result<CudaMdwf, Box<dyn Error>> {
        let mut op = CudaMdwf::new(lat, params.mdwf_params(mass))?;
        op.set_gauge(u)?;
        op.set_alpha(alpha);
        Ok(op)
    };
    let op = make(fo.mass(), fo.neff_alpha)?;
    let pv_alpha = make(fo.quark_mass_pauli_villars, fo.neff_alpha_pauli_villars)?;
    // The normal equations are the operator with D in place of D_PV.
    let (form, normal) = params.form();
    let pv = if normal { &op } else { &pv_alpha };
    let even = DeviceSpinor::zeros_half(lat, ls, Parity::Even)?;
    let odd = DeviceSpinor::zeros_half(lat, ls, Parity::Odd)?;
    let operator = || PvRatio::new(&op, pv, form, &even, &odd);
    let s = &params.solver;
    let start = Instant::now();
    // The multigrid of the CPU, for its setup and its coarse correction: the plain operators
    // it is built from, and those of the operator that is solved.
    let cpu_ops = if s.preconditioner_type == PreconditionerType::Multigrid {
        let cpu = |mass: f64, alpha: f64| -> Result<Mdwf<f64>, Box<dyn Error>> {
            let mut op = Mdwf::<f64>::new(lat, params.mdwf_params(mass));
            op.set_gauge(u)?;
            op.set_alpha(alpha);
            Ok(op)
        };
        let m_pv = fo.quark_mass_pauli_villars;
        Some([
            cpu(fo.mass(), 1.0)?,
            cpu(m_pv, 1.0)?,
            cpu(fo.mass(), fo.neff_alpha)?,
            cpu(m_pv, fo.neff_alpha_pauli_villars)?,
        ])
    } else {
        None
    };
    // (D, D_PV) plain and with α; the normal equations have D in place of D_PV.
    let cpu_pairs = cpu_ops.as_ref().map(|[d, pv, d_a, pv_a]| {
        if normal {
            ((d, d), (d_a, d_a))
        } else {
            ((d, pv), (d_a, pv_a))
        }
    });
    let basis = match cpu_pairs {
        Some((cpu, _)) => {
            let plain = make(fo.mass(), 1.0)?;
            let plain_pv = if normal {
                None
            } else {
                Some(make(fo.quark_mass_pauli_villars, 1.0)?)
            };
            let device = (&plain, plain_pv.as_ref().unwrap_or(&plain));
            Some(device_basis(
                params,
                file,
                lat,
                cpu,
                device,
                (&even, &odd),
                start,
                out,
            )?)
        }
        None => None,
    };
    let index = cpu_ops
        .as_ref()
        .and_then(|ops| ops[0].even_odd_index().cloned());
    let mut cpu_cycle = match (cpu_pairs, &basis, params.multigrid()) {
        (Some((plain, (d_a, pv_a))), Some(basis), Some((cycle_params, _, _))) => {
            let cycle = multigrid_cycle::<Aos>(
                params,
                lat,
                plain,
                (d_a, pv_a, form),
                (d_a, pv_a),
                basis,
                cycle_params,
                out,
            )?
            .0;
            Some((cycle, cycle_params))
        }
        _ => None,
    };
    let mut pre: Option<Box<dyn LinearFn<DeviceSpinor> + '_>> = match s.preconditioner_type {
        PreconditionerType::Chebyshev => {
            let c = s.chebyshev_settings();
            let mut start_vector = even.zeros_like();
            let random = DeviceSpinor::from_host(&random_fifth(lat, ls, 5))?;
            DeviceParities.extract(&mut start_vector, &random, Parity::Even);
            let hi = 1.1 * largest_magnitude(&mut operator(), &mut start_vector, 50);
            Some(Box::new(Chebyshev::new(
                operator(),
                c.number_of_steps,
                hi / c.interval_ratio,
                hi,
                &even,
            )))
        }
        PreconditionerType::None => None,
        PreconditionerType::Multigrid => {
            let (cycle, cycle_params) = cpu_cycle.as_mut().expect("made above");
            let index = index.as_deref().ok_or("no even-odd decomposition")?;
            let hi = cycle.top();
            let smoother = Chebyshev::new(
                operator(),
                cycle_params.chebyshev_steps,
                hi / cycle_params.chebyshev_ratio,
                hi,
                &even,
            );
            let correction = on_cpu(index, move |e, r| cycle.coarse_correction(e, r));
            Some(Box::new(TwoGrid::new(smoother, correction, &even)))
        }
    };
    writeln!(
        out,
        "  setup on the device: {:.0} s; {}",
        start.elapsed().as_secs_f64(),
        memory_line()
    )?;

    let mut a = operator();
    let stop: Stop = s.stop();
    let k = s.number_of_kept_directions;
    let (mut check_work, mut source_work) = (op.work(), op.work());
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let mut solve = |b: &DeviceSpinor,
                     label: &str,
                     out: &mut dyn Write|
     -> Result<DeviceSpinor, Box<dyn Error>> {
        let nv = b.num_vectors();
        let mut rhs = even.zeros_with(nv);
        a.right_hand_side(&DeviceParities, &mut rhs, b);
        let mut sol = rhs.zeros_like();
        let start = Instant::now();
        let m = pre
            .as_mut()
            .map(|m| m.as_mut() as &mut dyn LinearFn<DeviceSpinor>);
        let result = match s.solver_type {
            SolverType::Gcr => Gcr::new(stop, k).solve(&mut a, m, &mut sol, &rhs),
            SolverType::BlockGcr => BlockGcr::new(stop, k).solve(&mut a, m, &mut sol, &rhs),
            SolverType::BiCgStab => BiCgStab::new(stop).solve(&mut a, m, &mut sol, &rhs),
            SolverType::Fgmres => Fgmres::new(Stop {
                max_iterations: k,
                max_restarts: stop.max_iterations / k.max(1),
                ..stop
            })
            .solve(&mut a, m, &mut sol, &rhs),
            SolverType::Cg => Cg::new(stop).solve(&mut a, m, &mut sol, &rhs),
            SolverType::Minres => unreachable!("refused when the file is read"),
        };
        let seconds = start.elapsed().as_secs_f64();
        // x^(α), with D^(α) x^(α) = b on the whole lattice
        let mut x = b.zeros_like();
        a.solution(&DeviceParities, &mut x, &sol, b);
        let worst = {
            let mut r = b.zeros_like();
            op.mult(&mut check_work, &mut r, &x, Variant::D);
            r.aypx(&vec![C64::new(-1.0, 0.0); nv], b);
            let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
            r.norm_sqr(&mut rr);
            b.norm_sqr(&mut bb);
            rr.iter().zip(&bb).fold(0.0_f64, |m, (r, b)| m.max(r / b))
        };
        // x = Q x^(α)
        if op.alpha() != 1.0 {
            let mut plain = x.zeros_like();
            op.mult_q(&mut plain, &x, false);
            x = plain;
        }
        writeln!(
            out,
            "  {label}: {} {} iterations on the device; worst |Dx−b|² {worst:.0e}; {seconds:.1} s; {}",
            s.solver_type,
            iterations_text(&result),
            memory_line()
        )?;
        Ok(x)
    };

    let Some(src) = &params.source else {
        for seed in seed_lo..=seed_hi {
            let b = DeviceSpinor::from_host(&random_fifth(lat, ls, seed))?;
            solve(&b, &format!("seed {seed}"), out)?;
        }
        return Ok(());
    };

    // The propagator from the point source, `number_of_rhs_per_solve` columns at a time: the
    // four-dimensional quark field and the midpoint field of every column.
    let eta = point_source::<f64>(lat, src.source_position);
    let (mut prop, mut mid) = (eta.zeros_like(), eta.zeros_like());
    let columns: Vec<usize> = (0..NUM_SOURCES).collect();
    for chunk in columns.chunks(src.number_of_rhs_per_solve) {
        let eta_c = DeviceSpinor::from_host(&eta.select(chunk))?;
        let mut b = DeviceSpinor::zeros_fifth(lat, ls)?.zeros_with(chunk.len());
        op.source_5d(&mut source_work, &mut b, &eta_c);
        let psi = solve(&b, &format!("columns {chunk:?}"), out)?;
        let (mut q, mut m) = (eta_c.zeros_like(), eta_c.zeros_like());
        op.propagator_4d(&mut q, &psi);
        op.midpoint_4d(&mut m, &psi);
        prop.assign(chunk, &q.to_host()?);
        mid.assign(chunk, &m.to_host()?);
    }
    write_measurements(
        src.source_position,
        params.main.lattice_size[3],
        &prop,
        &mid,
        out,
    )
}
