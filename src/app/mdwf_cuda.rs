//! Möbius domain-wall solves on the GPU (`Main.backend : cuda_bridge`): the Pauli-Villars
//! preconditioned operator on the even sites and the outer solver on the device, without or
//! with the Chebyshev preconditioner. The five-dimensional source, the residual of the solution
//! (by the operator on the whole lattice, not the one that was solved) and its four-dimensional
//! fields are made on the device as well: the CPU has the four-dimensional source, the random
//! right-hand sides and the contractions.

use super::mdwf::{iterations_text, write_measurements};
use super::memory_line;
use super::params::{Params, PreconditionerType, SolverType};
use crate::backend::cuda_bridge::{CudaMdwf, DeviceParities, DeviceSpinor};
use crate::fermion::{FermionOp, PvRatio, SplitParity, Variant};
use crate::measure::{point_source, NUM_SOURCES};
use crate::solver::{
    largest_magnitude, BiCgStab, BlockGcr, Cg, Chebyshev, Fgmres, Gcr, Solver, Stop,
};
use crate::{random_spinor, Field, Gauge, Lattice, LinearFn, Parity, Spinor, Vector, C64};
use std::error::Error;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

type F = Field<f64, Spinor>;

/// A random field with `ls` spinors per site.
fn random_fifth(lat: &Arc<Lattice>, ls: usize, seed: u64) -> F {
    let mut b = F::with_shape(lat, 24, ls, 1, 1);
    b.as_mut_slice()
        .copy_from_slice(random_spinor(lat, ls, seed).as_slice());
    b
}

pub(super) fn run(
    params: &Params,
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
    let mut pre = match s.preconditioner_type {
        PreconditionerType::Chebyshev => {
            let c = s.chebyshev_settings();
            let mut start_vector = even.zeros_like();
            let random = DeviceSpinor::from_host(&random_fifth(lat, ls, 5))?;
            DeviceParities.extract(&mut start_vector, &random, Parity::Even);
            let hi = 1.1 * largest_magnitude(&mut operator(), &mut start_vector, 50);
            Some(Chebyshev::new(
                operator(),
                c.number_of_steps,
                hi / c.interval_ratio,
                hi,
                &even,
            ))
        }
        PreconditionerType::None => None,
        PreconditionerType::Multigrid => unreachable!("refused when the file is read"),
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
        let m = pre.as_mut().map(|m| m as &mut dyn LinearFn<DeviceSpinor>);
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
