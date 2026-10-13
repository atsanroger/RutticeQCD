//! Möbius domain-wall solves on the GPU (`Main.backend : cuda_bridge`): the Pauli-Villars
//! preconditioned operator on the even sites and the outer solver on the device, one
//! right-hand side at a time, without or with the Chebyshev preconditioner. The source, the
//! four-dimensional fields of the solution and the check of the residual are on the CPU.

use super::mdwf::{iterations_text, write_measurements};
use super::memory_line;
use super::params::{Params, PreconditionerType, SolverType};
use crate::backend::cuda_bridge::{CudaMdwf, DeviceParities, DeviceSpinor};
use crate::fermion::{FermionOp, Mdwf, PvRatio, SplitParity, Variant};
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
    // The plain operator on the CPU: the source, the four-dimensional fields, the check.
    let mut check = Mdwf::<f64>::new(lat, params.mdwf_params(fo.mass()));
    check.set_gauge(u)?;

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
    // Solves D ψ = b (one vector) and reports it in one line, under `label`.
    let mut solve = |b: &F, label: &str, out: &mut dyn Write| -> Result<F, Box<dyn Error>> {
        let device_b = DeviceSpinor::from_host(b)?;
        let mut rhs = even.zeros_like();
        a.right_hand_side(&DeviceParities, &mut rhs, &device_b);
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
        // x^(α), and x = Q x^(α)
        let mut x = device_b.zeros_like();
        a.solution(&DeviceParities, &mut x, &sol, &device_b);
        if op.alpha() != 1.0 {
            let mut plain = x.zeros_like();
            op.mult_q(&mut plain, &x, false);
            x = plain;
        }
        let x = x.to_host()?;
        let mut r = b.zeros_like();
        check.mult(&mut check.work(), &mut r, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], b);
        let (mut rr, mut bb) = ([0.0], [0.0]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        writeln!(
            out,
            "  {label}: {} {} iterations on the device; worst |Dx−b|² {:.0e}; {seconds:.1} s; {}",
            s.solver_type,
            iterations_text(&result),
            rr[0] / bb[0],
            memory_line()
        )?;
        Ok(x)
    };

    let Some(src) = &params.source else {
        for seed in seed_lo..=seed_hi {
            solve(&random_fifth(lat, ls, seed), &format!("seed {seed}"), out)?;
        }
        return Ok(());
    };

    // The propagator from the point source, a column at a time: the four-dimensional quark
    // field and the midpoint field of every column.
    let eta = point_source::<f64>(lat, src.source_position);
    let (mut prop, mut mid) = (eta.zeros_like(), eta.zeros_like());
    for column in 0..NUM_SOURCES {
        let eta_c = eta.select(&[column]);
        let mut b = F::with_shape(lat, 24, ls, 1, 1);
        check.source_5d(&mut check.work(), &mut b, &eta_c);
        let psi = solve(&b, &format!("column {column}"), out)?;
        let (mut q, mut m) = (eta_c.zeros_like(), eta_c.zeros_like());
        check.propagator_4d(&mut q, &psi);
        check.midpoint_4d(&mut m, &psi);
        prop.assign(&[column], &q);
        mid.assign(&[column], &m);
    }
    write_measurements(
        src.source_position,
        params.main.lattice_size[3],
        &prop,
        &mid,
        out,
    )
}
