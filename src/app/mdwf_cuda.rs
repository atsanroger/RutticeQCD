//! Möbius domain-wall solves on the GPU (`Main.backend : cuda_bridge`): the Pauli-Villars
//! preconditioned operator on the even sites and the outer solver on the device, without or
//! with the Chebyshev preconditioner or the multigrid.
//!
//! The multigrid is on the device: the filter of its test vectors and their bootstrap, the
//! prolongation, the coarse operators, and the cycle with its coarse solves; above the
//! coarse lattice in single precision unless `MultiGrid.fine_precision` says double (the
//! outer solver is in double precision either way), and so are levels below the coarse
//! lattice.
//!
//! The five-dimensional source, the residual of the solution (by the operator on the whole
//! lattice, not the one that was solved) and its four-dimensional fields are made on the
//! device as well: the CPU has the four-dimensional source, the random right-hand sides and
//! the contractions.

use super::mdwf::{iterations_text, load, write_measurements};
use super::memory_line;
use super::params::{BasisPrecision, Params, PrecisionType, PreconditionerType, SolverType};
use crate::backend::cuda_bridge::multigrid::Correction;
use crate::backend::cuda_bridge::multigrid::{
    bootstrap, device_correction, filtered_test_vectors, top_of, Stage,
};
use crate::backend::cuda_bridge::{
    CudaError, CudaMdwf, DeviceBasis, DeviceCoarse, DeviceCorrection, DeviceParities, DeviceReal,
    DeviceSpinor, LevelReal,
};
use crate::fermion::{EoForm, FermionOp, PvRatio};
use crate::measure::{point_source, NUM_SOURCES};
use crate::solver::preconditioner::multigrid::Aggregation;
use crate::solver::{BiCgStab, BlockGcr, Cg, Chebyshev, Convert, Fgmres, Gcr, Solver, Stop};
use crate::{
    random_fifth, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn, Parity, Spinor, Vector, C64,
};
use std::cell::Cell;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

/// With `RUTTICE_CHECKSUM` set, a line with the checksum of what a stage of the run made, to
/// compare runs bit by bit and find where they part.
fn checksum_line(
    out: &mut dyn Write,
    what: &str,
    sum: impl FnOnce() -> Result<u64, CudaError>,
) -> Result<(), Box<dyn Error>> {
    if std::env::var_os("RUTTICE_CHECKSUM").is_some() {
        writeln!(out, "  checksum of {what}: {:016x}", sum()?)?;
    }
    Ok(())
}

/// The memory of the host and that of the program's fields on the device, for a report.
fn memory() -> String {
    let (now, peak) = crate::backend::cuda_bridge::device_memory();
    let gb = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
    format!(
        "{}; device {:.2} GB, peak {:.2} GB",
        memory_line(),
        gb(now),
        gb(peak)
    )
}
type H = Field<f64, Spinor, Half>;

/// (D, the second factor): the operators a multigrid is built from, or is for. The second
/// factor is D_PV, or D itself for the normal equations.
type Pair<'a, Op> = (&'a Op, &'a Op);

/// The operators of a run on the device in the precision `T`, on the same links there, and
/// the shape of the fields on either parity.
struct Operators<T: DeviceReal> {
    /// With Neff's α: those of the operator that is solved.
    op: CudaMdwf<T>,
    pv: Option<CudaMdwf<T>>,
    /// Plain, for the multigrid.
    plain: Option<CudaMdwf<T>>,
    plain_pv: Option<CudaMdwf<T>>,
    even: DeviceSpinor<T>,
    odd: DeviceSpinor<T>,
}

impl<T: DeviceReal> Operators<T> {
    /// `with_plain`: with the plain operators as well.
    fn new(
        params: &Params,
        lat: &Arc<Lattice>,
        u: &Field<f64, Gauge>,
        with_plain: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let fo = &params.fopr;
        let mut op = CudaMdwf::new(lat, params.mdwf_params(fo.mass()))?;
        op.set_gauge(u)?;
        op.set_alpha(fo.neff_alpha);
        let make = |mass: f64, alpha: f64| -> Result<CudaMdwf<T>, Box<dyn Error>> {
            let mut other = CudaMdwf::on_the_links_of(&op, params.mdwf_params(mass))?;
            other.set_alpha(alpha);
            Ok(other)
        };
        let (m_pv, (_, normal)) = (fo.quark_mass_pauli_villars, params.form());
        Ok(Self {
            pv: match normal {
                true => None,
                false => Some(make(m_pv, fo.neff_alpha_pauli_villars)?),
            },
            plain: match with_plain {
                true => Some(make(fo.mass(), 1.0)?),
                false => None,
            },
            plain_pv: match with_plain && !normal {
                true => Some(make(m_pv, 1.0)?),
                false => None,
            },
            even: DeviceSpinor::shape_half(lat, fo.ls(), Parity::Even)?,
            odd: DeviceSpinor::shape_half(lat, fo.ls(), Parity::Odd)?,
            op,
        })
    }

    fn outer(&self) -> Pair<'_, CudaMdwf<T>> {
        (&self.op, self.pv.as_ref().unwrap_or(&self.op))
    }

    fn plain(&self) -> Result<Pair<'_, CudaMdwf<T>>, Box<dyn Error>> {
        let d = self
            .plain
            .as_ref()
            .ok_or("the plain operators were not made")?;
        Ok((d, self.plain_pv.as_ref().unwrap_or(d)))
    }
}

/// The vectors on the even sites the multigrid is built from: from a checkpoint, or filtered
/// here, and improved by the bootstrap if the file asks.
fn test_vectors<T: DeviceReal>(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    ops: &Operators<T>,
    index: &EvenOddIndex,
    start: Instant,
    out: &mut dyn Write,
) -> Result<DeviceSpinor<T>, Box<dyn Error>> {
    let mg = params.multigrid.as_ref().ok_or("no MultiGrid section")?;
    let (_, low, boot) = params.multigrid().ok_or("no multigrid")?;
    let (even, odd, device) = (&ops.even, &ops.odd, ops.plain()?);
    let ls = even.ls();
    let keep = mg.level1.test_vectors(true) / 2;
    let half = mg.basis_precision() == BasisPrecision::Half;
    // What the setup reports: the times and Ritz values, and with RUTTICE_CHECKSUM the
    // checksums of every stage, as the run always printed them.
    let mut failed: Option<Box<dyn Error>> = None;
    let mut report = |stage: Stage<'_, T>| {
        let line = match stage {
            Stage::Filtered { theta, vectors } => writeln!(
                out,
                "  filter on the device: Ritz values of Ã†Ã {:.3e} .. {:.3e} ({:.0} s); {}",
                theta[0],
                theta[theta.len() - 1],
                start.elapsed().as_secs_f64(),
                memory()
            )
            .map_err(Box::<dyn Error>::from)
            .and_then(|()| checksum_line(out, "the filtered vectors", || vectors.checksum())),
            Stage::Top(hi) => checksum_line(out, "the top of the smoother", || Ok(hi.to_bits())),
            Stage::Vector {
                round,
                index,
                vector,
            } if index < 2 => checksum_line(
                out,
                &format!("vector {index} of round {}", round + 1),
                || vector.checksum(),
            ),
            Stage::Vector { .. } => Ok(()),
            Stage::Round { round, vectors } => checksum_line(
                out,
                &format!("the vectors after round {}", round + 1),
                || vectors.checksum(),
            )
            .and_then(|()| {
                writeln!(
                    out,
                    "  bootstrap round {}: {:.0} s; {}",
                    round + 1,
                    start.elapsed().as_secs_f64(),
                    memory()
                )
                .map_err(Box::<dyn Error>::from)
            }),
        };
        if let Err(e) = line {
            failed.get_or_insert(e);
        }
    };
    let mut y = match &mg.setup.checkpoint {
        Some(path) => {
            let mut like = H::with_shape(lat, 24, ls, 1, 1);
            like.set_parity(Parity::Even);
            let path = file.parent().map_or(path.clone(), |dir| dir.join(path));
            DeviceSpinor::<T>::from_host_half(&load(&path, &like, keep)?, index)?
        }
        None => filtered_test_vectors(device, (even, odd), lat, index, low, keep, &mut report)?,
    };
    let agg = Aggregation::new(lat, mg.level1.block_size);
    let mut on_device = device_correction(device, &agg, boot.cycle, even, half);
    bootstrap(
        device,
        (even, odd),
        lat,
        &mut y,
        boot,
        &mut on_device,
        &mut report,
    )?;
    drop(report);
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(y)
}

/// Levels below the coarse lattice for the solves of P†DP of `c`, from the test vectors
/// `vectors` on the coarse lattice, as the file says.
fn add_levels<T: DeviceReal, CT: LevelReal>(
    c: &mut DeviceCorrection<'_, T, CT>,
    mg: &super::params::MultigridBlock,
    vectors: &DeviceCoarse<f64>,
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let (super_blocks, bootstrap) = (mg.super_blocks(), mg.level_bootstrap());
    let applications = c.add_levels(vectors, &super_blocks, &bootstrap)?;
    writeln!(
        out,
        "  levels below the coarse lattice {super_blocks:?} (refinement {:?}) on the device: {applications} coarse applications ({:.0} s); {}",
        bootstrap.iter().map(|b| (b.rounds, b.iterations)).collect::<Vec<_>>(),
        start.elapsed().as_secs_f64(),
        memory()
    )?;
    if mg.levels_for_pauli_villars {
        let start = Instant::now();
        let applications = c.add_pv_levels(vectors, &super_blocks)?;
        writeln!(
            out,
            "  the same levels for the Pauli-Villars factor on the device: {applications} coarse applications ({:.0} s); {}",
            start.elapsed().as_secs_f64(),
            memory()
        )?;
    }
    Ok(())
}

/// The cycle of the multigrid for the operator of `ops`, all of it on the device, with the
/// prolongation `basis` and levels below the coarse lattice from `level_vectors`; and the
/// applications of the coarse operators so far, after every cycle.
#[allow(clippy::type_complexity)]
fn two_grid<'a, T: DeviceReal>(
    params: &Params,
    lat: &Arc<Lattice>,
    ops: &'a Operators<T>,
    basis: DeviceBasis,
    level_vectors: Option<&DeviceCoarse<f64>>,
    form: EoForm,
    out: &mut dyn Write,
) -> Result<(impl LinearFn<DeviceSpinor<T>> + 'a, Rc<Cell<usize>>), Box<dyn Error>> {
    let mg = params.multigrid.as_ref().ok_or("no MultiGrid section")?;
    let (cycle_params, _, _) = params.multigrid().ok_or("no multigrid")?;
    let (even, odd, outer) = (&ops.even, &ops.odd, ops.outer());
    let count = Rc::new(Cell::new(0));
    let counted = count.clone();
    let device = ops.plain()?;
    let correction: Correction<'a, T> = match mg.coarse_precision() {
        PrecisionType::Single => {
            let mut c =
                DeviceCorrection::<T, f32>::new(basis, device, outer, form, cycle_params, even)?;
            if let Some(v) = level_vectors {
                add_levels(&mut c, mg, v, out)?;
            }
            Box::new(move |e, r| {
                c.correct(e, r);
                counted.set(c.coarse_applications());
            })
        }
        PrecisionType::Double => {
            let mut c =
                DeviceCorrection::<T, f64>::new(basis, device, outer, form, cycle_params, even)?;
            if let Some(v) = level_vectors {
                add_levels(&mut c, mg, v, out)?;
            }
            Box::new(move |e, r| {
                c.correct(e, r);
                counted.set(c.coarse_applications());
            })
        }
    };
    let cycle = crate::backend::cuda_bridge::multigrid::two_grid(
        outer,
        form,
        (even, odd),
        lat,
        cycle_params,
        correction,
        None,
    )?;
    Ok((cycle, count))
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
    let (form, _) = params.form();
    let s = &params.solver;
    let detailed = matches!(
        params.main.verbose_level,
        super::params::VerboseLevel::Detailed | super::params::VerboseLevel::Paranoiac
    );
    let start = Instant::now();

    // The multigrid, all of it on the device.
    let settings = match (params.multigrid(), &params.multigrid) {
        (Some((cycle, _, boot)), Some(mg)) => Some((cycle, boot, mg)),
        _ => None,
    };
    if settings.is_some_and(|(cycle, boot, _)| {
        !DeviceCorrection::<f64, f32>::supports(&cycle)
            || !DeviceCorrection::<f64, f32>::supports(&boot.cycle)
    }) {
        return Err("the coarse solvers of this multigrid are not on the device".into());
    }
    // Above the coarse lattice it is in single precision, with operators of its own, or in
    // double precision with those of the outer solver.
    let single = settings.is_some_and(|(_, _, mg)| mg.fine_precision() == PrecisionType::Single);
    let ops = Operators::<f64>::new(params, lat, u, settings.is_some() && !single)?;
    let ops_single = match single {
        true => Some(Operators::<f32>::new(params, lat, u, true)?),
        false => None,
    };
    let index = EvenOddIndex::new(lat);
    // The test vectors, in the precision of the multigrid.
    let (y_single, y_double) = match (settings, &ops_single) {
        (None, _) => (None, None),
        (Some(_), Some(o)) => (
            Some(test_vectors(params, file, lat, o, &index, start, out)?),
            None,
        ),
        (Some(_), None) => (
            None,
            Some(test_vectors(params, file, lat, &ops, &index, start, out)?),
        ),
    };
    // The prolongation on the device; the test vectors are not needed after it.
    let mut device_basis = match settings {
        Some((_, _, mg)) => {
            let agg = Aggregation::new(lat, mg.level1.block_size);
            let half = mg.basis_precision() == BasisPrecision::Half;
            Some(match (&ops_single, &y_single, &y_double) {
                (Some(o), Some(y), _) => {
                    let (d, pv) = o.plain()?;
                    DeviceBasis::pv_eo(d, pv, &agg, y, half)?
                }
                (_, _, Some(y)) => {
                    let (d, pv) = ops.plain()?;
                    DeviceBasis::pv_eo(d, pv, &agg, y, half)?
                }
                _ => return Err("no test vectors".into()),
            })
        }
        _ => None,
    };
    if let Some(b) = &device_basis {
        checksum_line(out, "P", || b.checksum())?;
    }
    // The test vectors of the levels below the coarse lattice on the device, P† of the vectors P
    // was made of.
    let level_vectors = match (&device_basis, settings) {
        (Some(b), Some((_, _, mg))) if !mg.super_blocks().is_empty() => {
            Some(match (&ops_single, &y_single, &y_double) {
                (Some(o), Some(y), _) => {
                    let (d, pv) = o.plain()?;
                    b.pv_eo_coarse_vectors(d, pv, y)?.to_precision::<f64>()?
                }
                (_, _, Some(y)) => {
                    let (d, pv) = ops.plain()?;
                    b.pv_eo_coarse_vectors(d, pv, y)?
                }
                _ => return Err("no test vectors".into()),
            })
        }
        _ => None,
    };
    drop((y_single, y_double));
    let (even, odd, outer) = (&ops.even, &ops.odd, ops.outer());
    let operator = || PvRatio::new(outer.0, outer.1, form, even, odd);
    // The applications of the coarse operators by the multigrid so far.
    let mut coarse_count: Option<Rc<Cell<usize>>> = None;
    let mut pre: Option<Box<dyn LinearFn<DeviceSpinor> + '_>> = match s.preconditioner_type {
        PreconditionerType::Chebyshev => {
            let c = s.chebyshev_settings();
            let hi = top_of(&mut operator(), lat, even, 5, 50)?;
            Some(Box::new(Chebyshev::new(
                operator(),
                c.number_of_steps,
                hi / c.interval_ratio,
                hi,
                even,
            )))
        }
        PreconditionerType::None => None,
        PreconditionerType::Multigrid => {
            let basis = device_basis.take().ok_or("the prolongation was not made")?;
            let levels = level_vectors.as_ref();
            match &ops_single {
                Some(o) => {
                    let (cycle, count) = two_grid(params, lat, o, basis, levels, form, out)?;
                    coarse_count = Some(count);
                    Some(Box::new(Convert::new(cycle, &o.even)))
                }
                None => {
                    let (cycle, count) = two_grid(params, lat, &ops, basis, levels, form, out)?;
                    coarse_count = Some(count);
                    Some(Box::new(cycle))
                }
            }
        }
    };
    writeln!(
        out,
        "  setup on the device: {:.0} s; {}",
        start.elapsed().as_secs_f64(),
        memory()
    )?;

    let op = &ops.op;
    let mut a = operator();
    let stop: Stop = s.stop();
    // Mixed precision (CGNE only): CG in single precision on K†K, K the left Schur factor of the
    // operator in single precision on the links of `op`, corrected in double precision.
    let mut single = match s.mixed() {
        Some(inner) => {
            let mut op32 = CudaMdwf::<f32>::converted_from(op, params.mdwf_params(fo.mass()))?;
            op32.set_alpha(op.alpha());
            Some(
                crate::backend::cuda_bridge::hmc_mixed::SingleCuda::from_operator(
                    op32, inner, true,
                ),
            )
        }
        None => None,
    };
    let mixed_stop = Stop {
        max_restarts: s.maximum_number_of_restart.max(50),
        ..stop
    };
    let k = s.number_of_kept_directions;
    let mut source_work = op.work();
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let mut solve = |b: DeviceSpinor,
                     label: &str,
                     out: &mut dyn Write|
     -> Result<DeviceSpinor, Box<dyn Error>> {
        let nv = b.num_vectors();
        let mut rhs = even.zeros_with(nv);
        a.right_hand_side(&DeviceParities, &mut rhs, &b);
        // b waits on the host during the solve.
        let b_host = b.stash()?;
        let mut sol = rhs.zeros_like();
        let coarse_before = coarse_count.as_ref().map(|c| c.get());
        let start = Instant::now();
        let m = pre
            .as_mut()
            .map(|m| m.as_mut() as &mut dyn LinearFn<DeviceSpinor>);
        let mut history: Option<Vec<Vec<f64>>> = None;
        let result = match (s.solver_type, single.as_mut()) {
            (SolverType::Cg, Some(sc)) => {
                let mut k = crate::fermion::SchurFactor::new(op, EoForm::Left, even, odd);
                crate::hmc::mixed_normal(&mut k, sc, &mut sol, &rhs, mixed_stop)
            }
            (t, _) => match t {
                SolverType::Gcr => {
                    let mut g = Gcr::new(stop, k);
                    if detailed {
                        g = g.recording();
                    }
                    let solved = g.solve(&mut a, m, &mut sol, &rhs);
                    history = g.history.take();
                    solved
                }
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
            },
        };
        let seconds = start.elapsed().as_secs_f64();
        let coarse = match (&coarse_count, coarse_before) {
            (Some(c), Some(before)) => format!("; {} coarse applications", c.get() - before),
            _ => String::new(),
        };
        checksum_line(out, &format!("the solution of {label}"), || sol.checksum())?;
        let b = b_host.restore()?;
        // x^(α), then x = Q x^(α), the solution of the plain operator; its residual by that
        // operator on the whole lattice, with three fields there at most.
        let mut x = b.zeros_like();
        a.solution(&DeviceParities, &mut x, &sol, &b);
        drop((rhs, sol));
        if op.alpha() != 1.0 {
            let mut plain = x.zeros_like();
            op.mult_q(&mut plain, &x, false);
            x = plain;
        }
        let worst = {
            let mut r = b.zeros_like();
            op.mult_plain(&mut r, &x);
            r.aypx(&vec![C64::new(-1.0, 0.0); nv], &b);
            let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
            r.norm_sqr(&mut rr);
            b.norm_sqr(&mut bb);
            rr.iter().zip(&bb).fold(0.0_f64, |m, (r, b)| m.max(r / b))
        };
        writeln!(
            out,
            "  {label}: {}{} {} iterations on the device{coarse}; worst |Dx−b|² {worst:.0e}; {seconds:.1} s; {}",
            s.solver_type,
            if single.is_some() { " (mixed precision)" } else { "" },
            iterations_text(&result),
            memory()
        )?;
        super::history_lines(out, label, history);
        Ok(x)
    };

    let Some(src) = &params.source else {
        for seed in seed_lo..=seed_hi {
            let b = DeviceSpinor::from_host(&random_fifth(lat, ls, seed))?;
            solve(b, &format!("seed {seed}"), out)?;
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
        let psi = solve(b, &format!("columns {chunk:?}"), out)?;
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
