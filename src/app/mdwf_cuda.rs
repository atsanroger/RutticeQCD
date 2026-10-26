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

use super::mdwf::{iterations_text, load, solve_sources, SourceFields};
use super::memory_line;
use super::params::{Params, PrecisionType, PreconditionerType, SolverType};
use crate::backend::cuda_bridge::mdwf_backend::OnDevice;
use crate::backend::cuda_bridge::multigrid::top_of;
use crate::backend::cuda_bridge::{
    CudaError, CudaMdwf, DeviceCorrection, DeviceParities, DeviceReal, DeviceSpinor,
};
use crate::fermion::{FermionOp, PvRatio};
use crate::solver::preconditioner::multigrid::{MdwfBackend, SetupStage};
use crate::solver::{Chebyshev, Stop};
use crate::{random_fifth, Field, Gauge, Half, Lattice, LinearFn, Parity, Spinor, Vector, C64};
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Whether `RUTTICE_CHECKSUM` asks for the checksums of the stages of a run.
fn checksums() -> bool {
    std::env::var_os("RUTTICE_CHECKSUM").is_some()
}

/// With `RUTTICE_CHECKSUM` set, a line with the checksum of what a stage of the run made, to
/// compare runs bit by bit and find where they part.
fn checksum_line(
    out: &mut dyn Write,
    what: &str,
    sum: impl FnOnce() -> Result<u64, CudaError>,
) -> Result<(), Box<dyn Error>> {
    if checksums() {
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

pub(super) fn run(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    seeds: (u64, u64),
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

    // The multigrid, all of it on the device (`mdwf_backend::OnDevice`, as in HMC).
    let mg = match (params.multigrid(), &params.multigrid) {
        (Some(_), Some(mg)) => Some(mg),
        _ => None,
    };
    let settings = mg.map(|mg| mg.mdwf_settings(PrecisionType::Single));
    if lat.comm.size() > 1 {
        if let Some(why) = settings
            .as_ref()
            .and_then(|m| m.refused_on_several(lat.local))
        {
            return Err(format!("Main.backend cuda_bridge on several processes: {why}").into());
        }
    }
    if settings.is_some_and(|m| {
        !DeviceCorrection::<f64, f32>::supports(&m.cycle)
            || !DeviceCorrection::<f64, f32>::supports(&m.bootstrap.cycle)
    }) {
        return Err("the coarse solvers of this multigrid are not on the device".into());
    }
    let ops = Operators::<f64>::new(params, lat, u, settings.is_some())?;
    let (even, odd, outer) = (&ops.even, &ops.odd, ops.outer());
    let outers = [(outer.0, outer.1, form)];
    let backend = OnDevice;
    // The test vectors from a checkpoint, or filtered; improved by the bootstrap, P, the levels
    // and the coarse parts made of them; reported as they are made.
    let built = match (mg, &settings) {
        (Some(mg), Some(m)) => {
            let kept = match &mg.setup.checkpoint {
                Some(path) => {
                    let mut like = H::with_shape(lat, 24, ls, 1, 1);
                    like.set_parity(Parity::Even);
                    let path = file.parent().map_or(path.clone(), |dir| dir.join(path));
                    Some(backend.vectors_from_host(&load(&path, &like, m.test_vectors)?, m)?)
                }
                None => None,
            };
            let mut failed: Option<std::io::Error> = None;
            let mut report = |stage: SetupStage<'_>| {
                let line = match stage {
                    SetupStage::Filtered(theta) => writeln!(
                        out,
                        "  filter on the device: Ritz values of Ã†Ã {:.3e} .. {:.3e} ({:.0} s); {}",
                        theta[0],
                        theta[theta.len() - 1],
                        start.elapsed().as_secs_f64(),
                        memory()
                    ),
                    SetupStage::Round(round) => writeln!(
                        out,
                        "  bootstrap round {}: {:.0} s; {}",
                        round + 1,
                        start.elapsed().as_secs_f64(),
                        memory()
                    ),
                    SetupStage::Levels {
                        pauli_villars: false,
                        applications,
                        seconds,
                    } => writeln!(
                        out,
                        "  levels below the coarse lattice {:?} (refinement {:?}) on the device: {applications} coarse applications ({seconds:.0} s); {}",
                        mg.super_blocks(),
                        mg.level_bootstrap().iter().map(|b| (b.rounds, b.iterations)).collect::<Vec<_>>(),
                        memory()
                    ),
                    SetupStage::Levels {
                        pauli_villars: true,
                        applications,
                        seconds,
                    } => writeln!(
                        out,
                        "  the same levels for the Pauli-Villars factor on the device: {applications} coarse applications ({seconds:.0} s); {}",
                        memory()
                    ),
                    SetupStage::Checksum(what, sum) => match checksums() {
                        true => writeln!(out, "  checksum of {what}: {:016x}", sum()),
                        false => Ok(()),
                    },
                };
                if let Err(e) = line {
                    failed.get_or_insert(e);
                }
            };
            let plain = ops.plain()?;
            let (_, parts, _) = backend.rebuild(plain, &outers, lat, m, kept, true, &mut report)?;
            if let Some(e) = failed {
                return Err(e.into());
            }
            Some((parts, *m))
        }
        _ => None,
    };
    let cycles = match &built {
        Some((parts, m)) => Some(backend.cycles(ops.plain()?, &outers, lat, m, parts)?),
        None => None,
    };
    let operator = || PvRatio::new(outer.0, outer.1, form, even, odd);
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
        PreconditionerType::Multigrid => Some(
            cycles
                .as_ref()
                .ok_or("the multigrid was not made")?
                .cycle(0),
        ),
    };
    writeln!(
        out,
        "  setup on the device: {:.0} s; {}",
        start.elapsed().as_secs_f64(),
        memory()
    )?;
    writeln!(
        out,
        "  basis lifts: {:?}",
        crate::solver::preconditioner::multigrid::Lifts::from_env()
    )?;

    let op = &ops.op;
    let mut a = operator();
    let stop: Stop = s.stop();
    // Mixed precision (CGNE only): CG in single precision on K†K, K the left Schur factor of the
    // operator in single precision on the links of `op`, with reliable updates in double
    // precision each time |r|² has fallen by `drop_sqr`.
    let mut single = match s.mixed() {
        Some(drop_sqr) => {
            let mut op32 = CudaMdwf::<f32>::converted_from(op, params.mdwf_params(fo.mass()))?;
            op32.set_alpha(op.alpha());
            Some((
                crate::fermion::SingleEven::new(op32, lat, drop_sqr),
                drop_sqr,
            ))
        }
        None => None,
    };
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let solve = |b: DeviceSpinor,
                 label: &str,
                 out: &mut dyn Write|
     -> Result<DeviceSpinor, Box<dyn Error>> {
        let nv = b.num_vectors();
        let mut rhs = even.zeros_with(nv);
        a.right_hand_side(&DeviceParities, &mut rhs, &b);
        // b waits on the host during the solve.
        let b_host = b.stash()?;
        let mut sol = rhs.zeros_like();
        let coarse_before = cycles.as_ref().map(|c| c.coarse_applications());
        let counted = crate::meter::Counts::now();
        let start = Instant::now();
        let m = pre
            .as_mut()
            .map(|m| m.as_mut() as &mut dyn LinearFn<DeviceSpinor>);
        let (result, history) = match (s.solver_type, single.as_mut()) {
            (SolverType::Cg, Some((sc, drop_sqr))) => {
                let solved = sc.reliable_normal(&mut a, &mut sol, &rhs, stop, *drop_sqr);
                (solved, None)
            }
            _ => s.solve(&mut a, m, &mut sol, &rhs, detailed),
        };
        let seconds = start.elapsed().as_secs_f64();
        let work = crate::meter::Counts::now().since(&counted);
        let coarse = match (&cycles, coarse_before) {
            (Some(c), Some(before)) => {
                format!("; {} coarse applications", c.coarse_applications() - before)
            }
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
            "  {label}: {}{} {} iterations on the device{coarse}; worst |Dx−b|² {worst:.0e}; {seconds:.1} s, {}; {}",
            s.solver_type,
            if single.is_some() { " (mixed precision)" } else { "" },
            iterations_text(&result),
            work.line(seconds),
            memory()
        )?;
        super::history_lines(out, label, history);
        Ok(x)
    };

    let mut fields = DeviceFields {
        lat,
        op,
        work: op.work(),
        ls,
    };
    solve_sources(params, lat, seeds, &mut fields, solve, out)
}

/// The fields of a run on the device, made with the operator `op` (its plain part).
struct DeviceFields<'a> {
    lat: &'a Arc<Lattice>,
    op: &'a CudaMdwf<f64>,
    work: <CudaMdwf<f64> as FermionOp<DeviceSpinor>>::Work,
    ls: usize,
}

impl SourceFields for DeviceFields<'_> {
    type Fifth = DeviceSpinor;

    fn random(&mut self, seed: u64) -> Result<DeviceSpinor, Box<dyn Error>> {
        Ok(DeviceSpinor::from_host(&random_fifth(
            self.lat, self.ls, seed,
        ))?)
    }

    fn source(&mut self, eta: &Field<f64, Spinor>) -> Result<DeviceSpinor, Box<dyn Error>> {
        let eta = DeviceSpinor::from_host(eta)?;
        let mut b = DeviceSpinor::zeros_fifth(self.lat, self.ls)?.zeros_with(eta.num_vectors());
        self.op.source_5d(&mut self.work, &mut b, &eta);
        Ok(b)
    }

    fn project(
        &mut self,
        psi: &DeviceSpinor,
        eta: &Field<f64, Spinor>,
    ) -> Result<(Field<f64, Spinor>, Field<f64, Spinor>), Box<dyn Error>> {
        let like = DeviceSpinor::from_host(eta)?;
        let (mut q, mut m) = (like.zeros_like(), like.zeros_like());
        self.op.propagator_4d(&mut q, psi);
        self.op.midpoint_4d(&mut m, psi);
        Ok((q.to_host()?, m.to_host()?))
    }
}
