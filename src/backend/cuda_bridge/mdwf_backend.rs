//! The device as a backend of the domain-wall multigrid
//! (`solver::preconditioner::multigrid::MdwfBackend`), for the propagator runs and HMC: the
//! fields, the test vectors, P, the coarse parts and the cycles on the device
//! (`backend::cuda_bridge::multigrid`).
//!
//! The test vectors wait on the host between the solves of HMC (`DeviceSpinor::stash`), so that
//! the device holds them only while P is made or the bootstrap runs. P is in half precision
//! unless `basis_single`, the coarse solves in single precision unless `coarse_double`, the
//! cycle in double precision unless `fine_single` (then on the operators converted to single
//! precision on the device; the outer solver stays in double precision).

use super::multigrid::{
    bootstrap, device_correction_with_levels, filtered_test_vectors, top_of, two_grid, Stage,
};
use super::{
    CudaMdwf, DeviceBasis, DeviceCoarse, DeviceCorrection, DeviceParities, DeviceParts, DeviceReal,
    DeviceSpinor, LevelReal, Stashed,
};
use crate::fermion::{EoForm, PvRatio};
use crate::solver::preconditioner::multigrid::{
    Aggregation, Cycles, MdwfBackend, MdwfMultigridSettings, Outer, PvEoParams, SetupStage,
};
use crate::solver::Convert;
use crate::{EvenOddIndex, Field, Half, Lattice, LinearFn, Parity, Spinor};
use std::cell::Cell;
use std::error::Error;
use std::rc::Rc;
use std::sync::Arc;

type V = DeviceSpinor<f64>;

/// The device as an `MdwfBackend`.
pub struct OnDevice;

/// The test vectors on the host, in the precision of the cycle.
pub enum DeviceVectors {
    Double(Stashed<f64>),
    Single(Stashed<f32>),
}

/// The coarse parts in the precision of the coarse solves, and the tops of the smoothers of the
/// outer operators.
pub enum DeviceKept {
    Single(DeviceParts<f32>, Vec<f64>),
    Double(DeviceParts<f64>, Vec<f64>),
}

/// D^(α) on the links of `d`.
fn with_alpha<T: DeviceReal>(d: &CudaMdwf<T>, alpha: f64) -> Result<CudaMdwf<T>, super::CudaError> {
    let mut a = CudaMdwf::on_the_links_of(d, *d.params())?;
    a.set_alpha(alpha);
    Ok(a)
}

/// `report` of a stage of the device's setup, its checksums computed when asked.
fn reported<T: DeviceReal>(report: &mut dyn FnMut(SetupStage<'_>), stage: Stage<'_, T>) {
    let sum = |v: &DeviceSpinor<T>| v.checksum().expect("a checksum on the device");
    match stage {
        Stage::Filtered { theta, vectors } => {
            report(SetupStage::Filtered(theta));
            report(SetupStage::Checksum("the filtered vectors", &|| {
                sum(vectors)
            }));
        }
        Stage::Top(hi) => report(SetupStage::Checksum("the top of the smoother", &|| {
            hi.to_bits()
        })),
        Stage::Vector {
            round,
            index,
            vector,
        } if index < 2 => {
            let what = format!("vector {index} of round {}", round + 1);
            report(SetupStage::Checksum(&what, &|| sum(vector)));
        }
        Stage::Vector { .. } => {}
        Stage::Round { round, vectors } => {
            let what = format!("the vectors after round {}", round + 1);
            report(SetupStage::Checksum(&what, &|| sum(vectors)));
            report(SetupStage::Round(round));
        }
    }
}

/// The test vectors for `ops` from those `kept`: a full setup without any, a bootstrap from
/// them when `due`, else as they are.
#[allow(clippy::too_many_arguments)]
fn vectors_in<T: DeviceReal>(
    ops: (&CudaMdwf<T>, &CudaMdwf<T>),
    (even, odd): (&DeviceSpinor<T>, &DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    agg: &Arc<Aggregation>,
    settings: &MdwfMultigridSettings,
    kept: Option<&Stashed<T>>,
    due: bool,
    report: &mut dyn FnMut(SetupStage<'_>),
) -> Result<DeviceSpinor<T>, Box<dyn Error>> {
    let mut stages = |s: Stage<'_, T>| reported(report, s);
    let mut y = match kept {
        None => filtered_test_vectors(
            ops,
            (even, odd),
            lat,
            &EvenOddIndex::new(lat),
            settings.low_modes,
            settings.test_vectors,
            &mut stages,
        )?,
        Some(k) if due => k.restore()?,
        Some(k) => return Ok(k.restore()?),
    };
    let (super_blocks, refine) = settings.level_lists();
    let for_pv = settings.levels.for_pauli_villars;
    let mut with = device_correction_with_levels(
        ops,
        agg,
        settings.bootstrap.cycle,
        even,
        !settings.basis_single,
        (super_blocks, refine, for_pv),
    );
    bootstrap(
        ops,
        (even, odd),
        lat,
        &mut y,
        settings.bootstrap,
        &mut with,
        &mut stages,
    )?;
    Ok(y)
}

/// `op` in single precision on its links, Neff's α kept (`converted_from` takes the parameters,
/// not α).
fn single(op: &CudaMdwf<f64>) -> Result<CudaMdwf<f32>, super::CudaError> {
    let mut c = CudaMdwf::<f32>::converted_from(op, *op.params())?;
    c.set_alpha(op.alpha());
    Ok(c)
}

/// The tops of the smoothers of `outers`, in their precision: 1.1 |λ_max| by the power iteration
/// (`multigrid::top_of`).
fn tops_in<T: DeviceReal>(
    outers: &[Outer<'_, CudaMdwf<T>>],
    (even, odd): (&DeviceSpinor<T>, &DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    params: PvEoParams,
) -> Result<Vec<f64>, super::CudaError> {
    outers
        .iter()
        .map(|(a, b, form)| {
            let mut ratio = PvRatio::new(*a, *b, *form, even, odd);
            top_of(&mut ratio, lat, even, params.seed, params.power_steps)
        })
        .collect()
}

/// The test vectors of the levels below the coarse lattice.
type LevelVectors = DeviceCoarse<f64>;

/// P made of `y` for `ops`, and the test vectors of the levels below the coarse lattice (P† of
/// the vectors P was made of) when the settings ask for levels.
fn basis_in<T: DeviceReal>(
    (d, pv): (&CudaMdwf<T>, &CudaMdwf<T>),
    agg: &Arc<Aggregation>,
    settings: &MdwfMultigridSettings,
    y: &DeviceSpinor<T>,
    report: &mut dyn FnMut(SetupStage<'_>),
) -> Result<(Rc<DeviceBasis>, Option<LevelVectors>), Box<dyn Error>> {
    let basis = Rc::new(DeviceBasis::pv_eo(d, pv, agg, y, !settings.basis_single)?);
    report(SetupStage::Checksum("P", &|| {
        basis.checksum().expect("a checksum on the device")
    }));
    let levels = match settings.levels.count {
        0 => None,
        _ => Some(
            basis
                .pv_eo_coarse_vectors(d, pv, y)?
                .to_precision::<f64>()?,
        ),
    };
    Ok((basis, levels))
}

/// The coarse parts on `basis` for the plain operators `(d, pv)` in the precision of the cycle
/// (P†DP probed with the operators the cycle works with, as on the CPU), with the levels below
/// the coarse lattice made of `level_vectors`.
fn coarse_parts<T: DeviceReal, CT: LevelReal>(
    basis: Rc<DeviceBasis>,
    (d, pv): (&CudaMdwf<T>, &CudaMdwf<T>),
    even: &DeviceSpinor<T>,
    settings: &MdwfMultigridSettings,
    level_vectors: Option<&LevelVectors>,
    report: &mut dyn FnMut(SetupStage<'_>),
) -> Result<DeviceParts<CT>, Box<dyn Error>> {
    let params = settings.cycle;
    let mut c =
        DeviceCorrection::<T, CT>::shared(basis, (d, pv), (d, pv), EoForm::Right, params, even)?;
    if let Some(v) = level_vectors {
        let (super_blocks, refine) = settings.level_lists();
        let start = std::time::Instant::now();
        let applications = c.add_levels(v, super_blocks, refine)?;
        report(SetupStage::Levels {
            pauli_villars: false,
            applications,
            seconds: start.elapsed().as_secs_f64(),
        });
        if settings.levels.for_pauli_villars {
            let start = std::time::Instant::now();
            let applications = c.add_pv_levels(v, super_blocks)?;
            report(SetupStage::Levels {
                pauli_villars: true,
                applications,
                seconds: start.elapsed().as_secs_f64(),
            });
        }
    }
    Ok(c.parts())
}

impl MdwfBackend for OnDevice {
    type Op = CudaMdwf<f64>;
    type Field = V;
    type Half = V;
    type Split = DeviceParities;
    type Vectors = DeviceVectors;
    type Parts = DeviceKept;

    fn split(&self) -> &DeviceParities {
        &DeviceParities
    }

    fn halves(&self, op: &CudaMdwf<f64>, lat: &Arc<Lattice>) -> (V, V) {
        let half = |parity| {
            DeviceSpinor::<f64>::shape_half(lat, op.params().ls, parity)
                .expect("fields on the device")
        };
        (half(Parity::Even), half(Parity::Odd))
    }

    fn gamma5(op: &CudaMdwf<f64>, out: &mut V, inp: &V) {
        op.mult_gamma5_on(out, inp);
    }

    fn mult_q(op: &CudaMdwf<f64>, out: &mut V, inp: &V) {
        op.mult_q(out, inp, false);
    }

    fn with_alpha(op: &CudaMdwf<f64>, alpha: f64) -> Result<CudaMdwf<f64>, Box<dyn Error>> {
        Ok(with_alpha(op, alpha)?)
    }

    fn vectors_from_host(
        &self,
        y: &Field<f64, Spinor, Half>,
        settings: &MdwfMultigridSettings,
    ) -> Result<DeviceVectors, Box<dyn Error>> {
        let index = EvenOddIndex::new(y.lattice());
        Ok(match settings.fine_single {
            false => {
                DeviceVectors::Double(DeviceSpinor::<f64>::from_host_half(y, &index)?.stash()?)
            }
            true => DeviceVectors::Single(DeviceSpinor::<f32>::from_host_half(y, &index)?.stash()?),
        })
    }

    fn rebuild(
        &self,
        (d, pv): (&CudaMdwf<f64>, &CudaMdwf<f64>),
        outers: &[Outer<'_, CudaMdwf<f64>>],
        lat: &Arc<Lattice>,
        settings: &MdwfMultigridSettings,
        kept: Option<DeviceVectors>,
        due: bool,
        report: &mut dyn FnMut(SetupStage<'_>),
    ) -> Result<(DeviceVectors, DeviceKept, f64), Box<dyn Error>> {
        let start = std::time::Instant::now();
        let ls = d.params().ls;
        let even = DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Even)?;
        let odd = DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Odd)?;
        let agg = Aggregation::new(lat, settings.block);
        // P and the coarse parts in the precision of the cycle: with `fine_single` the test
        // vectors are made and kept in single precision too, on the operators converted to it.
        macro_rules! made {
            ($ops:expr, $outers:expr, $halves:expr, $kept:expr, $stash:path) => {{
                let (ops, halves) = ($ops, $halves);
                let tops = tops_in($outers, halves, lat, settings.cycle)?;
                let y = vectors_in(ops, halves, lat, &agg, settings, $kept, due, report)?;
                let seconds = start.elapsed().as_secs_f64();
                let (basis, levels) = basis_in(ops, &agg, settings, &y, report)?;
                let levels = levels.as_ref();
                let parts = match settings.coarse_double {
                    false => DeviceKept::Single(
                        coarse_parts(basis, ops, halves.0, settings, levels, report)?,
                        tops,
                    ),
                    true => DeviceKept::Double(
                        coarse_parts(basis, ops, halves.0, settings, levels, report)?,
                        tops,
                    ),
                };
                ($stash(y.stash()?), parts, seconds)
            }};
        }
        let (vectors, parts, seconds) = match settings.fine_single {
            false => {
                let kept = match &kept {
                    Some(DeviceVectors::Double(k)) => Some(k),
                    _ => None,
                };
                made!((d, pv), outers, (&even, &odd), kept, DeviceVectors::Double)
            }
            true => {
                let d32 = single(d)?;
                let pv32 = single(pv)?;
                let outers32 = outers
                    .iter()
                    .map(|(a, b, form)| Ok((single(a)?, single(b)?, *form)))
                    .collect::<Result<Vec<_>, super::CudaError>>()?;
                let outers32: Vec<_> = outers32.iter().map(|(a, b, f)| (a, b, *f)).collect();
                let even32 = DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Even)?;
                let odd32 = DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Odd)?;
                let kept = match &kept {
                    Some(DeviceVectors::Single(k)) => Some(k),
                    _ => None,
                };
                made!(
                    (&d32, &pv32),
                    &outers32,
                    (&even32, &odd32),
                    kept,
                    DeviceVectors::Single
                )
            }
        };
        Ok((vectors, parts, seconds))
    }

    fn cycles<'a>(
        &'a self,
        (d, pv): (&'a CudaMdwf<f64>, &'a CudaMdwf<f64>),
        outers: &[Outer<'a, CudaMdwf<f64>>],
        lat: &Arc<Lattice>,
        settings: &MdwfMultigridSettings,
        parts: &'a DeviceKept,
    ) -> Result<Box<dyn Cycles<V> + 'a>, Box<dyn Error>> {
        let ls = d.params().ls;
        let single = match settings.fine_single {
            true => Some(Single {
                d: single(d)?,
                pv: single(pv)?,
                outers: outers
                    .iter()
                    .map(|(a, b, form)| Ok((single(a)?, single(b)?, *form)))
                    .collect::<Result<_, super::CudaError>>()?,
                even: DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Even)?,
                odd: DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Odd)?,
            }),
            false => None,
        };
        Ok(Box::new(DeviceCycles {
            plain: (d, pv),
            outers: outers.to_vec(),
            even: DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Even)?,
            odd: DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Odd)?,
            lat: lat.clone(),
            single,
            parts,
            params: settings.cycle,
            count: Rc::new(Cell::new(0)),
        }))
    }
}

/// The operators in single precision on the links of now and their fields on the even and odd
/// sites, for `fine_single`.
struct Single {
    d: CudaMdwf<f32>,
    pv: CudaMdwf<f32>,
    outers: Vec<(CudaMdwf<f32>, CudaMdwf<f32>, EoForm)>,
    even: DeviceSpinor<f32>,
    odd: DeviceSpinor<f32>,
}

/// The cycles of the outer operators on the device, made of the kept parts for the operators on
/// the links of now.
struct DeviceCycles<'a> {
    plain: (&'a CudaMdwf<f64>, &'a CudaMdwf<f64>),
    outers: Vec<Outer<'a, CudaMdwf<f64>>>,
    even: V,
    odd: V,
    lat: Arc<Lattice>,
    single: Option<Single>,
    parts: &'a DeviceKept,
    params: PvEoParams,
    /// Applications of the coarse operators by the cycles so far.
    count: Rc<Cell<usize>>,
}

impl DeviceCycles<'_> {
    /// The cycle of outer operator `which`, with the coarse solves in the precision `CT`.
    fn with_coarse<CT: LevelReal>(
        &self,
        parts: &DeviceParts<CT>,
        tops: &[f64],
        which: usize,
    ) -> Result<Box<dyn LinearFn<V> + '_>, Box<dyn Error>> {
        let hi = tops[which];
        let count = self.count.clone();
        Ok(match &self.single {
            None => {
                let (a, b, form) = self.outers[which];
                let mut corr = DeviceCorrection::<f64, CT>::from_parts(
                    parts.clone(),
                    self.plain,
                    (a, b),
                    form,
                    self.params,
                    &self.even,
                )?;
                Box::new(two_grid(
                    (a, b),
                    form,
                    (&self.even, &self.odd),
                    &self.lat,
                    self.params,
                    Box::new(move |e, r| {
                        let before = corr.coarse_applications();
                        corr.correct(e, r);
                        count.set(count.get() + corr.coarse_applications() - before);
                    }),
                    Some(hi),
                )?)
            }
            Some(s) => {
                let (a, b, form) = &s.outers[which];
                let mut corr = DeviceCorrection::<f32, CT>::from_parts(
                    parts.clone(),
                    (&s.d, &s.pv),
                    (a, b),
                    *form,
                    self.params,
                    &s.even,
                )?;
                let m = two_grid(
                    (a, b),
                    *form,
                    (&s.even, &s.odd),
                    &self.lat,
                    self.params,
                    Box::new(move |e, r| {
                        let before = corr.coarse_applications();
                        corr.correct(e, r);
                        count.set(count.get() + corr.coarse_applications() - before);
                    }),
                    Some(hi),
                )?;
                Box::new(Convert::new(m, &s.even))
            }
        })
    }
}

impl Cycles<V> for DeviceCycles<'_> {
    fn cycle(&self, which: usize) -> Box<dyn LinearFn<V> + '_> {
        match self.parts {
            DeviceKept::Single(p, tops) => self.with_coarse(p, tops, which),
            DeviceKept::Double(p, tops) => self.with_coarse(p, tops, which),
        }
        .unwrap_or_else(|e| panic!("the multigrid on the device failed: {e}"))
    }

    fn coarse_applications(&self) -> usize {
        self.count.get()
    }
}
