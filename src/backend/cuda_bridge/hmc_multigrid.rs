//! The multigrid of domain-wall fermions in the solves of HMC on the device: the device as a
//! `hmc::dwf_multigrid::Backend`. When to set up, bootstrap or rebuild, the cache, the retries
//! and the solves with D and D† are those of every backend (`hmc::dwf_multigrid`); here are the
//! fields, the test vectors, P, the coarse parts and the cycles on the device
//! (`backend::cuda_bridge::multigrid`).
//!
//! The test vectors wait on the host between the solves (`DeviceSpinor::stash`), so that the
//! device holds them only while P is made or the bootstrap runs; the cycles are made for each
//! solve from the coarse parts and dropped after it. P is in half precision as in the
//! propagator runs unless `basis_single`, the coarse solves in single precision unless
//! `coarse_double`, the cycle in double precision unless `fine_single` (then on the operators in
//! single precision, the links converted on the device; the outer GCR stays in double
//! precision).

use super::multigrid::{
    bootstrap, device_correction_with_levels, filtered_test_vectors, top_of, two_grid,
};
use super::{
    CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceParts, DeviceReal, DeviceSpinor,
    LevelReal, Stashed,
};
use crate::fermion::{EoForm, PvRatio};
use crate::hmc::dwf_multigrid::{
    two_solves, two_solves_half, with_solves, Backend, Cycles, DwfMultigridSetup,
};
use crate::solver::preconditioner::multigrid::{Aggregation, PvEoParams};
use crate::solver::{Convert, SolveError, SolveInfo, Stop};
use crate::{EvenOddIndex, Lattice, LinearFn, Parity};
use std::error::Error;
use std::rc::Rc;
use std::sync::Arc;

type V = DeviceSpinor<f64>;

/// x = (D†D)⁻¹ b for `d` over the Pauli-Villars operator `pv` with the multigrid of `setup`,
/// on the device; `x` holds the initial guess.
pub fn normal_solve(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    x: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lat.clone();
    with_solves(&OnDevice, (d, pv), setup, flags, &lat, |solves| {
        two_solves(d, solves, x, b, stop)
    })
}

/// z = (K†K)⁻¹ b on the even sites for K = T_ee⁻¹ Ŝ of `d`, on the device.
pub fn normal_solve_half(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    z: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lat.clone();
    with_solves(&OnDevice, (d, pv), setup, flags, &lat, |solves| {
        two_solves_half(d, &lat, solves, z, b, stop)
    })
}

/// The device as a `Backend`.
pub(crate) struct OnDevice;

/// The test vectors on the host between the solves, in the precision of the cycle.
pub(crate) enum DeviceVectors {
    Double(Stashed<f64>),
    Single(Stashed<f32>),
}

/// The coarse parts in the precision of the coarse solves, and the tops of the smoothers of Ã
/// and of Ã_L^(α).
pub(crate) enum DeviceKept {
    Single(DeviceParts<f32>, (f64, f64)),
    Double(DeviceParts<f64>, (f64, f64)),
}

/// D^(α) on the links of `d`.
fn with_alpha<T: DeviceReal>(d: &CudaMdwf<T>, alpha: f64) -> Result<CudaMdwf<T>, super::CudaError> {
    let mut a = CudaMdwf::on_the_links_of(d, *d.params())?;
    a.set_alpha(alpha);
    Ok(a)
}

/// The test vectors for `ops` from those `kept`: a full setup without any, a bootstrap from
/// them when `due`, else as they are.
fn vectors_in<T: DeviceReal>(
    ops: (&CudaMdwf<T>, &CudaMdwf<T>),
    (even, odd): (&DeviceSpinor<T>, &DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    agg: &Arc<Aggregation>,
    setup: &DwfMultigridSetup,
    kept: Option<&Stashed<T>>,
    due: bool,
) -> Result<DeviceSpinor<T>, Box<dyn Error>> {
    let quiet = &mut |_: super::multigrid::Stage<'_, T>| {};
    let mut y = match kept {
        None => filtered_test_vectors(
            ops,
            (even, odd),
            lat,
            &EvenOddIndex::new(lat),
            setup.low_modes,
            setup.test_vectors,
            quiet,
        )?,
        Some(k) if due => k.restore()?,
        Some(k) => return Ok(k.restore()?),
    };
    let (super_blocks, refine) = setup.level_lists();
    let for_pv = setup.levels.for_pauli_villars;
    let mut with = device_correction_with_levels(
        ops,
        agg,
        setup.bootstrap.cycle,
        even,
        true,
        (super_blocks, refine, for_pv),
    );
    bootstrap(
        ops,
        (even, odd),
        lat,
        &mut y,
        setup.bootstrap,
        &mut with,
        quiet,
    )?;
    Ok(y)
}

/// The test vectors of the levels below the coarse lattice.
type LevelVectors = super::DeviceCoarse<f64>;

/// P made of `y` for `ops`, and the test vectors of the levels below the coarse lattice (P† of
/// the vectors P was made of) when the setup asks for levels.
fn basis_in<T: DeviceReal>(
    (d, pv): (&CudaMdwf<T>, &CudaMdwf<T>),
    agg: &Arc<Aggregation>,
    setup: &DwfMultigridSetup,
    y: &DeviceSpinor<T>,
) -> Result<(Rc<DeviceBasis>, Option<LevelVectors>), Box<dyn Error>> {
    let basis = Rc::new(DeviceBasis::pv_eo(d, pv, agg, y, !setup.basis_single)?);
    let levels = match setup.levels.count {
        0 => None,
        _ => Some(
            basis
                .pv_eo_coarse_vectors(d, pv, y)?
                .to_precision::<f64>()?,
        ),
    };
    Ok((basis, levels))
}

/// The coarse parts on `basis` for the plain operators `(d, pv)`, with the levels below the
/// coarse lattice made of `level_vectors`.
fn coarse_parts<CT: LevelReal>(
    basis: Rc<DeviceBasis>,
    (d, pv): (&CudaMdwf<f64>, &CudaMdwf<f64>),
    even: &V,
    setup: &DwfMultigridSetup,
    level_vectors: Option<&super::DeviceCoarse<f64>>,
) -> Result<DeviceParts<CT>, Box<dyn Error>> {
    let mut c = DeviceCorrection::<f64, CT>::shared(
        basis,
        (d, pv),
        (d, pv),
        EoForm::Right,
        setup.cycle,
        even,
    )?;
    if let Some(v) = level_vectors {
        let (super_blocks, refine) = setup.level_lists();
        c.add_levels(v, super_blocks, refine)?;
        if setup.levels.for_pauli_villars {
            c.add_pv_levels(v, super_blocks)?;
        }
    }
    Ok(c.parts())
}

impl Backend for OnDevice {
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

    fn rebuild(
        &self,
        (d, pv): (&CudaMdwf<f64>, &CudaMdwf<f64>),
        d_alpha: Option<&CudaMdwf<f64>>,
        lat: &Arc<Lattice>,
        setup: &DwfMultigridSetup,
        kept: Option<DeviceVectors>,
        due: bool,
    ) -> Result<(DeviceVectors, DeviceKept, f64), Box<dyn Error>> {
        let start = std::time::Instant::now();
        let ls = d.params().ls;
        let even = DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Even)?;
        let odd = DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Odd)?;
        let agg = Aggregation::new(lat, setup.block);
        // With `fine_single` the test vectors are made and kept in single precision, on the
        // operators converted to it, as in the propagator runs.
        let (vectors, basis, level_vectors, seconds) = match setup.fine_single {
            false => {
                let kept = match &kept {
                    Some(DeviceVectors::Double(k)) => Some(k),
                    _ => None,
                };
                let y = vectors_in((d, pv), (&even, &odd), lat, &agg, setup, kept, due)?;
                let seconds = start.elapsed().as_secs_f64();
                let (basis, levels) = basis_in((d, pv), &agg, setup, &y)?;
                (DeviceVectors::Double(y.stash()?), basis, levels, seconds)
            }
            true => {
                let d32 = CudaMdwf::<f32>::converted_from(d, *d.params())?;
                let pv32 = CudaMdwf::<f32>::converted_from(pv, *pv.params())?;
                let even32 = DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Even)?;
                let odd32 = DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Odd)?;
                let kept = match &kept {
                    Some(DeviceVectors::Single(k)) => Some(k),
                    _ => None,
                };
                let ops = (&d32, &pv32);
                let y = vectors_in(ops, (&even32, &odd32), lat, &agg, setup, kept, due)?;
                let seconds = start.elapsed().as_secs_f64();
                let (basis, levels) = basis_in(ops, &agg, setup, &y)?;
                (DeviceVectors::Single(y.stash()?), basis, levels, seconds)
            }
        };
        let top = |d: &CudaMdwf<f64>, form: EoForm| {
            let mut a = PvRatio::new(d, pv, form, &even, &odd);
            top_of(
                &mut a,
                lat,
                &even,
                setup.cycle.seed,
                setup.cycle.power_steps,
            )
        };
        let hi = top(d, EoForm::Right)?;
        let tops = match d_alpha {
            Some(a) => (hi, top(a, EoForm::Left)?),
            None => (hi, hi),
        };
        let levels = level_vectors.as_ref();
        let parts = match setup.coarse_double {
            false => DeviceKept::Single(coarse_parts(basis, (d, pv), &even, setup, levels)?, tops),
            true => DeviceKept::Double(coarse_parts(basis, (d, pv), &even, setup, levels)?, tops),
        };
        Ok((vectors, parts, seconds))
    }

    fn cycles<'a>(
        &'a self,
        (d, pv): (&'a CudaMdwf<f64>, &'a CudaMdwf<f64>),
        d_alpha: Option<&'a CudaMdwf<f64>>,
        lat: &Arc<Lattice>,
        setup: &DwfMultigridSetup,
        parts: &'a DeviceKept,
    ) -> Result<Box<dyn Cycles<V> + 'a>, Box<dyn Error>> {
        let ls = d.params().ls;
        let single = match setup.fine_single {
            true => {
                let d32 = CudaMdwf::<f32>::converted_from(d, *d.params())?;
                let alpha = d_alpha.map(|_| with_alpha(&d32, setup.alpha)).transpose()?;
                Some(Single {
                    pv: CudaMdwf::<f32>::converted_from(pv, *pv.params())?,
                    d: d32,
                    alpha,
                    even: DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Even)?,
                    odd: DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Odd)?,
                })
            }
            false => None,
        };
        Ok(Box::new(DeviceCycles {
            d,
            pv,
            d_alpha,
            even: DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Even)?,
            odd: DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Odd)?,
            lat: lat.clone(),
            single,
            parts,
            params: setup.cycle,
        }))
    }
}

/// The operators in single precision on the links of now and their fields on the even and odd
/// sites, for `fine_single`.
struct Single {
    d: CudaMdwf<f32>,
    pv: CudaMdwf<f32>,
    alpha: Option<CudaMdwf<f32>>,
    even: DeviceSpinor<f32>,
    odd: DeviceSpinor<f32>,
}

/// The cycles of one solve on the device, made of the kept parts for the operators on the links
/// of now.
struct DeviceCycles<'a> {
    d: &'a CudaMdwf<f64>,
    pv: &'a CudaMdwf<f64>,
    d_alpha: Option<&'a CudaMdwf<f64>>,
    even: V,
    odd: V,
    lat: Arc<Lattice>,
    single: Option<Single>,
    parts: &'a DeviceKept,
    params: PvEoParams,
}

impl DeviceCycles<'_> {
    /// M of Ã, or with `neff` of Ã_L^(α), with the coarse solves in the precision `CT`.
    fn with_coarse<CT: LevelReal>(
        &self,
        parts: &DeviceParts<CT>,
        (hi, hi_alpha): (f64, f64),
        neff: bool,
    ) -> Result<Box<dyn LinearFn<V> + '_>, Box<dyn Error>> {
        let alpha = self.d_alpha.filter(|_| neff);
        let (form, hi) = match alpha {
            Some(_) => (EoForm::Left, hi_alpha),
            None => (EoForm::Right, hi),
        };
        Ok(match &self.single {
            None => {
                let plain = (self.d, self.pv);
                let outer = alpha.map_or(plain, |d| (d, self.pv));
                let mut corr = DeviceCorrection::<f64, CT>::from_parts(
                    parts.clone(),
                    plain,
                    outer,
                    form,
                    self.params,
                    &self.even,
                )?;
                Box::new(two_grid(
                    outer,
                    form,
                    (&self.even, &self.odd),
                    &self.lat,
                    self.params,
                    Box::new(move |e, r| corr.correct(e, r)),
                    Some(hi),
                )?)
            }
            Some(s) => {
                let plain = (&s.d, &s.pv);
                let outer = match alpha {
                    Some(_) => (s.alpha.as_ref().expect("D^(α) in single precision"), &s.pv),
                    None => plain,
                };
                let mut corr = DeviceCorrection::<f32, CT>::from_parts(
                    parts.clone(),
                    plain,
                    outer,
                    form,
                    self.params,
                    &s.even,
                )?;
                let m = two_grid(
                    outer,
                    form,
                    (&s.even, &s.odd),
                    &self.lat,
                    self.params,
                    Box::new(move |e, r| corr.correct(e, r)),
                    Some(hi),
                )?;
                Box::new(Convert::new(m, &s.even))
            }
        })
    }
}

impl Cycles<V> for DeviceCycles<'_> {
    fn cycle(&self, neff: bool) -> Box<dyn LinearFn<V> + '_> {
        match self.parts {
            DeviceKept::Single(p, tops) => self.with_coarse(p, *tops, neff),
            DeviceKept::Double(p, tops) => self.with_coarse(p, *tops, neff),
        }
        .unwrap_or_else(|e| panic!("the multigrid on the device failed: {e}"))
    }
}
