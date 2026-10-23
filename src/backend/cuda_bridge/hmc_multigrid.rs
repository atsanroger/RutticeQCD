//! The multigrid of domain-wall fermions in the solves of HMC on the device: what
//! `hmc::dwf_multigrid` does on the CPU, with the fields, the test vectors, P, the coarse
//! operators and the cycles on the device (`backend::cuda_bridge::multigrid`). The sweeps on
//! D†D and K†K are those of the CPU (`two_solves`, `two_solves_half`), over the two solves on
//! the even sites made here.
//!
//! The solves with D† use the cycle of D, between two Γ5 (`fermion::adjoint_by_gamma5`): there
//! is no cycle of Ã† and no coarse operator of it.
//!
//! With Neff's α (`DwfMultigridSetup::alpha`) the solves with D are those of D^(α) = D Q in the
//! left form, Ã_L^(α), as in the propagator runs, and x = Q x^(α): the setup, P and the coarse
//! operators are those of D, and only the smoother has a range of its own. The solves with D†
//! stay as they are: D^(α)† = Q D†, so α would only scale their right-hand side.
//!
//! Along a trajectory the test vectors are kept (on the host between the solves); P, the coarse
//! operators and the top of the smoother are made for the links of every solve, the vectors improved by the
//! bootstrap when the pseudofermions are drawn and after every `refresh_every` force solves, as
//! on the CPU. With `rebuild_growth` they are kept from solve to solve (`DeviceParts` and the
//! tops) and made again as on the CPU: after a bootstrap, or once the GCR iterations per solve
//! have grown by that factor. P is in half precision as in the propagator runs, the coarse solves in single
//! precision unless `coarse_double`, the cycle in double precision unless `fine_single` (then on
//! the operators in single precision, the links converted on the device; the outer GCR stays
//! in double precision).

use super::multigrid::{bootstrap, device_correction, filtered_test_vectors, top_of, two_grid};
use super::{
    CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceReal, DeviceSpinor, LevelReal,
    Stashed,
};
use crate::fermion::{adjoint_by_gamma5, EoForm, PvRatio};
use crate::hmc::dwf_multigrid::{two_solves, two_solves_half, EvenSolves};
use crate::hmc::DwfMultigridSetup;
use crate::solver::preconditioner::multigrid::{Aggregation, PvEoParams};
use crate::solver::{Convert, Gcr, SolveError, SolveInfo, Solver, Stop};
use crate::{EvenOddIndex, Lattice, LinearFn, Parity, Vector};

type V = DeviceSpinor<f64>;
use std::rc::Rc;
use std::sync::Arc;

/// The sweeps of a solve, over the solves on the even sites.
type Run<'r> =
    dyn FnMut(&mut dyn EvenSolves<DeviceSpinor<f64>>) -> Result<SolveInfo, SolveError> + 'r;

/// What the multigrid of an action keeps along a trajectory: the test vectors, on the host
/// between the solves (`DeviceSpinor::stash`), so that the device holds them only while P is
/// made or the bootstrap runs.
#[derive(Default)]
pub struct DeviceState {
    vectors: Option<Stashed<f64>>,
    /// With `fine_single`: in single precision.
    vectors32: Option<Stashed<f32>>,
    /// With the cache: the parts of the correction of Ã and the top of its smoother, `Kept<CT>`
    /// for the precision of the coarse solves.
    cache: Option<Box<dyn std::any::Any>>,
}

/// The test vectors `kept` in the precision of `ops`, taken out and back on the device: a full
/// setup the first time, a bootstrap from those of before when one is due, else as they are.
fn test_vectors<T: DeviceReal>(
    ops: (&CudaMdwf<T>, &CudaMdwf<T>),
    (even, odd): (&DeviceSpinor<T>, &DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    agg: &Arc<Aggregation>,
    setup: &mut DwfMultigridSetup,
    kept: &mut Option<Stashed<T>>,
    due: bool,
) -> Result<DeviceSpinor<T>, Box<dyn std::error::Error>> {
    let quiet = &mut |_: super::multigrid::Stage<'_, T>| {};
    let mut y = match kept.take().map(|k| k.restore()).transpose()? {
        None => {
            setup.count_setup();
            let index = EvenOddIndex::new(lat);
            filtered_test_vectors(
                ops,
                (even, odd),
                lat,
                &index,
                setup.low_modes,
                setup.test_vectors,
                quiet,
            )?
        }
        Some(y) if due => {
            setup.count_refresh();
            y
        }
        Some(y) => return Ok(y),
    };
    let mut with = device_correction(ops, agg, setup.bootstrap.cycle, even, true);
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

/// What the cache keeps on the device: the parts and the tops of Ã and of Ã_L^(α).
type Kept<CT> = (super::DeviceParts<CT>, f64, f64);

/// D^(α) on the links of `d`.
fn with_alpha<T: DeviceReal>(d: &CudaMdwf<T>, alpha: f64) -> Result<CudaMdwf<T>, super::CudaError> {
    let mut a = CudaMdwf::on_the_links_of(d, *d.params())?;
    a.set_alpha(alpha);
    Ok(a)
}

/// x = (D†D)⁻¹ b for `d` over the Pauli-Villars operator `pv` with the multigrid of `setup`,
/// on the device; `x` holds the initial guess.
pub fn normal_solve(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    x: &mut DeviceSpinor<f64>,
    b: &DeviceSpinor<f64>,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lat.clone();
    with_solves(d, pv, setup, flags, &lat, &mut |solves| {
        two_solves(d, solves, x, b, stop)
    })
}

/// z = (K†K)⁻¹ b on the even sites for K = T_ee⁻¹ Ŝ of `d`, on the device.
pub fn normal_solve_half(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    z: &mut DeviceSpinor<f64>,
    b: &DeviceSpinor<f64>,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let lat = b.lat.clone();
    with_solves(d, pv, setup, flags, &lat, &mut |solves| {
        two_solves_half(d, &lat, solves, z, b, stop)
    })
}

/// The operators in single precision and their fields on the even and odd sites, for
/// `fine_single`.
type Single = (
    CudaMdwf<f32>,
    CudaMdwf<f32>,
    DeviceSpinor<f32>,
    DeviceSpinor<f32>,
);

/// The solves on the even sites of one solve of D†D, both with the cycle M of Ã, made for each
/// solve from the coarse parts and dropped after it.
struct DeviceSolves<'a, CT: LevelReal> {
    d: &'a CudaMdwf<f64>,
    pv: &'a CudaMdwf<f64>,
    even: &'a DeviceSpinor<f64>,
    odd: &'a DeviceSpinor<f64>,
    lat: &'a Arc<Lattice>,
    single: Option<&'a Single>,
    /// With Neff's α: D^(α), in single precision too with `fine_single`, and the top of
    /// Ã_L^(α).
    alpha: Option<(&'a CudaMdwf<f64>, Option<&'a CudaMdwf<f32>>, f64)>,
    gcr: Gcr<DeviceSpinor<f64>>,
    parts: super::DeviceParts<CT>,
    params: PvEoParams,
    hi: f64,
    inner: usize,
    count: usize,
}

impl<'a, CT: LevelReal> DeviceSolves<'a, CT> {
    /// M of Ã, or with `neff` of Ã_L^(α): on the operators in double precision or, with
    /// `fine_single`, on those in single precision with the fields converted at its ends.
    fn cycle(&self, neff: bool) -> Box<dyn LinearFn<V> + 'a> {
        self.try_cycle(neff)
            .unwrap_or_else(|e| panic!("the multigrid on the device failed: {e}"))
    }

    fn try_cycle(
        &self,
        neff: bool,
    ) -> Result<Box<dyn LinearFn<V> + 'a>, Box<dyn std::error::Error>> {
        let alpha = self.alpha.filter(|_| neff);
        let (form, hi) = match alpha {
            Some((_, _, hi)) => (EoForm::Left, hi),
            None => (EoForm::Right, self.hi),
        };
        Ok(match self.single {
            None => {
                let plain = (self.d, self.pv);
                let outer = alpha.map_or(plain, |(d, ..)| (d, self.pv));
                let mut corr = DeviceCorrection::<f64, CT>::from_parts(
                    self.parts.clone(),
                    plain,
                    outer,
                    form,
                    self.params,
                    self.even,
                )?;
                Box::new(two_grid(
                    outer,
                    form,
                    (self.even, self.odd),
                    self.lat,
                    self.params,
                    Box::new(move |e, r| corr.correct(e, r)),
                    Some(hi),
                )?)
            }
            Some((d32, pv32, even32, odd32)) => {
                let plain = (d32, pv32);
                let outer = match alpha {
                    Some((_, d, _)) => (d.expect("D^(α) in single precision"), pv32),
                    None => plain,
                };
                let mut corr = DeviceCorrection::<f32, CT>::from_parts(
                    self.parts.clone(),
                    plain,
                    outer,
                    form,
                    self.params,
                    even32,
                )?;
                let m = two_grid(
                    outer,
                    form,
                    (even32, odd32),
                    self.lat,
                    self.params,
                    Box::new(move |e, r| corr.correct(e, r)),
                    Some(hi),
                )?;
                Box::new(Convert::new(m, even32))
            }
        })
    }

    fn counted(&mut self, info: Result<SolveInfo, SolveError>) -> Result<(), SolveError> {
        self.inner += match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(e) => return Err(e),
        };
        self.count += 1;
        Ok(())
    }
}

impl<CT: LevelReal> EvenSolves<DeviceSpinor<f64>> for DeviceSolves<'_, CT> {
    /// D e = r through Ã and M, or with Neff's α through Ã_L^(α) and its M: e = Q e^(α).
    fn forward(
        &mut self,
        e: &mut DeviceSpinor<f64>,
        r: &DeviceSpinor<f64>,
    ) -> Result<(), SolveError> {
        let (d, form) = match self.alpha {
            Some((d, ..)) => (d, EoForm::Left),
            None => (self.d, EoForm::Right),
        };
        let mut a = PvRatio::new(d, self.pv, form, self.even, self.odd);
        let mut rhs = self.even.zeros_with(r.num_vectors());
        a.right_hand_side(&DeviceParities, &mut rhs, r);
        let mut y = rhs.zeros_like();
        let mut m = self.cycle(self.alpha.is_some());
        let info = self.gcr.solve(&mut a, Some(m.as_mut()), &mut y, &rhs);
        drop(m);
        match self.alpha {
            Some(_) => {
                let mut x = e.zeros_like();
                a.solution(&DeviceParities, &mut x, &y, r);
                d.mult_q(e, &x, false);
            }
            None => a.solution(&DeviceParities, e, &y, r),
        }
        self.counted(info)
    }

    /// D† e = r with Γ5 M Γ5 (`adjoint_by_gamma5`).
    fn adjoint(
        &mut self,
        e: &mut DeviceSpinor<f64>,
        r: &DeviceSpinor<f64>,
    ) -> Result<(), SolveError> {
        let mut m = self.cycle(false);
        let d = self.d;
        let gamma5 = |out: &mut V, inp: &V| d.mult_gamma5_on(out, inp);
        let info = adjoint_by_gamma5(
            (self.d, self.pv),
            &DeviceParities,
            &gamma5,
            m.as_mut(),
            &mut self.gcr,
            (self.even, self.odd),
            e,
            r,
        );
        drop(m);
        self.counted(info)
    }
}

/// `run` with the solves on the even sites of `setup`'s multigrid for the current links, on the
/// device, and the bookkeeping of the setup; the iterations of the result are those of GCR.
fn with_solves(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    flags: (bool, bool),
    lat: &Arc<Lattice>,
    run: &mut Run<'_>,
) -> Result<SolveInfo, SolveError> {
    assert!(
        d.alpha() == 1.0 && pv.alpha() == 1.0,
        "the multigrid of HMC takes D without Neff's α"
    );
    let result = match setup.coarse_double {
        false => with_coarse::<f32>(d, pv, setup, flags, lat, run),
        true => with_coarse::<f64>(d, pv, setup, flags, lat, run),
    };
    result.unwrap_or_else(|e| panic!("the multigrid on the device failed: {e}"))
}

/// `with_solves` with the coarse solves in the precision `CT`.
fn with_coarse<CT: LevelReal>(
    d: &CudaMdwf<f64>,
    pv: &CudaMdwf<f64>,
    setup: &mut DwfMultigridSetup,
    (fresh, md): (bool, bool),
    lat: &Arc<Lattice>,
    run: &mut Run<'_>,
) -> Result<Result<SolveInfo, SolveError>, Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let ls = d.params().ls;
    let even = DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Even)?;
    let odd = DeviceSpinor::<f64>::shape_half(lat, ls, Parity::Odd)?;
    let agg = Aggregation::new(lat, setup.block);
    let due = setup.bootstrap_due(fresh);
    let mut state = setup.device.take().unwrap_or_default();
    let caching = setup.rebuild_growth > 0.0;
    let kept = state
        .cache
        .take()
        .filter(|_| caching && !due && !setup.rebuild_is_due())
        .and_then(|k| k.downcast::<Kept<CT>>().ok());
    let reuse = kept.is_some();
    let neff = setup.alpha != 1.0;
    let d_alpha = neff.then(|| with_alpha(d, setup.alpha)).transpose()?;
    // With `fine_single` the operators in single precision, for the setup (the test vectors are
    // made and kept in single precision, as in the propagator runs) and for the cycles.
    let single = match setup.fine_single {
        true => Some((
            CudaMdwf::<f32>::converted_from(d, *d.params())?,
            CudaMdwf::<f32>::converted_from(pv, *pv.params())?,
            DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Even)?,
            DeviceSpinor::<f32>::shape_half(lat, ls, Parity::Odd)?,
        )),
        false => None,
    };
    let d32_alpha = match (&single, neff) {
        (Some((d32, ..)), true) => Some(with_alpha(d32, setup.alpha)?),
        _ => None,
    };
    let kept = match kept {
        Some(k) => {
            setup.count_solve_for_refresh(false, md);
            setup.count_reuse();
            let (parts, hi, hi_alpha) = *k;
            (parts, (hi, hi_alpha), 0.0)
        }
        None => {
            let (basis, level_vectors, parts_start) = match &single {
                Some((d32, pv32, even32, odd32)) => {
                    let ops = (d32, pv32);
                    let y = test_vectors(
                        ops,
                        (even32, odd32),
                        lat,
                        &agg,
                        setup,
                        &mut state.vectors32,
                        due,
                    )?;
                    setup.count_solve_for_refresh(due, md);
                    setup.count_rebuild();
                    let parts_start = std::time::Instant::now();
                    let b = Rc::new(DeviceBasis::pv_eo(d32, pv32, &agg, &y, true)?);
                    let level_vectors = match setup.levels.count {
                        0 => None,
                        _ => Some(
                            b.pv_eo_coarse_vectors(d32, pv32, &y)?
                                .to_precision::<f64>()?,
                        ),
                    };
                    state.vectors32 = Some(y.stash()?);
                    (b, level_vectors, parts_start)
                }
                None => {
                    let y = test_vectors(
                        (d, pv),
                        (&even, &odd),
                        lat,
                        &agg,
                        setup,
                        &mut state.vectors,
                        due,
                    )?;
                    setup.count_solve_for_refresh(due, md);
                    setup.count_rebuild();
                    let parts_start = std::time::Instant::now();
                    let b = Rc::new(DeviceBasis::pv_eo(d, pv, &agg, &y, true)?);
                    let level_vectors = match setup.levels.count {
                        0 => None,
                        _ => Some(b.pv_eo_coarse_vectors(d, pv, &y)?),
                    };
                    state.vectors = Some(y.stash()?);
                    (b, level_vectors, parts_start)
                }
            };
            let mut c = DeviceCorrection::<f64, CT>::shared(
                basis,
                (d, pv),
                (d, pv),
                EoForm::Right,
                setup.cycle,
                &even,
            )?;
            // The levels below the coarse lattice, from P† of the vectors P was made of; kept
            // in the parts with the coarse operators they were built from.
            if let Some(v) = &level_vectors {
                let l = setup.levels;
                c.add_levels(v, &l.super_blocks[..l.count], &l.bootstrap[..l.count])?;
                if l.for_pauli_villars {
                    c.add_pv_levels(v, &l.super_blocks[..l.count])?;
                }
            }
            drop(level_vectors);
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
            let hi_alpha = match &d_alpha {
                Some(d) => top(d, EoForm::Left)?,
                None => hi,
            };
            (
                c.parts(),
                (hi, hi_alpha),
                parts_start.elapsed().as_secs_f64(),
            )
        }
    };
    // P and the coarse parts count as the rebuild, the vectors as the setup.
    let (parts, (hi, hi_alpha), parts_seconds) = kept;
    let setup_seconds = start.elapsed().as_secs_f64() - parts_seconds;
    let start = std::time::Instant::now();
    let rebuild_seconds = parts_seconds + start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let mut solves = DeviceSolves {
        d,
        pv,
        even: &even,
        odd: &odd,
        lat,
        single: single.as_ref(),
        alpha: d_alpha.as_ref().map(|d| (d, d32_alpha.as_ref(), hi_alpha)),
        gcr: Gcr::new(setup.inner, setup.directions),
        parts: parts.clone(),
        params: setup.cycle,
        hi,
        inner: 0,
        count: 0,
    };
    let result = run(&mut solves);
    let (inner, count) = (solves.inner, solves.count);
    drop(solves);
    if caching {
        setup.after_solve(reuse, inner as f64 / count.max(1) as f64);
        let kept: Kept<CT> = (parts, hi, hi_alpha);
        state.cache = Some(Box::new(kept));
    }
    setup.device = Some(state);
    setup.record(
        setup_seconds,
        rebuild_seconds,
        start.elapsed().as_secs_f64(),
        count,
    );
    Ok(result.map(|mut info| {
        info.iterations = inner;
        info
    }))
}
