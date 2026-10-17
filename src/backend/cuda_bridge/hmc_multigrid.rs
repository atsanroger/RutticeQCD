//! The multigrid of domain-wall fermions in the solves of HMC on the device: what
//! `hmc::dwf_multigrid` does on the CPU, with the fields, the test vectors, P, the coarse
//! operators and the cycles on the device (`backend::cuda_bridge::multigrid`). The sweeps on
//! D†D and K†K are those of the CPU (`two_solves`, `two_solves_half`), over the two solves on
//! the even sites made here.
//!
//! Along a trajectory the test vectors are kept on the device; P, the coarse operators and the
//! tops of the smoothers are made for the links of every solve, the vectors improved by the
//! bootstrap when the pseudofermions are drawn and after every `refresh_every` force solves, as
//! on the CPU. The fine fields are in double precision, P in half precision as in the
//! propagator runs, the coarse solves in single precision unless `coarse_double`.

use super::multigrid::{
    bootstrap, device_correction, filtered_test_vectors, two_grid, DeviceCycle,
};
use super::{CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceSpinor, LevelReal};
use crate::fermion::{adjoint_right_hand_side, adjoint_solution, EoForm, PvRatio};
use crate::hmc::dwf_multigrid::{two_solves, two_solves_half, EvenSolves};
use crate::hmc::DwfMultigridSetup;
use crate::solver::preconditioner::multigrid::{Aggregation, CoarseKrylov, PvEoParams};
use crate::solver::{Gcr, SolveError, SolveInfo, Solver, Stop};
use crate::{EvenOddIndex, Lattice, Parity, Vector};
use std::rc::Rc;
use std::sync::Arc;

/// The sweeps of a solve, over the solves on the even sites.
type Run<'r> =
    dyn FnMut(&mut dyn EvenSolves<DeviceSpinor<f64>>) -> Result<SolveInfo, SolveError> + 'r;

/// What the multigrid of an action keeps on the device along a trajectory: the test vectors.
#[derive(Default)]
pub struct DeviceState {
    vectors: Option<DeviceSpinor<f64>>,
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

/// The solves on the even sites of one solve of D†D, with their cycles.
struct DeviceSolves<'a> {
    d: &'a CudaMdwf<f64>,
    pv: &'a CudaMdwf<f64>,
    even: &'a DeviceSpinor<f64>,
    odd: &'a DeviceSpinor<f64>,
    gcr: Gcr<DeviceSpinor<f64>>,
    m: DeviceCycle<'a, f64>,
    m_dag: DeviceCycle<'a, f64>,
    inner: usize,
    count: usize,
}

impl DeviceSolves<'_> {
    fn counted(&mut self, info: Result<SolveInfo, SolveError>) -> Result<(), SolveError> {
        self.inner += match info {
            Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
            Err(e) => return Err(e),
        };
        self.count += 1;
        Ok(())
    }
}

impl EvenSolves<DeviceSpinor<f64>> for DeviceSolves<'_> {
    /// D e = r through Ã and M.
    fn forward(
        &mut self,
        e: &mut DeviceSpinor<f64>,
        r: &DeviceSpinor<f64>,
    ) -> Result<(), SolveError> {
        let mut a = PvRatio::new(self.d, self.pv, EoForm::Right, self.even, self.odd);
        let mut rhs = self.even.zeros_with(r.num_vectors());
        a.right_hand_side(&DeviceParities, &mut rhs, r);
        let mut y = rhs.zeros_like();
        let info = self.gcr.solve(&mut a, Some(&mut self.m), &mut y, &rhs);
        a.solution(&DeviceParities, e, &y, r);
        self.counted(info)
    }

    /// D† e = r through Ã† and M†.
    fn adjoint(
        &mut self,
        e: &mut DeviceSpinor<f64>,
        r: &DeviceSpinor<f64>,
    ) -> Result<(), SolveError> {
        let mut rhs = self.even.zeros_with(r.num_vectors());
        adjoint_right_hand_side(self.d, &DeviceParities, &mut rhs, r);
        let mut a_dag = PvRatio::new(self.pv, self.d, EoForm::Right, self.even, self.odd);
        let mut y = rhs.zeros_like();
        let info = self
            .gcr
            .solve(&mut a_dag, Some(&mut self.m_dag), &mut y, &rhs);
        adjoint_solution(self.d, self.pv, &DeviceParities, e, &y, r);
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
    // The test vectors: a full setup the first time, a bootstrap from those of before when
    // one is due, else as they are.
    let due = setup.bootstrap_due(fresh);
    let mut state = setup.device.take().unwrap_or_default();
    let quiet = &mut |_: super::multigrid::Stage<'_, f64>| {};
    let made = match state.vectors.take() {
        None => {
            setup.count_setup();
            let index = EvenOddIndex::new(lat);
            let mut y = filtered_test_vectors(
                (d, pv),
                (&even, &odd),
                lat,
                &index,
                setup.low_modes,
                setup.test_vectors,
                quiet,
            )?;
            let mut with = device_correction((d, pv), &agg, setup.bootstrap.cycle, &even, true);
            bootstrap(
                (d, pv),
                (&even, &odd),
                lat,
                &mut y,
                setup.bootstrap,
                &mut with,
                quiet,
            )?;
            y
        }
        Some(mut y) if due => {
            setup.count_refresh();
            let mut with = device_correction((d, pv), &agg, setup.bootstrap.cycle, &even, true);
            bootstrap(
                (d, pv),
                (&even, &odd),
                lat,
                &mut y,
                setup.bootstrap,
                &mut with,
                quiet,
            )?;
            y
        }
        Some(y) => y,
    };
    setup.count_solve_for_refresh(due, md);
    let setup_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    // P, shared by the cycles of Ã and Ã†, and the cycles.
    let basis = Rc::new(DeviceBasis::pv_eo(d, pv, &agg, &made, true)?);
    state.vectors = Some(made);
    let dual = PvEoParams {
        pv_solver: CoarseKrylov::BiCgStab,
        ..setup.cycle
    };
    let mut c = DeviceCorrection::<f64, CT>::shared(
        basis.clone(),
        (d, pv),
        (d, pv),
        EoForm::Right,
        setup.cycle,
        &even,
    )?;
    let mut c_dag =
        DeviceCorrection::<f64, CT>::shared(basis, (pv, d), (pv, d), EoForm::Right, dual, &even)?;
    let m = two_grid(
        (d, pv),
        EoForm::Right,
        (&even, &odd),
        lat,
        setup.cycle,
        Box::new(move |e, r| c.correct(e, r)),
        None,
    )?;
    let m_dag = two_grid(
        (pv, d),
        EoForm::Right,
        (&even, &odd),
        lat,
        dual,
        Box::new(move |e, r| c_dag.correct(e, r)),
        None,
    )?;
    let rebuild_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let mut solves = DeviceSolves {
        d,
        pv,
        even: &even,
        odd: &odd,
        gcr: Gcr::new(setup.inner, setup.directions),
        m,
        m_dag,
        inner: 0,
        count: 0,
    };
    let result = run(&mut solves);
    let (inner, count) = (solves.inner, solves.count);
    drop(solves);
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
