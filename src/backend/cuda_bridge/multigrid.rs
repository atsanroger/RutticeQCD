//! The setup and the cycle of the multigrid of domain-wall fermions on the device, for any
//! program: the test vectors (Chebyshev-filtered subspace iteration on Ã†Ã, then the bootstrap
//! with the cycle of the vectors of the round before), the top of the smoother's interval and the
//! two-grid cycle of a coarse correction. The prolongation and the coarse correction themselves
//! are `DeviceBasis::pv_eo` and `DeviceCorrection`.
//!
//! What a program reports of the setup (times, Ritz values, checksums) it does in `report`; the
//! coarse correction of a round of the bootstrap may come from elsewhere (the CPU, for settings
//! the device does not have) through `bootstrap`'s `with_correction`.

use super::{
    CudaError, CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceReal, DeviceSpinor,
};
use crate::fermion::{EoForm, PvRatio, SplitParity};
use crate::solver::preconditioner::multigrid::{
    bootstrap_vector, low_modes, orthonormalise_all, random_even, Aggregation, BootstrapParams,
    LevelBootstrap, LowModeParams, PvEoParams, TwoGrid,
};
use crate::solver::{largest_magnitude, Chebyshev};
use crate::{
    random_spinor, EvenOddIndex, Field, Lattice, LinearFn, Normal, Parity, Spinor, Vector,
};
use std::error::Error;
use std::sync::Arc;

/// A coarse correction as `TwoGrid` takes it: e = C r on the even sites.
pub type Correction<'c, T> = Box<dyn FnMut(&mut DeviceSpinor<T>, &DeviceSpinor<T>) + 'c>;

/// The vectors a round of the bootstrap returns.
pub type Grown<T> = Result<DeviceSpinor<T>, Box<dyn Error>>;

/// What a round of `bootstrap` does with the coarse correction of the current vectors: given it,
/// the top of the smoother if known, and the vectors themselves, it returns the new vectors.
pub type Round<'r, T> = dyn FnMut(Correction<'_, T>, Option<f64>, DeviceSpinor<T>) -> Grown<T> + 'r;

/// How `bootstrap` gets the coarse correction of the current vectors: it makes it and hands it
/// with the vectors to the round.
pub type WithCorrection<'w, T> = dyn FnMut(DeviceSpinor<T>, &mut Round<'_, T>) -> Grown<T> + 'w;

/// The cycle of the multigrid: the Chebyshev smoother on Ã and a coarse correction.
pub type DeviceCycle<'a, T> =
    TwoGrid<DeviceSpinor<T>, PvRatio<'a, CudaMdwf<T>, DeviceSpinor<T>>, Correction<'a, T>>;

/// What the setup has made, as it makes it.
pub enum Stage<'s, T: DeviceReal> {
    /// The filtered vectors and the Ritz values of Ã†Ã on them.
    Filtered {
        theta: &'s [f64],
        vectors: &'s DeviceSpinor<T>,
    },
    /// The top of the smoother's interval in a round of the bootstrap.
    Top(f64),
    /// Vector `index` after round `round` of the bootstrap (from 0).
    Vector {
        round: usize,
        index: usize,
        vector: &'s DeviceSpinor<T>,
    },
    /// All the vectors after round `round`.
    Round {
        round: usize,
        vectors: &'s DeviceSpinor<T>,
    },
}

/// 1.1 |λ_max| of `a`, the top of the interval of a smoother: by the power iteration from the
/// random vector of `seed` on the even sites, as on the CPU.
pub fn top_of<T: DeviceReal, A: LinearFn<DeviceSpinor<T>>>(
    a: &mut A,
    lat: &Arc<Lattice>,
    even: &DeviceSpinor<T>,
    seed: u64,
    steps: usize,
) -> Result<f64, CudaError> {
    let ls = even.ls();
    let mut host = Field::<f64, Spinor>::with_shape(lat, 24, ls, 1, 1);
    host.as_mut_slice()
        .copy_from_slice(random_spinor(lat, ls, seed).as_slice());
    let mut start = even.zeros_with(1);
    DeviceParities.extract(
        &mut start,
        &DeviceSpinor::<T>::from_host(&host)?,
        Parity::Even,
    );
    Ok(1.1 * largest_magnitude(a, &mut start, steps))
}

/// The lowest `keep` of `low.vectors` vectors of Chebyshev-filtered subspace iteration on Ã†Ã
/// (Ã of `d` over `pv`, the right form) from random vectors on the even sites, and the Ritz
/// values of all of them.
pub fn filtered_test_vectors<T: DeviceReal>(
    (d, pv): (&CudaMdwf<T>, &CudaMdwf<T>),
    (even, odd): (&DeviceSpinor<T>, &DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    index: &EvenOddIndex,
    low: LowModeParams,
    keep: usize,
    report: &mut dyn FnMut(Stage<'_, T>),
) -> Result<DeviceSpinor<T>, CudaError> {
    let random = random_even(lat, index, even.ls(), low.vectors, low.seed);
    let x = DeviceSpinor::<T>::from_host_half(&random, index)?;
    let ratio = || PvRatio::new(d, pv, EoForm::Right, even, odd);
    let (x, theta) = low_modes(|c: &DeviceSpinor<T>| Normal::new(ratio(), c), x, low);
    report(Stage::Filtered {
        theta: &theta,
        vectors: &x,
    });
    Ok(x.select(&(0..keep.min(x.num_vectors())).collect::<Vec<_>>()))
}

/// The coarse correction made on the device from the vectors `y` (`DeviceBasis::pv_eo`, P in
/// half precision with `half`), with coarse solves in single precision: what `bootstrap` takes
/// as `with_correction` when the device has the settings `cycle`.
pub fn device_correction<'o, T: DeviceReal>(
    ops: (&'o CudaMdwf<T>, &'o CudaMdwf<T>),
    agg: &'o Arc<Aggregation>,
    cycle: PvEoParams,
    even: &'o DeviceSpinor<T>,
    half: bool,
) -> impl FnMut(DeviceSpinor<T>, &mut Round<'_, T>) -> Grown<T> + 'o {
    device_correction_with_levels(ops, agg, cycle, even, half, (&[], &[], false))
}

/// `device_correction` with the levels below the coarse lattice for its coarse solves
/// (`DeviceCorrection::add_levels`, `add_pv_levels` with `for_pv`), from P† of the vectors of
/// each round: super-blocks and refinements as the files give them. At small masses the coarse
/// solves of the bootstrap are most of its time without them (24⁴, m 0.001: 65 of 101 s).
pub fn device_correction_with_levels<'o, T: DeviceReal>(
    (d, pv): (&'o CudaMdwf<T>, &'o CudaMdwf<T>),
    agg: &'o Arc<Aggregation>,
    cycle: PvEoParams,
    even: &'o DeviceSpinor<T>,
    half: bool,
    (super_blocks, refine, for_pv): (&'o [usize], &'o [LevelBootstrap], bool),
) -> impl FnMut(DeviceSpinor<T>, &mut Round<'_, T>) -> Grown<T> + 'o {
    let mut rounds = 0;
    move |y, round| {
        let basis = DeviceBasis::pv_eo(d, pv, agg, &y, half)?;
        // From the second round on: the coarse solves of the first, on the filtered vectors,
        // were as fast without them (24⁴, m 0.001: 19.8 s with, 19.7 s without, and 7 s to make
        // them).
        let level_vectors = match super_blocks.is_empty() || rounds == 0 {
            true => None,
            false => Some(basis.pv_eo_coarse_vectors(d, pv, &y)?),
        };
        rounds += 1;
        let mut c =
            DeviceCorrection::<T, f32>::new(basis, (d, pv), (d, pv), EoForm::Right, cycle, even)?;
        if let Some(v) = &level_vectors {
            c.add_levels(v, super_blocks, refine)?;
            if for_pv {
                c.add_pv_levels(v, super_blocks)?;
            }
        }
        drop(level_vectors);
        round(Box::new(|e, r| c.correct(e, r)), None, y)
    }
}

/// The bootstrap: `params.rounds` rounds of `params.steps` steps of FGMRES on Ã x = y_j with the
/// cycle of the vectors of the round before, for every vector, then orthonormalised. Each round
/// hands the current vectors to `with_correction`, which makes their coarse correction and
/// passes it on to the round (`Round`) with the vectors; the vectors wait on the host while the
/// cycle works on one at a time.
pub fn bootstrap<T: DeviceReal>(
    (d, pv): (&CudaMdwf<T>, &CudaMdwf<T>),
    (even, odd): (&DeviceSpinor<T>, &DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    y: &mut DeviceSpinor<T>,
    params: BootstrapParams,
    with_correction: &mut WithCorrection<'_, T>,
    report: &mut dyn FnMut(Stage<'_, T>),
) -> Result<(), Box<dyn Error>> {
    let ratio = || PvRatio::new(d, pv, EoForm::Right, even, odd);
    let cycle = params.cycle;
    for round in 0..params.rounds {
        let current = std::mem::replace(y, y.zeros_with(0));
        let mut run = |correction: Correction<'_, T>, hi: Option<f64>, vectors: DeviceSpinor<T>| {
            let hi = match hi {
                Some(hi) => hi,
                None => top_of(&mut ratio(), lat, even, cycle.seed, cycle.power_steps)?,
            };
            report(Stage::Top(hi));
            let smoother = Chebyshev::new(
                ratio(),
                cycle.chebyshev_steps,
                hi / cycle.chebyshev_ratio,
                hi,
                even,
            );
            let mut two_grid = TwoGrid::new(smoother, correction, even);
            let mut a = ratio();
            let mut stash = vectors.stash()?;
            for j in 0..stash.num_vectors() {
                let x = bootstrap_vector(&mut a, &mut two_grid, &stash.vector(j)?, params.steps);
                report(Stage::Vector {
                    round,
                    index: j,
                    vector: &x,
                });
                stash.set_vector(j, &x)?;
            }
            drop(two_grid);
            Ok(stash.restore()?)
        };
        *y = with_correction(current, &mut run)?;
        orthonormalise_all(y);
        report(Stage::Round { round, vectors: y });
    }
    Ok(())
}

/// The cycle for Ã of `outer` in the form `form`: the Chebyshev smoother of `cycle` on
/// [hi / ratio, hi] (hi given, or by the power iteration) and the coarse correction `correction`.
pub fn two_grid<'a, T: DeviceReal>(
    outer: (&'a CudaMdwf<T>, &'a CudaMdwf<T>),
    form: EoForm,
    (even, odd): (&'a DeviceSpinor<T>, &'a DeviceSpinor<T>),
    lat: &Arc<Lattice>,
    cycle: PvEoParams,
    correction: Correction<'a, T>,
    hi: Option<f64>,
) -> Result<DeviceCycle<'a, T>, CudaError> {
    let operator = || PvRatio::new(outer.0, outer.1, form, even, odd);
    let hi = match hi {
        Some(hi) => hi,
        None => top_of(&mut operator(), lat, even, cycle.seed, cycle.power_steps)?,
    };
    let smoother = Chebyshev::new(
        operator(),
        cycle.chebyshev_steps,
        hi / cycle.chebyshev_ratio,
        hi,
        even,
    );
    Ok(TwoGrid::new(smoother, correction, even))
}
