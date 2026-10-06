//! Checks: gauge transformations and Coulomb gauge fixing. These hold for any
//! configuration; the comparison with Bridge++ is `clover_two_flavours_coulomb_gauge` in
//! `stage2_reference_runs.rs`.

mod common;

use common::max_abs_diff;
use ruttice::gauge::{
    coulomb_violation, fix_coulomb, gauge_transform, link, plaquette, set_link, CoulombParams,
    GaugeFixError,
};
use ruttice::gauge::{random_gauge, random_su3};
use ruttice::{assemble, local_part, TestRng};
use ruttice::{reduce_sites, Boundary, Field, Gauge, Lattice, LocalComm, Stencil, ND};
use std::sync::Arc;

const GLOBAL: [usize; ND] = [4, 4, 4, 8];
const PERIODIC: [Boundary; ND] = [Boundary::Periodic; ND];

/// One process, one direction split, and several directions split, including time.
const GRIDS: [[usize; ND]; 4] = [[1, 1, 1, 1], [2, 1, 1, 1], [1, 2, 1, 4], [2, 2, 2, 2]];

fn whole() -> Arc<Lattice> {
    Lattice::single_node(GLOBAL)
}

/// Σ_x Σ_{i spatial} Re tr U_i(x): what Coulomb gauge fixing maximises.
fn spatial_trace(u: &Field<f64, Gauge>) -> f64 {
    reduce_sites(u.num_sites(), |x| {
        (0..ND - 1).map(|i| link(u, i, x).trace().re).sum()
    })
}

#[test]
fn gauge_transform_matches_the_reference_helper() {
    let lat = whole();
    let st = Stencil::nearest(&lat, PERIODIC);
    let u = random_gauge(&lat, 11);
    let mut rng = TestRng::new(12);
    let g: Vec<_> = (0..lat.num_sites()).map(|_| random_su3(&mut rng)).collect();
    let mut g_field = Field::<f64, Gauge>::with_shape(&lat, 18, 1, 1, 1);
    for (x, m) in g.iter().enumerate() {
        set_link(&mut g_field, 0, x, m);
    }
    let mut ours = random_gauge(&lat, 11);
    gauge_transform(&mut ours, &g_field, &st);
    let reference = common::gauge_transform(&u, &g, &st);
    assert!(max_abs_diff(&ours, &reference) < 1e-14);
}

/// From a random configuration: the violation ends below the tolerance on every time
/// slice, the plaquette does not change, and the functional has grown.
#[test]
fn coulomb_gauge_is_reached() {
    let lat = whole();
    let st = Stencil::nearest(&lat, PERIODIC);
    let original = random_gauge(&lat, 1);
    let mut u = random_gauge(&lat, 1);
    let params = CoulombParams::default();
    let report = fix_coulomb(&mut u, &params).unwrap();
    eprintln!("converged after {} sweeps", report.iterations);
    assert_eq!(report.iterations % params.measure_interval, 0);
    assert_eq!(report.violation.len(), GLOBAL[3]);
    let violation = coulomb_violation(&u, &st);
    assert_eq!(violation, report.violation);
    assert!(violation.iter().all(|v| *v < params.tolerance_sqr));
    assert!((plaquette(&u, &st) - plaquette(&original, &st)).abs() < 1e-13);
    assert!(spatial_trace(&u) > spatial_trace(&original));
}

#[test]
fn gauge_fixing_reports_failure() {
    let mut u = random_gauge(&whole(), 1);
    let params = CoulombParams {
        max_iterations: 30,
        ..CoulombParams::default()
    };
    let Err(GaugeFixError::NotConverged {
        iterations,
        largest_violation,
    }) = fix_coulomb(&mut u, &params)
    else {
        panic!("30 sweeps from a random configuration should not be enough");
    };
    assert_eq!(iterations, 30);
    assert!(largest_violation > params.tolerance_sqr);
}

/// Fix the gauge of `u` on every process of `grid` and put the result together.
fn distributed_fix(
    u: &Field<f64, Gauge>,
    grid: [usize; ND],
    params: &CoulombParams,
) -> (Field<f64, Gauge>, Option<usize>) {
    let parts = LocalComm::run(GLOBAL, grid, |lat| {
        let mut part = local_part(u, &lat);
        let iterations = fix_coulomb(&mut part, params).ok().map(|r| r.iterations);
        (part, iterations)
    });
    let iterations = parts[0].1;
    assert!(parts.iter().all(|p| p.1 == iterations));
    let fields: Vec<_> = parts.into_iter().map(|p| p.0).collect();
    (assemble(&whole(), &fields), iterations)
}

#[test]
fn same_result_on_several_processes() {
    let u = random_gauge(&whole(), 1);
    let params = CoulombParams::default();
    let (reference, n) = distributed_fix(&u, [1; ND], &params);
    assert!(n.is_some());
    for grid in GRIDS {
        let (fixed, m) = distributed_fix(&u, grid, &params);
        assert_eq!(m, n, "grid {grid:?}");
        let diff = max_abs_diff(&fixed, &reference);
        assert!(diff < 1e-12, "grid {grid:?}: {diff:e}");
    }
}

/// Resets every 100 sweeps; 250 sweeps are not enough to converge, so two resets happen.
/// The random transformations depend on the global site only, so every process grid gives
/// the same configuration, and it is still gauge equivalent to the original.
#[test]
fn resets_do_not_depend_on_the_process_grid() {
    let lat = whole();
    let st = Stencil::nearest(&lat, PERIODIC);
    let u = random_gauge(&lat, 1);
    let params = CoulombParams {
        max_iterations: 250,
        reset_interval: 100,
        naive_iterations: 20,
        ..CoulombParams::default()
    };
    let (reference, n) = distributed_fix(&u, [1; ND], &params);
    assert_eq!(n, None);
    assert!((plaquette(&reference, &st) - plaquette(&u, &st)).abs() < 1e-13);
    // Without the resets the result would differ.
    let no_reset = CoulombParams {
        reset_interval: 1000,
        ..params
    };
    let (smooth, _) = distributed_fix(&u, [1; ND], &no_reset);
    assert!(max_abs_diff(&smooth, &reference) > 1e-3);
    for grid in GRIDS {
        let (fixed, _) = distributed_fix(&u, grid, &params);
        let diff = max_abs_diff(&fixed, &reference);
        assert!(diff < 1e-12, "grid {grid:?}: {diff:e}");
    }
}
