//! Hybrid Monte Carlo against Bridge++: its quenched HMC (`sample_HMC_quench`, plaquette
//! action, β = 6.0, leapfrog of 200 steps of 0.0015, exp(iP) to order 8, M-series numbers
//! from trajectory number 0) from a cold start on 12^4. The numbers are those of its log
//! (`hmc_12_b6.0/hmc.log`, printed with 8 digits after the point, 6 for ΔH): the momenta of
//! the first two trajectories, and where the first one ends.

use ruttice::hmc::{
    cold_start, draw_momenta, force_statistics, kinetic_energy, Leapfrog, Mseries, PlaquetteAction,
    QuenchedHmc,
};
use ruttice::{Lattice, LocalComm, ND};

fn hmc() -> QuenchedHmc {
    QuenchedHmc {
        action: PlaquetteAction { beta: 6.0 },
        integrator: Leapfrog {
            step: 0.0015,
            steps: 200,
            exp_order: 8,
        },
        metropolis: true,
    }
}

/// The momenta of the first trajectory: the generator and the order of the draws are those
/// of Bridge++, on any process grid.
#[test]
fn momenta_are_those_of_bridge() {
    let lat = Lattice::single_node([12; ND]);
    let p = draw_momenta(&lat, &mut Mseries::new(0));
    let h = kinetic_energy(&p);
    eprintln!("H_kin {h:.8} (Bridge++ 331513.77780977)");
    assert!((h - 331513.77780977).abs() < 2e-7);
    for grid in [[1, 1, 1, 2], [2, 1, 3, 1]] {
        let split = LocalComm::run([12; ND], grid, |lat| {
            kinetic_energy(&draw_momenta(&lat, &mut Mseries::new(0)))
        });
        for s in split {
            assert!((s - h).abs() < 1e-7 * h, "grid {grid:?}: {s}");
        }
    }
}

/// The first two trajectories of the Bridge++ run: the first ends with these energies and
/// plaquette and is rejected; the second starts with these momenta.
#[test]
fn first_trajectories_are_those_of_bridge() {
    let lat = Lattice::single_node([12; ND]);
    let mut u = cold_start(&lat);
    let mut rng = Mseries::new(0);
    let hmc = hmc();
    // The force after the first step of U, as Bridge++ prints it (imdc = 1).
    {
        let mut p = draw_momenta(&lat, &mut Mseries::new(0));
        let mut v = cold_start(&lat);
        let lf = hmc.integrator;
        lf.update_p(0.5 * lf.step, &hmc.action, &v, &mut p);
        lf.update_u(lf.step, &p, &mut v);
        let (ave, max, dev) = force_statistics(&hmc.action.force(&v));
        eprintln!("force after one step: Fave {ave:.6} Fmax {max:.6} Fdev {dev:.6} (Bridge++ 0.060466 0.147245 0.015320)");
        assert!((ave - 0.060466).abs() < 1e-6 && (max - 0.147245).abs() < 1e-6);
        assert!((dev - 0.015320).abs() < 1e-6);
    }
    let t = hmc.trajectory(&mut u, &mut rng);
    eprintln!("{t:?}");
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() < tol;
    assert!(close(t.kinetic[0], 331513.77780977, 2e-7));
    assert!(close(t.gauge[0], 0.0, 1e-9));
    assert!(
        close(t.kinetic[1], 145028.93961247, 1e-4),
        "{}",
        t.kinetic[1]
    );
    assert!(close(t.gauge[1], 186486.69884699, 1e-4), "{}", t.gauge[1]);
    assert!(close(t.plaquette_trial, 0.75018393, 1e-8));
    assert!(close(t.delta_h, 1.860650, 1e-5), "{}", t.delta_h);
    assert!(close(t.random, 0.680828, 1e-6));
    assert!(!t.accepted);
    assert!(close(t.plaquette, 1.0, 1e-12));
    let p = draw_momenta(&lat, &mut rng);
    let h = kinetic_energy(&p);
    eprintln!("second trajectory: H_kin {h:.8} (Bridge++ 331541.78146248)");
    assert!(close(h, 331541.78146248, 2e-7));
}
