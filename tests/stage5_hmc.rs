//! Hybrid Monte Carlo against Bridge++: its quenched HMC (`sample_HMC_quench`, plaquette
//! action, β = 6.0, leapfrog of 200 steps of 0.0015, exp(iP) to order 8, M-series numbers
//! from trajectory number 0) from a cold start on 12^4. The numbers are those of its log
//! (`hmc_12_b6.0/hmc.log`, printed with 8 digits after the point, 6 for ΔH): the momenta of
//! the first two trajectories, and where the first one ends.

use ruttice::hmc::{
    cold_start, draw_momenta, exp_update, force_statistics, kinetic_energy, Hmc, Mseries,
    PlaquetteAction, Scheme, Term,
};
use ruttice::{Lattice, LocalComm, ND};

mod common;

const BETA: f64 = 6.0;

/// Quenched HMC with the plaquette action: one level of `steps` leapfrog steps of `step`,
/// exp(iP) to order 8, with the Metropolis test.
fn quenched(step: f64, steps: usize) -> Hmc<'static> {
    let gauge = Term {
        level: 0,
        action: Box::new(PlaquetteAction { beta: BETA }),
    };
    Hmc::new(
        vec![gauge],
        Scheme::Leapfrog,
        vec![steps],
        8,
        step * steps as f64,
    )
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
    let step = 0.0015;
    let mut hmc = quenched(step, 200);
    // The force after the first step of U, as Bridge++ prints it (imdc = 1).
    {
        let action = PlaquetteAction { beta: BETA };
        let mut p = draw_momenta(&lat, &mut Mseries::new(0));
        let mut v = cold_start(&lat);
        let f = action.force(&v);
        for (pv, fv) in p.as_mut_slice().iter_mut().zip(f.as_slice()) {
            *pv += 0.5 * step * fv;
        }
        exp_update(step, &p, &mut v, 8);
        let (ave, max, dev) = force_statistics(&action.force(&v));
        eprintln!("force after one step: Fave {ave:.6} Fmax {max:.6} Fdev {dev:.6} (Bridge++ 0.060466 0.147245 0.015320)");
        assert!((ave - 0.060466).abs() < 1e-6 && (max - 0.147245).abs() < 1e-6);
        assert!((dev - 0.015320).abs() < 1e-6);
    }
    let t = hmc.trajectory(&mut u, &mut rng);
    eprintln!("{t:?}");
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() < tol;
    assert!(close(t.kinetic[0], 331513.77780977, 2e-7));
    let gauge = t.actions[0];
    assert!(close(gauge[0], 0.0, 1e-9));
    assert!(
        close(t.kinetic[1], 145028.93961247, 1e-4),
        "{}",
        t.kinetic[1]
    );
    assert!(close(gauge[1], 186486.69884699, 1e-4), "{}", gauge[1]);
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

/// The Bridge++ test `test_HMC_Quenched_Leapfrog` (its configuration `conf_04040408.txt`,
/// plaquette action β = 6.0, 10 leapfrog steps of 0.01, M-series numbers from trajectory
/// number 200000, one trajectory): the plaquette after it, 0.623852461953823. Skipped
/// without the configuration.
#[test]
fn bridge_test_quenched_leapfrog() {
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut u = ruttice::io::read_bridge_text(&path, &lat).unwrap();
    let mut hmc = quenched(0.01, 10);
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    eprintln!("{t:?}");
    assert!(
        (t.plaquette - 0.623852461953823).abs() < 1e-13,
        "plaquette {:.15} against 0.623852461953823",
        t.plaquette
    );
}

/// The tests of Bridge++'s `HMC_General` on conf_04040408 (4³ × 8): the rectangle (Iwasaki)
/// gauge action at β = 2.3, one level of leapfrog with 5 steps over a trajectory of 0.1,
/// exp(iP) to order 8, the M-series numbers of trajectory 200000; the plaquette after one
/// trajectory against `expected_result` of its parameter file.
fn bridge_general(expected: f64, fermions: Option<Option<ruttice::gauge::GaugeSmearing>>) {
    bridge_general_with(expected, fermions, None)
}

/// `bridge_general` with clover fermions of this c_SW in place of Wilson fermions.
fn bridge_general_with(
    expected: f64,
    fermions: Option<Option<ruttice::gauge::GaugeSmearing>>,
    csw: Option<f64>,
) {
    use ruttice::fermion::{Clover, CloverParams, Wilson, WilsonParams};
    use ruttice::hmc::{Hmc, RectangleAction, Scheme, Term, TwoFlavor};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut u = ruttice::io::read_bridge_text(&path, &lat).unwrap();
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut terms: Vec<Term> = Vec::new();
    if let Some(smearing) = fermions {
        let params = WilsonParams {
            kappa: 0.12,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-28,
            max_iterations: 1000,
            max_restarts: 40,
        };
        let action: Box<dyn ruttice::hmc::MdAction> = match csw {
            None => Box::new(TwoFlavor::new(
                Wilson::<f64>::new(&lat, params),
                smearing,
                stop,
                stop,
            )),
            Some(csw) => Box::new(
                TwoFlavor::new(
                    Clover::<f64>::new(
                        &lat,
                        CloverParams {
                            kappa: params.kappa,
                            csw,
                            boundary: params.boundary,
                        },
                    ),
                    smearing,
                    stop,
                    stop,
                )
                .with_clover(params.kappa * csw),
            ),
        };
        terms.push(Term { level: 0, action });
    }
    terms.push(Term {
        level: 0,
        action: Box::new(gauge),
    });
    let mut hmc = Hmc::new(terms, Scheme::Leapfrog, vec![5], 8, 0.1);
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    eprintln!("{t:?}");
    assert!(
        (t.plaquette - expected).abs() < 1e-12,
        "plaquette {:.15} against {expected}",
        t.plaquette
    );
}

/// `test_HMC_Quenched`.
#[test]
fn bridge_test_quenched_rectangle() {
    bridge_general(0.627477459735022, None);
}

/// `test_HMC_Wilson_Nf2`: two flavours of Wilson fermions (κ = 0.12, antiperiodic in time,
/// Dirac γ matrices) on links smeared once by stout smearing with ρ = 0.1.
#[test]
fn bridge_test_wilson_nf2_stout() {
    use ruttice::gauge::smear::Stout;
    use ruttice::gauge::{GaugeSmearing, Smear};
    bridge_general(
        0.62763964820533,
        Some(Some(GaugeSmearing {
            smear: Smear::Stout(Stout::uniform(0.1)),
            steps: 1,
        })),
    );
}

/// `test_HMC_Clover_Nf2`: as `test_HMC_Wilson_Nf2` with clover fermions, c_SW = 1.
#[test]
fn bridge_test_clover_nf2_stout() {
    use ruttice::gauge::smear::Stout;
    use ruttice::gauge::{GaugeSmearing, Smear};
    bridge_general_with(
        0.627753790313323,
        Some(Some(GaugeSmearing {
            smear: Smear::Stout(Stout::uniform(0.1)),
            steps: 1,
        })),
        Some(1.0),
    );
}

/// Bridge++'s domain-wall HMC tests on conf_04040408: Shamir's operator (b = 1, c = 0),
/// M0 = 1.6, m = 0.1, Ls = 8, on links smeared twice (stout, ρ = 0.1), with the rectangle
/// gauge action. `pv`: the ratio over Pauli-Villars fields (`Action_F_Ratio_lex`) on the
/// outer level of two of Omelyan with 5 and 4 steps, the gauge action inside; otherwise two
/// flavours on one level of leapfrog with 5 steps.
fn bridge_domain_wall(expected: f64, pv: bool) {
    use ruttice::fermion::{Mdwf, MdwfParams};
    use ruttice::gauge::smear::Stout;
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::hmc::{Hmc, RectangleAction, Scheme, Term, TwoFlavor, TwoFlavorRatio};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut u = ruttice::io::read_bridge_text(&path, &lat).unwrap();
    let params = MdwfParams {
        ls: 8,
        m5: 1.6,
        mass: 0.1,
        b: 1.0,
        c: 0.0,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 4000,
        max_restarts: 40,
    };
    let smearing = Some(GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(0.1)),
        steps: 2,
    });
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut hmc = if pv {
        let fermions = TwoFlavorRatio::new(
            Mdwf::<f64>::new(&lat, params),
            Mdwf::<f64>::new(&lat, params.pauli_villars()),
            smearing,
            stop,
            stop,
        );
        Hmc::new(
            vec![
                Term {
                    level: 0,
                    action: Box::new(fermions),
                },
                Term {
                    level: 1,
                    action: Box::new(gauge),
                },
            ],
            Scheme::Omelyan {
                lambda: 0.1931833275,
            },
            vec![5, 4],
            8,
            0.1,
        )
    } else {
        let fermions = TwoFlavor::new(Mdwf::<f64>::new(&lat, params), smearing, stop, stop);
        Hmc::new(
            vec![
                Term {
                    level: 0,
                    action: Box::new(fermions),
                },
                Term {
                    level: 0,
                    action: Box::new(gauge),
                },
            ],
            Scheme::Leapfrog,
            vec![5],
            8,
            0.1,
        )
    };
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    eprintln!("{t:?}");
    // Bridge++'s own criterion (`Test::verify`): a relative difference of 1e-11. Its solves
    // (BiCGStab(L) to |r|² < 1e-28) and ours (CG, the same) differ in the last digits, and
    // the domain-wall operator amplifies that more than the Wilson one.
    assert!(
        ((t.plaquette - expected) / expected).abs() < 1e-11,
        "plaquette {:.15} against {expected}",
        t.plaquette
    );
}

/// `test_HMC_Domainwall_Nf2`.
#[test]
fn bridge_test_domain_wall_nf2() {
    bridge_domain_wall(0.62862305031787, false);
}

/// `test_HMC_Domainwall_Nf2_PV`.
#[test]
fn bridge_test_domain_wall_nf2_pv() {
    bridge_domain_wall(0.62754640357188, true);
}

/// `test_HMC_Clover_RHMC_Nf2p1`: 2 + 1 flavours of clover fermions (c_SW = 1, Dirac γ
/// matrices, antiperiodic in time) on links smeared twice (stout, ρ = 0.1), with the rectangle
/// gauge action. The light pair is Hasenbusch's: det(D_0.13)²/det(D_0.11)² as the ratio
/// (`Action_F_Ratio_lex`) on level 0, det(D_0.11)² as two flavours on level 1; the strange
/// quark det(D†D_0.12)^(1/2) by RHMC on level 1: S = φ† r(D†D) φ, r ≈ x^(−1/2), φ = r'(D†D) ξ,
/// r' ≈ x^(1/4), 16 poles on [0.01², 3.16227766²] (Bridge++'s `lower_bound`, `upper_bound`
/// are those of |D|); the gauge action on level 2; leapfrog with [5, 2, 3] steps over 0.1.
///
/// The coefficients are read from Bridge++'s files when they are in the test directory, as
/// Bridge++ does, and otherwise computed (`RationalApprox::minimax` gives the same numbers to
/// 2e-16, `numerics::rational`'s tests). All solves to |r|²/|b|² < 1e-28: CG and multi-shift
/// CG here, BiCGStab(L) and multi-shift CG in Bridge++.
#[test]
fn bridge_test_clover_rhmc_nf2p1() {
    use ruttice::fermion::{Clover, CloverParams};
    use ruttice::gauge::smear::Stout;
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::hmc::{Hmc, Rational, RectangleAction, Scheme, Term, TwoFlavor, TwoFlavorRatio};
    use ruttice::numerics::{RationalApprox, RationalSpec};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let mut u = ruttice::io::read_bridge_text(&path, &lat).unwrap();
    let clover = |kappa: f64| {
        Clover::<f64>::new(
            &lat,
            CloverParams {
                kappa,
                csw: 1.0,
                boundary: [
                    Boundary::Periodic,
                    Boundary::Periodic,
                    Boundary::Periodic,
                    Boundary::Antiperiodic,
                ],
            },
        )
    };
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 1000,
        max_restarts: 40,
    };
    let smearing = Some(GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(0.1)),
        steps: 2,
    });
    let spec = RationalSpec {
        poles: 16,
        p: -1,
        q: 2,
        x_min: 1.0e-2 * 1.0e-2,
        x_max: 3.16227766 * 3.16227766,
    };
    let coefficients = |spec: RationalSpec| match common::bridge_test_file(&spec.bridge_file_name())
    {
        Some(file) => {
            eprintln!("coefficients from {}", file.display());
            RationalApprox::read_bridge(&file, &spec).unwrap()
        }
        None => RationalApprox::minimax(&spec).unwrap(),
    };
    let (md, hb) = (coefficients(spec), coefficients(spec.with_power(1, 4)));
    let ratio = TwoFlavorRatio::new(clover(0.13), clover(0.11), smearing, stop, stop)
        .with_clover(0.13, 0.11);
    let light = TwoFlavor::new(clover(0.11), smearing, stop, stop).with_clover(0.11);
    let strange = Rational::new(clover(0.12), smearing, md, hb, stop, stop).with_clover(0.12);
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut hmc = Hmc::new(
        vec![
            Term {
                level: 0,
                action: Box::new(ratio),
            },
            Term {
                level: 1,
                action: Box::new(light),
            },
            Term {
                level: 1,
                action: Box::new(strange),
            },
            Term {
                level: 2,
                action: Box::new(gauge),
            },
        ],
        Scheme::Leapfrog,
        vec![5, 2, 3],
        8,
        0.1,
    );
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    eprintln!("{t:?}");
    let expected = 0.627668186553629;
    // Bridge++'s criterion (`Test::verify`): a relative difference of 1e-11.
    assert!(
        ((t.plaquette - expected) / expected).abs() < 1e-11,
        "plaquette {:.15} against {expected}",
        t.plaquette
    );
}
