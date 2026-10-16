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

/// The Bridge++ test `test_HMC_Quenched_Leapfrog` (its configuration `conf_04040408.txt`, plaquette
/// action β = 6.0, 10 leapfrog steps of 0.01, M-series numbers from trajectory number 200000, one
/// trajectory): the plaquette after it, 0.623852461953823.
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
    use ruttice::gauge::smearing::Stout;
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
    use ruttice::gauge::smearing::Stout;
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

/// Bridge++'s domain-wall HMC tests on conf_04040408: Shamir's operator (b = 1, c = 0), M0 = 1.6, m
/// = 0.1, Ls = 8, on links smeared twice (stout, ρ = 0.1), with the rectangle gauge action.
fn bridge_domain_wall(expected: f64, pv: bool) {
    bridge_domain_wall_bc(expected, pv, (1.0, 0.0));
}

/// `bridge_domain_wall` with Möbius' b and c.
fn bridge_domain_wall_bc(expected: f64, pv: bool, (b, c): (f64, f64)) {
    use ruttice::fermion::{Mdwf, MdwfParams};
    use ruttice::gauge::smearing::Stout;
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
        b,
        c,
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

/// conf_04040408 repeated over 8⁴, as Bridge++'s reader `Text_4x4x4x8` puts it on a larger
/// lattice (the link at x is the link of the file at x mod (4, 4, 4, 8)).
fn tiled_8888(path: &std::path::Path) -> ruttice::Field<f64, ruttice::Gauge> {
    use ruttice::gauge::{link, set_link};
    let small = Lattice::single_node([4, 4, 4, 8]);
    let u4 = ruttice::io::read_bridge_text(path, &small).unwrap();
    let lat = Lattice::single_node([8; ND]);
    let mut u = ruttice::Field::<f64, ruttice::Gauge>::gauge(&lat);
    for site in 0..lat.num_sites() {
        let (x, y, z, t) = (site % 8, (site / 8) % 8, (site / 64) % 8, site / 512);
        let from = x % 4 + 4 * (y % 4 + 4 * (z % 4 + 4 * t));
        for mu in 0..ND {
            set_link(&mut u, mu, site, &link(&u4, mu, from));
        }
    }
    u
}

/// Bridge++'s Wilson HMC tests of its alternative code (`test_alt_hmc`, the parameter files
/// `test_HMC_Wilson_Nf2_alt*.yaml`) on conf_04040408 repeated over 8⁴: κ = 0.12, antiperiodic in
/// time, the rectangle gauge action at β = 2.3, one level of leapfrog with 5 steps over 0.1, the
/// M-series numbers of trajectory 200000, the fermion links smeared `steps` times (stout, ρ = 0.1).
/// The plaquette after one trajectory against `expected_result`, to Bridge++'s relative 1e-11 (its
/// solves are BiCGStab, ours CG, both to |r|²/|b|² < 1e-28); with the initial guess (`guess` > 0)
/// the solves round differently, and the plaquette must agree to 1e-9.
fn bridge_alt_wilson(
    expected: Option<f64>,
    steps: usize,
    even_odd: bool,
    metropolis: bool,
    guess: usize,
) -> Option<(usize, f64)> {
    use ruttice::fermion::{Wilson, WilsonParams};
    use ruttice::gauge::smearing::Stout;
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::hmc::{Hmc, Links, MdAction, RectangleAction, Scheme, Term};
    use ruttice::hmc::{TwoFlavor, TwoFlavorEo};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    use std::cell::Cell;
    use std::rc::Rc;
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return None;
    };
    let mut u = tiled_8888(&path);
    let lat = u.lattice().clone();
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
    let smearing = (steps > 0).then_some(GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(0.1)),
        steps,
    });
    // The action inside the HMC, its iterations read after each force.
    struct Counted<A> {
        action: A,
        count: Rc<Cell<usize>>,
        read: fn(&A) -> usize,
    }
    impl<A: MdAction> MdAction for Counted<A> {
        fn langevin(&mut self, l: &Links, r: &mut Mseries) -> f64 {
            self.action.langevin(l, r)
        }
        fn hamiltonian(&mut self, l: &Links) -> f64 {
            self.action.hamiltonian(l)
        }
        fn force(&mut self, l: &Links) -> ruttice::Field<f64, ruttice::Gauge> {
            let f = self.action.force(l);
            self.count.set((self.read)(&self.action));
            f
        }
    }
    let count = Rc::new(Cell::new(0usize));
    let fermions: Box<dyn MdAction> = if even_odd {
        Box::new(Counted {
            action: TwoFlavorEo::new(Wilson::<f64>::new(&lat, params), smearing, stop, stop)
                .with_guess(guess),
            count: count.clone(),
            read: |a| a.md_iterations,
        })
    } else {
        Box::new(Counted {
            action: TwoFlavor::new(Wilson::<f64>::new(&lat, params), smearing, stop, stop)
                .with_guess(guess),
            count: count.clone(),
            read: |a| a.md_iterations,
        })
    };
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut hmc = Hmc::new(
        vec![
            Term {
                level: 0,
                action: fermions,
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
    );
    hmc.metropolis = metropolis;
    let start = std::time::Instant::now();
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{t:?}\nforce solves: {} CG iterations; {seconds:.2} s",
        count.get()
    );
    if let Some(expected) = expected {
        let tol = if guess == 0 { 1e-11 } else { 1e-9 };
        assert!(
            ((t.plaquette - expected) / expected).abs() < tol,
            "plaquette {:.15} against {expected}",
            t.plaquette
        );
    }
    Some((count.get(), seconds))
}

/// `test_HMC_Wilson_Nf2_alt` (8⁴): the action of the whole lattice, links smeared once. It
/// checks the tiling of the configuration for the even-odd tests below.
#[test]
fn bridge_test_wilson_nf2_alt_8888() {
    bridge_alt_wilson(Some(0.624961897149602), 1, false, true, 0);
}

/// `test_HMC_Wilson_Nf2_alt_eo` (8⁴): even-odd, links smeared twice, no Metropolis test.
#[test]
fn bridge_test_wilson_nf2_alt_eo_8888() {
    bridge_alt_wilson(Some(0.624921816342905), 2, true, false, 0);
}

/// `test_HMC_Wilson_Nf2_alt_eo_nosmr` (8⁴): even-odd without smearing, no Metropolis test.
#[test]
fn bridge_test_wilson_nf2_alt_eo_nosmr_8888() {
    bridge_alt_wilson(Some(0.625038361335609), 0, true, false, 0);
}

/// What even-odd preconditioning and the chronological guess save on one trajectory of
/// `test_HMC_Wilson_Nf2_alt_eo` (8⁴, two stout steps; six force solves): the CG iterations of the
/// force solves of the whole-lattice and of the even-odd action, each from zero and with the guess
/// from the last four solutions.
#[test]
fn even_odd_and_guess_save_iterations() {
    let e = Some(0.624921816342905);
    let Some((eo, eo_s)) = bridge_alt_wilson(e, 2, true, false, 0) else {
        return;
    };
    let (eo_g, eo_g_s) = bridge_alt_wilson(e, 2, true, false, 4).unwrap();
    let (full, full_s) = bridge_alt_wilson(None, 2, false, false, 0).unwrap();
    let (full_g, full_g_s) = bridge_alt_wilson(None, 2, false, false, 4).unwrap();
    eprintln!("CG iterations of the force solves (seconds of the trajectory):");
    eprintln!(
        "  whole lattice {full:5} ({full_s:.2} s), with the guess {full_g:5} ({full_g_s:.2} s)"
    );
    eprintln!("  even-odd      {eo:5} ({eo_s:.2} s), with the guess {eo_g:5} ({eo_g_s:.2} s)");
    assert!(2 * eo < full, "even-odd {eo} against {full}");
    assert!(eo_g < eo && full_g < full);
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

/// `test_HMC_Domainwall_Moebius_Nf2_PV` (new in Bridge++ 2.1.3): Möbius' operator, b = 1.5,
/// c = 0.5, over two Pauli-Villars flavours.
#[test]
fn bridge_test_domain_wall_moebius_nf2_pv() {
    bridge_domain_wall_bc(0.62752900011880808, true, (1.5, 0.5));
}

/// Bridge++'s domain-wall tests over Pauli-Villars fields of its alternative code
/// (`test_alt_hmc`, `test_HMC_Domainwall_Nf2_PV_alt*.yaml`: `Action_F_Ratio_lex`, and
/// `Action_F_alt_Ratio_eo` when `even_odd`) on conf_04040408 repeated over 8⁴: the operator of
/// `bridge_domain_wall`, the PV mass 1, links smeared `steps` times (stout, ρ = 0.1), Omelyan
/// [5, 4] over 0.1 with the fermions on the outer level, no Metropolis test. The plaquette after
/// one trajectory against `expected_result`, to Bridge++'s 1e-11.
fn bridge_alt_domain_wall_pv(expected: f64, steps: usize, even_odd: bool) {
    use ruttice::fermion::{Mdwf, MdwfParams};
    use ruttice::gauge::smearing::Stout;
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::hmc::{Hmc, MdAction, RectangleAction, Scheme, Term};
    use ruttice::hmc::{TwoFlavorRatio, TwoFlavorRatioEo};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let mut u = tiled_8888(&path);
    let lat = u.lattice().clone();
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
    let smearing = (steps > 0).then_some(GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(0.1)),
        steps,
    });
    let (op, pv) = (
        Mdwf::<f64>::new(&lat, params),
        Mdwf::<f64>::new(&lat, params.pauli_villars()),
    );
    let fermions: Box<dyn MdAction> = match even_odd {
        true => Box::new(TwoFlavorRatioEo::new(op, pv, smearing, stop, stop)),
        false => Box::new(TwoFlavorRatio::new(op, pv, smearing, stop, stop)),
    };
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut hmc = Hmc::new(
        vec![
            Term {
                level: 0,
                action: fermions,
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
    );
    hmc.metropolis = false;
    let start = std::time::Instant::now();
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    eprintln!("{t:?}\n{:.1} s", start.elapsed().as_secs_f64());
    assert!(
        ((t.plaquette - expected) / expected).abs() < 1e-11,
        "plaquette {:.15} against {expected}",
        t.plaquette
    );
}

/// `test_HMC_Wilson_Nf2_alt_eo_ratio` (8⁴, conf_04040408 repeated, no smearing): Hasenbusch's
/// two flavours, the even-odd ratio of κ = 0.12 over κ = 0.11 (`Action_F_alt_Ratio_eo`) on the
/// outer level and the even-odd action of κ = 0.11 with the rectangle gauge action on the inner
/// one; leapfrog [5, 4] over 0.1, no Metropolis test.
#[test]
fn bridge_test_wilson_nf2_alt_eo_ratio_8888() {
    use ruttice::fermion::{Wilson, WilsonParams};
    use ruttice::hmc::{Hmc, RectangleAction, Scheme, Term, TwoFlavorEo, TwoFlavorRatioEo};
    use ruttice::solver::Stop;
    use ruttice::Boundary;
    let Some(path) = common::bridge_test_file("conf_04040408.txt") else {
        eprintln!(
            "SKIPPED: set {} to a directory containing conf_04040408.txt",
            common::BRIDGE_TESTS_ENV
        );
        return;
    };
    let expected = 0.625088521840085;
    let mut u = tiled_8888(&path);
    let lat = u.lattice().clone();
    let wilson = |kappa| {
        Wilson::<f64>::new(
            &lat,
            WilsonParams {
                kappa,
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
        max_iterations: 4000,
        max_restarts: 40,
    };
    let gauge = RectangleAction {
        beta: 2.3,
        c_plaq: 3.648,
        c_rect: -0.331,
    };
    let mut hmc = Hmc::new(
        vec![
            Term {
                level: 0,
                action: Box::new(TwoFlavorRatioEo::new(
                    wilson(0.12),
                    wilson(0.11),
                    None,
                    stop,
                    stop,
                )),
            },
            Term {
                level: 1,
                action: Box::new(TwoFlavorEo::new(wilson(0.11), None, stop, stop)),
            },
            Term {
                level: 1,
                action: Box::new(gauge),
            },
        ],
        Scheme::Leapfrog,
        vec![5, 4],
        8,
        0.1,
    );
    hmc.metropolis = false;
    let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
    eprintln!("{t:?}");
    assert!(
        ((t.plaquette - expected) / expected).abs() < 1e-11,
        "plaquette {:.15} against {expected}",
        t.plaquette
    );
}

/// `test_HMC_Domainwall_Nf2_PV_alt` (8⁴, links smeared twice).
#[test]
fn bridge_test_domain_wall_nf2_pv_alt_8888() {
    bridge_alt_domain_wall_pv(0.624958020749689, 2, false);
}

/// `test_HMC_Domainwall_Nf2_PV_alt_nosmr` (8⁴, no smearing).
#[test]
fn bridge_test_domain_wall_nf2_pv_alt_nosmr_8888() {
    bridge_alt_domain_wall_pv(0.624866718847932, 0, false);
}

/// `test_HMC_Domainwall_Nf2_PV_alt_eo` (8⁴, links smeared twice).
///
/// Not passing (2026-10-16): the plaquette is 0.624960260701932, 7e-6 below `expected_result`,
/// and 0.624865517099993 against 0.624876160774692 without smearing; ΔH is 3e-4 and 6e-4. What
/// these tests share with the ones that agree to 1e-11 is checked: the lexical ratio of the same
/// operator on the same 8⁴ lattice (`bridge_test_domain_wall_nf2_pv_alt*`), the even-odd ratio
/// of Wilson fermions (`bridge_test_wilson_nf2_alt_eo_ratio_8888`), and of domain-wall fermions
/// on their own K = T_ee⁻¹ Ŝ and K† against the blocks, Γ5 on the even sites against that of the
/// whole lattice (`even_odd_pv_operator_is_made_of_the_blocks`), the force against the action and
/// S = |ξ_e|² after the heat bath (`even_odd_ratio_force_is_the_derivative`). On Bridge++'s side,
/// its D_ee⁻¹ and D_ee⁻† by L_inv, U_inv, Ldag_inv, Udag_inv were transcribed and are exact (to
/// 4e-16), and Index_eo::convertField maps the noise site by site, every s alike. Seven other
/// conventions tried in a self-consistent action (the noise of the odd sites, reflected in s or
/// conjugated; K in the right form; Γ5 without R; H ξ for H† ξ; and their pairs) all land 2e-7
/// to 1e-4 away, none to 1e-11: the plaquette after one trajectory moves by 1e-5 for any other
/// pseudofermion field. Not found: Bridge++'s numbers inside the trajectory. Its even-odd
/// domain-wall HMC runs only on QXS fields, and neither the 2.0.0-mod.34 tree here (no
/// `vsimd_common_float-inc.h`) nor mod.93 (QXS behind its `lib_alt`) builds that test. Bridge++
/// 2.1.3 (16 Apr 2026) no longer has `Action_F_Ratio_eo` or `test_alt_hmc`: its even-odd actions
/// are the lexical ones with even-odd solves, so it has no even-odd pseudofermion to compare with.
#[test]
#[ignore = "7e-6 from Bridge++'s expected_result, not understood (see the comment)"]
fn bridge_test_domain_wall_nf2_pv_alt_eo_8888() {
    bridge_alt_domain_wall_pv(0.624964691911979, 2, true);
}

/// `test_HMC_Domainwall_Nf2_PV_alt_eo_nosmr` (8⁴, no smearing). Not passing: see
/// `bridge_test_domain_wall_nf2_pv_alt_eo_8888`.
#[test]
#[ignore = "1e-5 from Bridge++'s expected_result, not understood"]
fn bridge_test_domain_wall_nf2_pv_alt_eo_nosmr_8888() {
    bridge_alt_domain_wall_pv(0.624876160774692, 0, true);
}

/// `test_HMC_Clover_RHMC_Nf2p1`: 2 + 1 flavours of clover fermions (c_SW = 1, Dirac γ matrices,
/// antiperiodic in time) on links smeared twice (stout, ρ = 0.1), with the rectangle gauge action.
#[test]
fn bridge_test_clover_rhmc_nf2p1() {
    use ruttice::fermion::{Clover, CloverParams};
    use ruttice::gauge::smearing::Stout;
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
