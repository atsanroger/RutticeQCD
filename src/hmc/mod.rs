//! Hybrid Monte Carlo: gauge configurations from the momenta, the actions and their forces,
//! a molecular-dynamics integrator and the Metropolis test.

pub(crate) mod dwf_multigrid;
mod even_odd;
mod fermion_action;
mod gauge_action;
mod guess;
mod integrator;
mod link_field;
mod links;
mod mixed;
mod mseries;
mod rational;
mod solve;

pub use dwf_multigrid::{normal_solve as dwf_multigrid_normal_solve, DwfMultigridSetup, Seconds};
pub use rational::Rational;
pub use solve::{MultigridSetup, NormalSolver, Pre, Preconditioner};

pub use even_odd::{
    log_det_diag, EvenOddForceOp, EvenOddHalves, HalfPsi, TwoFlavorEo, TwoFlavorRatioEo,
};
pub use fermion_action::{
    clover_derivative, force_from, hop_derivative, kernel_derivative, smeared_force,
    CloverInsertion, ForceOp, Single, TwoFlavor, TwoFlavorRatio,
};
pub use gauge_action::{
    force_statistics, path_product, staples, traceless_antihermitian, PlaquetteAction,
    RectangleAction, Step,
};
pub use guess::Chrono;
pub use integrator::{Hmc, HmcTrajectory, MdAction, Reversibility, Scheme, SolveCost, Term};
pub use link_field::LinkField;
pub use links::{Links, SmearingCache};
pub use mixed::{mixed_normal, SingleMdwf, SingleMdwfEo, SingleNormal};
pub use mseries::Mseries;

use crate::gauge::set_link;
use crate::kernel::{ColorMatrix, Cplx};
use crate::{for_sites, Field, Gauge, Lattice, ND};
use std::sync::Arc;

/// Conjugate momenta for every link: iP = i Σ_a 2 h_a T^a, with T^a = λ^a/2 the generators of
/// su(3) (Gell-Mann λ^a) and h_a Gaussian of variance 1/2 from `rng`, eight per link, drawn
/// for direction μ, then site after site of the global lattice (Bridge++'s `set_iP`).
pub fn draw_momenta(lat: &Arc<Lattice>, rng: &mut Mseries) -> Field<f64, Gauge> {
    let mut p = Field::<f64, Gauge>::gauge(lat);
    let r = 1.0 / 3.0_f64.sqrt();
    rng.gauss_lex_global(lat, 8, ND, |mu, x, h| {
        // P = Σ h_a λ^a; iP is stored.
        let p_mat = [
            [
                Cplx::new(h[2] + h[7] * r, 0.0),
                Cplx::new(h[0], -h[1]),
                Cplx::new(h[3], -h[4]),
            ],
            [
                Cplx::new(h[0], h[1]),
                Cplx::new(-h[2] + h[7] * r, 0.0),
                Cplx::new(h[5], -h[6]),
            ],
            [
                Cplx::new(h[3], h[4]),
                Cplx::new(h[5], h[6]),
                Cplx::new(-2.0 * h[7] * r, 0.0),
            ],
        ];
        let ip = ColorMatrix(p_mat.map(|row| row.map(|z| Cplx::new(-z.im, z.re))));
        set_link(&mut p, mu, x, &ip);
    });
    p
}

/// ½ |iP|², summed over all processes.
pub fn kinetic_energy(p: &Field<f64, Gauge>) -> f64 {
    let mut s = [p.as_slice().iter().map(|v| v * v).sum::<f64>()];
    p.lattice().comm.sum(&mut s);
    0.5 * s[0]
}

/// U made unitary again as Bridge++ does it for SU(3) (`reunit_SU3`): the first row
/// normalised, the second made orthogonal to it and normalised, the third the complex
/// conjugate of their cross product.
pub fn reunitarize(m: &ColorMatrix<f64>) -> ColorMatrix<f64> {
    let [r0, r1, _] = m.0;
    let norm = |r: &[Cplx<f64>; 3]| 1.0 / r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let n0 = norm(&r0);
    let r0 = r0.map(|z| z.scale(n0));
    let dot = r0[0].conj() * r1[0] + r0[1].conj() * r1[1] + r0[2].conj() * r1[2];
    let r1: [Cplx<f64>; 3] = core::array::from_fn(|k| r1[k] - dot * r0[k]);
    let n1 = norm(&r1);
    let r1 = r1.map(|z| z.scale(n1));
    let r2 = [
        (r0[1] * r1[2] - r0[2] * r1[1]).conj(),
        (r0[2] * r1[0] - r0[0] * r1[2]).conj(),
        (r0[0] * r1[1] - r0[1] * r1[0]).conj(),
    ];
    ColorMatrix([r0, r1, r2])
}

/// U ← exp(ε iP) U: u ← U + ε/(n − k) iP u for k = 0 … n − 1 (the Taylor series to order n,
/// nested), then made unitary again (Bridge++'s `mult_exp_Field_G`).
pub fn exp_update(eps: f64, p: &Field<f64, Gauge>, u: &mut Field<f64, Gauge>, n: usize) {
    for mu in 0..ND {
        let pm = p.slab(mu);
        for_sites(u.slab_mut(mu), 18, |x, out| {
            let u0 = ColorMatrix::from_reals(out);
            let h = ColorMatrix::from_reals(&pm[18 * x..18 * (x + 1)]);
            let mut u1 = u0;
            for k in 0..n {
                u1 = h.mul(&u1).scale(eps / (n - k) as f64).add(&u0);
            }
            reunitarize(&u1).write_reals(out);
        });
    }
}

/// The unit gauge field (a cold start).
pub fn cold_start(lat: &Arc<Lattice>) -> Field<f64, Gauge> {
    let mut u = Field::<f64, Gauge>::gauge(lat);
    let one = ColorMatrix::identity();
    for mu in 0..ND {
        for x in 0..lat.num_sites() {
            set_link(&mut u, mu, x, &one);
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauge::{link, random_gauge};
    use crate::Boundary;

    /// The force is minus the derivative of the action: along a direction X of the Lie
    /// algebra at one link, d/dt S(e^(tX) U) at t = 0 is −F·X, the dot product of their reals
    /// (by a central difference).
    #[test]
    fn force_is_the_derivative_of_the_action() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 3);
        let action = PlaquetteAction { beta: 5.5 };
        let f = action.force(&u);
        let mut rng = Mseries::new(7);
        let x_field = draw_momenta(&lat, &mut rng);
        let (mu, site) = (2, 37);
        let x = link(&x_field, mu, site);
        let s_at = |t: f64| {
            let mut v = Field::<f64, Gauge>::gauge(&lat);
            v.as_mut_slice().copy_from_slice(u.as_slice());
            // e^(tX) by its series, far beyond what t needs.
            let mut e = ColorMatrix::identity();
            let mut term = ColorMatrix::identity();
            for k in 1..20 {
                term = x.mul(&term).scale(t / k as f64);
                e = e.add(&term);
            }
            set_link(&mut v, mu, site, &e.mul(&link(&u, mu, site)));
            action.action(&v)
        };
        let h = 1e-5;
        let derivative = (s_at(h) - s_at(-h)) / (2.0 * h);
        // dS/dt = −Σ reals F · X: the momenta move as dP/dt = F, dU/dt = P U, and H is
        // conserved, so d(½|P|²)/dt = ⟨P, F⟩ = −dS/dt with P = X.
        let fx: f64 = link(&f, mu, site)
            .0
            .iter()
            .flatten()
            .zip(x.0.iter().flatten())
            .map(|(a, b)| a.re * b.re + a.im * b.im)
            .sum();
        assert!(
            (derivative + fx).abs() < 1e-6 * fx.abs().max(1.0),
            "dS/dt {derivative} against −F·X {}",
            -fx
        );
    }

    /// The rectangle action and force: the plaquette action for c_rect = 0, and its force the
    /// derivative of the action for Iwasaki's coefficients.
    #[test]
    fn rectangle_action_and_force() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 11);
        let plain = RectangleAction {
            beta: 5.5,
            c_plaq: 1.0,
            c_rect: 0.0,
        };
        let plaq = PlaquetteAction { beta: 5.5 };
        assert!((plain.action(&u) - plaq.action(&u)).abs() < 1e-9 * plaq.action(&u));
        let (a, b) = (plain.force(&u), plaq.force(&u));
        let d = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(d < 1e-12, "{d}");
        let iwasaki = RectangleAction {
            beta: 2.3,
            c_plaq: 3.648,
            c_rect: -0.331,
        };
        let f = iwasaki.force(&u);
        let x_field = draw_momenta(&lat, &mut Mseries::new(9));
        for (mu, site) in [(0, 5), (3, 200)] {
            let x = link(&x_field, mu, site);
            let s_at = |t: f64| {
                let mut v = Field::<f64, Gauge>::gauge(&lat);
                v.as_mut_slice().copy_from_slice(u.as_slice());
                let (mut e, mut term) = (ColorMatrix::identity(), ColorMatrix::identity());
                for k in 1..20 {
                    term = x.mul(&term).scale(t / k as f64);
                    e = e.add(&term);
                }
                set_link(&mut v, mu, site, &e.mul(&link(&u, mu, site)));
                iwasaki.action(&v)
            };
            let h = 1e-5;
            let derivative = (s_at(h) - s_at(-h)) / (2.0 * h);
            let fx: f64 = link(&f, mu, site)
                .0
                .iter()
                .flatten()
                .zip(x.0.iter().flatten())
                .map(|(a, b)| a.re * b.re + a.im * b.im)
                .sum();
            assert!(
                (derivative + fx).abs() < 1e-6 * fx.abs().max(1.0),
                "{derivative} {fx}"
            );
        }
    }

    /// d/dt S(e^(tX) U) at t = 0 by a central difference, and −F·X: they must agree.
    fn check_force(
        u: &Field<f64, Gauge>,
        f: &Field<f64, Gauge>,
        links: &[(usize, usize)],
        mut s: impl FnMut(&Field<f64, Gauge>) -> f64,
        tol: f64,
    ) {
        let lat = u.lattice().clone();
        let x_field = draw_momenta(&lat, &mut Mseries::new(21));
        for &(mu, site) in links {
            let x = link(&x_field, mu, site);
            let mut s_at = |t: f64| {
                let mut v = Field::<f64, Gauge>::gauge(&lat);
                v.as_mut_slice().copy_from_slice(u.as_slice());
                let (mut e, mut term) = (ColorMatrix::identity(), ColorMatrix::identity());
                for k in 1..20 {
                    term = x.mul(&term).scale(t / k as f64);
                    e = e.add(&term);
                }
                set_link(&mut v, mu, site, &e.mul(&link(u, mu, site)));
                s(&v)
            };
            // Fourth order in h; a larger h keeps the rounding of S (a solve for the fermions)
            // small against the difference.
            let h = 1e-3;
            let derivative =
                (8.0 * (s_at(h) - s_at(-h)) - (s_at(2.0 * h) - s_at(-2.0 * h))) / (12.0 * h);
            let fx: f64 = link(f, mu, site)
                .0
                .iter()
                .flatten()
                .zip(x.0.iter().flatten())
                .map(|(a, b)| a.re * b.re + a.im * b.im)
                .sum();
            assert!(
                (derivative + fx).abs() < tol * fx.abs().max(1.0),
                "link ({mu}, {site}): dS/dt {derivative} against −F·X {}",
                -fx
            );
        }
    }

    /// The chain rule of stout smearing: the plaquette action of the links smeared twice,
    /// differentiated with respect to the links before smearing.
    #[test]
    fn stout_chain_rule_is_the_derivative() {
        use crate::gauge::smearing::{spatial, Stout};
        use crate::gauge::{GaugeSmearing, Smear};
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 13);
        let action = PlaquetteAction { beta: 5.0 };
        for rho in [crate::gauge::smearing::uniform(0.1), spatial(0.15)] {
            let smearing = GaugeSmearing {
                smear: Smear::Stout(Stout { rho }),
                steps: 2,
            };
            let levels = smearing.levels(&u);
            let v = levels.last().unwrap();
            // dS = Re tr X M for δV = X V: M = −(β/3) V Σ(V)†.
            let sigma = staples(v);
            let mut m = Field::<f64, Gauge>::gauge(&lat);
            for mu in 0..ND {
                for_sites(m.slab_mut(mu), 18, |x, out| {
                    link(v, mu, x)
                        .mul(&link(&sigma, mu, x).adj())
                        .scale(-action.beta / 3.0)
                        .write_reals(out);
                });
            }
            let f = smeared_force(&m, &u, Some(&smearing), &levels);
            check_force(
                &u,
                &f,
                &[(0, 3), (2, 100), (3, 255)],
                |w| action.action(&smearing.apply(w)),
                1e-7,
            );
        }
    }

    /// The force of two flavours of Wilson fermions, on the links and on stout links.
    #[test]
    fn wilson_force_is_the_derivative() {
        use crate::fermion::{Wilson, WilsonParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 17);
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
            tolerance_sqr: 1e-30,
            max_iterations: 1000,
            max_restarts: 10,
        };
        for smearing in [
            None,
            Some(GaugeSmearing {
                smear: Smear::Stout(Stout::uniform(0.1)),
                steps: 1,
            }),
        ] {
            let mut action = TwoFlavor::new(Wilson::<f64>::new(&lat, params), smearing, stop, stop);
            let cache = SmearingCache::new();
            action.langevin(&Links::new(&u, &cache), &mut Mseries::new(3));
            let f = action.force(&Links::new(&u, &cache));
            assert_eq!(
                cache.len(),
                usize::from(smearing.is_some()),
                "smeared once, kept"
            );
            check_force(
                &u,
                &f,
                &[(0, 7), (3, 64), (3, 200)],
                |w| action.hamiltonian(&Links::new(w, &SmearingCache::new())),
                1e-7,
            );
        }
    }

    /// The force of two flavours of clover fermions, on the links and on stout links.
    #[test]
    fn clover_force_is_the_derivative() {
        use crate::fermion::{Clover, CloverParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 19);
        let params = CloverParams {
            kappa: 0.12,
            csw: 1.7,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-30,
            max_iterations: 1000,
            max_restarts: 10,
        };
        for smearing in [
            None,
            Some(GaugeSmearing {
                smear: Smear::Stout(Stout::uniform(0.1)),
                steps: 1,
            }),
        ] {
            let mut action = TwoFlavor::new(Clover::<f64>::new(&lat, params), smearing, stop, stop)
                .with_clover(params.kappa * params.csw);
            let cache = SmearingCache::new();
            action.langevin(&Links::new(&u, &cache), &mut Mseries::new(5));
            let f = action.force(&Links::new(&u, &cache));
            check_force(
                &u,
                &f,
                &[(0, 9), (1, 70), (3, 201)],
                |w| action.hamiltonian(&Links::new(w, &SmearingCache::new())),
                1e-7,
            );
        }
    }

    /// The even-odd actions of Wilson and clover fermions (the clover one with its
    /// −2 ln |det T_oo|): the force is the derivative of the action, on the links and on stout
    /// links.
    #[test]
    fn even_odd_forces_are_the_derivatives() {
        use crate::fermion::{Clover, CloverParams, Wilson, WilsonParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 29);
        let boundary = [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ];
        let stop = Stop {
            tolerance_sqr: 1e-30,
            max_iterations: 1000,
            max_restarts: 10,
        };
        for smearing in [
            None,
            Some(GaugeSmearing {
                smear: Smear::Stout(Stout::uniform(0.1)),
                steps: 1,
            }),
        ] {
            let mut wilson = TwoFlavorEo::new(
                Wilson::<f64>::new(
                    &lat,
                    WilsonParams {
                        kappa: 0.12,
                        boundary,
                    },
                ),
                smearing,
                stop,
                stop,
            );
            let cache = SmearingCache::new();
            wilson.langevin(&Links::new(&u, &cache), &mut Mseries::new(31));
            let f = wilson.force(&Links::new(&u, &cache));
            check_force(
                &u,
                &f,
                &[(0, 6), (2, 81), (3, 200)],
                |w| wilson.hamiltonian(&Links::new(w, &SmearingCache::new())),
                1e-7,
            );
            let params = CloverParams {
                kappa: 0.12,
                csw: 1.7,
                boundary,
            };
            let mut clover =
                TwoFlavorEo::new(Clover::<f64>::new(&lat, params), smearing, stop, stop)
                    .with_clover(params.kappa * params.csw);
            clover.langevin(&Links::new(&u, &cache), &mut Mseries::new(37));
            let f = clover.force(&Links::new(&u, &cache));
            check_force(
                &u,
                &f,
                &[(0, 9), (1, 70), (3, 201)],
                |w| clover.hamiltonian(&Links::new(w, &SmearingCache::new())),
                1e-7,
            );
        }
    }

    /// The det term alone: −2 ln |det T_oo| of the clover term against a numerical
    /// derivative, through the force of an action whose pseudofermions are zero.
    #[test]
    fn clover_log_det_force_is_the_derivative() {
        use crate::fermion::{Clover, CloverParams};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 41);
        let params = CloverParams {
            kappa: 0.13,
            csw: 2.0,
            boundary: [Boundary::Periodic; ND],
        };
        let stop = Stop::simple(1e-30, 1000);
        let mut a = TwoFlavorEo::new(Clover::<f64>::new(&lat, params), None, stop, stop)
            .with_clover(params.kappa * params.csw);
        let cache = SmearingCache::new();
        a.langevin(&Links::new(&u, &cache), &mut Mseries::new(1));
        // The pseudofermion part is exercised above; here only the det term.
        a.zero_pseudofermions();
        let f = a.force(&Links::new(&u, &cache));
        let mut op = Clover::<f64>::new(&lat, params);
        check_force(
            &u,
            &f,
            &[(0, 3), (1, 100), (2, 17), (3, 254)],
            |w| {
                use crate::fermion::FermionOp;
                op.set_gauge(w).unwrap();
                -2.0 * log_det_diag(&op, &lat, crate::Parity::Odd)
            },
            1e-8,
        );
    }

    /// The initial guess changes the start of the force solves, not their solutions: the force
    /// with a guess from earlier solves is the force without one, in fewer iterations.
    /// Every solver gives the action and the force of CG, on all sites and even-odd; the RHMC
    /// pole by pole gives those of the multi-shift solver.
    #[test]
    fn the_solvers_agree() {
        use crate::fermion::{Clover, CloverParams};
        use crate::numerics::{RationalApprox, RationalSpec};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 41);
        let boundary = [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ];
        let stop = Stop {
            tolerance_sqr: 1e-26,
            max_iterations: 2000,
            max_restarts: 10,
        };
        let op = || {
            Clover::<f64>::new(
                &lat,
                CloverParams {
                    kappa: 0.12,
                    csw: 1.0,
                    boundary,
                },
            )
        };
        let close = |a: &Field<f64, Gauge>, b: &Field<f64, Gauge>| {
            let worst = a
                .as_slice()
                .iter()
                .zip(b.as_slice())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(worst < 1e-9, "forces differ by {worst:e}");
        };
        let solvers = [
            NormalSolver::BiCgStab,
            NormalSolver::Fgmres(32),
            NormalSolver::Gcr(16),
        ];
        let cache = SmearingCache::new();
        let links = Links::new(&u, &cache);

        let run = |solver| {
            let mut a = TwoFlavor::new(op(), None, stop, stop)
                .with_clover(0.12)
                .with_solver(solver);
            a.langevin(&links, &mut Mseries::new(43));
            (a.hamiltonian(&links), a.force(&links))
        };
        let (h0, f0) = run(NormalSolver::Cg);
        for solver in solvers {
            let (h, f) = run(solver);
            assert!((h - h0).abs() < 1e-9 * h0.abs(), "{solver:?}: {h} {h0}");
            close(&f, &f0);
        }

        let run = |solver| {
            let mut a = TwoFlavorEo::new(op(), None, stop, stop)
                .with_clover(0.12)
                .with_solver(solver);
            a.langevin(&links, &mut Mseries::new(43));
            (a.hamiltonian(&links), a.force(&links))
        };
        let (h0, f0) = run(NormalSolver::Cg);
        for solver in solvers {
            let (h, f) = run(solver);
            assert!((h - h0).abs() < 1e-9 * h0.abs(), "{solver:?}: {h} {h0}");
            close(&f, &f0);
        }

        let run = |solver| {
            let mut a = TwoFlavorRatio::new(op(), op_heavy(&lat, boundary), None, stop, stop)
                .with_clover(0.12, 0.10)
                .with_solver(solver);
            a.langevin(&links, &mut Mseries::new(43));
            (a.hamiltonian(&links), a.force(&links))
        };
        let (h0, f0) = run(NormalSolver::Cg);
        for solver in solvers {
            let (h, f) = run(solver);
            assert!((h - h0).abs() < 1e-9 * h0.abs(), "{solver:?}: {h} {h0}");
            close(&f, &f0);
        }

        let spec = RationalSpec {
            poles: 10,
            p: -1,
            q: 2,
            x_min: 1e-3,
            x_max: 10.0,
        };
        let (md, hb) = (
            RationalApprox::minimax(&spec).unwrap(),
            RationalApprox::minimax(&spec.with_power(1, 4)).unwrap(),
        );
        let run = |by_pole| {
            let mut a = Rational::new(op(), None, md.clone(), hb.clone(), stop, stop)
                .with_clover(0.12)
                .with_pole_by_pole(by_pole);
            a.langevin(&links, &mut Mseries::new(43));
            (a.hamiltonian(&links), a.force(&links))
        };
        let (h0, f0) = run(false);
        let (h, f) = run(true);
        assert!((h - h0).abs() < 1e-9 * h0.abs(), "pole by pole: {h} {h0}");
        close(&f, &f0);
        // The poles through γ5 D, by BiCGStab.
        let mut a = Rational::new(op(), None, md.clone(), hb.clone(), stop, stop)
            .with_clover(0.12)
            .with_two_step(NormalSolver::BiCgStab);
        a.langevin(&links, &mut Mseries::new(43));
        let (h, f) = (a.hamiltonian(&links), a.force(&links));
        assert!((h - h0).abs() < 1e-9 * h0.abs(), "two-step: {h} {h0}");
        close(&f, &f0);
    }

    /// The preconditioned solves give the action and the force of plain CG: Chebyshev with CG
    /// (all sites and even-odd), the multigrid with BiCGStab and FGMRES.
    #[test]
    fn preconditioners_change_the_solve_not_the_action() {
        use crate::fermion::{Clover, CloverParams};
        use crate::solver::preconditioner::multigrid::MultigridParams;
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 53);
        let boundary = [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ];
        let stop = Stop {
            tolerance_sqr: 1e-26,
            max_iterations: 2000,
            max_restarts: 10,
        };
        let op = {
            let lat = lat.clone();
            move || {
                Clover::<f64>::new(
                    &lat,
                    CloverParams {
                        kappa: 0.12,
                        csw: 1.0,
                        boundary,
                    },
                )
            }
        };
        let cache = SmearingCache::new();
        let links = Links::new(&u, &cache);
        let same = |name: &str, a: &(f64, Field<f64, Gauge>), b: &(f64, Field<f64, Gauge>)| {
            assert!(
                (a.0 - b.0).abs() < 1e-9 * a.0.abs(),
                "{name}: {} {}",
                a.0,
                b.0
            );
            let worst =
                a.1.as_slice()
                    .iter()
                    .zip(b.1.as_slice())
                    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(worst < 1e-9, "{name}: forces differ by {worst:e}");
        };

        let run = |solver, pre: Preconditioner| {
            let mut a = TwoFlavor::new(op(), None, stop, stop)
                .with_clover(0.12)
                .with_solver(solver)
                .with_preconditioner(pre);
            a.langevin(&links, &mut Mseries::new(59));
            let h = a.hamiltonian(&links);
            (h, a.force(&links), a.last_iterations)
        };
        let plain = run(NormalSolver::Cg, Preconditioner::None);
        let reference = (plain.0, plain.1);
        let cheb = run(NormalSolver::Cg, Preconditioner::chebyshev(6, 30.0));
        assert!(
            cheb.2 < plain.2,
            "Chebyshev: {} against {}",
            cheb.2,
            plain.2
        );
        same("Chebyshev", &reference, &(cheb.0, cheb.1));
        let mut params = MultigridParams::bridge([2, 2, 2, 2]);
        params.num_test_vectors = 8;
        params.setup_iterations = 2;
        for solver in [NormalSolver::BiCgStab, NormalSolver::Fgmres(16)] {
            let plain = run(solver, Preconditioner::None);
            let mg = run(
                solver,
                Preconditioner::multigrid(MultigridSetup::new(params, 0)),
            );
            assert!(
                mg.2 < plain.2,
                "{solver:?} with the multigrid: {} against {}",
                mg.2,
                plain.2
            );
            same("multigrid", &reference, &(mg.0, mg.1));
        }

        let run = |solver, pre: Preconditioner| {
            let mut a = TwoFlavorEo::new(op(), None, stop, stop)
                .with_clover(0.12)
                .with_solver(solver)
                .with_preconditioner(pre);
            a.langevin(&links, &mut Mseries::new(59));
            let h = a.hamiltonian(&links);
            (h, a.force(&links), a.last_iterations)
        };
        let plain = run(NormalSolver::Cg, Preconditioner::None);
        let reference = (plain.0, plain.1);
        let cheb = run(NormalSolver::Cg, Preconditioner::chebyshev(6, 30.0));
        same("Chebyshev, even-odd", &reference, &(cheb.0, cheb.1));
        let plain = run(NormalSolver::BiCgStab, Preconditioner::None);
        let mg = run(
            NormalSolver::BiCgStab,
            Preconditioner::multigrid(MultigridSetup::new(params, 0)),
        );
        assert!(
            mg.2 < plain.2,
            "even-odd with the multigrid: {} against {}",
            mg.2,
            plain.2
        );
        same("multigrid, even-odd", &reference, &(mg.0, mg.1));
    }

    /// After the links have moved a little, a setup from the test vectors of the setup before
    /// preconditions about as well as one from scratch, and the solution is the same.
    #[test]
    fn multigrid_setup_reuses_the_vectors_of_the_one_before() {
        use crate::fermion::{Clover, CloverParams};
        use crate::solver::preconditioner::multigrid::MultigridParams;
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 61);
        // The links a few molecular-dynamics steps later.
        let mut moved = Field::<f64, Gauge>::gauge(&lat);
        moved.as_mut_slice().copy_from_slice(u.as_slice());
        let p = draw_momenta(&lat, &mut Mseries::new(63));
        exp_update(0.02, &p, &mut moved, 8);
        let boundary = [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ];
        let stop = Stop {
            tolerance_sqr: 1e-26,
            max_iterations: 2000,
            max_restarts: 10,
        };
        let op = {
            let lat = lat.clone();
            move || {
                Clover::<f64>::new(
                    &lat,
                    CloverParams {
                        kappa: 0.12,
                        csw: 1.0,
                        boundary,
                    },
                )
            }
        };
        let mut params = MultigridParams::bridge([2, 2, 2, 2]);
        params.num_test_vectors = 8;
        params.setup_iterations = 2;
        // A setup after every force solve; the action and the iterations of the second force.
        let run = |reuse: bool| {
            let mut cycle = MultigridSetup::new(params, 1);
            if reuse {
                cycle = cycle.reusing_vectors(0);
            }
            let mut a = TwoFlavor::new(op(), None, stop, stop)
                .with_clover(0.12)
                .with_solver(NormalSolver::BiCgStab)
                .with_preconditioner(Preconditioner::multigrid(cycle));
            let cache = SmearingCache::new();
            a.langevin(&Links::new(&u, &cache), &mut Mseries::new(67));
            a.force(&Links::new(&u, &cache));
            let cache = SmearingCache::new();
            a.force(&Links::new(&moved, &cache));
            let f = a.force(&Links::new(&moved, &cache));
            (f, a.last_iterations)
        };
        let (cold, cold_iterations) = run(false);
        let (warm, warm_iterations) = run(true);
        let worst = cold
            .as_slice()
            .iter()
            .zip(warm.as_slice())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst < 1e-9, "forces differ by {worst:e}");
        eprintln!(
            "iterations: {cold_iterations} from scratch, {warm_iterations} from the vectors before"
        );
        assert!(2 * warm_iterations <= 3 * cold_iterations);
    }

    #[cfg(test)]
    fn op_heavy(lat: &Arc<Lattice>, boundary: [Boundary; ND]) -> crate::fermion::Clover<f64> {
        crate::fermion::Clover::<f64>::new(
            lat,
            crate::fermion::CloverParams {
                kappa: 0.10,
                csw: 1.0,
                boundary,
            },
        )
    }

    #[test]
    fn chronological_guess_saves_iterations() {
        use crate::fermion::{Wilson, WilsonParams};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u0 = random_gauge(&lat, 43);
        let params = WilsonParams {
            kappa: 0.125,
            boundary: [Boundary::Periodic; ND],
        };
        let stop = Stop::simple(1e-24, 2000);
        let p = draw_momenta(&lat, &mut Mseries::new(2));
        let mut plain = TwoFlavorEo::new(Wilson::<f64>::new(&lat, params), None, stop, stop);
        let mut chrono =
            TwoFlavorEo::new(Wilson::<f64>::new(&lat, params), None, stop, stop).with_guess(3);
        let cache = SmearingCache::new();
        plain.langevin(&Links::new(&u0, &cache), &mut Mseries::new(4));
        chrono.langevin(&Links::new(&u0, &cache), &mut Mseries::new(4));
        let mut u = Field::<f64, Gauge>::gauge(&lat);
        u.as_mut_slice().copy_from_slice(u0.as_slice());
        for step in 0..6 {
            let a = plain.force(&Links::new(&u, &cache));
            let b = chrono.force(&Links::new(&u, &cache));
            let d = a
                .as_slice()
                .iter()
                .zip(b.as_slice())
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(d < 1e-9, "step {step}: forces differ by {d}");
            exp_update(0.02, &p, &mut u, 8);
            cache.clear();
        }
        eprintln!(
            "CG iterations of six force solves: {} from zero, {} with the guess",
            plain.md_iterations, chrono.md_iterations
        );
        // 168 against 130 when written (the first solve has nothing to start from).
        assert!(chrono.md_iterations * 10 < plain.md_iterations * 9);
    }

    /// The forces of Möbius domain-wall fermions: two flavours (on the whole lattice and
    /// even-odd), and two over two Pauli-Villars flavours, on stout links.
    /// Solving with Neff's D Q changes neither the action nor the force, on all sites,
    /// even-odd and for the ratio to Pauli-Villars.
    #[test]
    fn neff_alpha_changes_the_solve_not_the_action() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 23);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
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
            max_restarts: 10,
        };
        let op = |p: MdwfParams, alpha: f64| {
            let mut op = Mdwf::<f64>::new(&lat, p);
            op.set_alpha(alpha);
            op
        };
        let cache = SmearingCache::new();
        let links = Links::new(&u, &cache);
        let same =
            |name: &str, plain: (f64, Field<f64, Gauge>), alpha: (f64, Field<f64, Gauge>)| {
                assert!(
                    (plain.0 - alpha.0).abs() < 1e-9 * plain.0.abs(),
                    "{name}: {} {}",
                    plain.0,
                    alpha.0
                );
                let worst = plain
                    .1
                    .as_slice()
                    .iter()
                    .zip(alpha.1.as_slice())
                    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
                assert!(worst < 1e-9, "{name}: forces differ by {worst:e}");
            };

        let run = |alpha: Option<f64>| {
            let mut a = TwoFlavor::new(op(params, 1.0), None, stop, stop);
            if let Some(alpha) = alpha {
                a = a.with_conditioned(op(params, alpha));
            }
            a.langevin(&links, &mut Mseries::new(7));
            (a.hamiltonian(&links), a.force(&links))
        };
        same("all sites", run(None), run(Some(0.6)));

        let run = |alpha: Option<f64>| {
            let mut a = TwoFlavorEo::new(op(params, 1.0), None, stop, stop);
            if let Some(alpha) = alpha {
                a = a.with_conditioned(op(params, alpha));
            }
            a.langevin(&links, &mut Mseries::new(7));
            (a.hamiltonian(&links), a.force(&links))
        };
        same("even-odd", run(None), run(Some(0.6)));

        let run = |alpha: Option<f64>| {
            let pv = op(params.pauli_villars(), 1.0);
            let mut a = TwoFlavorRatio::new(op(params, 1.0), pv, None, stop, stop);
            if let Some(alpha) = alpha {
                a = a.with_conditioned(op(params, alpha));
            }
            a.langevin(&links, &mut Mseries::new(7));
            (a.hamiltonian(&links), a.force(&links))
        };
        same("ratio", run(None), run(Some(0.6)));
    }

    #[test]
    fn domain_wall_forces_are_the_derivatives() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 23);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-30,
            max_iterations: 2000,
            max_restarts: 10,
        };
        let smearing = Some(GaugeSmearing {
            smear: Smear::Stout(Stout::uniform(0.1)),
            steps: 1,
        });
        let links = [(0, 9), (2, 77), (3, 250)];
        let mut two = TwoFlavor::new(Mdwf::<f64>::new(&lat, params), smearing, stop, stop);
        let cache = SmearingCache::new();
        two.langevin(&Links::new(&u, &cache), &mut Mseries::new(7));
        let f = two.force(&Links::new(&u, &cache));
        check_force(
            &u,
            &f,
            &links,
            |w| two.hamiltonian(&Links::new(w, &SmearingCache::new())),
            1e-7,
        );
        // Even-odd: T is the site-local part of the fifth dimension, the same for all links.
        let mut eo = TwoFlavorEo::new(Mdwf::<f64>::new(&lat, params), smearing, stop, stop);
        eo.langevin(&Links::new(&u, &cache), &mut Mseries::new(9));
        let f = eo.force(&Links::new(&u, &cache));
        check_force(
            &u,
            &f,
            &links,
            |w| eo.hamiltonian(&Links::new(w, &SmearingCache::new())),
            1e-7,
        );
        let mut ratio = TwoFlavorRatio::new(
            Mdwf::<f64>::new(&lat, params),
            Mdwf::<f64>::new(&lat, params.pauli_villars()),
            smearing,
            stop,
            stop,
        );
        ratio.langevin(&Links::new(&u, &cache), &mut Mseries::new(8));
        let f = ratio.force(&Links::new(&u, &cache));
        check_force(
            &u,
            &f,
            &links,
            |w| ratio.hamiltonian(&Links::new(w, &SmearingCache::new())),
            1e-7,
        );
    }

    /// The Ratio action of domain-wall fermions over Pauli-Villars fields with the multigrid in
    /// its solves (D† through Ã† and M†, D through Ã and M) gives the energy and the force of CG
    /// to the tolerance of the solves, and keeps its test vectors along the solves.
    #[test]
    fn domain_wall_multigrid_gives_the_action_and_force_of_cg() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::solver::preconditioner::multigrid::PvEoParams;
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 41);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-24,
            max_iterations: 4000,
            max_restarts: 10,
        };
        let mdwf = |p| Mdwf::<f64>::new(&lat, p);
        let cache = SmearingCache::new();
        let links = Links::new(&u, &cache);
        let mut cg =
            TwoFlavorRatio::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop);
        cg.langevin(&links, &mut Mseries::new(13));
        let (h_cg, f_cg) = (cg.hamiltonian(&links), cg.force(&links));
        let setup = DwfMultigridSetup::new(
            [2, 2, 2, 2],
            8,
            PvEoParams::default(),
            Stop::simple(1e-10, 200),
        );
        let mut mg =
            TwoFlavorRatio::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop)
                .with_preconditioner(Preconditioner::DomainWallMultigrid(Box::new(setup)));
        mg.langevin(&links, &mut Mseries::new(13));
        let (h_mg, f_mg) = (mg.hamiltonian(&links), mg.force(&links));
        let f2 = mg.force(&links);
        let worst = |a: &Field<f64, Gauge>, b: &Field<f64, Gauge>| {
            a.as_slice()
                .iter()
                .zip(b.as_slice())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        let size = f_cg.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        eprintln!(
            "H: CG {h_cg}, multigrid {h_mg}; forces differ by {:e} of {size:e}, a second force by {:e}",
            worst(&f_cg, &f_mg),
            worst(&f_cg, &f2)
        );
        assert!((h_cg - h_mg).abs() < 1e-9 * h_cg.abs());
        assert!(worst(&f_cg, &f_mg) < 1e-8 * size && worst(&f_cg, &f2) < 1e-8 * size);
        let Preconditioner::DomainWallMultigrid(setup) = &mg.preconditioner else {
            unreachable!()
        };
        eprintln!(
            "setups {}, bootstraps again {}, solves on the even sites {}",
            setup.setups, setup.refreshes, setup.inner_solves
        );
        assert_eq!(setup.setups, 1, "the test vectors are kept");
    }

    /// The even-odd Ratio action of domain-wall fermions with the multigrid, (K†K)⁻¹ through
    /// the two solves of D† and D on the even sites, gives the energy and the force of CG.
    #[test]
    fn domain_wall_multigrid_of_the_even_odd_ratio_gives_those_of_cg() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::solver::preconditioner::multigrid::PvEoParams;
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 43);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.05,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-24,
            max_iterations: 4000,
            max_restarts: 10,
        };
        let mdwf = |p| Mdwf::<f64>::new(&lat, p);
        let cache = SmearingCache::new();
        let links = Links::new(&u, &cache);
        let mut cg =
            TwoFlavorRatioEo::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop);
        cg.langevin(&links, &mut Mseries::new(17));
        let (h_cg, f_cg) = (cg.hamiltonian(&links), cg.force(&links));
        let setup = DwfMultigridSetup::new(
            [2, 2, 2, 2],
            8,
            PvEoParams::default(),
            Stop::simple(1e-10, 200),
        );
        let mut mg =
            TwoFlavorRatioEo::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop)
                .with_solver(NormalSolver::Gcr(8))
                .with_preconditioner(Preconditioner::DomainWallMultigrid(Box::new(setup)));
        mg.langevin(&links, &mut Mseries::new(17));
        let (h_mg, f_mg) = (mg.hamiltonian(&links), mg.force(&links));
        let worst = |a: &Field<f64, Gauge>, b: &Field<f64, Gauge>| {
            a.as_slice()
                .iter()
                .zip(b.as_slice())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        let size = f_cg.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        eprintln!(
            "H: CG {h_cg}, multigrid {h_mg}; forces differ by {:e} of {size:e}",
            worst(&f_cg, &f_mg)
        );
        assert!((h_cg - h_mg).abs() < 1e-9 * h_cg.abs());
        assert!(worst(&f_cg, &f_mg) < 1e-8 * size);
        let Preconditioner::DomainWallMultigrid(setup) = &mg.preconditioner else {
            unreachable!()
        };
        assert_eq!(setup.setups, 1, "the test vectors are kept");
        assert!(setup.inner_solves >= 4, "two solves of each kind at least");
        // The coarse solves in double precision give the same.
        let mut double = DwfMultigridSetup::new(
            [2, 2, 2, 2],
            8,
            PvEoParams::default(),
            Stop::simple(1e-10, 200),
        );
        double.coarse_double = true;
        let mut mg =
            TwoFlavorRatioEo::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop)
                .with_solver(NormalSolver::Gcr(8))
                .with_preconditioner(Preconditioner::DomainWallMultigrid(Box::new(double)));
        mg.langevin(&links, &mut Mseries::new(17));
        let h_double = mg.hamiltonian(&links);
        assert!((h_cg - h_double).abs() < 1e-9 * h_cg.abs(), "{h_double}");
    }

    /// The multigrid as a cache (P and the coarse parts kept from solve to solve, rebuilt when
    /// the iterations grow) gives the energy and the force of CG on links that moved.
    #[test]
    fn multigrid_cache_gives_the_action_and_force_of_cg() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::solver::preconditioner::multigrid::PvEoParams;
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 53);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.05,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-24,
            max_iterations: 4000,
            max_restarts: 10,
        };
        let mdwf = |p| Mdwf::<f64>::new(&lat, p);
        let cache = SmearingCache::new();
        let mut setup = DwfMultigridSetup::new(
            [2, 2, 2, 2],
            8,
            PvEoParams::default(),
            Stop::simple(1e-10, 200),
        );
        setup.rebuild_growth = 1.5;
        let mut mg =
            TwoFlavorRatio::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop)
                .with_preconditioner(Preconditioner::DomainWallMultigrid(Box::new(setup)));
        let mut cg =
            TwoFlavorRatio::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop);
        let links = Links::new(&u, &cache);
        mg.langevin(&links, &mut Mseries::new(23));
        cg.langevin(&links, &mut Mseries::new(23));
        // The links move a little from solve to solve, as in a trajectory.
        let mut v = random_gauge(&lat, 53);
        let mut worst_h = 0.0f64;
        let mut worst_f = 0.0f64;
        let mut rng = Mseries::new(29);
        for _ in 0..4 {
            let p = draw_momenta(&lat, &mut rng);
            exp_update(0.02, &p, &mut v, 8);
            let links = Links::new(&v, &cache);
            let (h_mg, h_cg) = (mg.hamiltonian(&links), cg.hamiltonian(&links));
            let (f_mg, f_cg) = (mg.force(&links), cg.force(&links));
            let size = f_cg.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let df = f_cg
                .as_slice()
                .iter()
                .zip(f_mg.as_slice())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            worst_h = worst_h.max((h_mg - h_cg).abs() / h_cg.abs());
            worst_f = worst_f.max(df / size);
        }
        let Preconditioner::DomainWallMultigrid(setup) = &mg.preconditioner else {
            unreachable!()
        };
        eprintln!(
            "cache: H {worst_h:e}, force {worst_f:e} relative; rebuilds {}, reuses {}, solves on the even sites {}",
            setup.rebuilds, setup.reuses, setup.inner_solves
        );
        assert!(worst_h < 1e-9 && worst_f < 1e-8);
        assert!(setup.reuses > 0, "the coarse parts are reused");
    }

    /// Mixed precision, CG in single precision corrected in double, gives the energy and the
    /// force of CG in double precision, for the Ratio action on all sites and even-odd.
    #[test]
    fn mixed_precision_gives_the_action_and_force_of_cg() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = random_gauge(&lat, 47);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.05,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-24,
            max_iterations: 4000,
            max_restarts: 10,
        };
        let mdwf = |p| Mdwf::<f64>::new(&lat, p);
        let cache = SmearingCache::new();
        let links = Links::new(&u, &cache);
        let worst = |a: &Field<f64, Gauge>, b: &Field<f64, Gauge>| {
            a.as_slice()
                .iter()
                .zip(b.as_slice())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        // On all sites.
        let mut cg =
            TwoFlavorRatio::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop);
        cg.langevin(&links, &mut Mseries::new(19));
        let (h_cg, f_cg) = (cg.hamiltonian(&links), cg.force(&links));
        let mut mixed =
            TwoFlavorRatio::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop)
                .with_single(Box::new(SingleMdwf::<crate::kernel::Dirac>::new(
                    &lat, params, 1.0, 1e-10,
                )));
        mixed.langevin(&links, &mut Mseries::new(19));
        let (h_m, f_m) = (mixed.hamiltonian(&links), mixed.force(&links));
        let size = f_cg.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        eprintln!(
            "all sites: H {h_cg} and {h_m}; forces differ by {:e} of {size:e}",
            worst(&f_cg, &f_m)
        );
        assert!((h_cg - h_m).abs() < 1e-9 * h_cg.abs());
        assert!(worst(&f_cg, &f_m) < 1e-8 * size);
        // Even-odd.
        let mut cg =
            TwoFlavorRatioEo::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop);
        cg.langevin(&links, &mut Mseries::new(19));
        let (h_cg, f_cg) = (cg.hamiltonian(&links), cg.force(&links));
        let mut mixed =
            TwoFlavorRatioEo::new(mdwf(params), mdwf(params.pauli_villars()), None, stop, stop)
                .with_single(Box::new(SingleMdwfEo::<crate::kernel::Dirac>::new(
                    &lat, params, 1e-10,
                )));
        mixed.langevin(&links, &mut Mseries::new(19));
        let (h_m, f_m) = (mixed.hamiltonian(&links), mixed.force(&links));
        let size = f_cg.as_slice().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        eprintln!(
            "even-odd: H {h_cg} and {h_m}; forces differ by {:e} of {size:e}; iterations {} (double CG {})",
            worst(&f_cg, &f_m),
            mixed.last_iterations,
            cg.last_iterations
        );
        assert!((h_cg - h_m).abs() < 1e-9 * h_cg.abs());
        assert!(worst(&f_cg, &f_m) < 1e-8 * size);
    }

    /// The molecular dynamics is reversible, for the ratio of domain-wall fermions over
    /// Pauli-Villars fields on all sites and even-odd, on stout links with the rectangle action,
    /// on two levels of Omelyan's integrator: forward, P → −P, back gives the links and momenta
    /// of the start to rounding and H to the solves.
    #[test]
    fn molecular_dynamics_is_reversible() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 37);
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-30,
            max_iterations: 4000,
            max_restarts: 10,
        };
        let smearing = Some(GaugeSmearing {
            smear: Smear::Stout(Stout::uniform(0.1)),
            steps: 1,
        });
        let mdwf = |p| Mdwf::<f64>::new(&lat, p);
        let actions: [Box<dyn MdAction>; 2] = [
            Box::new(TwoFlavorRatio::new(
                mdwf(params),
                mdwf(params.pauli_villars()),
                smearing,
                stop,
                stop,
            )),
            Box::new(TwoFlavorRatioEo::new(
                mdwf(params),
                mdwf(params.pauli_villars()),
                smearing,
                stop,
                stop,
            )),
        ];
        for (name, fermions) in ["all sites", "even-odd"].into_iter().zip(actions) {
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
                vec![3, 2],
                12,
                0.5,
            );
            let r = hmc.reversibility(&u, &mut Mseries::new(11));
            eprintln!("{name}: {r:?}");
            assert!(
                r.max_link < 1e-12 && r.max_momentum < 1e-10,
                "{name}: {r:?}"
            );
            assert!(r.delta_h_back.abs() < 1e-8, "{name}: {r:?}");
        }
    }

    /// The even-odd ratio over Pauli-Villars fields (Bridge++'s `Action_F_alt_Ratio_eo`), of
    /// Möbius domain-wall and of Wilson fermions, on stout links: S = |ξ_e|² right after the
    /// heat bath, and the force is the derivative of the action.
    #[test]
    fn even_odd_ratio_force_is_the_derivative() {
        use crate::fermion::{Mdwf, MdwfParams, Wilson, WilsonParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 23);
        let boundary = [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ];
        let params = MdwfParams {
            ls: 4,
            m5: 1.6,
            mass: 0.1,
            b: 1.5,
            c: 0.5,
            boundary,
        };
        let stop = Stop {
            tolerance_sqr: 1e-30,
            max_iterations: 4000,
            max_restarts: 10,
        };
        let smearing = Some(GaugeSmearing {
            smear: Smear::Stout(Stout::uniform(0.1)),
            steps: 1,
        });
        let sites = [(0, 9), (2, 77), (3, 250)];
        let cache = SmearingCache::new();
        fn check<O: EvenOddForceOp>(
            mut a: TwoFlavorRatioEo<O>,
            u: &Field<f64, Gauge>,
            cache: &SmearingCache,
            sites: &[(usize, usize)],
        ) {
            let s0 = a.langevin(&Links::new(u, cache), &mut Mseries::new(8));
            let h = a.hamiltonian(&Links::new(u, cache));
            assert!((h - s0).abs() < 1e-9 * s0, "S {h} against |xi|^2 {s0}");
            let f = a.force(&Links::new(u, cache));
            check_force(
                u,
                &f,
                sites,
                |w| a.hamiltonian(&Links::new(w, &SmearingCache::new())),
                1e-7,
            );
        }
        let mdwf = |p| Mdwf::<f64>::new(&lat, p);
        check(
            TwoFlavorRatioEo::new(
                mdwf(params),
                mdwf(params.pauli_villars()),
                smearing,
                stop,
                stop,
            ),
            &u,
            &cache,
            &sites,
        );
        let wilson = |kappa| Wilson::<f64>::new(&lat, WilsonParams { kappa, boundary });
        check(
            TwoFlavorRatioEo::new(wilson(0.12), wilson(0.11), smearing, stop, stop),
            &u,
            &cache,
            &sites,
        );
    }

    /// One flavour of clover fermions by rational approximation (RHMC), on stout links: the
    /// force, and the heat bath (S = |ξ|² at the start, to the error of the approximations);
    /// and the ratio of two clover operators, which the 2 + 1 flavours need too.
    #[test]
    fn rational_and_clover_ratio_forces_are_the_derivatives() {
        use crate::fermion::{Clover, CloverParams};
        use crate::gauge::smearing::Stout;
        use crate::gauge::{GaugeSmearing, Smear};
        use crate::numerics::{RationalApprox, RationalSpec};
        use crate::solver::Stop;
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 29);
        let params = |kappa| CloverParams {
            kappa,
            csw: 1.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let stop = Stop {
            tolerance_sqr: 1e-30,
            max_iterations: 2000,
            max_restarts: 10,
        };
        let smearing = Some(GaugeSmearing {
            smear: Smear::Stout(Stout::uniform(0.1)),
            steps: 1,
        });
        // D†D of κ = 0.12 on these links lies in [1e-4, 16] (λ_max of D at most 1 + 8κ plus
        // the clover term).
        let spec = RationalSpec {
            poles: 16,
            p: -1,
            q: 2,
            x_min: 1e-4,
            x_max: 16.0,
        };
        let md = RationalApprox::minimax(&spec).unwrap();
        let hb = RationalApprox::minimax(&spec.with_power(1, 4)).unwrap();
        let links = [(0, 11), (2, 80), (3, 251)];
        let mut one = Rational::new(
            Clover::<f64>::new(&lat, params(0.12)),
            smearing,
            md,
            hb,
            stop,
            stop,
        )
        .with_clover(0.12 * 1.5);
        let cache = SmearingCache::new();
        let xi2 = one.langevin(&Links::new(&u, &cache), &mut Mseries::new(9));
        let s0 = one.hamiltonian(&Links::new(&u, &cache));
        eprintln!("|xi|^2 {xi2}, S {s0}");
        assert!(((s0 - xi2) / xi2).abs() < 1e-8, "{s0} against {xi2}");
        let f = one.force(&Links::new(&u, &cache));
        check_force(
            &u,
            &f,
            &links,
            |w| one.hamiltonian(&Links::new(w, &SmearingCache::new())),
            1e-7,
        );
        let mut ratio = TwoFlavorRatio::new(
            Clover::<f64>::new(&lat, params(0.12)),
            Clover::<f64>::new(&lat, params(0.10)),
            smearing,
            stop,
            stop,
        )
        .with_clover(0.12 * 1.5, 0.10 * 1.5);
        ratio.langevin(&Links::new(&u, &cache), &mut Mseries::new(10));
        let f = ratio.force(&Links::new(&u, &cache));
        check_force(
            &u,
            &f,
            &links,
            |w| ratio.hamiltonian(&Links::new(w, &SmearingCache::new())),
            1e-7,
        );
    }

    #[test]
    fn reunitarize_keeps_su3() {
        let lat = Lattice::single_node([2, 2, 2, 2]);
        let u = random_gauge(&lat, 5);
        let m = link(&u, 1, 3);
        let back = reunitarize(&m);
        for (a, b) in back.0.iter().flatten().zip(m.0.iter().flatten()) {
            assert!((*a - *b).norm_sqr() < 1e-24);
        }
    }
}
