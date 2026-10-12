//! Hybrid Monte Carlo: gauge configurations from the momenta, the actions and their forces,
//! a molecular-dynamics integrator and the Metropolis test.
//!
//! The conventions are those of Bridge++, so that its runs can be repeated here and checked
//! number by number (`tests/stage5_hmc.rs`): the momenta iP = i Σ_a 2 h_a T^a with h_a
//! Gaussian of variance 1/2 drawn by the M-series generator (`Mseries`) in Bridge++'s order,
//! H = ½ |iP|² + S; the leapfrog P(ε/2) [U(ε) P(ε)]… U(ε) P(ε/2); U ← exp(ε iP) U by the
//! Taylor series to order `exp_order`, then made unitary again row by row; and the Metropolis
//! test with the next number of the same generator.
//!
//! Actions: the plaquette and rectangle (Iwasaki, DBW2, …) gauge actions; two flavours of
//! Wilson-type or domain-wall fermions, and their ratio to Pauli-Villars fermions, on links
//! that may be stout smeared (`fermion`); fractional powers of the determinant, one flavour,
//! by rational approximation (`rational`). Several time scales with leapfrog or Omelyan
//! (`integrator`). An action is in the file of its kind, with what makes it an `MdAction`.

mod fermion;
mod gauge_force;
mod integrator;
mod links;
mod mseries;
mod rational;

pub use rational::Rational;

pub use fermion::{
    clover_derivative, force_from, hop_derivative, kernel_derivative, smeared_force, ForceOp,
    TwoFlavor, TwoFlavorRatio,
};
pub use gauge_force::{
    force_statistics, path_product, staples, traceless_antihermitian, PlaquetteAction,
    RectangleAction, Step,
};
pub use integrator::{Hmc, HmcTrajectory, MdAction, Scheme, Term};
pub use links::{Links, SmearingCache};
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
        use crate::gauge::smear::{spatial, Stout};
        use crate::gauge::{GaugeSmearing, Smear};
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let u = random_gauge(&lat, 13);
        let action = PlaquetteAction { beta: 5.0 };
        for rho in [crate::gauge::smear::uniform(0.1), spatial(0.15)] {
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
        use crate::gauge::smear::Stout;
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
        use crate::gauge::smear::Stout;
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

    /// The forces of Möbius domain-wall fermions: two flavours, and two over two
    /// Pauli-Villars flavours, on stout links.
    #[test]
    fn domain_wall_forces_are_the_derivatives() {
        use crate::fermion::{Mdwf, MdwfParams};
        use crate::gauge::smear::Stout;
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

    /// One flavour of clover fermions by rational approximation (RHMC), on stout links: the
    /// force, and the heat bath (S = |ξ|² at the start, to the error of the approximations);
    /// and the ratio of two clover operators, which the 2 + 1 flavours need too.
    #[test]
    fn rational_and_clover_ratio_forces_are_the_derivatives() {
        use crate::fermion::{Clover, CloverParams};
        use crate::gauge::smear::Stout;
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
