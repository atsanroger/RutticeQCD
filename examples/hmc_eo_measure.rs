//! Scratch: CG iterations and seconds of one trajectory, whole lattice against even-odd, with
//! and without the chronological guess. Not committed.
use ruttice::fermion::{Clover, CloverParams, Wilson, WilsonParams};
use ruttice::gauge::smearing::Stout;
use ruttice::gauge::{link, set_link, GaugeSmearing, Smear};
use ruttice::hmc::{
    Hmc, Links, MdAction, Mseries, RectangleAction, Scheme, Term, TwoFlavor, TwoFlavorEo,
};
use ruttice::solver::Stop;
use ruttice::{Boundary, Field, Gauge, Lattice, ND};
use std::cell::Cell;
use std::rc::Rc;

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
    fn force(&mut self, l: &Links) -> Field<f64, Gauge> {
        let f = self.action.force(l);
        self.count.set((self.read)(&self.action));
        f
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("conf_04040408.txt");
    let small = Lattice::single_node([4, 4, 4, 8]);
    let u4 = ruttice::io::read_bridge_text(std::path::Path::new(&path), &small).unwrap();
    let lat = Lattice::single_node([8; ND]);
    let mut u0 = Field::<f64, Gauge>::gauge(&lat);
    for site in 0..lat.num_sites() {
        let (x, y, z, t) = (site % 8, (site / 8) % 8, (site / 64) % 8, site / 512);
        let from = x % 4 + 4 * (y % 4 + 4 * (z % 4 + 4 * t));
        for mu in 0..ND {
            set_link(&mut u0, mu, site, &link(&u4, mu, from));
        }
    }
    let boundary = [
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Periodic,
        Boundary::Antiperiodic,
    ];
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 2000,
        max_restarts: 40,
    };
    let smearing = Some(GaugeSmearing {
        smear: Smear::Stout(Stout::uniform(0.1)),
        steps: 2,
    });
    let kappa = 0.12;
    for clover in [false, true] {
        for even_odd in [false, true] {
            for guess in [0, 4] {
                let count = Rc::new(Cell::new(0));
                let c = count.clone();
                let fermions: Box<dyn MdAction> = match (clover, even_odd) {
                    (false, false) => Box::new(Counted {
                        action: TwoFlavor::new(
                            Wilson::<f64>::new(&lat, WilsonParams { kappa, boundary }),
                            smearing,
                            stop,
                            stop,
                        )
                        .with_guess(guess),
                        count: c,
                        read: |a| a.md_iterations,
                    }),
                    (false, true) => Box::new(Counted {
                        action: TwoFlavorEo::new(
                            Wilson::<f64>::new(&lat, WilsonParams { kappa, boundary }),
                            smearing,
                            stop,
                            stop,
                        )
                        .with_guess(guess),
                        count: c,
                        read: |a| a.md_iterations,
                    }),
                    (true, false) => Box::new(Counted {
                        action: TwoFlavor::new(
                            Clover::<f64>::new(
                                &lat,
                                CloverParams {
                                    kappa,
                                    csw: 1.0,
                                    boundary,
                                },
                            ),
                            smearing,
                            stop,
                            stop,
                        )
                        .with_clover(kappa)
                        .with_guess(guess),
                        count: c,
                        read: |a| a.md_iterations,
                    }),
                    (true, true) => Box::new(Counted {
                        action: TwoFlavorEo::new(
                            Clover::<f64>::new(
                                &lat,
                                CloverParams {
                                    kappa,
                                    csw: 1.0,
                                    boundary,
                                },
                            ),
                            smearing,
                            stop,
                            stop,
                        )
                        .with_clover(kappa)
                        .with_guess(guess),
                        count: c,
                        read: |a| a.md_iterations,
                    }),
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
                    vec![10, 4],
                    8,
                    1.0,
                );
                let mut u = Field::<f64, Gauge>::gauge(&lat);
                u.as_mut_slice().copy_from_slice(u0.as_slice());
                let start = std::time::Instant::now();
                let t = hmc.trajectory(&mut u, &mut Mseries::new(200000));
                println!(
                    "{} {} guess {guess}: {:5} CG iterations, {:6.2} s, dH {:+.6}, plaq {:.12}",
                    if clover { "clover" } else { "wilson" },
                    if even_odd { "even-odd" } else { "whole   " },
                    count.get(),
                    start.elapsed().as_secs_f64(),
                    t.delta_h,
                    t.plaquette_trial
                );
            }
        }
    }
}
