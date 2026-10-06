//! Checks: a lattice split over several processes gives the same results as one process.
//!
//! The processes here are threads of the test (`LocalComm`); each has its own lattice,
//! fields and operator, and data crosses only through halo exchanges and global sums. No
//! MPI library is involved, so these tests say nothing about an MPI backend.

mod common;

use ruttice::fermion::{
    solve_even_odd, Bound, Dagger, EvenOdd, FermionOp, HopTerms, Variant, Wilson, WilsonParams,
};
use ruttice::gauge::random_gauge;
use ruttice::solver::{solve_cgne, BiCgStab, Cg, Solver, Stop};
use ruttice::{assemble, local_part, random_spinor};
use ruttice::{
    extract_parity, insert_parity, shift, Boundary, ConvertFrom, Direction, Field, Gauge, Half,
    Kind, Lattice, LocalComm, Parity, Spinor, Stencil, Storage, Vector, C64, ND,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;

const GLOBAL: [usize; ND] = [4, 4, 4, 8];

/// One process, one direction split, and several directions split. With two processes
/// along a direction the neighbour is the same on both sides.
const GRIDS: [[usize; ND]; 5] = [
    [1, 1, 1, 1],
    [2, 1, 1, 1],
    [1, 2, 1, 4],
    [2, 2, 2, 2],
    [1, 1, 2, 4],
];

const MIXED: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];
const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];

fn whole() -> Arc<Lattice> {
    Lattice::single_node(GLOBAL)
}

/// Run `f` on every process of the grid and put the fields it returns together.
fn distributed<T, K, F>(grid: [usize; ND], f: F) -> Field<T, K>
where
    T: Storage,
    K: Kind,
    F: Fn(&Arc<Lattice>) -> Field<T, K> + Sync,
{
    let parts = LocalComm::run(GLOBAL, grid, |lat| f(&lat));
    assemble(&whole(), &parts)
}

fn wilson<T: Storage>(
    lat: &Arc<Lattice>,
    boundary: [Boundary; ND],
    u: &Field<f64, Gauge>,
) -> Wilson<T> {
    let mut op = Wilson::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary,
        },
    );
    op.set_gauge(&local_part(u, lat)).unwrap();
    op
}

#[test]
fn shift_across_processes() {
    let lat = whole();
    let a = random_gauge(&lat, 501);
    let mut boundary = [Boundary::Periodic; ND];
    boundary[3] = Boundary::Dirichlet;
    for mu in 0..ND {
        for dir in [Direction::Forward, Direction::Backward] {
            let mut want = Field::<f64, Gauge>::gauge(&lat);
            shift(&mut want, &a, &Stencil::nearest(&lat, boundary), mu, dir);
            for grid in GRIDS {
                let got = distributed(grid, |lat| {
                    let mut out = Field::<f64, Gauge>::gauge(lat);
                    let st = Stencil::nearest(lat, boundary);
                    shift(&mut out, &local_part(&a, lat), &st, mu, dir);
                    out
                });
                assert!(
                    got.as_slice() == want.as_slice(),
                    "grid {grid:?}, mu {mu}, {dir:?}"
                );
            }
        }
    }
}

/// D, D† and γ5 D on two spinors at once, bit for bit.
#[test]
fn wilson_mult_is_identical() {
    let lat = whole();
    let u = random_gauge(&lat, 511);
    let psi = random_spinor(&lat, 2, 512);
    for boundary in [MIXED, ANTIPERIODIC] {
        for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
            let op = wilson::<f64>(&lat, boundary, &u);
            let mut want = psi.zeros_like();
            op.mult(&mut op.work(), &mut want, &psi, variant);
            for grid in GRIDS {
                let got = distributed(grid, |lat| {
                    let op = wilson::<f64>(lat, boundary, &u);
                    let inp = local_part(&psi, lat);
                    let mut out = inp.zeros_like();
                    let mut work = op.work();
                    // Twice with the same work object: the exchange buffers are reused.
                    op.mult(&mut work, &mut out, &inp, variant);
                    op.mult(&mut work, &mut out, &inp, variant);
                    out
                });
                assert!(
                    got.as_slice() == want.as_slice(),
                    "grid {grid:?}, {variant:?}"
                );
            }
        }
    }
}

#[test]
fn wilson_mult_is_identical_in_single_precision() {
    let lat = whole();
    let u = random_gauge(&lat, 521);
    let mut psi = Field::<f32, Spinor>::spinor(&lat, 1);
    psi.convert_from(&random_spinor(&lat, 1, 522));
    let op = wilson::<f32>(&lat, MIXED, &u);
    let mut want = psi.zeros_like();
    op.mult(&mut op.work(), &mut want, &psi, Variant::Dagger);
    for grid in GRIDS {
        let got = distributed(grid, |lat| {
            let op = wilson::<f32>(lat, MIXED, &u);
            let inp = local_part(&psi, lat);
            let mut out = inp.zeros_like();
            op.mult(&mut op.work(), &mut out, &inp, Variant::Dagger);
            out
        });
        assert!(got.as_slice() == want.as_slice(), "grid {grid:?}");
    }
}

#[test]
fn hopping_terms_are_identical() {
    let lat = whole();
    let u = random_gauge(&lat, 531);
    let psi = random_spinor(&lat, 2, 532);
    let op = wilson::<f64>(&lat, MIXED, &u);
    for mu in 0..ND {
        for dir in [Direction::Forward, Direction::Backward] {
            let apply = |op: &Wilson<f64>, inp: &V| {
                // The terms add to what is already there.
                let mut out = inp.zeros_like();
                out.copy_from(inp);
                let mut work = op.work();
                match dir {
                    Direction::Forward => op.hop_forward(&mut work, mu, &mut out, inp),
                    Direction::Backward => op.hop_backward(&mut work, mu, &mut out, inp),
                }
                out
            };
            let want = apply(&op, &psi);
            for grid in GRIDS {
                let got = distributed(grid, |lat| {
                    apply(&wilson::<f64>(lat, MIXED, &u), &local_part(&psi, lat))
                });
                assert!(
                    got.as_slice() == want.as_slice(),
                    "grid {grid:?}, mu {mu}, {dir:?}"
                );
            }
        }
    }
}

/// The even-odd hopping blocks, written back into a full field for comparison.
#[test]
fn even_odd_blocks_are_identical() {
    let lat = whole();
    let u = random_gauge(&lat, 541);
    let psi = random_spinor(&lat, 2, 542);
    let block = |op: &Wilson<f64>, full: &V, to: Parity, dagger: Dagger| {
        let index = op.even_odd_index().unwrap();
        let mut inp = H::spinor_half(full.lattice(), 2, to.flip());
        let mut out = H::spinor_half(full.lattice(), 2, to);
        extract_parity(&mut inp, full, index, to.flip());
        op.hop(&mut op.even_odd_work(), &mut out, &inp, to, dagger);
        let mut result = full.zeros_like();
        insert_parity(&mut result, &out, index);
        result
    };
    let op = wilson::<f64>(&lat, MIXED, &u);
    for to in [Parity::Even, Parity::Odd] {
        for dagger in [Dagger::No, Dagger::Yes] {
            let want = block(&op, &psi, to, dagger);
            // Every local extent must be even for an even-odd decomposition.
            for grid in [[2, 1, 1, 1], [1, 2, 1, 4], [2, 2, 2, 2]] {
                let got = distributed(grid, |lat| {
                    let op = wilson::<f64>(lat, MIXED, &u);
                    block(&op, &local_part(&psi, lat), to, dagger)
                });
                assert!(
                    got.as_slice() == want.as_slice(),
                    "grid {grid:?}, to {to:?}, {dagger:?}"
                );
            }
        }
    }
}

/// Inner products and norms are sums over all processes, and every process gets the same
/// bits. They differ from the single-process value by rounding only: the order of the sum
/// follows the process grid.
#[test]
fn reductions_are_global() {
    let lat = whole();
    let a = random_spinor(&lat, 2, 551);
    let b = random_spinor(&lat, 2, 552);
    let mut want_dot = [C64::ZERO; 2];
    let mut want_norm = [0.0; 2];
    a.dot(&b, &mut want_dot);
    a.norm_sqr(&mut want_norm);
    for grid in GRIDS {
        let got = LocalComm::run(GLOBAL, grid, |lat| {
            let (a, b) = (local_part(&a, &lat), local_part(&b, &lat));
            let mut dot = [C64::ZERO; 2];
            let mut norm = [0.0; 2];
            a.dot(&b, &mut dot);
            a.norm_sqr(&mut norm);
            (dot.map(|z| [z.re, z.im]), norm)
        });
        for (dot, norm) in &got {
            assert_eq!((dot, norm), (&got[0].0, &got[0].1), "grid {grid:?}");
            for j in 0..2 {
                assert!((dot[j][0] - want_dot[j].re).abs() < 1e-11 * want_dot[j].re.abs());
                assert!((dot[j][1] - want_dot[j].im).abs() < 1e-11);
                assert!((norm[j] - want_norm[j]).abs() < 1e-12 * want_norm[j]);
            }
        }
    }
}

/// `|b - D x|^2 / |b|^2` per vector on the single-process lattice.
fn residual_sqr(u: &Field<f64, Gauge>, x: &V, b: &V) -> Vec<f64> {
    let op = wilson::<f64>(&whole(), ANTIPERIODIC, u);
    let nv = b.num_vectors();
    let mut r = b.zeros_like();
    op.mult(&mut op.work(), &mut r, x, Variant::D);
    r.scale(&vec![-1.0; nv]);
    r.axpy(&vec![C64::new(1.0, 0.0); nv], b);
    let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    rr.iter().zip(&bb).map(|(r, b)| r / b).collect()
}

/// The solvers run unchanged on several processes: they only see vectors and operators.
#[test]
fn solvers_converge_on_several_processes() {
    let lat = whole();
    let u = random_gauge(&lat, 561);
    let b = random_spinor(&lat, 2, 562);
    let stop = Stop {
        tolerance_sqr: 1e-24,
        max_iterations: 1000,
        max_restarts: 3,
    };
    for grid in [[2, 1, 1, 2], [2, 2, 2, 2]] {
        let x = distributed(grid, |lat| {
            let op = wilson::<f64>(lat, ANTIPERIODIC, &u);
            let b = local_part(&b, lat);
            let mut x = b.zeros_like();
            let mut d = Bound::new(&op, Variant::D);
            solve_cgne(&mut Cg::new(stop), &mut d, &mut x, &b).unwrap();
            x
        });
        let r = residual_sqr(&u, &x, &b);
        assert!(r.iter().all(|r| *r < 2e-24), "CG, grid {grid:?}: {r:?}");

        let x = distributed(grid, |lat| {
            let op = wilson::<f64>(lat, ANTIPERIODIC, &u);
            let b = local_part(&b, lat);
            let mut x = b.zeros_like();
            let mut d = Bound::new(&op, Variant::D);
            BiCgStab::new(stop).solve(&mut d, None, &mut x, &b).unwrap();
            x
        });
        let r = residual_sqr(&u, &x, &b);
        assert!(
            r.iter().all(|r| *r < 2e-24),
            "BiCGStab, grid {grid:?}: {r:?}"
        );

        let x = distributed(grid, |lat| {
            let op = wilson::<f64>(lat, ANTIPERIODIC, &u);
            let b = local_part(&b, lat);
            let mut x = b.zeros_like();
            let index = op.even_odd_index().unwrap();
            solve_even_odd(&op, index, &mut x, &b, |schur, x_e, rhs| {
                solve_cgne(&mut Cg::new(stop), schur, x_e, rhs)
            })
            .unwrap();
            x
        });
        let r = residual_sqr(&u, &x, &b);
        assert!(
            r.iter().all(|r| *r < 1e-22),
            "even-odd, grid {grid:?}: {r:?}"
        );
    }
}
