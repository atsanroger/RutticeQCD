//! Three-point functions by a sequential source through the sink (DESIGN §7.10). Bridge++ has
//! none, so the checks come from outside the code another way: the conserved vector current
//! (its charge does not change between the source and the sink, and jumps there by the number
//! of quarks of its flavour times the two-point function, a Ward identity exact on every
//! configuration), and the sequential source through the current, which must give the same
//! numbers with momenta and a current that is not the charge.

use ruttice::fermion::{solve_even_odd, Clover, CloverParams, FermionOp};
use ruttice::gauge::random_gauge;
use ruttice::kernel::{Dirac, Gamma, GammaRepre};
use ruttice::measure::{
    conserved_charge, current_insertion, meson_correlator_with_momentum, point_source,
    sequential_source, Flavour, MesonSink, ProtonSink, Sink,
};
use ruttice::solver::{solve_cgne, Cg, Stop};
use ruttice::{Boundary, Field, Gauge, Lattice, Spinor, Vector, C64, ND};
use std::marker::PhantomData;
use std::sync::Arc;

type V = Field<f64, Spinor>;

const TIME_ANTIPERIODIC: [Boundary; ND] = [
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Periodic,
    Boundary::Antiperiodic,
];
const SOURCE: [usize; ND] = [1, 2, 3, 1];
const T_SINK: usize = 5;
const KAPPA: [f64; 2] = [0.12, 0.115];

/// The clover operator of flavour `f` (u: κ 0.12, d: κ 0.115) on `u`.
fn clover(lat: &Arc<Lattice>, u: &Field<f64, Gauge>, f: usize) -> Clover<f64> {
    let mut op = Clover::<f64>::new(
        lat,
        CloverParams {
            kappa: KAPPA[f],
            csw: 1.0,
            boundary: TIME_ANTIPERIODIC,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

/// D x = b for every vector of `b`, even-odd CGNE to a relative residual of 1e-14.
fn solve(op: &Clover<f64>, b: &V) -> V {
    let stop = Stop {
        tolerance_sqr: 1e-28,
        max_iterations: 4000,
        max_restarts: 3,
    };
    let mut x = b.zeros_like();
    solve_even_odd(
        op,
        op.even_odd_index().unwrap(),
        &mut x,
        b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap();
    x
}

struct Setup {
    u: Field<f64, Gauge>,
    ops: [Clover<f64>; 2],
    props: [V; 2],
}

fn setup() -> Setup {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 701);
    let ops = [clover(&lat, &u, 0), clover(&lat, &u, 1)];
    let b = point_source::<f64>(&lat, SOURCE);
    let props = [solve(&ops[0], &b), solve(&ops[1], &b)];
    Setup { u, ops, props }
}

/// The charge of flavour `f` (0 = u, 1 = d) through the sink `sink`: constant on the links
/// between the source and the sink and on the others, the jump at the sink `quarks` times the
/// two-point function there.
fn check_charge<K: Sink>(s: &Setup, sink: &K, f: usize, quarks: f64) {
    let props = [&s.props[0], &s.props[1]];
    let eta = sequential_source::<Dirac, K>(sink, props, T_SINK, [0; 3], SOURCE);
    let sigma = solve(&s.ops[f], &eta);
    let q = conserved_charge::<Dirac>(&s.props[f], &sigma, &s.u, KAPPA[f], Boundary::Antiperiodic);
    let c2 = sink.two_point(props, [0; 3], SOURCE);
    let nt = q.len();
    let inside = |t: usize| (SOURCE[3]..T_SINK).contains(&t);
    let size = q.iter().fold(c2[T_SINK].norm_sqr().sqrt(), |m, z| {
        m.max(z.norm_sqr().sqrt())
    });
    let tolerance = 1e-9 * size;
    let (a, b) = (q[SOURCE[3]], q[T_SINK]);
    for (t, z) in q.iter().enumerate() {
        let expected = if inside(t) { a } else { b };
        let d = (*z - expected).norm_sqr().sqrt();
        assert!(
            d <= tolerance,
            "Q on link {t}: {z:?}, expected {expected:?} (of {nt})"
        );
    }
    let jump = a - b;
    let d = (jump - c2[T_SINK].scale(quarks)).norm_sqr().sqrt();
    eprintln!(
        "jump {jump:?}, {quarks} C2 {:?}, difference {d:.2e}",
        c2[T_SINK].scale(quarks)
    );
    assert!(
        d <= tolerance,
        "jump {jump:?} against {quarks} C2 = {:?}",
        c2[T_SINK].scale(quarks)
    );
}

#[test]
fn meson_charge_is_conserved() {
    let s = setup();
    let g5 = Dirac::GAMMA5;
    let sink = MesonSink::<Dirac> {
        sink: g5,
        source: g5,
        gammas: PhantomData,
    };
    check_charge(&s, &sink, 0, 1.0);
}

#[test]
fn proton_charges_are_conserved() {
    let s = setup();
    let identity = Dirac::GAMMA[3].mul(&Dirac::GAMMA[3]);
    let half = C64::new(0.5, 0.0);
    let projection = vec![(identity, half), (Dirac::GAMMA[3], half)];
    for (current, f, quarks) in [(Flavour::U, 0, 2.0), (Flavour::D, 1, 1.0)] {
        let sink = ProtonSink::<Dirac> {
            projection: projection.clone(),
            current,
            gammas: PhantomData,
        };
        check_charge(&s, &sink, f, quarks);
    }
}

/// Through the sink against through the current: C3(τ) is the two-point function with S_u
/// replaced by D_u⁻¹ (e^{+i q·(y − x0)} Γ_J S_u on time slice τ).
#[test]
fn sequential_source_through_the_sink_is_that_through_the_current() {
    let s = setup();
    let (g, g5) = (Dirac::GAMMA, Dirac::GAMMA5);
    // A pion-like sink and source that differ, so a swap of the two would show.
    let (sink_gamma, source_gamma): (Gamma, Gamma) = (g5, g[3].mul(&g5));
    let sink = MesonSink::<Dirac> {
        sink: sink_gamma,
        source: source_gamma,
        gammas: PhantomData,
    };
    let (p_sink, q, current) = ([1, 0, 0], [0, 1, 1], g[0]);
    let props = [&s.props[0], &s.props[1]];
    let eta = sequential_source::<Dirac, _>(&sink, props, T_SINK, p_sink, SOURCE);
    let sigma = solve(&s.ops[0], &eta);
    let c3 = current_insertion::<Dirac>(&s.props[0], &sigma, &current, q, SOURCE);

    let lat = s.props[0].lattice().clone();
    for tau in [SOURCE[3], 3, T_SINK - 1] {
        // e^{+i q·(y − x0)} Γ_J S_u(y) on time slice τ.
        let mut b = s.props[0].zeros_like();
        for site in 0..lat.num_sites() {
            let y = lat.coords(site);
            if y[3] != tau {
                continue;
            }
            let angle: f64 = (0..3)
                .map(|mu| {
                    2.0 * std::f64::consts::PI * q[mu] as f64 / lat.global[mu] as f64
                        * (y[mu] as f64 - SOURCE[mu] as f64)
                })
                .sum();
            let phase = C64::new(angle.cos(), angle.sin());
            let from = s.props[0].site(0, site).to_vec();
            let to = b.site_mut(0, site);
            for k in 0..12 {
                for sp in 0..4 {
                    for c in 0..3 {
                        let i = k * 24 + 2 * (c + 3 * current.col[sp]);
                        let v = phase * current.val[sp].times(C64::new(from[i], from[i + 1]));
                        let o = k * 24 + 2 * (c + 3 * sp);
                        to[o] = v.re;
                        to[o + 1] = v.im;
                    }
                }
            }
        }
        let psi = solve(&s.ops[0], &b);
        let through_current = meson_correlator_with_momentum(
            &psi,
            &s.props[1],
            &sink_gamma,
            &source_gamma,
            &g5,
            p_sink,
            SOURCE,
        )[T_SINK];
        let d = (c3[tau] - through_current).norm_sqr().sqrt();
        let size = through_current.norm_sqr().sqrt();
        eprintln!(
            "tau {tau}: through the sink {:?}, through the current {through_current:?}",
            c3[tau]
        );
        assert!(
            size > 0.0 && d <= 1e-9 * size,
            "tau {tau}: differ by {d:e} of {size:e}"
        );
    }
}
