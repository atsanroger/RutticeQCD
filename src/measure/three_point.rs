//! Connected three-point functions by a sequential source through the sink (DESIGN §7.10).
//!
//! A sink is a contraction c(x) at the sink site x, multilinear in the propagators from the
//! source. The current ū Γ_J u at y on the line of one flavour f replaces one S_f(x, 0) by
//! Σ_y S_f(x, y) Γ_J S_f(y, 0). With G(x)_{ik} = ∂c(x)/∂S_f(x)_{ik} and S_f(x, y) =
//! γ5 S_f(y, x)† γ5, the sum over x becomes one solve:
//!
//!   η(x) = e^{−i p·(x − x0)} γ5 conj(G(x)) on the sink time slice,   D_f Σ = η,
//!   C3(τ) = Σ_{y, y4 = τ} e^{+i q·(y − x0)} Tr[Γ_J S_f(y, 0) Σ(y)† γ5].
//!
//! The same three lines serve every hadron; a sink supplies only c and G.

use super::{element, slice_sum, Propagator, NUM_SOURCES};
use crate::gauge::link;
use crate::kernel::{Cplx, Gamma, GammaRepre, Unit, NC, NS, SPINOR_REALS};
use crate::{shift, Boundary, Direction, Field, Gauge, Spinor, Stencil, Storage, Vector, C64, ND};

/// A 12 × 12 complex matrix at one site: sink index `colour + 3 spin`, source vector.
pub type SiteMatrix = [[C64; NUM_SOURCES]; NUM_SOURCES];

/// The flavour a current acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavour {
    U,
    D,
}

/// A hadron operator at the sink, for three-point functions.
pub trait Sink {
    /// The two-point function of the operator, the sink projected on momentum n (Bridge++'s
    /// phase e^{+i p·(x − x0)}).
    fn two_point(
        &self,
        props: [&Field<f64, Spinor>; 2],
        n: [i64; 3],
        source_position: [usize; ND],
    ) -> Vec<C64>;
    /// Adds G_{ik} = ∂c/∂S_{ik}, the derivative of the contraction at one site by the
    /// propagator of the flavour the current acts on; `sites` are the two propagators there.
    fn derivative(&self, sites: [&[f64]; 2], g: &mut SiteMatrix);
}

/// The sequential source of `sink` on time slice `t_sink`, momentum n at the sink:
/// η(x) = e^{−i p·(x − x0)} γ5 conj(G(x)). Solving D Σ = η with the operator of the flavour the
/// current acts on gives the Σ of `current_insertion` and `conserved_charge`.
pub fn sequential_source<G: GammaRepre, K: Sink>(
    sink: &K,
    props: [&Field<f64, Spinor>; 2],
    t_sink: usize,
    n: [i64; 3],
    source_position: [usize; ND],
) -> Field<f64, Spinor> {
    let (p0, p1) = (Propagator::new(props[0]), Propagator::new(props[1]));
    let lat = p0.lattice();
    assert!(t_sink < lat.global[3], "sink time outside the lattice");
    let g5 = G::GAMMA5;
    let unit: [f64; 3] =
        std::array::from_fn(|mu| 2.0 * std::f64::consts::PI * n[mu] as f64 / lat.global[mu] as f64);
    let mut eta = props[0].zeros_like();
    for site in 0..lat.num_sites() {
        let x = lat.global_coords(site);
        if x[3] != t_sink {
            continue;
        }
        let mut g = [[Cplx::ZERO; NUM_SOURCES]; NUM_SOURCES];
        sink.derivative([p0.site(site), p1.site(site)], &mut g);
        let angle: f64 = (0..3)
            .map(|mu| unit[mu] * (x[mu] as f64 - source_position[mu] as f64))
            .sum();
        let phase = Cplx::new(angle.cos(), -angle.sin());
        let out = eta.site_mut(0, site);
        for s in 0..NS {
            for c in 0..NC {
                let from = c + NC * g5.col[s];
                for k in 0..NUM_SOURCES {
                    let z = phase * g5.val[s].times(g[from][k].conj());
                    let at = k * SPINOR_REALS + 2 * (c + NC * s);
                    out[at] = z.re;
                    out[at + 1] = z.im;
                }
            }
        }
    }
    eta
}

/// Σ_{ik} (γ5 X)_{ik} conj(Σ_{ik}) = Tr[X Σ† γ5], X given by its rows on a site.
#[inline(always)]
fn trace_with<T: Storage>(g5: &Gamma, x: &SiteMatrix, seq: &[T]) -> C64 {
    let mut sum = Cplx::ZERO;
    for s in 0..NS {
        for c in 0..NC {
            let row = &x[c + NC * g5.col[s]];
            for (k, xk) in row.iter().enumerate() {
                sum += g5.val[s].times(*xk * element(seq, s, c, k).conj());
            }
        }
    }
    sum
}

/// The local current Γ_J inserted on the line of `prop` (S_f from the source), `seq` the Σ of
/// `sequential_source`: C3(τ) = Σ_y e^{+i q·(y − x0)} Tr[Γ_J S_f(y, 0) Σ(y)† γ5], q = 2π n / L.
pub fn current_insertion<G: GammaRepre>(
    prop: &Field<f64, Spinor>,
    seq: &Field<f64, Spinor>,
    current: &Gamma,
    n: [i64; 3],
    source_position: [usize; ND],
) -> Vec<C64> {
    let (s, sigma) = (Propagator::new(prop), Propagator::new(seq));
    assert_eq!(s.num_sites(), sigma.num_sites());
    let g5 = G::GAMMA5;
    slice_sum(s.lattice(), n, source_position, |site| {
        let sy = s.site(site);
        let mut x = [[Cplx::ZERO; NUM_SOURCES]; NUM_SOURCES];
        for sp in 0..NS {
            for c in 0..NC {
                for (k, v) in x[c + NC * sp].iter_mut().enumerate() {
                    *v = current.val[sp].times(element(sy, current.col[sp], c, k));
                }
            }
        }
        trace_with(&g5, &x, sigma.site(site))
    })
}

/// The rows of (1 + sign γ) W S at one site, W a colour matrix, S a propagator site:
/// ((1 + sign γ) W S)_{(s,c),k} = (W S)_{(s,c),k} + sign γ.val[s] (W S)_{(γ.col[s], c),k}.
#[inline(always)]
fn spin_colour_rows(gamma: &Gamma, sign: Unit, w: &[[C64; NC]; NC], s: &[f64]) -> SiteMatrix {
    let mut ws = [[Cplx::ZERO; NUM_SOURCES]; NUM_SOURCES];
    for sp in 0..NS {
        for c in 0..NC {
            for (k, v) in ws[c + NC * sp].iter_mut().enumerate() {
                for (cp, wc) in w[c].iter().enumerate() {
                    *v += *wc * element(s, sp, cp, k);
                }
            }
        }
    }
    let mut out = ws;
    for sp in 0..NS {
        let (from, v) = (gamma.col[sp], sign.mul(gamma.val[sp]));
        for c in 0..NC {
            for k in 0..NUM_SOURCES {
                out[c + NC * sp][k] += v.times(ws[c + NC * from][k]);
            }
        }
    }
    out
}

/// The conserved vector current of the Wilson and clover operators D = 1 − κ H (H the hops),
/// its time component inserted on the line of `prop`, `seq` the Σ of `sequential_source`:
/// on link τ → τ + 1, summed over space,
///
///   J_4(y) = −κ [ū(y) (1 − γ4) U_4(y) u(y + 4) − ū(y + 4) (1 + γ4) U_4(y)† u(y)],
///
/// the Noether current of the vector rotation of that flavour (the clover term is on one site
/// and does not enter). `boundary` is the time boundary of the operator: the link from the
/// last time slice to the first carries its phase. The charge is conserved: C3 does not
/// depend on τ between the source and the sink, and jumps there by the number of quarks of
/// that flavour times the two-point function.
pub fn conserved_charge<G: GammaRepre>(
    prop: &Field<f64, Spinor>,
    seq: &Field<f64, Spinor>,
    gauge: &Field<f64, Gauge>,
    kappa: f64,
    boundary: Boundary,
) -> Vec<C64> {
    let (s, sigma) = (Propagator::new(prop), Propagator::new(seq));
    assert_eq!(s.num_sites(), sigma.num_sites());
    let lat = s.lattice();
    let nt = lat.global[3];
    // S(y + 4) and Σ(y + 4); the boundary phase goes with the link below.
    let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
    let (mut s_up, mut sigma_up) = (prop.zeros_like(), seq.zeros_like());
    shift(&mut s_up, prop, &st, 3, Direction::Forward);
    shift(&mut sigma_up, seq, &st, 3, Direction::Forward);
    let (s_up, sigma_up) = (Propagator::new(&s_up), Propagator::new(&sigma_up));
    let phase = match boundary {
        Boundary::Periodic => Cplx::new(1.0, 0.0),
        Boundary::Antiperiodic => Cplx::new(-1.0, 0.0),
        Boundary::Phase(re, im) => Cplx::new(re, im),
        Boundary::Dirichlet => Cplx::ZERO,
    };
    let (g4, g5) = (G::GAMMA[3], G::GAMMA5);
    slice_sum(lat, [0; 3], [0; ND], |site| {
        let mut u = link(gauge, 3, site).0;
        if lat.global_coords(site)[3] == nt - 1 {
            for row in u.iter_mut() {
                for v in row.iter_mut() {
                    *v = phase * *v;
                }
            }
        }
        let u_dag: [[C64; NC]; NC] =
            std::array::from_fn(|a| std::array::from_fn(|b| u[b][a].conj()));
        // ū(y) (1 − γ4) U u(y + 4): the current takes S at y + 4, Σ at y.
        let forward = spin_colour_rows(&g4, Unit::MinusOne, &u, s_up.site(site));
        // ū(y + 4) (1 + γ4) U† u(y): S at y, Σ at y + 4.
        let backward = spin_colour_rows(&g4, Unit::One, &u_dag, s.site(site));
        let t1 = trace_with(&g5, &forward, sigma.site(site));
        let t2 = trace_with(&g5, &backward, sigma_up.site(site));
        (t1 - t2).scale(-kappa)
    })
}
