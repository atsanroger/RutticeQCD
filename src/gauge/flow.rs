//! The gradient flow of the gauge field (Lüscher's Wilson flow) and what is measured along it:
//! the energy density and the topological charge. The conventions are those of Bridge++
//! (`GradientFlow`, `EnergyDensity`, `TopologicalCharge`, `FieldStrength`).

use super::paths::{Extended, Step};
use super::smearing::traceless_antihermitian;
use super::{plaquette, PLANES};
use crate::hmc::{exp_update, PlaquetteAction};
use crate::kernel::ColorMatrix;
use crate::{reduce_sites, Boundary, Field, Gauge, Stencil, Vector, ND};

/// The Wilson flow ∂_t V = Z(V) V, Z the force of the plaquette action at β = N_c (Bridge++
/// fixes β so in `GradientFlow`), integrated by Lüscher's third-order Runge-Kutta scheme; the
/// exponentials are Taylor series to order `exp_order` (Bridge++'s
/// `order_of_approx_for_exp_iP`) and the links are made unitary again after each.
#[derive(Clone, Copy, Debug)]
pub struct WilsonFlow {
    /// ε, the flow time of one step.
    pub step: f64,
    pub exp_order: usize,
}

impl WilsonFlow {
    /// `u` from flow time t to t + ε: W1 = exp(¼ Z0) W0, W2 = exp(8/9 Z1 − 17/36 Z0) W1,
    /// W3 = exp(¾ Z2 − 8/9 Z1 + 17/36 Z0) W2, Z_i = ε Z(W_i).
    pub fn step(&self, u: &mut Field<f64, Gauge>) {
        let force = |v: &Field<f64, Gauge>| PlaquetteAction { beta: 3.0 }.force(v);
        let combine = |terms: &[(f64, &Field<f64, Gauge>)]| {
            let mut out = terms[0].1.zeros_like();
            for (c, z) in terms {
                for (o, v) in out.as_mut_slice().iter_mut().zip(z.as_slice()) {
                    *o += c * v;
                }
            }
            out
        };
        let (n, eps) = (self.exp_order, self.step);
        let z0 = force(u);
        exp_update(eps, &combine(&[(0.25, &z0)]), u, n);
        let z1 = force(u);
        exp_update(
            eps,
            &combine(&[(8.0 / 9.0, &z1), (-17.0 / 36.0, &z0)]),
            u,
            n,
        );
        let z2 = force(u);
        let last = combine(&[(0.75, &z2), (-8.0 / 9.0, &z1), (17.0 / 36.0, &z0)]);
        exp_update(eps, &last, u, n);
    }
}

/// E = 36 (1 − ⟨plaquette⟩), the energy density from the plaquette (Bridge++ `E_plaq`).
pub fn energy_density_plaq(u: &Field<f64, Gauge>) -> f64 {
    let st = Stencil::nearest(u.lattice(), [Boundary::Periodic; ND]);
    36.0 * (1.0 - plaquette(u, &st))
}

/// The loops through every site in the μν plane that make a clover: the four plaquettes
/// (`rect` false), or the eight rectangles of 2 × 1 and 1 × 2 sites (`rect` true), all
/// starting and ending at the site and turning the same way.
fn clover_loops(mu: usize, nu: usize, rect: bool) -> Vec<Vec<Step>> {
    let (f, b) = (true, false);
    // The four quarters of the plane, each with its first two directions and how far the
    // loop goes in each: from the site along `a` then `c`, back along `a` then `c`.
    let quarters = [
        ((mu, f), (nu, f)),
        ((nu, f), (mu, b)),
        ((mu, b), (nu, b)),
        ((nu, b), (mu, f)),
    ];
    let sizes: &[(usize, usize)] = if rect { &[(2, 1), (1, 2)] } else { &[(1, 1)] };
    let mut loops = Vec::new();
    for ((a, a_fwd), (c, c_fwd)) in quarters {
        for &(along_mu, along_nu) in sizes {
            // The extent along `a` and along `c`: a is μ in the first and third quarter.
            let (la, lc) = if a == mu {
                (along_mu, along_nu)
            } else {
                (along_nu, along_mu)
            };
            let mut path = Vec::new();
            path.extend(std::iter::repeat_n((a, a_fwd), la));
            path.extend(std::iter::repeat_n((c, c_fwd), lc));
            path.extend(std::iter::repeat_n((a, !a_fwd), la));
            path.extend(std::iter::repeat_n((c, !c_fwd), lc));
            loops.push(path);
        }
    }
    loops
}

/// The loops of the clovers of the six planes (in the order of `PLANES`).
fn all_clover_loops(rect: bool) -> Vec<Vec<Vec<Step>>> {
    PLANES
        .iter()
        .map(|&(mu, nu)| clover_loops(mu, nu, rect))
        .collect()
}

/// The clover field strength of the six planes at one site, as Bridge++'s
/// `construct_Fmunu_1x1_traceless` (or `_1x2_traceless` with the rectangles):
/// F = −i k TA(Σ loops), k = ¼ for the plaquettes and ⅛ for the rectangles, kept as the
/// anti-Hermitian A = −k TA(Σ loops), F = i A. `ext` reaches as far as the loops go.
fn site_clovers(
    ext: &Extended,
    loops: &[Vec<Vec<Step>>],
    x: usize,
    k: f64,
) -> [ColorMatrix<f64>; 6] {
    let e = ext.at(x);
    core::array::from_fn(|plane| {
        let sum = loops[plane]
            .iter()
            .map(|path| ext.walk(e, path))
            .reduce(|s, m| s.add(&m))
            .expect("a clover has loops");
        traceless_antihermitian(&sum).scale(-k)
    })
}

/// Σ_x f(the field strength of the six planes at x) over all processes, for the plaquette
/// clover (`rect` false) or the rectangle one.
fn sum_over_sites(
    u: &Field<f64, Gauge>,
    rect: bool,
    f: impl Fn(&[ColorMatrix<f64>; 6]) -> f64 + Sync,
) -> f64 {
    let (depth, k) = if rect { (2, 0.125) } else { (1, 0.25) };
    let ext = Extended::new(u, depth);
    let loops = all_clover_loops(rect);
    let local = reduce_sites(u.num_sites(), |x| f(&site_clovers(&ext, &loops, x, k)));
    let mut sum = [local];
    u.lattice().comm.sum(&mut sum);
    sum[0]
}

/// Re tr(F_a F_b) of two field strengths kept as F = i A: −Re tr(A_a A_b).
fn re_tr(a: &ColorMatrix<f64>, b: &ColorMatrix<f64>) -> f64 {
    -a.mul(b).trace().re
}

/// E from the clover field strength, c_plaq E_1x1 + c_rect E_1x2 with E = Σ_μ<ν Re tr F_μν²
/// averaged over the sites (Bridge++ `E_clover`; the 1 × 2 part counted twice as there).
pub fn energy_density_clover(u: &Field<f64, Gauge>, c_plaq: f64, c_rect: f64) -> f64 {
    let sites = u.lattice().num_global_sites() as f64;
    let part =
        |rect: bool| sum_over_sites(u, rect, |f| f.iter().map(|a| re_tr(a, a)).sum()) / sites;
    let mut e = c_plaq * part(false);
    if c_rect != 0.0 {
        e += c_rect * 2.0 * part(true);
    }
    e
}

/// The topological charge Q = 1/(32 π²) Σ_x ε_μνρσ tr F_μν F_ρσ of the clover field strength,
/// c_plaq Q_1x1 + c_rect Q_1x2 (Bridge++ `TopologicalCharge::measure`; the 1 × 2 part counted
/// twice as there). c_plaq = 1, c_rect = 0 is the clover charge; 5/3, −1/12 the improved one.
pub fn topological_charge(u: &Field<f64, Gauge>, c_plaq: f64, c_rect: f64) -> f64 {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    // The planes (01)(23), (02)(13), (03)(12): indices 0 5, 1 4, 2 3 of PLANES.
    let part = |rect: bool| {
        let q = sum_over_sites(u, rect, |f| {
            re_tr(&f[0], &f[5]) - re_tr(&f[1], &f[4]) + re_tr(&f[2], &f[3])
        });
        8.0 * q / (32.0 * pi2)
    };
    let mut q = c_plaq * part(false);
    if c_rect != 0.0 {
        q += c_rect * 2.0 * part(true);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lattice;

    /// The clovers are made of loops that close.
    #[test]
    fn clover_loops_close() {
        for rect in [false, true] {
            for path in clover_loops(1, 3, rect) {
                let mut pos = [0i64; ND];
                for (mu, f) in &path {
                    pos[*mu] += if *f { 1 } else { -1 };
                }
                assert_eq!(pos, [0; ND], "{path:?}");
            }
        }
    }

    /// The unit field has no field strength, and the flow leaves it alone.
    #[test]
    fn the_unit_field_stays() {
        let lat = Lattice::single_node([4, 4, 4, 4]);
        let mut u = crate::hmc::cold_start(&lat);
        assert!(energy_density_clover(&u, 5.0 / 3.0, -1.0 / 12.0).abs() < 1e-14);
        assert!(topological_charge(&u, 1.0, 0.0).abs() < 1e-14);
        WilsonFlow {
            step: 0.01,
            exp_order: 8,
        }
        .step(&mut u);
        assert!(energy_density_plaq(&u).abs() < 1e-13);
    }
}
