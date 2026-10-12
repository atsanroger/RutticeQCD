//! Two flavours with even-odd preconditioning: the pseudofermions live on the even sites and
//! the action is that of the Schur complement there,
//!
//!   S = φ_e† (Ŝ†Ŝ)⁻¹ φ_e − 2 ln |det T_oo|,   Ŝ = T_ee − M_eo T_oo⁻¹ M_oe,
//!
//! with D = [[T_ee, M_eo], [M_oe, T_oo]] split by parity (`fermion::EvenOdd`, the blocks the
//! even-odd solvers use).
//!
//! The force: with η = (Ŝ†Ŝ)⁻¹ φ_e and ζ = Ŝ η, δS = −2 Re ζ† δŜ η − 2 Re Tr T_oo⁻¹ δT_oo.
//! The first term is −2 Re Y† δD X for the vectors of the whole lattice
//!
//!   X = (η, −T_oo⁻¹ M_oe η),   Y = (ζ, −T_oo⁻† (M_eo)† ζ),

use super::fermion_action::{draw, smeared_force, CloverInsertion, ForceOp};
use super::{Chrono, Links, MdAction, Mseries};
use crate::fermion::{Dagger, EvenOdd, Schur};
use crate::gauge::GaugeSmearing;
use crate::kernel::Cplx;
use crate::solver::{Cg, SolveError, Solver, Stop};
use crate::{
    extract_parity, insert_parity, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn, LinearOp,
    Normal, Parity, Spinor, Vector, C64,
};
use std::rc::Rc;
use std::sync::Arc;

type Psi = Field<f64, Spinor>;
/// A spinor field on the sites of one parity.
pub type HalfPsi = Field<f64, Spinor, Half>;

/// What the even-odd actions need of an operator besides its blocks.
pub trait EvenOddForceOp: ForceOp + EvenOdd<HalfPsi> {
    /// The numbering of the sites of each parity the operator uses.
    fn eo_index(&self) -> &EvenOddIndex;
    /// A zero field on the sites of `parity` of the shape the operator acts on.
    fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> HalfPsi {
        HalfPsi::spinor_half(lat, 1, parity)
    }
}

impl<G: crate::kernel::GammaRepre, D: crate::fermion::DiagTerm<f64>> EvenOddForceOp
    for crate::fermion::WilsonOp<f64, G, D>
{
    fn eo_index(&self) -> &EvenOddIndex {
        self.even_odd_index()
            .expect("even-odd preconditioning needs even extents")
    }
}

impl<G: crate::kernel::GammaRepre> EvenOddForceOp for crate::fermion::Mdwf<f64, G> {
    fn eo_index(&self) -> &EvenOddIndex {
        self.even_odd_index()
            .expect("even-odd preconditioning needs even extents")
    }
    fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> HalfPsi {
        let mut f = HalfPsi::with_shape(lat, 24, self.params().ls, 1, 1);
        f.set_parity(parity);
        f
    }
}

/// The unit vector e_k (k = colour + 3 spin) on every site of `parity`.
fn unit<O: EvenOddForceOp>(op: &O, lat: &Arc<Lattice>, parity: Parity, k: usize) -> HalfPsi {
    let mut e = op.new_half(lat, parity);
    for site in e.slab_mut(0).as_chunks_mut::<24>().0 {
        site[2 * k] = 1.0;
    }
    e
}

/// Σ over the sites x of `parity` of ln |det T(x)|, T(x) the 12 × 12 site-diagonal block of
/// the operator, from its columns T e_k; summed over all processes.
pub fn log_det_diag<O: EvenOddForceOp>(op: &O, lat: &Arc<Lattice>, parity: Parity) -> f64 {
    let mut work = op.even_odd_work();
    let cols: Vec<HalfPsi> = (0..12)
        .map(|k| {
            let e = unit(op, lat, parity, k);
            let mut t = e.zeros_like();
            op.diag(&mut work, &mut t, &e, parity, Dagger::No);
            t
        })
        .collect();
    let sites = cols[0].slab(0).len() / 24;
    let mut sum = [0.0];
    for x in 0..sites {
        let mut m = [[Cplx::new(0.0, 0.0); 12]; 12];
        for (k, col) in cols.iter().enumerate() {
            let c = &col.slab(0)[24 * x..24 * (x + 1)];
            for (r, row) in m.iter_mut().enumerate() {
                row[k] = Cplx::new(c[2 * r], c[2 * r + 1]);
            }
        }
        sum[0] += log_abs_det(&mut m);
    }
    lat.comm.sum(&mut sum);
    sum[0]
}

/// ln |det m| by Gaussian elimination with partial pivoting (m is overwritten).
fn log_abs_det(m: &mut [[C64; 12]; 12]) -> f64 {
    let n = m.len();
    let mut s = 0.0;
    for col in 0..n {
        let p = (col..n)
            .max_by(|&a, &b| m[a][col].norm_sqr().total_cmp(&m[b][col].norm_sqr()))
            .expect("rows left");
        m.swap(p, col);
        let pivot = m[col][col];
        s += 0.5 * pivot.norm_sqr().ln();
        for r in col + 1..n {
            let f = m[r][col] / pivot;
            for j in col..n {
                let t = m[col][j];
                m[r][j] = m[r][j] - f * t;
            }
        }
    }
    s
}

/// Adds to `b` the clover insertion of the derivative of −2 ln |det T_p| (the sites of
/// `parity`): the pairs (T⁻¹ e_k, e_k) for the twelve unit vectors.
fn add_log_det_insertion<O: EvenOddForceOp>(
    op: &O,
    lat: &Arc<Lattice>,
    kappa_csw: f64,
    parity: Parity,
    b: &mut CloverInsertion,
) {
    let mut work = op.even_odd_work();
    let index = op.eo_index();
    let (mut zf, mut ef) = (op.new_field(lat), op.new_field(lat));
    for k in 0..12 {
        let e = unit(op, lat, parity, k);
        let mut z = e.zeros_like();
        op.diag_inv(&mut work, &mut z, &e, parity, Dagger::No);
        insert_parity(&mut zf, &z, index);
        insert_parity(&mut ef, &e, index);
        b.add(kappa_csw, &zf, &ef);
    }
}

/// Two degenerate flavours with even-odd preconditioning (module documentation), D the
/// operator `op` on the links smeared by `smearing`. (Ŝ†Ŝ)⁻¹ by CG to |r|²/|b|² < `stop`.
pub struct TwoFlavorEo<O> {
    pub op: O,
    pub smearing: Option<GaugeSmearing>,
    pub stop_md: Stop,
    pub stop_h: Stop,
    phi: Option<HalfPsi>,
    /// CG iterations of the last solve, and of all force solves so far.
    pub last_iterations: usize,
    pub md_iterations: usize,
    /// κ c_SW when the operator has a clover term (`with_clover`).
    clover: Option<f64>,
    /// The initial guess of the force solves (none by default).
    pub guess: Chrono<HalfPsi>,
}

impl<O: EvenOddForceOp> TwoFlavorEo<O> {
    pub fn new(op: O, smearing: Option<GaugeSmearing>, stop_md: Stop, stop_h: Stop) -> Self {
        Self {
            op,
            smearing,
            stop_md,
            stop_h,
            phi: None,
            last_iterations: 0,
            md_iterations: 0,
            clover: None,
            guess: Chrono::new(0),
        }
    }

    /// The operator has the clover term with coefficient κ c_SW: its force and the term
    /// −2 ln |det T_oo| are added.
    pub fn with_clover(mut self, kappa_csw: f64) -> Self {
        self.clover = Some(kappa_csw);
        self
    }

    /// Start the force solves from the best combination of the last `depth` solutions
    /// (`Chrono`); 0: from zero.
    pub fn with_guess(mut self, depth: usize) -> Self {
        self.guess = Chrono::new(depth);
        self
    }

    fn set_links(&mut self, links: &Links) -> Rc<Vec<Field<f64, Gauge>>> {
        let levels = links.levels(self.smearing.as_ref());
        self.op
            .set_gauge(levels.last().unwrap_or(links.thin))
            .expect("the links suit the operator");
        levels
    }

    /// φ_e = 0: what is left of the action is the det term (a test of it alone).
    #[cfg(test)]
    pub(super) fn zero_pseudofermions(&mut self) {
        if let Some(p) = self.phi.as_mut() {
            p.set_zero();
        }
    }

    /// −2 ln |det T_oo|, for the operator on the current links.
    fn det_term(&self, lat: &Arc<Lattice>) -> f64 {
        match self.clover {
            Some(_) => -2.0 * log_det_diag(&self.op, lat, Parity::Odd),
            None => 0.0,
        }
    }

    /// (Ŝ†Ŝ)⁻¹ φ_e; for the force (`md`) from the guess.
    fn solve(&mut self, stop: Stop, md: bool) -> Result<HalfPsi, SolveError> {
        let phi = self.phi.as_ref().expect("pseudofermions drawn");
        let odd = self.op.new_half(phi.lattice(), Parity::Odd);
        let mut schur = Schur::new(&self.op, Parity::Even, phi, &odd);
        let mut normal = Normal::new(&mut schur, phi);
        let mut x = phi.zeros_like();
        if md {
            self.guess.guess(&mut normal, phi, &mut x);
        }
        let info = Cg::new(stop).solve(&mut normal, None, &mut x, phi)?;
        self.last_iterations = info.iterations;
        if md {
            self.md_iterations += info.iterations;
            self.guess.record(&x);
        }
        Ok(x)
    }

    /// The heat bath: ξ Gaussian (variance ½ per real component) drawn on the whole lattice
    /// in Bridge++'s order, φ_e = Ŝ† ξ_e; returns |ξ_e|² (plus the det term), the action.
    pub fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let lat = links.thin.lattice();
        let mut xi = self.op.new_field(lat);
        draw(&mut xi, rng);
        let mut xi_e = self.op.new_half(lat, Parity::Even);
        extract_parity(&mut xi_e, &xi, self.op.eo_index(), Parity::Even);
        let mut phi = xi_e.zeros_like();
        {
            let odd = self.op.new_half(lat, Parity::Odd);
            Schur::new(&self.op, Parity::Even, &xi_e, &odd).mult_dag(&mut phi, &xi_e);
        }
        self.phi = Some(phi);
        self.guess.clear();
        let mut n = [0.0];
        xi_e.norm_sqr(&mut n);
        n[0] + self.det_term(lat)
    }

    /// S = φ_e† (Ŝ†Ŝ)⁻¹ φ_e − 2 ln |det T_oo|.
    pub fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.set_links(links);
        let v = self
            .solve(self.stop_h, false)
            .expect("the solve for the energy converges");
        let mut d = [C64::ZERO];
        v.dot(self.phi.as_ref().expect("drawn"), &mut d);
        d[0].re + self.det_term(links.thin.lattice())
    }

    pub fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        let levels = self.set_links(links);
        let eta = self
            .solve(self.stop_md, true)
            .expect("the solve for the force converges");
        let lat = links.thin.lattice();
        let (even, odd) = (Parity::Even, Parity::Odd);
        let like_odd = self.op.new_half(lat, odd);
        let mut zeta = eta.zeros_like();
        Schur::new(&self.op, even, &eta, &like_odd).mult(&mut zeta, &eta);
        // X_o = −T_oo⁻¹ M_oe η, Y_o = −T_oo⁻† (M_eo)† ζ.
        let mut work = self.op.even_odd_work();
        let (mut t, mut x_o, mut y_o) = (
            like_odd.zeros_like(),
            like_odd.zeros_like(),
            like_odd.zeros_like(),
        );
        self.op.hop(&mut work, &mut t, &eta, odd, Dagger::No);
        self.op.diag_inv(&mut work, &mut x_o, &t, odd, Dagger::No);
        x_o.scale(&[-1.0]);
        self.op.hop(&mut work, &mut t, &zeta, odd, Dagger::Yes);
        self.op.diag_inv(&mut work, &mut y_o, &t, odd, Dagger::Yes);
        y_o.scale(&[-1.0]);
        let index = self.op.eo_index();
        let (mut x, mut y): (Psi, Psi) = (self.op.new_field(lat), self.op.new_field(lat));
        insert_parity(&mut x, &eta, index);
        insert_parity(&mut x, &x_o, index);
        insert_parity(&mut y, &zeta, index);
        insert_parity(&mut y, &y_o, index);
        let mut m = self.op.derivative(&y, &x);
        if let Some(kc) = self.clover {
            let v = levels.last().unwrap_or(links.thin);
            let mut b = CloverInsertion::new(lat);
            b.add(kc, &y, &x);
            add_log_det_insertion(&self.op, lat, kc, odd, &mut b);
            let c = b.leaves(v);
            for (a, b) in m.as_mut_slice().iter_mut().zip(c.as_slice()) {
                *a += b;
            }
        }
        smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

impl<O: EvenOddForceOp> MdAction for TwoFlavorEo<O> {
    fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        TwoFlavorEo::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links) -> f64 {
        TwoFlavorEo::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        TwoFlavorEo::force(self, links)
    }
}
