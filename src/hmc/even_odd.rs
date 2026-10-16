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

use super::fermion_action::{smeared_force, CloverInsertion, ForceOp, Gamma5};
use super::solve::{chebyshev_cg, Pre};
use super::{Chrono, Links, MdAction, Mseries};
use super::{NormalSolver, Preconditioner};
use crate::fermion::{Dagger, EvenOdd, OnParity, Schur};
use crate::gauge::GaugeSmearing;
use crate::kernel::Cplx;
use crate::solver::{SolveError, Stop};
use crate::{
    extract_parity, insert_parity, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn, LinearOp,
    Normal, Parity, Spinor, Vector, C64,
};
use std::rc::Rc;
use std::sync::Arc;

type Psi = Field<f64, Spinor>;
/// A spinor field on the sites of one parity.
pub type HalfPsi = Field<f64, Spinor, Half>;

/// What the even-odd actions need of an operator besides its blocks. `Half` is its kind of
/// field on the sites of one parity: as with `ForceOp::Field`, the action reaches its fields
/// only through this trait and `Vector`, so an operator in another layout or on another backend
/// brings its own.
pub trait EvenOddForceOp: ForceOp<Links = Field<f64, Gauge>> + EvenOdd<Self::Half> {
    type Half: Vector;
    /// A zero field on the sites of `parity` of the shape the operator acts on.
    fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> Self::Half;
    /// `ForceOp::right` on the sites of one parity.
    fn right_half(&self, out: &mut Self::Half, inp: &Self::Half);
    /// `half` = the sites of `parity` of `whole`.
    fn half_of(&self, half: &mut Self::Half, whole: &Self::Field, parity: Parity);
    /// The sites of the parity of `half` in `whole` = `half`.
    fn put_half(&self, whole: &mut Self::Field, half: &Self::Half);
    /// Σ over the sites x of `parity` of ln |det T(x)|, T the site-diagonal block of the
    /// operator; summed over all processes.
    fn log_det_diag(&self, lat: &Arc<Lattice>, parity: Parity) -> f64;
    /// The clover term's part (coefficient κ c_SW) of the M of the even-odd action on the links
    /// `v`: that of −2 Re y† δD x, and that of the derivative of −2 ln |det T| on the sites
    /// of `parity`.
    fn clover_derivative_eo(
        &self,
        kappa_csw: f64,
        v: &Field<f64, Gauge>,
        parity: Parity,
        y: &Self::Field,
        x: &Self::Field,
    ) -> Field<f64, Gauge>;
    /// A map on the even sites from one on all sites: the even sites of `inner` applied to a
    /// field that is zero on the odd ones.
    fn on_even<'a>(
        &'a self,
        inner: &'a mut dyn LinearFn<Self::Field>,
        lat: &Arc<Lattice>,
    ) -> Box<dyn LinearFn<Self::Half> + 'a>;
}

/// The operators of the usual layout: what `EvenOddForceOp` is made of for them.
pub trait EvenOddSites: ForceOp<Field = Psi, Links = Field<f64, Gauge>> + EvenOdd<HalfPsi> {
    /// The numbering of the sites of each parity the operator uses.
    fn eo_index(&self) -> &EvenOddIndex;
    fn half(&self, lat: &Arc<Lattice>, parity: Parity) -> HalfPsi {
        HalfPsi::spinor_half(lat, 1, parity)
    }
    fn right_on_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        out.copy_from(inp);
    }
}

impl<G, D> EvenOddSites for crate::fermion::WilsonOp<f64, G, D>
where
    G: crate::kernel::GammaRepre + 'static,
    D: crate::fermion::DiagTerm<f64> + 'static,
{
    fn eo_index(&self) -> &EvenOddIndex {
        self.even_odd_index()
            .expect("even-odd preconditioning needs even extents")
    }
}

impl<G: crate::kernel::GammaRepre> EvenOddSites for crate::fermion::Mdwf<f64, G> {
    fn eo_index(&self) -> &EvenOddIndex {
        self.even_odd_index()
            .expect("even-odd preconditioning needs even extents")
    }
    fn half(&self, lat: &Arc<Lattice>, parity: Parity) -> HalfPsi {
        let mut f = HalfPsi::with_shape(lat, 24, self.params().ls, 1, 1);
        f.set_parity(parity);
        f
    }
    fn right_on_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        self.mult_q(out, inp, false);
    }
}

impl<O: EvenOddSites> EvenOddForceOp for O {
    type Half = HalfPsi;
    fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> HalfPsi {
        self.half(lat, parity)
    }
    fn right_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        self.right_on_half(out, inp);
    }
    fn half_of(&self, half: &mut HalfPsi, whole: &Psi, parity: Parity) {
        extract_parity(half, whole, self.eo_index(), parity);
    }
    fn put_half(&self, whole: &mut Psi, half: &HalfPsi) {
        insert_parity(whole, half, self.eo_index());
    }
    fn log_det_diag(&self, lat: &Arc<Lattice>, parity: Parity) -> f64 {
        log_det_diag(self, lat, parity)
    }
    fn clover_derivative_eo(
        &self,
        kappa_csw: f64,
        v: &Field<f64, Gauge>,
        parity: Parity,
        y: &Psi,
        x: &Psi,
    ) -> Field<f64, Gauge> {
        let lat = v.lattice();
        let mut b = CloverInsertion::new(lat);
        b.add(kappa_csw, y, x);
        add_log_det_insertion(self, lat, kappa_csw, parity, &mut b);
        b.leaves(v)
    }
    fn on_even<'a>(
        &'a self,
        inner: &'a mut dyn LinearFn<Psi>,
        lat: &Arc<Lattice>,
    ) -> Box<dyn LinearFn<HalfPsi> + 'a> {
        Box::new(OnParity::new(inner, self.eo_index(), &self.new_field(lat)))
    }
}

/// The unit vector e_k (k = colour + 3 spin) on every site of `parity`.
fn unit<O: EvenOddSites>(op: &O, lat: &Arc<Lattice>, parity: Parity, k: usize) -> HalfPsi {
    let mut e = op.half(lat, parity);
    for site in e.slab_mut(0).as_chunks_mut::<24>().0 {
        site[2 * k] = 1.0;
    }
    e
}

/// Σ over the sites x of `parity` of ln |det T(x)|, T(x) the 12 × 12 site-diagonal block of
/// the operator, from its columns T e_k; summed over all processes.
pub fn log_det_diag<O: EvenOddSites>(op: &O, lat: &Arc<Lattice>, parity: Parity) -> f64 {
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
fn add_log_det_insertion<O: EvenOddSites>(
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
pub struct TwoFlavorEo<O: EvenOddForceOp> {
    pub op: O,
    pub smearing: Option<GaugeSmearing>,
    pub stop_md: Stop,
    pub stop_h: Stop,
    /// How (Ŝ†Ŝ)⁻¹ is computed (CG by default).
    pub solver: NormalSolver,
    /// Chebyshev with CG; the multigrid of the whole operator with the two-step solvers: its
    /// even sites approximate Ŝ⁻¹, the block of D⁻¹ there.
    pub preconditioner: Preconditioner,
    conditioned: Option<O>,
    fresh: bool,
    phi: Option<O::Half>,
    /// CG iterations of the last solve, and of all force solves so far.
    pub last_iterations: usize,
    pub md_iterations: usize,
    /// κ c_SW when the operator has a clover term (`with_clover`).
    clover: Option<f64>,
    /// The initial guess of the force solves (none by default).
    pub guess: Chrono<O::Half>,
}

impl<O: EvenOddForceOp> TwoFlavorEo<O> {
    pub fn new(op: O, smearing: Option<GaugeSmearing>, stop_md: Stop, stop_h: Stop) -> Self {
        Self {
            op,
            smearing,
            stop_md,
            stop_h,
            solver: NormalSolver::Cg,
            preconditioner: Preconditioner::None,
            conditioned: None,
            fresh: true,
            phi: None,
            last_iterations: 0,
            md_iterations: 0,
            clover: None,
            guess: Chrono::new(0),
        }
    }

    pub fn with_solver(mut self, solver: NormalSolver) -> Self {
        self.solver = solver;
        self
    }

    /// The preconditioner of the solves; it must suit the solver (`Preconditioner`).
    pub fn with_preconditioner(mut self, preconditioner: Preconditioner) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    /// Solve with `op`, which applies D R for a Hermitian R within a site
    /// (`TwoFlavor::with_conditioned`): its Schur complement is Ŝ R.
    pub fn with_conditioned(mut self, op: O) -> Self {
        self.conditioned = Some(op);
        self
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
        if let Some(a) = self.conditioned.as_mut() {
            a.set_gauge(levels.last().unwrap_or(links.thin))
                .expect("the links suit the operator");
        }
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
            Some(_) => -2.0 * self.op.log_det_diag(lat, Parity::Odd),
            None => 0.0,
        }
    }

    /// (Ŝ†Ŝ)⁻¹ φ_e; for the force (`md`) from the guess.
    fn solve(&mut self, lat: &Arc<Lattice>, stop: Stop, md: bool) -> Result<O::Half, SolveError> {
        let phi = self.phi.as_ref().expect("pseudofermions drawn");
        let a = self.conditioned.as_ref().unwrap_or(&self.op);
        let odd = a.new_half(lat, Parity::Odd);
        let (mut b, mut z) = (phi.zeros_like(), phi.zeros_like());
        a.right_half(&mut b, phi);
        let mut schur = Schur::new(a, Parity::Even, phi, &odd);
        if md {
            self.guess
                .guess(&mut Normal::new(&mut schur, &b), &b, &mut z);
        }
        let fresh = std::mem::take(&mut self.fresh);
        let info = match &mut self.preconditioner {
            Preconditioner::None => self.solver.solve(&mut schur, Pre::None, &mut z, &b, stop)?,
            Preconditioner::Chebyshev { steps, ratio, hi } => chebyshev_cg(
                &mut schur,
                Schur::new(a, Parity::Even, phi, &odd),
                (*steps, *ratio, hi),
                fresh,
                &mut z,
                &b,
                stop,
            )?,
            Preconditioner::Multigrid(setup) => {
                // The even sites of the cycle approximate Ŝ⁻¹, the block of D⁻¹ there; γ5 acts
                // within a site.
                let mut cycle = a
                    .multigrid(setup, fresh, md)
                    .expect("the operator has a multigrid");
                let mut gamma5 = Gamma5 {
                    op: a,
                    work: <O as crate::fermion::FermionOp<O::Field>>::work(a),
                };
                let mut m = a.on_even(&mut *cycle, lat);
                let mut gamma = a.on_even(&mut gamma5, lat);
                let pre = Pre::Hermitian {
                    m: &mut *m,
                    gamma: &mut *gamma,
                };
                self.solver.solve(&mut schur, pre, &mut z, &b, stop)?
            }
        };
        self.last_iterations = info.iterations;
        if md {
            self.md_iterations += info.iterations;
            self.guess.record(&z);
        }
        let mut x = phi.zeros_like();
        a.right_half(&mut x, &z);
        Ok(x)
    }

    /// The heat bath: ξ Gaussian (variance ½ per real component) drawn on the whole lattice
    /// in Bridge++'s order, φ_e = Ŝ† ξ_e; returns |ξ_e|² (plus the det term), the action.
    pub fn langevin(&mut self, links: &Links, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let lat = links.thin.lattice();
        let mut xi = self.op.new_field(lat);
        self.op.draw(&mut xi, rng);
        let mut xi_e = self.op.new_half(lat, Parity::Even);
        self.op.half_of(&mut xi_e, &xi, Parity::Even);
        let mut phi = xi_e.zeros_like();
        {
            let odd = self.op.new_half(lat, Parity::Odd);
            Schur::new(&self.op, Parity::Even, &xi_e, &odd).mult_dag(&mut phi, &xi_e);
        }
        self.phi = Some(phi);
        self.fresh = true;
        self.guess.clear();
        let mut n = [0.0];
        xi_e.norm_sqr(&mut n);
        n[0] + self.det_term(lat)
    }

    /// S = φ_e† (Ŝ†Ŝ)⁻¹ φ_e − 2 ln |det T_oo|.
    pub fn hamiltonian(&mut self, links: &Links) -> f64 {
        self.set_links(links);
        let v = self
            .solve(links.thin.lattice(), self.stop_h, false)
            .expect("the solve for the energy converges");
        let mut d = [C64::ZERO];
        v.dot(self.phi.as_ref().expect("drawn"), &mut d);
        d[0].re + self.det_term(links.thin.lattice())
    }

    pub fn force(&mut self, links: &Links) -> Field<f64, Gauge> {
        let levels = self.set_links(links);
        let lat = links.thin.lattice();
        let eta = self
            .solve(lat, self.stop_md, true)
            .expect("the solve for the force converges");
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
        let (mut x, mut y) = (self.op.new_field(lat), self.op.new_field(lat));
        self.op.put_half(&mut x, &eta);
        self.op.put_half(&mut x, &x_o);
        self.op.put_half(&mut y, &zeta);
        self.op.put_half(&mut y, &y_o);
        let mut m = self.op.derivative(&y, &x);
        if let Some(kc) = self.clover {
            let v = levels.last().unwrap_or(links.thin);
            let c = self.op.clover_derivative_eo(kc, v, odd, &y, &x);
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
