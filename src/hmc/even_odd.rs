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

use super::fermion_action::{CloverInsertion, ForceOp, Gamma5};
use super::solve::{cg_after_multigrid, chebyshev_cg, Pre};
use super::{Chrono, LinkField, Links, MdAction, Mseries};
use super::{NormalSolver, Preconditioner};
use crate::fermion::{AfterQ, Dagger, EoForm, EvenOdd, OnParity, Schur, SchurFactor};
use crate::gauge::GaugeSmearing;
use crate::kernel::Cplx;
use crate::solver::{SolveError, SolveInfo, Stop};
use crate::{
    extract_parity, for_sites, insert_parity, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn,
    LinearOp, Normal, Parity, Spinor, Vector, C64,
};
use std::rc::Rc;
use std::sync::Arc;

type Psi = Field<f64, Spinor>;
/// A spinor field on the sites of one parity.
pub type HalfPsi = Field<f64, Spinor, Half>;

/// What the even-odd actions need of an operator besides its blocks, on the host or on a
/// device. `Half` is its kind of field on the sites of one parity: as with `ForceOp::Field`,
/// the action reaches its fields only through this trait and `Vector`, so an operator in another
/// layout or on another backend brings its own.
pub trait EvenOddHalves: ForceOp + EvenOdd<Self::Half> {
    type Half: Vector;
    /// A zero field on the sites of `parity` of the shape the operator acts on.
    fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> Self::Half;
    /// `ForceOp::right` on the sites of one parity.
    fn right_half(&self, out: &mut Self::Half, inp: &Self::Half);
    /// `ForceOp::gamma5` on the sites of one parity (Γ5 acts within a site).
    fn gamma5_half(&self, out: &mut Self::Half, inp: &Self::Half);
    /// `half` = the sites of `parity` of `whole`.
    fn half_of(&self, half: &mut Self::Half, whole: &Self::Field, parity: Parity);
    /// The sites of the parity of `half` in `whole` = `half`.
    fn put_half(&self, whole: &mut Self::Field, half: &Self::Half);
    /// (K†K)⁻¹ b on the even sites (K the left `SchurFactor`) with the domain-wall multigrid
    /// `setup` over the Pauli-Villars operator `pv`, from the guess in `z`, for an operator that
    /// has it (as `ForceOp::domain_wall_multigrid` on the whole lattice); None otherwise.
    #[allow(clippy::too_many_arguments)]
    fn domain_wall_multigrid_half(
        &self,
        _pv: &Self,
        _setup: &mut super::DwfMultigridSetup,
        _flags: (bool, bool),
        _z: &mut Self::Half,
        _b: &Self::Half,
        _stop: Stop,
    ) -> Option<Result<SolveInfo, SolveError>> {
        None
    }
}

/// What the even-odd two-flavour action needs besides: the clover term and the multigrid of
/// Wilson and clover fermions, which the operators of the usual layout on the CPU have
/// (`EvenOddSites`); the others panic, refused when the parameter file is read.
pub trait EvenOddForceOp: EvenOddHalves {
    /// Σ over the sites x of `parity` of ln |det T(x)|, T the site-diagonal block of the
    /// operator; summed over all processes. Only the clover term makes T depend on the links.
    fn log_det_diag(&self, _lat: &Arc<Lattice>, _parity: Parity) -> f64 {
        panic!("ln det T for an operator with a clover term")
    }
    /// The clover term's part (coefficient κ c_SW) of the M of the even-odd action on the links
    /// `v`: that of −2 Re y† δD x, and that of the derivative of −2 ln |det T| on the sites
    /// of `parity`.
    fn clover_derivative_eo(
        &self,
        _kappa_csw: f64,
        _v: &Self::Links,
        _parity: Parity,
        _y: &Self::Field,
        _x: &Self::Field,
    ) -> Self::Links {
        panic!("the clover term of an operator without one")
    }
    /// A map on the even sites from one on all sites: the even sites of `inner` applied to a
    /// field that is zero on the odd ones (for the multigrid of the whole operator).
    fn on_even<'a>(
        &'a self,
        _inner: &'a mut dyn LinearFn<Self::Field>,
        _lat: &Arc<Lattice>,
    ) -> Box<dyn LinearFn<Self::Half> + 'a> {
        panic!("the multigrid of the whole operator is on the CPU")
    }
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
    fn gamma5_on_half(&self, out: &mut HalfPsi, inp: &HalfPsi);
    /// `EvenOddHalves::domain_wall_multigrid_half`.
    #[allow(clippy::too_many_arguments)]
    fn multigrid_on_half(
        &self,
        _pv: &Self,
        _setup: &mut super::DwfMultigridSetup,
        _flags: (bool, bool),
        _z: &mut HalfPsi,
        _b: &HalfPsi,
        _stop: Stop,
    ) -> Option<Result<SolveInfo, SolveError>> {
        None
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
    fn gamma5_on_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        // (γ5 ψ)_s = v_s ψ_(col s), colour by colour.
        let (g5, from) = (G::GAMMA5, inp.slab(0));
        for_sites(out.slab_mut(0), 24, |x, o| {
            let i = &from[24 * x..24 * (x + 1)];
            for spin in 0..4 {
                let (col, v) = (g5.col[spin], g5.val[spin]);
                for c in 0..3 {
                    let j = 2 * (c + 3 * col);
                    let w = v.times(Cplx::new(i[j], i[j + 1]));
                    o[2 * (c + 3 * spin)] = w.re;
                    o[2 * (c + 3 * spin) + 1] = w.im;
                }
            }
        });
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
    fn gamma5_on_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        self.mult_gamma5_on(out, inp);
    }
    fn multigrid_on_half(
        &self,
        pv: &Self,
        setup: &mut super::DwfMultigridSetup,
        flags: (bool, bool),
        z: &mut HalfPsi,
        b: &HalfPsi,
        stop: Stop,
    ) -> Option<Result<SolveInfo, SolveError>> {
        Some(super::dwf_multigrid::normal_solve_half(
            self, pv, setup, flags, z, b, stop,
        ))
    }
}

impl<O: EvenOddSites> EvenOddHalves for O {
    type Half = HalfPsi;
    fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> HalfPsi {
        self.half(lat, parity)
    }
    fn right_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        self.right_on_half(out, inp);
    }
    fn gamma5_half(&self, out: &mut HalfPsi, inp: &HalfPsi) {
        self.gamma5_on_half(out, inp);
    }
    fn half_of(&self, half: &mut HalfPsi, whole: &Psi, parity: Parity) {
        extract_parity(half, whole, self.eo_index(), parity);
    }
    fn put_half(&self, whole: &mut Psi, half: &HalfPsi) {
        insert_parity(whole, half, self.eo_index());
    }
    fn domain_wall_multigrid_half(
        &self,
        pv: &Self,
        setup: &mut super::DwfMultigridSetup,
        flags: (bool, bool),
        z: &mut HalfPsi,
        b: &HalfPsi,
        stop: Stop,
    ) -> Option<Result<SolveInfo, SolveError>> {
        self.multigrid_on_half(pv, setup, flags, z, b, stop)
    }
}

impl<O: EvenOddSites> EvenOddForceOp for O {
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
pub(crate) fn log_det_diag<O: EvenOddSites>(op: &O, lat: &Arc<Lattice>, parity: Parity) -> f64 {
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
    pub(crate) stop_md: Stop,
    pub(crate) stop_h: Stop,
    /// How (Ŝ†Ŝ)⁻¹ is computed (CG by default).
    pub solver: NormalSolver,
    /// Chebyshev with CG; the multigrid of the whole operator with the two-step solvers: its
    /// even sites approximate Ŝ⁻¹, the block of D⁻¹ there.
    pub preconditioner: Preconditioner,
    conditioned: Option<O>,
    fresh: bool,
    phi: Option<O::Half>,
    /// CG iterations of the last solve, and of all force solves so far.
    pub(crate) last_iterations: usize,
    pub md_iterations: usize,
    /// κ c_SW when the operator has a clover term (`with_clover`).
    clover: Option<f64>,
    /// The initial guess of the force solves (none by default).
    pub(crate) guess: Chrono<O::Half>,
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

    pub(crate) fn with_solver(mut self, solver: NormalSolver) -> Self {
        self.solver = solver;
        self
    }

    /// The preconditioner of the solves; it must suit the solver (`Preconditioner`).
    pub(crate) fn with_preconditioner(mut self, preconditioner: Preconditioner) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    /// Solve with `op`, which applies D R for a Hermitian R within a site
    /// (`TwoFlavor::with_conditioned`): its Schur complement is Ŝ R.
    pub(crate) fn with_conditioned(mut self, op: O) -> Self {
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

    fn set_links(&mut self, links: &Links<O::Links>) -> Rc<Vec<O::Links>> {
        let levels = links.levels(self.smearing.as_ref());
        let v = levels.last().unwrap_or(links.thin);
        self.op.set_links(v).expect("the links suit the operator");
        if let Some(a) = self.conditioned.as_mut() {
            a.set_links(v).expect("the links suit the operator");
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
            Preconditioner::DomainWallMultigrid(_) => {
                panic!("the domain-wall multigrid is for the Ratio action on all sites")
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
    pub fn langevin(&mut self, links: &Links<O::Links>, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let lat = LinkField::lattice(links.thin);
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
    pub fn hamiltonian(&mut self, links: &Links<O::Links>) -> f64 {
        self.set_links(links);
        let lat = LinkField::lattice(links.thin);
        let v = self
            .solve(lat, self.stop_h, false)
            .expect("the solve for the energy converges");
        let mut d = [C64::ZERO];
        v.dot(self.phi.as_ref().expect("drawn"), &mut d);
        d[0].re + self.det_term(lat)
    }

    pub fn force(&mut self, links: &Links<O::Links>) -> O::Links {
        let levels = self.set_links(links);
        let lat = LinkField::lattice(links.thin);
        let eta = self
            .solve(lat, self.stop_md, true)
            .expect("the solve for the force converges");
        let like_odd = self.op.new_half(lat, Parity::Odd);
        let mut zeta = eta.zeros_like();
        Schur::new(&self.op, Parity::Even, &eta, &like_odd).mult(&mut zeta, &eta);
        let (y, x) = lift_to_whole(&self.op, lat, &zeta, &eta);
        let mut m = self.op.derivative(&y, &x);
        if let Some(kc) = self.clover {
            let v = levels.last().unwrap_or(links.thin);
            let c = self.op.clover_derivative_eo(kc, v, Parity::Odd, &y, &x);
            m.add_scaled(1.0, &c);
        }
        O::Links::smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

impl<O: EvenOddForceOp> MdAction<O::Links> for TwoFlavorEo<O> {
    fn langevin(&mut self, links: &Links<O::Links>, rng: &mut Mseries) -> f64 {
        TwoFlavorEo::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links<O::Links>) -> f64 {
        TwoFlavorEo::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links<O::Links>) -> O::Links {
        TwoFlavorEo::force(self, links)
    }
}

/// The fields of the whole lattice whose −2 Re Y† δD X is −2 Re y_e† δŜ x_e (module
/// documentation): X = (x_e, −T_oo⁻¹ M_oe x_e), Y = (y_e, −T_oo⁻† (M_eo)† y_e); returns (Y, X).
fn lift_to_whole<O: EvenOddHalves>(
    op: &O,
    lat: &Arc<Lattice>,
    y_e: &O::Half,
    x_e: &O::Half,
) -> (O::Field, O::Field) {
    let odd = Parity::Odd;
    let like_odd = op.new_half(lat, odd);
    let mut work = op.even_odd_work();
    let (mut t, mut x_o, mut y_o) = (
        like_odd.zeros_like(),
        like_odd.zeros_like(),
        like_odd.zeros_like(),
    );
    op.hop(&mut work, &mut t, x_e, odd, Dagger::No);
    op.diag_inv(&mut work, &mut x_o, &t, odd, Dagger::No);
    x_o.scale(&[-1.0]);
    op.hop(&mut work, &mut t, y_e, odd, Dagger::Yes);
    op.diag_inv(&mut work, &mut y_o, &t, odd, Dagger::Yes);
    y_o.scale(&[-1.0]);
    let (mut x, mut y) = (op.new_field(lat), op.new_field(lat));
    op.put_half(&mut x, x_e);
    op.put_half(&mut x, &x_o);
    op.put_half(&mut y, y_e);
    op.put_half(&mut y, &y_o);
    (y, x)
}

/// Two flavours over two Pauli-Villars flavours with even-odd preconditioning, as Bridge++'s
/// `Action_F_alt_Ratio_eo` writes it: with K = T_ee⁻¹ Ŝ = 1 − T_ee⁻¹ M_eo T_oo⁻¹ M_oe (the left
/// form of `SchurFactor`, Bridge++'s even-odd operator) and H = Γ5 K,
///
///   S = φ_e† H_PV (H†H)⁻¹ H_PV† φ_e,   φ_e = H_PV (H_PV†H_PV)⁻¹ H† ξ_e,
///
/// so that S = |ξ_e|² at the start. det D = det T_ee det T_oo det K, and T does not depend on
/// the links for the operators this takes (Wilson and domain-wall fermions; not clover).
///
/// The force: with χ = Γ5 φ_e, v1 = K_PV† χ and v2 = (K†K)⁻¹ v1,
/// δS = 2 Re χ† δK_PV v2 − 2 Re (K v2)† δK v2, and y† δK x = (T_ee⁻† y)† δŜ x.
pub struct TwoFlavorRatioEo<O: EvenOddHalves> {
    pub op: O,
    pub pv: O,
    pub smearing: Option<GaugeSmearing>,
    pub(crate) stop_md: Stop,
    pub(crate) stop_h: Stop,
    /// How (K†K)⁻¹ is computed (CG by default); the solve with the denominator in the heat
    /// bath uses it with no preconditioner.
    pub solver: NormalSolver,
    /// None, Chebyshev, or the domain-wall multigrid.
    pub preconditioner: Preconditioner,
    fresh: bool,
    phi: Option<O::Half>,
    /// Iterations of the last solve with the operator, and of all its force solves so far.
    pub(crate) last_iterations: usize,
    pub md_iterations: usize,
    /// The initial guess of the force solves (none by default).
    pub(crate) guess: Chrono<O::Half>,
    /// Mixed precision: K in single precision (`with_single`).
    single: Option<Box<dyn super::SingleNormal<O::Half, O::Links>>>,
    /// Neff's Q on the even sites (`with_neff`).
    neff: Option<NeffHalfQ<O::Half>>,
    meter: super::solve::CostMeter,
}

/// Neff's Q on the fields of one parity: `q(out, inp, inverse)`.
pub(crate) type NeffHalfQ<H> = Box<dyn Fn(&mut H, &H, bool)>;

impl<O: EvenOddHalves> TwoFlavorRatioEo<O> {
    pub fn new(op: O, pv: O, smearing: Option<GaugeSmearing>, stop_md: Stop, stop_h: Stop) -> Self {
        Self {
            op,
            pv,
            smearing,
            stop_md,
            stop_h,
            solver: NormalSolver::Cg,
            preconditioner: Preconditioner::None,
            fresh: true,
            phi: None,
            last_iterations: 0,
            md_iterations: 0,
            guess: Chrono::new(0),
            single: None,
            neff: None,
            meter: Default::default(),
        }
    }

    pub(crate) fn with_solver(mut self, solver: NormalSolver) -> Self {
        self.solver = solver;
        self
    }

    /// The solves with the operator by Neff's α, Q of D^(α) = D Q on the even sites: CG on the
    /// normal equations of K Q_e (`fermion::AfterQ`), z = Q_e ((K Q_e)†(K Q_e))⁻¹ Q_e b; the
    /// action and its force are those of K. Mixed precision (`with_single`) solves the same
    /// equations in single precision. Not with a preconditioner.
    pub(crate) fn with_neff(mut self, q: NeffHalfQ<O::Half>) -> Self {
        self.neff = Some(q);
        self
    }

    /// Mixed precision in the solves with the operator: CG on K†K in single precision with
    /// `single`, corrected in double; with CG and no preconditioner.
    pub(crate) fn with_single(
        mut self,
        single: Box<dyn super::SingleNormal<O::Half, O::Links>>,
    ) -> Self {
        self.single = Some(single);
        self
    }

    /// None, Chebyshev, or the domain-wall multigrid (`EvenOddHalves::domain_wall_multigrid_half`);
    /// the multigrid of Wilson fermions is not written for K.
    pub(crate) fn with_preconditioner(mut self, preconditioner: Preconditioner) -> Self {
        assert!(
            !matches!(preconditioner, Preconditioner::Multigrid(_)),
            "the even-odd ratio action has no Wilson multigrid"
        );
        self.preconditioner = preconditioner;
        self
    }

    /// Start the force solves from the best combination of the last `depth` solutions
    /// (`Chrono`); 0: from zero.
    pub fn with_guess(mut self, depth: usize) -> Self {
        self.guess = Chrono::new(depth);
        self
    }

    fn set_links(&mut self, links: &Links<O::Links>) -> Rc<Vec<O::Links>> {
        let levels = links.levels(self.smearing.as_ref());
        let v = levels.last().unwrap_or(links.thin);
        self.op.set_links(v).expect("the links suit the operator");
        self.pv.set_links(v).expect("the links suit the operator");
        if let Some(s) = self.single.as_mut() {
            s.set_links(v);
        }
        levels
    }

    /// K v or K† v for the operator `op`.
    fn factor(op: &O, lat: &Arc<Lattice>, v: &O::Half, dagger: Dagger) -> O::Half {
        let odd = op.new_half(lat, Parity::Odd);
        let mut k = SchurFactor::new(op, EoForm::Left, v, &odd);
        let mut out = v.zeros_like();
        match dagger {
            Dagger::No => k.mult(&mut out, v),
            Dagger::Yes => k.mult_dag(&mut out, v),
        }
        out
    }

    fn gamma5(op: &O, v: &O::Half) -> O::Half {
        let mut out = v.zeros_like();
        op.gamma5_half(&mut out, v);
        out
    }

    /// (K†K)⁻¹ b for the operator `op` (over `pv` for the domain-wall multigrid), from `z`.
    /// `fresh`: the first solve since the pseudofermions were drawn; `md`: a force solve.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        solver: NormalSolver,
        (op, pv): (&O, Option<&O>),
        lat: &Arc<Lattice>,
        (pre, fresh, md): (&mut Preconditioner, bool, bool),
        z: &mut O::Half,
        b: &O::Half,
        stop: Stop,
    ) -> Result<SolveInfo, SolveError> {
        let odd = op.new_half(lat, Parity::Odd);
        let mut k = SchurFactor::new(op, EoForm::Left, b, &odd);
        match pre {
            Preconditioner::None => solver.solve(&mut k, Pre::None, z, b, stop),
            Preconditioner::Chebyshev { steps, ratio, hi } => chebyshev_cg(
                &mut k,
                SchurFactor::new(op, EoForm::Left, b, &odd),
                (*steps, *ratio, hi),
                fresh,
                z,
                b,
                stop,
            ),
            Preconditioner::DomainWallMultigrid(setup) => {
                let result = op
                    .domain_wall_multigrid_half(
                        pv.expect("the multigrid of the ratio over Pauli-Villars fields"),
                        setup,
                        (fresh, md),
                        z,
                        b,
                        stop,
                    )
                    .expect("an operator with the domain-wall multigrid");
                cg_after_multigrid(&mut k, result, z, b, stop)
            }
            Preconditioner::Multigrid(_) => unreachable!("refused by with_preconditioner"),
        }
    }

    /// (K†K)⁻¹ v1 for the operator of the action; for the force (`md`) from the guess.
    fn solve_op(&mut self, lat: &Arc<Lattice>, v1: &O::Half, stop: Stop, md: bool) -> O::Half {
        let mut z = v1.zeros_like();
        if md {
            let odd = self.op.new_half(lat, Parity::Odd);
            let mut k = SchurFactor::new(&self.op, EoForm::Left, v1, &odd);
            self.guess.guess(&mut Normal::new(&mut k, v1), v1, &mut z);
        }
        let flags = (
            &mut self.preconditioner,
            std::mem::take(&mut self.fresh),
            md,
        );
        let ops = (&self.op, Some(&self.pv));
        let start = std::time::Instant::now();
        let info = match (self.neff.as_ref(), self.single.as_mut()) {
            (Some(q), single) => {
                assert!(
                    self.solver == NormalSolver::Cg && matches!(flags.0, Preconditioner::None),
                    "Neff's α is for CG without a preconditioner"
                );
                // z = Q_e y, (K Q_e)†(K Q_e) y = Q_e b.
                let (mut b, mut y) = (v1.zeros_like(), z.zeros_like());
                q(&mut b, v1, false);
                q(&mut y, &z, true);
                let odd = self.op.new_half(lat, Parity::Odd);
                let mut k = SchurFactor::new(&self.op, EoForm::Left, v1, &odd);
                let mut a = AfterQ::new(&mut k, q.as_ref(), v1);
                let info = match single {
                    Some(single) => super::mixed_normal(&mut a, single.as_mut(), &mut y, &b, stop),
                    None => self.solver.solve(&mut a, Pre::None, &mut y, &b, stop),
                };
                q(&mut z, &y, false);
                info
            }
            (None, Some(single)) => {
                assert!(
                    self.solver == NormalSolver::Cg && matches!(flags.0, Preconditioner::None),
                    "mixed precision is for CG without a preconditioner"
                );
                let odd = self.op.new_half(lat, Parity::Odd);
                let mut k = SchurFactor::new(&self.op, EoForm::Left, v1, &odd);
                super::mixed_normal(&mut k, single.as_mut(), &mut z, v1, stop)
            }
            (None, None) => Self::solve(self.solver, ops, lat, flags, &mut z, v1, stop),
        }
        .expect("the solve with the operator converges");
        self.meter.add(start, info.iterations);
        self.last_iterations = info.iterations;
        if md {
            self.md_iterations += info.iterations;
            self.guess.record(&z);
        }
        z
    }

    /// v1 = H_PV† φ_e = K_PV† Γ5 φ_e.
    fn v1(&self, lat: &Arc<Lattice>) -> O::Half {
        let phi = self.phi.as_ref().expect("pseudofermions drawn");
        Self::factor(&self.pv, lat, &Self::gamma5(&self.pv, phi), Dagger::Yes)
    }

    /// The heat bath: ξ Gaussian (variance ½ per real component) drawn on the whole lattice
    /// in Bridge++'s order, φ_e = H_PV (H_PV†H_PV)⁻¹ H† ξ_e; returns |ξ_e|².
    pub fn langevin(&mut self, links: &Links<O::Links>, rng: &mut Mseries) -> f64 {
        self.set_links(links);
        let lat = LinkField::lattice(links.thin);
        let mut xi = self.op.new_field(lat);
        self.op.draw(&mut xi, rng);
        let mut xi_e = self.op.new_half(lat, Parity::Even);
        self.op.half_of(&mut xi_e, &xi, Parity::Even);
        let v2 = Self::factor(&self.op, lat, &Self::gamma5(&self.op, &xi_e), Dagger::Yes);
        let mut v1 = v2.zeros_like();
        let none = (&mut Preconditioner::None, false, false);
        let ops = (&self.pv, None);
        // CG for the denominator, whatever the operator's solver: the two-step solvers do
        // not converge on the domain-wall operator itself.
        let solver = match self.preconditioner {
            Preconditioner::DomainWallMultigrid(_) => NormalSolver::Cg,
            _ => self.solver,
        };
        let start = std::time::Instant::now();
        let info = Self::solve(solver, ops, lat, none, &mut v1, &v2, self.stop_h)
            .expect("the solve with the denominator converges");
        self.meter.add(start, info.iterations);
        let k1 = Self::factor(&self.pv, lat, &v1, Dagger::No);
        self.phi = Some(Self::gamma5(&self.pv, &k1));
        self.fresh = true;
        self.guess.clear();
        let mut n = [0.0];
        xi_e.norm_sqr(&mut n);
        n[0]
    }

    /// S = v1† (K†K)⁻¹ v1.
    pub fn hamiltonian(&mut self, links: &Links<O::Links>) -> f64 {
        self.set_links(links);
        let lat = LinkField::lattice(links.thin);
        let v1 = self.v1(lat);
        let v2 = self.solve_op(lat, &v1, self.stop_h, false);
        let mut d = [C64::ZERO];
        v1.dot(&v2, &mut d);
        d[0].re
    }

    pub fn force(&mut self, links: &Links<O::Links>) -> O::Links {
        let levels = self.set_links(links);
        let lat = LinkField::lattice(links.thin);
        let v1 = self.v1(lat);
        let v2 = self.solve_op(lat, &v1, self.stop_md, true);
        // y_e = T_ee⁻† (K v2) for the operator, T_PV,ee⁻† χ for the denominator.
        let even_inv_dag = |op: &O, v: &O::Half| {
            let mut out = v.zeros_like();
            op.diag_inv(
                &mut op.even_odd_work(),
                &mut out,
                v,
                Parity::Even,
                Dagger::Yes,
            );
            out
        };
        let y = even_inv_dag(&self.op, &Self::factor(&self.op, lat, &v2, Dagger::No));
        let chi = Self::gamma5(&self.pv, self.phi.as_ref().expect("drawn"));
        let y_pv = even_inv_dag(&self.pv, &chi);
        let (y, x) = lift_to_whole(&self.op, lat, &y, &v2);
        let mut m = self.op.derivative(&y, &x);
        let (y_pv, x_pv) = lift_to_whole(&self.pv, lat, &y_pv, &v2);
        m.add_scaled(-1.0, &self.pv.derivative(&y_pv, &x_pv));
        O::Links::smeared_force(&m, links.thin, self.smearing.as_ref(), &levels)
    }
}

impl<O: EvenOddHalves> MdAction<O::Links> for TwoFlavorRatioEo<O> {
    fn langevin(&mut self, links: &Links<O::Links>, rng: &mut Mseries) -> f64 {
        TwoFlavorRatioEo::langevin(self, links, rng)
    }
    fn hamiltonian(&mut self, links: &Links<O::Links>) -> f64 {
        TwoFlavorRatioEo::hamiltonian(self, links)
    }
    fn force(&mut self, links: &Links<O::Links>) -> O::Links {
        TwoFlavorRatioEo::force(self, links)
    }
    fn take_cost(&mut self) -> Option<super::SolveCost> {
        Some(self.meter.take(&self.preconditioner))
    }
}
