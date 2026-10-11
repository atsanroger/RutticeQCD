//! The multigrid for the even-odd Pauli-Villars operator of Möbius domain-wall fermions
//! (`PvEvenOdd`; docs/MDWF_MULTIGRID_LOG.md, section 0, "factorized coarse correction").
//!
//! With S the Schur complement of D on the even sites, S_PV that of D_PV, ι the injection
//! of the even sites into the whole lattice and π its adjoint, S⁻¹ = π D⁻¹ ι and
//! S_PV†⁻¹ = π D_PV†⁻¹ ι, so
//!
//! ```text
//!   Ã⁻¹   = C_e S⁻¹ S_PV†⁻¹ C_PV,e†                    (right form)
//!   Ã_L⁻¹ = Q_e⁻¹ S⁻¹ (C_e^(α) C_PV,e^(α)†) S_PV†⁻¹ O_R  (left form, Neff's α)
//! ```
//!
//! (C^(α) = C Q: C_e^(α) C_PV,e^(α)† = C_e Q_e^k C_PV,e†, k = 2 with α in D_PV too, and then
//! O_R = Q_e⁻¹; k = 1 and O_R = 1 with α in D alone). The coarse correction replaces
//! D⁻¹ and D_PV†⁻¹ by their Galerkin inverses P (P†DP)⁻¹ P† and P (P†D_PV†P)⁻¹ P†:
//!
//! ```text
//!   e = O_L π P (P†DP)⁻¹ P† ι M π P (P†D_PV†P)⁻¹ P† ι O_R r,
//! ```
//!
//! with O_L = C_e, M = 1, O_R = C_PV,e† for the right form. P and the two coarse factors
//! are those of the plain operators: one setup serves every α and both forms.
//!
//! P is built from vectors y on the even sites (approximate low modes of Ã): for each,
//! s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y), whose image D s lives on the even sites, and
//! t̂ = (π D s, −(D_PV†)_oo⁻¹ (D_PV†)_oe π D s), whose image under D_PV† does; with these
//! the correction is exact on the vectors it is built from. The aggregates keep the whole
//! fifth dimension and are split by the chirality of Γ5 = γ5 R; P is stored folded and in
//! single precision.
//!
//! The coarse solves: the quark factor by BiCGStab with the exact inverses of the diagonal
//! blocks as right preconditioner; the Pauli-Villars factor, held as P†D_PV P and applied
//! as its Γ5-adjoint, by MINRES on H = Γ5 (P†D_PV P)† with the block |H_xx|⁻¹. The cycle:
//! y = S b, y += e(b − Ã y), y += S(b − Ã y), S a Chebyshev polynomial in the operator.

use super::{Aggregation, Basis, BasisBuilder, CoarseOp, MultigridError, Reflection};
use crate::fermion::{
    Dagger, EoForm, EvenOdd, FermionOp, HopTerms, Mdwf, MdwfHalfWork, PvEvenOdd, Variant,
};
use crate::kernel::{Dirac, GammaRepre, Number, Real};
use crate::numerics::{hermitian_eigen, invert_complex};
use crate::solver::{largest_magnitude, BiCgStab, Chebyshev, Minres, Solver, Stop};
use crate::{
    extract_parity, insert_parity, random_spinor, Coarse, ConvertFrom, EvenOddIndex, Field, Half,
    Lattice, LinearFn, LinearOp, Parity, Spinor, Storage, Vector, C64,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

type HalfField = Field<f64, Spinor, Half>;
type CoarseField = Field<f64, Coarse>;

/// P: double-precision fields, single-precision storage, folded on the fifth dimension.
pub type PvEoBasis = Basis<f64, f32>;

/// The prolongation from vectors `y` on the even sites (see the top of the file): 2 n
/// test vectors for n vectors in `y`, on the aggregates `agg`.
pub fn pv_eo_basis<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    agg: &Arc<Aggregation>,
    y: &HalfField,
) -> Result<PvEoBasis, MultigridError> {
    let index = op
        .even_odd_index()
        .expect("an even-odd decomposition")
        .clone();
    let n = y.num_vectors();
    let lat = agg.fine_lattice().clone();
    let ls = op.params().ls;
    let full = || Field::<f64, Spinor>::with_shape(&lat, 24, ls, 1, 1);
    let half = |parity: Parity| {
        let mut h = HalfField::with_shape(&lat, 24, ls, 1, 1);
        h.set_parity(parity);
        h
    };
    let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
    let mut full_work = op.work();
    let mut builder = BasisBuilder::<f64, f32>::new(agg, 2 * n).folded(Reflection(G::GAMMA5));
    let mut g5 = full();
    let mut add = |i: usize, v: &Field<f64, Spinor>, work: &mut _| {
        op.mult_gamma5(work, &mut g5, v);
        builder.add(i, v, &g5);
    };
    for j in 0..n {
        let yj = y.select(&[j]);
        // s = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y)
        let mut x_e = half(Parity::Even);
        op.diag_inv(&mut work, &mut x_e, &yj, Parity::Even, Dagger::No);
        let mut t = half(Parity::Odd);
        op.hop(&mut work, &mut t, &x_e, Parity::Odd, Dagger::No);
        let mut x_o = half(Parity::Odd);
        op.diag_inv(&mut work, &mut x_o, &t, Parity::Odd, Dagger::No);
        x_o.scale(&[-1.0]);
        let mut s = full();
        insert_parity(&mut s, &x_e, &index);
        insert_parity(&mut s, &x_o, &index);
        // t̂ = (π D s, −(D_PV†)_oo⁻¹ (D_PV†)_oe π D s)
        let mut ds = full();
        op.mult(&mut full_work, &mut ds, &s, Variant::D);
        add(j, &s, &mut full_work);
        drop(s);
        let mut a = half(Parity::Even);
        extract_parity(&mut a, &ds, &index, Parity::Even);
        drop(ds);
        pv.hop(&mut work_pv, &mut t, &a, Parity::Odd, Dagger::Yes);
        pv.diag_inv(&mut work_pv, &mut x_o, &t, Parity::Odd, Dagger::Yes);
        x_o.scale(&[-1.0]);
        let mut t_hat = full();
        insert_parity(&mut t_hat, &a, &index);
        insert_parity(&mut t_hat, &x_o, &index);
        add(n + j, &t_hat, &mut full_work);
    }
    drop(add);
    builder.finish()
}

/// Dense matrices on the coarse sites, applied site by site.
struct BlockMap {
    dim: usize,
    /// Per coarse site, row-major.
    blocks: Vec<Vec<C64>>,
}

impl BlockMap {
    /// The exact inverses of the diagonal blocks of `a`.
    fn inverse_diagonal(a: &CoarseOp<f64>) -> Result<Self, MultigridError> {
        let dim = a.dim();
        let blocks = (0..a.lattice().num_sites())
            .map(|x| {
                let mut m: Vec<f64> = a
                    .diagonal_block(x)
                    .iter()
                    .flat_map(|z| [z.re, z.im])
                    .collect();
                let mut inv = vec![0.0; 2 * dim * dim];
                invert_complex(dim, &mut m, &mut inv)
                    .map_err(|_| MultigridError::SingularCoarseDiagonal { coarse_site: x })?;
                Ok(inv
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|z| C64::new(z[0], z[1]))
                    .collect())
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { dim, blocks })
    }

    /// The inverses of the diagonal blocks of the exact adjoint of `a`: (A_xx†)⁻¹ = (A_xx⁻¹)†.
    fn inverse_diagonal_adjoint(a: &CoarseOp<f64>) -> Result<Self, MultigridError> {
        let mut m = Self::inverse_diagonal(a)?;
        let dim = m.dim;
        for b in m.blocks.iter_mut() {
            *b = (0..dim * dim)
                .map(|ik| b[(ik % dim) * dim + ik / dim].conj())
                .collect();
        }
        Ok(m)
    }

    /// |H_xx|⁻¹ = V |Λ|⁻¹ V† for H = Γ5 (Γ5 A Γ5) = A Γ5, the operator MINRES works on
    /// when `a` holds P†D_PV P and is applied as its Γ5-adjoint (Γ5 = +1 on the first half
    /// of the components). Hermitian positive definite, as MINRES needs of a preconditioner.
    fn abs_inverse_of_a_gamma5(a: &CoarseOp<f64>) -> Self {
        let dim = a.dim();
        let blocks = (0..a.lattice().num_sites())
            .map(|x| {
                let mut h = a.diagonal_block(x);
                for row in 0..dim {
                    for col in dim / 2..dim {
                        h[row * dim + col] = -h[row * dim + col];
                    }
                }
                let (lam, v) = hermitian_eigen(dim, &h);
                let mut m = vec![C64::ZERO; dim * dim];
                for i in 0..dim {
                    for k in 0..dim {
                        let mut z = C64::ZERO;
                        for (j, l) in lam.iter().enumerate() {
                            z += v[i * dim + j].scale(1.0 / l.abs().max(1e-300))
                                * v[k * dim + j].conj();
                        }
                        m[i * dim + k] = z;
                    }
                }
                m
            })
            .collect();
        Self { dim, blocks }
    }
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for BlockMap {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        let dim = self.dim;
        let site_len = inp.site_len();
        crate::for_sites(out.slab_mut(0), site_len, |x, dst| {
            let m = &self.blocks[x];
            let src = inp.site(0, x);
            for (dst, src) in dst.chunks_exact_mut(2 * dim).zip(src.chunks_exact(2 * dim)) {
                for i in 0..dim {
                    let mut z = C64::ZERO;
                    for k in 0..dim {
                        let v =
                            C64::new(src[2 * k].load().to_f64(), src[2 * k + 1].load().to_f64());
                        z += m[i * dim + k] * v;
                    }
                    dst[2 * i] = T::store(Number::from_f64(z.re));
                    dst[2 * i + 1] = T::store(Number::from_f64(z.im));
                }
            }
        });
    }
}

/// Γ5 on a coarse field: −1 on the second half of the components.
fn coarse_gamma5<T: Storage>(v: &mut Field<T, Coarse>) {
    let dim = v.internal_dof() / 2;
    for site in v.as_mut_slice().chunks_exact_mut(2 * dim) {
        for r in &mut site[dim..] {
            *r = T::store(-r.load());
        }
    }
}

/// How a coarse factor is applied.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Apply {
    /// D_c.
    Plain,
    /// H = Γ5 (Γ5 D_c Γ5) = D_c Γ5, the Γ5-adjoint (for MINRES).
    Gamma5Adjoint,
    /// D_c† exactly.
    Adjoint,
}

/// A coarse factor as a linear map counting its applications.
struct Counted<'a, T: Storage> {
    a: &'a mut CoarseOp<T>,
    apply: Apply,
    n: &'a Cell<usize>,
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for Counted<'_, T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.n.set(self.n.get() + inp.num_vectors());
        match self.apply {
            Apply::Plain => self.a.mult(out, inp),
            Apply::Gamma5Adjoint => {
                self.a.mult_dag(out, inp);
                coarse_gamma5(out);
            }
            Apply::Adjoint => self.a.mult_adjoint(out, inp),
        }
    }
}

/// The Krylov solver of a coarse factor, right-preconditioned by the inverses of its
/// diagonal blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseKrylov {
    BiCgStab,
    /// Flexible GCR keeping the last `k` directions.
    Gcr {
        k: usize,
    },
    /// MINRES on the Γ5-Hermitian form H = Γ5 (Γ5 A Γ5) with the block |H_xx|⁻¹: only for
    /// the second (Pauli-Villars) factor, and exact only where A† = Γ5 A Γ5, which for
    /// Möbius domain-wall fermions holds approximately. The other solvers take that factor
    /// on the exact adjoint of the stored matrix.
    Minres,
}

/// x = A⁻¹ b approximately by kind, preconditioned by m; not reaching the tolerance
/// is not an error.
fn coarse_solve<T: Storage>(
    kind: CoarseKrylov,
    stop: Stop,
    a: &mut dyn LinearFn<Field<T, Coarse>>,
    m: &mut BlockMap,
    x: &mut Field<T, Coarse>,
    b: &Field<T, Coarse>,
) {
    let _ = match kind {
        CoarseKrylov::BiCgStab => BiCgStab::new(stop).solve(a, Some(m), x, b),
        CoarseKrylov::Gcr { k } => crate::solver::Gcr::new(stop, k).solve(a, Some(m), x, b),
        CoarseKrylov::Minres => Minres::new(stop).solve(a, Some(m), x, b),
    };
}

/// Settings of the cycle.
#[derive(Clone, Copy, Debug)]
pub struct PvEoParams {
    /// Steps of the Chebyshev smoother (applications of the operator: one fewer).
    pub chebyshev_steps: usize,
    /// The smoother acts on [hi/ratio, hi], hi = 1.1 |λ_max|.
    pub chebyshev_ratio: f64,
    /// The solve of the quark factor stops at |r|²/|b|² below this ...
    pub coarse_tolerance_sqr: f64,
    /// ... or after this many iterations;
    pub coarse_max_iterations: usize,
    /// that of the second factor likewise (MINRES: in its own norm).
    pub pv_tolerance_sqr: f64,
    pub pv_max_iterations: usize,
    /// The solver of the quark factor P†DP (not `Minres`) ...
    pub d_solver: CoarseKrylov,
    /// ... and that of the second factor, P†D_PV†P (P†D†P for the normal equations).
    pub pv_solver: CoarseKrylov,
    /// Steps of the power iteration for |λ_max|, from a random vector with this seed.
    pub power_steps: usize,
    pub seed: u64,
}

impl Default for PvEoParams {
    /// The settings of the 12⁴ and 16⁴ runs in the log: Chebyshev 5 steps on [hi/30, hi],
    /// coarse solves to 1e-2, the quark factor by BiCGStab, the Pauli-Villars one by MINRES.
    fn default() -> Self {
        Self {
            chebyshev_steps: 5,
            chebyshev_ratio: 30.0,
            coarse_tolerance_sqr: 1e-2,
            coarse_max_iterations: 2000,
            pv_tolerance_sqr: 1e-2,
            pv_max_iterations: 2000,
            d_solver: CoarseKrylov::BiCgStab,
            pv_solver: CoarseKrylov::Minres,
            power_steps: 50,
            seed: 5,
        }
    }
}

/// The operator of a cycle (`PvEoCycle::operator`), shared with the solver the cycle
/// preconditions, so that the two do not each hold its work fields. The solver and the
/// cycle apply it one after the other, never inside each other, which the `RefCell`
/// checks.
pub struct SharedOperator<'a, G: GammaRepre = Dirac>(Rc<RefCell<PvEvenOdd<'a, f64, G>>>);

impl<G: GammaRepre> Clone for SharedOperator<'_, G> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, G: GammaRepre> SharedOperator<'a, G> {
    pub fn new(a: PvEvenOdd<'a, f64, G>) -> Self {
        Self(Rc::new(RefCell::new(a)))
    }

    /// See `PvEvenOdd::right_hand_side`.
    pub fn right_hand_side(&self, rhs: &mut HalfField, b: &Field<f64, Spinor>) {
        self.0.borrow_mut().right_hand_side(rhs, b);
    }

    /// See `PvEvenOdd::solution`.
    pub fn solution(&self, x: &mut Field<f64, Spinor>, y: &HalfField, b: &Field<f64, Spinor>) {
        self.0.borrow_mut().solution(x, y, b);
    }
}

impl<G: GammaRepre> LinearFn<HalfField> for SharedOperator<'_, G> {
    fn mult(&mut self, out: &mut HalfField, inp: &HalfField) {
        self.0.borrow_mut().mult(out, inp);
    }
}

impl<G: GammaRepre> LinearOp<HalfField> for SharedOperator<'_, G> {
    fn mult_dag(&mut self, out: &mut HalfField, inp: &HalfField) {
        self.0.borrow_mut().mult_dag(out, inp);
    }
}

/// The multigrid cycle for `PvEvenOdd::new(op_a, pv_a, form, ..)`, as the preconditioner of
/// a flexible solver (`Gcr`, `Fgmres`, flexible `BiCgStab`) on that operator. `op` and `pv`
/// are the plain operators (α = 1) the basis and the coarse factors are built from;
/// `op_a` and `pv_a` may carry Neff's α (it matters only in the left form). The coarse
/// operators and solves are stored as `CT`: single precision by default, enough for
/// solves to 1e-2 and half the memory traffic of the bandwidth-limited coarse operator.
pub struct PvEoCycle<'a, G: GammaRepre = Dirac, CT: Storage = f32> {
    op: &'a Mdwf<f64, G>,
    pv: &'a Mdwf<f64, G>,
    op_a: &'a Mdwf<f64, G>,
    pv_a: &'a Mdwf<f64, G>,
    form: EoForm,
    index: Arc<EvenOddIndex>,
    basis: &'a PvEoBasis,
    coarse_d: CoarseOp<CT>,
    coarse_pv: CoarseOp<CT>,
    blocks_d: BlockMap,
    blocks_pv: BlockMap,
    smoother: Chebyshev<SharedOperator<'a, G>, HalfField>,
    params: PvEoParams,
    hi: f64,
    work: [MdwfHalfWork<f64>; 2],
    /// Scratch on the even sites (taken out while the cycle runs) and on the coarse lattice:
    /// `g` in the precision of P, the others in that of the coarse solves.
    r: Option<HalfField>,
    e: Option<HalfField>,
    h: HalfField,
    g: CoarseField,
    gc: Field<CT, Coarse>,
    w: Field<CT, Coarse>,
    c: Field<CT, Coarse>,
    coarse_applications: Cell<usize>,
}

/// A zero field on the coarse lattice of `basis`, stored as `T`.
fn coarse_field<T: Storage>(basis: &PvEoBasis, num_vectors: usize) -> Field<T, Coarse> {
    Field::with_shape(
        basis.aggregation().coarse_lattice(),
        2 * basis.coarse_dim(),
        1,
        num_vectors,
        1,
    )
}

impl<'a, G: GammaRepre> PvEoCycle<'a, G, f32> {
    /// The cycle with single-precision coarse solves.
    pub fn new(
        op: &'a Mdwf<f64, G>,
        pv: &'a Mdwf<f64, G>,
        outer: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>, EoForm),
        basis: &'a PvEoBasis,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        Self::with_coarse_storage(op, pv, outer, basis, params)
    }
}

impl<'a, G: GammaRepre, CT: Storage> PvEoCycle<'a, G, CT> {
    /// The cycle with the coarse solves stored as `CT`.
    pub fn with_coarse_storage(
        op: &'a Mdwf<f64, G>,
        pv: &'a Mdwf<f64, G>,
        (op_a, pv_a, form): (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>, EoForm),
        basis: &'a PvEoBasis,
        params: PvEoParams,
    ) -> Result<Self, MultigridError> {
        let index = op
            .even_odd_index()
            .expect("an even-odd decomposition")
            .clone();
        let lat = basis.aggregation().fine_lattice().clone();
        let ls = op.params().ls;
        let mut like = HalfField::with_shape(&lat, 24, ls, 1, 1);
        like.set_parity(Parity::Even);
        let coarse_d = CoarseOp::build(op, &mut op.work(), basis);
        let coarse_pv = CoarseOp::build(pv, &mut pv.work(), basis);
        let blocks_d = BlockMap::inverse_diagonal(&coarse_d)?;
        assert!(
            params.d_solver != CoarseKrylov::Minres,
            "MINRES needs a Hermitian operator: not for the quark factor"
        );
        let blocks_pv = match params.pv_solver {
            CoarseKrylov::Minres => BlockMap::abs_inverse_of_a_gamma5(&coarse_pv),
            _ => BlockMap::inverse_diagonal_adjoint(&coarse_pv)?,
        };
        let (coarse_d, coarse_pv) = (coarse_d.to_storage(), coarse_pv.to_storage());
        let mut start = like.zeros_like();
        extract_parity(
            &mut start,
            &random_spinor(&lat, ls, params.seed),
            &index,
            Parity::Even,
        );
        let hi = 1.1
            * largest_magnitude(
                &mut PvEvenOdd::new(op_a, pv_a, form, &like),
                &mut start,
                params.power_steps,
            );
        let smoother = Chebyshev::new(
            SharedOperator::new(PvEvenOdd::new(op_a, pv_a, form, &like)),
            params.chebyshev_steps,
            hi / params.chebyshev_ratio,
            hi,
            &like,
        );
        Ok(Self {
            op,
            pv,
            op_a,
            pv_a,
            form,
            index,
            basis,
            g: basis.coarse_field(1),
            gc: coarse_field(basis, 1),
            w: coarse_field(basis, 1),
            c: coarse_field(basis, 1),
            coarse_d,
            coarse_pv,
            blocks_d,
            blocks_pv,
            smoother,
            params,
            hi,
            work: [op.even_odd_work(), op_a.even_odd_work()],
            r: Some(like.zeros_like()),
            e: Some(like.zeros_like()),
            h: like,
            coarse_applications: Cell::new(0),
        })
    }

    /// The operator the cycle approximates the inverse of, to be used by the solver it
    /// preconditions (one copy of its work fields for both).
    pub fn operator(&mut self) -> SharedOperator<'a, G> {
        self.smoother.operator().clone()
    }

    /// 1.1 |λ_max| of the operator, the top of the smoother's interval.
    pub fn top(&self) -> f64 {
        self.hi
    }

    /// Coarse-operator applications so far (both factors, the preconditioners' included).
    pub fn coarse_applications(&self) -> usize {
        self.coarse_applications.get()
    }

    /// e = the coarse correction of r (see the top of the file).
    fn coarse_correction(&mut self, e: &mut HalfField, r: &HalfField) {
        let nv = r.num_vectors();
        if self.g.num_vectors() != nv {
            self.g = self.basis.coarse_field(nv);
        }
        for f in [&mut self.gc, &mut self.w, &mut self.c] {
            if f.num_vectors() != nv {
                *f = coarse_field(self.basis, nv);
            }
        }
        crate::fit(&mut self.h, r);
        let [work, work_a] = &mut self.work;
        // h = O_R r, g = P† ι h.
        match self.form {
            EoForm::Right => self
                .pv
                .diag(work, &mut self.h, r, Parity::Even, Dagger::Yes),
            EoForm::Left => self.pv_a.mult_q(&mut self.h, r, true),
        }
        self.basis.restrict_half(&mut self.g, &self.h, &self.index);
        self.gc.convert_from(&self.g);
        let stop = Stop::simple(
            self.params.coarse_tolerance_sqr,
            self.params.coarse_max_iterations,
        );
        let pv_stop = Stop::simple(self.params.pv_tolerance_sqr, self.params.pv_max_iterations);
        let n = &self.coarse_applications;
        // w = (P†D_PV†P)⁻¹ g: on the exact adjoint, or by MINRES on
        // H = Γ5 (Γ5 P†D_PV P Γ5) with H w = Γ5 g.
        self.w.set_zero();
        let apply = if self.params.pv_solver == CoarseKrylov::Minres {
            coarse_gamma5(&mut self.gc);
            Apply::Gamma5Adjoint
        } else {
            Apply::Adjoint
        };
        coarse_solve(
            self.params.pv_solver,
            pv_stop,
            &mut Counted {
                a: &mut self.coarse_pv,
                apply,
                n,
            },
            &mut self.blocks_pv,
            &mut self.w,
            &self.gc,
        );
        // g = P† ι M π P w; M = 1 (right form) or C_e^(α) C_PV,e^(α)† (left form).
        self.g.convert_from(&self.w);
        self.basis.prolong_half(&mut self.h, &self.g, &self.index);
        if self.form == EoForm::Left {
            self.pv_a
                .diag(work_a, e, &self.h, Parity::Even, Dagger::Yes);
            self.op_a
                .diag(work_a, &mut self.h, e, Parity::Even, Dagger::No);
        }
        self.basis.restrict_half(&mut self.g, &self.h, &self.index);
        self.gc.convert_from(&self.g);
        // c = (P†DP)⁻¹ g, right-preconditioned by the block inverses.
        self.c.set_zero();
        coarse_solve(
            self.params.d_solver,
            stop,
            &mut Counted {
                a: &mut self.coarse_d,
                apply: Apply::Plain,
                n,
            },
            &mut self.blocks_d,
            &mut self.c,
            &self.gc,
        );
        // e = O_L π P c: C_e (right form) or Q_e⁻¹ (left form).
        self.g.convert_from(&self.c);
        self.basis.prolong_half(&mut self.h, &self.g, &self.index);
        match self.form {
            EoForm::Right => self.op.diag(work, e, &self.h, Parity::Even, Dagger::No),
            EoForm::Left => self.op_a.mult_q(e, &self.h, true),
        }
    }
}

impl<G: GammaRepre, CT: Storage> LinearFn<HalfField> for PvEoCycle<'_, G, CT> {
    /// y = S b; y += e(b − Ã y); y += S(b − Ã y).
    fn mult(&mut self, y: &mut HalfField, b: &HalfField) {
        let nv = b.num_vectors();
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus = vec![C64::new(-1.0, 0.0); nv];
        let (mut r, mut e) = (
            self.r.take().expect("returned after every cycle"),
            self.e.take().expect("returned after every cycle"),
        );
        crate::fit(&mut r, b);
        crate::fit(&mut e, b);
        self.smoother.mult(y, b);
        self.smoother.operator().mult_xpay(&mut r, y, &minus, b);
        self.coarse_correction(&mut e, &r);
        y.axpy(&one, &e);
        self.smoother.operator().mult_xpay(&mut r, y, &minus, b);
        self.smoother.mult(&mut e, &r);
        y.axpy(&one, &e);
        self.r = Some(r);
        self.e = Some(e);
    }
}

/// Settings of the Chebyshev-filtered subspace iteration for the test vectors.
#[derive(Clone, Copy, Debug)]
pub struct LowModeParams {
    /// Vectors iterated (M).
    pub vectors: usize,
    /// Rounds of filtering and Rayleigh-Ritz.
    pub rounds: usize,
    /// Vectors filtered at a time (memory: the work fields of Ã are this large).
    pub chunk: usize,
    pub seed: u64,
}

impl Default for LowModeParams {
    /// The cheap setup of the log: M = 16, two rounds (with a bootstrap after them).
    fn default() -> Self {
        Self {
            vectors: 16,
            rounds: 2,
            chunk: 4,
            seed: 21,
        }
    }
}

/// `out = Σ_i x_i c[i n + j]` for every j, in place, site by site (no copy of x).
fn rotate(x: &mut HalfField, c: &[C64]) {
    let n = x.num_vectors();
    let site_len = x.site_len();
    let reals = site_len / n;
    crate::for_sites(x.slab_mut(0), site_len, |_, s| {
        let old: Vec<C64> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| C64::new(z[0], z[1]))
            .collect();
        for j in 0..n {
            for d in 0..reals / 2 {
                let mut z = C64::ZERO;
                for i in 0..n {
                    z += old[i * reals / 2 + d] * c[i * n + j];
                }
                s[j * reals + 2 * d] = z.re;
                s[j * reals + 2 * d + 1] = z.im;
            }
        }
    });
}

/// ⟨x_i, y_j⟩ for all i, j (row-major, x rows), one pair of vectors at a time.
fn gram(x: &HalfField, y: &HalfField) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let mut g = vec![C64::ZERO; nx * ny];
    let mut d = [C64::ZERO];
    for j in 0..ny {
        let yj = y.select(&[j]);
        for i in 0..nx {
            x.select(&[i]).dot(&yj, &mut d);
            g[i * ny + j] = d[0];
        }
    }
    g
}

/// Eigenvalues ascending, with the eigenvectors (columns) in the same order.
fn sorted_eigen(n: usize, a: &[C64]) -> (Vec<f64>, Vec<C64>) {
    let (values, v) = hermitian_eigen(n, a);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|i, j| values[*i].total_cmp(&values[*j]));
    let mut vectors = vec![C64::ZERO; n * n];
    for (j, o) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + j] = v[i * n + o];
        }
    }
    (order.iter().map(|o| values[*o]).collect(), vectors)
}

/// Orthonormalise the vectors of x through their Gram matrix (Löwdin, x ← x G^-1/2 V).
fn orthonormalise_all(x: &mut HalfField) {
    let n = x.num_vectors();
    let (values, mut c) = sorted_eigen(n, &gram(x, x));
    for (j, l) in values.iter().enumerate() {
        assert!(*l > 1e-13 * values[n - 1], "the subspace lost rank");
        for i in 0..n {
            c[i * n + j] = c[i * n + j].scale(1.0 / l.sqrt());
        }
    }
    rotate(x, &c);
}

/// Approximate low modes of Ã (the right singular vectors of its smallest singular
/// values, eigenvectors of Ã†Ã) for the basis: Chebyshev-filtered subspace iteration with
/// Rayleigh-Ritz on `params.vectors` random vectors on the even sites, the degree following
/// √(hi/lo) (at least 40), lo the largest Ritz value of the round before (hi/1000 at
/// first). Returns the Ritz vectors, lowest first, and the Ritz values of Ã†Ã. One round
/// costs 2 × degree applications of Ã per vector, plus two for the Rayleigh-Ritz step.
pub fn pv_eo_low_modes<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> (HalfField, Vec<f64>) {
    let index = op
        .even_odd_index()
        .expect("an even-odd decomposition")
        .clone();
    let (ls, m) = (op.params().ls, params.vectors);
    let mut x = HalfField::with_shape(lat, 24, ls, m, 1);
    x.set_parity(Parity::Even);
    extract_parity(
        &mut x,
        &random_spinor(lat, ls * m, params.seed),
        &index,
        Parity::Even,
    );
    let like = x.select(&[0]);
    let normal = |c: &HalfField| crate::Normal::new(PvEvenOdd::new(op, pv, EoForm::Right, c), c);
    let mut start = like.zeros_like();
    start.copy_from(&x.select(&[0]));
    let hi = 1.1 * largest_magnitude(&mut normal(&like), &mut start, 50);
    let mut lo = hi / 1000.0;
    let mut theta = vec![0.0; m];
    let chunks: Vec<Vec<usize>> = (0..m)
        .collect::<Vec<_>>()
        .chunks(params.chunk.max(1))
        .map(<[usize]>::to_vec)
        .collect();
    for _ in 0..params.rounds {
        let degree = ((1.5 * (hi / lo).sqrt()).ceil() as usize).clamp(40, 1000);
        for c in &chunks {
            let xc = x.select(c);
            let mut y = xc.zeros_like();
            crate::solver::ChebyshevFilter::new(normal(&xc), degree, lo, hi, &xc).mult(&mut y, &xc);
            x.assign(c, &y);
        }
        orthonormalise_all(&mut x);
        // G = X† Ã†Ã X, a chunk of columns at a time.
        let mut g = vec![C64::ZERO; m * m];
        for c in &chunks {
            let xc = x.select(c);
            let mut w = xc.zeros_like();
            normal(&xc).mult(&mut w, &xc);
            let part = gram(&x, &w);
            for i in 0..m {
                for (k, j) in c.iter().enumerate() {
                    g[i * m + j] = part[i * c.len() + k];
                }
            }
        }
        let (values, v) = sorted_eigen(m, &g);
        rotate(&mut x, &v);
        theta = values;
        lo = theta[m - 1];
    }
    (x, theta)
}

/// Settings of the multigrid-accelerated improvement of the test vectors.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapParams {
    pub rounds: usize,
    /// Steps of FGMRES on Ã x = y with the cycle, per vector and round.
    pub steps: usize,
    /// Settings of the cycle used (its coarse tolerance, smoother).
    pub cycle: PvEoParams,
}

impl Default for BootstrapParams {
    /// Two rounds of four steps (the log: as good as two more filter rounds, cheaper).
    fn default() -> Self {
        Self {
            rounds: 2,
            steps: 4,
            cycle: PvEoParams::default(),
        }
    }
}

/// Improve the test vectors y with the multigrid built on them: every round, the basis and
/// the cycle (right form) from the current y, then y_j ← (`steps` steps of FGMRES on
/// Ã x = y_j preconditioned by the cycle), one vector at a time, and y orthonormalised.
/// Inverse subspace iteration with the multigrid as the approximate inverse. Returns the
/// basis built from the final y.
pub fn pv_eo_bootstrap<G: GammaRepre>(
    op: &Mdwf<f64, G>,
    pv: &Mdwf<f64, G>,
    agg: &Arc<Aggregation>,
    y: &mut HalfField,
    params: BootstrapParams,
) -> Result<PvEoBasis, MultigridError> {
    let like = y.select(&[0]);
    for _ in 0..params.rounds {
        let basis = pv_eo_basis(op, pv, agg, y)?;
        let mut cycle = PvEoCycle::new(op, pv, (op, pv, EoForm::Right), &basis, params.cycle)?;
        let mut a = PvEvenOdd::new(op, pv, EoForm::Right, &like);
        let mut fgmres = crate::solver::Fgmres::new(Stop {
            tolerance_sqr: 1e-30,
            max_iterations: params.steps,
            max_restarts: 0,
        });
        for j in 0..y.num_vectors() {
            let yj = y.select(&[j]);
            let mut x = yj.zeros_like();
            let _ = fgmres.solve(&mut a, Some(&mut cycle), &mut x, &yj);
            y.assign(&[j], &x);
        }
        orthonormalise_all(y);
    }
    pv_eo_basis(op, pv, agg, y)
}
