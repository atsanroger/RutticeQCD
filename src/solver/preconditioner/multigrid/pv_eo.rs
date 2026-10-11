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
use crate::kernel::{Dirac, GammaRepre};
use crate::numerics::{hermitian_eigen, invert_complex};
use crate::solver::{largest_magnitude, BiCgStab, Chebyshev, Minres, Solver, Stop};
use crate::{
    extract_parity, insert_parity, random_spinor, Coarse, EvenOddIndex, Field, Half, LinearFn,
    LinearOp, Parity, Spinor, Vector, C64,
};
use std::cell::Cell;
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

impl LinearFn<CoarseField> for BlockMap {
    fn mult(&mut self, out: &mut CoarseField, inp: &CoarseField) {
        let dim = self.dim;
        for (x, m) in self.blocks.iter().enumerate() {
            let src = inp.site(0, x).to_vec();
            let dst = out.site_mut(0, x);
            for (dst, src) in dst.chunks_exact_mut(2 * dim).zip(src.chunks_exact(2 * dim)) {
                for i in 0..dim {
                    let mut z = C64::ZERO;
                    for k in 0..dim {
                        z += m[i * dim + k] * C64::new(src[2 * k], src[2 * k + 1]);
                    }
                    dst[2 * i] = z.re;
                    dst[2 * i + 1] = z.im;
                }
            }
        }
    }
}

/// Γ5 on a coarse field: −1 on the second half of the components.
fn coarse_gamma5(v: &mut CoarseField) {
    let dim = v.internal_dof() / 2;
    for site in v.as_mut_slice().chunks_exact_mut(2 * dim) {
        for r in &mut site[dim..] {
            *r = -*r;
        }
    }
}

/// A coarse factor as a linear map counting its applications: D_c, or H = Γ5 D_c† for
/// the Pauli-Villars factor (`adjoint`).
struct Counted<'a> {
    a: &'a mut CoarseOp<f64>,
    adjoint: bool,
    n: &'a Cell<usize>,
}

impl LinearFn<CoarseField> for Counted<'_> {
    fn mult(&mut self, out: &mut CoarseField, inp: &CoarseField) {
        self.n.set(self.n.get() + inp.num_vectors());
        if self.adjoint {
            self.a.mult_dag(out, inp);
            coarse_gamma5(out);
        } else {
            self.a.mult(out, inp);
        }
    }
}

/// Settings of the cycle.
#[derive(Clone, Copy, Debug)]
pub struct PvEoParams {
    /// Steps of the Chebyshev smoother (applications of the operator: one fewer).
    pub chebyshev_steps: usize,
    /// The smoother acts on [hi/ratio, hi], hi = 1.1 |λ_max|.
    pub chebyshev_ratio: f64,
    /// The coarse solves stop at |r|²/|b|² below this (MINRES: in its own norm).
    pub coarse_tolerance_sqr: f64,
    /// Steps of the power iteration for |λ_max|, from a random vector with this seed.
    pub power_steps: usize,
    pub seed: u64,
}

impl Default for PvEoParams {
    /// The settings of the 12⁴ and 16⁴ runs in the log: Chebyshev 5 steps on [hi/30, hi],
    /// coarse solves to 1e-2.
    fn default() -> Self {
        Self {
            chebyshev_steps: 5,
            chebyshev_ratio: 30.0,
            coarse_tolerance_sqr: 1e-2,
            power_steps: 50,
            seed: 5,
        }
    }
}

/// The multigrid cycle for `PvEvenOdd::new(op_a, pv_a, form, ..)`, as the preconditioner of
/// a flexible solver (`Gcr`, `Fgmres`, flexible `BiCgStab`) on that operator. `op` and `pv`
/// are the plain operators (α = 1) the basis and the coarse factors are built from;
/// `op_a` and `pv_a` may carry Neff's α (it matters only in the left form).
pub struct PvEoCycle<'a, G: GammaRepre = Dirac> {
    op: &'a Mdwf<f64, G>,
    pv: &'a Mdwf<f64, G>,
    op_a: &'a Mdwf<f64, G>,
    pv_a: &'a Mdwf<f64, G>,
    form: EoForm,
    index: Arc<EvenOddIndex>,
    basis: &'a PvEoBasis,
    coarse_d: CoarseOp<f64>,
    coarse_pv: CoarseOp<f64>,
    blocks_d: BlockMap,
    blocks_pv: BlockMap,
    smoother: Chebyshev<PvEvenOdd<'a, f64, G>, HalfField>,
    params: PvEoParams,
    hi: f64,
    work: [MdwfHalfWork<f64>; 2],
    /// Scratch on the even sites (taken out while the cycle runs) and on the coarse lattice.
    r: Option<HalfField>,
    e: Option<HalfField>,
    h: HalfField,
    g: CoarseField,
    w: CoarseField,
    c: CoarseField,
    coarse_applications: Cell<usize>,
}

impl<'a, G: GammaRepre> PvEoCycle<'a, G> {
    pub fn new(
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
        let blocks_pv = BlockMap::abs_inverse_of_a_gamma5(&coarse_pv);
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
            PvEvenOdd::new(op_a, pv_a, form, &like),
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
            w: basis.coarse_field(1),
            c: basis.coarse_field(1),
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
        for f in [&mut self.g, &mut self.w, &mut self.c] {
            if f.num_vectors() != nv {
                *f = self.basis.coarse_field(nv);
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
        let stop = Stop::simple(self.params.coarse_tolerance_sqr, 5000);
        let n = &self.coarse_applications;
        // w = (P†D_PV†P)⁻¹ g: MINRES on H = Γ5 (P†D_PV P)† with H w = Γ5 g.
        coarse_gamma5(&mut self.g);
        self.w.set_zero();
        let _ = Minres::new(stop).solve(
            &mut Counted {
                a: &mut self.coarse_pv,
                adjoint: true,
                n,
            },
            Some(&mut self.blocks_pv),
            &mut self.w,
            &self.g,
        );
        // g = P† ι M π P w; M = 1 (right form) or C_e^(α) C_PV,e^(α)† (left form).
        self.basis.prolong_half(&mut self.h, &self.w, &self.index);
        if self.form == EoForm::Left {
            self.pv_a
                .diag(work_a, e, &self.h, Parity::Even, Dagger::Yes);
            self.op_a
                .diag(work_a, &mut self.h, e, Parity::Even, Dagger::No);
        }
        self.basis.restrict_half(&mut self.g, &self.h, &self.index);
        // c = (P†DP)⁻¹ g: BiCGStab right-preconditioned by the block inverses.
        self.c.set_zero();
        let _ = BiCgStab::new(stop).solve(
            &mut Counted {
                a: &mut self.coarse_d,
                adjoint: false,
                n,
            },
            Some(&mut self.blocks_d),
            &mut self.c,
            &self.g,
        );
        // e = O_L π P c: C_e (right form) or Q_e⁻¹ (left form).
        self.basis.prolong_half(&mut self.h, &self.c, &self.index);
        match self.form {
            EoForm::Right => self.op.diag(work, e, &self.h, Parity::Even, Dagger::No),
            EoForm::Left => self.op_a.mult_q(e, &self.h, true),
        }
    }
}

impl<G: GammaRepre> LinearFn<HalfField> for PvEoCycle<'_, G> {
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
