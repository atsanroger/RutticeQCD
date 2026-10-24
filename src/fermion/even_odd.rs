//! Even-odd preconditioning, written once for every fermion.
//!
//! Split the operator by parity, M = [[M_ee, M_eo], [M_oe, M_oo]]. The diagonal blocks act
//! within a site; the off-diagonal blocks are hops. A fermion supplies the blocks through
//! `EvenOdd`; the Schur complement and the solve around it are generic.

use crate::{
    extract_parity, insert_parity, Cpu, EvenOddIndex, Field, Half, Kind, Layout, LinearFn,
    LinearOp, Parity, Storage, Vector, C64,
};

/// Whether a block of M or of M† is meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dagger {
    No,
    Yes,
}

/// The blocks of an operator in the even-odd decomposition. `H` is a field on one parity.
/// With `Dagger::Yes` every method applies the corresponding block of M†.
pub trait EvenOdd<H> {
    type Work;
    fn even_odd_work(&self) -> Self::Work;
    /// `out = M_{to, other} inp`: a hop from the other parity onto `to`.
    fn hop(&self, work: &mut Self::Work, out: &mut H, inp: &H, to: Parity, dagger: Dagger);
    /// `out = M_{pp} inp`.
    fn diag(&self, work: &mut Self::Work, out: &mut H, inp: &H, parity: Parity, dagger: Dagger);
    /// `out = M_{pp}^{-1} inp`.
    fn diag_inv(&self, work: &mut Self::Work, out: &mut H, inp: &H, parity: Parity, dagger: Dagger);

    /// `out = [from −] [M_{to,to}⁻¹] M_{to, other} inp`: the hop and then what `then` asks of its
    /// result. An operator with a kernel that does it in one pass over the fields provides
    /// this; otherwise it is made of the blocks, through `scratch` (a field on `to`).
    #[allow(clippy::too_many_arguments)]
    fn hop_then(
        &self,
        work: &mut Self::Work,
        out: &mut H,
        inp: &H,
        to: Parity,
        dagger: Dagger,
        then: Then<'_, H>,
        scratch: &mut H,
    ) where
        H: Vector,
    {
        hop_then_of_blocks(self, work, out, inp, to, dagger, then, scratch);
    }

    /// `out = inp − M_ee⁻¹ M_eo M_oo⁻¹ M_oe inp` on the even sites (or with the adjoint hops and
    /// blocks): the part of a Schur factor that goes through the odd sites, `o1`, `o2` scratch
    /// on the odd sites and `e` on the even ones. By default two `hop_then`
    /// (`inverse_hops_twice_of`); an operator whose hops act on a matrix within a site of their
    /// input can fold that matrix of the second hops into the first (`Mdwf` and `CudaMdwf` with
    /// Neff's α).
    fn inverse_hops_twice(
        &self,
        work: &mut Self::Work,
        (o1, o2, e): (&mut H, &mut H, &mut H),
        out: &mut H,
        inp: &H,
        dagger: Dagger,
    ) where
        H: Vector,
    {
        inverse_hops_twice_of(self, work, (o1, o2, e), out, inp, dagger);
    }
}

/// `EvenOdd::inverse_hops_twice` by two `hop_then`.
pub fn inverse_hops_twice_of<F: EvenOdd<H> + ?Sized, H: Vector>(
    op: &F,
    work: &mut F::Work,
    (o1, o2, e): (&mut H, &mut H, &mut H),
    out: &mut H,
    inp: &H,
    dagger: Dagger,
) {
    let inverse = Then {
        site_inv: true,
        from: Some(inp),
    };
    op.hop_then(work, o2, inp, Parity::Odd, dagger, Then::INVERSE, o1);
    op.hop_then(work, out, o2, Parity::Even, dagger, inverse, e);
}

/// What `EvenOdd::hop_then` does to the result of the hop.
pub struct Then<'f, H> {
    /// The inverse of the block within a site (of its adjoint after the adjoint hop).
    pub site_inv: bool,
    /// Then the result is subtracted from this field.
    pub from: Option<&'f H>,
}

impl<H> Then<'_, H> {
    /// The hop alone.
    pub(crate) const NOTHING: Self = Then {
        site_inv: false,
        from: None,
    };
    /// The inverse of the block within a site after the hop.
    pub const INVERSE: Self = Then {
        site_inv: true,
        from: None,
    };
}

/// `EvenOdd::hop_then` from the blocks of the operator one after the other.
#[allow(clippy::too_many_arguments)]
pub fn hop_then_of_blocks<F: EvenOdd<H> + ?Sized, H: Vector>(
    op: &F,
    work: &mut F::Work,
    out: &mut H,
    inp: &H,
    to: Parity,
    dagger: Dagger,
    then: Then<'_, H>,
    scratch: &mut H,
) {
    if then.site_inv {
        op.hop(work, scratch, inp, to, dagger);
        op.diag_inv(work, out, scratch, to, dagger);
    } else {
        op.hop(work, out, inp, to, dagger);
    }
    if let Some(from) = then.from {
        let minus = vec![C64::new(-1.0, 0.0); inp.num_vectors()];
        out.aypx(&minus, from);
    }
}

/// The Schur complement on one parity p (q is the other):
/// `S = M_pp - M_pq M_qq^{-1} M_qp`.
pub struct Schur<'a, F: EvenOdd<H>, H> {
    op: &'a F,
    work: F::Work,
    parity: Parity,
    other_a: H,
    other_b: H,
    same: H,
    minus_one: Vec<C64>,
}

impl<'a, F: EvenOdd<H>, H: Vector> Schur<'a, F, H> {
    /// `like_same` and `like_other` only provide the shapes (and parities) of the
    /// intermediate vectors on `parity` and on the other parity.
    pub fn new(op: &'a F, parity: Parity, like_same: &H, like_other: &H) -> Self {
        Self {
            op,
            work: op.even_odd_work(),
            parity,
            other_a: like_other.zeros_like(),
            other_b: like_other.zeros_like(),
            same: like_same.zeros_like(),
            minus_one: vec![C64::new(-1.0, 0.0); like_same.num_vectors()],
        }
    }

    fn apply(&mut self, out: &mut H, inp: &H, dagger: Dagger) {
        let (p, q) = (self.parity, self.parity.flip());
        // The solvers go on with fewer vectors once some have converged.
        if self.same.num_vectors() != inp.num_vectors() {
            let n = inp.num_vectors();
            self.other_a = self.other_a.zeros_with(n);
            self.other_b = self.other_b.zeros_with(n);
            self.same = self.same.zeros_with(n);
            self.minus_one = vec![C64::new(-1.0, 0.0); n];
        }
        // (S†) has the same form with every block replaced by the block of M†.
        self.op.hop_then(
            &mut self.work,
            &mut self.other_b,
            inp,
            q,
            dagger,
            Then::INVERSE,
            &mut self.other_a,
        );
        self.op
            .hop(&mut self.work, &mut self.same, &self.other_b, p, dagger);
        self.op.diag(&mut self.work, out, inp, p, dagger);
        out.axpy(&self.minus_one, &self.same);
    }
}

impl<F: EvenOdd<H>, H: Vector> LinearFn<H> for Schur<'_, F, H> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.apply(out, inp, Dagger::No);
    }
}

impl<F: EvenOdd<H>, H: Vector> LinearOp<H> for Schur<'_, F, H> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        self.apply(out, inp, Dagger::Yes);
    }
}

/// The even-odd form of a Pauli-Villars preconditioned operator (see [`PvRatio`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EoForm {
    /// `Ã = C_PV,e†⁻¹ S_PV† S C_e⁻¹`, S = C_e − D_eo C_o⁻¹ D_oe the Schur complement of D on
    /// the even sites: the even-odd version of A. Neff's α cancels in it.
    Right,
    /// `Ã_L = (C_PV,e⁻¹ S_PV)† (C_e⁻¹ S)`, which sees α: with D^(α) = D Q it is
    /// `Q_e (C_PV,e⁻¹ S_PV)† Q_e⁻² (C_e⁻¹ S) Q_e` (α in D_PV too).
    Left,
}

/// The Schur factor `1 − M_eo M_oo⁻¹ M_oe M_ee⁻¹` (right form) or `1 − M_ee⁻¹ M_eo M_oo⁻¹ M_oe`
/// (left form) of `op` on the even sites, or its adjoint, with two scratch fields on the odd
/// sites and one on the even ones: three passes over the fields for one form, two for the
/// other, with an operator that has `hop_then` in one.
pub fn schur_factor<F: EvenOdd<H>, H: Vector>(
    op: &F,
    work: &mut F::Work,
    form: EoForm,
    ([o1, o2], e): (&mut [H; 2], &mut H),
    out: &mut H,
    inp: &H,
    dagger: Dagger,
) {
    let (even, odd) = (Parity::Even, Parity::Odd);
    // The adjoint takes the factors in the opposite order.
    if (form == EoForm::Right) == (dagger == Dagger::No) {
        let subtract = Then {
            site_inv: false,
            from: Some(inp),
        };
        op.diag_inv(work, e, inp, even, dagger);
        op.hop_then(work, o2, e, odd, dagger, Then::INVERSE, o1);
        op.hop_then(work, out, o2, even, dagger, subtract, e);
    } else {
        op.inverse_hops_twice(work, (o1, o2, e), out, inp, dagger);
    }
}

/// The Schur factor of `op` on the even sites (`schur_factor`) as a map. In the left form it is
/// K = 1 − M_ee⁻¹ M_eo M_oo⁻¹ M_oe = M_ee⁻¹ Ŝ, Bridge++'s even-odd operator ("D" of its `_eo`
/// operators).
pub struct SchurFactor<'a, F: EvenOdd<H>, H> {
    op: &'a F,
    form: EoForm,
    work: F::Work,
    odd: [H; 2],
    even: H,
}

impl<'a, F: EvenOdd<H>, H: Vector> SchurFactor<'a, F, H> {
    /// `like_even` and `like_odd` give the shapes of the fields on the two parities.
    pub fn new(op: &'a F, form: EoForm, like_even: &H, like_odd: &H) -> Self {
        Self {
            op,
            form,
            work: op.even_odd_work(),
            odd: [like_odd.zeros_like(), like_odd.zeros_like()],
            even: like_even.zeros_like(),
        }
    }

    fn apply(&mut self, out: &mut H, inp: &H, dagger: Dagger) {
        for f in self.odd.iter_mut().chain([&mut self.even]) {
            crate::fit(f, inp);
        }
        let scratch = (&mut self.odd, &mut self.even);
        schur_factor(
            self.op,
            &mut self.work,
            self.form,
            scratch,
            out,
            inp,
            dagger,
        );
    }
}

impl<F: EvenOdd<H>, H: Vector> LinearFn<H> for SchurFactor<'_, F, H> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.apply(out, inp, Dagger::No);
    }
}

impl<F: EvenOdd<H>, H: Vector> LinearOp<H> for SchurFactor<'_, F, H> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        self.apply(out, inp, Dagger::Yes);
    }
}

/// The Pauli-Villars preconditioned operator on the even sites in the form `form`, the Schur
/// factor of `op` followed by the adjoint of that of `pv` (normally the same operator at mass
/// 1), for any operator with even-odd blocks. `fermion::mdwf::PvEvenOdd` is this for the
/// operator on the CPU, with the blocks fused.
pub struct PvRatio<'a, F: EvenOdd<H>, H> {
    op: &'a F,
    pv: &'a F,
    form: EoForm,
    work: F::Work,
    work_pv: F::Work,
    odd: [H; 2],
    even: [H; 2],
}

impl<'a, F: EvenOdd<H>, H: Vector> PvRatio<'a, F, H> {
    /// `like_even` and `like_odd` give the shapes of the fields on the two parities.
    pub fn new(op: &'a F, pv: &'a F, form: EoForm, like_even: &H, like_odd: &H) -> Self {
        Self {
            op,
            pv,
            form,
            work: op.even_odd_work(),
            work_pv: pv.even_odd_work(),
            odd: [like_odd.zeros_like(), like_odd.zeros_like()],
            even: [like_even.zeros_like(), like_even.zeros_like()],
        }
    }

    /// The work fields for as many vectors as `like` has: the solvers go on with fewer once
    /// some have converged.
    fn fit_all(&mut self, like: &H) {
        for f in self.odd.iter_mut().chain(self.even.iter_mut()) {
            crate::fit(f, like);
        }
    }

    /// `out = second(first(inp))`, the factors given as (Pauli-Villars?, dagger).
    fn product(&mut self, out: &mut H, inp: &H, first: (bool, Dagger), second: (bool, Dagger)) {
        self.fit_all(inp);
        let [e0, e1] = &mut self.even;
        for (which, (pv, dagger)) in [first, second].into_iter().enumerate() {
            let (op, work) = if pv {
                (self.pv, &mut self.work_pv)
            } else {
                (self.op, &mut self.work)
            };
            let scratch = (&mut self.odd, &mut *e0);
            if which == 0 {
                schur_factor(op, work, self.form, scratch, e1, inp, dagger);
            } else {
                schur_factor(op, work, self.form, scratch, out, e1, dagger);
            }
        }
    }
}

impl<F: EvenOdd<H>, H: Vector> PvRatio<'_, F, H> {
    /// The right-hand side of `Ã y = rhs` for `D x = b`: with b′_e = b_e − D_eo C_o⁻¹ b_o,
    /// `(S_PV C_PV,e⁻¹)† b′_e` for the right form and `(C_PV,e⁻¹ S_PV)† C_e⁻¹ b′_e` for the left
    /// one.
    pub fn right_hand_side<V, P: SplitParity<V, H>>(&mut self, split: &P, rhs: &mut H, b: &V) {
        let (even, odd) = (Parity::Even, Parity::Odd);
        let (mut b_e, mut b_o) = (split.half_like(b, even), split.half_like(b, odd));
        split.extract(&mut b_e, b, even);
        split.extract(&mut b_o, b, odd);
        let (mut t_o, mut t_e) = (split.half_like(b, odd), split.half_like(b, even));
        self.op
            .diag_inv(&mut self.work, &mut t_o, &b_o, odd, Dagger::No);
        self.op
            .hop(&mut self.work, &mut t_e, &t_o, even, Dagger::No);
        let minus = vec![C64::new(-1.0, 0.0); b_e.num_vectors()];
        t_e.aypx(&minus, &b_e);
        if self.form == EoForm::Left {
            self.op
                .diag_inv(&mut self.work, &mut b_e, &t_e, even, Dagger::No);
            t_e = b_e;
        }
        self.fit_all(&t_e);
        let scratch = (&mut self.odd, &mut self.even[0]);
        schur_factor(
            self.pv,
            &mut self.work_pv,
            self.form,
            scratch,
            rhs,
            &t_e,
            Dagger::Yes,
        );
    }

    /// The solution x of `M x = b` from the solution y of `Ã y = rhs`, M the operator this was
    /// made of: x_e = M_ee⁻¹ y (right form) or y (left form), x_o = M_oo⁻¹ (b_o − M_oe x_e).
    /// For an operator D Q (Neff's α) this is x^(α); the solution of D x = b is Q x^(α).
    pub fn solution<V, P: SplitParity<V, H>>(&mut self, split: &P, x: &mut V, y: &H, b: &V) {
        let (even, odd) = (Parity::Even, Parity::Odd);
        let mut x_e = split.half_like(b, even);
        match self.form {
            EoForm::Right => self
                .op
                .diag_inv(&mut self.work, &mut x_e, y, even, Dagger::No),
            EoForm::Left => x_e.copy_from(y),
        }
        let (mut b_o, mut t_o) = (split.half_like(b, odd), split.half_like(b, odd));
        split.extract(&mut b_o, b, odd);
        self.op.hop(&mut self.work, &mut t_o, &x_e, odd, Dagger::No);
        let minus = vec![C64::new(-1.0, 0.0); y.num_vectors()];
        t_o.aypx(&minus, &b_o);
        let mut x_o = b_o;
        self.op
            .diag_inv(&mut self.work, &mut x_o, &t_o, odd, Dagger::No);
        split.insert(x, &x_e);
        split.insert(x, &x_o);
    }
}

/// The operator `a` after Neff's Q (`q(out, inp, inverse)`, Hermitian, within a site): A Q, its
/// adjoint Q A†. With K the left Schur factor of D and K^(α) that of D^(α) = D Q,
/// K^(α) = Q_e⁻¹ K Q_e, so (K†K)⁻¹ = Q_e ((K Q_e)†(K Q_e))⁻¹ Q_e: CG on the normal equations of
/// K Q_e solves those of K with Neff's better conditioned operator, the action unchanged.
pub struct AfterQ<'a, H> {
    pub a: &'a mut dyn LinearOp<H>,
    pub q: &'a dyn Fn(&mut H, &H, bool),
    t: H,
}

impl<'a, H: Vector> AfterQ<'a, H> {
    /// `like`: a field of the shape of the operator's.
    pub fn new(a: &'a mut dyn LinearOp<H>, q: &'a dyn Fn(&mut H, &H, bool), like: &H) -> Self {
        Self {
            a,
            q,
            t: like.zeros_like(),
        }
    }
}

impl<H: Vector> LinearFn<H> for AfterQ<'_, H> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        crate::fit(&mut self.t, inp);
        (self.q)(&mut self.t, inp, false);
        self.a.mult(out, &self.t);
    }
}

impl<H: Vector> LinearOp<H> for AfterQ<'_, H> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        crate::fit(&mut self.t, inp);
        self.a.mult_dag(&mut self.t, inp);
        (self.q)(out, &self.t, false);
    }
}

/// D† x = b by the adjoint of the Pauli-Villars operator: `PvRatio::new(pv, op, EoForm::Right,
/// ..)` (the roles exchanged) is Ã† = C_e⁻† S† S_PV C_PV,e⁻¹, the adjoint of Ã of `op` over `pv`,
/// and the cycle made the same way on the P of Ã (`PvEoCycle::new(pv, op, ..)`) is M†. The
/// Schur complement of D† on the even sites is S†, so with b′_e = b_e − (M_oe)† C_o⁻† b_o and
/// x_e = S_PV C_PV,e⁻¹ y the system is Ã† y = C_e⁻† b′_e. This is its right-hand side; `op` and
/// `pv` without Neff's α.
pub fn adjoint_right_hand_side<F, H, V, P>(op: &F, split: &P, rhs: &mut H, b: &V)
where
    F: EvenOdd<H>,
    H: Vector,
    P: SplitParity<V, H>,
{
    let (even, odd) = (Parity::Even, Parity::Odd);
    let mut work = op.even_odd_work();
    let (mut b_e, mut b_o) = (split.half_like(b, even), split.half_like(b, odd));
    split.extract(&mut b_e, b, even);
    split.extract(&mut b_o, b, odd);
    let (mut t_o, mut t_e) = (split.half_like(b, odd), split.half_like(b, even));
    op.diag_inv(&mut work, &mut t_o, &b_o, odd, Dagger::Yes);
    op.hop(&mut work, &mut t_e, &t_o, even, Dagger::Yes);
    let minus = vec![C64::new(-1.0, 0.0); b_e.num_vectors()];
    t_e.aypx(&minus, &b_e);
    crate::fit(rhs, &t_e);
    op.diag_inv(&mut work, rhs, &t_e, even, Dagger::Yes);
}

/// The solution x of D† x = b from the solution y of Ã† y = `adjoint_right_hand_side`:
/// x_e = S_PV C_PV,e⁻¹ y, x_o = C_o⁻† (b_o − (M_eo)† x_e).
pub fn adjoint_solution<F, H, V, P>(op: &F, pv: &F, split: &P, x: &mut V, y: &H, b: &V)
where
    F: EvenOdd<H>,
    H: Vector,
    P: SplitParity<V, H>,
{
    let (even, odd) = (Parity::Even, Parity::Odd);
    let (mut work, mut work_pv) = (op.even_odd_work(), pv.even_odd_work());
    let mut t_e = split.half_like(b, even);
    crate::fit(&mut t_e, y);
    pv.diag_inv(&mut work_pv, &mut t_e, y, even, Dagger::No);
    let mut like_odd = split.half_like(b, odd);
    crate::fit(&mut like_odd, y);
    let mut x_e = t_e.zeros_like();
    Schur::new(pv, even, &t_e, &like_odd).mult(&mut x_e, &t_e);
    let (mut b_o, mut t_o) = (split.half_like(b, odd), split.half_like(b, odd));
    split.extract(&mut b_o, b, odd);
    op.hop(&mut work, &mut t_o, &x_e, odd, Dagger::Yes);
    let minus = vec![C64::new(-1.0, 0.0); y.num_vectors()];
    t_o.aypx(&minus, &b_o);
    let mut x_o = b_o;
    op.diag_inv(&mut work, &mut x_o, &t_o, odd, Dagger::Yes);
    split.insert(x, &x_e);
    split.insert(x, &x_o);
}

/// D† x = b by the multigrid of D (`m`, a cycle of Ã of `op` over `pv`, right form), without
/// one of its own. For Möbius fermions D† = Γ5 D₋⁻¹ D Γ5 D₋† (D₋ = c W − 1), so
/// D† Γ5 D⁻¹ Γ5 = Γ5 (1 − c [Y, W] D⁻¹) Γ5: Γ5 D⁻¹ Γ5 is a preconditioner of D† whose error is
/// a term that is large only on the low modes of D and takes them out of their span (the note,
/// Sec. "Solving with D†"). On the even sites, with S the Schur complement of D: S† x_e = b′_e,
/// b′_e = b_e − (M_oe)† C_o⁻† b_o; put x_e = C_e⁻† z, so that K† z = b′_e for the left Schur
/// factor K = C_e⁻¹ S; precondition with C_e† Γ5 S⁻¹ Γ5, since S⁻† = π D⁻† ι ≈ Γ5 S⁻¹ Γ5 (Γ5
/// acts within a site). S⁻¹ c = C_e⁻¹ Ã⁻¹ X_PV† c (X_PV = S_PV C_PV,e⁻¹, the right-hand side of
/// Ã for the source (c, 0)), with Ã⁻¹ one application of `m`. Then x_o = C_o⁻† (b_o −
/// (M_eo)† x_e). `gamma5` is Γ5 on fields of one parity; `like` fields on the even and the odd
/// sites; `op` and `pv` without Neff's α. Returns what `solver` returns for z.
#[allow(clippy::too_many_arguments)]
pub fn adjoint_by_gamma5<F, H, V, P>(
    (op, pv): (&F, &F),
    split: &P,
    gamma5: &dyn Fn(&mut H, &H),
    m: &mut dyn LinearFn<H>,
    solver: &mut dyn crate::solver::Solver<H>,
    (like_even, like_odd): (&H, &H),
    x: &mut V,
    b: &V,
) -> Result<crate::solver::SolveInfo, crate::solver::SolveError>
where
    F: EvenOdd<H>,
    H: Vector,
    V: Vector,
    P: SplitParity<V, H>,
{
    let (even, odd) = (Parity::Even, Parity::Odd);
    let mut work = op.even_odd_work();
    let nv = b.num_vectors();
    let minus = vec![C64::new(-1.0, 0.0); nv];
    // b′_e.
    let (mut b_e, mut b_o) = (split.half_like(b, even), split.half_like(b, odd));
    split.extract(&mut b_e, b, even);
    split.extract(&mut b_o, b, odd);
    let (mut t_o, mut rhs) = (split.half_like(b, odd), split.half_like(b, even));
    op.diag_inv(&mut work, &mut t_o, &b_o, odd, Dagger::Yes);
    op.hop(&mut work, &mut rhs, &t_o, even, Dagger::Yes);
    rhs.aypx(&minus, &b_e);
    drop(b_e);
    // K† z = b′_e with the preconditioner C_e† Γ5 S⁻¹ Γ5.
    let mut k = SchurFactor::new(op, EoForm::Left, like_even, like_odd);
    let mut a = crate::Adjoint(&mut k);
    // Its fields are made once: the solves on 24⁴ are short of device memory.
    struct Prec<'p, F: EvenOdd<H>, H> {
        op: &'p F,
        gamma5: &'p dyn Fn(&mut H, &H),
        m: &'p mut dyn LinearFn<H>,
        /// X_PV = S_PV C_PV,e⁻¹.
        x_pv: SchurFactor<'p, F, H>,
        t: H,
        rho: H,
        y: H,
        work: F::Work,
    }
    impl<F: EvenOdd<H>, H: Vector> LinearFn<H> for Prec<'_, F, H> {
        fn mult(&mut self, out: &mut H, inp: &H) {
            let even = Parity::Even;
            for f in [&mut self.t, &mut self.rho, &mut self.y] {
                crate::fit(f, inp);
            }
            (self.gamma5)(&mut self.t, inp);
            // ρ = X_PV† t: the right-hand side of Ã for the source (t, 0) (`PvRatio::
            // right_hand_side`, with nothing on the odd sites).
            self.x_pv.mult_dag(&mut self.rho, &self.t);
            self.m.mult(&mut self.y, &self.rho);
            // S⁻¹ t ≈ C_e⁻¹ y, then Γ5 and C_e†.
            self.op
                .diag_inv(&mut self.work, &mut self.t, &self.y, even, Dagger::No);
            (self.gamma5)(&mut self.y, &self.t);
            self.op
                .diag(&mut self.work, out, &self.y, even, Dagger::Yes);
        }
    }
    let mut prec = Prec {
        op,
        gamma5,
        m,
        x_pv: SchurFactor::new(pv, EoForm::Right, like_even, like_odd),
        t: rhs.zeros_like(),
        rho: rhs.zeros_like(),
        y: rhs.zeros_like(),
        work: op.even_odd_work(),
    };
    let mut z = rhs.zeros_like();
    let info = solver.solve(&mut a, Some(&mut prec), &mut z, &rhs);
    drop(prec);
    // x_e = C_e⁻† z, x_o = C_o⁻† (b_o − (M_eo)† x_e).
    let mut x_e = rhs.zeros_like();
    op.diag_inv(&mut work, &mut x_e, &z, even, Dagger::Yes);
    op.hop(&mut work, &mut t_o, &x_e, odd, Dagger::Yes);
    t_o.aypx(&minus, &b_o);
    let mut x_o = b_o;
    op.diag_inv(&mut work, &mut x_o, &t_o, odd, Dagger::Yes);
    split.insert(x, &x_e);
    split.insert(x, &x_o);
    info
}

impl<F: EvenOdd<H>, H: Vector> LinearFn<H> for PvRatio<'_, F, H> {
    /// `out = Ã inp`: the factor of the operator, then the adjoint of the Pauli-Villars one.
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.product(out, inp, (false, Dagger::No), (true, Dagger::Yes));
    }
}

impl<F: EvenOdd<H>, H: Vector> LinearOp<H> for PvRatio<'_, F, H> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        self.product(out, inp, (true, Dagger::No), (false, Dagger::Yes));
    }
}

type HalfField<T, K, L> = Field<T, K, Half, L, Cpu>;

/// How a field on the whole lattice (`V`) splits into fields on one parity (`H`).
pub trait SplitParity<V, H> {
    /// A zero field on `parity`, for the part of `like` there.
    fn half_like(&self, like: &V, parity: Parity) -> H;
    /// `half` = the part of `full` on `parity`.
    fn extract(&self, half: &mut H, full: &V, parity: Parity);
    /// The part of `full` on the parity of `half` = `half`.
    fn insert(&self, full: &mut V, half: &H);
}

/// Fields on the CPU split by the numbering of the sites of each parity.
impl<T: Storage, K: Kind, L: Layout> SplitParity<Field<T, K, crate::Lexical, L>, HalfField<T, K, L>>
    for EvenOddIndex
{
    fn half_like(
        &self,
        like: &Field<T, K, crate::Lexical, L>,
        parity: Parity,
    ) -> HalfField<T, K, L> {
        let mut f = HalfField::<T, K, L>::with_shape(
            like.lattice(),
            like.internal_dof(),
            like.ls(),
            Field::num_vectors(like),
            like.outer(),
        );
        f.set_parity(parity);
        f
    }

    fn extract(
        &self,
        half: &mut HalfField<T, K, L>,
        full: &Field<T, K, crate::Lexical, L>,
        parity: Parity,
    ) {
        extract_parity(half, full, self, parity);
    }

    fn insert(&self, full: &mut Field<T, K, crate::Lexical, L>, half: &HalfField<T, K, L>) {
        insert_parity(full, half, self);
    }
}

impl<V, H, P: SplitParity<V, H>> SplitParity<V, H> for std::sync::Arc<P> {
    fn half_like(&self, like: &V, parity: Parity) -> H {
        (**self).half_like(like, parity)
    }
    fn extract(&self, half: &mut H, full: &V, parity: Parity) {
        (**self).extract(half, full, parity);
    }
    fn insert(&self, full: &mut V, half: &H) {
        (**self).insert(full, half);
    }
}

/// Solve `M x = b` on the full lattice through the Schur complement on the even sites:
/// `solve` solves `S x_e = rhs`. `split` is the numbering of the operator (`EvenOddIndex`) for
/// fields on the CPU.
pub fn solve_even_odd<V, H, F, P, R, E>(
    op: &F,
    split: &P,
    x: &mut V,
    b: &V,
    solve: impl FnOnce(&mut Schur<'_, F, H>, &mut H, &H) -> Result<R, E>,
) -> Result<R, E>
where
    H: Vector,
    F: EvenOdd<H>,
    P: SplitParity<V, H>,
{
    let half = |parity: Parity| split.half_like(b, parity);
    let (even, odd) = (Parity::Even, Parity::Odd);
    let mut work = op.even_odd_work();

    let (mut b_e, mut b_o) = (half(even), half(odd));
    split.extract(&mut b_e, b, even);
    split.extract(&mut b_o, b, odd);
    let minus_one = vec![C64::new(-1.0, 0.0); b_e.num_vectors()];

    // rhs = b_e - M_eo M_oo^{-1} b_o
    let (mut t_o, mut rhs) = (half(odd), half(even));
    op.diag_inv(&mut work, &mut t_o, &b_o, odd, Dagger::No);
    op.hop(&mut work, &mut rhs, &t_o, even, Dagger::No);
    rhs.aypx(&minus_one, &b_e);

    let mut x_e = half(even);
    let result = {
        let mut schur = Schur::new(op, even, &x_e, &t_o);
        solve(&mut schur, &mut x_e, &rhs)?
    };

    // x_o = M_oo^{-1} (b_o - M_oe x_e)
    let mut x_o = half(odd);
    op.hop(&mut work, &mut t_o, &x_e, odd, Dagger::No);
    t_o.aypx(&minus_one, &b_o);
    op.diag_inv(&mut work, &mut x_o, &t_o, odd, Dagger::No);

    split.insert(x, &x_e);
    split.insert(x, &x_o);
    Ok(result)
}

/// An approximate inverse of the Schur complement S on one parity from one of the whole operator:
/// the block of M⁻¹ on that parity is exactly S⁻¹, so with an approximate inverse `inner` of M,
/// `[inner (r ⊕ 0)]_p` approximates S⁻¹ r as well as `inner` approximates M⁻¹. `split` splits
/// the fields by parity: `EvenOddIndex` on the CPU, `DeviceParities` on the device.
pub struct OnParity<'m, V, H, P> {
    inner: &'m mut dyn LinearFn<V>,
    split: &'m P,
    parity: Parity,
    inp: V,
    out: V,
    half: std::marker::PhantomData<H>,
}

impl<'m, V: Vector, H, P: SplitParity<V, H>> OnParity<'m, V, H, P> {
    /// On the even sites; `like` gives the shape of the whole fields.
    pub fn new(inner: &'m mut dyn LinearFn<V>, split: &'m P, like: &V) -> Self {
        Self {
            inner,
            split,
            parity: Parity::Even,
            inp: like.zeros_like(),
            out: like.zeros_like(),
            half: std::marker::PhantomData,
        }
    }
}

impl<V: Vector, H: Vector, P: SplitParity<V, H>> LinearFn<H> for OnParity<'_, V, H, P> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        let n = inp.num_vectors();
        if self.inp.num_vectors() != n {
            self.inp = self.inp.zeros_with(n);
            self.out = self.out.zeros_with(n);
        }
        self.inp.set_zero();
        self.split.insert(&mut self.inp, inp);
        self.inner.mult(&mut self.out, &self.inp);
        self.split.extract(out, &self.out, self.parity);
    }
}
