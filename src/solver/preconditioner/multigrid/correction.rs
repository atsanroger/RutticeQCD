//! The coarse correction of the multigrid for the even-odd Pauli-Villars operator of Möbius
//! domain-wall fermions (the top of `mdwf_even_odd`), and the solve of a coarse system, for
//! any kind of field: on the CPU and on a device they are this.

use super::mdwf_even_odd::{coarse_solve, CoarseKrylov, PvEoParams};
use crate::fermion::{Dagger, EoForm, EvenOdd};
use crate::solver::Stop;
use crate::{fit, ConvertFrom, LinearFn, Parity, Vector, C64};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// A closure as a linear map.
pub struct FnMap<F>(pub F);

impl<V: Vector, F: FnMut(&mut V, &V)> LinearFn<V> for FnMap<F> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        (self.0)(out, inp);
    }
}

/// A coarse operator A = P† D P on coarse fields `C`, with its even-odd form where the coarse
/// lattice has one (`CoarseOp`): D_c = [[A_ee, B_eo], [B_oe, A_oo]], S = A_ee − B_eo A_oo⁻¹ B_oe.
pub trait CoarseOperator<C> {
    /// Whether `schur` and the like are there.
    fn has_even_odd(&self) -> bool;
    /// `out = A inp`.
    fn mult(&self, out: &mut C, inp: &C);
    /// `out = H inp` for H = Γ5 (Γ5 A Γ5) = A Γ5, the Γ5-Hermitian form.
    fn mult_hermitian(&self, out: &mut C, inp: &C);
    /// `v = Γ5 v`.
    fn gamma5(&self, v: &mut C);
    /// `out = S inp` on the even sites; `odd` is scratch.
    fn schur(&self, out: &mut C, inp: &C, odd: &mut C);
    /// `out = g_e − B_eo A_oo⁻¹ g_o`.
    fn schur_source(&self, out: &mut C, g: &C, odd: &mut C);
    /// `c` holds c_e and gets `c_o = A_oo⁻¹ (g_o − B_oe c_e)` added.
    fn schur_solution(&self, c: &mut C, g: &C, odd: &mut C);
    /// `out = inp` on the even sites, zero on the odd ones.
    fn even_part(&self, out: &mut C, inp: &C);
    /// `out = A_ee⁻¹ inp` on the even sites, zero on the odd ones.
    fn even_block_inverse(&self, out: &mut C, inp: &C);
}

impl<C, D: CoarseOperator<C>> CoarseOperator<C> for Rc<D> {
    fn has_even_odd(&self) -> bool {
        (**self).has_even_odd()
    }
    fn mult(&self, out: &mut C, inp: &C) {
        (**self).mult(out, inp);
    }
    fn mult_hermitian(&self, out: &mut C, inp: &C) {
        (**self).mult_hermitian(out, inp);
    }
    fn gamma5(&self, v: &mut C) {
        (**self).gamma5(v);
    }
    fn schur(&self, out: &mut C, inp: &C, odd: &mut C) {
        (**self).schur(out, inp, odd);
    }
    fn schur_source(&self, out: &mut C, g: &C, odd: &mut C) {
        (**self).schur_source(out, g, odd);
    }
    fn schur_solution(&self, c: &mut C, g: &C, odd: &mut C) {
        (**self).schur_solution(c, g, odd);
    }
    fn even_part(&self, out: &mut C, inp: &C) {
        (**self).even_part(out, inp);
    }
    fn even_block_inverse(&self, out: &mut C, inp: &C) {
        (**self).even_block_inverse(out, inp);
    }
}

/// Levels below the coarse lattice, as a map of a coarse field.
pub type Levels<'a, C> = dyn Fn(&mut C, &C) + 'a;

/// A solver of a coarse system, as `solve_shared` calls it: operator, preconditioner, solution
/// (zero on entry), right-hand side.
pub type CoarseSolve<'a, C> =
    dyn FnMut(&mut dyn LinearFn<C>, Option<&mut dyn LinearFn<C>>, &mut C, &C) + 'a;

/// c = A⁻¹ g approximately for the coarse operator `a`, by `solve`: on the even coarse sites
/// where `a` has the even-odd form (preconditioned by the inverses of the blocks there, or
/// with `levels` by the even sites of their cycle), otherwise on all of them, right-
/// preconditioned by `blocks` (the inverses of the blocks) and the levels as far as there are
/// any: z₁ = L r, z = z₁ + B⁻¹ (r − A z₁), one more application of A; an additive combination
/// made BiCGStab worse. `n` counts the applications of `a`.
pub fn solve_shared<C: Vector, D: CoarseOperator<C> + ?Sized>(
    a: &D,
    mut blocks: Option<&mut dyn LinearFn<C>>,
    levels: Option<&Levels<'_, C>>,
    n: &Cell<usize>,
    solve: &mut CoarseSolve<'_, C>,
    c: &mut C,
    g: &C,
) {
    let plain = blocks.is_none() && levels.is_none();
    let mut levels_and_blocks = |out: &mut C, inp: &C| {
        let Some(levels) = levels else {
            match blocks.as_mut() {
                Some(b) => b.mult(out, inp),
                None => out.copy_from(inp),
            }
            return;
        };
        let nv = inp.num_vectors();
        let mut z1 = inp.zeros_like();
        levels(&mut z1, inp);
        let mut r = inp.zeros_like();
        a.mult(&mut r, &z1);
        n.set(n.get() + nv);
        r.aypx(&vec![C64::new(-1.0, 0.0); nv], inp);
        match blocks.as_mut() {
            Some(b) => b.mult(out, &r),
            None => out.copy_from(&r),
        }
        out.axpy(&vec![C64::new(1.0, 0.0); nv], &z1);
    };
    if !a.has_even_odd() {
        let mut whole = FnMap(|out: &mut C, inp: &C| {
            n.set(n.get() + inp.num_vectors());
            a.mult(out, inp);
        });
        if plain {
            return solve(&mut whole, None, c, g);
        }
        return solve(&mut whole, Some(&mut FnMap(levels_and_blocks)), c, g);
    }
    let (mut source, mut odd) = (g.zeros_like(), g.zeros_like());
    a.schur_source(&mut source, g, &mut odd);
    let mut scratch = g.zeros_like();
    let mut schur = FnMap(|out: &mut C, inp: &C| {
        n.set(n.get() + inp.num_vectors());
        a.schur(out, inp, &mut scratch);
    });
    match levels {
        None => {
            let mut m = FnMap(|out: &mut C, inp: &C| a.even_block_inverse(out, inp));
            solve(&mut schur, Some(&mut m), c, &source);
        }
        // The block of A⁻¹ on the even sites is the inverse of the Schur complement: the even
        // sites of what approximates A⁻¹, applied to a field that is zero on the odd sites.
        Some(_) => {
            let mut whole = g.zeros_like();
            let mut m = FnMap(|out: &mut C, inp: &C| {
                fit(&mut whole, inp);
                levels_and_blocks(&mut whole, inp);
                a.even_part(out, &whole);
            });
            solve(&mut schur, Some(&mut m), c, &source);
        }
    }
    a.schur_solution(c, g, &mut odd);
    // The source and the odd sites of the solution: one application's work.
    n.set(n.get() + g.num_vectors());
}

/// P between fine fields `V` on the even sites and coarse fields `G`.
pub trait Transfer<V, G> {
    /// A zero field on the coarse lattice.
    fn coarse_field(&self, vectors: usize) -> G;
    /// `coarse = P† ι fine`, ι putting the even sites onto the whole lattice.
    fn restrict(&self, coarse: &mut G, fine: &V);
    /// `fine = π P coarse`, π taking the even sites.
    fn prolong(&self, fine: &mut V, coarse: &G);
}

/// A P shared by several corrections (the cycles of Ã and Ã† in HMC).
impl<V, G, T: Transfer<V, G>> Transfer<V, G> for std::rc::Rc<T> {
    fn coarse_field(&self, vectors: usize) -> G {
        (**self).coarse_field(vectors)
    }
    fn restrict(&self, coarse: &mut G, fine: &V) {
        (**self).restrict(coarse, fine)
    }
    fn prolong(&self, fine: &mut V, coarse: &G) {
        (**self).prolong(fine, coarse)
    }
}

/// u = A⁻¹ g approximately for a coarse operator `a` that is nearly its own Γ5-adjoint
/// (A† ≈ Γ5 A Γ5, the Pauli-Villars operator), by MINRES on the Hermitian form H = A Γ5,
/// u = Γ5 w for H w = g: on the even coarse sites where `a` has the even-odd form (the Schur
/// complement of H is S Γ5), else on the whole coarse lattice. `blocks` = |H_xx|⁻¹; with
/// `levels` (Hermitian positive) the even sites of blocks plus levels on the whole coarse
/// lattice. `n` counts the applications of `a`.
#[allow(clippy::too_many_arguments)]
pub fn minres_on_hermitian<C: Vector, D: CoarseOperator<C> + ?Sized>(
    a: &D,
    blocks: &mut dyn LinearFn<C>,
    levels: Option<&Levels<'_, C>>,
    n: &Cell<usize>,
    stop: Stop,
    u: &mut C,
    g: &C,
) {
    let nv = g.num_vectors();
    let mut blocks = FnMap(|out: &mut C, inp: &C| {
        blocks.mult(out, inp);
        if let Some(levels) = levels {
            levels(out, inp);
        }
    });
    if a.has_even_odd() {
        // The blocks |H_xx|⁻¹ act within a site and keep a field on the even sites there.
        let (mut source, mut odd) = (g.zeros_like(), g.zeros_like());
        a.schur_source(&mut source, g, &mut odd);
        let (mut turned, mut scratch) = (g.zeros_like(), g.zeros_like());
        let mut h = FnMap(|out: &mut C, inp: &C| {
            n.set(n.get() + inp.num_vectors());
            fit(&mut turned, inp);
            turned.copy_from(inp);
            a.gamma5(&mut turned);
            a.schur(out, &turned, &mut scratch);
        });
        let mut whole = g.zeros_like();
        let mut m = FnMap(|out: &mut C, inp: &C| match levels {
            None => blocks.mult(out, inp),
            Some(_) => {
                fit(&mut whole, inp);
                blocks.mult(&mut whole, inp);
                a.even_part(out, &whole);
            }
        });
        coarse_solve(CoarseKrylov::Minres, stop, &mut h, Some(&mut m), u, &source);
        a.gamma5(u);
        a.schur_solution(u, g, &mut odd);
        // The source and the odd sites of the solution: one application's work.
        n.set(n.get() + nv);
    } else {
        let mut h = FnMap(|out: &mut C, inp: &C| {
            n.set(n.get() + inp.num_vectors());
            a.mult_hermitian(out, inp);
        });
        coarse_solve(CoarseKrylov::Minres, stop, &mut h, Some(&mut blocks), u, g);
        a.gamma5(u);
    }
}

/// A domain-wall operator as the correction needs it: the even-odd blocks, and Neff's Q.
pub trait NeffQ<V>: EvenOdd<V> {
    /// `out = Q inp`, or `Q⁻¹ inp`.
    fn mult_q(&self, out: &mut V, inp: &V, inverse: bool);
}

/// e = O_L π P (P†DP)⁻¹ P† ι M π P (P†D_PV†P)⁻¹ P† ι O_R r, the coarse correction for the
/// operator `PvEvenOdd` of `outer` in the form `form` (the top of `mdwf_even_odd`):
///
/// - fine fields `V` on the even sites, with the operators `O`: `plain` = (D, the second
///   factor), which the coarse operators are those of, and `outer`, the same as they are in
///   the operator that is solved (with Neff's α);
/// - P, `T`, into coarse fields `G`; the coarse systems are solved on fields `C`, which may
///   be of a lower precision;
/// - the coarse operators `D`, the quark factor by `params.d_solver` (`solve_shared`; MINRES on
///   its Γ5-Hermitian form, `minres_on_hermitian`, only where `plain` is exchanged, D_PV first,
///   as in the dual cycle for D†), the second by MINRES on H = Γ5 (P†D_PV P)† with `blocks_pv` = |H_xx|⁻¹, on the even coarse
///   sites where there is the even-odd form; or, with another solver, the second coarse
///   operator is the exact adjoint P†D_PV†P itself (`CoarseOp::adjoint`), solved as the first
///   (`solve_shared`, on the even coarse sites where it has the even-odd form) with
///   `blocks_pv` the inverses of its blocks;
/// - levels below the coarse lattice for either factor, if they are set.
pub struct Correction<'a, V, G, C, O: NeffQ<V>, T, D> {
    plain: (&'a O, &'a O),
    outer: (&'a O, &'a O),
    form: EoForm,
    transfer: T,
    coarse_d: D,
    coarse_pv: D,
    blocks_d: Box<dyn LinearFn<C> + 'a>,
    blocks_pv: Box<dyn LinearFn<C> + 'a>,
    levels_d: Option<Box<Levels<'a, C>>>,
    levels_pv: Option<Box<Levels<'a, C>>>,
    params: PvEoParams,
    work: (O::Work, O::Work),
    /// Scratch on the even sites and on the coarse lattice: `g` as P gives it, the others
    /// as the coarse systems are solved.
    h: V,
    g: G,
    gc: C,
    w: C,
    c: C,
    coarse_applications: Cell<usize>,
}

impl<'a, V, G, C, O, T, D> Correction<'a, V, G, C, O, T, D>
where
    V: Vector,
    G: Vector + ConvertFrom<C>,
    C: Vector + ConvertFrom<G>,
    O: NeffQ<V>,
    T: Transfer<V, G>,
    D: CoarseOperator<C>,
{
    /// `blocks_d`: the inverses of the blocks of P†DP. `like`: a field on the even sites;
    /// `coarse_like`: a coarse field as the coarse systems are solved.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plain: (&'a O, &'a O),
        outer: (&'a O, &'a O),
        form: EoForm,
        transfer: T,
        (coarse_d, coarse_pv): (D, D),
        (blocks_d, blocks_pv): (Box<dyn LinearFn<C> + 'a>, Box<dyn LinearFn<C> + 'a>),
        params: PvEoParams,
        like: &V,
        coarse_like: &C,
    ) -> Self {
        Self {
            work: (plain.0.even_odd_work(), outer.0.even_odd_work()),
            plain,
            outer,
            form,
            g: transfer.coarse_field(1),
            transfer,
            coarse_d,
            coarse_pv,
            blocks_d,
            blocks_pv,
            levels_d: None,
            levels_pv: None,
            params,
            h: like.zeros_with(1),
            gc: coarse_like.zeros_with(1),
            w: coarse_like.zeros_with(1),
            c: coarse_like.zeros_with(1),
            coarse_applications: Cell::new(0),
        }
    }

    pub fn params(&self) -> &PvEoParams {
        &self.params
    }

    /// The coarse operators of the quark factor and of the second one.
    pub fn coarse_mut(&mut self) -> (&mut D, &mut D) {
        (&mut self.coarse_d, &mut self.coarse_pv)
    }

    /// The coarse operators, the correction given up.
    pub fn into_coarse(self) -> (D, D) {
        (self.coarse_d, self.coarse_pv)
    }

    /// Levels below the coarse lattice for the solves of the quark factor: an approximate
    /// inverse of P†DP on the whole coarse lattice.
    pub fn set_levels_d(&mut self, levels: Box<Levels<'a, C>>) {
        self.levels_d = Some(levels);
    }

    /// The same for the second factor, solved by MINRES: `levels(out, inp)` adds to `out`
    /// what the levels give of `inp`, Hermitian positive.
    pub fn set_levels_pv(&mut self, levels: Box<Levels<'a, C>>) {
        assert!(
            self.params.pv_solver == CoarseKrylov::Minres,
            "levels for the second factor go with MINRES"
        );
        self.levels_pv = Some(levels);
    }

    /// Coarse-operator applications so far (both factors, the preconditioners' included).
    pub fn coarse_applications(&self) -> usize {
        self.coarse_applications.get()
    }

    /// `e` = the coarse correction of `r`, fields on the even sites.
    pub fn correct(&mut self, e: &mut V, r: &V) {
        let nv = r.num_vectors();
        let Self {
            plain: (op, pv),
            outer: (op_a, pv_a),
            form,
            transfer,
            coarse_d,
            coarse_pv,
            blocks_d,
            blocks_pv,
            levels_d,
            levels_pv,
            params,
            work: (work, work_a),
            h,
            g,
            gc,
            w,
            c,
            coarse_applications: n,
        } = self;
        fit(h, r);
        if g.num_vectors() != nv {
            *g = transfer.coarse_field(nv);
            for f in [&mut *gc, &mut *w, &mut *c] {
                *f = f.zeros_with(nv);
            }
        }
        let (even, coarse_pv, n) = (Parity::Even, &*coarse_pv, &*n);
        let stop = Stop::simple(params.coarse_tolerance_sqr, params.coarse_max_iterations);
        let pv_stop = Stop::simple(params.pv_tolerance_sqr, params.pv_max_iterations);

        // h = O_R r, g = P† ι h.
        match form {
            EoForm::Right => pv.diag(work, h, r, even, Dagger::Yes),
            EoForm::Left => pv_a.mult_q(h, r, true),
        }
        transfer.restrict(g, h);
        gc.convert_from(g);

        // w = (P†D_PV†P)⁻¹ g: by MINRES, w = Γ5 u with P†D_PV P u = Γ5 g (P†D_PV P taken as its
        // own Γ5-adjoint), or on the exact adjoint, which `coarse_pv` then is. Preconditioned
        // by the blocks, and the levels below if there are any.
        w.set_zero();
        if params.pv_solver == CoarseKrylov::Minres {
            coarse_pv.gamma5(gc);
            minres_on_hermitian(
                coarse_pv,
                &mut **blocks_pv,
                levels_pv.as_deref(),
                n,
                pv_stop,
                w,
                gc,
            );
            coarse_pv.gamma5(w);
        } else {
            let pv_solver = params.pv_solver;
            solve_shared(
                coarse_pv,
                Some(&mut **blocks_pv),
                levels_pv.as_deref(),
                n,
                &mut |a, m, x, b| coarse_solve(pv_solver, pv_stop, a, m, x, b),
                w,
                gc,
            );
        }

        // g = P† ι M π P w; M = 1 (right form) or C_e^(α) C_PV,e^(α)† (left form).
        g.convert_from(w);
        transfer.prolong(h, g);
        if *form == EoForm::Left {
            pv_a.diag(work_a, e, h, even, Dagger::Yes);
            op_a.diag(work_a, h, e, even, Dagger::No);
        }
        transfer.restrict(g, h);
        gc.convert_from(g);

        // c = (P†DP)⁻¹ g; by MINRES on its Γ5-Hermitian form where the first operator is D_PV
        // (the dual cycle, `plain` exchanged), with `blocks_d` = |H_xx|⁻¹.
        c.set_zero();
        let d_solver = params.d_solver;
        if d_solver == CoarseKrylov::Minres {
            minres_on_hermitian(&*coarse_d, &mut **blocks_d, None, n, stop, c, gc);
        } else {
            solve_shared(
                &*coarse_d,
                Some(&mut **blocks_d),
                levels_d.as_deref(),
                n,
                &mut |a, m, x, b| coarse_solve(d_solver, stop, a, m, x, b),
                c,
                gc,
            );
        }

        // e = O_L π P c: C_e (right form) or Q_e⁻¹ (left form).
        g.convert_from(c);
        transfer.prolong(h, g);
        match form {
            EoForm::Right => op.diag(work, e, h, even, Dagger::No),
            EoForm::Left => op_a.mult_q(e, h, true),
        }
    }
}

/// A coarse operator of the CPU that its solver and the preconditioner of that share.
impl<T: crate::Storage> CoarseOperator<crate::Field<T, crate::Coarse>>
    for RefCell<super::CoarseOp<T>>
{
    fn has_even_odd(&self) -> bool {
        self.borrow().has_even_odd()
    }
    fn mult(&self, out: &mut crate::Field<T, crate::Coarse>, inp: &crate::Field<T, crate::Coarse>) {
        self.borrow_mut().mult(out, inp);
    }
    fn mult_hermitian(
        &self,
        out: &mut crate::Field<T, crate::Coarse>,
        inp: &crate::Field<T, crate::Coarse>,
    ) {
        use crate::LinearOp;
        self.borrow_mut().mult_dag(out, inp);
        self.borrow().gamma5(out);
    }
    fn gamma5(&self, v: &mut crate::Field<T, crate::Coarse>) {
        self.borrow().gamma5(v);
    }
    fn schur(
        &self,
        out: &mut crate::Field<T, crate::Coarse>,
        inp: &crate::Field<T, crate::Coarse>,
        odd: &mut crate::Field<T, crate::Coarse>,
    ) {
        self.borrow().schur(out, inp, odd);
    }
    fn schur_source(
        &self,
        out: &mut crate::Field<T, crate::Coarse>,
        g: &crate::Field<T, crate::Coarse>,
        odd: &mut crate::Field<T, crate::Coarse>,
    ) {
        self.borrow().schur_source(out, g, odd);
    }
    fn schur_solution(
        &self,
        c: &mut crate::Field<T, crate::Coarse>,
        g: &crate::Field<T, crate::Coarse>,
        odd: &mut crate::Field<T, crate::Coarse>,
    ) {
        self.borrow().schur_solution(c, g, odd);
    }
    fn even_part(
        &self,
        out: &mut crate::Field<T, crate::Coarse>,
        inp: &crate::Field<T, crate::Coarse>,
    ) {
        self.borrow().even_part(out, inp);
    }
    fn even_block_inverse(
        &self,
        out: &mut crate::Field<T, crate::Coarse>,
        inp: &crate::Field<T, crate::Coarse>,
    ) {
        self.borrow().even_block_inverse(out, inp);
    }
}
