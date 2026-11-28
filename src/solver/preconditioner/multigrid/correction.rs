//! The coarse correction of the multigrid for the even-odd Pauli-Villars operator of Möbius
//! domain-wall fermions (the top of `mdwf_even_odd`), and the solve of a coarse system, for
//! any kind of field: on the CPU and on a device they are this.

use super::mdwf_even_odd::{coarse_solve, CoarseKrylov, FactorSolve, PvEoParams};
use crate::fermion::{Dagger, EoForm, EvenOdd};
use crate::solver::{Solver, Stop};
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

/// `out = A† inp` for the regularized solve of a coarse factor A: by a coarse operator that is
/// A† (a copy), or from the blocks of A themselves.
pub trait ApplyAdjoint<C> {
    fn apply_adjoint(&self, out: &mut C, inp: &C);
}

/// A copy of the adjoint: its `mult`.
impl<C, D: CoarseOperator<C>> ApplyAdjoint<C> for D {
    fn apply_adjoint(&self, out: &mut C, inp: &C) {
        self.mult(out, inp);
    }
}

/// The adjoint of a coarse factor, for its regularized solve.
pub type Adjoint<'a, C> = Box<dyn ApplyAdjoint<C> + 'a>;

/// A solver of a coarse system, as `solve_shared` calls it: operator, preconditioner, solution
/// (zero on entry), right-hand side.
pub type CoarseSolve<'a, C> =
    dyn FnMut(&mut dyn LinearFn<C>, Option<&mut dyn LinearFn<C>>, &mut C, &C) + 'a;

/// x = (A†A + ε)⁻¹ A† g by CG on all coarse sites, from x = 0: against A⁻¹ g, the component along
/// a right singular vector of A is multiplied by σ²/(σ² + ε). `adj` applies A†; `n` counts the
/// applications of either.
fn regularized_solve<C: Vector, D: CoarseOperator<C> + ?Sized>(
    a: &D,
    adj: &dyn ApplyAdjoint<C>,
    eps: f64,
    stop: Stop,
    n: &Cell<usize>,
    x: &mut C,
    g: &C,
) {
    let nv = g.num_vectors();
    let mut b = g.zeros_like();
    adj.apply_adjoint(&mut b, g);
    n.set(n.get() + nv);
    // CG drops the vectors that have converged, so `t` and the shift follow `inp`.
    let mut t = g.zeros_like();
    let mut normal = FnMap(|out: &mut C, inp: &C| {
        fit(&mut t, inp);
        a.mult(&mut t, inp);
        adj.apply_adjoint(out, &t);
        n.set(n.get() + 2 * inp.num_vectors());
        out.axpy(&vec![C64::new(eps, 0.0); inp.num_vectors()], inp);
    });
    let _ = crate::solver::Cg::inner(stop).solve(&mut normal, None, x, &b);
}

/// The largest |b − A x|²/|b|² over the vectors, for the diagnostics of `correct`.
fn diag_residual<C: Vector>(apply: impl FnOnce(&mut C, &C), x: &C, b: &C) -> f64 {
    let nv = b.num_vectors();
    let mut t = b.zeros_like();
    apply(&mut t, x);
    t.axpy(&vec![C64::new(-1.0, 0.0); nv], b);
    let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
    t.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    rr.iter().zip(&bb).fold(0.0, |m, (r, b)| f64::max(m, r / b))
}

thread_local! {
    /// exp/ls16-diag: when set, `correct` only stores |π P P† ι O_R r|²/|O_R r|² of its input.
    pub(crate) static DIAG_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static DIAG_SHARE: std::cell::Cell<f64> = const { std::cell::Cell::new(f64::NAN) };
    /// exp/ls16-diag: 0 the whole correction, 1 the quark factor alone (no PV solve), 2 the
    /// Pauli-Villars factor alone (no quark solve).
    pub(crate) static DIAG_FACTOR: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// |x|/|b| of the first vector, for the diagnostics of `correct`.
fn norm_ratio<C: Vector>(x: &C, b: &C) -> f64 {
    let nv = b.num_vectors();
    let (mut xx, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
    x.norm_sqr(&mut xx);
    b.norm_sqr(&mut bb);
    (xx[0] / bb[0]).sqrt()
}

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
pub(crate) fn minres_on_hermitian<C: Vector, D: CoarseOperator<C> + ?Sized>(
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
    /// exp/ls16-diag (RUTTICE_DSYM): `out = B⁻¹ inp` with B = b − cY of the quark operator,
    /// whose coarse operator is then P† D B⁻¹ P; false (and nothing done) otherwise.
    fn dsym_right(&self, _out: &mut V, _inp: &V) -> bool {
        false
    }
    /// exp/ls16-diag (RUTTICE_DSYM_PV): `out = B_PV⁻† inp` for the Pauli-Villars operator,
    /// whose factor is then B⁻¹P (P† D_PV,sym† B⁻¹ P)⁻¹ P† B_PV⁻†; false otherwise.
    fn dsym_pv_left(&self, _out: &mut V, _inp: &V) -> bool {
        false
    }
    /// exp/ls16-diag (RUTTICE_DL): `out = B5,e inp` for the quark operator, whose coarse
    /// factor is then P (P† B5 D P)⁻¹ P† ι B5,e; false otherwise.
    fn dl_left(&self, _out: &mut V, _inp: &V) -> bool {
        false
    }
}

/// e = O_L π P (P†DP)⁻¹ P† ι M π P (P†D_PV†P)⁻¹ P† ι O_R r, the coarse correction for the
/// operator `PvEvenOdd` of `outer` in the form `form` (the top of `mdwf_even_odd`):
///
/// - fine fields `V` on the even sites, with the operators `O`: `plain` = (D, the second
///   factor), which the coarse operators are those of, and `outer`, the same as they are in
///   the operator that is solved (with Neff's α);
/// - P, `T`, into coarse fields `G`; the coarse systems are solved on fields `C`, which may
///   be of a lower precision;
/// - the coarse operators `D`, the quark factor by `params.d_solver` (`solve_shared`), the
///   second by MINRES on H = Γ5 (P†D_PV P)† with `blocks_pv` = |H_xx|⁻¹, on the even coarse
///   sites where there is the even-odd form; or, with another solver, the second coarse
///   operator is the exact adjoint P†D_PV†P itself (`CoarseOp::adjoint`), solved as the first
///   (`solve_shared`, on the even coarse sites where it has the even-odd form) with
///   `blocks_pv` the inverses of its blocks;
/// - a factor solved by `FactorSolve::Regularized` applies its adjoint as well (`adjoints`:
///   (P†DP)† for the first, P†D_PV P for the second), however it is applied;
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
    /// The adjoints of the coarse factors solved by `FactorSolve::Regularized`.
    coarse_d_adj: Option<Adjoint<'a, C>>,
    coarse_pv_adj: Option<Adjoint<'a, C>>,
    params: PvEoParams,
    work: (O::Work, O::Work),
    /// Scratch on the even sites and on the coarse lattice: `g` as P gives it, the others
    /// as the coarse systems are solved.
    h: V,
    g: G,
    gc: C,
    w: C,
    c: C,
    /// exp/ls16-diag (RUTTICE_SPLIT): the coarse fields of the Pauli-Villars step, of the
    /// dimension of P_v (that of P otherwise).
    g_pv: G,
    gc_pv: C,
    coarse_applications: Cell<usize>,
    /// exp/mr-coarse: the low directions of the quark coarse operator, solved by minimal
    /// residual (`LowModes`).
    low: Option<LowModes<'a, V, C>>,
}

/// exp/mr-coarse: k singular triplets (u_j, σ_j, v_j) of the quark coarse operator D_c = P†AP
/// and, with Z = A P V_k on the whole lattice, `g` = P†A†Z, `ztz_inv` = (Z†Z)⁻¹ (k × k,
/// row-major; Z†Z = V†G) and `adj`, the map h ↦ P†A†ι h of a field h on the even sites. The
/// quark solve then takes the left test space [P U_⊥, Z] in place of P: y_⊥ = D_c⁻¹ (1 −
/// U_k U_k†) g on the other directions, and on these α = (Z†Z)⁻¹ (V† P†A†ι h − G† y_⊥), the
/// minimal residual over span A P V_k. Z itself is not kept (on 24⁴ at L_s = 16 it is a GB).
pub struct LowModes<'a, V, C> {
    pub v: C,
    pub u: C,
    pub g: C,
    pub ztz_inv: Vec<crate::C64>,
    pub adj: Box<dyn Fn(&V) -> C + 'a>,
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
    /// `blocks_d`: the inverses of the blocks of P†DP; `adjoints`: those of the factors solved
    /// by `FactorSolve::Regularized`, exactly those. `like`: a field on the even sites;
    /// `coarse_like`: a coarse field as the coarse systems are solved.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plain: (&'a O, &'a O),
        outer: (&'a O, &'a O),
        form: EoForm,
        transfer: T,
        (coarse_d, coarse_pv): (D, D),
        (blocks_d, blocks_pv): (Box<dyn LinearFn<C> + 'a>, Box<dyn LinearFn<C> + 'a>),
        (coarse_d_adj, coarse_pv_adj): (Option<Adjoint<'a, C>>, Option<Adjoint<'a, C>>),
        params: PvEoParams,
        like: &V,
        coarse_like: &C,
    ) -> Self {
        assert!(
            params.d_solver != CoarseKrylov::Minres,
            "MINRES needs a Hermitian operator: not for the quark factor"
        );
        for (name, solve, adjoint) in [
            ("first", params.d_solver, coarse_d_adj.is_some()),
            ("second", params.pv_solver, coarse_pv_adj.is_some()),
        ] {
            assert_eq!(
                solve.regularized().is_some(),
                adjoint,
                "the {name} coarse factor: its adjoint goes with the regularized solve, and only with it"
            );
        }
        crate::env::PV_TRANSFER.with(|f| f.set(true));
        let g_pv = transfer.coarse_field(1);
        crate::env::PV_TRANSFER.with(|f| f.set(false));
        Self {
            g_pv,
            gc_pv: coarse_like.zeros_with(1),
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
            coarse_d_adj,
            coarse_pv_adj,
            params,
            h: like.zeros_with(1),
            gc: coarse_like.zeros_with(1),
            w: coarse_like.zeros_with(1),
            c: coarse_like.zeros_with(1),
            coarse_applications: Cell::new(0),
            low: None,
        }
    }

    /// exp/mr-coarse: c = D_c⁻¹ g for the quark coarse operator, by the solver of the cycle
    /// (its preconditioners and levels) to |r|²/|b|² < `tolerance_sqr`.
    pub(crate) fn solve_quark_coarse(&mut self, c: &mut C, g: &C, tolerance_sqr: f64, max: usize) {
        let stop = Stop::simple(tolerance_sqr, max);
        let d_solver = self
            .params
            .d_solver
            .krylov()
            .expect("a Krylov solver for the quark factor");
        c.set_zero();
        solve_shared(
            &self.coarse_d,
            Some(&mut *self.blocks_d),
            self.levels_d.as_deref(),
            &self.coarse_applications,
            &mut |a, m, x, b| coarse_solve(d_solver, stop, a, m, x, b),
            c,
            g,
        );
    }

    /// exp/mr-coarse (RUTTICE_MR_K=k): the k smallest singular triplets D_c v_j = σ_j u_j of
    /// the quark coarse operator, by subspace iteration with D_c⁻¹ on the 2k vectors `x` (`iters`
    /// rounds, each a solve of the cycle's coarse solver to |r|²/|b|² < `tol`, then
    /// Gram–Schmidt), and the Rayleigh–Ritz step for D_c†D_c on their span. Returns V, U (D_c V
    /// made orthonormal), the σ_j and the coarse-operator applications it took.
    pub(crate) fn quark_low_singular(
        &mut self,
        mut x: C,
        k: usize,
        iters: usize,
        tol: f64,
    ) -> (C, C, Vec<f64>, usize) {
        let m = x.num_vectors();
        assert!(m >= k);
        let n0 = self.coarse_applications();
        orthonormalize(&mut x);
        for _ in 0..iters {
            let mut y = x.zeros_like();
            self.solve_quark_coarse(&mut y, &x, tol, 5000);
            orthonormalize(&mut y);
            x = y;
        }
        let mut b = x.zeros_like();
        self.coarse_d.mult(&mut b, &x);
        let mut gram = vec![C64::new(0.0, 0.0); m * m];
        for i in 0..m {
            let bi = b.select(&[i]);
            for j in 0..m {
                let mut d = [C64::new(0.0, 0.0)];
                bi.dot(&b.select(&[j]), &mut d);
                gram[i * m + j] = d[0];
            }
        }
        let (lam, w) = crate::numerics::hermitian_eigen(m, &gram);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| lam[a].total_cmp(&lam[b]));
        let mut v = x.zeros_with(k);
        for (j, &c) in order.iter().take(k).enumerate() {
            let mut vj = x.zeros_with(1);
            for l in 0..m {
                vj.axpy(&[w[l * m + c]], &x.select(&[l]));
            }
            v.assign(&[j], &vj);
        }
        let mut u = v.zeros_like();
        self.coarse_d.mult(&mut u, &v);
        let mut nu = vec![0.0; k];
        u.norm_sqr(&mut nu);
        let sigma: Vec<f64> = nu.iter().map(|x| x.sqrt()).collect();
        orthonormalize(&mut u);
        (v, u, sigma, self.coarse_applications() - n0)
    }

    /// exp/mr-coarse: solve the low directions of the quark factor by minimal residual.
    pub(crate) fn set_low(&mut self, low: LowModes<'a, V, C>) {
        self.low = Some(low);
    }

    pub fn params(&self) -> &PvEoParams {
        &self.params
    }

    /// exp/ls16-diag (RUTTICE_SPLIT): the coarse fields of the Pauli-Villars step like `c`
    /// (the dimension of P_v).
    pub(crate) fn set_pv_like(&mut self, c: &C) {
        self.gc_pv = c.zeros_with(1);
        self.w = c.zeros_with(1);
    }

    /// Levels below the coarse lattice for the solves of the quark factor: an approximate
    /// inverse of P†DP on the whole coarse lattice.
    pub(crate) fn set_levels_d(&mut self, levels: Box<Levels<'a, C>>) {
        self.levels_d = Some(levels);
    }

    /// The same for the second factor, solved by MINRES: `levels(out, inp)` adds to `out`
    /// what the levels give of `inp`, Hermitian positive.
    pub(crate) fn set_levels_pv(&mut self, levels: Box<Levels<'a, C>>) {
        assert!(
            self.params.pv_solver == CoarseKrylov::Minres,
            "levels for the second factor go with MINRES"
        );
        self.levels_pv = Some(levels);
    }

    /// exp/ls16-diag: after `correct`, the quark solution c in the fine precision (`g`), the
    /// same in the coarse precision, P, D and P†DP.
    #[cfg(feature = "cuda-bridge")]
    /// exp/mr-coarse: the quark operator that the quark factor coarsens.
    pub(crate) fn plain_op(&self) -> &'a O {
        self.plain.0
    }

    pub(crate) fn diag_parts(&self) -> (&G, &C, &T, &O, &D) {
        (
            &self.g,
            &self.c,
            &self.transfer,
            self.plain.0,
            &self.coarse_d,
        )
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
            coarse_pv_adj,
            coarse_d_adj,
            params,
            work: (work, work_a),
            h,
            g,
            gc,
            w,
            c,
            g_pv,
            gc_pv,
            coarse_applications: n,
            low,
        } = self;
        fit(h, r);
        if g.num_vectors() != nv {
            *g = transfer.coarse_field(nv);
            for f in [&mut *gc, &mut *w, &mut *c, &mut *gc_pv] {
                *f = f.zeros_with(nv);
            }
            *g_pv = g_pv.zeros_with(nv);
        }
        let (even, coarse_pv, n) = (Parity::Even, &*coarse_pv, &*n);
        let stop = Stop::simple(params.coarse_tolerance_sqr, params.coarse_max_iterations);
        let pv_stop = Stop::simple(params.pv_tolerance_sqr, params.pv_max_iterations);

        // h = O_R r, g = P† ι h.
        match form {
            EoForm::Right => pv.diag(work, h, r, even, Dagger::Yes),
            EoForm::Left => pv_a.mult_q(h, r, true),
        }
        // exp/ls16-diag (RUTTICE_DSYM_PV): h = B_PV,e⁻† h first (site-local, so it commutes
        // with ι); the coarse system is then σ3 B σ3 w = g, solved as B (σ3 w) = σ3 g.
        let dsym_pv = pv.dsym_pv_left(e, h);
        if dsym_pv {
            h.copy_from(e);
        }
        // exp/ls16-diag (RUTTICE_SPLIT): the Pauli-Villars step restricts and prolongs with P_v.
        crate::env::PV_TRANSFER.with(|f| f.set(true));
        transfer.restrict(g_pv, h);
        crate::env::PV_TRANSFER.with(|f| f.set(false));
        gc_pv.convert_from(g_pv);
        if dsym_pv {
            coarse_pv.gamma5(gc_pv);
        }

        // RUTTICE_DIAG=1 probe (set by the two-grid cycle): only the share of O_R r in range P.
        if DIAG_PROBE.with(|p| p.get()) {
            let mut back = h.zeros_like();
            crate::env::PV_TRANSFER.with(|f| f.set(true));
            transfer.prolong(&mut back, g_pv);
            crate::env::PV_TRANSFER.with(|f| f.set(false));
            let (mut pb, mut hh) = (vec![0.0; nv], vec![0.0; nv]);
            back.norm_sqr(&mut pb);
            h.norm_sqr(&mut hh);
            DIAG_SHARE.with(|s| s.set(pb[0] / hh[0]));
            e.set_zero();
            return;
        }

        // RUTTICE_DIAG=1 (exp/ls16-diag): per correction, the coarse applications of each solve
        // and |b − A x|²/|b|² of its system, at the cost of one more application of each; and
        // |π P P† ι h|²/|h|², the share of the residual (after O_R) that range P holds.
        let diag = crate::env::switch("RUTTICE_DIAG");
        let in_range = diag.then(|| {
            let mut back = h.zeros_like();
            crate::env::PV_TRANSFER.with(|f| f.set(true));
            transfer.prolong(&mut back, g_pv);
            crate::env::PV_TRANSFER.with(|f| f.set(false));
            let (mut pb, mut hh) = (vec![0.0; nv], vec![0.0; nv]);
            back.norm_sqr(&mut pb);
            h.norm_sqr(&mut hh);
            pb.iter()
                .zip(&hh)
                .fold(f64::INFINITY, |m, (p, h)| m.min(p / h))
        });
        let n_start = n.get();
        // w = (P†D_PV†P)⁻¹ g: by CG on the regularized normal equations of the exact adjoint
        // (`coarse_pv`), by MINRES, w = Γ5 u with P†D_PV P u = Γ5 g (P†D_PV P taken as its own
        // Γ5-adjoint), or by a Krylov solver on the exact adjoint. The last two preconditioned by
        // the blocks, and the levels below if there are any.
        let factor = DIAG_FACTOR.with(|f| f.get());
        if factor == 1 {
            // exp/ls16-diag: the quark factor alone, w = g.
            w.copy_from(gc_pv);
        } else {
            w.set_zero();
            if let (FactorSolve::Regularized { eps }, Some(adj)) = (params.pv_solver, coarse_pv_adj)
            {
                regularized_solve(coarse_pv, &**adj, eps, pv_stop, n, w, gc_pv);
            } else if params.pv_solver == CoarseKrylov::Minres {
                coarse_pv.gamma5(gc_pv);
                minres_on_hermitian(
                    coarse_pv,
                    &mut **blocks_pv,
                    levels_pv.as_deref(),
                    n,
                    pv_stop,
                    w,
                    gc_pv,
                );
                coarse_pv.gamma5(w);
            } else {
                let pv_solver = params.pv_solver.krylov().expect("a Krylov solver");
                solve_shared(
                    coarse_pv,
                    Some(&mut **blocks_pv),
                    levels_pv.as_deref(),
                    n,
                    &mut |a, m, x, b| coarse_solve(pv_solver, pv_stop, a, m, x, b),
                    w,
                    gc_pv,
                );
            }
        }

        let (n_pv, pv_rel) = (
            n.get() - n_start,
            diag.then(|| diag_residual(|o, i| coarse_pv.mult(o, i), w, gc_pv)),
        );
        let pv_gain = diag.then(|| norm_ratio(w, gc_pv));
        if dsym_pv {
            coarse_pv.gamma5(w);
        }
        if factor == 2 {
            // exp/ls16-diag: the Pauli-Villars factor alone, c = w.
            c.copy_from(w);
        } else {
            // g = P† ι M π P w; M = 1 (right form) or C_e^(α) C_PV,e^(α)† (left form); with
            // RUTTICE_DSYM_PV π B⁻¹ P w = B_e⁻¹ π P w in place of π P w.
            g_pv.convert_from(w);
            crate::env::PV_TRANSFER.with(|f| f.set(true));
            transfer.prolong(h, g_pv);
            crate::env::PV_TRANSFER.with(|f| f.set(false));
            if dsym_pv && op.dsym_right(e, h) {
                h.copy_from(e);
            }
            if *form == EoForm::Left {
                pv_a.diag(work_a, e, h, even, Dagger::Yes);
                op_a.diag(work_a, h, e, even, Dagger::No);
            }
            if op.dl_left(e, h) {
                h.copy_from(e);
            }
            transfer.restrict(g, h);
            gc.convert_from(g);

            // exp/mr-coarse: g ← (1 − U_k U_k†) g before the solve.
            if let Some(lm) = low.as_ref() {
                low_project(gc, &lm.u, &lm.u);
            }
            // c = (P†DP)⁻¹ g, or regularized as the second factor.
            c.set_zero();
            if let (FactorSolve::Regularized { eps }, Some(adj)) = (params.d_solver, coarse_d_adj) {
                regularized_solve(&*coarse_d, &**adj, eps, stop, n, c, gc);
            } else {
                let d_solver = params.d_solver.krylov().expect("a Krylov solver");
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
        }
        // exp/mr-coarse: c ← (1 − V_k V_k†) c + V_k α, α = (Z†Z)⁻¹ (Z† ι h − G† c), with h the
        // field on the even sites that g was restricted from.
        if let (Some(lm), true) = (low.as_ref(), factor != 2) {
            low_project(c, &lm.v, &lm.v);
            low_minres(c, h, lm);
        }
        if let Some(pv_rel) = pv_rel {
            // Γ5-Hermiticity of the coarse quark operator A: <c, Γ5 A g> against <Γ5 A c, g>.
            let (mut ag, mut ac) = (c.zeros_like(), c.zeros_like());
            coarse_d.mult(&mut ag, gc);
            coarse_d.gamma5(&mut ag);
            coarse_d.mult(&mut ac, c);
            coarse_d.gamma5(&mut ac);
            let (mut x1, mut x2) = (
                vec![crate::C64::new(0.0, 0.0); nv],
                vec![crate::C64::new(0.0, 0.0); nv],
            );
            c.dot(&ag, &mut x1);
            ac.dot(gc, &mut x2);
            eprintln!(
                "diag hermiticity of Γ5 P†DP: |<c,Γ5Ag> − <Γ5Ac,g>|/|<c,Γ5Ag>| {:.3e}",
                ((x1[0] - x2[0]).norm_sqr() / x1[0].norm_sqr()).sqrt()
            );
            let d_rel = diag_residual(|o, i| coarse_d.mult(o, i), c, gc);
            eprintln!(
                "diag gain: pv |w|/|g| {:.3e}; d |c|/|g| {:.3e}",
                pv_gain.unwrap_or(f64::NAN),
                norm_ratio(c, gc)
            );
            eprintln!(
                "diag coarse: pv {n_pv} applications |r|^2/|b|^2 {pv_rel:.2e}; d {} applications |r|^2/|b|^2 {d_rel:.2e}; in range P {:.3e}",
                n.get() - n_start - n_pv,
                in_range.unwrap_or(f64::NAN)
            );
        }
        // e = O_L π P c: C_e (right form) or Q_e⁻¹ (left form); with RUTTICE_DSYM
        // e = O_L B_e⁻¹ π P c.
        g.convert_from(c);
        transfer.prolong(h, g);
        if op.dsym_right(e, h) {
            h.copy_from(e);
        }
        match form {
            EoForm::Right => op.diag(work, e, h, even, Dagger::No),
            EoForm::Left => op_a.mult_q(e, h, true),
        }
    }
}

/// exp/mr-coarse: x_r ← x_r − Σ_j a_j ⟨b_j, x_r⟩ for each vector x_r of `x` (a, b: k vectors).
fn low_project<C: Vector>(x: &mut C, a: &C, b: &C) {
    let k = a.num_vectors();
    for r in 0..x.num_vectors() {
        let mut xr = x.select(&[r]);
        for j in 0..k {
            let mut d = [crate::C64::new(0.0, 0.0)];
            b.select(&[j]).dot(&xr, &mut d);
            xr.axpy(&[-d[0]], &a.select(&[j]));
        }
        x.assign(&[r], &xr);
    }
}

/// exp/mr-coarse: c_r ← c_r + Σ_j α_j v_j, α = (Z†Z)⁻¹ (Z† ι h_r − G† c_r).
fn low_minres<V: Vector, C: Vector>(c: &mut C, h: &V, lm: &LowModes<'_, V, C>) {
    let k = lm.v.num_vectors();
    for r in 0..c.num_vectors() {
        let mut cr = c.select(&[r]);
        let w = (lm.adj)(&h.select(&[r]));
        let mut rhs = vec![crate::C64::new(0.0, 0.0); k];
        for (j, x) in rhs.iter_mut().enumerate() {
            let (mut zf, mut gy) = ([crate::C64::new(0.0, 0.0)], [crate::C64::new(0.0, 0.0)]);
            lm.v.select(&[j]).dot(&w, &mut zf);
            lm.g.select(&[j]).dot(&cr, &mut gy);
            *x = zf[0] - gy[0];
        }
        for j in 0..k {
            let a = (0..k).fold(crate::C64::new(0.0, 0.0), |a, l| {
                a + lm.ztz_inv[j * k + l] * rhs[l]
            });
            cr.axpy(&[a], &lm.v.select(&[j]));
        }
        c.assign(&[r], &cr);
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

/// exp/mr-coarse: the start of the search of `Correction::quark_low_singular`, `m` vectors of
/// `n` complex numbers one after the other, the same on every backend.
pub(crate) fn mr_start(n: usize, m: usize) -> Vec<C64> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rnd = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    (0..m * n).map(|_| C64::new(rnd(), rnd())).collect()
}

/// Gram–Schmidt, twice, on the vectors of `x`.
pub(crate) fn orthonormalize<C: Vector>(x: &mut C) {
    for _ in 0..2 {
        for j in 0..x.num_vectors() {
            let mut xj = x.select(&[j]);
            for l in 0..j {
                let xl = x.select(&[l]);
                let mut d = [C64::new(0.0, 0.0)];
                xl.dot(&xj, &mut d);
                xj.axpy(&[C64::new(-d[0].re, -d[0].im)], &xl);
            }
            let mut nn = [0.0];
            xj.norm_sqr(&mut nn);
            xj.scale(&[1.0 / nn[0].sqrt()]);
            x.assign(&[j], &xj);
        }
    }
}
