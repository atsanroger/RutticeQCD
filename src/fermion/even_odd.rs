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
        self.op
            .hop(&mut self.work, &mut self.other_a, inp, q, dagger);
        self.op
            .diag_inv(&mut self.work, &mut self.other_b, &self.other_a, q, dagger);
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

type HalfField<T, K, L> = Field<T, K, Half, L, Cpu>;

/// Solve `M x = b` on the full lattice through the Schur complement on the even sites:
///
/// 1. `b' = b_e - M_eo M_oo^{-1} b_o`
/// 2. `S x_e = b'`, done by `solve`
/// 3. `x_o = M_oo^{-1} (b_o - M_oe x_e)`
///
/// `solve` receives the Schur complement, the unknown (zero on entry) and the right-hand
/// side, and may use any solver. Its result is passed through.
///
/// `index` is the decomposition the operator uses (its `even_odd_index`): of the lattice
/// for the usual layout, of the lattice of blocks for the SIMD layout.
pub fn solve_even_odd<T, K, L, F, R, E>(
    op: &F,
    index: &EvenOddIndex,
    x: &mut Field<T, K, crate::Lexical, L>,
    b: &Field<T, K, crate::Lexical, L>,
    solve: impl FnOnce(
        &mut Schur<'_, F, HalfField<T, K, L>>,
        &mut HalfField<T, K, L>,
        &HalfField<T, K, L>,
    ) -> Result<R, E>,
) -> Result<R, E>
where
    T: Storage,
    K: Kind,
    L: Layout,
    F: EvenOdd<HalfField<T, K, L>>,
{
    let half = |parity: Parity| {
        let mut f = HalfField::<T, K, L>::with_shape(
            b.lattice(),
            b.internal_dof(),
            b.ls(),
            Field::num_vectors(b),
            b.outer(),
        );
        f.set_parity(parity);
        f
    };
    let (even, odd) = (Parity::Even, Parity::Odd);
    let minus_one = vec![C64::new(-1.0, 0.0); Field::num_vectors(b)];
    let mut work = op.even_odd_work();

    let (mut b_e, mut b_o) = (half(even), half(odd));
    extract_parity(&mut b_e, b, index, even);
    extract_parity(&mut b_o, b, index, odd);

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

    insert_parity(x, &x_e, index);
    insert_parity(x, &x_o, index);
    Ok(result)
}

/// An approximate inverse of the Schur complement S on one parity from one of the whole
/// operator: the block of M⁻¹ on that parity is exactly S⁻¹, so with an approximate inverse
/// `inner` of M, `[inner (r ⊕ 0)]_p` approximates S⁻¹ r as well as `inner` approximates
/// M⁻¹. This puts a preconditioner of the whole lattice (a multigrid cycle) under a solver
/// on one parity (`solve_even_odd`).
pub struct OnParity<'m, T: Storage, K: Kind, L: Layout> {
    inner: &'m mut dyn LinearFn<Field<T, K, crate::Lexical, L>>,
    index: &'m EvenOddIndex,
    inp: Field<T, K, crate::Lexical, L>,
    out: Field<T, K, crate::Lexical, L>,
}

impl<'m, T: Storage, K: Kind, L: Layout> OnParity<'m, T, K, L> {
    /// `like` gives the shape of the whole fields; `index` is the operator's decomposition.
    pub fn new(
        inner: &'m mut dyn LinearFn<Field<T, K, crate::Lexical, L>>,
        index: &'m EvenOddIndex,
        like: &Field<T, K, crate::Lexical, L>,
    ) -> Self {
        Self {
            inner,
            index,
            inp: like.zeros_like(),
            out: like.zeros_like(),
        }
    }
}

impl<T: Storage, K: Kind, L: Layout> LinearFn<HalfField<T, K, L>> for OnParity<'_, T, K, L> {
    fn mult(&mut self, out: &mut HalfField<T, K, L>, inp: &HalfField<T, K, L>) {
        let n = Field::num_vectors(inp);
        if Field::num_vectors(&self.inp) != n {
            self.inp = self.inp.zeros_with(n);
            self.out = self.out.zeros_with(n);
        }
        self.inp.set_zero();
        insert_parity(&mut self.inp, inp, self.index);
        self.inner.mult(&mut self.out, &self.inp);
        extract_parity(out, &self.out, self.index, inp.parity());
    }
}
