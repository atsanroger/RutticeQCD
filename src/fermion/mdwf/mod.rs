//! Möbius domain-wall fermions (MDWF).
//!
//! The field has a fifth dimension of extent Ls; ψ_s, s = 0 .. Ls-1, are four-dimensional
//! spinor fields. With P± = (1 ± γ5)/2 and the hop in the fifth dimension
//!
//!   (Y ψ)_s = - P- ψ_{s+1} - P+ ψ_{s-1},
//!
//! in which a term that leaves the interval comes back from the other wall multiplied by
//! -m: (Y ψ)_{Ls-1} has + m P- ψ_0 and (Y ψ)_0 has + m P+ ψ_{Ls-1},
//!
//! the operator is
//!
//!   D ψ = ψ + Y ψ + W (b ψ - c Y ψ),        W = (4 - M5) Dw,
//!
//! where Dw = 1 - κ Σ hops is the Wilson operator with κ = 1 / (8 - 2 M5), so that W is the
//! Wilson operator with mass -M5 in the normalisation with unit hopping coefficient 1/2.
//! Written out, D ψ_s = (b W + 1) ψ_s + (c W - 1) [P- ψ_{s+1} + P+ ψ_{s-1}]. b = 1, c = 0
//! is the domain-wall operator of Shamir. The Pauli-Villars operator is D at m = 1.
//!
//! Provides: `mult` for D, D† and Γ5 D with Γ5 = γ5 R, R the reflection s -> Ls-1-s. For
//! c = 0, Γ5 D is Hermitian; for c ≠ 0 it is not (W does not commute with P±).
//!
//! The part of D that stays within a four-dimensional site is
//!
//!   C = d+ + d- Y,    d+ = 1 + b (4 - M5),    d- = 1 - c (4 - M5),
//!
//! an Ls x Ls matrix in the fifth dimension for each chirality: P+ ψ only sees its
//! neighbour below and P- ψ its neighbour above, plus the mass term in one corner.
//! `mult_site` applies C and `mult_site_inv` its inverse, by substitution along s with the
//! corner taken out first, so the inverse is exact and costs a few operations per s.
//!
//! `PvPreconditioned` is the operator
//!
//!   A = (D_PV C_PV^-1)† D C^-1 = X† Y,    X = D_PV C_PV^-1,  Y = D C^-1,
//!
//! with D_PV and C_PV the same at m = 1 (Pauli-Villars). D alone has eigenvalues on both
//! sides of the imaginary axis and Krylov solvers for general matrices stall on it; the
//! factor in front turns the spectrum into the right half plane (Brower, Clark, Howarth,
//! Weinberg; Kanamori, Chen, Matsufuru, PoS(LATTICE2024)414, whose Fig. 4 is this A). C^-1
//! is a right preconditioner: solving A y = X† b and setting x = C^-1 y gives the solution
//! of D x = b. X and Y have the same form, so at m = 1 A is the Hermitian X† X.
//!
//! An earlier version had C^-1 on the left, A = X† C^-1 D, as the CUDA implementation and
//! the working note `pvprec_amg` do; that is not the paper's operator. On a 12^4 quenched
//! configuration (Ls = 8, M5 = 1, m = 0.005) BiCGStab needs about the same number of
//! iterations on both (4 % fewer on this one over five right-hand sides).
//!
//! A site of the field holds its Ls spinors next to each other (`Field::ls`), so the
//! four-dimensional part is the Wilson sweep over Ls times as many spinors per site, with
//! the links read once per site. The fifth dimension is local to a site. So far the
//! operator is put together from three passes (the combination before W, the Wilson sweep,
//! the combination after it); a sweep that does all three at a site is not written yet.
//! The even-odd blocks do have it: the hop takes (4 − M5)(b − c Y) into the sweep of the
//! Wilson kernel, and the Schur factors of `PvEvenOdd` also what follows a hop within a
//! site (C⁻¹, the difference with the input), so a factor is two or three passes over the
//! field. The kernels are limited by the memory they move, not by the arithmetic.
//! The fields may be in the usual layout or in blocks of W sites (`MdwfLayout`); the fifth
//! dimension is the same code on numbers of W lanes.
//!
//! Files: this one has the operator, its layouts and constructors; `fifth.rs` the fifth
//! dimension (Q, Y, C and C⁻¹); `full.rs` the operator on the whole lattice (D at any mass,
//! the parts for the multigrid, `PvPreconditioned`); `even_odd.rs` the even-odd blocks and
//! `PvEvenOdd`; `measure.rs` the five-dimensional source and the four-dimensional fields.
//!
//! Solvers: BiCGStab or FGMRES on `PvPreconditioned` (no breakdown, in double or single
//! precision); CG on the normal equations of D.
//! Tools needed: none beyond what is here.
//!
//! Neff's better conditioned form (arXiv:1501.04950; Chen, Kanamori, Matsufuru, Neff,
//! arXiv:2602.14454) is `set_alpha`: D^(α) = D Q with Q = P A P†, which works out diagonal
//! in the fifth dimension and local to a site,
//!
//!   Q_0 = P- + α P+,   Q_s = α (0 < s < Ls-1),   Q_(Ls-1) = P+ + α P-,
//!
//! so D^(α) has the blocks of eq. (13) of 2602.14454 (those of D multiplied by α except the
//! four corners). The four-dimensional solution is unchanged: x = Q x^(α), and Q is the
//! identity on P- x_0 and on P+ x_(Ls-1). With α = 1 (the default) Q is skipped. Every
//! part of D below carries Q: C^(α) = C Q, the even-odd blocks, the hops of `HopTerms`.
//! Right-preconditioned forms do not see α (D^(α) C^(α)⁻¹ = D C⁻¹, and the same for the
//! Schur complements): it has to be used with left preconditioning to make a difference.
//! (2602.14454 writes D (P A⁻¹ P†) once in its even-odd section; eq. (14), D^(α) P = D P A,
//! and the blocks of eq. (13) give P A P†.)
//!
//! The normalisation and the placement of the mass term follow Bridge++
//! (`Fopr_Domainwall`), so that results can be compared. Written from the formulas above.

use super::{
    fifth_hop, After, Dagger, EvenOdd, FermionOp, Fifth, GaugeError, HopTerms, UnitDiag, Variant,
    Wilson, WilsonLayout, WilsonOp, WilsonParams, WilsonWork,
};
use crate::kernel::{
    Cplx, Dirac, GammaRepre, Lane, LaneOf, Number, Spinor as SiteSpinor, NC, NS, SPINOR_REALS,
};
use crate::{
    extract_parity, fit, for_sites, insert_parity, Aos, Aosoa, Boundary, EvenOddIndex, Field, Full,
    Gauge, Half, Lattice, Layout, LinearFn, LinearOp, Parity, Sites, Spinor, Storage, Vector, C64,
    ND,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MdwfParams {
    /// Extent of the fifth dimension.
    pub ls: usize,
    /// Height of the domain wall, the negative mass of the Wilson kernel.
    pub m5: f64,
    /// Quark mass m.
    pub mass: f64,
    /// Möbius coefficients; (1, 0) is Shamir's operator.
    pub b: f64,
    pub c: f64,
    pub boundary: [Boundary; ND],
}

impl MdwfParams {
    /// The parameters of the Pauli-Villars operator: the same operator at unit mass.
    pub fn pauli_villars(self) -> Self {
        Self { mass: 1.0, ..self }
    }
}

/// Largest fifth dimension.
pub const MAX_LS: usize = 64;

/// The Möbius domain-wall operator on fields with `ls` spinors per site, in the layout `L`:
/// the usual one (`new`), or blocks of `W` sites for SIMD (`Aosoa<W>`, `in_blocks`), in which
/// the same code runs on the `W` sites of a block at once.
pub struct Mdwf<T: Storage, G: GammaRepre = Dirac, L: MdwfLayout<T> = Aos> {
    params: MdwfParams,
    wilson: WilsonOp<T, G, UnitDiag, L>,
    /// Neff's α (1: the plain operator).
    alpha: f64,
}

/// Scratch space of `Mdwf`: that of the Wilson kernel and four fields, made on first use.
pub struct MdwfWork<T: Storage, L: Layout = Aos> {
    wilson: WilsonWork<T>,
    fields: Option<[Field<T, Spinor, Full, L>; 4]>,
}

/// A layout `Mdwf` works in: the Wilson kernel has a sweep for it, and the reals of one
/// spinor (of one site, or of the `W` sites of a block) load into a spinor of `Num`, the
/// compute type or `W` lanes of it, on which the fifth dimension is the same code.
pub trait MdwfLayout<T: Storage>: WilsonLayout<T> {
    type Num: Number;
    fn load(reals: &[T]) -> SiteSpinor<Self::Num>;
    fn store(p: &SiteSpinor<Self::Num>, reals: &mut [T]);
    /// The Wilson kernel in this layout, if the lattice allows it.
    fn kernel<G: GammaRepre>(
        lat: &Arc<Lattice>,
        params: WilsonParams,
    ) -> Option<WilsonOp<T, G, UnitDiag, Self>>;
    fn even_odd_index<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
    ) -> Option<&Arc<EvenOddIndex>>;
    fn work_of<G: GammaRepre>(w: &WilsonOp<T, G, UnitDiag, Self>) -> WilsonWork<T>;
    fn set_gauge<G: GammaRepre>(
        w: &mut WilsonOp<T, G, UnitDiag, Self>,
        u: &Field<f64, Gauge>,
    ) -> Result<(), GaugeError>;
    fn mult<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Full, Self>,
        inp: &Field<T, Spinor, Full, Self>,
        variant: Variant,
    );
    fn hop<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half, Self>,
        inp: &Field<T, Spinor, Half, Self>,
        to: Parity,
        dagger: Dagger,
    );
    /// The hop with the factor in the fifth dimension in the same sweep, and `after` on each
    /// site (block) of the result (`WilsonOp::even_odd_hop_fifth`).
    #[allow(clippy::too_many_arguments)]
    fn hop_fifth<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half, Self>,
        inp: &Field<T, Spinor, Half, Self>,
        to: Parity,
        dagger: Dagger,
        fifth: Fifth,
        after: After<'_, T>,
    );
}

impl<T: Storage> MdwfLayout<T> for Aos {
    type Num = T::Compute;
    #[inline(always)]
    fn load(reals: &[T]) -> SiteSpinor<T::Compute> {
        load::<T>(reals)
    }
    #[inline(always)]
    fn store(p: &SiteSpinor<T::Compute>, reals: &mut [T]) {
        store::<T>(p, reals);
    }
    fn kernel<G: GammaRepre>(
        lat: &Arc<Lattice>,
        params: WilsonParams,
    ) -> Option<WilsonOp<T, G, UnitDiag, Self>> {
        Some(Wilson::new(lat, params))
    }
    fn even_odd_index<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
    ) -> Option<&Arc<EvenOddIndex>> {
        w.even_odd_index()
    }
    fn work_of<G: GammaRepre>(w: &WilsonOp<T, G, UnitDiag, Self>) -> WilsonWork<T> {
        FermionOp::<Field<T, Spinor>>::work(w)
    }
    fn set_gauge<G: GammaRepre>(
        w: &mut WilsonOp<T, G, UnitDiag, Self>,
        u: &Field<f64, Gauge>,
    ) -> Result<(), GaugeError> {
        FermionOp::<Field<T, Spinor>>::set_gauge(w, u)
    }
    fn mult<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
        variant: Variant,
    ) {
        FermionOp::mult(w, work, out, inp, variant);
    }
    fn hop<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        to: Parity,
        dagger: Dagger,
    ) {
        EvenOdd::hop(w, work, out, inp, to, dagger);
    }
    #[allow(clippy::too_many_arguments)]
    fn hop_fifth<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        to: Parity,
        dagger: Dagger,
        fifth: Fifth,
        after: After<'_, T>,
    ) {
        w.even_odd_hop_fifth(work, out, inp, to, dagger, fifth, after);
    }
}

impl<T: Storage, const W: usize> MdwfLayout<T> for Aosoa<W>
where
    T::Compute: LaneOf<W>,
{
    type Num = <T::Compute as LaneOf<W>>::Lane;
    #[inline(always)]
    fn load(reals: &[T]) -> SiteSpinor<Self::Num> {
        let lanes = reals.as_chunks::<W>().0;
        let at = |k: usize| Lane::from_array(lanes[k].map(Storage::load));
        let mut p = SiteSpinor::zero();
        for a in 0..NS {
            for c in 0..NC {
                let k = 2 * (c + NC * a);
                p.0[a][c] = Cplx::new(at(k), at(k + 1));
            }
        }
        p
    }
    #[inline(always)]
    fn store(p: &SiteSpinor<Self::Num>, reals: &mut [T]) {
        let lanes = reals.as_chunks_mut::<W>().0;
        for a in 0..NS {
            for c in 0..NC {
                let k = 2 * (c + NC * a);
                lanes[k] = p.0[a][c].re.to_array().map(T::store);
                lanes[k + 1] = p.0[a][c].im.to_array().map(T::store);
            }
        }
    }
    fn kernel<G: GammaRepre>(
        lat: &Arc<Lattice>,
        params: WilsonParams,
    ) -> Option<WilsonOp<T, G, UnitDiag, Self>> {
        WilsonOp::<T, G, UnitDiag, Self>::new(lat, params)
    }
    fn even_odd_index<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
    ) -> Option<&Arc<EvenOddIndex>> {
        w.even_odd_index()
    }
    fn work_of<G: GammaRepre>(w: &WilsonOp<T, G, UnitDiag, Self>) -> WilsonWork<T> {
        FermionOp::<Field<T, Spinor, Full, Self>>::work(w)
    }
    fn set_gauge<G: GammaRepre>(
        w: &mut WilsonOp<T, G, UnitDiag, Self>,
        u: &Field<f64, Gauge>,
    ) -> Result<(), GaugeError> {
        FermionOp::<Field<T, Spinor, Full, Self>>::set_gauge(w, u)
    }
    fn mult<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Full, Self>,
        inp: &Field<T, Spinor, Full, Self>,
        variant: Variant,
    ) {
        FermionOp::mult(w, work, out, inp, variant);
    }
    fn hop<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half, Self>,
        inp: &Field<T, Spinor, Half, Self>,
        to: Parity,
        dagger: Dagger,
    ) {
        EvenOdd::hop(w, work, out, inp, to, dagger);
    }
    #[allow(clippy::too_many_arguments)]
    fn hop_fifth<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half, Self>,
        inp: &Field<T, Spinor, Half, Self>,
        to: Parity,
        dagger: Dagger,
        fifth: Fifth,
        after: After<'_, T>,
    ) {
        w.even_odd_hop_fifth(work, out, inp, to, dagger, fifth, after);
    }
}

/// A term of `Mdwf::combine`: a field, its coefficient and that of its hop in the fifth
/// dimension.
type Term<'f, T, S, L> = (&'f Field<T, Spinor, S, L>, f64, f64);

/// The reals of block (or site, in the usual layout) `x` of a field, all its vectors.
#[inline(always)]
fn block<T: Storage, S: Sites, L: Layout>(f: &Field<T, Spinor, S, L>, x: usize) -> &[T] {
    let n = f.block_len();
    &f.slab(0)[x * n..(x + 1) * n]
}

impl<T: Storage, G: GammaRepre> Mdwf<T, G> {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams) -> Self {
        Self::in_layout(lat, params).expect("every lattice has the usual layout")
    }
}

impl<T: Storage, G: GammaRepre, const W: usize> Mdwf<T, G, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    /// The operator on fields in blocks of `W` sites, the SIMD layout (one process, as the
    /// Wilson kernel in blocks). `None` if the lattice cannot be cut into `W` sub-lattices
    /// of equal shape.
    pub fn in_blocks(lat: &Arc<Lattice>, params: MdwfParams) -> Option<Self> {
        Self::in_layout(lat, params)
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    /// The operator in the layout `L`; `None` if the lattice does not allow it (see
    /// `in_blocks`).
    pub fn in_layout(lat: &Arc<Lattice>, params: MdwfParams) -> Option<Self> {
        assert!(
            (2..=MAX_LS).contains(&params.ls),
            "the fifth dimension must have between 2 and {MAX_LS} sites"
        );
        let kernel = WilsonParams {
            kappa: 1.0 / (8.0 - 2.0 * params.m5),
            boundary: params.boundary,
        };
        Some(Self {
            params,
            wilson: L::kernel(lat, kernel)?,
            alpha: 1.0,
        })
    }

    pub fn params(&self) -> &MdwfParams {
        &self.params
    }

    /// Neff's form D^(α) = D Q (see the top of the file); 1 is the plain operator.
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(alpha > 0.0, "α must be positive");
        self.alpha = alpha;
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[inline(always)]
fn load<T: Storage>(reals: &[T]) -> SiteSpinor<T::Compute> {
    let mut p = SiteSpinor::zero();
    for a in 0..crate::kernel::NS {
        for c in 0..NC {
            let k = 2 * (c + NC * a);
            p.0[a][c] = crate::kernel::Cplx::new(reals[k].load(), reals[k + 1].load());
        }
    }
    p
}

#[inline(always)]
fn store<T: Storage>(p: &SiteSpinor<T::Compute>, reals: &mut [T]) {
    for a in 0..crate::kernel::NS {
        for c in 0..NC {
            let k = 2 * (c + NC * a);
            reals[k] = T::store(p.0[a][c].re);
            reals[k + 1] = T::store(p.0[a][c].im);
        }
    }
}

/// The four scratch fields of a work object, in the shape of `like`.
fn scratch<'w, T: Storage, L: Layout>(
    fields: &'w mut Option<[Field<T, Spinor, Full, L>; 4]>,
    like: &Field<T, Spinor, Full, L>,
) -> &'w mut [Field<T, Spinor, Full, L>; 4] {
    if !fields.as_ref().is_some_and(|f| f[0].same_shape(like)) {
        *fields = Some([
            like.zeros_like(),
            like.zeros_like(),
            like.zeros_like(),
            like.zeros_like(),
        ]);
    }
    fields.as_mut().expect("the fields were just made")
}

mod even_odd;
mod fifth;
mod full;
mod measure;

pub use even_odd::{EoForm, MdwfHalfWork, PvEvenOdd};
pub use full::{MdwfAt, PvPreconditioned, SiteInverse};
