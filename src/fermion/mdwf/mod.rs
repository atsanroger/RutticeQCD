//! Möbius domain-wall fermions (MDWF).
//!
//! The field has a fifth dimension of extent Ls; ψ_s, s = 0 .. Ls-1, are four-dimensional
//! spinor fields. With P± = (1 ± γ5)/2 and the hop in the fifth dimension
//!
//!   (Y ψ)_s = - P- ψ_{s+1} - P+ ψ_{s-1},
//!
//! the operator is
//!
//!   D ψ = ψ + Y ψ + W (b ψ - c Y ψ),        W = (4 - M5) Dw,
//!
//! where Dw = 1 - κ Σ hops is the Wilson operator with κ = 1 / (8 - 2 M5), so that W is the Wilson
//! operator with mass -M5 in the normalisation with unit hopping coefficient 1/2.
//!
//! The part of D that stays within a four-dimensional site is
//!
//!   C = d+ + d- Y,    d+ = 1 + b (4 - M5),    d- = 1 - c (4 - M5),
//!
//! `PvPreconditioned` is the operator
//!
//!   A = (D_PV C_PV^-1)† D C^-1 = X† Y,    X = D_PV C_PV^-1,  Y = D C^-1,
//!
//! with D_PV and C_PV the same at m = 1 (Pauli-Villars).
//!
//! Neff's better conditioned form (arXiv:1501.04950; Chen, Kanamori, Matsufuru, Neff,
//! arXiv:2602.14454) is `set_alpha`: D^(α) = D Q with Q = P A P†, which works out diagonal in the
//! fifth dimension and local to a site,
//!
//!   Q_0 = P- + α P+,   Q_s = α (0 < s < Ls-1),   Q_(Ls-1) = P+ + α P-,

use super::{
    fifth_hop, After, Dagger, EvenOdd, FermionOp, Fifth, GaugeError, HopTerms, UnitDiag, Variant,
    Wilson, WilsonLayout, WilsonOp, WilsonParams, WilsonWork,
};
use crate::kernel::{
    Cplx, Dirac, GammaRepre, Lane, LaneOf, Number, Spinor as SiteSpinor, NC, NS, SPINOR_REALS,
};
use crate::{
    extract_parity, fit, for_sites, insert_parity, Aos, Aosoa, Boundary, EvenOddIndex, Field,
    Gauge, Half, Lattice, Layout, Lexical, LinearFn, LinearOp, Parity, Sites, Spinor, Storage,
    Vector, C64, ND,
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
    fields: Option<[Field<T, Spinor, Lexical, L>; 4]>,
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
        out: &mut Field<T, Spinor, Lexical, Self>,
        inp: &Field<T, Spinor, Lexical, Self>,
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
    /// The hop with the factor in the fifth dimension on its input (`before`) and `after` on
    /// each site (block) of the result, in the same sweep (`WilsonOp::even_odd_hop_fifth`).
    #[allow(clippy::too_many_arguments)]
    fn hop_fifth<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half, Self>,
        inp: &Field<T, Spinor, Half, Self>,
        to: Parity,
        dagger: Dagger,
        before: Option<Fifth>,
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
        before: Option<Fifth>,
        after: After<'_, T>,
    ) {
        w.even_odd_hop_fifth(work, out, inp, to, dagger, before, after);
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
        FermionOp::<Field<T, Spinor, Lexical, Self>>::work(w)
    }
    fn set_gauge<G: GammaRepre>(
        w: &mut WilsonOp<T, G, UnitDiag, Self>,
        u: &Field<f64, Gauge>,
    ) -> Result<(), GaugeError> {
        FermionOp::<Field<T, Spinor, Lexical, Self>>::set_gauge(w, u)
    }
    fn mult<G: GammaRepre>(
        w: &WilsonOp<T, G, UnitDiag, Self>,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Lexical, Self>,
        inp: &Field<T, Spinor, Lexical, Self>,
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
        before: Option<Fifth>,
        after: After<'_, T>,
    ) {
        w.even_odd_hop_fifth(work, out, inp, to, dagger, before, after);
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

    /// The count Bridge++ uses for its domain-wall operator, `(464 Nc + 240) Ls`.
    pub fn flops_per_site(&self) -> usize {
        (464 * crate::kernel::NC + 240) * self.params.ls
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
    fields: &'w mut Option<[Field<T, Spinor, Lexical, L>; 4]>,
    like: &Field<T, Spinor, Lexical, L>,
) -> &'w mut [Field<T, Spinor, Lexical, L>; 4] {
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
mod lexical;
mod quark_4d;
mod site;

pub use even_odd::{MdwfHalfWork, PvEvenOdd};
pub use lexical::{MdwfAt, PvPreconditioned, SiteInverse};
