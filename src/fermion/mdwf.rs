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
//! A site of the field holds its Ls spinors next to each other (`Field::ls`), so the
//! four-dimensional part is the Wilson sweep over Ls times as many spinors per site, with
//! the links read once per site. The fifth dimension is local to a site. So far the
//! operator is put together from three passes (the combination before W, the Wilson sweep,
//! the combination after it); a sweep that does all three at a site is not written yet.
//!
//! Solvers: BiCGStab or FGMRES on the Pauli-Villars preconditioned operator; CG on the
//! normal equations. Tools needed: the inverse of the site-local part (not here yet).
//!
//! The normalisation and the placement of the mass term follow Bridge++
//! (`Fopr_Domainwall`), so that results can be compared. Written from the formulas above.

use super::{FermionOp, GaugeError, Variant, Wilson, WilsonParams, WilsonWork};
use crate::kernel::{Dirac, GammaRepre, Number, Spinor as SiteSpinor, NC, SPINOR_REALS};
use crate::{for_sites, Boundary, Field, Gauge, Lattice, Spinor, Storage, Vector, ND};
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

/// Largest fifth dimension: the spinors of a site are gathered on the stack.
pub const MAX_LS: usize = 64;

/// The Möbius domain-wall operator on fields with `ls` spinors per site.
pub struct Mdwf<T: Storage, G: GammaRepre = Dirac> {
    params: MdwfParams,
    wilson: Wilson<T, G>,
}

/// Scratch space of `Mdwf`: that of the Wilson kernel and three fields, made on first use.
pub struct MdwfWork<T: Storage> {
    wilson: WilsonWork<T>,
    fields: Option<[Field<T, Spinor>; 3]>,
}

impl<T: Storage, G: GammaRepre> Mdwf<T, G> {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams) -> Self {
        assert!(
            (2..=MAX_LS).contains(&params.ls),
            "the fifth dimension must have between 2 and {MAX_LS} sites"
        );
        let kernel = WilsonParams {
            kappa: 1.0 / (8.0 - 2.0 * params.m5),
            boundary: params.boundary,
        };
        Self {
            params,
            wilson: Wilson::new(lat, kernel),
        }
    }

    pub fn params(&self) -> &MdwfParams {
        &self.params
    }

    /// `(Y φ)_s`, or `(Y† φ)_s`, from the `ls` spinors of one site.
    #[inline(always)]
    fn fifth(
        &self,
        phi: &[SiteSpinor<T::Compute>],
        s: usize,
        dagger: bool,
    ) -> SiteSpinor<T::Compute> {
        let ls = phi.len();
        let number = <T::Compute as Number>::from_f64;
        // The coefficient of the projected neighbour: -1, or +m across the walls.
        let across = |wall: bool| number(0.5 * if wall { self.params.mass } else { -1.0 });
        // Y takes P- from above and P+ from below; Y† the other way round.
        let (above, below) = if dagger { (1.0, -1.0) } else { (-1.0, 1.0) };
        let projected = |from: usize, sign: f64, k: T::Compute| {
            // k (1 + sign γ5) φ
            let p = &phi[from];
            let both = p.sub_scaled(number(-sign), &G::GAMMA5.apply(p));
            SiteSpinor::zero().sub_scaled(-k, &both)
        };
        projected((s + 1) % ls, above, across(s == ls - 1)).add(&projected(
            (s + ls - 1) % ls,
            below,
            across(s == 0),
        ))
    }

    /// `out_s = Σ_k a_k x_k,s + Σ_k y_k (Y x_k)_s` for the spinors of every site and vector:
    /// each `(field, a, y)` of `terms` contributes its spinor times `a` and its hop in the
    /// fifth dimension (Y, or Y† if `dagger`) times `y`. With `reflect`, the result at s is
    /// γ5 times what would be at Ls-1-s: Γ5 applied last.
    fn combine(
        &self,
        out: &mut Field<T, Spinor>,
        terms: &[(&Field<T, Spinor>, f64, f64)],
        dagger: bool,
        reflect: bool,
    ) {
        let ls = self.params.ls;
        let number = <T::Compute as Number>::from_f64;
        let site_len = out.site_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            // One vector at a time: its `ls` spinors are next to each other.
            for (v, o) in o.chunks_exact_mut(ls * SPINOR_REALS).enumerate() {
                let mut result = [SiteSpinor::<T::Compute>::zero(); MAX_LS];
                for (field, a, y) in terms {
                    let from =
                        &field.site(0, x)[v * ls * SPINOR_REALS..(v + 1) * ls * SPINOR_REALS];
                    let mut phi = [SiteSpinor::<T::Compute>::zero(); MAX_LS];
                    for (p, reals) in phi.iter_mut().zip(from.as_chunks::<SPINOR_REALS>().0) {
                        *p = load::<T>(reals);
                    }
                    for s in 0..ls {
                        let mut term = SiteSpinor::zero().sub_scaled(number(-*a), &phi[s]);
                        if *y != 0.0 {
                            let hop = self.fifth(&phi[..ls], s, dagger);
                            term = term.sub_scaled(number(-*y), &hop);
                        }
                        result[s] = result[s].add(&term);
                    }
                }
                for (s, reals) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                    if reflect {
                        store::<T>(&G::GAMMA5.apply(&result[ls - 1 - s]), reals);
                    } else {
                        store::<T>(&result[s], reals);
                    }
                }
            }
        });
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

impl<T: Storage, G: GammaRepre> FermionOp<Field<T, Spinor>> for Mdwf<T, G> {
    type Work = MdwfWork<T>;

    fn work(&self) -> MdwfWork<T> {
        MdwfWork {
            wilson: self.wilson.work(),
            fields: None,
        }
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        self.wilson.set_gauge(u)
    }

    /// The count Bridge++ uses for its domain-wall operator, `(464 Nc + 240) Ls`.
    fn flops_per_site(&self) -> usize {
        (464 * NC + 240) * self.params.ls
    }

    fn mult(
        &self,
        work: &mut MdwfWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
        variant: Variant,
    ) {
        assert_eq!(inp.ls(), self.params.ls, "the field has another Ls");
        assert!(out.same_shape(inp));
        if !work.fields.as_ref().is_some_and(|f| f[0].same_shape(inp)) {
            work.fields = Some([inp.zeros_like(), inp.zeros_like(), inp.zeros_like()]);
        }
        let Some([w, v, d]) = work.fields.as_mut() else {
            unreachable!("the fields were just made");
        };
        let MdwfParams { m5, b, c, .. } = self.params;
        let height = 4.0 - m5;
        match variant {
            // D ψ = ψ + Y ψ + W (b ψ - c Y ψ)
            Variant::D | Variant::Gamma5D => {
                self.combine(w, &[(inp, b, -c)], false, false);
                self.wilson.mult(&mut work.wilson, v, w, Variant::D);
                let reflect = variant == Variant::Gamma5D;
                // Γ5 goes on last, so it needs D ψ whole before it is reflected.
                let to = if reflect { &mut *d } else { &mut *out };
                self.combine(to, &[(inp, 1.0, 1.0), (v, height, 0.0)], false, false);
                if reflect {
                    self.combine(out, &[(d, 1.0, 0.0)], false, true);
                }
            }
            // D† ψ = ψ + Y† ψ + (b - c Y†) W† ψ
            Variant::Dagger => {
                self.wilson.mult(&mut work.wilson, v, inp, Variant::Dagger);
                self.combine(
                    out,
                    &[(inp, 1.0, 1.0), (v, height * b, -height * c)],
                    true,
                    false,
                );
            }
        }
    }
}
