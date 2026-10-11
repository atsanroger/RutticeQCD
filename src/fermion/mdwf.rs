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
    Dagger, EvenOdd, FermionOp, GaugeError, HopTerms, Variant, Wilson, WilsonParams, WilsonWork,
};
use crate::kernel::{Dirac, GammaRepre, Number, Spinor as SiteSpinor, NC, SPINOR_REALS};
use crate::{
    fit, for_sites, Boundary, EvenOddIndex, Field, Gauge, Half, Lattice, LinearFn, LinearOp,
    Parity, Sites, Spinor, Storage, Vector, C64, ND,
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

/// Largest fifth dimension: the spinors of a site are gathered on the stack.
pub const MAX_LS: usize = 64;

/// The Möbius domain-wall operator on fields with `ls` spinors per site.
pub struct Mdwf<T: Storage, G: GammaRepre = Dirac> {
    params: MdwfParams,
    wilson: Wilson<T, G>,
    /// Neff's α (1: the plain operator).
    alpha: f64,
}

/// Scratch space of `Mdwf`: that of the Wilson kernel and four fields, made on first use.
pub struct MdwfWork<T: Storage> {
    wilson: WilsonWork<T>,
    fields: Option<[Field<T, Spinor>; 4]>,
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
            alpha: 1.0,
        }
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

    /// `out = Q inp`, or `Q⁻¹ inp` with `inverse`: Q_0 = P- + α P+, Q_s = α inside,
    /// Q_(Ls-1) = P+ + α P-. Q is Hermitian. The solution of D x = b is x = Q x^(α) for
    /// D^(α) x^(α) = b.
    pub fn mult_q<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        inp: &Field<T, Spinor, S>,
        inverse: bool,
    ) {
        assert!(inp.ls() == self.params.ls && out.same_shape(inp));
        let ls = self.params.ls;
        let a = if inverse {
            1.0 / self.alpha
        } else {
            self.alpha
        };
        let number = <T::Compute as Number>::from_f64;
        // P- + a P+ = (1 + a)/2 + (a - 1)/2 γ5; P+ + a P- = (1 + a)/2 + (1 - a)/2 γ5.
        let (even, odd) = (number(0.5 * (1.0 + a)), number(0.5 * (a - 1.0)));
        let inside = number(a);
        let site_len = out.site_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            for (v, o) in o.chunks_exact_mut(ls * SPINOR_REALS).enumerate() {
                let from = &inp.site(0, x)[v * ls * SPINOR_REALS..(v + 1) * ls * SPINOR_REALS];
                for (s, (out_reals, in_reals)) in o
                    .as_chunks_mut::<SPINOR_REALS>()
                    .0
                    .iter_mut()
                    .zip(from.as_chunks::<SPINOR_REALS>().0)
                    .enumerate()
                {
                    let p = load::<T>(in_reals);
                    let q = if s == 0 || s == ls - 1 {
                        let g = G::GAMMA5.apply(&p);
                        let k = if s == 0 { odd } else { -odd };
                        SiteSpinor::zero().sub_scaled(-even, &p).sub_scaled(-k, &g)
                    } else {
                        SiteSpinor::zero().sub_scaled(-inside, &p)
                    };
                    store::<T>(&q, out_reals);
                }
            }
        });
    }

    fn has_alpha(&self) -> bool {
        self.alpha != 1.0
    }

    /// `(Y φ)_s`, or `(Y† φ)_s`, at mass `mass`, from the `ls` spinors of one site.
    #[inline(always)]
    fn fifth(
        phi: &[SiteSpinor<T::Compute>],
        s: usize,
        dagger: bool,
        mass: f64,
    ) -> SiteSpinor<T::Compute> {
        let ls = phi.len();
        let number = <T::Compute as Number>::from_f64;
        // The coefficient of the projected neighbour: -1, or +m across the walls.
        let across = |wall: bool| number(0.5 * if wall { mass } else { -1.0 });
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
    /// γ5 times what would be at Ls-1-s: Γ5 applied last. Y is taken at mass `mass`.
    fn combine<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        terms: &[(&Field<T, Spinor, S>, f64, f64)],
        (dagger, reflect): (bool, bool),
        mass: f64,
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
                            let hop = Self::fifth(&phi[..ls], s, dagger, mass);
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

impl<T: Storage, G: GammaRepre> Mdwf<T, G> {
    /// The coefficients (d+, d-) of the site-local part C = d+ + d- Y.
    fn site_coefficients(&self) -> (f64, f64) {
        let height = 4.0 - self.params.m5;
        (1.0 + self.params.b * height, 1.0 - self.params.c * height)
    }

    /// `out = C inp`, or `C† inp`: the part of D (at mass `mass`) that stays within a
    /// four-dimensional site. The fields may be on the whole lattice or on one parity.
    pub fn mult_site<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        inp: &Field<T, Spinor, S>,
        mass: f64,
        dagger: bool,
    ) {
        assert!(inp.ls() == self.params.ls && out.same_shape(inp));
        let (d_plus, d_minus) = self.site_coefficients();
        if !self.has_alpha() {
            self.combine(out, &[(inp, d_plus, d_minus)], (dagger, false), mass);
            return;
        }
        // C^(α) = C Q and (C Q)† = Q C†.
        let mut t = inp.zeros_like();
        if dagger {
            self.combine(&mut t, &[(inp, d_plus, d_minus)], (true, false), mass);
            self.mult_q(out, &t, false);
        } else {
            self.mult_q(&mut t, inp, false);
            self.combine(out, &[(&t, d_plus, d_minus)], (false, false), mass);
        }
    }

    /// `out = C^-1 inp`, or `C†^-1 inp`, exactly.
    ///
    /// For one chirality C is `d+` on the diagonal, `-d-` next to it on one side and
    /// `m d-` in the far corner. Substitution along s gives every x_s as a known part plus
    /// a multiple of the x at the wall the corner refers to; the corner row then fixes
    /// that one. P+ runs upward in s and P- downward; for C† they change places.
    pub fn mult_site_inv<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        inp: &Field<T, Spinor, S>,
        mass: f64,
        dagger: bool,
    ) {
        assert!(inp.ls() == self.params.ls && out.same_shape(inp));
        if self.has_alpha() {
            // (C Q)⁻¹ = Q⁻¹ C⁻¹ and (Q C†)⁻¹ = C†⁻¹ Q⁻¹.
            let mut t = inp.zeros_like();
            if dagger {
                self.mult_q(&mut t, inp, true);
                self.site_inv_plain(out, &t, mass, true);
            } else {
                self.site_inv_plain(&mut t, inp, mass, false);
                self.mult_q(out, &t, true);
            }
            return;
        }
        self.site_inv_plain(out, inp, mass, dagger);
    }

    /// C⁻¹ or C†⁻¹ of the plain operator (α = 1).
    fn site_inv_plain<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        inp: &Field<T, Spinor, S>,
        mass: f64,
        dagger: bool,
    ) {
        let ls = self.params.ls;
        let number = <T::Compute as Number>::from_f64;
        let (d_plus, d_minus) = self.site_coefficients();
        let q = d_minus / d_plus;
        // x at the first row of the substitution, from the corner row:
        // d+ x_first + m d- (p_last + q^(Ls-1) x_first) = r_first.
        let corner = number(
            1.0 / (d_plus
                + mass * d_minus * q.powi(i32::try_from(ls - 1).expect("Ls is at most MAX_LS"))),
        );
        let (to_last, per_step) = (number(-mass * d_minus), number(1.0 / d_plus));
        let scaled =
            |k: T::Compute, x: &SiteSpinor<T::Compute>| SiteSpinor::zero().sub_scaled(-k, x);
        let site_len = out.site_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            for (v, o) in o.chunks_exact_mut(ls * SPINOR_REALS).enumerate() {
                let from = &inp.site(0, x)[v * ls * SPINOR_REALS..(v + 1) * ls * SPINOR_REALS];
                let mut result = [SiteSpinor::<T::Compute>::zero(); MAX_LS];
                // The two chiralities, P+ first. `up`: the substitution runs from s = 0.
                for (sign, up) in [(1.0, !dagger), (-1.0, dagger)] {
                    let at = |i: usize| if up { i } else { ls - 1 - i };
                    let mut r = [SiteSpinor::<T::Compute>::zero(); MAX_LS];
                    for i in 0..ls {
                        let reals = &from[at(i) * SPINOR_REALS..(at(i) + 1) * SPINOR_REALS];
                        let phi = load::<T>(reals);
                        // (1 + sign γ5) / 2
                        let both = phi.sub_scaled(number(-sign), &G::GAMMA5.apply(&phi));
                        r[i] = scaled(number(0.5), &both);
                    }
                    // p_0 = 0, p_i = (r_i + d- p_(i-1)) / d+
                    let mut part = [SiteSpinor::<T::Compute>::zero(); MAX_LS];
                    for i in 1..ls {
                        let sum = r[i].sub_scaled(number(-d_minus), &part[i - 1]);
                        part[i] = scaled(per_step, &sum);
                    }
                    let first = scaled(corner, &r[0].sub_scaled(-to_last, &part[ls - 1]));
                    let mut power = 1.0;
                    for i in 0..ls {
                        let x_i = part[i].sub_scaled(number(-power), &first);
                        result[at(i)] = result[at(i)].add(&x_i);
                        power *= q;
                    }
                }
                for (s, reals) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                    store::<T>(&result[s], reals);
                }
            }
        });
    }

    /// `mult` at another mass: the Pauli-Villars operator is this at `mass = 1`.
    pub fn mult_at(
        &self,
        work: &mut MdwfWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
        variant: Variant,
        mass: f64,
    ) {
        assert_eq!(inp.ls(), self.params.ls, "the field has another Ls");
        assert!(out.same_shape(inp));
        let [w, v, d, qf] = scratch(&mut work.fields, inp);
        // D^(α) ψ = D (Q ψ); D^(α)† ψ = Q (D† ψ), Q applied at the end below.
        let inp: &Field<T, Spinor> = if self.has_alpha() && variant != Variant::Dagger {
            self.mult_q(qf, inp, false);
            qf
        } else {
            inp
        };
        let MdwfParams { m5, b, c, .. } = self.params;
        let height = 4.0 - m5;
        let plain = (false, false);
        match variant {
            // D ψ = ψ + Y ψ + W (b ψ - c Y ψ)
            Variant::D | Variant::Gamma5D => {
                self.combine(w, &[(inp, b, -c)], plain, mass);
                self.wilson.mult(&mut work.wilson, v, w, Variant::D);
                let reflect = variant == Variant::Gamma5D;
                // Γ5 goes on last, so it needs D ψ whole before it is reflected.
                let to = if reflect { &mut *d } else { &mut *out };
                self.combine(to, &[(inp, 1.0, 1.0), (v, height, 0.0)], plain, mass);
                if reflect {
                    self.combine(out, &[(d, 1.0, 0.0)], (false, true), mass);
                }
            }
            // D† ψ = ψ + Y† ψ + (b - c Y†) W† ψ
            Variant::Dagger => {
                self.wilson.mult(&mut work.wilson, v, inp, Variant::Dagger);
                let terms = [(inp, 1.0, 1.0), (&*v, height * b, -height * c)];
                if self.has_alpha() {
                    self.combine(d, &terms, (true, false), mass);
                    self.mult_q(out, d, false);
                } else {
                    self.combine(out, &terms, (true, false), mass);
                }
            }
        }
    }
}

/// The Pauli-Villars preconditioned operator `A = (D_PV C_PV^-1)† D C^-1` of an `Mdwf`,
/// with what goes with it: the right-hand side `X† b`, `X = D_PV C_PV^-1`, and the
/// solution `x = C^-1 y` of `D x = b` from the solution y of `A y = X† b`. See the top of
/// the file.
///
/// One application is two sweeps of the Wilson kernel (through D and through D_PV†) and
/// two exact site-local inverses.
pub struct PvPreconditioned<'a, T: Storage, G: GammaRepre = Dirac> {
    op: &'a Mdwf<T, G>,
    work: MdwfWork<T>,
    first: Field<T, Spinor>,
    second: Field<T, Spinor>,
}

impl<'a, T: Storage, G: GammaRepre> PvPreconditioned<'a, T, G> {
    /// `like` only provides the shape of the intermediate fields.
    pub fn new(op: &'a Mdwf<T, G>, like: &Field<T, Spinor>) -> Self {
        Self {
            op,
            work: op.work(),
            first: like.zeros_like(),
            second: like.zeros_like(),
        }
    }

    /// `out = X† inp = C_PV†^-1 D_PV† inp`: the right-hand side of `A y = X† b` for
    /// `D x = b`.
    pub fn right_hand_side(&mut self, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        fit(&mut self.first, inp);
        self.op
            .mult_at(&mut self.work, &mut self.first, inp, Variant::Dagger, 1.0);
        self.op.mult_site_inv(out, &self.first, 1.0, true);
    }

    /// `x = C^-1 y`: the solution of `D x = b` from that of `A y = X† b`.
    pub fn solution(&self, x: &mut Field<T, Spinor>, y: &Field<T, Spinor>) {
        self.op.mult_site_inv(x, y, self.op.params.mass, false);
    }
}

impl<T: Storage, G: GammaRepre> LinearFn<Field<T, Spinor>> for PvPreconditioned<'_, T, G> {
    /// `out = A inp = C_PV†^-1 D_PV† D C^-1 inp`.
    fn mult(&mut self, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        let mass = self.op.params.mass;
        fit(&mut self.first, inp);
        fit(&mut self.second, inp);
        self.op.mult_site_inv(&mut self.first, inp, mass, false);
        self.op.mult_at(
            &mut self.work,
            &mut self.second,
            &self.first,
            Variant::D,
            mass,
        );
        self.op.mult_at(
            &mut self.work,
            &mut self.first,
            &self.second,
            Variant::Dagger,
            1.0,
        );
        self.op.mult_site_inv(out, &self.first, 1.0, true);
    }
}

impl<T: Storage, G: GammaRepre> LinearOp<Field<T, Spinor>> for PvPreconditioned<'_, T, G> {
    /// `out = A† inp = C†^-1 D† D_PV C_PV^-1 inp`.
    fn mult_dag(&mut self, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        let mass = self.op.params.mass;
        fit(&mut self.first, inp);
        fit(&mut self.second, inp);
        self.op.mult_site_inv(&mut self.first, inp, 1.0, false);
        self.op.mult_at(
            &mut self.work,
            &mut self.second,
            &self.first,
            Variant::D,
            1.0,
        );
        self.op.mult_at(
            &mut self.work,
            &mut self.first,
            &self.second,
            Variant::Dagger,
            mass,
        );
        self.op.mult_site_inv(out, &self.first, mass, true);
    }
}

/// The parts of D, as multigrid coarsening needs them: with `w = b ψ - c Y ψ`,
/// `D ψ = C ψ - (4 - M5) κ Σ hops(w)`, the hops being those of the Wilson kernel. The
/// part that stays within a site is C; Γ5 is γ5 times the reflection of the fifth
/// dimension.
impl<T: Storage, G: GammaRepre> HopTerms<Field<T, Spinor>> for Mdwf<T, G> {
    fn hop_forward(
        &self,
        work: &mut MdwfWork<T>,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        let MdwfWork { wilson, fields } = work;
        let w = self.before_kernel(fields, inp);
        self.wilson.hop_forward(wilson, mu, out, w);
    }

    fn hop_backward(
        &self,
        work: &mut MdwfWork<T>,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        let MdwfWork { wilson, fields } = work;
        let w = self.before_kernel(fields, inp);
        self.wilson.hop_backward(wilson, mu, out, w);
    }

    fn mult_diag(
        &self,
        work: &mut MdwfWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        let [c, ..] = scratch(&mut work.fields, inp);
        self.mult_site(c, inp, self.params.mass, false);
        out.axpy(&vec![C64::new(1.0, 0.0); inp.num_vectors()], c);
    }

    fn mult_gamma5(
        &self,
        _work: &mut MdwfWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.combine(out, &[(inp, 1.0, 0.0)], (false, true), self.params.mass);
    }

    fn hop_coefficient(&self) -> f64 {
        -(4.0 - self.params.m5) / (8.0 - 2.0 * self.params.m5)
    }
}

/// Scratch space of the even-odd blocks of `Mdwf`: one field on each parity, made on first
/// use.
pub struct MdwfHalfWork<T: Storage> {
    wilson: WilsonWork<T>,
    fields: [Option<Field<T, Spinor, Half>>; 2],
}

/// The scratch field on the parity of `like`, in its shape.
fn half_scratch<'w, T: Storage>(
    fields: &'w mut [Option<Field<T, Spinor, Half>>; 2],
    like: &Field<T, Spinor, Half>,
) -> &'w mut Field<T, Spinor, Half> {
    let slot = &mut fields[like.parity().index()];
    if !slot.as_ref().is_some_and(|f| f.same_shape(like)) {
        *slot = Some(like.zeros_like());
    }
    slot.as_mut().expect("the field was just made")
}

impl<T: Storage, G: GammaRepre> Mdwf<T, G> {
    /// The even-odd numbering this operator uses, if the lattice allows one.
    pub fn even_odd_index(&self) -> Option<&Arc<EvenOddIndex>> {
        self.wilson.even_odd_index()
    }
}

/// The even-odd blocks of D, in four dimensions: all `ls` spinors of a site have the
/// parity of the site. With `B = b - c Y` and the hops `M_pq` of the Wilson kernel,
///
///   D = [[C, (4 - M5) M_eo B], [(4 - M5) M_oe B, C]],
///
/// so the diagonal blocks are the site-local part C, whose inverse is exact, and
/// `(D†)_pq = B† (4 - M5) (M†)_pq`.
impl<T: Storage, G: GammaRepre> EvenOdd<Field<T, Spinor, Half>> for Mdwf<T, G> {
    type Work = MdwfHalfWork<T>;

    fn even_odd_work(&self) -> MdwfHalfWork<T> {
        MdwfHalfWork {
            wilson: self.wilson.even_odd_work(),
            fields: [None, None],
        }
    }

    fn hop(
        &self,
        work: &mut MdwfHalfWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        to: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.ls(), self.params.ls, "the field has another Ls");
        let MdwfParams { m5, b, c, mass, .. } = self.params;
        let height = 4.0 - m5;
        match dagger {
            Dagger::No => {
                // (D Q)_pq = D_pq Q_q.
                let q_inp;
                let inp = if self.has_alpha() {
                    let mut t = inp.zeros_like();
                    self.mult_q(&mut t, inp, false);
                    q_inp = t;
                    &q_inp
                } else {
                    inp
                };
                let w = half_scratch(&mut work.fields, inp);
                self.combine(w, &[(inp, height * b, -height * c)], (false, false), mass);
                self.wilson.hop(&mut work.wilson, out, w, to, dagger);
            }
            Dagger::Yes => {
                // ((D Q)†)_pq = Q_p (D†)_pq.
                let v = half_scratch(&mut work.fields, out);
                self.wilson.hop(&mut work.wilson, v, inp, to, dagger);
                if self.has_alpha() {
                    let mut t = out.zeros_like();
                    self.combine(
                        &mut t,
                        &[(&*v, height * b, -height * c)],
                        (true, false),
                        mass,
                    );
                    self.mult_q(out, &t, false);
                } else {
                    self.combine(out, &[(&*v, height * b, -height * c)], (true, false), mass);
                }
            }
        }
    }

    fn diag(
        &self,
        _work: &mut MdwfHalfWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert!(inp.parity() == parity && out.parity() == parity);
        self.mult_site(out, inp, self.params.mass, dagger == Dagger::Yes);
    }

    fn diag_inv(
        &self,
        _work: &mut MdwfHalfWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert!(inp.parity() == parity && out.parity() == parity);
        self.mult_site_inv(out, inp, self.params.mass, dagger == Dagger::Yes);
    }
}

/// The four scratch fields of a work object, in the shape of `like`.
fn scratch<'w, T: Storage>(
    fields: &'w mut Option<[Field<T, Spinor>; 4]>,
    like: &Field<T, Spinor>,
) -> &'w mut [Field<T, Spinor>; 4] {
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

impl<T: Storage, G: GammaRepre> Mdwf<T, G> {
    /// `b ψ - c Y ψ`, what the Wilson kernel is applied to, in a scratch field.
    fn before_kernel<'w>(
        &self,
        fields: &'w mut Option<[Field<T, Spinor>; 4]>,
        inp: &Field<T, Spinor>,
    ) -> &'w Field<T, Spinor> {
        let [w, _, _, qf] = scratch(fields, inp);
        let MdwfParams { b, c, mass, .. } = self.params;
        // The hops of D^(α) = D Q act on Q ψ.
        if self.has_alpha() {
            self.mult_q(qf, inp, false);
            self.combine(w, &[(&*qf, b, -c)], (false, false), mass);
        } else {
            self.combine(w, &[(inp, b, -c)], (false, false), mass);
        }
        w
    }
}

/// D, D† or Γ5 D of an `Mdwf` at a chosen mass, as a linear map: the Pauli-Villars
/// operator is this at mass 1.
pub struct MdwfAt<'a, T: Storage, G: GammaRepre = Dirac> {
    op: &'a Mdwf<T, G>,
    work: MdwfWork<T>,
    variant: Variant,
    mass: f64,
}

impl<'a, T: Storage, G: GammaRepre> MdwfAt<'a, T, G> {
    pub fn new(op: &'a Mdwf<T, G>, variant: Variant, mass: f64) -> Self {
        Self {
            op,
            work: op.work(),
            variant,
            mass,
        }
    }
}

impl<T: Storage, G: GammaRepre> LinearFn<Field<T, Spinor>> for MdwfAt<'_, T, G> {
    fn mult(&mut self, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        self.op
            .mult_at(&mut self.work, out, inp, self.variant, self.mass);
    }
}

/// The inverse of the site-local part of an `Mdwf` at a chosen mass, `C^-1` or `C†^-1`, as
/// a linear map.
pub struct SiteInverse<'a, T: Storage, G: GammaRepre = Dirac> {
    pub op: &'a Mdwf<T, G>,
    pub mass: f64,
    pub dagger: bool,
}

impl<T: Storage, G: GammaRepre> LinearFn<Field<T, Spinor>> for SiteInverse<'_, T, G> {
    fn mult(&mut self, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        self.op.mult_site_inv(out, inp, self.mass, self.dagger);
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
        self.mult_at(work, out, inp, variant, self.params.mass);
    }
}
