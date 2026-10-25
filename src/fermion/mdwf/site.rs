//! The fifth dimension, local to a four-dimensional site: Neff's Q, the hop Y, and the
//! site-local part C and its exact inverse.

use super::*;

/// The flops of the fifth-dimension arithmetic, as this code does it, per spinor (24 reals):
/// `sub_scaled` is 2 a real, `add` 1. The device does the same matrices as dense Ls × Ls
/// products (96 Ls a spinor), so the two count differently, each what it computes.
pub(super) mod flops {
    const AXPY: f64 = 48.0;
    const ADD: f64 = 24.0;
    /// k (1 ± γ5) φ: φ ∓ γ5 φ, then times k.
    const PROJECT: f64 = 2.0 * AXPY;
    /// (Y φ)_s: two projected neighbours added (`fifth_hop`).
    const Y: f64 = 2.0 * PROJECT + ADD;

    /// Neff's Q on the spinors of a vector, a mean per spinor: a scaling inside, a scaling and
    /// a γ5 term at the two walls (`neff_q`).
    pub(in crate::fermion::mdwf) fn q(ls: usize) -> f64 {
        let ls = ls as f64;
        (AXPY * (ls - 2.0) + 2.0 * 2.0 * AXPY) / ls
    }

    /// The factor B = a + y Y, with Neff's Q on its three inputs when `with_q` (`Fifth::apply`).
    pub(in crate::fermion::mdwf) fn factor(ls: usize, y: bool, with_q: bool) -> f64 {
        term(y) + if with_q { 3.0 * q(ls) } else { 0.0 }
    }

    /// One term `a x + y Y x` of `combine`, or the factor B without Q.
    pub(in crate::fermion::mdwf) fn term(y: bool) -> f64 {
        AXPY + if y { Y + AXPY } else { 0.0 }
    }

    /// C⁻¹ in place (`site_inv_in_place`): for each chirality, the first recursion
    /// ((Ls − 1) projected rows and steps), the first x, then every row projected, stepped,
    /// solved and written back: 504 Ls − 96 a chirality; a mean per spinor.
    pub(in crate::fermion::mdwf) fn site_inv_in_place(ls: usize) -> f64 {
        let ls = ls as f64;
        2.0 * (504.0 * ls - 96.0) / ls
    }

    /// C⁻¹ in its own pass (`site_inv_plain`): for each chirality 432 Ls − 192 (the projected
    /// row recomputed in each step), the second adding 24 a spinor; Q on its 2 Ls − 1 inputs and
    /// its Ls outputs a chirality when asked. A mean per spinor.
    pub(in crate::fermion::mdwf) fn site_inv_plain(ls: usize, q_in: bool, q_out: bool) -> f64 {
        let n = ls as f64;
        let mut total = 2.0 * (432.0 * n - 192.0) + ADD * n;
        if q_in {
            total += 2.0 * (2.0 * n - 1.0) * q(ls);
        }
        if q_out {
            total += 2.0 * n * q(ls);
        }
        total / n
    }
}

/// Counts fifth-dimension work (`meter`): `sites` sites with `spinors` spinors each, `flops` and
/// `moved` spinors read or written for every spinor.
pub(super) fn count_fifth<T: Storage>(sites: usize, spinors: usize, flops: f64, moved: f64) {
    let n = (sites * spinors) as f64;
    let size = std::mem::size_of::<T>();
    crate::meter::add(
        crate::meter::Work::Fine,
        size,
        n * flops,
        n * moved * SPINOR_REALS as f64 * size as f64,
    );
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    /// `out = Γ5 inp`, Γ5 = γ5 R (R reverses the fifth dimension), on the sites `inp` has.
    pub fn mult_gamma5_on<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
    ) {
        self.combine(out, &[(inp, 1.0, 0.0)], (false, true), self.params.mass);
    }

    /// `out = Q inp`, or `Q⁻¹ inp` with `inverse`: Q_0 = P- + α P+, Q_s = α inside,
    /// Q_(Ls-1) = P+ + α P-. Q is Hermitian. The solution of D x = b is x = Q x^(α) for
    /// D^(α) x^(α) = b.
    pub fn mult_q<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
        inverse: bool,
    ) {
        assert!(inp.ls() == self.params.ls && out.same_shape(inp));
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        let a = if inverse {
            1.0 / self.alpha
        } else {
            self.alpha
        };
        count_fifth::<T>(out.num_sites(), out.copies(), flops::q(ls), 2.0);
        let site_len = out.block_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            for (v, o) in o.chunks_exact_mut(ls * sp).enumerate() {
                let from = &block(inp, x)[v * ls * sp..(v + 1) * ls * sp];
                for (s, (out_reals, in_reals)) in o
                    .chunks_exact_mut(sp)
                    .zip(from.chunks_exact(sp))
                    .enumerate()
                {
                    let q = neff_q::<L::Num, G>(a, s, ls, &L::load(in_reals));
                    L::store(&q, out_reals);
                }
            }
        });
    }

    /// Q at α = `a` (`neff_q`) on the spinors of one site, `ls` per vector, in place: for a
    /// result in the cache, in the pass that made it.
    #[inline(always)]
    pub(super) fn q_in_place(a: f64, site: &mut [T], ls: usize) {
        let sp = SPINOR_REALS * L::LANES;
        for v in site.chunks_exact_mut(ls * sp) {
            for (s, reals) in v.chunks_exact_mut(sp).enumerate() {
                let q = neff_q::<L::Num, G>(a, s, ls, &L::load(reals));
                L::store(&q, reals);
            }
        }
    }

    pub(super) fn has_alpha(&self) -> bool {
        self.alpha != 1.0
    }

    /// `(Y φ)_s`, or `(Y† φ)_s`, at mass `mass`, from the `ls` spinors φ_k = `phi(k)` of
    /// one site (loaded where they are needed: a site's spinors stay in the cache).
    #[inline(always)]
    fn fifth(
        phi: impl Fn(usize) -> SiteSpinor<L::Num>,
        ls: usize,
        s: usize,
        dagger: bool,
        mass: f64,
    ) -> SiteSpinor<L::Num> {
        let (above, below) = (phi((s + 1) % ls), phi((s + ls - 1) % ls));
        fifth_hop::<L::Num, G>(s, ls, dagger, mass, &above, &below)
    }

    /// `out_s = Σ_k a_k x_k,s + Σ_k y_k (Y x_k)_s` for the spinors of every site and vector: each
    /// `(field, a, y)` of `terms` contributes its spinor times `a` and its hop in the fifth
    /// dimension (Y, or Y† if `dagger`) times `y`.
    pub(super) fn combine<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        terms: &[Term<'_, T, S, L>],
        (dagger, reflect): (bool, bool),
        mass: f64,
    ) {
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        let number = <L::Num as Number>::from_f64;
        let per_spinor: f64 = terms.iter().map(|(_, _, y)| flops::term(*y != 0.0)).sum();
        count_fifth::<T>(
            out.num_sites(),
            out.copies(),
            per_spinor,
            terms.len() as f64 + 1.0,
        );
        let site_len = out.block_len();
        let load = |from: &[T], s: usize| L::load(&from[s * sp..(s + 1) * sp]);
        for_sites(out.slab_mut(0), site_len, |x, o| {
            // One vector at a time: its `ls` spinors are next to each other. The spinors are
            // loaded where they are used (up to three times, from the cache), with no
            // buffer of a site's spinors to fill.
            for (v, o) in o.chunks_exact_mut(ls * sp).enumerate() {
                for (s_out, reals) in o.chunks_exact_mut(sp).enumerate() {
                    let s = if reflect { ls - 1 - s_out } else { s_out };
                    let mut result = SiteSpinor::<L::Num>::zero();
                    for (field, a, y) in terms {
                        let from = &block(field, x)[v * ls * sp..(v + 1) * ls * sp];
                        result = result.sub_scaled(number(-*a), &load(from, s));
                        if *y != 0.0 {
                            let hop = Self::fifth(|k| load(from, k), ls, s, dagger, mass);
                            result = result.sub_scaled(number(-*y), &hop);
                        }
                    }
                    if reflect {
                        L::store(&G::GAMMA5.apply(&result), reals);
                    } else {
                        L::store(&result, reals);
                    }
                }
            }
        });
    }
}

/// The constants of the substitution along s for C⁻¹ at one mass (see `mult_site_inv`).
pub(super) struct Substitution<N> {
    /// x at the first row of the substitution, from the corner row:
    /// d+ x_first + m d- (p_last + q^(Ls-1) x_first) = r_first.
    corner: N,
    to_last: N,
    per_step: N,
    d_minus: f64,
    /// d- / d+
    q: f64,
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    pub(super) fn substitution(&self, mass: f64) -> Substitution<L::Num> {
        let number = <L::Num as Number>::from_f64;
        let ls = self.params.ls;
        let (d_plus, d_minus) = self.site_coefficients();
        let q = d_minus / d_plus;
        Substitution {
            corner: number(
                1.0 / (d_plus
                    + mass
                        * d_minus
                        * q.powi(i32::try_from(ls - 1).expect("Ls is at most MAX_LS"))),
            ),
            to_last: number(-mass * d_minus),
            per_step: number(1.0 / d_plus),
            d_minus,
            q,
        }
    }

    /// `fifth` on the spinors of one site, `ls` per vector, in place: for the result of a
    /// sweep of the Wilson kernel while it is in the cache. A spinor is read before it is
    /// overwritten; the first and the last of a vector are kept for the wrap-around.
    #[inline(always)]
    pub(super) fn factor_in_place(fifth: &Fifth, site: &mut [T]) {
        let (ls, sp) = (fifth.ls, SPINOR_REALS * L::LANES);
        let at = |v: &[T], s: usize| L::load(&v[s * sp..(s + 1) * sp]);
        for v in site.chunks_exact_mut(ls * sp) {
            let (first, mut below) = (at(v, 0), at(v, ls - 1));
            for s in 0..ls {
                let here = at(v, s);
                let above = if s + 1 < ls { at(v, s + 1) } else { first };
                let result = fifth.apply::<L::Num, G>(s, &below, &here, &above);
                L::store(&result, &mut v[s * sp..(s + 1) * sp]);
                below = here;
            }
        }
    }

    /// C⁻¹ (C†⁻¹ with `dagger`) of the plain operator on the `ls` spinors of one vector at
    /// one site, in place: for a result that is in the cache, in the pass that made it.
    #[inline(always)]
    pub(super) fn site_inv_in_place(
        k: &Substitution<L::Num>,
        v: &mut [T],
        ls: usize,
        dagger: bool,
    ) {
        let sp = SPINOR_REALS * L::LANES;
        let number = <L::Num as Number>::from_f64;
        let scaled = |c: L::Num, x: &SiteSpinor<L::Num>| SiteSpinor::zero().sub_scaled(-c, x);
        for (sign, up) in [(1.0, !dagger), (-1.0, dagger)] {
            let at = |i: usize| if up { i } else { ls - 1 - i };
            // (1 + sign γ5) / 2 of a spinor
            let project = |phi: &SiteSpinor<L::Num>| {
                let both = phi.sub_scaled(number(-sign), &G::GAMMA5.apply(phi));
                scaled(number(0.5), &both)
            };
            // p_0 = 0, p_i = (r_i + d- p_(i-1)) / d+
            let step = |r: &SiteSpinor<L::Num>, p: &SiteSpinor<L::Num>| {
                scaled(k.per_step, &r.sub_scaled(number(-k.d_minus), p))
            };
            let row = |v: &[T], i: usize| project(&L::load(&v[at(i) * sp..(at(i) + 1) * sp]));
            let mut last = SiteSpinor::zero();
            for i in 1..ls {
                last = step(&row(v, i), &last);
            }
            let first = scaled(k.corner, &row(v, 0).sub_scaled(-k.to_last, &last));
            let (mut p, mut power) = (SiteSpinor::zero(), 1.0);
            for i in 0..ls {
                let reals = &mut v[at(i) * sp..(at(i) + 1) * sp];
                let phi = L::load(reals);
                let r = project(&phi);
                if i > 0 {
                    p = step(&r, &p);
                }
                let x_i = p.sub_scaled(number(-power), &first);
                L::store(&phi.sub_scaled(number(1.0), &r).add(&x_i), reals);
                power *= k.q;
            }
        }
    }

    /// The coefficients (d+, d-) of the site-local part C = d+ + d- Y.
    fn site_coefficients(&self) -> (f64, f64) {
        let height = 4.0 - self.params.m5;
        (1.0 + self.params.b * height, 1.0 - self.params.c * height)
    }

    /// `out = C inp`, or `C† inp`: the part of D (at mass `mass`) that stays within a
    /// four-dimensional site. The fields may be on the whole lattice or on one parity.
    pub fn mult_site<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
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
    pub fn mult_site_inv<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
        mass: f64,
        dagger: bool,
    ) {
        assert!(inp.ls() == self.params.ls && out.same_shape(inp));
        // (C Q)⁻¹ = Q⁻¹ C⁻¹ and (Q C†)⁻¹ = C†⁻¹ Q⁻¹, Q⁻¹ in the same pass.
        let a = 1.0 / self.alpha;
        let q = if dagger { (a, 1.0) } else { (1.0, a) };
        self.site_inv_plain(out, inp, mass, dagger, q);
    }

    /// C⁻¹ or C†⁻¹ of the plain operator (α = 1), between Q at `q.0` on its input and Q at `q.1`
    /// on its output (`neff_q`; 1 for none).
    fn site_inv_plain<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
        mass: f64,
        dagger: bool,
        (q_in, q_out): (f64, f64),
    ) {
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        let number = <L::Num as Number>::from_f64;
        let Substitution {
            corner,
            to_last,
            per_step,
            d_minus,
            q,
        } = self.substitution(mass);
        let per_spinor = flops::site_inv_plain(ls, q_in != 1.0, q_out != 1.0);
        count_fifth::<T>(out.num_sites(), out.copies(), per_spinor, 2.0);
        let scaled = |k: L::Num, x: &SiteSpinor<L::Num>| SiteSpinor::zero().sub_scaled(-k, x);
        let site_len = out.block_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            for (v, o) in o.chunks_exact_mut(ls * sp).enumerate() {
                let from = &block(inp, x)[v * ls * sp..(v + 1) * ls * sp];
                // The two chiralities, P+ first. `up`: the substitution runs from s = 0. The
                // recursion runs twice, first for the last p only (which fixes the first
                // x), then writing every x: no buffer of a site's spinors is needed.
                for (pass, (sign, up)) in [(1.0, !dagger), (-1.0, dagger)].into_iter().enumerate() {
                    let at = |i: usize| if up { i } else { ls - 1 - i };
                    // (1 + sign γ5) / 2 of the spinor at row i
                    let r = |i: usize| {
                        let mut phi = L::load(&from[at(i) * sp..(at(i) + 1) * sp]);
                        if q_in != 1.0 {
                            phi = neff_q::<L::Num, G>(q_in, at(i), ls, &phi);
                        }
                        let both = phi.sub_scaled(number(-sign), &G::GAMMA5.apply(&phi));
                        scaled(number(0.5), &both)
                    };
                    // p_0 = 0, p_i = (r_i + d- p_(i-1)) / d+
                    let step = |i: usize, p: &SiteSpinor<L::Num>| {
                        scaled(per_step, &r(i).sub_scaled(number(-d_minus), p))
                    };
                    let mut last = SiteSpinor::zero();
                    for i in 1..ls {
                        last = step(i, &last);
                    }
                    let first = scaled(corner, &r(0).sub_scaled(-to_last, &last));
                    let (mut p, mut power) = (SiteSpinor::zero(), 1.0);
                    for i in 0..ls {
                        if i > 0 {
                            p = step(i, &p);
                        }
                        let mut x_i = p.sub_scaled(number(-power), &first);
                        // x_i has one chirality, which Q only scales.
                        if q_out != 1.0 {
                            x_i = neff_q::<L::Num, G>(q_out, at(i), ls, &x_i);
                        }
                        let reals = &mut o[at(i) * sp..(at(i) + 1) * sp];
                        if pass == 0 {
                            L::store(&x_i, reals);
                        } else {
                            L::store(&L::load(reals).add(&x_i), reals);
                        }
                        power *= q;
                    }
                }
            }
        });
    }
}
