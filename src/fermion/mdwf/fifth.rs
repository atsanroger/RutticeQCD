//! The fifth dimension, local to a four-dimensional site: Neff's Q, the hop Y, and the
//! site-local part C and its exact inverse.

use super::*;

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
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
        let number = <L::Num as Number>::from_f64;
        // P- + a P+ = (1 + a)/2 + (a - 1)/2 γ5; P+ + a P- = (1 + a)/2 + (1 - a)/2 γ5.
        let (even, odd) = (number(0.5 * (1.0 + a)), number(0.5 * (a - 1.0)));
        let inside = number(a);
        let site_len = out.block_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            for (v, o) in o.chunks_exact_mut(ls * sp).enumerate() {
                let from = &block(inp, x)[v * ls * sp..(v + 1) * ls * sp];
                for (s, (out_reals, in_reals)) in o
                    .chunks_exact_mut(sp)
                    .zip(from.chunks_exact(sp))
                    .enumerate()
                {
                    let p = L::load(in_reals);
                    let q = if s == 0 || s == ls - 1 {
                        let g = G::GAMMA5.apply(&p);
                        let k = if s == 0 { odd } else { -odd };
                        SiteSpinor::zero().sub_scaled(-even, &p).sub_scaled(-k, &g)
                    } else {
                        SiteSpinor::zero().sub_scaled(-inside, &p)
                    };
                    L::store(&q, out_reals);
                }
            }
        });
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
        let number = <L::Num as Number>::from_f64;
        // The coefficient of the projected neighbour: -1, or +m across the walls.
        let across = |wall: bool| number(0.5 * if wall { mass } else { -1.0 });
        // Y takes P- from above and P+ from below; Y† the other way round.
        let (above, below) = if dagger { (1.0, -1.0) } else { (-1.0, 1.0) };
        let projected = |from: usize, sign: f64, k: L::Num| {
            // k (1 + sign γ5) φ
            let p = phi(from);
            let both = p.sub_scaled(number(-sign), &G::GAMMA5.apply(&p));
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
    pub(super) fn combine<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        terms: &[Term<'_, T, S, L>],
        (dagger, reflect): (bool, bool),
        mass: f64,
    ) {
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        let number = <L::Num as Number>::from_f64;
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

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
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
    ///
    /// For one chirality C is `d+` on the diagonal, `-d-` next to it on one side and
    /// `m d-` in the far corner. Substitution along s gives every x_s as a known part plus
    /// a multiple of the x at the wall the corner refers to; the corner row then fixes
    /// that one. P+ runs upward in s and P- downward; for C† they change places.
    pub fn mult_site_inv<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
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
        out: &mut Field<T, Spinor, S, L>,
        inp: &Field<T, Spinor, S, L>,
        mass: f64,
        dagger: bool,
    ) {
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        let number = <L::Num as Number>::from_f64;
        let (d_plus, d_minus) = self.site_coefficients();
        let q = d_minus / d_plus;
        // x at the first row of the substitution, from the corner row:
        // d+ x_first + m d- (p_last + q^(Ls-1) x_first) = r_first.
        let corner = number(
            1.0 / (d_plus
                + mass * d_minus * q.powi(i32::try_from(ls - 1).expect("Ls is at most MAX_LS"))),
        );
        let (to_last, per_step) = (number(-mass * d_minus), number(1.0 / d_plus));
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
                        let phi = L::load(&from[at(i) * sp..(at(i) + 1) * sp]);
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
                        let x_i = p.sub_scaled(number(-power), &first);
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
