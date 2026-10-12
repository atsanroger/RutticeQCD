//! The five-dimensional source of a four-dimensional one, and the four-dimensional fields
//! read from a five-dimensional solution.

use super::*;

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    /// The five-dimensional right-hand side for a four-dimensional source η (`eta`, Ls = 1,
    /// any number of vectors): `out = D₋ b`, b = P+ η at s = 0 and P- η at s = Ls-1, zero
    /// in between, D₋ = 1 - c W. With D = D₋ K, K of the form of Shamir's operator, the
    /// solution ψ of D ψ = out has the four-dimensional quark field
    /// q = P- ψ_0 + P+ ψ_(Ls-1) (`propagator_4d`) of the source η: the fields
    /// q̄ = ψ̄_(Ls-1) P- + ψ̄_0 P+ (Furman and Shamir) put η on these walls.
    pub fn source_5d(
        &self,
        work: &mut MdwfWork<T, L>,
        out: &mut Field<T, Spinor, Lexical, L>,
        eta: &Field<T, Spinor, Lexical, L>,
    ) {
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        assert!(eta.ls() == 1 && out.ls() == ls && out.num_vectors() == eta.num_vectors());
        let [b, v, _, _] = scratch(&mut work.fields, out);
        let site_len = b.block_len();
        let half = <L::Num as Number>::from_f64(0.5);
        for_sites(b.slab_mut(0), site_len, |x, o| {
            o.fill(T::default());
            let from = block(eta, x);
            for (v, o) in o.chunks_exact_mut(ls * sp).enumerate() {
                let p = L::load(&from[v * sp..(v + 1) * sp]);
                let g = G::GAMMA5.apply(&p);
                // P± η = (η ± γ5 η) / 2
                let plus = SiteSpinor::zero()
                    .sub_scaled(-half, &p)
                    .sub_scaled(-half, &g);
                let minus = SiteSpinor::zero()
                    .sub_scaled(-half, &p)
                    .sub_scaled(half, &g);
                L::store(&plus, &mut o[..sp]);
                L::store(&minus, &mut o[(ls - 1) * sp..]);
            }
        });
        // out = b - c W b, W = (4 - M5) Dw.
        L::mult(&self.wilson, &mut work.wilson, v, b, Variant::D);
        let MdwfParams { m5, c, .. } = self.params;
        self.combine(
            out,
            &[(b, 1.0, 0.0), (v, -c * (4.0 - m5), 0.0)],
            (false, false),
            self.params.mass,
        );
    }

    /// The four-dimensional quark field q = P- ψ_0 + P+ ψ_(Ls-1) of a five-dimensional
    /// solution ψ (of the plain operator: x = Q x^(α) for D^(α)), into `out` (Ls = 1).
    pub fn propagator_4d(
        &self,
        out: &mut Field<T, Spinor, Lexical, L>,
        psi: &Field<T, Spinor, Lexical, L>,
    ) {
        let ls = self.params.ls;
        self.walls_to_4d(out, psi, (0, ls - 1));
    }

    /// The field at the middle of the fifth dimension, q_mid = P- ψ_(Ls/2) + P+ ψ_(Ls/2-1),
    /// of the pseudoscalar density J5q whose correlator gives the residual mass.
    pub fn midpoint_4d(
        &self,
        out: &mut Field<T, Spinor, Lexical, L>,
        psi: &Field<T, Spinor, Lexical, L>,
    ) {
        let ls = self.params.ls;
        assert!(ls.is_multiple_of(2), "the midpoint needs an even Ls");
        self.walls_to_4d(out, psi, (ls / 2, ls / 2 - 1));
    }

    /// `out = P- ψ_(minus) + P+ ψ_(plus)`.
    fn walls_to_4d(
        &self,
        out: &mut Field<T, Spinor, Lexical, L>,
        psi: &Field<T, Spinor, Lexical, L>,
        (minus, plus): (usize, usize),
    ) {
        let (ls, sp) = (self.params.ls, SPINOR_REALS * L::LANES);
        assert!(psi.ls() == ls && out.ls() == 1 && out.num_vectors() == psi.num_vectors());
        let half = <L::Num as Number>::from_f64(0.5);
        let site_len = out.block_len();
        for_sites(out.slab_mut(0), site_len, |x, o| {
            let from = block(psi, x);
            for (v, o) in o.chunks_exact_mut(sp).enumerate() {
                let base = v * ls * sp;
                let at = |s: usize| L::load(&from[base + s * sp..base + (s + 1) * sp]);
                let (m, p) = (at(minus), at(plus));
                // P- m + P+ p = (m + p)/2 + γ5 (p - m)/2
                let sum = SiteSpinor::zero()
                    .sub_scaled(-half, &m)
                    .sub_scaled(-half, &p);
                let diff = SiteSpinor::zero()
                    .sub_scaled(-half, &p)
                    .sub_scaled(half, &m);
                let q = sum.add(&G::GAMMA5.apply(&diff));
                L::store(&q, o);
            }
        });
    }
}
