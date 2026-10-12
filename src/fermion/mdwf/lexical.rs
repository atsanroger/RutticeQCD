//! The operator on the whole lattice: D, D† and Γ5 D at any mass, its parts for the
//! multigrid, and the Pauli-Villars preconditioned operator A.

use super::*;

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    /// `mult` at another mass: the Pauli-Villars operator is this at `mass = 1`.
    pub fn mult_at(
        &self,
        work: &mut MdwfWork<T, L>,
        out: &mut Field<T, Spinor, Lexical, L>,
        inp: &Field<T, Spinor, Lexical, L>,
        variant: Variant,
        mass: f64,
    ) {
        assert_eq!(inp.ls(), self.params.ls, "the field has another Ls");
        assert!(out.same_shape(inp));
        let [w, v, d, qf] = scratch(&mut work.fields, inp);
        // D^(α) ψ = D (Q ψ); D^(α)† ψ = Q (D† ψ), Q applied at the end below.
        let inp: &Field<T, Spinor, Lexical, L> = if self.has_alpha() && variant != Variant::Dagger {
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
                L::mult(&self.wilson, &mut work.wilson, v, w, Variant::D);
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
                L::mult(&self.wilson, &mut work.wilson, v, inp, Variant::Dagger);
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
pub struct PvPreconditioned<'a, T: Storage, G: GammaRepre = Dirac, L: MdwfLayout<T> = Aos> {
    op: &'a Mdwf<T, G, L>,
    work: MdwfWork<T, L>,
    first: Field<T, Spinor, Lexical, L>,
    second: Field<T, Spinor, Lexical, L>,
}

impl<'a, T: Storage, G: GammaRepre, L: MdwfLayout<T>> PvPreconditioned<'a, T, G, L> {
    /// `like` only provides the shape of the intermediate fields.
    pub fn new(op: &'a Mdwf<T, G, L>, like: &Field<T, Spinor, Lexical, L>) -> Self {
        Self {
            op,
            work: op.work(),
            first: like.zeros_like(),
            second: like.zeros_like(),
        }
    }

    /// `out = X† inp = C_PV†^-1 D_PV† inp`: the right-hand side of `A y = X† b` for
    /// `D x = b`.
    pub fn right_hand_side(
        &mut self,
        out: &mut Field<T, Spinor, Lexical, L>,
        inp: &Field<T, Spinor, Lexical, L>,
    ) {
        fit(&mut self.first, inp);
        self.op
            .mult_at(&mut self.work, &mut self.first, inp, Variant::Dagger, 1.0);
        self.op.mult_site_inv(out, &self.first, 1.0, true);
    }

    /// `x = C^-1 y`: the solution of `D x = b` from that of `A y = X† b`.
    pub fn solution(&self, x: &mut Field<T, Spinor, Lexical, L>, y: &Field<T, Spinor, Lexical, L>) {
        self.op.mult_site_inv(x, y, self.op.params.mass, false);
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> LinearFn<Field<T, Spinor, Lexical, L>>
    for PvPreconditioned<'_, T, G, L>
{
    /// `out = A inp = C_PV†^-1 D_PV† D C^-1 inp`.
    fn mult(&mut self, out: &mut Field<T, Spinor, Lexical, L>, inp: &Field<T, Spinor, Lexical, L>) {
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

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> LinearOp<Field<T, Spinor, Lexical, L>>
    for PvPreconditioned<'_, T, G, L>
{
    /// `out = A† inp = C†^-1 D† D_PV C_PV^-1 inp`.
    fn mult_dag(
        &mut self,
        out: &mut Field<T, Spinor, Lexical, L>,
        inp: &Field<T, Spinor, Lexical, L>,
    ) {
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

impl<T: Storage, G: GammaRepre> Mdwf<T, G> {
    /// The Wilson kernel whose hops make up D: D ψ = C ψ + `hop_coefficient` Σ hops(w), with
    /// w = `kernel_input(ψ)`. What the force of HMC differentiates.
    pub fn kernel(&self) -> &WilsonOp<T, G, UnitDiag> {
        &self.wilson
    }

    /// w = b ψ − c Y ψ (of Q ψ with Neff's α), the field the kernel's hops act on in D ψ.
    pub fn kernel_input(&self, inp: &Field<T, Spinor>) -> Field<T, Spinor> {
        let mut fields = None;
        let w = self.before_kernel(&mut fields, inp);
        let mut out = inp.zeros_like();
        out.copy_from(w);
        out
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    /// `b ψ - c Y ψ`, what the Wilson kernel is applied to, in a scratch field.
    fn before_kernel<'w>(
        &self,
        fields: &'w mut Option<[Field<T, Spinor, Lexical, L>; 4]>,
        inp: &Field<T, Spinor, Lexical, L>,
    ) -> &'w Field<T, Spinor, Lexical, L> {
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
pub struct MdwfAt<'a, T: Storage, G: GammaRepre = Dirac, L: MdwfLayout<T> = Aos> {
    op: &'a Mdwf<T, G, L>,
    work: MdwfWork<T, L>,
    variant: Variant,
    mass: f64,
}

impl<'a, T: Storage, G: GammaRepre, L: MdwfLayout<T>> MdwfAt<'a, T, G, L> {
    pub fn new(op: &'a Mdwf<T, G, L>, variant: Variant, mass: f64) -> Self {
        Self {
            op,
            work: op.work(),
            variant,
            mass,
        }
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> LinearFn<Field<T, Spinor, Lexical, L>>
    for MdwfAt<'_, T, G, L>
{
    fn mult(&mut self, out: &mut Field<T, Spinor, Lexical, L>, inp: &Field<T, Spinor, Lexical, L>) {
        self.op
            .mult_at(&mut self.work, out, inp, self.variant, self.mass);
    }
}

/// The inverse of the site-local part of an `Mdwf` at a chosen mass, `C^-1` or `C†^-1`, as
/// a linear map.
pub struct SiteInverse<'a, T: Storage, G: GammaRepre = Dirac, L: MdwfLayout<T> = Aos> {
    pub op: &'a Mdwf<T, G, L>,
    pub mass: f64,
    pub dagger: bool,
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> LinearFn<Field<T, Spinor, Lexical, L>>
    for SiteInverse<'_, T, G, L>
{
    fn mult(&mut self, out: &mut Field<T, Spinor, Lexical, L>, inp: &Field<T, Spinor, Lexical, L>) {
        self.op.mult_site_inv(out, inp, self.mass, self.dagger);
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> FermionOp<Field<T, Spinor, Lexical, L>>
    for Mdwf<T, G, L>
{
    type Work = MdwfWork<T, L>;

    fn work(&self) -> MdwfWork<T, L> {
        MdwfWork {
            wilson: L::work_of(&self.wilson),
            fields: None,
        }
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        L::set_gauge(&mut self.wilson, u)
    }

    /// The count Bridge++ uses for its domain-wall operator, `(464 Nc + 240) Ls`.
    fn flops_per_site(&self) -> usize {
        (464 * NC + 240) * self.params.ls
    }

    fn mult(
        &self,
        work: &mut MdwfWork<T, L>,
        out: &mut Field<T, Spinor, Lexical, L>,
        inp: &Field<T, Spinor, Lexical, L>,
        variant: Variant,
    ) {
        self.mult_at(work, out, inp, variant, self.params.mass);
    }
}
