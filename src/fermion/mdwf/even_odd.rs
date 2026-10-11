//! The even-odd blocks of D and the even-odd Pauli-Villars preconditioned operator Ã.

use super::*;

/// The even-odd form of the Pauli-Villars preconditioned operator (see [`PvEvenOdd`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EoForm {
    /// `Ã = C_PV,e†⁻¹ S_PV† S C_e⁻¹`, S = C_e − D_eo C_o⁻¹ D_oe the Schur complement of D on
    /// the even sites: the even-odd version of A. Neff's α cancels in it.
    Right,
    /// `Ã_L = (C_PV,e⁻¹ S_PV)† (C_e⁻¹ S)`, which sees α: with D^(α) = D Q it is
    /// `Q_e (C_PV,e⁻¹ S_PV)† Q_e⁻² (C_e⁻¹ S) Q_e` (α in D_PV too).
    Left,
}

type HalfSpinorField<T, L = Aos> = Field<T, Spinor, Half, L>;

/// The Pauli-Villars preconditioned operator on the even sites, in the form `form`, of a
/// quark operator and a Pauli-Villars operator (normally the same at mass 1), each with
/// its own α (`Mdwf::set_alpha`; 1 for the plain operator). With it: the right-hand side
/// for `D x = b` and the solution x of `D x = b`, through `x = Q x^(α)` when the quark
/// operator carries α.
///
/// One application is four half-lattice hops, two Wilson sweeps, as for the full-lattice A.
pub struct PvEvenOdd<'a, T: Storage, G: GammaRepre = Dirac, L: MdwfLayout<T> = Aos> {
    op: &'a Mdwf<T, G, L>,
    pv: &'a Mdwf<T, G, L>,
    form: EoForm,
    work: MdwfHalfWork<T, L>,
    work_pv: MdwfHalfWork<T, L>,
    odd: [HalfSpinorField<T, L>; 2],
    even: [HalfSpinorField<T, L>; 2],
}

impl<'a, T: Storage, G: GammaRepre, L: MdwfLayout<T>> PvEvenOdd<'a, T, G, L> {
    /// `like`: a field on the even sites, for the shape of the intermediate fields.
    pub fn new(
        op: &'a Mdwf<T, G, L>,
        pv: &'a Mdwf<T, G, L>,
        form: EoForm,
        like: &HalfSpinorField<T, L>,
    ) -> Self {
        assert_eq!(op.params.ls, pv.params.ls, "the two operators differ in Ls");
        let mut odd = like.zeros_like();
        odd.set_parity(Parity::Odd);
        let mut even = like.zeros_like();
        even.set_parity(Parity::Even);
        Self {
            op,
            pv,
            form,
            work: op.even_odd_work(),
            work_pv: pv.even_odd_work(),
            odd: [odd.zeros_like(), odd],
            even: [even.zeros_like(), even],
        }
    }

    pub fn form(&self) -> EoForm {
        self.form
    }

    fn fit_all(&mut self, like: &HalfSpinorField<T, L>) {
        for f in self.odd.iter_mut().chain(self.even.iter_mut()) {
            fit(f, like);
        }
    }

    /// The Schur factor of the quark operator (`pv` false) or of the Pauli-Villars one in
    /// the form of this operator, or its adjoint.
    fn factor(
        &mut self,
        out: &mut HalfSpinorField<T, L>,
        inp: &HalfSpinorField<T, L>,
        pv: bool,
        dagger: Dagger,
    ) {
        let (op, work) = if pv {
            (self.pv, &mut self.work_pv)
        } else {
            (self.op, &mut self.work)
        };
        schur_factor(
            op,
            work,
            self.form,
            (&mut self.odd, &mut self.even[0]),
            out,
            inp,
            dagger,
        );
    }

    /// The right-hand side of `Ã y = rhs` for `D x = b`: with b′_e = b_e − D_eo C_o⁻¹ b_o
    /// (which does not depend on α), `(S_PV C_PV,e⁻¹)† b′_e` for the right form and
    /// `(C_PV,e⁻¹ S_PV)† C_e⁻¹ b′_e` for the left one.
    pub fn right_hand_side(
        &mut self,
        rhs: &mut HalfSpinorField<T, L>,
        b: &Field<T, Spinor, Full, L>,
    ) {
        let index = self.op.even_odd_index().expect("an even-odd decomposition");
        let mut b_e = rhs.zeros_like();
        b_e.set_parity(Parity::Even);
        let mut b_o = rhs.zeros_like();
        b_o.set_parity(Parity::Odd);
        extract_parity(&mut b_e, b, index, Parity::Even);
        extract_parity(&mut b_o, b, index, Parity::Odd);
        self.fit_all(&b_e);
        let mut t_o = b_o.zeros_like();
        self.op
            .diag_inv(&mut self.work, &mut t_o, &b_o, Parity::Odd, Dagger::No);
        let mut t_e = b_e.zeros_like();
        self.op
            .hop(&mut self.work, &mut t_e, &t_o, Parity::Even, Dagger::No);
        let minus = vec![C64::new(-1.0, 0.0); b_e.num_vectors()];
        t_e.aypx(&minus, &b_e);
        if self.form == EoForm::Left {
            self.op
                .diag_inv(&mut self.work, &mut b_e, &t_e, Parity::Even, Dagger::No);
            t_e = b_e;
        }
        self.factor(rhs, &t_e, true, Dagger::Yes);
    }

    /// The solution x of `D x = b` from the solution y of `Ã y = rhs`: x_e^(α) = C_e⁻¹ y
    /// (right form) or y (left form), x_o^(α) = C_o⁻¹ (b_o − D_oe x_e^(α)), x = Q x^(α).
    pub fn solution(
        &mut self,
        x: &mut Field<T, Spinor, Full, L>,
        y: &HalfSpinorField<T, L>,
        b: &Field<T, Spinor, Full, L>,
    ) {
        let index = self.op.even_odd_index().expect("an even-odd decomposition");
        let x_e = match self.form {
            EoForm::Right => {
                let mut x_e = y.zeros_like();
                self.op
                    .diag_inv(&mut self.work, &mut x_e, y, Parity::Even, Dagger::No);
                x_e
            }
            EoForm::Left => {
                let mut x_e = y.zeros_like();
                x_e.copy_from(y);
                x_e
            }
        };
        let mut b_o = y.zeros_like();
        b_o.set_parity(Parity::Odd);
        extract_parity(&mut b_o, b, index, Parity::Odd);
        let mut t_o = b_o.zeros_like();
        self.op
            .hop(&mut self.work, &mut t_o, &x_e, Parity::Odd, Dagger::No);
        let minus = vec![C64::new(-1.0, 0.0); y.num_vectors()];
        t_o.aypx(&minus, &b_o);
        let mut x_o = b_o;
        self.op
            .diag_inv(&mut self.work, &mut x_o, &t_o, Parity::Odd, Dagger::No);
        let mut x_alpha = x.zeros_like();
        insert_parity(&mut x_alpha, &x_e, index);
        insert_parity(&mut x_alpha, &x_o, index);
        self.op.mult_q(x, &x_alpha, false);
    }
}

/// The Schur factor `1 − D_eo C_o⁻¹ D_oe C_e⁻¹` (right form) or `1 − C_e⁻¹ D_eo C_o⁻¹ D_oe`
/// (left form) of `op`, or its adjoint, with two odd scratch fields and one even one.
fn schur_factor<T: Storage, G: GammaRepre, L: MdwfLayout<T>>(
    op: &Mdwf<T, G, L>,
    work: &mut MdwfHalfWork<T, L>,
    form: EoForm,
    scratch: (&mut [HalfSpinorField<T, L>; 2], &mut HalfSpinorField<T, L>),
    out: &mut HalfSpinorField<T, L>,
    inp: &HalfSpinorField<T, L>,
    dagger: Dagger,
) {
    let ([o1, o2], e) = scratch;
    let (even, odd) = (Parity::Even, Parity::Odd);
    // The adjoint takes the factors in the opposite order.
    if (form == EoForm::Right) == (dagger == Dagger::No) {
        op.diag_inv(work, e, inp, even, dagger);
        op.hop(work, o1, e, odd, dagger);
        op.diag_inv(work, o2, o1, odd, dagger);
        op.hop(work, out, o2, even, dagger);
    } else {
        op.hop(work, o1, inp, odd, dagger);
        op.diag_inv(work, o2, o1, odd, dagger);
        op.hop(work, e, o2, even, dagger);
        op.diag_inv(work, out, e, even, dagger);
    }
    let minus = vec![C64::new(-1.0, 0.0); inp.num_vectors()];
    out.aypx(&minus, inp);
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> PvEvenOdd<'_, T, G, L> {
    /// `out = second(first(inp))`, the factors given as (Pauli-Villars?, dagger), through
    /// `even[1]`.
    fn product(
        &mut self,
        out: &mut HalfSpinorField<T, L>,
        inp: &HalfSpinorField<T, L>,
        first: (bool, Dagger),
        second: (bool, Dagger),
    ) {
        assert!(first.0 != second.0, "one factor of each operator");
        self.fit_all(inp);
        let [e0, e1] = &mut self.even;
        let (op1, op2, w1, w2) = if first.0 {
            (self.pv, self.op, &mut self.work_pv, &mut self.work)
        } else {
            (self.op, self.pv, &mut self.work, &mut self.work_pv)
        };
        schur_factor(op1, w1, self.form, (&mut self.odd, e0), e1, inp, first.1);
        schur_factor(op2, w2, self.form, (&mut self.odd, e0), out, e1, second.1);
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> LinearFn<HalfSpinorField<T, L>>
    for PvEvenOdd<'_, T, G, L>
{
    /// `out = Ã inp`: the quark factor, then the adjoint of the Pauli-Villars one.
    fn mult(&mut self, out: &mut HalfSpinorField<T, L>, inp: &HalfSpinorField<T, L>) {
        self.product(out, inp, (false, Dagger::No), (true, Dagger::Yes));
    }
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> LinearOp<HalfSpinorField<T, L>>
    for PvEvenOdd<'_, T, G, L>
{
    /// `out = Ã† inp`.
    fn mult_dag(&mut self, out: &mut HalfSpinorField<T, L>, inp: &HalfSpinorField<T, L>) {
        self.product(out, inp, (true, Dagger::No), (false, Dagger::Yes));
    }
}

/// Scratch space of the even-odd blocks of `Mdwf`: one field on each parity, made on first
/// use.
pub struct MdwfHalfWork<T: Storage, L: Layout = Aos> {
    wilson: WilsonWork<T>,
    fields: [Option<Field<T, Spinor, Half, L>>; 2],
}

/// The scratch field on the parity of `like`, in its shape.
fn half_scratch<'w, T: Storage, L: Layout>(
    fields: &'w mut [Option<Field<T, Spinor, Half, L>>; 2],
    like: &Field<T, Spinor, Half, L>,
) -> &'w mut Field<T, Spinor, Half, L> {
    let slot = &mut fields[like.parity().index()];
    if !slot.as_ref().is_some_and(|f| f.same_shape(like)) {
        *slot = Some(like.zeros_like());
    }
    slot.as_mut().expect("the field was just made")
}

impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> Mdwf<T, G, L> {
    /// The even-odd numbering this operator uses, if the lattice allows one.
    pub fn even_odd_index(&self) -> Option<&Arc<EvenOddIndex>> {
        L::even_odd_index(&self.wilson)
    }
}

/// The even-odd blocks of D, in four dimensions: all `ls` spinors of a site have the
/// parity of the site. With `B = b - c Y` and the hops `M_pq` of the Wilson kernel,
///
///   D = [[C, (4 - M5) M_eo B], [(4 - M5) M_oe B, C]],
///
/// so the diagonal blocks are the site-local part C, whose inverse is exact, and
/// `(D†)_pq = B† (4 - M5) (M†)_pq`.
impl<T: Storage, G: GammaRepre, L: MdwfLayout<T>> EvenOdd<Field<T, Spinor, Half, L>>
    for Mdwf<T, G, L>
{
    type Work = MdwfHalfWork<T, L>;

    fn even_odd_work(&self) -> MdwfHalfWork<T, L> {
        MdwfHalfWork {
            wilson: L::work_of(&self.wilson),
            fields: [None, None],
        }
    }

    fn hop(
        &self,
        work: &mut MdwfHalfWork<T, L>,
        out: &mut Field<T, Spinor, Half, L>,
        inp: &Field<T, Spinor, Half, L>,
        to: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.ls(), self.params.ls, "the field has another Ls");
        let MdwfParams {
            ls, m5, b, c, mass, ..
        } = self.params;
        let height = 4.0 - m5;
        // B = (4 - M5)(b - c Y) goes into the sweep of the Wilson kernel: on its input for
        // D_pq = M_pq B, on its output for (D†)_pq = B† M†_pq.
        let fifth = Fifth {
            ls,
            a: height * b,
            y: -height * c,
            mass,
            dagger: dagger == Dagger::Yes,
        };
        match dagger {
            Dagger::No => {
                // (D Q)_pq = D_pq Q_q.
                if self.has_alpha() {
                    let q = half_scratch(&mut work.fields, inp);
                    self.mult_q(q, inp, false);
                    L::hop_fifth(&self.wilson, &mut work.wilson, out, q, to, dagger, fifth);
                } else {
                    L::hop_fifth(&self.wilson, &mut work.wilson, out, inp, to, dagger, fifth);
                }
            }
            Dagger::Yes => {
                // ((D Q)†)_pq = Q_p (D†)_pq.
                if self.has_alpha() {
                    let v = half_scratch(&mut work.fields, out);
                    L::hop_fifth(&self.wilson, &mut work.wilson, v, inp, to, dagger, fifth);
                    self.mult_q(out, v, false);
                } else {
                    L::hop_fifth(&self.wilson, &mut work.wilson, out, inp, to, dagger, fifth);
                }
            }
        }
    }

    fn diag(
        &self,
        _work: &mut MdwfHalfWork<T, L>,
        out: &mut Field<T, Spinor, Half, L>,
        inp: &Field<T, Spinor, Half, L>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert!(inp.parity() == parity && out.parity() == parity);
        self.mult_site(out, inp, self.params.mass, dagger == Dagger::Yes);
    }

    fn diag_inv(
        &self,
        _work: &mut MdwfHalfWork<T, L>,
        out: &mut Field<T, Spinor, Half, L>,
        inp: &Field<T, Spinor, Half, L>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert!(inp.parity() == parity && out.parity() == parity);
        self.mult_site_inv(out, inp, self.params.mass, dagger == Dagger::Yes);
    }
}
