//! An operator in single precision as the inner solver of mixed precision, the same on every
//! backend: CG on D†D on all sites (`SingleAll`), or on K†K on the even sites, K = T_ee⁻¹ Ŝ the
//! left `SchurFactor`, with Neff's Q when asked (`SingleEven`: CG on (K Q_e)†(K Q_e), as
//! `AfterQ`). The backend supplies only the operator (`SingleOperator`, for `Mdwf<f32>` and the
//! device's `CudaMdwf<f32>`). They make the corrections of the defect correction of HMC
//! (`hmc::SingleNormal`) and the CGNE of the propagators with reliable updates
//! (`SingleEven::reliable_normal`).

use super::{AfterQ, Bound, EoForm, EvenOdd, FermionOp, SchurFactor, Variant};
use crate::solver::{reliable_cg, Cg, SolveError, SolveInfo, Solver, Stop};
use crate::{ConvertFrom, Lattice, LinearFn, Normal, Parity, Vector};
use std::sync::Arc;

/// What a backend gives of its operator in single precision.
pub trait SingleOperator: FermionOp<Self::Whole> + EvenOdd<Self::Half> + Sized {
    /// Fields on all sites and on the sites of one parity.
    type Whole: Vector;
    type Half: Vector;
    /// The links of the action, in double precision, where the backend keeps them.
    type Links;
    fn set_links(&mut self, links: &Self::Links);
    /// Zero fields of `n` vectors of the shape the operator acts on.
    fn whole(&self, lat: &Arc<Lattice>, n: usize) -> Self::Whole;
    fn half(&self, lat: &Arc<Lattice>, parity: Parity, n: usize) -> Self::Half;
    /// `out = Q inp`, or `Q⁻¹ inp` with `inverse`, Neff's Q at the α of this operator. Q acts
    /// within a site and does not take the links.
    fn mult_q(&self, out: &mut Self::Half, inp: &Self::Half, inverse: bool);
}

/// D (or Neff's D Q, if the operator has α) in single precision on all sites: CG on D†D.
pub struct SingleAll<O: SingleOperator> {
    op: O,
    lat: Arc<Lattice>,
    /// The inner solves stop at |r|²/|b|² below this at the latest.
    pub tolerance_sqr: f64,
}

impl<O: SingleOperator> SingleAll<O> {
    pub fn new(op: O, lat: &Arc<Lattice>, tolerance_sqr: f64) -> Self {
        Self {
            op,
            lat: lat.clone(),
            tolerance_sqr,
        }
    }

    pub fn set_links(&mut self, links: &O::Links) {
        self.op.set_links(links);
    }

    /// e ≈ (D†D)⁻¹ r by CG in single precision from zero, to |r − D†D e|²/|r|² below
    /// `tolerance_sqr` or the inner tolerance, whichever is larger; its iterations.
    pub fn normal<V>(
        &mut self,
        e: &mut V,
        r: &V,
        tolerance_sqr: f64,
        max_iterations: usize,
    ) -> usize
    where
        V: Vector + ConvertFrom<O::Whole>,
        O::Whole: ConvertFrom<V>,
    {
        let mut r32 = self.op.whole(&self.lat, r.num_vectors());
        r32.convert_from(r);
        let mut e32 = r32.zeros_like();
        let mut d = Bound::new(&self.op, Variant::D);
        let stop = Stop::simple(self.tolerance_sqr.max(tolerance_sqr), max_iterations);
        let info = Cg::new(stop).solve(&mut Normal::new(&mut d, &r32), None, &mut e32, &r32);
        e.convert_from(&e32);
        iterations(info)
    }
}

/// K = T_ee⁻¹ Ŝ of the operator in single precision on the even sites (the even-odd Ratio
/// action, the CGNE of the propagators): CG on K†K, or with Neff's Q on (K Q_e)†(K Q_e).
pub struct SingleEven<O: SingleOperator> {
    op: O,
    lat: Arc<Lattice>,
    pub tolerance_sqr: f64,
    /// The operator at Neff's α whose Q the solves take (`with_neff`).
    neff: Option<O>,
}

impl<O: SingleOperator> SingleEven<O> {
    pub fn new(op: O, lat: &Arc<Lattice>, tolerance_sqr: f64) -> Self {
        Self {
            op,
            lat: lat.clone(),
            tolerance_sqr,
            neff: None,
        }
    }

    /// The solves on K Q_e, Q that of `alpha_op` (an operator at Neff's α; only its Q is used),
    /// as `TwoFlavorRatioEo::with_neff` asks.
    pub(crate) fn with_neff(mut self, alpha_op: O) -> Self {
        self.neff = Some(alpha_op);
        self
    }

    pub fn set_links(&mut self, links: &O::Links) {
        self.op.set_links(links);
    }

    /// e ≈ (K†K)⁻¹ r (with Q: ((K Q_e)†(K Q_e))⁻¹ r) by CG in single precision from zero, as
    /// `SingleAll::normal`.
    pub fn normal<V>(
        &mut self,
        e: &mut V,
        r: &V,
        tolerance_sqr: f64,
        max_iterations: usize,
    ) -> usize
    where
        V: Vector + ConvertFrom<O::Half>,
        O::Half: ConvertFrom<V>,
    {
        let mut r32 = self.op.half(&self.lat, Parity::Even, r.num_vectors());
        r32.convert_from(r);
        let mut e32 = r32.zeros_like();
        let stop = Stop::simple(self.tolerance_sqr.max(tolerance_sqr), max_iterations);
        let info = self.with_normal(&r32, |a| Cg::new(stop).solve(a, None, &mut e32, &r32));
        e.convert_from(&e32);
        iterations(info)
    }

    /// `a x = b`, `a` the normal operator in double precision of which this one is the single
    /// precision: CG in single precision with reliable updates (`solver::reliable_cg`), each when
    /// |r|² has fallen by `drop_sqr`. `x` holds the initial guess.
    pub(crate) fn reliable_normal<V>(
        &mut self,
        a: &mut dyn LinearFn<V>,
        x: &mut V,
        b: &V,
        stop: Stop,
        drop_sqr: f64,
    ) -> Result<SolveInfo, SolveError>
    where
        V: Vector + ConvertFrom<O::Half>,
        O::Half: ConvertFrom<V>,
    {
        let like = self.op.half(&self.lat, Parity::Even, b.num_vectors());
        self.with_normal(&like, |low| {
            reliable_cg(a, low, &like, x, b, stop, drop_sqr)
        })
    }

    /// `run` with the normal operator of K (with Q, of K Q_e) on fields shaped like `like`.
    fn with_normal<R>(
        &self,
        like: &O::Half,
        run: impl FnOnce(&mut dyn LinearFn<O::Half>) -> R,
    ) -> R {
        let odd = self.op.half(&self.lat, Parity::Odd, 1);
        let mut k = SchurFactor::new(&self.op, EoForm::Left, like, &odd);
        match &self.neff {
            Some(n) => {
                let q = |out: &mut O::Half, inp: &O::Half, inverse| n.mult_q(out, inp, inverse);
                let mut a = AfterQ::new(&mut k, &q, like);
                run(&mut Normal::new(&mut a, like))
            }
            None => run(&mut Normal::new(&mut k, like)),
        }
    }
}

/// The iterations of an inner solve; not reaching its tolerance is left to the outer residual.
fn iterations(info: Result<SolveInfo, SolveError>) -> usize {
    match info {
        Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
        Err(e) => panic!("the single-precision solve broke down: {e:?}"),
    }
}

impl<G, L> SingleOperator for super::Mdwf<f32, G, L>
where
    G: crate::kernel::GammaRepre,
    L: super::MdwfLayout<f32>,
    Self: FermionOp<crate::Field<f32, crate::Spinor, crate::Lexical, L>>
        + EvenOdd<crate::Field<f32, crate::Spinor, crate::Half, L>>,
{
    type Whole = crate::Field<f32, crate::Spinor, crate::Lexical, L>;
    type Half = crate::Field<f32, crate::Spinor, crate::Half, L>;
    type Links = crate::Field<f64, crate::Gauge>;

    fn set_links(&mut self, links: &Self::Links) {
        FermionOp::<Self::Whole>::set_gauge(self, links).expect("the links suit the operator");
    }

    fn whole(&self, lat: &Arc<Lattice>, n: usize) -> Self::Whole {
        crate::Field::with_shape(lat, 24, self.params().ls, n, 1)
    }

    fn half(&self, lat: &Arc<Lattice>, parity: Parity, n: usize) -> Self::Half {
        let mut h = crate::Field::with_shape(lat, 24, self.params().ls, n, 1);
        h.set_parity(parity);
        h
    }

    fn mult_q(&self, out: &mut Self::Half, inp: &Self::Half, inverse: bool) {
        super::Mdwf::mult_q(self, out, inp, inverse);
    }
}
