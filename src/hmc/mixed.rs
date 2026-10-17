//! Mixed precision in the solves of the pseudofermion actions: (A†A)⁻¹ b by defect correction,
//! the residual of A†A in double precision and each correction by CG in single precision on
//! the same operator on the same links (`SingleNormal`), until the residual is below the
//! tolerance of the solve.

use super::even_odd::HalfPsi;
use super::fermion_action::Psi;
use crate::fermion::{Bound, EoForm, FermionOp, Mdwf, MdwfParams, SchurFactor, Variant};
use crate::kernel::{Dirac, GammaRepre};
use crate::solver::{Cg, SolveError, SolveInfo, Solver, Stop};
use crate::C64;
use crate::{ConvertFrom, Field, Gauge, Half, Lattice, LinearOp, Normal, Parity, Spinor, Vector};
use std::sync::Arc;

/// The operator of an action in single precision: the inner solves of mixed precision.
pub trait SingleNormal<V, L> {
    /// Takes the links the operator of the action has now.
    fn set_links(&mut self, links: &L);
    /// e ≈ (A†A)⁻¹ r by CG in single precision from zero, to the tolerance of the inner solves;
    /// its iterations.
    fn normal(&mut self, e: &mut V, r: &V, max_iterations: usize) -> usize;
}

/// The domain-wall operator in single precision, D (or Neff's D Q) on all sites, CG on D†D to
/// |r|²/|b|² < `tolerance_sqr`.
pub struct SingleMdwf<G: GammaRepre = Dirac> {
    op: Mdwf<f32, G>,
    pub tolerance_sqr: f64,
}

impl<G: GammaRepre> SingleMdwf<G> {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams, alpha: f64, tolerance_sqr: f64) -> Self {
        let mut op = Mdwf::<f32, G>::new(lat, params);
        op.set_alpha(alpha);
        Self { op, tolerance_sqr }
    }
}

impl<G: GammaRepre> SingleNormal<Psi, Field<f64, Gauge>> for SingleMdwf<G> {
    fn set_links(&mut self, links: &Field<f64, Gauge>) {
        self.op
            .set_gauge(links)
            .expect("the links suit the operator");
    }

    fn normal(&mut self, e: &mut Psi, r: &Psi, max_iterations: usize) -> usize {
        let mut r32 = Field::<f32, Spinor>::with_shape(
            r.lattice(),
            24,
            self.op.params().ls,
            r.num_vectors(),
            1,
        );
        r32.convert_from(r);
        let mut e32 = r32.zeros_like();
        let mut d = Bound::new(&self.op, Variant::D);
        let stop = Stop::simple(self.tolerance_sqr, max_iterations);
        let iterations =
            iterations(Cg::new(stop).solve(&mut Normal::new(&mut d, &r32), None, &mut e32, &r32));
        e.convert_from(&e32);
        iterations
    }
}

/// The same for the even-odd Ratio action: K = T_ee⁻¹ Ŝ (the left `SchurFactor`) on the even
/// sites, CG on K†K.
pub struct SingleMdwfEo<G: GammaRepre = Dirac> {
    op: Mdwf<f32, G>,
    pub tolerance_sqr: f64,
}

impl<G: GammaRepre> SingleMdwfEo<G> {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams, tolerance_sqr: f64) -> Self {
        Self {
            op: Mdwf::<f32, G>::new(lat, params),
            tolerance_sqr,
        }
    }
}

impl<G: GammaRepre> SingleNormal<HalfPsi, Field<f64, Gauge>> for SingleMdwfEo<G> {
    fn set_links(&mut self, links: &Field<f64, Gauge>) {
        self.op
            .set_gauge(links)
            .expect("the links suit the operator");
    }

    fn normal(&mut self, e: &mut HalfPsi, r: &HalfPsi, max_iterations: usize) -> usize {
        let mut r32 = Field::<f32, Spinor, Half>::with_shape(
            r.lattice(),
            24,
            self.op.params().ls,
            r.num_vectors(),
            1,
        );
        r32.set_parity(Parity::Even);
        r32.convert_from(r);
        let mut odd = r32.zeros_like();
        odd.set_parity(Parity::Odd);
        let mut e32 = r32.zeros_like();
        let mut k = SchurFactor::new(&self.op, EoForm::Left, &r32, &odd);
        let stop = Stop::simple(self.tolerance_sqr, max_iterations);
        let iterations =
            iterations(Cg::new(stop).solve(&mut Normal::new(&mut k, &r32), None, &mut e32, &r32));
        e.convert_from(&e32);
        iterations
    }
}

/// The iterations of an inner solve; not reaching its tolerance is left to the outer residual.
fn iterations(info: Result<SolveInfo, SolveError>) -> usize {
    match info {
        Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
        Err(e) => panic!("the single-precision solve broke down: {e:?}"),
    }
}

/// A†A x = b by defect correction: e = (A†A)⁻¹ r in single precision on the residual r of
/// double precision, until |r|²/|b|² < `stop.tolerance_sqr` (at most `stop.max_restarts` + 1
/// residuals). `x` holds the initial guess; the iterations are those of the inner CG.
pub fn mixed_normal<V: Vector, L>(
    a: &mut dyn LinearOp<V>,
    single: &mut dyn SingleNormal<V, L>,
    x: &mut V,
    b: &V,
    stop: Stop,
) -> Result<SolveInfo, SolveError> {
    let n = b.num_vectors();
    let mut b_norm = vec![0.0; n];
    b.norm_sqr(&mut b_norm);
    let (mut t, mut r, mut e) = (b.zeros_like(), b.zeros_like(), b.zeros_like());
    let mut info = SolveInfo {
        iterations: 0,
        restarts: 0,
        residual_sqr: vec![0.0; n],
    };
    for sweep in 0..=stop.max_restarts {
        info.restarts = sweep;
        a.mult(&mut t, x);
        a.mult_dag(&mut r, &t);
        r.aypx(&vec![C64::new(-1.0, 0.0); n], b);
        r.norm_sqr(&mut info.residual_sqr);
        for (r, b) in info.residual_sqr.iter_mut().zip(&b_norm) {
            *r = if *b > 0.0 { *r / b } else { 0.0 };
        }
        if info.residual_sqr.iter().all(|r| *r < stop.tolerance_sqr) {
            return Ok(info);
        }
        if sweep == stop.max_restarts {
            break;
        }
        info.iterations += single.normal(&mut e, &r, stop.max_iterations);
        x.axpy(&vec![C64::new(1.0, 0.0); n], &e);
    }
    Err(SolveError::NotConverged(info))
}
