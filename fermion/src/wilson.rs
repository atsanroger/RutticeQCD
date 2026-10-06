//! Wilson fermions.
//!
//! D ψ(x) = ψ(x) - κ Σ_mu [ (1 - γ_mu) U_mu(x) ψ(x + mu) + (1 + γ_mu) U_mu(x - mu)^† ψ(x - mu) ]
//!
//! Provides: `mult` (fused, one sweep) for D, D† and γ5 D; the hopping terms one direction
//! at a time. Solvers: CG on D†D, BiCGStab on D. Tools needed: none.
//!
//! The normalisation (hopping parameter κ, unit diagonal) and the placement of the boundary
//! phase follow Bridge++ so that results can be compared. Written from the formula above.

use crate::{FermionOp, GaugeError, HopTerms, Variant};
use ruttice_core::{
    for_sites, Boundary, Direction, Field, Gauge, Lattice, Source, Spinor, Stencil, Storage, ND,
};
use ruttice_kernel::{
    sink_spinor, transport, ColorMatrix, Cplx, Dirac, GammaRepre, LinkUse, Real, Sign,
    Spinor as SiteSpinor, SPINOR_REALS,
};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct WilsonParams {
    /// Hopping parameter κ.
    pub kappa: f64,
    pub boundary: [Boundary; ND],
}

/// Scratch space for `Wilson`. Empty on a single node; communication buffers go here.
pub struct WilsonWork {}

pub struct Wilson<T: Storage, G: GammaRepre = Dirac> {
    params: WilsonParams,
    lat: Arc<Lattice>,
    stencil: Stencil,
    u: Option<Field<T, Gauge>>,
    _gamma: PhantomData<G>,
}

/// Where one leg of the stencil comes from, resolved for one site.
#[derive(Clone, Copy)]
struct Leg<R> {
    site: usize,
    /// Boundary phase to apply, if the hop crossed a boundary with a non-trivial phase.
    phase: Option<Cplx<R>>,
}

impl<T: Storage, G: GammaRepre> Wilson<T, G> {
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Self {
        Self {
            params,
            lat: lat.clone(),
            stencil: Stencil::nearest(lat, params.boundary),
            u: None,
            _gamma: PhantomData,
        }
    }

    pub fn params(&self) -> &WilsonParams {
        &self.params
    }

    fn gauge(&self) -> &Field<T, Gauge> {
        self.u
            .as_ref()
            .expect("Wilson: set_gauge must be called before the operator is applied")
    }

    /// The neighbour in direction (mu, dir) of `site`, or `None` across a Dirichlet boundary.
    /// A hop backward across the boundary picks up the conjugate phase.
    #[inline(always)]
    fn leg(&self, site: usize, mu: usize, dir: Direction) -> Option<Leg<T::Compute>> {
        match self.stencil.source(site, mu, dir) {
            Source::Zero => None,
            Source::Local {
                site,
                wrapped: false,
            } => Some(Leg { site, phase: None }),
            Source::Local {
                site,
                wrapped: true,
            } => {
                let phase = match self.params.boundary[mu] {
                    Boundary::Periodic => None,
                    Boundary::Antiperiodic => Some((-1.0, 0.0)),
                    Boundary::Phase(re, im) => Some(match dir {
                        Direction::Forward => (re, im),
                        Direction::Backward => (re, -im),
                    }),
                    Boundary::Dirichlet => unreachable!("Dirichlet hops are Source::Zero"),
                };
                let phase = phase.map(|(re, im)| Cplx::new(Real::from_f64(re), Real::from_f64(im)));
                Some(Leg { site, phase })
            }
            Source::Buffer { .. } => unreachable!("no communication buffers on a single node"),
        }
    }

    #[inline(always)]
    fn link(&self, mu: usize, site: usize) -> ColorMatrix<T::Compute> {
        let s = self.gauge().site(mu, site);
        let v: [T::Compute; 18] = core::array::from_fn(|k| s[k].load());
        ColorMatrix::from_reals(&v)
    }

    fn check(&self, out: &Field<T, Spinor>, inp: &Field<T, Spinor>) {
        let n = self.lat.num_sites();
        assert!(
            out.num_sites() == n && inp.num_sites() == n,
            "field on a different lattice"
        );
        assert!(out.outer() == 1 && inp.outer() == 1);
        assert_eq!(
            out.copies(),
            inp.copies(),
            "out and inp hold different numbers of spinors"
        );
    }

    /// Add one hopping term to `out`. Shared by `hop_forward` and `hop_backward`.
    fn hop(&self, mu: usize, dir: Direction, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        self.check(out, inp);
        let gamma = &G::GAMMA[mu];
        let (sign, link_use) = match dir {
            Direction::Forward => (Sign::Minus, LinkUse::Direct),
            Direction::Backward => (Sign::Plus, LinkUse::Adjoint),
        };
        let copies = inp.copies();
        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            let Some(leg) = self.leg(x, mu, dir) else {
                return;
            };
            let link = match dir {
                Direction::Forward => self.link(mu, x),
                Direction::Backward => self.link(mu, leg.site),
            };
            let neighbour = inp.site(0, leg.site);
            let o = o.as_chunks_mut::<SPINOR_REALS>().0;
            for (o, n) in o.iter_mut().zip(neighbour.as_chunks::<SPINOR_REALS>().0) {
                let mut acc = load_spinor::<T>(o);
                let h = transport(
                    gamma,
                    sign,
                    &load_spinor::<T>(n),
                    &link,
                    link_use,
                    leg.phase,
                );
                sink_spinor(&mut acc, &h, gamma, sign);
                store_spinor::<T>(&acc, o);
            }
        });
    }
}

#[inline(always)]
fn load_spinor<T: Storage>(s: &[T]) -> SiteSpinor<T::Compute> {
    let v: [T::Compute; SPINOR_REALS] = core::array::from_fn(|k| s[k].load());
    SiteSpinor::from_reals(&v)
}

#[inline(always)]
fn store_spinor<T: Storage>(p: &SiteSpinor<T::Compute>, s: &mut [T]) {
    let mut v = [T::Compute::ZERO; SPINOR_REALS];
    p.write_reals(&mut v);
    for (dst, src) in s.iter_mut().zip(v) {
        *dst = T::store(src);
    }
}

impl<T: Storage, G: GammaRepre> FermionOp<Field<T, Spinor>> for Wilson<T, G> {
    type Work = WilsonWork;

    fn work(&self) -> WilsonWork {
        WilsonWork {}
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.num_sites() != self.lat.num_sites() || u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let mut own = Field::<T, Gauge>::gauge(&self.lat);
        for (dst, src) in own.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *dst = T::store(Real::from_f64(*src));
        }
        self.u = Some(own);
        Ok(())
    }

    fn mult(
        &self,
        _work: &mut WilsonWork,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
        variant: Variant,
    ) {
        self.check(out, inp);
        // γ5 applied to the input (pre) and to the result (post) of the same sweep.
        let (pre, post) = match variant {
            Variant::D => (false, false),
            Variant::Gamma5D => (false, true),
            Variant::Dagger => (true, true),
        };
        let kappa: T::Compute = Real::from_f64(self.params.kappa);
        let copies = inp.copies();

        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            // Geometry and links are resolved once per site and shared by all copies.
            let forward: [_; ND] = core::array::from_fn(|mu| self.leg(x, mu, Direction::Forward));
            let backward: [_; ND] = core::array::from_fn(|mu| self.leg(x, mu, Direction::Backward));
            let link_forward: [_; ND] = core::array::from_fn(|mu| self.link(mu, x));
            let link_backward: [_; ND] =
                core::array::from_fn(|mu| backward[mu].map(|leg| self.link(mu, leg.site)));

            for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                let at = |site: usize| {
                    let s = &inp.site(0, site)[r * SPINOR_REALS..(r + 1) * SPINOR_REALS];
                    let psi = load_spinor::<T>(s);
                    if pre {
                        G::GAMMA5.apply(&psi)
                    } else {
                        psi
                    }
                };

                let mut hops = SiteSpinor::zero();
                for mu in 0..ND {
                    let gamma = &G::GAMMA[mu];
                    if let Some(leg) = forward[mu] {
                        let h = transport(
                            gamma,
                            Sign::Minus,
                            &at(leg.site),
                            &link_forward[mu],
                            LinkUse::Direct,
                            leg.phase,
                        );
                        sink_spinor(&mut hops, &h, gamma, Sign::Minus);
                    }
                    if let (Some(leg), Some(link)) = (backward[mu], &link_backward[mu]) {
                        let h = transport(
                            gamma,
                            Sign::Plus,
                            &at(leg.site),
                            link,
                            LinkUse::Adjoint,
                            leg.phase,
                        );
                        sink_spinor(&mut hops, &h, gamma, Sign::Plus);
                    }
                }

                let mut result = at(x).sub_scaled(kappa, &hops);
                if post {
                    result = G::GAMMA5.apply(&result);
                }
                store_spinor::<T>(&result, o);
            }
        });
    }
}

impl<T: Storage, G: GammaRepre> HopTerms<Field<T, Spinor>> for Wilson<T, G> {
    fn hop_forward(
        &self,
        _work: &mut WilsonWork,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.hop(mu, Direction::Forward, out, inp);
    }

    fn hop_backward(
        &self,
        _work: &mut WilsonWork,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.hop(mu, Direction::Backward, out, inp);
    }

    fn mult_diag(
        &self,
        _work: &mut WilsonWork,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(out, inp);
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            let sum = load_spinor::<T>(o).add(&load_spinor::<T>(s));
            store_spinor::<T>(&sum, o);
        });
    }

    fn mult_gamma5(
        &self,
        _work: &mut WilsonWork,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(out, inp);
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            store_spinor::<T>(&G::GAMMA5.apply(&load_spinor::<T>(s)), o);
        });
    }

    fn hop_coefficient(&self) -> f64 {
        -self.params.kappa
    }
}
