//! The Möbius domain-wall operator on the device.
//!
//! In the notation of `fermion::mdwf`, with H the hops of the Wilson kernel times (4 − M5),
//! the plain operator is `D = d+ + d− Y + H (b − c Y)`, `D† = d+ + d− Y† + (b − c Y†) H†`, and
//! the part of it within a site `C = d+ + d− Y`. With Neff's α (`set_alpha`) the operator is
//! `D Q`, Q diagonal in the fifth dimension and Hermitian: every block of it is the plain one
//! with Q on its input, and the adjoint of that.

use super::wilson::{gamma5, scratch};
use super::*;
use crate::fermion::{Dagger, EvenOdd, FermionOp, MdwfParams, Variant};
use crate::kernel::NC;
use crate::{Parity, Vector};

pub struct CudaMdwf<T = f64> {
    params: MdwfParams,
    geometry: Geometry,
    gauge: DeviceGauge<T>,
    /// Neff's α (1: the plain operator).
    alpha: f64,
}

/// Scratch fields of `CudaMdwf`, made on first use.
type Work<T> = [Option<DeviceSpinor<T>>; 2];

impl<T: DeviceReal> CudaMdwf<T> {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams) -> Result<Self, CudaError> {
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, params.ls),
            gauge: DeviceGauge::new(lat)?,
            alpha: 1.0,
        })
    }

    /// Neff's form D^(α) = D Q; 1 is the plain operator.
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(alpha > 0.0, "α must be positive");
        self.alpha = alpha;
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    fn has_alpha(&self) -> bool {
        self.alpha != 1.0
    }

    fn kernel_params(&self) -> MdwfKernelParams {
        MdwfKernelParams {
            m5: self.params.m5,
            mass: self.params.mass,
            alpha: 1.0,
            b5: self.params.b,
            c5: self.params.c,
        }
    }

    /// The coefficients (d+, d−) of the part of D within a site, C = d+ + d− Y.
    fn site_coefficients(&self) -> (f64, f64) {
        let height = 4.0 - self.params.m5;
        (1.0 + self.params.b * height, 1.0 - self.params.c * height)
    }

    /// `out = Q inp`, or `Q⁻¹ inp`: Q_0 = P− + α P+, Q_s = α inside, Q_(Ls−1) = P+ + α P−.
    /// The solution of D x = b is x = Q x^(α) for D^(α) x^(α) = b.
    pub fn mult_q(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, inverse: bool) {
        assert!(out.same_shape(inp) && inp.ls == self.params.ls);
        out.parity = inp.parity;
        let a = if inverse {
            1.0 / self.alpha
        } else {
            self.alpha
        };
        // SAFETY: two buffers of the same shape, not the same one (`out` is borrowed mutably).
        unsafe {
            T::mdwf_q(
                out.data.ptr,
                inp.data.ptr,
                inp.sites() as i32,
                inp.ls as i32,
                a,
            );
        }
    }

    /// `out = H (b − c Y) inp`, or with `H†`, for the given b and c: on the whole lattice, or
    /// `out` on the sites of parity `to` and `inp` on the others.
    fn hops(
        &self,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        (b, c): (f64, f64),
        dagger: bool,
        to: Option<Parity>,
    ) {
        let fifth = FifthInHops {
            mass: self.params.mass,
            b,
            c,
        };
        hops(
            (&self.gauge, &self.geometry),
            out,
            inp,
            0.5,
            fifth,
            dagger,
            to,
        );
    }

    /// `out = (a1 + y1 Y) x + (a2 + y2 Y) z`, or with `Y†`; the second term is optional.
    fn fifth(
        &self,
        out: &mut DeviceSpinor<T>,
        (x, a1, y1): (&DeviceSpinor<T>, f64, f64),
        second: Option<(&DeviceSpinor<T>, f64, f64)>,
        dagger: bool,
    ) {
        assert!(out.same_shape(x) && x.ls == self.params.ls);
        out.parity = x.parity;
        let (z, a2, y2) = match second {
            Some((z, a2, y2)) => {
                assert!(z.same_shape(x) && z.parity == x.parity);
                (z.data.ptr.cast_const(), a2, y2)
            }
            None => (std::ptr::null(), 0.0, 0.0),
        };
        // SAFETY: buffers of the same shape, `out` none of the inputs; a null `z` is not read.
        unsafe {
            T::mdwf_fifth(
                out.data.ptr,
                x.data.ptr,
                a1,
                y1,
                z,
                a2,
                y2,
                x.sites() as i32,
                x.ls as i32,
                self.params.mass,
                i32::from(dagger),
            );
        }
    }

    /// `out = D inp` of the plain operator on the whole lattice, in one pass.
    fn one_pass(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>) {
        let ls = self.params.ls;
        assert!(inp.is(out, ls, None) && out.is(inp, ls, None));
        let params = self.kernel_params();
        // SAFETY: as in `CudaWilson::apply`, the spinor buffers holding `ls` spinors per site.
        unsafe {
            T::mdwf_dslash(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                &self.geometry,
                &params,
            );
        }
    }

    /// `out = C⁻¹ inp`, or `C†⁻¹ inp`, of the plain operator: exactly, by substitution along
    /// the fifth dimension. On the whole lattice or on one parity.
    fn site_inv_plain(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, dagger: bool) {
        assert!(out.same_shape(inp) && inp.ls == self.params.ls);
        out.parity = inp.parity;
        let params = self.kernel_params();
        // SAFETY: both buffers hold the sites of `inp` with `ls` spinors each (padded), and
        // are not the same one.
        unsafe {
            T::mdwf_site_inv(
                out.data.ptr,
                inp.data.ptr,
                inp.sites() as i32,
                inp.ls as i32,
                &params,
                i32::from(dagger),
            );
        }
    }

    /// `out = plain(Q inp)` (`q_first`) or `Q plain(inp)`, with Q⁻¹ for Q if `inverse`, through
    /// the scratch field `t`; without α, `plain` alone.
    fn with_q(
        &self,
        t: &mut Option<DeviceSpinor<T>>,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        q_first: bool,
        inverse: bool,
        plain: impl FnOnce(&mut DeviceSpinor<T>, &DeviceSpinor<T>),
    ) {
        if !self.has_alpha() {
            plain(out, inp);
        } else if q_first {
            let t = scratch(t, inp);
            self.mult_q(t, inp, inverse);
            plain(out, t);
        } else {
            let t = scratch(t, out);
            plain(t, inp);
            self.mult_q(out, t, inverse);
        }
    }
}

/// The even-odd blocks of D, in four dimensions (all `ls` spinors of a site have its parity):
/// the hops `D_pq = H_pq (b − c Y)` and the part within a site `C = d+ + d− Y`; with α,
/// `(D Q)_pq = D_pq Q` and `C Q`.
impl<T: DeviceReal> EvenOdd<DeviceSpinor<T>> for CudaMdwf<T> {
    type Work = Work<T>;

    fn even_odd_work(&self) -> Self::Work {
        [None, None]
    }

    fn hop(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        to: Parity,
        dagger: Dagger,
    ) {
        let MdwfParams { b, c, .. } = self.params;
        let [h, t] = work;
        match dagger {
            Dagger::No => self.with_q(t, out, inp, true, false, |out, inp| {
                self.hops(out, inp, (b, c), false, Some(to));
            }),
            // (D†)_pq = (b − c Y†) H†_pq, and Q after it
            Dagger::Yes => self.with_q(t, out, inp, false, false, |out, inp| {
                let h = scratch(h, out);
                self.hops(h, inp, (1.0, 0.0), true, Some(to));
                self.fifth(out, (h, b, -c), None, true);
            }),
        }
    }

    fn diag(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        let (d_plus, d_minus) = self.site_coefficients();
        let adjoint = dagger == Dagger::Yes;
        // C Q, and its adjoint Q C†
        self.with_q(&mut work[1], out, inp, !adjoint, false, |out, inp| {
            self.fifth(out, (inp, d_plus, d_minus), None, adjoint);
        });
    }

    fn diag_inv(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        let adjoint = dagger == Dagger::Yes;
        // (C Q)⁻¹ = Q⁻¹ C⁻¹ and (Q C†)⁻¹ = C†⁻¹ Q⁻¹
        self.with_q(&mut work[1], out, inp, adjoint, true, |out, inp| {
            self.site_inv_plain(out, inp, adjoint);
        });
    }
}

impl<T: DeviceReal> FermionOp<DeviceSpinor<T>> for CudaMdwf<T> {
    type Work = Work<T>;

    fn work(&self) -> Self::Work {
        [None, None]
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        self.gauge.set(u, &mut self.geometry)
    }

    fn mult(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        variant: Variant,
    ) {
        let [h, t] = work;
        match variant {
            // D Q
            Variant::D => self.with_q(t, out, inp, true, false, |out, inp| {
                self.one_pass(out, inp);
            }),
            // Q D†
            Variant::Dagger => self.with_q(t, out, inp, false, false, |out, inp| {
                let MdwfParams { b, c, .. } = self.params;
                let (d_plus, d_minus) = self.site_coefficients();
                let h = scratch(h, inp);
                self.hops(h, inp, (1.0, 0.0), true, None);
                self.fifth(out, (inp, d_plus, d_minus), Some((h, b, -c)), true);
            }),
            // Γ5 D Q
            Variant::Gamma5D => self.with_q(t, out, inp, true, false, |out, inp| {
                let d = scratch(h, inp);
                self.one_pass(d, inp);
                gamma5(out, d, true);
            }),
        }
    }

    /// The count Bridge++ uses for its domain-wall operator, `(464 Nc + 240) Ls`.
    fn flops_per_site(&self) -> usize {
        (464 * NC + 240) * self.params.ls
    }
}
