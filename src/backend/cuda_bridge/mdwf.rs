//! The Möbius domain-wall operator on the device: the plain operator (no α of Neff).
//!
//! In the notation of `fermion::mdwf`, with H the hops of the Wilson kernel times (4 − M5),
//! `D = d+ + d− Y + H (b − c Y)` and `D† = d+ + d− Y† + (b − c Y†) H†`.

use super::wilson::{gamma5, scratch};
use super::*;
use crate::fermion::{Dagger, FermionOp, MdwfParams, Variant};
use crate::kernel::NC;
use crate::{Parity, Vector, C64};

pub struct CudaMdwf {
    params: MdwfParams,
    geometry: Geometry,
    gauge: DeviceGauge,
}

impl CudaMdwf {
    pub fn new(lat: &Arc<Lattice>, params: MdwfParams) -> Result<Self, CudaError> {
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, params.ls),
            gauge: DeviceGauge::new(lat)?,
        })
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

    /// `out = H (b − c Y) inp`, or with `H†`, for the given b and c: on the whole lattice, or
    /// `out` on the sites of parity `to` and `inp` on the others.
    fn hops(
        &self,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        (b, c): (f64, f64),
        dagger: bool,
        to: Option<Parity>,
    ) {
        let ls = self.params.ls;
        assert!(inp.is(out, ls, to.map(Parity::flip)) && out.ls == ls);
        assert_eq!(out.data.len, inp.data.len);
        out.parity = to;
        // SAFETY: the spinor buffers hold the sites of the lattice or of one parity, as the
        // kernel is told; the links are those of the whole lattice; `out` is not `inp`.
        unsafe {
            bridge_cuda_mdwf_hop_f64(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                &self.geometry,
                self.params.mass,
                b,
                c,
                i32::from(dagger),
                to.map_or(-1, |p| p.index() as i32),
                STREAM,
            );
        }
        finish();
    }

    /// `out = (a1 + y1 Y) x + (a2 + y2 Y) z`, or with `Y†`; the second term is optional.
    fn fifth(
        &self,
        out: &mut DeviceSpinor,
        (x, a1, y1): (&DeviceSpinor, f64, f64),
        second: Option<(&DeviceSpinor, f64, f64)>,
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
            bridge_cuda_mdwf_fifth_f64(
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
                STREAM,
            );
        }
        finish();
    }

    /// `out = D_pq inp`, or `(D†)_pq inp`: `out` on the sites of parity `to`, `inp` on the
    /// others.
    pub fn hop(
        &self,
        work: &mut Option<DeviceSpinor>,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        to: Parity,
        dagger: Dagger,
    ) {
        let MdwfParams { b, c, .. } = self.params;
        match dagger {
            Dagger::No => self.hops(out, inp, (b, c), false, Some(to)),
            // (D†)_pq = (b − c Y†) H†_pq
            Dagger::Yes => {
                let h = scratch(work, out);
                self.hops(h, inp, (1.0, 0.0), true, Some(to));
                self.fifth(out, (h, b, -c), None, true);
            }
        }
    }

    /// `out = C⁻¹ inp`, or `C†⁻¹ inp`: the inverse of the part of D within a site, on the
    /// whole lattice or on one parity.
    pub fn site_inv(&self, out: &mut DeviceSpinor, inp: &DeviceSpinor, dagger: Dagger) {
        assert!(out.same_shape(inp) && inp.ls == self.params.ls);
        out.parity = inp.parity;
        let params = self.kernel_params();
        // SAFETY: both buffers hold the sites of `inp` with `ls` spinors each (padded), and
        // are not the same one.
        unsafe {
            bridge_cuda_mdwf_site_inv_f64(
                out.data.ptr,
                inp.data.ptr,
                inp.sites() as i32,
                inp.ls as i32,
                &params,
                i32::from(dagger == Dagger::Yes),
                STREAM,
            );
        }
        finish();
    }

    /// `out = (1 − D_eo C_o⁻¹ D_oe C_e⁻¹) inp` on the even sites, the Schur complement of D
    /// times C_e⁻¹, with two scratch fields.
    pub fn schur_factor(
        &self,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        scratch: &mut [DeviceSpinor; 2],
    ) {
        let [a, b] = scratch;
        self.site_inv(a, inp, Dagger::No);
        self.hop(&mut None, b, a, Parity::Odd, Dagger::No);
        self.site_inv(a, b, Dagger::No);
        self.hop(&mut None, b, a, Parity::Even, Dagger::No);
        out.copy_from(inp);
        out.axpy(&[C64::new(-1.0, 0.0)], b);
    }
}

impl FermionOp<DeviceSpinor> for CudaMdwf {
    type Work = Option<DeviceSpinor>;

    fn work(&self) -> Self::Work {
        None
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        self.gauge.set(u)
    }

    fn mult(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        variant: Variant,
    ) {
        let ls = self.params.ls;
        assert!(inp.is(out, ls, None) && out.is(inp, ls, None));
        let one_pass = |out: &mut DeviceSpinor| {
            let params = self.kernel_params();
            // SAFETY: as in `CudaWilson::apply`, the spinor buffers holding `ls` spinors
            // per site.
            unsafe {
                bridge_cuda_mdwf_dslash_1pass_f64(
                    out.data.ptr,
                    inp.data.ptr,
                    self.gauge.links.ptr,
                    &self.geometry,
                    &params,
                    STREAM,
                );
            }
            finish();
        };
        match variant {
            Variant::D => one_pass(out),
            Variant::Dagger => {
                let MdwfParams { b, c, .. } = self.params;
                let (d_plus, d_minus) = self.site_coefficients();
                let h = scratch(work, inp);
                self.hops(h, inp, (1.0, 0.0), true, None);
                self.fifth(out, (inp, d_plus, d_minus), Some((h, b, -c)), true);
            }
            Variant::Gamma5D => {
                let d = scratch(work, inp);
                one_pass(d);
                gamma5(out, d, true);
            }
        }
    }

    /// The count Bridge++ uses for its domain-wall operator, `(464 Nc + 240) Ls`.
    fn flops_per_site(&self) -> usize {
        (464 * NC + 240) * self.params.ls
    }
}
