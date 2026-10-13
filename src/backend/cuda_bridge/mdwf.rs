//! The Möbius domain-wall operator on the device: the plain operator (no α of Neff).
//!
//! In the notation of `fermion::mdwf`, with H the hops of the Wilson kernel times (4 − M5),
//! `D = d+ + d− Y + H (b − c Y)` and `D† = d+ + d− Y† + (b − c Y†) H†`.

use super::wilson::{gamma5, scratch};
use super::*;
use crate::fermion::{Dagger, EvenOdd, FermionOp, MdwfParams, Variant};
use crate::kernel::NC;
use crate::{Parity, Vector};

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
}

/// The even-odd blocks of D, in four dimensions (all `ls` spinors of a site have its parity):
/// the hops `D_pq = H_pq (b − c Y)` and the part within a site `C = d+ + d− Y`.
impl EvenOdd<DeviceSpinor> for CudaMdwf {
    type Work = Option<DeviceSpinor>;

    fn even_odd_work(&self) -> Self::Work {
        None
    }

    fn hop(
        &self,
        work: &mut Self::Work,
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

    fn diag(
        &self,
        _work: &mut Self::Work,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        let (d_plus, d_minus) = self.site_coefficients();
        self.fifth(out, (inp, d_plus, d_minus), None, dagger == Dagger::Yes);
    }

    /// Exactly, by substitution along the fifth dimension.
    fn diag_inv(
        &self,
        _work: &mut Self::Work,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        parity: Parity,
        dagger: Dagger,
    ) {
        assert!(out.same_shape(inp) && inp.ls == self.params.ls);
        assert_eq!(inp.parity, Some(parity));
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
