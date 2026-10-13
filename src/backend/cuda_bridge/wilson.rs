//! The Wilson and the clover operator on the device.

use super::*;
use crate::fermion::{Clover, CloverParams, Dagger, EvenOdd, FermionOp, Variant, WilsonParams};
use crate::kernel::{NC, NS};
use crate::{Parity, Spinor, Vector};

/// Reals of the block of the clover operator within a site.
const CLOVER: usize = 72;

/// What Bridge++ counts for one application of the Wilson operator at a site (as the
/// operator on the CPU).
const WILSON_FLOPS: usize = NC * NS * (4 + 6 * (4 * NC + 2) + 2 * (4 * NC + 1));

/// The scratch field of an operator, in the shape of `like`.
pub(super) fn scratch<'w>(
    work: &'w mut Option<DeviceSpinor>,
    like: &DeviceSpinor,
) -> &'w mut DeviceSpinor {
    if !work.as_ref().is_some_and(|f| f.same_shape(like)) {
        *work = Some(like.zeros_like());
    }
    work.as_mut().expect("the field was just made")
}

/// `out = γ5 inp` (`ls` 1), or `Γ5 inp` with the fifth dimension reflected.
pub(super) fn gamma5(out: &mut DeviceSpinor, inp: &DeviceSpinor, reflect: bool) {
    assert!(out.same_shape(inp));
    out.parity = inp.parity;
    // SAFETY: two buffers of the same shape, not the same one (`out` is borrowed mutably).
    unsafe {
        bridge_cuda_gamma5_f64(
            out.data.ptr,
            inp.data.ptr,
            inp.sites() as i32,
            inp.ls as i32,
            i32::from(reflect),
            STREAM,
        );
    }
    finish();
}

/// The Wilson operator on the device.
pub struct CudaWilson {
    params: WilsonParams,
    geometry: Geometry,
    gauge: DeviceGauge,
}

impl CudaWilson {
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Result<Self, CudaError> {
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, 1),
            gauge: DeviceGauge::new(lat)?,
        })
    }

    /// `out = D inp`, or `D† inp`.
    fn apply(&self, out: &mut DeviceSpinor, inp: &DeviceSpinor, dagger: bool) {
        assert!(inp.is(out, 1, None) && out.is(inp, 1, None));
        let params = WilsonKernelParams {
            kappa: self.params.kappa,
        };
        // SAFETY: the three buffers are device memory of the sizes the kernel reads and writes
        // for this geometry (`padded(sites)` blocks of spinors and of links), `out` is not
        // `inp` (one is borrowed mutably), and the structures outlive the call.
        unsafe {
            bridge_cuda_wilson_dslash_f64(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                &self.geometry,
                &params,
                i32::from(dagger),
                STREAM,
            );
        }
        finish();
    }
}

impl FermionOp<DeviceSpinor> for CudaWilson {
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
        match variant {
            Variant::D => self.apply(out, inp, false),
            Variant::Dagger => self.apply(out, inp, true),
            Variant::Gamma5D => {
                let d = scratch(work, inp);
                self.apply(d, inp, false);
                gamma5(out, d, false);
            }
        }
    }

    fn flops_per_site(&self) -> usize {
        WILSON_FLOPS
    }
}

/// The even-odd blocks: the hops between the parities; the block within a site is 1.
impl EvenOdd<DeviceSpinor> for CudaWilson {
    type Work = ();

    fn even_odd_work(&self) {}

    fn hop(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        to: Parity,
        dagger: Dagger,
    ) {
        hops(
            (&self.gauge, &self.geometry),
            out,
            inp,
            self.params.kappa,
            FifthInHops::NONE,
            dagger == Dagger::Yes,
            Some(to),
        );
    }

    fn diag(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        _parity: Parity,
        _dagger: Dagger,
    ) {
        out.copy_from(inp);
    }

    fn diag_inv(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        _parity: Parity,
        _dagger: Dagger,
    ) {
        out.copy_from(inp);
    }
}

/// The clover operator on the device. The blocks of the operator within a site and their
/// inverses are made on the CPU, by the operator there, and sent to the device.
pub struct CudaClover {
    params: CloverParams,
    geometry: Geometry,
    gauge: DeviceGauge,
    cpu: Clover<f64>,
    term: DeviceBuffer,
    inverse: DeviceBuffer,
}

impl CudaClover {
    pub fn new(lat: &Arc<Lattice>, params: CloverParams) -> Result<Self, CudaError> {
        let blocks = padded(lat.num_sites()) * CLOVER;
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, 1),
            gauge: DeviceGauge::new(lat)?,
            cpu: Clover::new(lat, params),
            term: DeviceBuffer::zeros(blocks)?,
            inverse: DeviceBuffer::zeros(blocks)?,
        })
    }

    /// `out = D inp`, or `D† inp`.
    fn apply(&self, out: &mut DeviceSpinor, inp: &DeviceSpinor, dagger: bool) {
        assert!(inp.is(out, 1, None) && out.is(inp, 1, None));
        let params = CloverKernelParams {
            kappa: self.params.kappa,
            csw: self.params.csw,
        };
        // SAFETY: as in `CudaWilson::apply`, with the blocks of the clover term of every site.
        unsafe {
            bridge_cuda_clover_dslash_f64(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                self.term.ptr,
                &self.geometry,
                &params,
                i32::from(dagger),
                STREAM,
            );
        }
        finish();
    }

    /// `out = T inp`, the block of D within a site, or `T⁻¹ inp`, on the whole lattice or on
    /// one parity.
    pub fn site(&self, out: &mut DeviceSpinor, inp: &DeviceSpinor, inverse: bool) {
        assert!(out.same_shape(inp) && inp.ls == 1);
        out.parity = inp.parity;
        let blocks = if inverse { &self.inverse } else { &self.term };
        // SAFETY: two spinor buffers on the sites the kernel is told, and the blocks of
        // every site of the lattice.
        unsafe {
            bridge_cuda_clover_term_f64(
                out.data.ptr,
                inp.data.ptr,
                blocks.ptr,
                &self.geometry,
                parity_number(inp.parity),
                STREAM,
            );
        }
        finish();
    }
}

/// The even-odd blocks: the hops between the parities, and T, which is Hermitian.
impl EvenOdd<DeviceSpinor> for CudaClover {
    type Work = ();

    fn even_odd_work(&self) {}

    fn hop(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        to: Parity,
        dagger: Dagger,
    ) {
        hops(
            (&self.gauge, &self.geometry),
            out,
            inp,
            self.params.kappa,
            FifthInHops::NONE,
            dagger == Dagger::Yes,
            Some(to),
        );
    }

    fn diag(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        self.site(out, inp, false);
    }

    fn diag_inv(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        self.site(out, inp, true);
    }
}

impl FermionOp<DeviceSpinor> for CudaClover {
    type Work = Option<DeviceSpinor>;

    fn work(&self) -> Self::Work {
        None
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        FermionOp::<Field<f64, Spinor>>::set_gauge(&mut self.cpu, u)?;
        self.gauge.set(u)?;
        let diag = self.cpu.diag_term();
        self.term = must(DeviceBuffer::from_host(&in_blocks(diag.blocks(), CLOVER)));
        self.inverse = must(DeviceBuffer::from_host(&in_blocks(
            diag.inverse_blocks(),
            CLOVER,
        )));
        Ok(())
    }

    fn mult(
        &self,
        work: &mut Self::Work,
        out: &mut DeviceSpinor,
        inp: &DeviceSpinor,
        variant: Variant,
    ) {
        match variant {
            Variant::D => self.apply(out, inp, false),
            Variant::Dagger => self.apply(out, inp, true),
            Variant::Gamma5D => {
                let d = scratch(work, inp);
                self.apply(d, inp, false);
                gamma5(out, d, false);
            }
        }
    }

    fn flops_per_site(&self) -> usize {
        FermionOp::<Field<f64, Spinor>>::flops_per_site(&self.cpu)
    }
}
