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
pub(super) fn scratch<'w, T: DeviceReal>(
    work: &'w mut Option<DeviceSpinor<T>>,
    like: &DeviceSpinor<T>,
) -> &'w mut DeviceSpinor<T> {
    if !work.as_ref().is_some_and(|f| f.same_shape(like)) {
        *work = Some(like.zeros_like());
    }
    work.as_mut().expect("the field was just made")
}

/// `out = γ5 inp` (`ls` 1), or `Γ5 inp` with the fifth dimension reflected. The vectors of a
/// field are whole blocks of the device layout one after the other: more sites to the kernel.
pub(super) fn gamma5<T: DeviceReal>(
    out: &mut DeviceSpinor<T>,
    inp: &DeviceSpinor<T>,
    reflect: bool,
) {
    assert!(out.same_shape(inp));
    out.parity = inp.parity;
    // SAFETY: two buffers of the same shape, not the same one (`out` is borrowed mutably).
    unsafe {
        T::gamma5(
            out.data.ptr,
            inp.data.ptr,
            (inp.vectors * padded(inp.sites())) as i32,
            inp.ls as i32,
            i32::from(reflect),
        );
    }
}

/// The Wilson operator on the device.
pub struct CudaWilson<T = f64> {
    params: WilsonParams,
    geometry: Geometry,
    gauge: DeviceGauge<T>,
}

impl<T: DeviceReal> CudaWilson<T> {
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Result<Self, CudaError> {
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, 1),
            gauge: DeviceGauge::new(lat)?,
        })
    }

    /// `out = D inp`, or `D† inp`.
    fn apply(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, dagger: bool) {
        assert!(inp.is(out, 1, None) && out.is(inp, 1, None) && out.same_shape(inp));
        let params = WilsonKernelParams {
            kappa: self.params.kappa,
        };
        // SAFETY: the three buffers are device memory of the sizes the kernel reads and writes
        // for this geometry (`padded(sites)` blocks of spinors and of links), `out` is not
        // `inp` (one is borrowed mutably), and the structures outlive the call.
        unsafe {
            T::wilson_dslash(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                &self.geometry,
                &params,
                i32::from(dagger),
                inp.vectors as i32,
            );
        }
    }
}

impl<T: DeviceReal> FermionOp<DeviceSpinor<T>> for CudaWilson<T> {
    type Work = Option<DeviceSpinor<T>>;

    fn work(&self) -> Self::Work {
        None
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
impl<T: DeviceReal> EvenOdd<DeviceSpinor<T>> for CudaWilson<T> {
    type Work = ();

    fn even_odd_work(&self) {}

    fn hop(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        to: Parity,
        dagger: Dagger,
    ) {
        hops(
            (&self.gauge, &self.geometry),
            out,
            inp,
            None,
            self.params.kappa,
            FifthInHops::NONE,
            dagger == Dagger::Yes,
            Some(to),
        );
    }

    fn diag(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        _parity: Parity,
        _dagger: Dagger,
    ) {
        out.copy_from(inp);
    }

    fn diag_inv(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        _parity: Parity,
        _dagger: Dagger,
    ) {
        out.copy_from(inp);
    }
}

/// The clover operator on the device. The blocks of the operator within a site and their
/// inverses are made on the CPU, by the operator there, and sent to the device.
pub struct CudaClover<T = f64> {
    params: CloverParams,
    geometry: Geometry,
    gauge: DeviceGauge<T>,
    cpu: Clover<f64>,
    term: DeviceBuffer<T>,
    inverse: DeviceBuffer<T>,
}

impl<T: DeviceReal> CudaClover<T> {
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
    fn apply(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, dagger: bool) {
        assert!(inp.is(out, 1, None) && out.is(inp, 1, None) && out.same_shape(inp));
        let params = CloverKernelParams {
            kappa: self.params.kappa,
            csw: self.params.csw,
        };
        // SAFETY: as in `CudaWilson::apply`, with the blocks of the clover term of every site.
        unsafe {
            T::clover_dslash(
                out.data.ptr,
                inp.data.ptr,
                self.gauge.links.ptr,
                self.term.ptr,
                &self.geometry,
                &params,
                i32::from(dagger),
                inp.vectors as i32,
            );
        }
    }

    /// `out = T inp`, the block of D within a site, or `T⁻¹ inp`, on the whole lattice or on
    /// one parity.
    pub fn site(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, inverse: bool) {
        assert!(out.same_shape(inp) && inp.ls == 1);
        out.parity = inp.parity;
        let blocks = if inverse { &self.inverse } else { &self.term };
        // SAFETY: two spinor buffers on the sites the kernel is told, and the blocks of
        // every site of the lattice.
        unsafe {
            T::clover_term(
                out.data.ptr,
                inp.data.ptr,
                blocks.ptr,
                &self.geometry,
                parity_number(inp.parity),
                inp.vectors as i32,
            );
        }
    }
}

/// The even-odd blocks: the hops between the parities, and T, which is Hermitian.
impl<T: DeviceReal> EvenOdd<DeviceSpinor<T>> for CudaClover<T> {
    type Work = ();

    fn even_odd_work(&self) {}

    fn hop(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        to: Parity,
        dagger: Dagger,
    ) {
        hops(
            (&self.gauge, &self.geometry),
            out,
            inp,
            None,
            self.params.kappa,
            FifthInHops::NONE,
            dagger == Dagger::Yes,
            Some(to),
        );
    }

    fn diag(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        self.site(out, inp, false);
    }

    fn diag_inv(
        &self,
        _work: &mut (),
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity, Some(parity));
        self.site(out, inp, true);
    }
}

impl<T: DeviceReal> FermionOp<DeviceSpinor<T>> for CudaClover<T> {
    type Work = Option<DeviceSpinor<T>>;

    fn work(&self) -> Self::Work {
        None
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        FermionOp::<Field<f64, Spinor>>::set_gauge(&mut self.cpu, u)?;
        self.gauge.set(u, &mut self.geometry)?;
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
        out: &mut DeviceSpinor<T>,
        inp: &DeviceSpinor<T>,
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
