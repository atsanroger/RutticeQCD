//! The Wilson and the clover operator on the device.

use super::halo::Halos;
use super::*;
use crate::fermion::{CloverParams, Dagger, EvenOdd, FermionOp, Variant, WilsonParams};
use crate::hmc::{EvenOddForceOp, EvenOddHalves, ForceOp, Mseries};
use crate::kernel::{NC, NS};
use crate::solver::preconditioner::multigrid::CoarseOp;
use crate::{Parity, Vector};
use std::rc::Rc;

/// Reals of the block of the clover operator within a site.
const CLOVER: usize = 72;

/// What Bridge++ counts for one application of the Wilson operator at a site (as the
/// operator on the CPU).
const WILSON_FLOPS: usize = NC * NS * (4 + 6 * (4 * NC + 2) + 2 * (4 * NC + 1));

/// The same with the clover term, `Nc Nd (2 + 48 Nc)` and the `2 Nc Nd` additions that join it to
/// the hops (as the operator on the CPU).
const CLOVER_FLOPS: usize = WILSON_FLOPS + NC * NS * (2 + 48 * NC) + 2 * NC * NS;

extern "C" {
    fn bridge_cuda_clover_blocks(
        t: *mut f64,
        t_inv: *mut f64,
        ln_det: *mut f64,
        singular: *mut f64,
        u: *const f64,
        kappa_csw: f64,
        dims: *const i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_clover_insertion(
        b_lo: *mut f64,
        b_hi: *mut f64,
        zeta: *const f64,
        eta: *const f64,
        coef: f64,
        vol: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_clover_log_det_insertion(
        b_lo: *mut f64,
        b_hi: *mut f64,
        t_inv: *const f64,
        coef: f64,
        parity: i32,
        dims: *const i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_clover_leaves(
        m: *mut f64,
        u: *const f64,
        b_lo: *const f64,
        b_hi: *const f64,
        dims: *const i32,
        stream: *mut c_void,
    );
}

/// Gaussian numbers of variance ½ for every real of a field of `ls` spinors a site, drawn on the
/// host in Bridge++'s order (`hmc::fermion_action::draw`) and sent to the device.
pub(super) fn drawn(lat: &Arc<Lattice>, ls: usize, rng: &mut Mseries) -> DeviceSpinor<f64> {
    let mut host = Field::<f64, crate::Spinor>::with_shape(lat, 24, ls, 1, 1);
    rng.gauss_lex_global(lat, 24, ls, |s, x, h| {
        host.site_mut(0, x)[24 * s..24 * (s + 1)].copy_from_slice(h);
    });
    must(DeviceSpinor::from_host(&host))
}

/// M of −2 Re ζ† δD η for the hops −κ H of a Wilson-type operator on the links `gauge`
/// (`fermion/force.cuh` with one spinor a site): `ForceOp::derivative` of the CPU's operators.
/// On several processes the forward legs of η and γ5 ζ across a cut come from their faces.
fn hop_derivative(
    (gauge, geometry, halos): (&DeviceGauge<f64>, &Geometry, Option<&Halos<f64>>),
    kappa: f64,
    zeta: &DeviceSpinor<f64>,
    eta: &DeviceSpinor<f64>,
) -> DeviceLinks {
    assert!(zeta.same_shape(eta) && eta.vectors == 1 && eta.ls == 1 && eta.parity.is_none());
    let (mut g5zeta, mut g5eta) = (eta.zeros_like(), eta.zeros_like());
    gamma5(&mut g5zeta, zeta, false);
    gamma5(&mut g5eta, eta, false);
    let m = must(DeviceLinks::zeros(&gauge.lat));
    let links = gauge.links.ptr.cast_const();
    let plain = [0.0, 1.0, 0.0];
    let faces = halos.map(|h| {
        (
            h.exchanged(geometry, links, eta, plain, 1),
            h.exchanged(geometry, links, &g5zeta, plain, 2),
        )
    });
    let (h_eta, h_g5zeta) = faces
        .as_ref()
        .map_or((std::ptr::null(), std::ptr::null()), |(a, b)| {
            (a as *const _, b as *const _)
        });
    // SAFETY: four spinor fields of one vector on the lattice of the links, and a field of
    // matrices of that lattice; the faces, if any, those of η and γ5 ζ.
    unsafe {
        super::mdwf::bridge_cuda_hop_force(
            m.data.ptr,
            zeta.data.ptr,
            eta.data.ptr,
            g5zeta.data.ptr,
            g5eta.data.ptr,
            gauge.links.ptr,
            geometry.link_rows,
            geometry.dims.as_ptr(),
            geometry.bc.as_ptr(),
            1,
            -kappa,
            h_eta,
            h_g5zeta,
            STREAM,
        );
    }
    m
}

/// The even-odd fields of a Wilson-type operator on the device: one spinor a site.
macro_rules! wilson_halves {
    ($op:ty) => {
        impl EvenOddHalves for $op {
            type Half = DeviceSpinor<f64>;
            fn new_half(&self, lat: &Arc<Lattice>, parity: Parity) -> DeviceSpinor<f64> {
                must(DeviceSpinor::zeros_half(lat, 1, parity))
            }
            fn right_half(&self, out: &mut DeviceSpinor<f64>, inp: &DeviceSpinor<f64>) {
                out.copy_from(inp);
            }
            fn gamma5_half(&self, out: &mut DeviceSpinor<f64>, inp: &DeviceSpinor<f64>) {
                gamma5(out, inp, false);
            }
            fn half_of(
                &self,
                half: &mut DeviceSpinor<f64>,
                whole: &DeviceSpinor<f64>,
                parity: Parity,
            ) {
                crate::fermion::SplitParity::extract(&DeviceParities, half, whole, parity);
            }
            fn put_half(&self, whole: &mut DeviceSpinor<f64>, half: &DeviceSpinor<f64>) {
                crate::fermion::SplitParity::insert(&DeviceParities, whole, half);
            }
        }
    };
}

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
    count_fine::<T>(inp.sites(), inp.ls * inp.vectors, 0.0, 2.0, 0.0);
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
    /// How the lattice is cut between processes; none on one.
    halos: Option<Rc<Halos<T>>>,
}

impl<T: DeviceReal> CudaWilson<T> {
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Result<Self, CudaError> {
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, 1),
            gauge: DeviceGauge::new(lat)?,
            halos: Halos::of(lat, params.boundary),
        })
    }

    /// The links `u`, already on the device.
    pub fn set_gauge_on_device(&mut self, u: &DeviceLinks) -> Result<(), GaugeError> {
        self.gauge.set_on_device(u, &mut self.geometry)
    }

    /// The blocks and links of P† D P on `basis` (`coarse::coarse_blocks`): the part of D
    /// within a site is 1.
    fn coarse_blocks<L: DeviceReal>(
        &self,
        basis: &DeviceBasis,
    ) -> Result<(DeviceBuffer<f64>, DeviceBuffer<L>), CudaError> {
        super::coarse::coarse_blocks(
            basis,
            (&self.gauge, &self.geometry, self.halos.as_deref()),
            [self.params.kappa, 0.0, 1.0, 0.0],
            &|out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>| out.copy_from(inp),
        )
    }

    /// The coarse operator P† D P on the host, as `CoarseOp::build` makes it on the CPU.
    pub fn coarse_operator(&self, basis: &DeviceBasis) -> Result<CoarseOp<f64>, CudaError> {
        let (diag, links) = self.coarse_blocks::<f64>(basis)?;
        Ok(CoarseOp::from_parts(
            &basis.coarse,
            basis.coarse_dim(),
            diag.download()?,
            links.download()?,
        ))
    }

    /// The same, made and kept on the device in the precision `CT` (`even_odd` as in
    /// `DeviceCoarseOp::new`).
    pub fn coarse_operator_on_device<CT: DeviceReal>(
        &self,
        basis: &DeviceBasis,
        even_odd: bool,
    ) -> Result<DeviceCoarseOp<CT>, Box<dyn std::error::Error>> {
        let (diag, links) = self.coarse_blocks::<CT>(basis)?;
        DeviceCoarseOp::on_lattice(&basis.coarse, diag, links, basis.coarse_dim(), even_odd)
    }

    /// `out = D inp = inp + H inp`, or `D† inp`: `inp − H inp` of the hop kernel at −κ, which
    /// is the same number to the last bit.
    fn apply(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, dagger: bool) {
        assert!(inp.is(out, 1, None) && out.same_shape(inp));
        hops(
            (&self.gauge, &self.geometry, self.halos.as_deref()),
            out,
            inp,
            Some(inp),
            None,
            -self.params.kappa,
            FifthInHops::NONE,
            dagger,
            None,
        );
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
            (&self.gauge, &self.geometry, self.halos.as_deref()),
            out,
            inp,
            None,
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
/// inverses are made from the links on the device (`fermion/clover_links.cuh`).
pub struct CudaClover<T = f64> {
    params: CloverParams,
    geometry: Geometry,
    gauge: DeviceGauge<T>,
    term: DeviceBuffer<T>,
    inverse: DeviceBuffer<T>,
    halos: Option<Rc<Halos<T>>>,
    /// ln |det| of every chiral block, block w of site x at `w vol + x`: for the even-odd action.
    ln_det: DeviceBuffer<f64>,
}

impl<T: DeviceReal> CudaClover<T> {
    pub fn new(lat: &Arc<Lattice>, params: CloverParams) -> Result<Self, CudaError> {
        let sites = lat.num_sites();
        let blocks = padded(sites) * CLOVER;
        Ok(Self {
            params,
            geometry: geometry(lat, params.boundary, 1),
            gauge: DeviceGauge::new(lat)?,
            term: DeviceBuffer::zeros(blocks)?,
            inverse: DeviceBuffer::zeros(blocks)?,
            halos: Halos::of(lat, params.boundary),
            ln_det: DeviceBuffer::zeros(2 * sites)?,
        })
    }

    /// The links `u`, already on the device, and the blocks made of them.
    pub fn set_gauge_on_device(&mut self, u: &DeviceLinks) -> Result<(), GaugeError> {
        self.gauge.set_on_device(u, &mut self.geometry)?;
        self.set_blocks(u)
    }

    /// The blocks and links of P† D P on `basis` (`coarse::coarse_blocks`): the part of D
    /// within a site is T.
    fn coarse_blocks<L: DeviceReal>(
        &self,
        basis: &DeviceBasis,
    ) -> Result<(DeviceBuffer<f64>, DeviceBuffer<L>), CudaError> {
        super::coarse::coarse_blocks(
            basis,
            (&self.gauge, &self.geometry, self.halos.as_deref()),
            [self.params.kappa, 0.0, 1.0, 0.0],
            &|out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>| self.site(out, inp, false),
        )
    }

    /// The coarse operator P† D P on the host, as `CoarseOp::build` makes it on the CPU.
    pub fn coarse_operator(&self, basis: &DeviceBasis) -> Result<CoarseOp<f64>, CudaError> {
        let (diag, links) = self.coarse_blocks::<f64>(basis)?;
        Ok(CoarseOp::from_parts(
            &basis.coarse,
            basis.coarse_dim(),
            diag.download()?,
            links.download()?,
        ))
    }

    /// The same, made and kept on the device in the precision `CT` (`even_odd` as in
    /// `DeviceCoarseOp::new`).
    pub fn coarse_operator_on_device<CT: DeviceReal>(
        &self,
        basis: &DeviceBasis,
        even_odd: bool,
    ) -> Result<DeviceCoarseOp<CT>, Box<dyn std::error::Error>> {
        let (diag, links) = self.coarse_blocks::<CT>(basis)?;
        DeviceCoarseOp::on_lattice(&basis.coarse, diag, links, basis.coarse_dim(), even_odd)
    }

    /// T and T⁻¹ from the links `u` (in double precision, then in that of the operator), and
    /// ln |det| of their blocks.
    fn set_blocks(&mut self, u: &DeviceLinks) -> Result<(), GaugeError> {
        if u.lat.local != self.gauge.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let sites = u.lat.num_sites();
        let reals = padded(sites) * CLOVER;
        let t = must(DeviceBuffer::<f64>::zeros(reals));
        let t_inv = must(DeviceBuffer::<f64>::zeros(reals));
        let ln_det = must(DeviceBuffer::<f64>::zeros(2 * sites));
        let singular = must(DeviceBuffer::<f64>::zeros(1));
        let n = u.with_halo(1);
        // SAFETY: the blocks of T and of T⁻¹ of every site, two numbers a site, one number, and
        // the links extended one site deep, as `n.dims` says.
        unsafe {
            bridge_cuda_clover_blocks(
                t.ptr,
                t_inv.ptr,
                ln_det.ptr,
                singular.ptr,
                n.ptr(),
                self.params.kappa * self.params.csw,
                n.dims.as_ptr(),
                STREAM,
            );
        }
        let flag = must(singular.download())[0];
        if flag != 0.0 {
            return Err(GaugeError::SingularDiagonal {
                site: flag as usize - 1,
            });
        }
        let in_precision = |src: &DeviceBuffer<f64>| {
            let mut dst = must(DeviceBuffer::<T>::uninitialised(reals));
            convert(&mut dst, src);
            dst
        };
        self.term = in_precision(&t);
        self.inverse = in_precision(&t_inv);
        self.ln_det = ln_det;
        Ok(())
    }

    /// `out = D inp = T inp + H inp`, or `D† inp` (T is Hermitian), in one pass.
    fn apply(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, dagger: bool) {
        assert!(inp.is(out, 1, None) && out.same_shape(inp));
        hops(
            (&self.gauge, &self.geometry, self.halos.as_deref()),
            out,
            inp,
            Some(inp),
            Some(&self.term),
            self.params.kappa,
            FifthInHops::NONE,
            dagger,
            None,
        );
    }

    /// `out = T inp`, the block of D within a site, or `T⁻¹ inp`, on the whole lattice or on
    /// one parity.
    pub fn site(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, inverse: bool) {
        assert!(out.same_shape(inp) && inp.ls == 1);
        out.parity = inp.parity;
        let blocks = if inverse { &self.inverse } else { &self.term };
        let flops = (NC * NS * (2 + 48 * NC)) as f64;
        count_fine::<T>(inp.sites(), inp.vectors, flops, 2.0, CLOVER as f64);
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
            (&self.gauge, &self.geometry, self.halos.as_deref()),
            out,
            inp,
            None,
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
        self.gauge.set(u, &mut self.geometry)?;
        self.set_blocks(&must(DeviceLinks::from_host(u)))
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
        CLOVER_FLOPS
    }
}

impl CudaClover<f64> {
    /// The clover term's part of Σ w M(ζ, η) on the links `v`, with that of −2 ln |det T| on the
    /// sites of `log_det` if given: the insertions B of all pairs, then the leaves walked once.
    fn clover_part(
        &self,
        kappa_csw: f64,
        v: &DeviceLinks,
        pairs: &[(f64, &DeviceSpinor<f64>, &DeviceSpinor<f64>)],
        log_det: Option<Parity>,
    ) -> DeviceLinks {
        let lat = &v.lat;
        let sites = lat.num_sites();
        let b_lo = must(DeviceLinks::zeros(lat));
        let b_hi = must(DeviceLinks::zeros(lat));
        for &(w, zeta, eta) in pairs {
            assert!(zeta.same_shape(eta) && eta.vectors == 1 && eta.ls == 1);
            assert!(eta.parity.is_none() && eta.sites() == sites);
            // SAFETY: two fields of matrices and two spinor fields of one vector on `lat`.
            unsafe {
                bridge_cuda_clover_insertion(
                    b_lo.data.ptr,
                    b_hi.data.ptr,
                    zeta.data.ptr,
                    eta.data.ptr,
                    w * kappa_csw / 4.0,
                    sites as i32,
                    STREAM,
                );
            }
        }
        if let Some(parity) = log_det {
            // The parity of a local site from its local coordinates: that of the origin added.
            let origin = lat.global_coords(0).iter().sum::<usize>();
            let local = (parity.index() + origin) % 2;
            let dims: [i32; 8] =
                core::array::from_fn(|k| if k < 4 { lat.local[k] as i32 } else { 0 });
            // SAFETY: two fields of matrices on `lat` and the blocks of T⁻¹ of its sites.
            unsafe {
                bridge_cuda_clover_log_det_insertion(
                    b_lo.data.ptr,
                    b_hi.data.ptr,
                    self.inverse.ptr,
                    kappa_csw / 4.0,
                    local as i32,
                    dims.as_ptr(),
                    STREAM,
                );
            }
        }
        let m = must(DeviceLinks::zeros(lat));
        let (nv, nlo, nhi) = (v.with_halo(1), b_lo.with_halo(1), b_hi.with_halo(1));
        // SAFETY: the matrices of `lat`, and the links and the two insertions extended one site
        // deep alike (`nv.dims`).
        unsafe {
            bridge_cuda_clover_leaves(
                m.data.ptr,
                nv.ptr(),
                nlo.ptr(),
                nhi.ptr(),
                nv.dims.as_ptr(),
                STREAM,
            );
        }
        m
    }
}

/// The Wilson operator on the device in the pseudofermion actions of HMC: the links, the solves
/// and the derivative on the device; the pseudofermions drawn on the host in Bridge++'s order.
impl ForceOp for CudaWilson<f64> {
    type Field = DeviceSpinor<f64>;
    type Links = DeviceLinks;
    fn set_links(&mut self, u: &DeviceLinks) -> Result<(), GaugeError> {
        self.set_gauge_on_device(u)
    }
    fn derivative(&self, zeta: &DeviceSpinor<f64>, eta: &DeviceSpinor<f64>) -> DeviceLinks {
        let op = (&self.gauge, &self.geometry, self.halos.as_deref());
        hop_derivative(op, self.params.kappa, zeta, eta)
    }
    fn clover_derivative(
        &self,
        _kappa_csw: f64,
        _v: &DeviceLinks,
        _pairs: &[(f64, &DeviceSpinor<f64>, &DeviceSpinor<f64>)],
    ) -> DeviceLinks {
        unreachable!("the Wilson operator has no clover term")
    }
    fn new_field(&self, lat: &Arc<Lattice>) -> DeviceSpinor<f64> {
        must(DeviceSpinor::zeros(lat))
    }
    fn draw(&self, xi: &mut DeviceSpinor<f64>, rng: &mut Mseries) {
        *xi = drawn(&xi.lat, 1, rng);
    }
    fn gamma5(&self, _work: &mut Self::Work, out: &mut DeviceSpinor<f64>, inp: &DeviceSpinor<f64>) {
        gamma5(out, inp, false);
    }
}

/// Without a clover term ln det T is zero and has no derivative; the action asks for neither.
impl EvenOddForceOp for CudaWilson<f64> {}

wilson_halves!(CudaWilson<f64>);

/// The clover operator on the device in the actions of HMC, as `CudaWilson`; the clover term's
/// part of the force on the device too.
impl ForceOp for CudaClover<f64> {
    type Field = DeviceSpinor<f64>;
    type Links = DeviceLinks;
    fn set_links(&mut self, u: &DeviceLinks) -> Result<(), GaugeError> {
        self.set_gauge_on_device(u)
    }
    fn derivative(&self, zeta: &DeviceSpinor<f64>, eta: &DeviceSpinor<f64>) -> DeviceLinks {
        let op = (&self.gauge, &self.geometry, self.halos.as_deref());
        hop_derivative(op, self.params.kappa, zeta, eta)
    }
    fn clover_derivative(
        &self,
        kappa_csw: f64,
        v: &DeviceLinks,
        pairs: &[(f64, &DeviceSpinor<f64>, &DeviceSpinor<f64>)],
    ) -> DeviceLinks {
        self.clover_part(kappa_csw, v, pairs, None)
    }
    fn new_field(&self, lat: &Arc<Lattice>) -> DeviceSpinor<f64> {
        must(DeviceSpinor::zeros(lat))
    }
    fn draw(&self, xi: &mut DeviceSpinor<f64>, rng: &mut Mseries) {
        *xi = drawn(&xi.lat, 1, rng);
    }
    fn gamma5(&self, _work: &mut Self::Work, out: &mut DeviceSpinor<f64>, inp: &DeviceSpinor<f64>) {
        gamma5(out, inp, false);
    }
}

impl EvenOddForceOp for CudaClover<f64> {
    fn log_det_diag(&self, lat: &Arc<Lattice>, parity: Parity) -> f64 {
        let ln_det = must(self.ln_det.download());
        let sites = lat.num_sites();
        let mut sum = [0.0];
        for x in 0..sites {
            if lat.global_coords(x).iter().sum::<usize>() % 2 == parity.index() {
                sum[0] += ln_det[x] + ln_det[sites + x];
            }
        }
        lat.comm.sum(&mut sum);
        sum[0]
    }
    fn clover_derivative_eo(
        &self,
        kappa_csw: f64,
        v: &DeviceLinks,
        parity: Parity,
        y: &DeviceSpinor<f64>,
        x: &DeviceSpinor<f64>,
    ) -> DeviceLinks {
        self.clover_part(kappa_csw, v, &[(1.0, y, x)], Some(parity))
    }
}

wilson_halves!(CudaClover<f64>);
