//! Gauge links on the device, whole matrices in double precision, and stout smearing of them
//! with the chain rule of its force (`gauge/stout.cuh`): `gauge::smearing` of the CPU, step
//! for step.

use super::*;
use crate::gauge::smearing::{Stout, Weights};
use crate::gauge::{GaugeSmearing, Smear};
use std::error::Error;

extern "C" {
    fn bridge_cuda_links_axpy(y: *mut f64, a: f64, x: *const f64, n: usize, stream: *mut c_void);
    fn bridge_cuda_links_exp_update(
        u: *mut f64,
        p: *const f64,
        eps: f64,
        order: i32,
        vol: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_links_ta(out: *mut f64, m: *const f64, s: f64, vol: i32, stream: *mut c_void);
    fn bridge_cuda_links_mul(
        out: *mut f64,
        a: *const f64,
        b: *const f64,
        adjoint_a: i32,
        vol: i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_gauge_force(
        f: *mut f64,
        u: *const f64,
        beta: f64,
        c_plaq: f64,
        c_rect: f64,
        dims: *const i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_gauge_traces(
        plaq: *mut f64,
        rect: *mut f64,
        u: *const f64,
        with_rect: i32,
        dims: *const i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_links_norm2(out: *mut f64, m: *const f64, vol: i32, stream: *mut c_void);
    fn bridge_cuda_stout_step(
        v: *mut f64,
        u: *const f64,
        dims: *const i32,
        rho: *const f64,
        stream: *mut c_void,
    );
    fn bridge_cuda_stout_chain(
        sigma: *mut f64,
        itheta: *mut f64,
        sigma_prime: *const f64,
        u: *const f64,
        dims: *const i32,
        rho: *const f64,
        stream: *mut c_void,
    );
}

/// Reals of the matrices of one site, four directions of 18.
const SITE_REALS: usize = 4 * 18;

/// A field of 3 × 3 complex matrices, one for each link, on the device: links, or the
/// matrices of a force. Real c of the matrix of direction μ at site x is at `at(18 μ + c, x,
/// 72)`: the layout of the links of the operators kept whole (three rows).
pub struct DeviceLinks {
    pub(super) lat: Arc<Lattice>,
    pub(super) data: DeviceBuffer<f64>,
}

impl DeviceLinks {
    pub fn zeros(lat: &Arc<Lattice>) -> Result<Self, CudaError> {
        assert_eq!(lat.comm.size(), 1, "the CUDA backend runs on one process");
        Ok(Self {
            lat: lat.clone(),
            data: DeviceBuffer::zeros(padded(lat.num_sites()) * SITE_REALS)?,
        })
    }

    pub fn from_host(u: &Field<f64, Gauge>) -> Result<Self, CudaError> {
        let lat = u.lattice();
        let sites = lat.num_sites();
        let mut host = vec![0.0; padded(sites) * SITE_REALS];
        for mu in 0..ND {
            for x in 0..sites {
                for (c, v) in u.site(mu, x)[..18].iter().enumerate() {
                    host[at(18 * mu + c, x, SITE_REALS)] = *v;
                }
            }
        }
        Ok(Self {
            lat: lat.clone(),
            data: DeviceBuffer::from_host(&host)?,
        })
    }

    pub fn to_host(&self) -> Result<Field<f64, Gauge>, CudaError> {
        let host = self.data.download()?;
        let mut u = Field::<f64, Gauge>::gauge(&self.lat);
        for mu in 0..ND {
            for x in 0..self.lat.num_sites() {
                for (c, v) in u.site_mut(mu, x)[..18].iter_mut().enumerate() {
                    *v = host[at(18 * mu + c, x, SITE_REALS)];
                }
            }
        }
        Ok(u)
    }

    pub fn lattice(&self) -> &Arc<Lattice> {
        &self.lat
    }

    fn dims(&self) -> [i32; ND] {
        self.lat.local.map(|l| l as i32)
    }
}

impl Stout {
    /// One step on the device: `Stout::apply`.
    pub fn apply_on_device(&self, u: &DeviceLinks) -> Result<DeviceLinks, CudaError> {
        let v = DeviceLinks::zeros(&u.lat)?;
        let (dims, rho) = (u.dims(), flat(&self.rho));
        // SAFETY: two fields of the links of this lattice.
        unsafe {
            bridge_cuda_stout_step(v.data.ptr, u.data.ptr, dims.as_ptr(), rho.as_ptr(), STREAM);
        }
        Ok(v)
    }

    /// One step of the chain rule on the device: `Stout::chain`.
    pub fn chain_on_device(
        &self,
        sigma_prime: &DeviceLinks,
        u: &DeviceLinks,
    ) -> Result<DeviceLinks, CudaError> {
        let (sigma, itheta) = (DeviceLinks::zeros(&u.lat)?, DeviceLinks::zeros(&u.lat)?);
        let (dims, rho) = (u.dims(), flat(&self.rho));
        // SAFETY: four fields of the links of this lattice.
        unsafe {
            bridge_cuda_stout_chain(
                sigma.data.ptr,
                itheta.data.ptr,
                sigma_prime.data.ptr,
                u.data.ptr,
                dims.as_ptr(),
                rho.as_ptr(),
                STREAM,
            );
        }
        Ok(sigma)
    }
}

fn flat(rho: &Weights) -> [f64; 16] {
    core::array::from_fn(|k| rho[k / 4][k % 4])
}

fn stout_of(s: &GaugeSmearing) -> Result<Stout, Box<dyn Error>> {
    match s.smear {
        Smear::Stout(stout) => Ok(stout),
        Smear::Ape(_) => Err("APE smearing is not on the device; stout is".into()),
    }
}

impl GaugeSmearing {
    /// `levels` on the device: the smeared fields after each step.
    pub fn levels_on_device(&self, u: &DeviceLinks) -> Result<Vec<DeviceLinks>, Box<dyn Error>> {
        let stout = stout_of(self)?;
        let mut levels: Vec<DeviceLinks> = Vec::with_capacity(self.steps);
        for _ in 0..self.steps {
            let next = stout.apply_on_device(levels.last().unwrap_or(u))?;
            levels.push(next);
        }
        Ok(levels)
    }

    /// `chain` on the device: Σ of the links `u` from Σ' of the smeared ones.
    pub fn chain_on_device(
        &self,
        sigma: DeviceLinks,
        u: &DeviceLinks,
        levels: &[DeviceLinks],
    ) -> Result<DeviceLinks, Box<dyn Error>> {
        let stout = stout_of(self)?;
        let mut sigma = sigma;
        for k in (0..self.steps).rev() {
            let below = if k == 0 { u } else { &levels[k - 1] };
            sigma = stout.chain_on_device(&sigma, below)?;
        }
        Ok(sigma)
    }
}

impl DeviceLinks {
    fn vol(&self) -> i32 {
        self.lat.num_sites() as i32
    }

    /// out = a b, or a† b, link by link.
    fn product(&self, a: &Self, b: &Self, adjoint_a: bool) -> Self {
        let out = must(Self::zeros(&self.lat));
        // SAFETY: three fields of matrices of this lattice.
        unsafe {
            bridge_cuda_links_mul(
                out.data.ptr,
                a.data.ptr,
                b.data.ptr,
                i32::from(adjoint_a),
                self.vol(),
                STREAM,
            );
        }
        out
    }

    /// s TA(self).
    fn projected(&self, s: f64) -> Self {
        let out = must(Self::zeros(&self.lat));
        // SAFETY: two fields of matrices of this lattice.
        unsafe { bridge_cuda_links_ta(out.data.ptr, self.data.ptr, s, self.vol(), STREAM) };
        out
    }

    /// A per-site quantity summed on the host, site after site.
    fn site_sum(per_site: &DeviceBuffer<f64>, sites: usize) -> f64 {
        must(per_site.download())[..sites].iter().sum()
    }
}

/// HMC with the links and the momenta on the device: what the integrator, the gauge actions and
/// the smeared fermion actions do with them (`gauge/links.cuh`, `gauge/stout.cuh`). The
/// momenta are drawn on the host in Bridge++'s order and go over; sums are taken on the host
/// over the sums of the sites.
impl crate::hmc::LinkField for DeviceLinks {
    fn lattice(&self) -> &Arc<Lattice> {
        &self.lat
    }
    fn zeros_like(&self) -> Self {
        must(Self::zeros(&self.lat))
    }
    fn duplicate(&self) -> Self {
        let mut v = must(Self::zeros(&self.lat));
        must(v.data.copy_from(&self.data));
        v
    }
    fn add_scaled(&mut self, a: f64, x: &Self) {
        // SAFETY: two fields of matrices of this lattice, of `len` reals each.
        unsafe { bridge_cuda_links_axpy(self.data.ptr, a, x.data.ptr, self.data.len, STREAM) };
    }
    fn kinetic_energy(&self) -> f64 {
        let sums = must(DeviceBuffer::<f64>::zeros(self.lat.num_sites()));
        // SAFETY: a field of matrices and a real for each site.
        unsafe { bridge_cuda_links_norm2(sums.ptr, self.data.ptr, self.vol(), STREAM) };
        0.5 * Self::site_sum(&sums, self.lat.num_sites())
    }
    fn exp_update(&mut self, eps: f64, p: &Self, order: usize) {
        // SAFETY: the links and the momenta of this lattice.
        unsafe {
            bridge_cuda_links_exp_update(
                self.data.ptr,
                p.data.ptr,
                eps,
                order as i32,
                self.vol(),
                STREAM,
            );
        }
    }
    fn plaquette(&self) -> f64 {
        let sites = self.lat.num_sites();
        let (plaq, rect) = (
            must(DeviceBuffer::<f64>::zeros(sites)),
            must(DeviceBuffer::<f64>::zeros(sites)),
        );
        let dims = self.dims();
        // SAFETY: the links of this lattice and a real for each site, twice.
        unsafe {
            bridge_cuda_gauge_traces(plaq.ptr, rect.ptr, self.data.ptr, 0, dims.as_ptr(), STREAM)
        };
        Self::site_sum(&plaq, sites) / (3.0 * 6.0 * sites as f64)
    }
    fn draw_momenta(lat: &Arc<Lattice>, rng: &mut crate::hmc::Mseries) -> Self {
        must(Self::from_host(&crate::hmc::draw_momenta(lat, rng)))
    }
    fn smear_levels(&self, smearing: &GaugeSmearing) -> Vec<Self> {
        smearing
            .levels_on_device(self)
            .unwrap_or_else(|e| panic!("smearing on the device: {e}"))
    }
    fn smeared_force(
        m_v: &Self,
        u: &Self,
        smearing: Option<&GaugeSmearing>,
        levels: &[Self],
    ) -> Self {
        let Some(smearing) = smearing.filter(|s| s.steps > 0) else {
            return m_v.projected(1.0);
        };
        let v = levels.last().expect("the smeared links");
        // Σ' = V† M, carried back through the steps; M of the thin links U Σ.
        let sigma = u.product(v, m_v, true);
        let sigma = smearing
            .chain_on_device(sigma, u, levels)
            .unwrap_or_else(|e| panic!("the chain rule on the device: {e}"));
        u.product(u, &sigma, false).projected(1.0)
    }
    fn gauge_action(&self, beta: f64, c_plaq: f64, c_rect: f64) -> f64 {
        let sites = self.lat.num_sites();
        let (plaq, rect) = (
            must(DeviceBuffer::<f64>::zeros(sites)),
            must(DeviceBuffer::<f64>::zeros(sites)),
        );
        let dims = self.dims();
        let with_rect = i32::from(c_rect != 0.0);
        // SAFETY: the links of this lattice and a real for each site, twice.
        unsafe {
            bridge_cuda_gauge_traces(
                plaq.ptr,
                rect.ptr,
                self.data.ptr,
                with_rect,
                dims.as_ptr(),
                STREAM,
            )
        };
        let v = sites as f64;
        let (p, r) = (Self::site_sum(&plaq, sites), Self::site_sum(&rect, sites));
        if c_rect == 0.0 && c_plaq == 1.0 {
            // As PlaquetteAction::action: β (1 − ⟨plaquette⟩) 6 V.
            beta * (1.0 - p / (3.0 * 6.0 * v)) * v * 6.0
        } else {
            beta * (c_plaq * (6.0 * v - p / 3.0) + c_rect * (12.0 * v - r / 3.0))
        }
    }
    fn gauge_force(&self, beta: f64, c_plaq: f64, c_rect: f64) -> Self {
        let f = must(Self::zeros(&self.lat));
        let dims = self.dims();
        // SAFETY: the links and the force of this lattice.
        unsafe {
            bridge_cuda_gauge_force(
                f.data.ptr,
                self.data.ptr,
                beta,
                c_plaq,
                c_rect,
                dims.as_ptr(),
                STREAM,
            );
        }
        f
    }
    fn to_host(&self) -> Field<f64, Gauge> {
        must(DeviceLinks::to_host(self))
    }
    fn from_host(u: &Field<f64, Gauge>) -> Self {
        must(DeviceLinks::from_host(u))
    }
}
