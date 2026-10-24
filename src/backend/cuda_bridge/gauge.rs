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
    fn bridge_cuda_stout_chain_local(
        sigma: *mut f64,
        itheta: *mut f64,
        sigma_prime: *const f64,
        u: *const f64,
        dims: *const i32,
        rho: *const f64,
        stream: *mut c_void,
    );
    fn bridge_cuda_stout_chain_staples(
        sigma: *mut f64,
        u: *const f64,
        itheta: *const f64,
        dims: *const i32,
        rho: *const f64,
        stream: *mut c_void,
    );
    fn bridge_cuda_links_extend(
        ext: *mut f64,
        local: *const f64,
        dims: *const i32,
        stream: *mut c_void,
    );
    fn bridge_cuda_links_layers(
        buf: *mut f64,
        ext: *mut f64,
        mu: i32,
        first: i32,
        pack: i32,
        dims: *const i32,
        stream: *mut c_void,
    );
}

/// A field of matrices as the kernels of `gauge/stout.cuh` read it at the neighbours of a site:
/// on one process the field itself; on several the extended field, the local lattice with a halo
/// `depth` sites deep on both sides of every direction cut between processes, corners included
/// (`gauge::paths::Extended` of the CPU). `dims` is what the kernels take: the local extents and
/// the depths.
struct Neighbourhood<'a> {
    field: &'a DeviceLinks,
    extended: Option<DeviceBuffer<f64>>,
    dims: [i32; 8],
}

impl Neighbourhood<'_> {
    fn ptr(&self) -> *const f64 {
        self.extended
            .as_ref()
            .map_or(self.field.data.ptr, |e| e.ptr)
            .cast_const()
    }
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

    /// The local extents and the depth of the halo of an extended field: `depth` in the
    /// directions cut between processes, 0 in the others (and everywhere on one process).
    fn dims_with(&self, depth: usize) -> [i32; 8] {
        core::array::from_fn(|k| {
            if k < ND {
                self.lat.local[k] as i32
            } else if self.lat.grid[k - ND] > 1 {
                depth as i32
            } else {
                0
            }
        })
    }

    /// The field with a halo `depth` sites deep, for the kernels that walk paths from a site (a
    /// staple: 1; a rectangle: 2). The halo is filled one direction after the other, the layers
    /// sent in a direction including the halos of the directions before it, so the corners are
    /// filled too: the first `depth` layers of the interior go to the neighbour behind, the last
    /// to the one ahead.
    fn with_halo(&self, depth: usize) -> Neighbourhood<'_> {
        let dims = self.dims_with(depth);
        if self.lat.comm.size() == 1 {
            return Neighbourhood {
                field: self,
                extended: None,
                dims,
            };
        }
        let local = self.lat.local;
        let d: [usize; ND] = core::array::from_fn(|mu| dims[ND + mu] as usize);
        assert!(
            (0..ND).all(|mu| local[mu] >= d[mu]),
            "a local extent {local:?} smaller than the halo depth {depth}"
        );
        let ext: [usize; ND] = core::array::from_fn(|mu| local[mu] + 2 * d[mu]);
        let mut extended = must(DeviceBuffer::<f64>::zeros(
            padded(ext.iter().product()) * SITE_REALS,
        ));
        // SAFETY: the local field and the extended one of these extents.
        unsafe { bridge_cuda_links_extend(extended.ptr, self.data.ptr, dims.as_ptr(), STREAM) };
        for mu in 0..ND {
            if d[mu] == 0 {
                continue;
            }
            let layers = d[mu]
                * (0..ND)
                    .filter(|&nu| nu != mu)
                    .map(|nu| ext[nu])
                    .product::<usize>();
            let len = padded(layers) * SITE_REALS;
            let buffers = |faces: [usize; 2]| -> Vec<Option<DeviceBuffer<f64>>> {
                (0..2 * ND)
                    .map(|f| faces.contains(&f).then(|| must(DeviceBuffer::zeros(len))))
                    .collect()
            };
            let (forward, backward) = (mu, ND + mu);
            let (send, recv) = (buffers([forward, backward]), buffers([forward, backward]));
            let layers_of = |buf: &DeviceBuffer<f64>,
                             first: usize,
                             pack: bool,
                             ext_buf: &mut DeviceBuffer<f64>| {
                // SAFETY: the buffer holds the layers of this direction; the extended field
                // has these extents.
                unsafe {
                    bridge_cuda_links_layers(
                        buf.ptr,
                        ext_buf.ptr,
                        mu as i32,
                        first as i32,
                        i32::from(pack),
                        dims.as_ptr(),
                        STREAM,
                    )
                };
            };
            let sent = |f: usize| send[f].as_ref().expect("a buffer of the direction");
            let got = |f: usize| recv[f].as_ref().expect("a buffer of the direction");
            layers_of(sent(backward), d[mu], true, &mut extended);
            layers_of(sent(forward), ext[mu] - 2 * d[mu], true, &mut extended);
            super::halo::exchange_now(&self.lat, &send, &recv);
            // What arrives from ahead is the neighbour's first layers, the halo above; from
            // behind its last, the halo below.
            layers_of(got(forward), ext[mu] - d[mu], false, &mut extended);
            layers_of(got(backward), 0, false, &mut extended);
        }
        Neighbourhood {
            field: self,
            extended: Some(extended),
            dims,
        }
    }
}

impl Stout {
    /// One step on the device: `Stout::apply`.
    pub(crate) fn apply_on_device(&self, u: &DeviceLinks) -> Result<DeviceLinks, CudaError> {
        let v = DeviceLinks::zeros(&u.lat)?;
        let (n, rho) = (u.with_halo(1), flat(&self.rho));
        // SAFETY: the links of this lattice, `n` extended as its dims say.
        unsafe {
            bridge_cuda_stout_step(v.data.ptr, n.ptr(), n.dims.as_ptr(), rho.as_ptr(), STREAM);
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
        let (n, rho) = (u.with_halo(1), flat(&self.rho));
        // SAFETY: four fields of the links of this lattice, `n` extended as its dims say.
        unsafe {
            bridge_cuda_stout_chain_local(
                sigma.data.ptr,
                itheta.data.ptr,
                sigma_prime.data.ptr,
                n.ptr(),
                n.dims.as_ptr(),
                rho.as_ptr(),
                STREAM,
            );
        }
        // i Theta at the neighbours, with the same halo.
        let t = itheta.with_halo(1);
        // SAFETY: as above.
        unsafe {
            bridge_cuda_stout_chain_staples(
                sigma.data.ptr,
                n.ptr(),
                t.ptr(),
                n.dims.as_ptr(),
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

    /// A per-site quantity summed on the host, site after site, then over the processes.
    fn site_sum(&self, per_site: &DeviceBuffer<f64>) -> f64 {
        let sites = self.lat.num_sites();
        let mut sum = [must(per_site.download())[..sites].iter().sum()];
        if self.lat.comm.size() > 1 {
            self.lat.comm.sum(&mut sum);
        }
        sum[0]
    }

    /// The sites of the whole lattice.
    fn global_sites(&self) -> f64 {
        self.lat.global.iter().product::<usize>() as f64
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
        0.5 * self.site_sum(&sums)
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
        let n = self.with_halo(1);
        // SAFETY: the links of this lattice, extended as `n` says, and a real for each site,
        // twice.
        unsafe {
            bridge_cuda_gauge_traces(plaq.ptr, rect.ptr, n.ptr(), 0, n.dims.as_ptr(), STREAM)
        };
        self.site_sum(&plaq) / (3.0 * 6.0 * self.global_sites())
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
        let with_rect = i32::from(c_rect != 0.0);
        let n = self.with_halo(if c_rect != 0.0 { 2 } else { 1 });
        // SAFETY: the links of this lattice, extended as `n` says, and a real for each site,
        // twice.
        unsafe {
            bridge_cuda_gauge_traces(
                plaq.ptr,
                rect.ptr,
                n.ptr(),
                with_rect,
                n.dims.as_ptr(),
                STREAM,
            )
        };
        let v = self.global_sites();
        let (p, r) = (self.site_sum(&plaq), self.site_sum(&rect));
        if c_rect == 0.0 && c_plaq == 1.0 {
            // As PlaquetteAction::action: β (1 − ⟨plaquette⟩) 6 V.
            beta * (1.0 - p / (3.0 * 6.0 * v)) * v * 6.0
        } else {
            beta * (c_plaq * (6.0 * v - p / 3.0) + c_rect * (12.0 * v - r / 3.0))
        }
    }
    fn gauge_force(&self, beta: f64, c_plaq: f64, c_rect: f64) -> Self {
        let f = must(Self::zeros(&self.lat));
        let n = self.with_halo(if c_rect != 0.0 { 2 } else { 1 });
        // SAFETY: the links (extended as `n` says) and the force of this lattice.
        unsafe {
            bridge_cuda_gauge_force(
                f.data.ptr,
                n.ptr(),
                beta,
                c_plaq,
                c_rect,
                n.dims.as_ptr(),
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
