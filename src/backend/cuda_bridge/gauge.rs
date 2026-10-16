//! Gauge links on the device, whole matrices in double precision, and stout smearing of them
//! with the chain rule of its force (`gauge/stout.cuh`): `gauge::smearing` of the CPU, step
//! for step.

use super::*;
use crate::gauge::smearing::{Stout, Weights};
use crate::gauge::{GaugeSmearing, Smear};
use std::error::Error;

extern "C" {
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
