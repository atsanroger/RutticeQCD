//! Sources of point-to-all propagators: twelve vectors, one for each spin and colour.

use super::NUM_SOURCES;
use crate::kernel::{Number, SPINOR_REALS};
use crate::{Field, Lattice, Spinor, Storage, ND};
use std::sync::Arc;

/// Twelve unit sources at one site: vector `k` has a real 1 in component `k`. `position`
/// is in global coordinates; the process that holds the site sets it.
pub fn point_source<T: Storage>(lat: &Arc<Lattice>, position: [usize; ND]) -> Field<T, Spinor> {
    for mu in 0..ND {
        assert!(
            position[mu] < lat.global[mu],
            "source position outside the lattice"
        );
    }
    let mut b = Field::<T, Spinor>::spinor(lat, NUM_SOURCES);
    if let Some(site) = lat.local_site(position) {
        let site = b.site_mut(0, site);
        for k in 0..NUM_SOURCES {
            site[k * SPINOR_REALS + 2 * k] = T::store(Number::from_f64(1.0));
        }
    }
    b
}

/// Twelve smeared sources on one time slice: vector `k` has the same real profile in
/// component `k` at every site of the time slice of `position`.
pub fn exponential_source<T: Storage>(
    lat: &Arc<Lattice>,
    position: [usize; ND],
    slope: f64,
    power: f64,
) -> Field<T, Spinor> {
    for mu in 0..ND {
        assert!(
            position[mu] < lat.global[mu],
            "source position outside the lattice"
        );
    }
    // The profile is computed on the whole time slice by every process.
    let l = [lat.global[0], lat.global[1], lat.global[2]];
    let mut profile = vec![0.0f64; l[0] * l[1] * l[2]];
    let range = |n: usize| -(n as i64) + 1..n as i64;
    let wrap = |x0: usize, d: i64, n: usize| (x0 as i64 + d).rem_euclid(n as i64) as usize;
    for dz in range(l[2]) {
        for dy in range(l[1]) {
            for dx in range(l[0]) {
                let r = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                let x = wrap(position[0], dx, l[0]);
                let y = wrap(position[1], dy, l[1]);
                let z = wrap(position[2], dz, l[2]);
                profile[x + l[0] * (y + l[1] * z)] += (-slope * r.powf(power)).exp();
            }
        }
    }
    let norm = profile.iter().map(|w| w * w).sum::<f64>().sqrt();

    let mut b = Field::<T, Spinor>::spinor(lat, NUM_SOURCES);
    for site in 0..lat.num_sites() {
        let [x, y, z, t] = lat.global_coords(site);
        if t != position[3] {
            continue;
        }
        let w = profile[x + l[0] * (y + l[1] * z)] / norm;
        let site = b.site_mut(0, site);
        for k in 0..NUM_SOURCES {
            site[k * SPINOR_REALS + 2 * k] = T::store(Number::from_f64(w));
        }
    }
    b
}
