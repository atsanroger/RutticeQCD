//! Random gauge fields for checks and examples (see `field::random` for the generator).

use super::set_link;
use crate::kernel::{ColorMatrix, Cplx};
use crate::{Field, Gauge, Lattice, TestRng, ND};
use std::sync::Arc;

/// A random SU(3) matrix: two orthonormalised random rows, third row from their cross product.
pub fn random_su3(rng: &mut TestRng) -> ColorMatrix<f64> {
    type Row = [Cplx<f64>; 3];
    let dot = |a: &Row, b: &Row| a[0].conj() * b[0] + a[1].conj() * b[1] + a[2].conj() * b[2];
    let unit = |a: &mut Row| {
        let n = dot(a, a).re.sqrt();
        for z in a.iter_mut() {
            *z = z.scale(1.0 / n);
        }
    };
    let mut random_row =
        || -> Row { core::array::from_fn(|_| Cplx::new(rng.uniform(), rng.uniform())) };
    let mut r0 = random_row();
    let mut r1 = random_row();
    unit(&mut r0);
    let p = dot(&r0, &r1);
    for k in 0..3 {
        r1[k] = r1[k] - p * r0[k];
    }
    unit(&mut r1);
    let r2 = [
        (r0[1] * r1[2] - r0[2] * r1[1]).conj(),
        (r0[2] * r1[0] - r0[0] * r1[2]).conj(),
        (r0[0] * r1[1] - r0[1] * r1[0]).conj(),
    ];
    ColorMatrix([r0, r1, r2])
}

/// Every link an independent random SU(3) matrix.
pub fn random_gauge(lat: &Arc<Lattice>, seed: u64) -> Field<f64, Gauge> {
    let mut rng = TestRng::new(seed);
    let mut u = Field::<f64, Gauge>::gauge(lat);
    for mu in 0..ND {
        for site in 0..lat.num_sites() {
            set_link(&mut u, mu, site, &random_su3(&mut rng));
        }
    }
    u
}
