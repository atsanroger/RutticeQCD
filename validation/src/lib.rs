//! Helpers for the validation tests (DESIGN.md §8).
//!
//! The random numbers here only build test inputs. They are not the generators of the
//! physics code and make no attempt to match any other program.

use ruttice_core::{Direction, Field, Gauge, Lattice, Source, Spinor, Stencil, ND};
use ruttice_gauge::{link, set_link};
use ruttice_kernel::{ColorMatrix, Cplx, Spinor as SiteSpinor, SPINOR_REALS};
use std::path::PathBuf;
use std::sync::Arc;

/// Environment variable naming a directory that holds the Bridge++ test inputs
/// (`conf_04040408.txt`). Comparisons against Bridge++ are skipped when it is not set;
/// those files are not part of this repository.
pub const BRIDGE_TESTS_ENV: &str = "RUTTICE_BRIDGE_TESTS";

pub fn bridge_test_file(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(BRIDGE_TESTS_ENV)?).join(name);
    path.is_file().then_some(path)
}

/// SplitMix64.
pub struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (-1, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

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

/// U_mu(x) -> G(x) U_mu(x) G(x+mu)^†, with periodic neighbours.
pub fn gauge_transform(
    u: &Field<f64, Gauge>,
    g: &[ColorMatrix<f64>],
    st: &Stencil,
) -> Field<f64, Gauge> {
    let mut out = Field::<f64, Gauge>::gauge(u.lattice());
    for mu in 0..ND {
        for site in 0..u.num_sites() {
            let Source::Local { site: n, .. } = st.source(site, mu, Direction::Forward) else {
                panic!("gauge_transform needs periodic neighbours");
            };
            set_link(
                &mut out,
                mu,
                site,
                &g[site].mul(&link(u, mu, site)).mul(&g[n].adj()),
            );
        }
    }
    out
}

/// `num_vectors` spinor fields with every real uniform in (-1, 1).
pub fn random_spinor(lat: &Arc<Lattice>, num_vectors: usize, seed: u64) -> Field<f64, Spinor> {
    let mut rng = TestRng::new(seed);
    let mut psi = Field::<f64, Spinor>::spinor(lat, num_vectors);
    for x in psi.as_mut_slice() {
        *x = rng.uniform();
    }
    psi
}

/// ψ(x) -> G(x) ψ(x).
pub fn gauge_transform_spinor(
    psi: &Field<f64, Spinor>,
    g: &[ColorMatrix<f64>],
) -> Field<f64, Spinor> {
    let mut out = Field::<f64, Spinor>::spinor(psi.lattice(), psi.num_vectors());
    for (site, g) in g.iter().enumerate() {
        let src = psi.site(0, site).as_chunks::<SPINOR_REALS>().0;
        let dst = out.site_mut(0, site).as_chunks_mut::<SPINOR_REALS>().0;
        for (o, s) in dst.iter_mut().zip(src) {
            SiteSpinor::from_reals(s).color_mul(g).write_reals(o);
        }
    }
    out
}

/// Largest absolute difference between two fields of the same shape.
pub fn max_abs_diff<K: ruttice_core::Kind, S: ruttice_core::Sites>(
    a: &Field<f64, K, S>,
    b: &Field<f64, K, S>,
) -> f64 {
    assert_eq!(a.as_slice().len(), b.as_slice().len());
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}
