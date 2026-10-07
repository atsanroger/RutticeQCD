//! Helpers used only by the tests: where the Bridge++ test inputs are, a reference gauge
//! transformation written independently of the library's, and comparing fields. Random
//! fields and splitting fields over processes are in the library (`ruttice::gauge::random_gauge`,
//! `ruttice::random_spinor`, `ruttice::local_part` …).
#![allow(dead_code)]

use ruttice::gauge::{link, set_link};
use ruttice::kernel::{ColorMatrix, Spinor as SiteSpinor, SPINOR_REALS};
use ruttice::{Direction, Field, Gauge, Kind, Sites, Source, Spinor, Stencil, ND};
use std::path::PathBuf;

/// Environment variable naming a directory that holds the Bridge++ test inputs
/// (`conf_04040408.txt`). Without it they are looked for in `run/` at the top of the
/// repository. Comparisons against Bridge++ are skipped when the files are in neither
/// place; they are not part of this repository.
pub const BRIDGE_TESTS_ENV: &str = "RUTTICE_BRIDGE_TESTS";

pub fn bridge_test_file(name: &str) -> Option<PathBuf> {
    let dir = match std::env::var_os(BRIDGE_TESTS_ENV) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("run"),
    };
    let path = dir.join(name);
    path.is_file().then_some(path)
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
pub fn max_abs_diff<K: Kind, S: Sites>(a: &Field<f64, K, S>, b: &Field<f64, K, S>) -> f64 {
    assert_eq!(a.as_slice().len(), b.as_slice().len());
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// `mpirun -np processes`, allowing more processes than cores. Open MPI has to be told
/// so; other launchers do not know its option.
pub fn mpirun(processes: usize) -> std::process::Command {
    let open_mpi = std::process::Command::new("mpirun")
        .arg("--version")
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("Open MPI"));
    let mut command = std::process::Command::new("mpirun");
    if open_mpi {
        command.arg("--oversubscribe");
    }
    command.args(["-np", &processes.to_string()]);
    command
}
