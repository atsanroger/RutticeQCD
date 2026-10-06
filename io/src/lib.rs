//! Reading and writing gauge configurations.

use ruttice_core::{Field, Gauge, Lattice, ND};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub enum IoError {
    Read(std::io::Error),
    Parse { index: usize, token: String },
    Count { expected: usize, found: usize },
    Unsupported(&'static str),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Read(e) => write!(f, "cannot read file: {e}"),
            IoError::Parse { index, token } => {
                write!(f, "value {index} is not a number: {token:?}")
            }
            IoError::Count { expected, found } => {
                write!(f, "expected {expected} values, found {found}")
            }
            IoError::Unsupported(what) => write!(f, "not supported yet: {what}"),
        }
    }
}

impl std::error::Error for IoError {}

/// Read a gauge configuration in the plain-text format used by Bridge++ (one real per
/// line).
///
/// File order, outermost first: global site (x fastest), direction, then the 18 reals of
/// the link, row-major with the real part first (the ILDG ordering). This interoperates
/// with the file format only; the reader is written from that description.
pub fn read_bridge_text(path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
    if lat.local != lat.global {
        return Err(IoError::Unsupported(
            "reading a configuration on more than one process",
        ));
    }
    let text = std::fs::read_to_string(path).map_err(IoError::Read)?;
    let mut u = Field::<f64, Gauge>::gauge(lat);
    let num_sites = lat.num_sites();
    let expected = ND * num_sites * 18;
    let out = u.as_mut_slice();
    let mut found = 0;
    for token in text.split_ascii_whitespace() {
        if found < expected {
            // The field keeps the direction outermost; the file keeps the site outermost.
            let (site, mu, c) = (found / (ND * 18), (found / 18) % ND, found % 18);
            out[c + 18 * (site + num_sites * mu)] = token.parse().map_err(|_| IoError::Parse {
                index: found,
                token: token.to_string(),
            })?;
        }
        found += 1;
    }
    if found != expected {
        return Err(IoError::Count { expected, found });
    }
    Ok(u)
}

/// Write a gauge configuration in the same plain-text format, one real per line, with
/// enough digits that reading it back gives the same bits.
pub fn write_bridge_text(path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
    use std::fmt::Write;
    let lat = u.lattice();
    if lat.local != lat.global {
        return Err(IoError::Unsupported(
            "writing a configuration on more than one process",
        ));
    }
    let num_sites = lat.num_sites();
    let mut text = String::with_capacity(ND * num_sites * 18 * 25);
    for site in 0..num_sites {
        for mu in 0..ND {
            for x in u.site(mu, site) {
                // Writing to a String cannot fail.
                let _ = writeln!(text, "{x:.16e}");
            }
        }
    }
    std::fs::write(path, text).map_err(IoError::Read)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_in_field_order_and_checks_count() {
        let lat = Lattice::single_node([2, 2, 2, 2]);
        let n = ND * 16 * 18;
        let dir = std::env::temp_dir();
        let good = dir.join("ruttice_io_test_good.txt");
        let text: String = (0..n)
            .map(|i| format!("{:.14e}\n", i as f64 * 0.5))
            .collect();
        std::fs::write(&good, &text).unwrap();
        let u = read_bridge_text(&good, &lat).unwrap();
        // direction 1, site 3, component 4
        assert_eq!(u.site(1, 3)[4], (3 * 72 + 18 + 4) as f64 * 0.5);

        // Writing and reading back is exact.
        let copy = dir.join("ruttice_io_test_copy.txt");
        let mut v = u;
        for (i, x) in v.as_mut_slice().iter_mut().enumerate() {
            *x = (i as f64 * 0.37).sin() / 3.0;
        }
        write_bridge_text(&copy, &v).unwrap();
        assert_eq!(
            read_bridge_text(&copy, &lat).unwrap().as_slice(),
            v.as_slice()
        );
        let _ = std::fs::remove_file(copy);

        let short = dir.join("ruttice_io_test_short.txt");
        std::fs::write(&short, "1.0\n2.0\n").unwrap();
        assert!(matches!(
            read_bridge_text(&short, &lat),
            Err(IoError::Count { found: 2, .. })
        ));
        let _ = std::fs::remove_file(good);
        let _ = std::fs::remove_file(short);
    }
}
