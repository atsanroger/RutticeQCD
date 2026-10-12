//! Binary without a header, the format Bridge++ calls `Binary`: the reals of the text format
//! as big-endian doubles.

use super::{write_on_root, IoError, LINK_REALS};
use crate::{Field, Gauge, Lattice, ND};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const REAL: usize = std::mem::size_of::<f64>();

/// Read a gauge configuration in the binary format of Bridge++.
pub fn read_binary(path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
    let bytes = std::fs::read(path).map_err(IoError::Read)?;
    let expected = ND * lat.num_global_sites() * LINK_REALS;
    if bytes.len() != expected * REAL {
        return Err(IoError::Count {
            expected,
            found: bytes.len() / REAL,
        });
    }
    let mut u = Field::<f64, Gauge>::gauge(lat);
    let num_sites = lat.num_sites();
    let out = u.as_mut_slice();
    let (reals, _) = bytes.as_chunks::<REAL>();
    let (sites, _) = reals.as_chunks::<{ ND * LINK_REALS }>();
    for (global, links) in sites.iter().enumerate() {
        let mut rest = global;
        let x = lat.global.map(|l| {
            let x = rest % l;
            rest /= l;
            x
        });
        let Some(site) = lat.local_site(x) else {
            continue;
        };
        // The field keeps the direction outermost; the file keeps the site outermost.
        for (mu, link) in links.as_chunks::<LINK_REALS>().0.iter().enumerate() {
            for (c, value) in link.iter().enumerate() {
                out[c + LINK_REALS * (site + num_sites * mu)] = f64::from_be_bytes(*value);
            }
        }
    }
    Ok(u)
}

/// Write a gauge configuration in the same format.
pub fn write_binary(path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
    write_on_root(
        u,
        || Ok(BufWriter::new(File::create(path)?)),
        |out, _, slice| {
            for x in slice {
                out.write_all(&x.to_be_bytes())?;
            }
            Ok(())
        },
        |mut out| Ok(out.flush()?),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_in_field_order_and_checks_the_length() {
        let lat = Lattice::single_node([2, 2, 2, 2]);
        let n = ND * 16 * LINK_REALS;
        let dir = std::env::temp_dir();
        let good = dir.join(format!("ruttice_io_binary_{}.dat", std::process::id()));
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| (i as f64 * 0.5).to_be_bytes())
            .collect();
        std::fs::write(&good, &bytes).unwrap();
        let u = read_binary(&good, &lat).unwrap();
        // direction 1, site 3, component 4
        assert_eq!(u.site(1, 3)[4], (3 * 72 + 18 + 4) as f64 * 0.5);

        // Writing gives the same bytes back.
        let copy = dir.join(format!("ruttice_io_binary_copy_{}.dat", std::process::id()));
        write_binary(&copy, &u).unwrap();
        assert_eq!(std::fs::read(&copy).unwrap(), bytes);
        let _ = std::fs::remove_file(copy);

        std::fs::write(&good, &bytes[..n * REAL - 8]).unwrap();
        assert!(matches!(
            read_binary(&good, &lat),
            Err(IoError::Count { .. })
        ));
        let _ = std::fs::remove_file(good);
    }
}
