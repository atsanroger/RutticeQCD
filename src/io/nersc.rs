//! The NERSC (Gauge Connection) format: a text header between `BEGIN_HEADER` and `END_HEADER`,
//! then the links, the site outermost with x running fastest, then the direction, then the
//! matrix row by row. `4D_SU3_GAUGE_3x3` holds whole matrices, `4D_SU3_GAUGE` their first two
//! rows. The header's checksum is the sum of the data as 32-bit words.
//!
//! Written from the description of the format; there is no file of another program in the
//! tests.

use super::{write_on_root, FilePrecision, IoError, LINK_REALS};
use crate::gauge::plaquette;
use crate::{Boundary, Field, Gauge, Lattice, Stencil, ND};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const END: &[u8] = b"END_HEADER\n";

fn malformed<T>(what: impl Into<String>) -> Result<T, IoError> {
    Err(IoError::Format(what.into()))
}

/// The third row of an SU(3) matrix from its first two: the conjugate of their cross product.
fn third_row(m: &mut [f64]) {
    let rows: [f64; 12] = m[..12].try_into().expect("two rows");
    let c = |i: usize| (rows[2 * i], rows[2 * i + 1]);
    let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for k in 0..3 {
        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
        let (p, q) = (mul(c(i), c(3 + j)), mul(c(j), c(3 + i)));
        m[12 + 2 * k] = p.0 - q.0;
        m[12 + 2 * k + 1] = -(p.1 - q.1);
    }
}

/// Read a gauge configuration in the NERSC format. The checksum is verified when the header
/// has one.
pub fn read_nersc(path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
    let bytes = std::fs::read(path).map_err(IoError::Read)?;
    let Some(end) = bytes.windows(END.len()).position(|w| w == END) else {
        return malformed("NERSC: no END_HEADER");
    };
    let header = String::from_utf8_lossy(&bytes[..end]);
    let value = |key: &str| -> Option<String> {
        header.lines().find_map(|line| {
            let (k, v) = line.split_once('=')?;
            (k.trim() == key).then(|| v.trim().to_string())
        })
    };
    let required = |key: &str| value(key).ok_or(IoError::Format(format!("NERSC: no {key}")));
    let mut dims = [0; ND];
    for (mu, d) in dims.iter_mut().enumerate() {
        let text = required(&format!("DIMENSION_{}", mu + 1))?;
        *d = text
            .parse()
            .map_err(|_| IoError::Format(format!("NERSC: DIMENSION_{} = {text:?}", mu + 1)))?;
    }
    if dims != lat.global {
        return Err(IoError::Lattice {
            file: dims,
            expected: lat.global,
        });
    }
    let rows = match required("DATATYPE")?.as_str() {
        "4D_SU3_GAUGE_3x3" => 3,
        "4D_SU3_GAUGE" => 2,
        other => return malformed(format!("NERSC: DATATYPE {other:?} is not supported")),
    };
    let (precision, big) = match required("FLOATING_POINT")?.as_str() {
        "IEEE64BIG" => (FilePrecision::Double, true),
        "IEEE64LITTLE" | "IEEE64" => (FilePrecision::Double, false),
        "IEEE32BIG" | "IEEE32" => (FilePrecision::Single, true),
        "IEEE32LITTLE" => (FilePrecision::Single, false),
        other => return malformed(format!("NERSC: FLOATING_POINT {other:?} is not supported")),
    };
    let data = &bytes[end + END.len()..];
    let (real, link) = (precision.bytes(), rows * 6);
    let expected = ND * lat.num_global_sites() * link;
    if data.len() != expected * real {
        return Err(IoError::Count {
            expected,
            found: data.len() / real,
        });
    }
    if let Some(text) = value("CHECKSUM") {
        let recorded = u32::from_str_radix(text.trim_start_matches("0x"), 16)
            .map_err(|_| IoError::Format(format!("NERSC: CHECKSUM = {text:?}")))?;
        let computed = data.as_chunks::<4>().0.iter().fold(0u32, |sum, w| {
            sum.wrapping_add(if big {
                u32::from_be_bytes(*w)
            } else {
                u32::from_le_bytes(*w)
            })
        });
        if recorded != computed {
            return Err(IoError::Checksum {
                recorded: (recorded, 0),
                computed: (computed, 0),
            });
        }
    }
    let load = |at: usize| -> f64 {
        let b = &data[at * real..(at + 1) * real];
        match (precision, big) {
            (FilePrecision::Double, true) => f64::from_be_bytes(b.try_into().expect("8 bytes")),
            (FilePrecision::Double, false) => f64::from_le_bytes(b.try_into().expect("8 bytes")),
            (FilePrecision::Single, true) => {
                f64::from(f32::from_be_bytes(b.try_into().expect("4 bytes")))
            }
            (FilePrecision::Single, false) => {
                f64::from(f32::from_le_bytes(b.try_into().expect("4 bytes")))
            }
        }
    };
    let mut u = Field::<f64, Gauge>::gauge(lat);
    let num_sites = lat.num_sites();
    let out = u.as_mut_slice();
    for global in 0..lat.num_global_sites() {
        let Some(site) = lat.local_site(lat.global_site_coords(global)) else {
            continue;
        };
        // The field keeps the direction outermost; the file keeps the site outermost.
        for mu in 0..ND {
            let to = &mut out[LINK_REALS * (site + num_sites * mu)..][..LINK_REALS];
            for (c, v) in to[..link].iter_mut().enumerate() {
                *v = load((global * ND + mu) * link + c);
            }
            if rows == 2 {
                third_row(to);
            }
        }
    }
    Ok(u)
}

/// Write a gauge configuration in the NERSC format: whole matrices, big-endian doubles.
pub fn write_nersc(path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
    let lat = u.lattice();
    // The sum of 32-bit words does not depend on their order: every process adds its own.
    let (mut words, mut trace) = (0u32, 0.0);
    for mu in 0..ND {
        for site in 0..lat.num_sites() {
            let m = u.site(mu, site);
            trace += m[0] + m[8] + m[16];
            for x in m {
                let b = x.to_be_bytes();
                words = words
                    .wrapping_add(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
                    .wrapping_add(u32::from_be_bytes([b[4], b[5], b[6], b[7]]));
            }
        }
    }
    let mut sums = [f64::from(words), trace];
    lat.comm.sum(&mut sums);
    let checksum = (sums[0] % 4294967296.0) as u32;
    let link_trace = sums[1] / (3 * ND * lat.num_global_sites()) as f64;
    let plaq = plaquette(u, &Stencil::nearest(lat, [Boundary::Periodic; ND]));
    let [lx, ly, lz, lt] = lat.global;
    let header = format!(
        "BEGIN_HEADER\nHDR_VERSION = 1.0\nDATATYPE = 4D_SU3_GAUGE_3x3\nSTORAGE_FORMAT = 1.0\n\
         DIMENSION_1 = {lx}\nDIMENSION_2 = {ly}\nDIMENSION_3 = {lz}\nDIMENSION_4 = {lt}\n\
         CHECKSUM = {checksum:x}\nLINK_TRACE = {link_trace:.12}\nPLAQUETTE = {plaq:.12}\n\
         BOUNDARY_1 = PERIODIC\nBOUNDARY_2 = PERIODIC\nBOUNDARY_3 = PERIODIC\n\
         BOUNDARY_4 = PERIODIC\nENSEMBLE_ID = ruttice\nSEQUENCE_NUMBER = 0\n\
         FLOATING_POINT = IEEE64BIG\nEND_HEADER\n"
    );
    write_on_root(
        u,
        || {
            let mut out = BufWriter::new(File::create(path)?);
            out.write_all(header.as_bytes())?;
            Ok(out)
        },
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
    use crate::gauge::random_gauge;

    #[test]
    fn two_rows_and_single_precision_are_read_and_the_checksum_is_checked() {
        let lat = Lattice::single_node([2, 2, 2, 4]);
        let u = random_gauge(&lat, 97);
        let dir = std::env::temp_dir();
        let whole = dir.join(format!("ruttice_nersc_{}.dat", std::process::id()));
        write_nersc(&whole, &u).unwrap();
        assert_eq!(read_nersc(&whole, &lat).unwrap().as_slice(), u.as_slice());

        // The same links as two rows in single precision, little-endian, written here.
        let mut data = Vec::new();
        for site in 0..lat.num_sites() {
            for mu in 0..ND {
                for x in &u.site(mu, site)[..12] {
                    data.extend((*x as f32).to_le_bytes());
                }
            }
        }
        let sum = data
            .as_chunks::<4>()
            .0
            .iter()
            .fold(0u32, |s, w| s.wrapping_add(u32::from_le_bytes(*w)));
        let header = |sum: u32| {
            format!(
                "BEGIN_HEADER\nDATATYPE = 4D_SU3_GAUGE\nDIMENSION_1 = 2\nDIMENSION_2 = 2\n\
                 DIMENSION_3 = 2\nDIMENSION_4 = 4\nCHECKSUM = {sum:x}\n\
                 FLOATING_POINT = IEEE32LITTLE\nEND_HEADER\n"
            )
        };
        let two = dir.join(format!("ruttice_nersc_two_{}.dat", std::process::id()));
        std::fs::write(&two, [header(sum).as_bytes(), &data].concat()).unwrap();
        let v = read_nersc(&two, &lat).unwrap();
        let worst = v
            .as_slice()
            .iter()
            .zip(u.as_slice())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-6, "{worst:e}");

        std::fs::write(&two, [header(sum ^ 1).as_bytes(), &data].concat()).unwrap();
        assert!(matches!(
            read_nersc(&two, &lat),
            Err(IoError::Checksum { .. })
        ));
        let _ = std::fs::remove_file(whole);
        let _ = std::fs::remove_file(two);
    }
}
