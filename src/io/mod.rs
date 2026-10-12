//! Reading and writing gauge configurations.

mod bridge;
#[cfg(feature = "hdf5")]
mod hdf5;
mod lime;

pub use bridge::{read_bridge_text, write_bridge_text};
#[cfg(feature = "hdf5")]
pub use hdf5::{read_hdf5, write_hdf5};
pub use lime::{read_lime, write_ildg, write_scidac};

use crate::{Field, Gauge, Lattice, ND};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub enum IoError {
    /// The file could not be read or written.
    Read(std::io::Error),
    Parse {
        index: usize,
        token: String,
    },
    Count {
        expected: usize,
        found: usize,
    },
    /// The file is not what the format says it should be.
    Format(String),
    /// The file is for another lattice.
    Lattice {
        file: [usize; ND],
        expected: [usize; ND],
    },
    /// The data do not have the checksum recorded in the file.
    Checksum {
        recorded: (u32, u32),
        computed: (u32, u32),
    },
    Unsupported(&'static str),
    /// Rank 0, the process that writes the file, failed; it has the error itself.
    Elsewhere,
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Read(e) => write!(f, "cannot read or write the file: {e}"),
            IoError::Parse { index, token } => {
                write!(f, "value {index} is not a number: {token:?}")
            }
            IoError::Count { expected, found } => {
                write!(f, "expected {expected} values, found {found}")
            }
            IoError::Format(what) => write!(f, "{what}"),
            IoError::Lattice { file, expected } => {
                write!(f, "the file holds a {file:?} lattice, not {expected:?}")
            }
            IoError::Checksum { recorded, computed } => write!(
                f,
                "checksum mismatch: the file records {:x} {:x}, its data give {:x} {:x}",
                recorded.0, recorded.1, computed.0, computed.1
            ),
            IoError::Unsupported(what) => write!(f, "not supported yet: {what}"),
            IoError::Elsewhere => write!(f, "rank 0 could not write the file"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Read(e)
    }
}

/// A file format for gauge configurations.
pub trait GaugeFile {
    /// Read the configuration of the lattice `lat`. Every process makes this call and gets
    /// its part.
    fn read(&self, path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError>;
    /// Write the configuration `u`. Every process makes this call with its part; rank 0
    /// writes the file, and all of them get the same outcome.
    fn write(&self, path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError>;
}

/// The precision of the reals in a binary file. Fields in memory are double precision
/// either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilePrecision {
    Single,
    #[default]
    Double,
}

impl FilePrecision {
    /// Bytes of one real.
    pub fn bytes(self) -> usize {
        match self {
            FilePrecision::Single => 4,
            FilePrecision::Double => 8,
        }
    }
}

/// Plain text, one real per line: the text format of Bridge++.
#[derive(Clone, Copy, Debug, Default)]
pub struct BridgeText;

impl GaugeFile for BridgeText {
    fn read(&self, path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
        read_bridge_text(path, lat)
    }
    fn write(&self, path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
        write_bridge_text(path, u)
    }
}

/// The ILDG format: a LIME container with the records `ildg-format` and
/// `ildg-binary-data`. Written the way QIO writes ILDG files, with the SciDAC records and
/// the checksum next to them. Reading takes the precision from the file.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ildg {
    /// The precision to write in.
    pub precision: FilePrecision,
}

impl GaugeFile for Ildg {
    fn read(&self, path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
        read_lime(path, lat)
    }
    fn write(&self, path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
        write_ildg(path, u, self.precision)
    }
}

/// A lattice-colour-matrix record of the SciDAC QIO library: a LIME container with the
/// record `scidac-binary-data`. Reading is the same as for `Ildg`: either kind of binary
/// record is accepted.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scidac {
    /// The precision to write in.
    pub precision: FilePrecision,
}

impl GaugeFile for Scidac {
    fn read(&self, path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
        read_lime(path, lat)
    }
    fn write(&self, path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
        write_scidac(path, u, self.precision)
    }
}

/// One dataset of an HDF5 file; see `hdf5` for its shape.
#[cfg(feature = "hdf5")]
#[derive(Clone, Debug)]
pub struct Hdf5 {
    /// Name of the dataset in the file.
    pub dataset: String,
}

#[cfg(feature = "hdf5")]
impl Default for Hdf5 {
    fn default() -> Self {
        Self {
            dataset: "gauge".to_string(),
        }
    }
}

#[cfg(feature = "hdf5")]
impl GaugeFile for Hdf5 {
    fn read(&self, path: &Path, lat: &Arc<Lattice>) -> Result<Field<f64, Gauge>, IoError> {
        read_hdf5(path, &self.dataset, lat)
    }
    fn write(&self, path: &Path, u: &Field<f64, Gauge>) -> Result<(), IoError> {
        write_hdf5(path, &self.dataset, u)
    }
}

/// Reals of one link: a 3x3 complex matrix.
const LINK_REALS: usize = 18;

/// Write a configuration file on rank 0 from the parts held by all processes. Every
/// process makes this call.
fn write_on_root<W>(
    u: &Field<f64, Gauge>,
    open: impl FnOnce() -> Result<W, IoError>,
    mut slice: impl FnMut(&mut W, usize, &[f64]) -> Result<(), IoError>,
    close: impl FnOnce(W) -> Result<(), IoError>,
) -> Result<(), IoError> {
    let lat = u.lattice();
    let root = lat.comm.rank() == 0;
    let [lx, ly, lz, lt] = lat.global;
    let (local, origin) = (lat.local, lat.origin);
    let site_reals = ND * LINK_REALS;
    // A line of sites along x on one process: the pieces the slices are made of.
    let line = local[0] * site_reals;

    let mut out = if root { Some(open()) } else { None };
    let mut whole = if root {
        vec![0.0; lx * ly * lz * site_reals]
    } else {
        Vec::new()
    };
    let mut mine = Vec::with_capacity(local[0] * local[1] * local[2] * site_reals);
    for t in 0..lt {
        mine.clear();
        if (origin[3]..origin[3] + local[3]).contains(&t) {
            for z in 0..local[2] {
                for y in 0..local[1] {
                    for x in 0..local[0] {
                        let site = lat.site([x, y, z, t - origin[3]]);
                        for mu in 0..ND {
                            mine.extend_from_slice(u.site(mu, site));
                        }
                    }
                }
            }
        }
        let parts = lat.comm.gather(&mine);
        if !root {
            continue;
        }
        for (rank, part) in parts.iter().enumerate() {
            let o = lat.origin_of(rank);
            for (k, piece) in part.chunks_exact(line).enumerate() {
                let (y, z) = (o[1] + k % local[1], o[2] + k / local[1]);
                let first = (o[0] + lx * (y + ly * z)) * site_reals;
                whole[first..first + line].copy_from_slice(piece);
            }
        }
        if let Some(Ok(w)) = &mut out {
            if let Err(e) = slice(w, t, &whole) {
                out = Some(Err(e));
            }
        }
    }

    let outcome = match out {
        None => Ok(()),
        Some(Ok(w)) => close(w),
        Some(Err(e)) => Err(e),
    };
    let mut failed = [f64::from(u8::from(outcome.is_err()))];
    lat.comm.sum(&mut failed);
    match outcome {
        Ok(()) if failed[0] > 0.0 => Err(IoError::Elsewhere),
        outcome => outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauge::random_gauge;
    use crate::{local_part, LocalComm};
    use std::path::PathBuf;

    const GLOBAL: [usize; ND] = [4, 6, 4, 8];
    /// Process grids that split the lattice in every direction, alone and together.
    const GRIDS: [[usize; ND]; 5] = [
        [2, 1, 1, 1],
        [1, 3, 1, 1],
        [1, 1, 2, 4],
        [2, 3, 1, 2],
        [2, 2, 2, 2],
    ];

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ruttice_io_parts_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// The bytes of a file with the date of the SciDAC record blanked out: it is the time
    /// of writing. The date always has the same length, so nothing else moves.
    fn without_date(path: &Path) -> Vec<u8> {
        let mut bytes = std::fs::read(path).unwrap();
        let find = |b: &[u8], what: &[u8]| b.windows(what.len()).position(|w| w == what);
        if let Some(open) = find(&bytes, b"<date>") {
            let close = open + find(&bytes[open..], b"</date>").unwrap();
            bytes[open..close].fill(b'-');
        }
        bytes
    }

    /// Write the same configuration on one process and on several grids: the files are the
    /// same bytes, and every one of them reads back on another grid as the one-process file
    /// does (for the binary formats this checks the SciDAC checksum as well).
    fn same_file_on_every_grid<F: GaugeFile + Sync>(name: &str, format: &F, bytewise: bool) {
        let whole_lat = Lattice::single_node(GLOBAL);
        let mut whole = random_gauge(&whole_lat, 4242);
        // A negative zero must come through the gather with its sign.
        whole.site_mut(2, 17)[5] = -0.0;
        let reference = scratch(&format!("{name}_1"));
        format.write(&reference, &whole).unwrap();
        let read_whole = format.read(&reference, &whole_lat).unwrap();

        for grid in GRIDS {
            let [a, b, c, d] = grid;
            let path = scratch(&format!("{name}_{a}{b}{c}{d}"));
            let results = LocalComm::run(GLOBAL, grid, |lat| {
                format
                    .write(&path, &local_part(&whole, &lat))
                    .map_err(|e| e.to_string())
            });
            assert!(
                results.iter().all(Result::is_ok),
                "{name} on {grid:?}: {results:?}"
            );
            if bytewise {
                assert!(
                    without_date(&path) == without_date(&reference),
                    "{name}: the file written on {grid:?} differs from the one of one process"
                );
            }
            // Read back on a grid that splits the lattice differently.
            let other = if grid == [2, 2, 2, 2] {
                [1, 3, 2, 1]
            } else {
                [2, 2, 2, 2]
            };
            LocalComm::run(GLOBAL, other, |lat| {
                let back = format.read(&path, &lat).unwrap();
                let expected = local_part(&read_whole, &lat);
                let same = back
                    .as_slice()
                    .iter()
                    .zip(expected.as_slice())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{name} written on {grid:?}, read on {other:?}");
            });
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_file(reference);
    }

    #[test]
    fn bridge_text_on_several_processes() {
        same_file_on_every_grid("bridge", &BridgeText, true);
    }

    #[test]
    fn ildg_on_several_processes() {
        for (name, precision) in [
            ("ildg_d", FilePrecision::Double),
            ("ildg_f", FilePrecision::Single),
        ] {
            same_file_on_every_grid(name, &Ildg { precision }, true);
        }
    }

    #[test]
    fn scidac_on_several_processes() {
        for (name, precision) in [
            ("scidac_d", FilePrecision::Double),
            ("scidac_f", FilePrecision::Single),
        ] {
            same_file_on_every_grid(name, &Scidac { precision }, true);
        }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn hdf5_on_several_processes() {
        same_file_on_every_grid("hdf5", &Hdf5::default(), true);
    }

    /// The checksum written on several processes is the one of the data: flipping one bit
    /// of the links makes reading fail.
    #[test]
    fn checksum_written_on_several_processes_guards_the_data() {
        let whole = random_gauge(&Lattice::single_node(GLOBAL), 77);
        let path = scratch("checksum");
        LocalComm::run(GLOBAL, [2, 3, 1, 2], |lat| {
            Scidac::default()
                .write(&path, &local_part(&whole, &lat))
                .unwrap()
        });
        let lat = Lattice::single_node(GLOBAL);
        assert!(Scidac::default().read(&path, &lat).is_ok());
        let mut bytes = std::fs::read(&path).unwrap();
        let name: &[u8] = b"scidac-binary-data";
        let header = bytes.windows(name.len()).position(|w| w == name).unwrap();
        // The data follow the 128 bytes of the name field; flip a bit well inside them.
        bytes[header + 128 + 1000] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            Scidac::default().read(&path, &lat),
            Err(IoError::Checksum { .. })
        ));
        let _ = std::fs::remove_file(path);
    }

    /// When rank 0 cannot write, every process returns an error (rank 0 its own), so that
    /// none of them goes on alone.
    #[test]
    fn every_process_hears_that_writing_failed() {
        let whole = random_gauge(&Lattice::single_node(GLOBAL), 5);
        let path = scratch("no_such_directory").join("conf");
        let results = LocalComm::run(GLOBAL, [1, 1, 2, 2], |lat| {
            BridgeText.write(&path, &local_part(&whole, &lat))
        });
        assert!(matches!(results[0], Err(IoError::Read(_))));
        for r in &results[1..] {
            assert!(matches!(r, Err(IoError::Elsewhere)));
        }
    }
}
