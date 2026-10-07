//! Reading and writing gauge configurations.
//!
//! A file format is a type that implements `GaugeFile`:
//!
//! - `BridgeText`: plain text, one real per line (`bridge`)
//! - `Ildg` and `Scidac`: binary data in a LIME container, as the ILDG specifies it and as
//!   the SciDAC QIO library writes it (`lime`)
//! - `Hdf5`: one dataset in an HDF5 file, read by hyperslabs; only with the feature `hdf5`
//!
//! All of them keep the links in the same order, the one of ILDG: the site outermost with x
//! running fastest, then the direction, then the 3x3 matrix row by row, real part first.
//!
//! Reading works on several processes; each keeps the part of the lattice it holds.
//! Writing is for one process so far.

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
    /// Write a configuration held by one process.
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

/// Writing needs the whole lattice on the calling process.
fn on_one_process(lat: &Lattice) -> Result<(), IoError> {
    if lat.local == lat.global {
        Ok(())
    } else {
        Err(IoError::Unsupported(
            "writing a configuration on more than one process",
        ))
    }
}
