//! Gauge configurations in HDF5 files, through the `hdf5` crate. Built only with the
//! feature `hdf5`, which needs the HDF5 library.
//!
//! A configuration is one dataset of double-precision reals with the shape
//! `[Lt, Lz, Ly, Lx, 4, 3, 3, 2]`: the site with x running fastest, the direction, the row
//! and the column of the link, and its real and imaginary part. Stored in the row-major
//! order of HDF5 this is the order of the ILDG binary record. The layout is this program's
//! own; HDF5 files of other programs arrange a gauge field in other ways.
//!
//! Reading takes from the file only the part of the lattice the calling process holds, as
//! one hyperslab. The processes do not use parallel HDF5: each opens the file for itself.

use super::{on_one_process, IoError, LINK_REALS};
use crate::{Field, Gauge, Lattice, ND};
use ::hdf5::{File, Hyperslab, Selection, SliceOrIndex};
use ndarray::{ArrayD, IxDyn};
use std::path::Path;
use std::sync::Arc;

fn failed(e: ::hdf5::Error) -> IoError {
    IoError::Format(format!("HDF5: {e}"))
}

/// The shape of the dataset of a lattice with these extents.
fn shape(dims: [usize; ND]) -> [usize; 8] {
    [dims[3], dims[2], dims[1], dims[0], ND, 3, 3, 2]
}

/// Read the dataset `dataset` of an HDF5 file as the gauge configuration of `lat`.
pub fn read_hdf5(
    path: &Path,
    dataset: &str,
    lat: &Arc<Lattice>,
) -> Result<Field<f64, Gauge>, IoError> {
    let file = File::open(path).map_err(failed)?;
    let data = file.dataset(dataset).map_err(failed)?;
    let found = data.shape();
    let expected = shape(lat.global);
    if found != expected {
        return match found[..] {
            [lt, lz, ly, lx, ND, 3, 3, 2] => Err(IoError::Lattice {
                file: [lx, ly, lz, lt],
                expected: lat.global,
            }),
            _ => Err(IoError::Format(format!(
                "the dataset {dataset:?} has the shape {found:?}, a gauge field has {expected:?}"
            ))),
        };
    }

    // The part of this process: a range of sites in each direction, everything of a site.
    let (o, l) = (lat.origin, lat.local);
    let mut ranges: Vec<SliceOrIndex> = (0..ND)
        .rev()
        .map(|mu| SliceOrIndex::from(o[mu]..o[mu] + l[mu]))
        .collect();
    ranges.extend([ND, 3, 3, 2].map(|n| SliceOrIndex::from(0..n)));
    let part: ArrayD<f64> = data
        .read_slice(Selection::from(Hyperslab::from(ranges)))
        .map_err(failed)?;

    let mut u = Field::<f64, Gauge>::gauge(lat);
    let num_sites = lat.num_sites();
    let out = u.as_mut_slice();
    // The part comes in the order of its shape, which is the order of the local sites.
    for (k, value) in part.iter().enumerate() {
        let (site, mu, c) = (k / (ND * LINK_REALS), k / LINK_REALS % ND, k % LINK_REALS);
        out[c + LINK_REALS * (site + num_sites * mu)] = *value;
    }
    Ok(u)
}

/// Write a gauge configuration as the dataset `dataset` of a new HDF5 file.
pub fn write_hdf5(path: &Path, dataset: &str, u: &Field<f64, Gauge>) -> Result<(), IoError> {
    let lat = u.lattice();
    on_one_process(lat)?;
    let shape = shape(lat.global);
    let mut values = Vec::with_capacity(shape.iter().product());
    for site in 0..lat.num_sites() {
        for mu in 0..ND {
            values.extend_from_slice(u.site(mu, site));
        }
    }
    let values = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .expect("one value for every element of the shape");
    let file = File::create(path).map_err(failed)?;
    let data = file
        .new_dataset::<f64>()
        .shape(shape)
        .create(dataset)
        .map_err(failed)?;
    data.write(&values).map_err(failed)
}
