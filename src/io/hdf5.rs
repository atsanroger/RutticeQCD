//! Gauge configurations in HDF5 files, through the `hdf5` crate. Built only with the
//! feature `hdf5`, which needs the HDF5 library.

use super::{write_on_root, IoError, LINK_REALS};
use crate::{Field, Gauge, Lattice, ND};
use ::hdf5::{File, Hyperslab, Selection, SliceOrIndex};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
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
    let shape = shape(u.lattice().global);
    let mut one_slice = shape;
    one_slice[0] = 1;
    write_on_root(
        u,
        || {
            let file = File::create(path).map_err(failed)?;
            let data = file
                .new_dataset::<f64>()
                .shape(shape)
                .create(dataset)
                .map_err(failed)?;
            // The file stays open as long as the dataset does.
            Ok((file, data))
        },
        |(_, data), t, slice| {
            let values = ArrayViewD::from_shape(IxDyn(&one_slice), slice)
                .expect("one value for every element of a time slice");
            let mut ranges = vec![SliceOrIndex::from(t..t + 1)];
            ranges.extend(shape[1..].iter().map(|n| SliceOrIndex::from(0..*n)));
            data.write_slice(values, Selection::from(Hyperslab::from(ranges)))
                .map_err(failed)
        },
        |(file, _)| file.flush().map_err(failed),
    )
}
