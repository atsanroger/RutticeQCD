//! Moving fields between a single-process lattice and the parts of the same global lattice
//! held by several processes: for checks that compare a split run with one process.

use crate::{Field, Kind, Lattice, Storage};
use std::sync::Arc;

/// The part of a field on a single-process lattice that belongs to the process owning
/// `lat`, a part of the same global lattice.
pub fn local_part<T: Storage, K: Kind>(whole: &Field<T, K>, lat: &Arc<Lattice>) -> Field<T, K> {
    let whole_lat = whole.lattice();
    assert_eq!(whole_lat.local, lat.global);
    let mut part = Field::with_shape(
        lat,
        whole.internal_dof(),
        whole.ls(),
        whole.num_vectors(),
        whole.outer(),
    );
    for o in 0..whole.outer() {
        for site in 0..lat.num_sites() {
            let from = whole_lat.site(lat.global_coords(site));
            part.site_mut(o, site).copy_from_slice(whole.site(o, from));
        }
    }
    part
}

/// Put the parts held by all processes together into one field on a single-process lattice.
pub fn assemble<T: Storage, K: Kind>(
    whole_lat: &Arc<Lattice>,
    parts: &[Field<T, K>],
) -> Field<T, K> {
    let first = &parts[0];
    let mut whole = Field::with_shape(
        whole_lat,
        first.internal_dof(),
        first.ls(),
        first.num_vectors(),
        first.outer(),
    );
    let mut covered = 0;
    for part in parts {
        let lat = part.lattice();
        assert_eq!(lat.global, whole_lat.local);
        for o in 0..part.outer() {
            for site in 0..lat.num_sites() {
                let to = whole_lat.site(lat.global_coords(site));
                whole.site_mut(o, to).copy_from_slice(part.site(o, site));
            }
        }
        covered += lat.num_sites();
    }
    assert_eq!(
        covered,
        whole_lat.num_sites(),
        "the parts do not cover the lattice"
    );
    whole
}
