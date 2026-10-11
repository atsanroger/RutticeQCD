//! Products of links along paths, site by site, on one process: a table of the neighbours of
//! every site (periodic), and a walk along a path that multiplies the links it meets. One
//! parallel pass over the sites computes any sum of paths without shifting whole fields,
//! which is what staples, smearing and the gauge forces are made of. On several processes
//! the neighbours of the sites on a face are elsewhere: `Neighbors::new` gives `None` and the
//! callers use shifts.

use super::link;
use crate::kernel::ColorMatrix;
use crate::{Boundary, Direction, Field, Gauge, Lattice, Source, Stencil, ND};
use std::sync::Arc;

/// A step of a path of links: direction μ, forward (to x + μ̂, through U_μ(x)) or backward
/// (to x − μ̂, through U_μ(x − μ̂)†).
pub type Step = (usize, bool);

/// The neighbours of every site of a lattice on one process, periodic in every direction.
pub struct Neighbors {
    fwd: Vec<[u32; ND]>,
    bwd: Vec<[u32; ND]>,
}

impl Neighbors {
    /// `None` when a neighbour is on another process.
    pub fn new(lat: &Arc<Lattice>) -> Option<Self> {
        if lat.comm.size() != 1 {
            return None;
        }
        let st = Stencil::nearest(lat, [Boundary::Periodic; ND]);
        let n = lat.num_sites();
        let table = |dir: Direction| -> Option<Vec<[u32; ND]>> {
            (0..n)
                .map(|x| {
                    let mut row = [0u32; ND];
                    for (mu, r) in row.iter_mut().enumerate() {
                        match st.source(x, mu, dir) {
                            Source::Local { site, .. } => *r = site as u32,
                            _ => return None,
                        }
                    }
                    Some(row)
                })
                .collect()
        };
        Some(Self {
            fwd: table(Direction::Forward)?,
            bwd: table(Direction::Backward)?,
        })
    }

    #[inline]
    pub fn up(&self, x: usize, mu: usize) -> usize {
        self.fwd[x][mu] as usize
    }

    #[inline]
    pub fn down(&self, x: usize, mu: usize) -> usize {
        self.bwd[x][mu] as usize
    }

    /// The link of a step from `x` and the site it leads to.
    #[inline]
    pub fn step(
        &self,
        u: &Field<f64, Gauge>,
        x: usize,
        (mu, forward): Step,
    ) -> (ColorMatrix<f64>, usize) {
        if forward {
            (link(u, mu, x), self.up(x, mu))
        } else {
            let y = self.down(x, mu);
            (link(u, mu, y).adj(), y)
        }
    }

    /// The product of the links along `path` from `x`.
    #[inline]
    pub fn walk(&self, u: &Field<f64, Gauge>, x: usize, path: &[Step]) -> ColorMatrix<f64> {
        let (mut m, mut y) = self.step(u, x, path[0]);
        for &s in &path[1..] {
            let (l, z) = self.step(u, y, s);
            m = m.mul(&l);
            y = z;
        }
        m
    }
}
