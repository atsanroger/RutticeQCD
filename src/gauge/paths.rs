//! Products of links along paths, site by site: the links of this process with a halo of `depth`
//! sites on every side, corners included (an extended field), and walks along paths that multiply
//! the links they meet.

use crate::kernel::ColorMatrix;
use crate::{for_sites, opposite_face, Direction, Field, Gauge, Stencil, ND};
use rayon::prelude::*;

/// A step of a path of links: direction μ, forward (to x + μ̂, through U_μ(x)) or backward
/// (to x − μ̂, through U_μ(x − μ̂)†).
pub type Step = (usize, bool);

const LINK: usize = 18;
const SITE: usize = ND * LINK;

/// The links of a gauge field (or any field of one colour matrix per site and direction) on
/// this process and a halo `depth` sites deep around it.
pub struct Extended {
    depth: usize,
    ext: [usize; ND],
    stride: [usize; ND],
    data: Vec<f64>,
    /// The extended index of every local site.
    inner: Vec<u32>,
}

impl Extended {
    pub fn new(u: &Field<f64, Gauge>, depth: usize) -> Self {
        let lat = u.lattice();
        let local = lat.local;
        assert!(
            local.iter().all(|l| *l >= depth),
            "a local extent {local:?} smaller than the halo depth {depth}"
        );
        let ext: [usize; ND] = core::array::from_fn(|mu| local[mu] + 2 * depth);
        let mut stride = [1; ND];
        for mu in 1..ND {
            stride[mu] = stride[mu - 1] * ext[mu - 1];
        }
        let index = |c: [usize; ND]| (0..ND).map(|mu| c[mu] * stride[mu]).sum::<usize>();
        let inner: Vec<u32> = (0..lat.num_sites())
            .map(|x| {
                let c = lat.coords(x);
                index(core::array::from_fn(|mu| c[mu] + depth)) as u32
            })
            .collect();
        let mut data = vec![0.0; ext.iter().product::<usize>() * SITE];
        {
            // The interior, in parallel over the sites.
            let slabs: [&[f64]; ND] = core::array::from_fn(|mu| u.slab(mu));
            let ptr = SyncPtr(data.as_mut_ptr());
            inner.par_iter().enumerate().for_each(|(x, &e)| {
                let ptr = &ptr;
                for (mu, slab) in slabs.iter().enumerate() {
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            ptr.0.add(e as usize * SITE + mu * LINK),
                            LINK,
                        )
                    };
                    dst.copy_from_slice(&slab[x * LINK..(x + 1) * LINK]);
                }
            });
        }
        let mut me = Self {
            depth,
            ext,
            stride,
            data,
            inner,
        };
        if depth > 0 {
            for mu in 0..ND {
                me.fill(u, mu);
            }
        }
        me
    }

    /// The extended indices of the sites with coordinate μ in `range` and, in the other
    /// directions, the full extent (directions before μ, already filled) or the interior.
    fn slab(&self, local: [usize; ND], mu: usize, range: std::ops::Range<usize>) -> Vec<usize> {
        let d = self.depth;
        let ranges: [std::ops::Range<usize>; ND] = core::array::from_fn(|nu| {
            if nu == mu {
                range.clone()
            } else if nu < mu {
                0..self.ext[nu]
            } else {
                d..d + local[nu]
            }
        });
        let mut out = Vec::new();
        for t in ranges[3].clone() {
            for z in ranges[2].clone() {
                for y in ranges[1].clone() {
                    for x in ranges[0].clone() {
                        out.push(
                            x * self.stride[0]
                                + y * self.stride[1]
                                + z * self.stride[2]
                                + t * self.stride[3],
                        );
                    }
                }
            }
        }
        out
    }

    /// The halo in direction μ, both sides.
    fn fill(&mut self, u: &Field<f64, Gauge>, mu: usize) {
        let lat = u.lattice();
        let (d, l) = (self.depth, lat.local[mu]);
        // What the neighbours need from this process: its first and its last d layers.
        let first = self.slab(lat.local, mu, d..2 * d);
        let last = self.slab(lat.local, mu, l..l + d);
        let low = self.slab(lat.local, mu, 0..d);
        let high = self.slab(lat.local, mu, l + d..l + 2 * d);
        let ptr = SyncPtr(self.data.as_mut_ptr());
        if lat.grid[mu] == 1 {
            // Periodic copies within this process; sources and destinations do not overlap.
            low.par_iter()
                .zip(last.par_iter())
                .chain(high.par_iter().zip(first.par_iter()))
                .for_each(|(&to, &from)| {
                    let ptr = &ptr;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            ptr.0.add(from * SITE),
                            ptr.0.add(to * SITE),
                            SITE,
                        );
                    }
                });
            return;
        }
        let (fwd, bwd) = (
            Stencil::leg(mu, Direction::Forward),
            Stencil::leg(mu, Direction::Backward),
        );
        let st = Stencil::nearest(lat, [crate::Boundary::Periodic; ND]);
        let mut spec = st.halo_spec(SITE);
        for f in 0..spec.len.len() {
            spec.len[f] = if f == fwd || f == bwd {
                first.len() * SITE
            } else {
                0
            };
        }
        debug_assert_eq!(opposite_face(fwd), bwd);
        let mut ex = lat.comm.halo_f64(&spec);
        let data = &self.data;
        let gather = |buf: &mut [f64], from: &[usize]| {
            buf.par_chunks_mut(SITE)
                .zip(from.par_iter())
                .for_each(|(b, &e)| b.copy_from_slice(&data[e * SITE..(e + 1) * SITE]));
        };
        // The first layers go to the neighbour behind (its high halo), the last ones to the
        // neighbour ahead (its low halo).
        gather(ex.send_buffer(bwd), &first);
        gather(ex.send_buffer(fwd), &last);
        ex.start();
        ex.wait();
        for (face, to) in [(fwd, &high), (bwd, &low)] {
            let buf = ex.recv_buffer(face);
            buf.par_chunks(SITE).zip(to.par_iter()).for_each(|(b, &e)| {
                let ptr = &ptr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b.as_ptr(), ptr.0.add(e * SITE), SITE);
                }
            });
        }
    }
    /// The extended index of local site `x`.
    #[inline]
    pub fn at(&self, x: usize) -> usize {
        self.inner[x] as usize
    }

    #[inline]
    pub fn up(&self, e: usize, mu: usize) -> usize {
        e + self.stride[mu]
    }

    #[inline]
    pub fn down(&self, e: usize, mu: usize) -> usize {
        e - self.stride[mu]
    }

    /// The link U_μ at extended index `e`.
    #[inline]
    pub fn link(&self, mu: usize, e: usize) -> ColorMatrix<f64> {
        let o = e * SITE + mu * LINK;
        ColorMatrix::from_reals(&self.data[o..o + LINK])
    }

    /// The link of a step from `e` and the index it leads to.
    #[inline]
    pub fn step(&self, e: usize, (mu, forward): Step) -> (ColorMatrix<f64>, usize) {
        if forward {
            (self.link(mu, e), self.up(e, mu))
        } else {
            let y = self.down(e, mu);
            (self.link(mu, y).adj(), y)
        }
    }

    /// The product of the links along `path` from extended index `e` (at most `depth` steps
    /// away from the local sites in every direction).
    #[inline]
    pub fn walk(&self, e: usize, path: &[Step]) -> ColorMatrix<f64> {
        let (mut m, mut y) = self.step(e, path[0]);
        for &s in &path[1..] {
            let (l, z) = self.step(y, s);
            m = m.mul(&l);
            y = z;
        }
        m
    }
}

struct SyncPtr(*mut f64);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

/// The product of the links along the path `steps` from every site, as one colour matrix per
/// site (in the first slab of a gauge field). Periodic boundary conditions.
pub fn path_product(u: &Field<f64, Gauge>, steps: &[Step]) -> Field<f64, Gauge> {
    // The halo must reach as far as the path goes in any direction.
    let mut depth = 0i64;
    let mut pos = [0i64; ND];
    for &(mu, f) in steps {
        pos[mu] += if f { 1 } else { -1 };
        depth = depth.max(pos[mu].abs());
    }
    let ext = Extended::new(u, depth.max(1) as usize);
    let mut p = Field::<f64, Gauge>::gauge(u.lattice());
    for_sites(p.slab_mut(0), 18, |x, out| {
        ext.walk(ext.at(x), steps).write_reals(out)
    });
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauge::{link, random_gauge};
    use crate::{assemble, local_part, Lattice, LocalComm};

    /// Every link of the halo is the link of the neighbouring site, on one process and on
    /// several, corners included.
    #[test]
    fn the_halo_holds_the_neighbours() {
        let global = [4, 6, 4, 4];
        let whole = random_gauge(&Lattice::single_node(global), 3);
        let check = |u: &Field<f64, Gauge>, whole: &Field<f64, Gauge>| {
            let lat = u.lattice();
            let ext = Extended::new(u, 2);
            let wl = whole.lattice();
            for x in 0..lat.num_sites() {
                let g = lat.global_coords(x);
                for off in [
                    [1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [-1, -2, 0, 0],
                    [2, -1, 1, -2],
                    [0, 0, -1, 1],
                ] {
                    let mut e = ext.at(x);
                    let mut c = g;
                    for mu in 0..ND {
                        let o: i64 = off[mu];
                        for _ in 0..o.unsigned_abs() {
                            e = if o > 0 {
                                ext.up(e, mu)
                            } else {
                                ext.down(e, mu)
                            };
                        }
                        c[mu] = ((g[mu] as i64 + o).rem_euclid(global[mu] as i64)) as usize;
                    }
                    let site = wl.site(c);
                    for mu in 0..ND {
                        assert_eq!(ext.link(mu, e), link(whole, mu, site), "{g:?} + {off:?}");
                    }
                }
            }
        };
        check(&whole, &whole);
        for grid in [[2, 1, 1, 1], [1, 3, 2, 1], [2, 2, 2, 2]] {
            let parts = LocalComm::run(global, grid, |lat| {
                let u = local_part(&whole, &lat);
                check(&u, &whole);
                u
            });
            let back = assemble(whole.lattice(), &parts);
            assert_eq!(back.as_slice(), whole.as_slice());
        }
    }
}
