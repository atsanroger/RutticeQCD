use crate::comm::{opposite_face, HaloSpec, FACES};
use crate::field::storage::Storage;
use crate::field::{Aos, Cpu, Field, Kind, Sites};
use crate::lattice::even_odd::{EvenOddIndex, Parity};
use crate::lattice::for_sites::for_sites;
use crate::lattice::{Lattice, ND};

/// Boundary condition in one direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    Periodic,
    Antiperiodic,
    /// A general complex phase (re, im).
    Phase(f64, f64),
    /// Hops across the boundary contribute zero.
    Dirichlet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// Where the neighbour of a site comes from. `wrapped` is true if the hop crossed the
/// boundary of the global lattice, so the caller can apply the boundary phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Another site on this process.
    Local {
        site: usize,
        wrapped: bool,
    },
    /// A site of the neighbouring process: position `index` in what was received on the
    /// face in the direction of the hop.
    Buffer {
        index: usize,
        wrapped: bool,
    },
    Zero,
}

const WRAPPED: u8 = 1;
const ZERO: u8 = 2;
const BUFFER: u8 = 4;

/// Nearest-neighbour geometry of a lattice, built once and shared by kernels and `shift`.
///
/// So far: unit displacements. Whether this is a table or arithmetic is an internal choice
/// that callers do not see.
///
/// In a direction with more than one process, the neighbours beyond the local lattice are
/// `Source::Buffer`. The sites a process has to send are listed by `send_sites`, in the
/// order the receiving side indexes them: both sides order a face by increasing site
/// index, which is the same order because their local lattices have the same shape.
pub struct Stencil {
    num_sites: usize,
    neighbor: Vec<u32>,
    flags: Vec<u8>,
    boundary: [Boundary; ND],
    /// Rank on the other side of each face.
    neighbour_rank: [usize; FACES],
    /// Whether leaving through each face crosses the boundary of the global lattice.
    at_global_boundary: [bool; FACES],
    /// For each face, the input sites whose data the neighbour on that side needs.
    send_sites: [Vec<u32>; FACES],
}

impl Stencil {
    pub const LEGS: usize = FACES;

    #[inline(always)]
    pub fn leg(mu: usize, dir: Direction) -> usize {
        match dir {
            Direction::Forward => mu,
            Direction::Backward => ND + mu,
        }
    }

    /// All sites of the lattice; neighbours are full-lattice indices.
    pub fn nearest(lat: &Lattice, boundary: [Boundary; ND]) -> Self {
        Self::build(lat, boundary, 0..lat.num_sites(), Some)
    }

    /// The sites of one parity; neighbours are indices within the opposite parity.
    pub fn nearest_parity(
        lat: &Lattice,
        boundary: [Boundary; ND],
        index: &EvenOddIndex,
        parity: Parity,
    ) -> Self {
        let sites = index.full_sites(parity).iter().map(|s| *s as usize);
        Self::build(lat, boundary, sites, |site| {
            let (p, half) = index.half_site(site);
            (p != parity).then_some(half)
        })
    }

    /// `sites` are the full-lattice indices this stencil covers, in order. `input_index`
    /// gives the index of a full-lattice site in the field the neighbours are read from, or
    /// `None` if that field does not live on the site; it must increase with the site.
    fn build(
        lat: &Lattice,
        boundary: [Boundary; ND],
        sites: impl ExactSizeIterator<Item = usize>,
        input_index: impl Fn(usize) -> Option<usize>,
    ) -> Self {
        assert!(lat.num_sites() <= u32::MAX as usize);
        let legs = (0..ND).flat_map(|mu| [(mu, Direction::Forward), (mu, Direction::Backward)]);
        let exchanged = |mu: usize| lat.grid[mu] > 1;
        let at_global_boundary: [bool; FACES] = {
            let mut at = [false; FACES];
            for (mu, dir) in legs.clone() {
                at[Self::leg(mu, dir)] = lat.at_global_boundary(mu, dir);
            }
            at
        };

        let num_sites = sites.len();
        let mut neighbor = vec![0u32; num_sites * Self::LEGS];
        let mut flags = vec![0u8; num_sites * Self::LEGS];
        let mut received = [0u32; FACES];
        for (i, site) in sites.enumerate() {
            let x = lat.coords(site);
            for (mu, dir) in legs.clone() {
                let l = lat.local[mu];
                let (xn, leaves) = match dir {
                    Direction::Forward => ((x[mu] + 1) % l, x[mu] + 1 == l),
                    Direction::Backward => ((x[mu] + l - 1) % l, x[mu] == 0),
                };
                let leg = Self::leg(mu, dir);
                let k = i * Self::LEGS + leg;
                let wrapped = leaves && at_global_boundary[leg];
                if wrapped && boundary[mu] == Boundary::Dirichlet {
                    flags[k] = ZERO;
                } else if leaves && exchanged(mu) {
                    neighbor[k] = received[leg];
                    received[leg] += 1;
                    flags[k] = BUFFER | if wrapped { WRAPPED } else { 0 };
                } else {
                    let mut y = x;
                    y[mu] = xn;
                    let n = input_index(lat.site(y))
                        .expect("the neighbour of a covered site holds no input");
                    neighbor[k] = n as u32;
                    flags[k] = if wrapped { WRAPPED } else { 0 };
                }
            }
        }

        // What leaves through a face are the input sites in the outermost layer on that side.
        let mut send_sites: [Vec<u32>; FACES] = Default::default();
        for (mu, dir) in legs.filter(|(mu, _)| exchanged(*mu)) {
            let layer = match dir {
                Direction::Forward => lat.local[mu] - 1,
                Direction::Backward => 0,
            };
            send_sites[Self::leg(mu, dir)] = (0..lat.num_sites())
                .filter(|site| lat.coords(*site)[mu] == layer)
                .filter_map(&input_index)
                .map(|n| n as u32)
                .collect();
        }

        Self {
            num_sites,
            neighbor,
            flags,
            boundary,
            neighbour_rank: {
                let mut rank = [lat.comm.rank(); FACES];
                for mu in 0..ND {
                    rank[mu] = lat.neighbour_rank(mu, Direction::Forward);
                    rank[ND + mu] = lat.neighbour_rank(mu, Direction::Backward);
                }
                rank
            },
            at_global_boundary,
            send_sites,
        }
    }

    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    pub fn boundary(&self, mu: usize) -> Boundary {
        self.boundary[mu]
    }

    #[inline(always)]
    pub fn source(&self, site: usize, mu: usize, dir: Direction) -> Source {
        let k = site * Self::LEGS + Self::leg(mu, dir);
        let (flags, n) = (self.flags[k], self.neighbor[k] as usize);
        let wrapped = flags & WRAPPED != 0;
        if flags & ZERO != 0 {
            Source::Zero
        } else if flags & BUFFER != 0 {
            Source::Buffer { index: n, wrapped }
        } else {
            Source::Local { site: n, wrapped }
        }
    }

    /// Whether any neighbour comes from another process.
    pub fn has_buffers(&self) -> bool {
        self.send_sites.iter().any(|s| !s.is_empty())
    }

    /// The input sites to send through the face (mu, dir), in the order the neighbour
    /// indexes them. Empty if that direction has a single process.
    pub fn send_sites(&self, mu: usize, dir: Direction) -> &[u32] {
        &self.send_sites[Self::leg(mu, dir)]
    }

    /// Whether what is sent through the face (mu, dir) crosses the boundary of the global
    /// lattice.
    pub fn sends_across_boundary(&self, mu: usize, dir: Direction) -> bool {
        self.at_global_boundary[Self::leg(mu, dir)]
    }

    /// The exchange that carries `reals_per_site` reals for every site of every face. A
    /// face receives as many sites as the opposite face sends.
    pub fn halo_spec(&self, reals_per_site: usize) -> HaloSpec {
        HaloSpec {
            neighbour: self.neighbour_rank,
            len: core::array::from_fn(|face| {
                debug_assert_eq!(
                    self.send_sites[face].len(),
                    self.send_sites[opposite_face(face)].len()
                );
                reals_per_site * self.send_sites[face].len()
            }),
        }
    }
}

/// `out(x) = inp(x ± mu)` for every outer slab. Sites whose neighbour is `Zero` are set to
/// the default value. No boundary phase is applied.
///
/// Meant to be simple and general, not fast: with several processes it sets up its own
/// exchange on every call.
pub fn shift<T, K, S>(
    out: &mut Field<T, K, S, Aos, Cpu>,
    inp: &Field<T, K, S, Aos, Cpu>,
    st: &Stencil,
    mu: usize,
    dir: Direction,
) where
    T: Storage,
    K: Kind,
    S: Sites,
{
    assert_eq!(out.as_slice().len(), inp.as_slice().len());
    assert_eq!(out.site_len(), inp.site_len());
    assert_eq!(st.num_sites(), inp.num_sites());
    let site_len = inp.site_len();
    // What this process receives from direction `dir` is what the neighbour there sends
    // through its opposite face; only that pair of faces carries data.
    let (face, back) = (Stencil::leg(mu, dir), opposite_face(Stencil::leg(mu, dir)));
    let back_dir = match dir {
        Direction::Forward => Direction::Backward,
        Direction::Backward => Direction::Forward,
    };
    let mut spec = st.halo_spec(site_len);
    for f in (0..FACES).filter(|f| *f != face && *f != back) {
        spec.len[f] = 0;
    }
    let mut exchange = T::halo(inp.lattice().comm.as_ref(), &spec);
    for o in 0..inp.outer() {
        let send = exchange.send_buffer(back);
        for (v, site) in send
            .chunks_exact_mut(site_len)
            .zip(st.send_sites(mu, back_dir))
        {
            v.copy_from_slice(inp.site(o, *site as usize));
        }
        exchange.start();
        exchange.wait();
        let received = exchange.recv_buffer(face);
        for_sites(out.slab_mut(o), site_len, |site, v| {
            match st.source(site, mu, dir) {
                Source::Local { site: n, .. } => v.copy_from_slice(inp.site(o, n)),
                Source::Zero => v.fill(T::default()),
                Source::Buffer { index, .. } => {
                    v.copy_from_slice(&received[index * site_len..(index + 1) * site_len])
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Gauge;

    #[test]
    fn forward_then_backward_returns() {
        let lat = Lattice::single_node([4, 2, 6, 8]);
        let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
        for s in 0..lat.num_sites() {
            for mu in 0..ND {
                let Source::Local { site: f, .. } = st.source(s, mu, Direction::Forward) else {
                    panic!()
                };
                let Source::Local { site: b, .. } = st.source(f, mu, Direction::Backward) else {
                    panic!()
                };
                assert_eq!(b, s);
            }
        }
    }

    #[test]
    fn wrap_flag_and_dirichlet() {
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let mut bc = [Boundary::Periodic; ND];
        bc[3] = Boundary::Dirichlet;
        let st = Stencil::nearest(&lat, bc);
        let last_x = lat.site([3, 1, 2, 5]);
        assert_eq!(
            st.source(last_x, 0, Direction::Forward),
            Source::Local {
                site: lat.site([0, 1, 2, 5]),
                wrapped: true
            }
        );
        assert_eq!(
            st.source(last_x, 1, Direction::Forward),
            Source::Local {
                site: lat.site([3, 2, 2, 5]),
                wrapped: false
            }
        );
        assert_eq!(
            st.source(lat.site([0, 0, 0, 7]), 3, Direction::Forward),
            Source::Zero
        );
        assert_eq!(
            st.source(lat.site([0, 0, 0, 0]), 3, Direction::Backward),
            Source::Zero
        );
    }

    #[test]
    fn parity_stencil_agrees_with_the_full_one() {
        let lat = Lattice::single_node([4, 2, 6, 4]);
        let index = EvenOddIndex::new(&lat);
        let mut bc = [Boundary::Antiperiodic; ND];
        bc[1] = Boundary::Dirichlet;
        let full = Stencil::nearest(&lat, bc);
        for parity in [Parity::Even, Parity::Odd] {
            let half = Stencil::nearest_parity(&lat, bc, &index, parity);
            assert_eq!(half.num_sites(), lat.num_sites() / 2);
            for h in 0..half.num_sites() {
                let site = index.full_site(parity, h);
                for mu in 0..ND {
                    for dir in [Direction::Forward, Direction::Backward] {
                        let want = match full.source(site, mu, dir) {
                            Source::Local { site: n, wrapped } => {
                                let (p, i) = index.half_site(n);
                                assert_eq!(p, parity.flip());
                                Source::Local { site: i, wrapped }
                            }
                            other => other,
                        };
                        assert_eq!(half.source(h, mu, dir), want);
                    }
                }
            }
        }
    }

    #[test]
    fn shift_moves_sites_and_round_trips() {
        let lat = Lattice::single_node([4, 2, 6, 4]);
        let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
        let mut a = Field::<f64, Gauge>::gauge(&lat);
        for (i, x) in a.as_mut_slice().iter_mut().enumerate() {
            *x = i as f64;
        }
        let mut b = Field::<f64, Gauge>::gauge(&lat);
        let mut c = Field::<f64, Gauge>::gauge(&lat);
        for mu in 0..ND {
            shift(&mut b, &a, &st, mu, Direction::Forward);
            let x = [1, 1, 5, 3];
            let mut y = x;
            y[mu] = (x[mu] + 1) % lat.local[mu];
            assert_eq!(b.site(2, lat.site(x)), a.site(2, lat.site(y)));
            shift(&mut c, &b, &st, mu, Direction::Backward);
            assert_eq!(c.as_slice(), a.as_slice());
        }
    }
}
