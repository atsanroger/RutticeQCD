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

/// Where the neighbour of a site comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Another site on this process. `wrapped` is true if the hop crossed the global
    /// boundary, so the caller can apply the boundary phase.
    Local {
        site: usize,
        wrapped: bool,
    },
    /// A position in a communication buffer. Not produced until MPI is implemented.
    Buffer {
        index: usize,
    },
    Zero,
}

const WRAPPED: u8 = 1;
const ZERO: u8 = 2;

/// Nearest-neighbour geometry of a lattice, built once and shared by kernels and `shift`.
///
/// Stage 0: unit displacements, all sites, single node. Whether this is a table or
/// arithmetic is an internal choice that callers do not see.
pub struct Stencil {
    num_sites: usize,
    neighbor: Vec<u32>,
    flags: Vec<u8>,
    boundary: [Boundary; ND],
}

impl Stencil {
    pub const LEGS: usize = 2 * ND;

    #[inline(always)]
    pub fn leg(mu: usize, dir: Direction) -> usize {
        match dir {
            Direction::Forward => mu,
            Direction::Backward => ND + mu,
        }
    }

    pub fn nearest(lat: &Lattice, boundary: [Boundary; ND]) -> Self {
        assert_eq!(
            lat.grid, [1; ND],
            "stencils across processes are not implemented yet"
        );
        let num_sites = lat.num_sites();
        assert!(num_sites <= u32::MAX as usize);
        let mut neighbor = vec![0u32; num_sites * Self::LEGS];
        let mut flags = vec![0u8; num_sites * Self::LEGS];
        for site in 0..num_sites {
            let x = lat.coords(site);
            for mu in 0..ND {
                let l = lat.local[mu];
                for dir in [Direction::Forward, Direction::Backward] {
                    let (xn, wrapped) = match dir {
                        Direction::Forward => ((x[mu] + 1) % l, x[mu] + 1 == l),
                        Direction::Backward => ((x[mu] + l - 1) % l, x[mu] == 0),
                    };
                    let mut y = x;
                    y[mu] = xn;
                    let k = site * Self::LEGS + Self::leg(mu, dir);
                    neighbor[k] = lat.site(y) as u32;
                    if wrapped {
                        flags[k] = match boundary[mu] {
                            Boundary::Dirichlet => ZERO,
                            _ => WRAPPED,
                        };
                    }
                }
            }
        }
        Self {
            num_sites,
            neighbor,
            flags,
            boundary,
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
        match self.flags[k] {
            ZERO => Source::Zero,
            f => Source::Local {
                site: self.neighbor[k] as usize,
                wrapped: f == WRAPPED,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
