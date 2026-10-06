use super::Basis;
use crate::fermion::HopTerms;
use crate::kernel::{Cplx, Real};
use crate::{
    for_sites, Boundary, Coarse, Direction, Field, Lattice, LinearFn, LinearOp, Source, Spinor,
    Stencil, Storage, ND,
};
use std::sync::Arc;

/// The eight hopping directions, in the order the coarse links are stored.
const DIRECTIONS: [(usize, Direction); 2 * ND] = [
    (0, Direction::Forward),
    (1, Direction::Forward),
    (2, Direction::Forward),
    (3, Direction::Forward),
    (0, Direction::Backward),
    (1, Direction::Backward),
    (2, Direction::Backward),
    (3, Direction::Backward),
];

/// The Galerkin coarse operator `P† D P` for a nearest-neighbour fine operator D.
///
/// It is itself a nearest-neighbour operator on the coarse lattice:
/// `(D_c ψ)(X) = A(X) ψ(X) + Σ_dir B_dir(X) ψ(X + dir)`, with dense matrices A and B_dir at
/// every coarse site. A collects the fine couplings inside a block, B_dir those that cross
/// into the neighbouring block.
pub struct CoarseOp<T: Storage> {
    lat: Arc<Lattice>,
    stencil: Stencil,
    /// Complex numbers per coarse site.
    dim: usize,
    /// `[coarse site][row][column]`, row-major (re, im).
    diag: Vec<T>,
    /// `[direction][coarse site][row][column]`.
    links: Vec<T>,
}

impl<T: Storage> CoarseOp<T> {
    /// Build `P† D P` from the parts of the fine operator, one column at a time: prolong
    /// the coarse unit vector of one component on every coarse site at once, apply the
    /// diagonal block and each hopping term, sort what comes out by whether the hop
    /// stayed inside the block, and restrict.
    pub fn build<F>(op: &F, work: &mut F::Work, basis: &Basis<T>) -> Self
    where
        F: HopTerms<Field<T, Spinor>>,
    {
        let agg = basis.aggregation();
        let (fine_lat, lat) = (agg.fine_lattice(), agg.coarse_lattice().clone());
        let dim = basis.coarse_dim();
        let n_coarse = lat.num_sites();
        let matrix = 2 * dim * dim;
        let mut diag = vec![T::default(); n_coarse * matrix];
        let mut links = vec![T::default(); 2 * ND * n_coarse * matrix];
        let hop_coefficient: T::Compute = Real::from_f64(op.hop_coefficient());
        let one = T::store(T::Compute::ONE);

        let mut unit = basis.coarse_field(1);
        let mut restricted = basis.coarse_field(1);
        let mut phi = Field::<T, Spinor>::spinor(fine_lat, 1);
        let mut inside = Field::<T, Spinor>::spinor(fine_lat, 1);
        let mut hop = Field::<T, Spinor>::spinor(fine_lat, 1);
        let mut crossing = Field::<T, Spinor>::spinor(fine_lat, 1);

        // Write column `column` of every site's matrix from a coarse field.
        let store_column = |target: &mut [T], column: usize, values: &Field<T, Coarse>| {
            for site in 0..n_coarse {
                let v = values.site(0, site);
                for row in 0..dim {
                    let at = site * matrix + 2 * (dim * row + column);
                    target[at] = v[2 * row];
                    target[at + 1] = v[2 * row + 1];
                }
            }
        };

        for column in 0..dim {
            unit.as_mut_slice().fill(T::default());
            for site in 0..n_coarse {
                unit.site_mut(0, site)[2 * column] = one;
            }
            basis.prolong(&mut phi, &unit);

            inside.as_mut_slice().fill(T::default());
            op.mult_diag(work, &mut inside, &phi);

            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                hop.as_mut_slice().fill(T::default());
                match dir {
                    Direction::Forward => op.hop_forward(work, mu, &mut hop, &phi),
                    Direction::Backward => op.hop_backward(work, mu, &mut hop, &phi),
                }
                crossing.as_mut_slice().fill(T::default());
                for x in 0..fine_lat.num_sites() {
                    let scaled = hop.site(0, x).iter().map(|h| h.load() * hop_coefficient);
                    if agg.crosses(x, mu, dir) {
                        for (c, h) in crossing.site_mut(0, x).iter_mut().zip(scaled) {
                            *c = T::store(h);
                        }
                    } else {
                        for (i, h) in inside.site_mut(0, x).iter_mut().zip(scaled) {
                            *i = T::store(i.load() + h);
                        }
                    }
                }
                basis.restrict(&mut restricted, &crossing);
                let range = d * n_coarse * matrix..(d + 1) * n_coarse * matrix;
                store_column(&mut links[range], column, &restricted);
            }

            basis.restrict(&mut restricted, &inside);
            store_column(&mut diag, column, &restricted);
        }

        Self {
            stencil: Stencil::nearest(&lat, [Boundary::Periodic; ND]),
            lat,
            dim,
            diag,
            links,
        }
    }

    pub fn lattice(&self) -> &Arc<Lattice> {
        &self.lat
    }

    /// Complex numbers per coarse site.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `out = D_c inp`, or `Γ5 D_c Γ5 inp` when `gamma5` is set, where Γ5 is +1 on the first
    /// half of the components (positive chirality) and -1 on the second.
    fn apply(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>, gamma5: bool) {
        let (dim, matrix) = (self.dim, 2 * self.dim * self.dim);
        assert!(out.num_sites() == self.lat.num_sites() && inp.num_sites() == self.lat.num_sites());
        assert_eq!(inp.internal_dof(), 2 * dim);
        assert_eq!(out.as_slice().len(), inp.as_slice().len());
        let n_coarse = self.lat.num_sites();
        let site_len = inp.site_len();
        let sign = |row: usize| -> T::Compute {
            if gamma5 && row >= dim / 2 {
                -T::Compute::ONE
            } else {
                T::Compute::ONE
            }
        };

        for_sites(out.slab_mut(0), site_len, |x, out| {
            out.fill(T::default());
            // (matrix, the coarse site it multiplies)
            let mut terms: [(&[T], usize); 2 * ND + 1] =
                [(&self.diag[x * matrix..(x + 1) * matrix], x); 2 * ND + 1];
            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                let Source::Local { site, .. } = self.stencil.source(x, mu, dir) else {
                    unreachable!("the coarse stencil is periodic and on one process");
                };
                let at = (d * n_coarse + x) * matrix;
                terms[d + 1] = (&self.links[at..at + matrix], site);
            }

            for (m, neighbour) in terms {
                let neighbour = inp.site(0, neighbour);
                for (out, v) in out
                    .chunks_exact_mut(2 * dim)
                    .zip(neighbour.chunks_exact(2 * dim))
                {
                    for row in 0..dim {
                        let mut s = Cplx::ZERO;
                        for col in 0..dim {
                            let k = 2 * (dim * row + col);
                            let z = Cplx::new(v[2 * col].load(), v[2 * col + 1].load())
                                .scale(sign(col));
                            s += Cplx::new(m[k].load(), m[k + 1].load()) * z;
                        }
                        let s = s.scale(sign(row));
                        out[2 * row] = T::store(out[2 * row].load() + s.re);
                        out[2 * row + 1] = T::store(out[2 * row + 1].load() + s.im);
                    }
                }
            }
        });
    }
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for CoarseOp<T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.apply(out, inp, false);
    }
}

/// The prolongation commutes with γ5, so `D_c† = Γ5 D_c Γ5` whenever the fine operator
/// satisfies `D† = γ5 D γ5`.
impl<T: Storage> LinearOp<Field<T, Coarse>> for CoarseOp<T> {
    fn mult_dag(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.apply(out, inp, true);
    }
}
