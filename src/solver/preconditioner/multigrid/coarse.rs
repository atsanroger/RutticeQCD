use super::{Basis, MultigridError};
use crate::fermion::HopTerms;
use crate::kernel::{Cplx, Number, Real};
use crate::numerics::invert_complex;
use crate::{
    for_sites, Boundary, Coarse, Direction, Field, Lattice, LinearFn, LinearOp, Source, Spinor,
    Stencil, Storage, Vector, C64, ND,
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
    /// The inverse of `diag` when the operator is preconditioned by it: the operator is
    /// then A^{-1} D_c (see `build_with`).
    diag_inv: Option<Vec<T>>,
    scratch: Option<Field<T, Coarse>>,
    /// Whether `D_c† = Γ5 D_c Γ5`: true for an operator built from the parts of a fine
    /// operator with that property, not for a probed one.
    adjoint_from_gamma5: bool,
}

/// The inverses of the diagonal blocks of a coarse operator.
fn invert_blocks<T: Storage>(
    diag: &[T],
    dim: usize,
    n_coarse: usize,
) -> Result<Vec<T>, MultigridError> {
    let matrix = 2 * dim * dim;
    let mut inv = vec![T::default(); n_coarse * matrix];
    let mut a = vec![0.0; matrix];
    let mut a_inv = vec![0.0; matrix];
    for site in 0..n_coarse {
        let block = site * matrix..(site + 1) * matrix;
        for (x, d) in a.iter_mut().zip(&diag[block.clone()]) {
            *x = d.load().to_f64();
        }
        invert_complex(dim, &mut a, &mut a_inv)
            .map_err(|_| MultigridError::SingularCoarseDiagonal { coarse_site: site })?;
        for (o, x) in inv[block].iter_mut().zip(&a_inv) {
            *o = T::store(Number::from_f64(*x));
        }
    }
    Ok(inv)
}

impl<T: Storage> CoarseOp<T> {
    /// Build `P† D P` from the parts of the fine operator, one column at a time: prolong
    /// the coarse unit vector of one component on every coarse site at once, apply the
    /// diagonal block and each hopping term, sort what comes out by whether the hop
    /// stayed inside the block, and restrict.
    pub fn build<F, S: Storage>(op: &F, work: &mut F::Work, basis: &Basis<T, S>) -> Self
    where
        F: HopTerms<Field<T, Spinor>>,
    {
        Self::build_with(op, work, basis, None, false)
            .expect("nothing is inverted without diagonal preconditioning")
    }

    /// `build` with the two options of the domain-wall multigrid (DESIGN.md §7.6.2, after
    /// `AFopr_Domainwall_coarse::generate_coarse_op` of the CUDA version):
    ///
    /// - `right`: a factor R applied before D, giving `P† D R P`. It must not couple
    ///   different 4D sites (domain-wall: the exact C_PV^{-1}), so that the coarse operator
    ///   still reaches only the 8 neighbours.
    /// - `precondition_diagonal`: the operator becomes A^{-1} D_c, where A is the diagonal
    ///   block of D_c at every coarse site. A coarse solve then needs its right-hand side
    ///   multiplied by A^{-1} first (`apply_diagonal_inverse`).
    pub fn build_with<F, S: Storage>(
        op: &F,
        work: &mut F::Work,
        basis: &Basis<T, S>,
        mut right: Option<&mut dyn LinearFn<Field<T, Spinor>>>,
        precondition_diagonal: bool,
    ) -> Result<Self, MultigridError>
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
        let hop_coefficient: T::Compute = Number::from_f64(op.hop_coefficient());
        let one = T::store(T::Compute::ONE);

        let mut unit = basis.coarse_field(1);
        let mut restricted = basis.coarse_field(1);
        let mut phi = basis.fine_field(1);
        let mut phi_right = basis.fine_field(1);
        let mut inside = basis.fine_field(1);
        let mut hop = basis.fine_field(1);
        let mut crossing = basis.fine_field(1);

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
            let phi = match right.as_mut() {
                Some(r) => {
                    r.mult(&mut phi_right, &phi);
                    &phi_right
                }
                None => &phi,
            };

            inside.as_mut_slice().fill(T::default());
            op.mult_diag(work, &mut inside, phi);

            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                hop.as_mut_slice().fill(T::default());
                match dir {
                    Direction::Forward => op.hop_forward(work, mu, &mut hop, phi),
                    Direction::Backward => op.hop_backward(work, mu, &mut hop, phi),
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

        let diag_inv = if precondition_diagonal {
            Some(invert_blocks(&diag, dim, n_coarse)?)
        } else {
            None
        };

        Ok(Self {
            stencil: Stencil::nearest(&lat, [Boundary::Periodic; ND]),
            lat,
            dim,
            diag,
            links,
            diag_inv,
            scratch: None,
            adjoint_from_gamma5: true,
        })
    }

    /// The coarse operator `R A P` of any map A on the fine lattice, cut down to the
    /// couplings of a coarse site with itself and its eight neighbours. P is the
    /// prolongation of `basis` and R the restriction of `restriction`; with the same basis
    /// for both this is the Galerkin operator `P† A P`.
    ///
    /// A is used as a black box: the coarse unit vectors of one component are prolonged,
    /// A is applied, and the result is restricted. Sites that are far enough apart can be
    /// probed together: with A reaching no further than the neighbouring aggregates, also
    /// diagonally, sources three coarse sites apart in every direction do not disturb the
    /// neighbours of one another. That is at most 81 applications of A per component (more
    /// on a coarse lattice with an extent of four, where every position is probed by
    /// itself). Couplings to diagonal neighbours are left out; for an A that is a product
    /// of two nearest-neighbour operators they exist, and leaving them out is an
    /// approximation.
    ///
    /// The adjoint of such an operator is not available (`mult_dag` panics): it does not
    /// follow from Γ5 as it does for `build`.
    pub fn probe<S: Storage, R: Storage>(
        a: &mut dyn LinearFn<Field<T, Spinor>>,
        basis: &Basis<T, S>,
        restriction: &Basis<T, R>,
        precondition_diagonal: bool,
    ) -> Result<Self, MultigridError> {
        assert_eq!(basis.coarse_dim(), restriction.coarse_dim());
        let agg = basis.aggregation();
        let lat = agg.coarse_lattice().clone();
        let stencil = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
        let dim = basis.coarse_dim();
        let n_coarse = lat.num_sites();
        let matrix = 2 * dim * dim;
        let mut diag = vec![T::default(); n_coarse * matrix];
        let mut links = vec![T::default(); 2 * ND * n_coarse * matrix];
        let one = T::store(T::Compute::ONE);
        // Sources of one colour are `colours[mu]` apart in direction mu: three, or a
        // divisor of the extent above that, or the whole extent when it is small.
        let colours = lat.local.map(|l| {
            if l <= 4 {
                l
            } else {
                (3..=l).find(|k| l.is_multiple_of(*k)).unwrap_or(l)
            }
        });

        let mut unit = basis.coarse_field(1);
        let mut restricted = basis.coarse_field(1);
        let mut phi = basis.fine_field(1);
        let mut a_phi = basis.fine_field(1);
        let column_of = |target: &mut [T], site: usize, column: usize, values: &[T]| {
            for row in 0..dim {
                let at = site * matrix + 2 * (dim * row + column);
                target[at] = values[2 * row];
                target[at + 1] = values[2 * row + 1];
            }
        };

        for colour in 0..colours.iter().product::<usize>() {
            let mut rest = colour;
            let want = colours.map(|k| {
                let c = rest % k;
                rest /= k;
                c
            });
            let sources: Vec<usize> = (0..n_coarse)
                .filter(|site| {
                    let x = lat.coords(*site);
                    (0..ND).all(|mu| x[mu] % colours[mu] == want[mu])
                })
                .collect();
            for column in 0..dim {
                unit.as_mut_slice().fill(T::default());
                for site in &sources {
                    unit.site_mut(0, *site)[2 * column] = one;
                }
                basis.prolong(&mut phi, &unit);
                a.mult(&mut a_phi, &phi);
                restriction.restrict(&mut restricted, &a_phi);
                for source in &sources {
                    column_of(&mut diag, *source, column, restricted.site(0, *source));
                    for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                        // The link in direction `dir` at X multiplies the field at
                        // X + dir, so the source is X + dir. With an extent of one the
                        // neighbour is the site itself, with two both neighbours are the
                        // same site: then everything is in the diagonal block, or in the
                        // forward link.
                        let extent = lat.local[mu];
                        if extent == 1 || (extent == 2 && dir == Direction::Backward) {
                            continue;
                        }
                        let back = match dir {
                            Direction::Forward => Direction::Backward,
                            Direction::Backward => Direction::Forward,
                        };
                        let Source::Local { site: x, .. } = stencil.source(*source, mu, back)
                        else {
                            unreachable!("the coarse stencil is periodic and on one process");
                        };
                        let range = d * n_coarse * matrix..(d + 1) * n_coarse * matrix;
                        column_of(&mut links[range], x, column, restricted.site(0, x));
                    }
                }
            }
        }

        let diag_inv = if precondition_diagonal {
            Some(invert_blocks(&diag, dim, n_coarse)?)
        } else {
            None
        };
        Ok(Self {
            stencil,
            lat,
            dim,
            diag,
            links,
            diag_inv,
            scratch: None,
            adjoint_from_gamma5: false,
        })
    }

    /// Whether the operator is A^{-1} D_c rather than D_c.
    pub fn is_diagonal_preconditioned(&self) -> bool {
        self.diag_inv.is_some()
    }

    /// `out = A^{-1} inp` with A the diagonal block at every coarse site. Only for an
    /// operator built with `precondition_diagonal`.
    pub fn apply_diagonal_inverse(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        let inv = self
            .diag_inv
            .as_ref()
            .expect("the coarse operator was built without diagonal preconditioning");
        let (dim, matrix) = (self.dim, 2 * self.dim * self.dim);
        let site_len = inp.site_len();
        for_sites(out.slab_mut(0), site_len, |x, out| {
            let m = &inv[x * matrix..(x + 1) * matrix];
            for (out, v) in out
                .chunks_exact_mut(2 * dim)
                .zip(inp.site(0, x).chunks_exact(2 * dim))
            {
                for row in 0..dim {
                    let mut s = Cplx::ZERO;
                    for col in 0..dim {
                        let k = 2 * (dim * row + col);
                        s += Cplx::new(m[k].load(), m[k + 1].load())
                            * Cplx::new(v[2 * col].load(), v[2 * col + 1].load());
                    }
                    out[2 * row] = T::store(s.re);
                    out[2 * row + 1] = T::store(s.im);
                }
            }
        });
    }

    /// Γ5 on the coarse lattice: +1 on the first half of the components, -1 on the second.
    fn gamma5(&self, v: &mut Field<T, Coarse>) {
        let dim = self.dim;
        let site_len = v.site_len();
        for_sites(v.slab_mut(0), site_len, |_, s| {
            for chunk in s.chunks_exact_mut(2 * dim) {
                for r in &mut chunk[dim..] {
                    *r = T::store(-r.load());
                }
            }
        });
    }

    fn scratch_like(&mut self, like: &Field<T, Coarse>) -> Field<T, Coarse> {
        match self.scratch.take() {
            Some(s) if s.same_shape(like) => s,
            _ => like.zeros_like(),
        }
    }

    pub fn lattice(&self) -> &Arc<Lattice> {
        &self.lat
    }

    /// The same operator stored as `U`, for example single precision: the coarse operator
    /// is limited by memory bandwidth, so half the bytes is nearly twice the speed.
    pub fn to_storage<U: Storage>(&self) -> CoarseOp<U> {
        let convert = |v: &[T]| -> Vec<U> {
            v.iter()
                .map(|x| U::store(Number::from_f64(x.load().to_f64())))
                .collect()
        };
        CoarseOp {
            stencil: Stencil::nearest(&self.lat, [Boundary::Periodic; ND]),
            lat: self.lat.clone(),
            dim: self.dim,
            diag: convert(&self.diag),
            links: convert(&self.links),
            diag_inv: self.diag_inv.as_deref().map(convert),
            scratch: None,
            adjoint_from_gamma5: self.adjoint_from_gamma5,
        }
    }

    /// `out = D_c† inp` exactly, the conjugate transpose of the stored matrix:
    /// `(D_c† ψ)(X) = A(X)† ψ(X) + Σ_dir B_-dir(X + dir)† ψ(X + dir)`. For a Galerkin operator
    /// this is P† D† P, whether or not D† = γ5 D γ5 (`mult_dag` assumes it: for Möbius
    /// domain-wall fermions with c ≠ 0 it holds only approximately). Not for an operator
    /// preconditioned by its diagonal blocks.
    pub fn mult_adjoint(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        assert!(
            self.diag_inv.is_none(),
            "the exact adjoint of a diagonally preconditioned coarse operator"
        );
        let (dim, matrix) = (self.dim, 2 * self.dim * self.dim);
        let n_coarse = self.lat.num_sites();
        let site_len = inp.site_len();
        for_sites(out.slab_mut(0), site_len, |x, out| {
            out.fill(T::default());
            // (the block applied as its conjugate transpose, the site it multiplies)
            let mut terms: [(&[T], usize); 2 * ND + 1] =
                [(&self.diag[x * matrix..(x + 1) * matrix], x); 2 * ND + 1];
            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                let Source::Local { site, .. } = self.stencil.source(x, mu, dir) else {
                    unreachable!("the coarse stencil is periodic and on one process");
                };
                let back = (d + ND) % (2 * ND);
                let at = (back * n_coarse + site) * matrix;
                terms[d + 1] = (&self.links[at..at + matrix], site);
            }
            for (m, neighbour) in terms {
                let v = inp.site(0, neighbour);
                for (out, v) in out.chunks_exact_mut(2 * dim).zip(v.chunks_exact(2 * dim)) {
                    for row in 0..dim {
                        // (M†)_{row, col} = conj(M_{col, row})
                        let mut s = Cplx::<T::Compute>::ZERO;
                        for col in 0..dim {
                            let k = 2 * (dim * col + row);
                            s += Cplx::new(m[k].load(), -m[k + 1].load())
                                * Cplx::new(v[2 * col].load(), v[2 * col + 1].load());
                        }
                        out[2 * row] = T::store(out[2 * row].load() + s.re);
                        out[2 * row + 1] = T::store(out[2 * row + 1].load() + s.im);
                    }
                }
            }
        });
    }

    /// The diagonal block of D_c at coarse site `x` (of the operator as built, before any
    /// diagonal preconditioning), row-major `dim × dim`.
    pub fn diagonal_block(&self, x: usize) -> Vec<C64> {
        let matrix = 2 * self.dim * self.dim;
        self.diag[x * matrix..(x + 1) * matrix]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| C64::new(z[0].load().to_f64(), z[1].load().to_f64()))
            .collect()
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
        let half = dim / 2;

        for_sites(out.slab_mut(0), site_len, |x, out| {
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

            // The sums of the nine terms for every vector, in the compute type, real and
            // imaginary parts apart; the inputs are loaded once per term with Γ5 already
            // applied, so the inner loop is a plain multiply-add the compiler can vectorise.
            // Each row of a block is read once and used for all the vectors: the operator is
            // limited by memory bandwidth (8 flops per stored element and vector), so several
            // right-hand sides at once raise the arithmetic intensity in proportion.
            let nv = out.len() / (2 * dim);
            let zero = T::Compute::ZERO;
            let (mut acc_re, mut acc_im) = (vec![zero; nv * dim], vec![zero; nv * dim]);
            let (mut in_re, mut in_im) = (vec![zero; nv * dim], vec![zero; nv * dim]);
            for (m, neighbour) in terms {
                let v = inp.site(0, neighbour);
                for j in 0..nv * dim {
                    let (re, im) = (v[2 * j].load(), v[2 * j + 1].load());
                    let negate = gamma5 && j % dim >= half;
                    in_re[j] = if negate { -re } else { re };
                    in_im[j] = if negate { -im } else { im };
                }
                for row in 0..dim {
                    let m_row = &m[2 * dim * row..2 * dim * (row + 1)];
                    for k in 0..nv {
                        let (x_re, x_im) = (
                            &in_re[k * dim..(k + 1) * dim],
                            &in_im[k * dim..(k + 1) * dim],
                        );
                        let (mut s_re, mut s_im) = (zero, zero);
                        for col in 0..dim {
                            let (a, b) = (m_row[2 * col].load(), m_row[2 * col + 1].load());
                            s_re = s_re + a * x_re[col] - b * x_im[col];
                            s_im = s_im + a * x_im[col] + b * x_re[col];
                        }
                        acc_re[k * dim + row] += s_re;
                        acc_im[k * dim + row] += s_im;
                    }
                }
            }
            for j in 0..nv * dim {
                let negate = gamma5 && j % dim >= half;
                let (re, im) = (acc_re[j], acc_im[j]);
                out[2 * j] = T::store(if negate { -re } else { re });
                out[2 * j + 1] = T::store(if negate { -im } else { im });
            }
        });
    }
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for CoarseOp<T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        if self.diag_inv.is_none() {
            self.apply(out, inp, false);
            return;
        }
        let mut t = self.scratch_like(inp);
        self.apply(&mut t, inp, false);
        self.apply_diagonal_inverse(out, &t);
        self.scratch = Some(t);
    }
}

/// The prolongation commutes with γ5, so `D_c† = Γ5 D_c Γ5` whenever the fine operator
/// satisfies `D† = γ5 D γ5`.
impl<T: Storage> LinearOp<Field<T, Coarse>> for CoarseOp<T> {
    fn mult_dag(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        assert!(
            self.adjoint_from_gamma5,
            "the adjoint of a probed coarse operator is not implemented"
        );
        if self.diag_inv.is_none() {
            self.apply(out, inp, true);
            return;
        }
        // The diagonal block inherits γ5-hermiticity, A^† = Γ5 A Γ5, so
        // (A^{-1} D_c)^† = Γ5 D_c Γ5 (Γ5 A^{-1} Γ5).
        let mut t = self.scratch_like(inp);
        t.copy_from(inp);
        self.gamma5(&mut t);
        self.apply_diagonal_inverse(out, &t);
        self.gamma5(out);
        t.copy_from(out);
        self.apply(out, &t, true);
        self.scratch = Some(t);
    }
}
