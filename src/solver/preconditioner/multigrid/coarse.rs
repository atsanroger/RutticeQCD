use super::{Basis, MultigridError};
use crate::fermion::{HopTerms, Variant};
use crate::kernel::{Cplx, Number, Real};
use crate::numerics::invert_complex;
use crate::{
    fit, for_sites, Boundary, Coarse, Direction, Field, HaloExchange, Lattice, LinearFn, LinearOp,
    Source, Spinor, Stencil, Storage, Vector, C64, FACES, ND,
};
use std::sync::{Arc, Mutex, OnceLock};

/// The eight hopping directions, in the order the coarse links are stored (the order of the
/// faces and legs of `Stencil`: forward 0..4, backward 4..8).
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
/// The hops join coarse sites of opposite parity (when every extent of the coarse lattice is even),
/// so with D_c = [[A_ee, B_eo], [B_oe, A_oo]] a solve of D_c c = g is one of the Schur complement
/// on the even sites,
///
///   S c_e = g_e − B_eo A_oo⁻¹ g_o,    S = A_ee − B_eo A_oo⁻¹ B_oe,
///
/// followed by c_o = A_oo⁻¹ (g_o − B_oe c_e) (`schur`, `schur_source`, `schur_solution`, after
/// `prepare_even_odd`).
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
    /// The buffers of the halo exchange, for fields of this many reals per site.
    exchange: Mutex<Option<Exchange<T>>>,
    /// The links B_dir(Y) of the neighbours Y on other processes, by the face they arrive
    /// on (the hop from here to Y), for `mult_adjoint`.
    ghost_links: OnceLock<[Vec<T>; FACES]>,
    /// What the even-odd form needs (`prepare_even_odd`).
    even_odd: Option<EvenOdd<T>>,
}

/// What the even-odd form of a coarse operator needs besides the operator.
struct EvenOdd<T> {
    /// Whether each coarse site is odd, by its coordinates on the global coarse lattice.
    odd: Vec<bool>,
    /// The inverses of the diagonal blocks, `[coarse site][row][column]`.
    inverse: Vec<T>,
}

/// `acc += M v` for every vector: `m` a dim × dim block (row-major, re and im), the vectors'
/// components in `(in_re, in_im)`. A plain multiply-add the compiler can vectorise; each row
/// of the block is read once and used for all the vectors.
#[inline(always)]
fn add_block<T: Storage>(
    dim: usize,
    m: &[T],
    (in_re, in_im): (&[T::Compute], &[T::Compute]),
    (acc_re, acc_im): (&mut [T::Compute], &mut [T::Compute]),
) {
    let nv = in_re.len() / dim;
    let zero = T::Compute::ZERO;
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

/// Halo buffers for fields of this many reals per site.
type Exchange<T> = (usize, Box<dyn HaloExchange<T>>);

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
    /// Build `P† D P`, a few columns at a time as the vectors of the fields: P of the coarse
    /// unit vectors of one component on every coarse site at once, and of that
    ///
    /// - each of the eight hops where it comes from another block, for the links. With φ kept
    ///   on the face of every block that the hop comes from and zero elsewhere, D of it on
    ///   the opposite face is that hop and nothing else (in blocks of more than two sites:
    ///   the part within a site stays on the first face, the other hops do not reach the
    ///   second), so this is the operator itself, in one sweep; in blocks of one or two sites
    ///   it is the hop of `HopTerms`. P† reads the one face;
    /// - D of all of φ: P† of it less the eight links is the block within the coarse site.
    pub fn build<F, S: Storage>(op: &F, work: &mut F::Work, basis: &Basis<T, S>) -> Self
    where
        F: HopTerms<Field<T, Spinor>>,
    {
        Self::build_with(op, work, basis, None, false)
            .expect("nothing is inverted without diagonal preconditioning")
    }

    /// `build` with the two options of the domain-wall multigrid (the design notes, §7.6.2, after
    /// `AFopr_Domainwall_coarse::generate_coarse_op` of the CUDA version):
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
        let (dim, block) = (basis.coarse_dim(), agg.block());
        let n_coarse = lat.num_sites();
        let matrix = 2 * dim * dim;
        let mut diag = vec![T::default(); n_coarse * matrix];
        let mut links = vec![T::default(); 2 * ND * n_coarse * matrix];
        let one = T::store(T::Compute::ONE);

        // As many columns at a time as keep the five fine fields (three here, two in the
        // operator) within about a GB, eight at most (P is read once for all of them).
        let reals = basis.fine_field(1).as_slice().len();
        let batch = ((1usize << 30) / (5 * reals * size_of::<T>())).clamp(1, 8.min(dim));
        // Write the columns `first`, `first + 1`, ... of every site's matrix from the vectors
        // of a coarse field.
        let store_columns = |target: &mut [T], first: usize, values: &Field<T, Coarse>| {
            for site in 0..n_coarse {
                for (v, part) in values.site(0, site).chunks_exact(2 * dim).enumerate() {
                    for row in 0..dim {
                        let at = site * matrix + 2 * (dim * row + first + v);
                        target[at] = part[2 * row];
                        target[at + 1] = part[2 * row + 1];
                    }
                }
            }
        };
        // Where a site is in its block, in direction mu.
        let in_block = |x: usize, mu: usize| fine_lat.coords(x)[mu] % block[mu];

        for first in (0..dim).step_by(batch) {
            let nv = batch.min(dim - first);
            let mut unit = basis.coarse_field(nv);
            for site in 0..n_coarse {
                for (v, part) in unit.site_mut(0, site).chunks_exact_mut(2 * dim).enumerate() {
                    part[2 * (first + v)] = one;
                }
            }
            let mut phi = basis.fine_field(nv);
            basis.prolong(&mut phi, &unit);
            if let Some(r) = right.as_mut() {
                let mut phi_right = phi.zeros_like();
                r.mult(&mut phi_right, &phi);
                phi = phi_right;
            }
            let (mut masked, mut out) = (phi.zeros_like(), phi.zeros_like());
            let (mut restricted, mut inside) = (unit.zeros_like(), unit.zeros_like());
            let site_len = phi.site_len();

            op.mult(work, &mut out, &phi, Variant::D);
            basis.restrict(&mut inside, &out);
            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                // The hop comes from the face `from` of a block onto the face `onto` of the
                // block before or after it.
                let last = block[mu] - 1;
                let (from, onto) = match dir {
                    Direction::Forward => (0, last),
                    Direction::Backward => (last, 0),
                };
                if block[mu] > 2 {
                    for_sites(masked.slab_mut(0), site_len, |x, m| {
                        if in_block(x, mu) == from {
                            m.copy_from_slice(phi.site(0, x));
                        } else {
                            m.fill(T::default());
                        }
                    });
                    op.mult(work, &mut out, &masked, Variant::D);
                    basis.restrict_where(&mut restricted, &out, |x| in_block(x, mu) == onto);
                } else {
                    out.as_mut_slice().fill(T::default());
                    match dir {
                        Direction::Forward => op.hop_forward(work, mu, &mut out, &phi),
                        Direction::Backward => op.hop_backward(work, mu, &mut out, &phi),
                    }
                    basis.restrict_where(&mut restricted, &out, |x| agg.crosses(x, mu, dir));
                    restricted.scale(&vec![op.hop_coefficient(); nv]);
                }
                let range = d * n_coarse * matrix..(d + 1) * n_coarse * matrix;
                store_columns(&mut links[range], first, &restricted);
                inside.axpy(&vec![C64::new(-1.0, 0.0); nv], &restricted);
            }
            store_columns(&mut diag, first, &inside);
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
            exchange: Mutex::new(None),
            ghost_links: OnceLock::new(),
            even_odd: None,
        })
    }

    /// The operator from its blocks and links as `blocks` gives them, made elsewhere (on a
    /// device), for a fine operator with `D† = γ5 D γ5`.
    pub fn from_parts(lat: &Arc<Lattice>, dim: usize, diag: Vec<T>, links: Vec<T>) -> Self {
        let matrix = 2 * dim * dim;
        assert_eq!(diag.len(), lat.num_sites() * matrix);
        assert_eq!(links.len(), 2 * ND * lat.num_sites() * matrix);
        Self {
            stencil: Stencil::nearest(lat, [Boundary::Periodic; ND]),
            lat: lat.clone(),
            dim,
            diag,
            links,
            diag_inv: None,
            scratch: None,
            adjoint_from_gamma5: true,
            exchange: Mutex::new(None),
            ghost_links: OnceLock::new(),
            even_odd: None,
        }
    }

    /// The coarse operator `R A P` of any map A on the fine lattice, cut down to the couplings of a
    /// coarse site with itself and its eight neighbours.
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
        assert_eq!(
            lat.grid, [1; ND],
            "probing a coarse operator on more than one process is not implemented"
        );
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
                        // The link in direction `dir` at X multiplies the field at X + dir, so the
                        // source is X + dir.
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
                            unreachable!("probing is on one process");
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
            exchange: Mutex::new(None),
            ghost_links: OnceLock::new(),
            even_odd: None,
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
    pub(super) fn gamma5(&self, v: &mut Field<T, Coarse>) {
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

    /// The neighbours of `inp` on other processes, by the face they arrive on, `site_len`
    /// reals each in the order of `Source::Buffer`; empty on one process.
    fn halo(&self, inp: &Field<T, Coarse>) -> [Vec<T>; FACES] {
        if !self.stencil.has_buffers() {
            return Default::default();
        }
        let site_len = inp.site_len();
        let mut slot = self.exchange.lock().unwrap_or_else(|e| e.into_inner());
        if slot.as_ref().is_none_or(|(len, _)| *len != site_len) {
            let spec = self.stencil.halo_spec(site_len);
            *slot = Some((site_len, T::halo(&*self.lat.comm, &spec)));
        }
        let ex = &mut slot.as_mut().expect("just set").1;
        for (face, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
            let buffer = ex.send_buffer(face);
            for (k, s) in self.stencil.send_sites(mu, dir).iter().enumerate() {
                buffer[k * site_len..(k + 1) * site_len].copy_from_slice(inp.site(0, *s as usize));
            }
        }
        ex.start();
        ex.wait();
        core::array::from_fn(|face| ex.recv_buffer(face).to_vec())
    }

    /// The links of the neighbours on other processes that `mult_adjoint` needs: for the
    /// hop from X to Y = X + dir, B_-dir(Y). Every process sends, through each face, the
    /// links in the direction of that face of the sites on it.
    fn exchange_links(&self) -> [Vec<T>; FACES] {
        if !self.stencil.has_buffers() {
            return Default::default();
        }
        let (matrix, n_coarse) = (2 * self.dim * self.dim, self.lat.num_sites());
        let mut ex = T::halo(&*self.lat.comm, &self.stencil.halo_spec(matrix));
        for (face, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
            let buffer = ex.send_buffer(face);
            for (k, s) in self.stencil.send_sites(mu, dir).iter().enumerate() {
                let at = (face * n_coarse + *s as usize) * matrix;
                buffer[k * matrix..(k + 1) * matrix].copy_from_slice(&self.links[at..at + matrix]);
            }
        }
        ex.start();
        ex.wait();
        core::array::from_fn(|face| ex.recv_buffer(face).to_vec())
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
            exchange: Mutex::new(None),
            ghost_links: OnceLock::new(),
            even_odd: self.even_odd.as_ref().map(|eo| EvenOdd {
                odd: eo.odd.clone(),
                inverse: convert(&eo.inverse),
            }),
        }
    }

    /// `out = D_c† inp` exactly, the conjugate transpose of the stored matrix: `(D_c† ψ)(X) = A(X)†
    /// ψ(X) + Σ_dir B_-dir(X + dir)† ψ(X + dir)`. For a Galerkin operator this is P† D† P, whether
    /// or not D† = γ5 D γ5 (`mult_dag` assumes it: for Möbius domain-wall fermions with c ≠ 0 it
    /// holds only approximately). Not for an operator preconditioned by its diagonal blocks.
    pub fn mult_adjoint(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        assert!(
            self.diag_inv.is_none(),
            "the exact adjoint of a diagonally preconditioned coarse operator"
        );
        let (dim, matrix) = (self.dim, 2 * self.dim * self.dim);
        let n_coarse = self.lat.num_sites();
        let site_len = inp.site_len();
        let ghost = self.halo(inp);
        let ghost_links = self.ghost_links.get_or_init(|| self.exchange_links());
        for_sites(out.slab_mut(0), site_len, |x, out| {
            out.fill(T::default());
            // (the block applied as its conjugate transpose, the field it multiplies)
            let mut terms: [(&[T], &[T]); 2 * ND + 1] =
                [(&self.diag[x * matrix..(x + 1) * matrix], inp.site(0, x)); 2 * ND + 1];
            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                let back = (d + ND) % (2 * ND);
                terms[d + 1] = match self.stencil.source(x, mu, dir) {
                    Source::Local { site, .. } => {
                        let at = (back * n_coarse + site) * matrix;
                        (&self.links[at..at + matrix], inp.site(0, site))
                    }
                    Source::Buffer { index, .. } => (
                        &ghost_links[d][index * matrix..(index + 1) * matrix],
                        &ghost[d][index * site_len..(index + 1) * site_len],
                    ),
                    Source::Zero => unreachable!("the coarse stencil is periodic"),
                };
            }
            for (m, v) in terms {
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

    /// The exact adjoint D_c† as an operator of its own, the conjugate transposes of the blocks
    /// of `mult_adjoint`: A(X)† and, for the hop from X to X + dir, B_-dir(X + dir)† (from
    /// another process where X + dir is there). Its hops join the same sites, so it has the
    /// even-odd form where D_c has it (`prepare_even_odd`), and the inverses of its blocks are
    /// (A(X)⁻¹)†. Not for an operator preconditioned by its diagonal blocks.
    pub fn adjoint(&self) -> Self {
        assert!(
            self.diag_inv.is_none(),
            "the exact adjoint of a diagonally preconditioned coarse operator"
        );
        let (dim, matrix) = (self.dim, 2 * self.dim * self.dim);
        let n_coarse = self.lat.num_sites();
        // (M†)_{row, col} = conj(M_{col, row})
        let dagger = |m: &[T], out: &mut [T]| {
            for row in 0..dim {
                for col in 0..dim {
                    let k = 2 * (dim * col + row);
                    out[2 * (dim * row + col)] = m[k];
                    out[2 * (dim * row + col) + 1] = T::store(T::Compute::ZERO - m[k + 1].load());
                }
            }
        };
        let mut diag = vec![T::default(); self.diag.len()];
        for x in 0..n_coarse {
            let at = x * matrix..(x + 1) * matrix;
            dagger(&self.diag[at.clone()], &mut diag[at]);
        }
        let ghost_links = self.exchange_links();
        let mut links = vec![T::default(); self.links.len()];
        for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
            let back = (d + ND) % (2 * ND);
            for x in 0..n_coarse {
                let from = match self.stencil.source(x, mu, dir) {
                    Source::Local { site, .. } => {
                        let at = (back * n_coarse + site) * matrix;
                        &self.links[at..at + matrix]
                    }
                    Source::Buffer { index, .. } => {
                        &ghost_links[d][index * matrix..(index + 1) * matrix]
                    }
                    Source::Zero => unreachable!("the coarse stencil is periodic"),
                };
                let at = (d * n_coarse + x) * matrix;
                dagger(from, &mut links[at..at + matrix]);
            }
        }
        Self {
            stencil: Stencil::nearest(&self.lat, [Boundary::Periodic; ND]),
            lat: self.lat.clone(),
            dim,
            diag,
            links,
            diag_inv: None,
            scratch: None,
            adjoint_from_gamma5: self.adjoint_from_gamma5,
            exchange: Mutex::new(None),
            ghost_links: OnceLock::new(),
            even_odd: None,
        }
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

    /// Make the even-odd form available (see the type): the parity of the coarse sites and
    /// the inverses of the diagonal blocks. `false`, and nothing done, where it does not
    /// exist: a coarse lattice with an odd extent (its hops then join sites of the same
    /// parity across the boundary), or an operator preconditioned by its diagonal blocks.
    pub fn prepare_even_odd(&mut self) -> Result<bool, MultigridError> {
        if self.diag_inv.is_some() || self.lat.global.iter().any(|l| l % 2 != 0) {
            return Ok(false);
        }
        let n_coarse = self.lat.num_sites();
        let odd = (0..n_coarse)
            .map(|x| self.lat.global_coords(x).iter().sum::<usize>() % 2 == 1)
            .collect();
        let inverse = invert_blocks(&self.diag, self.dim, n_coarse)?;
        self.even_odd = Some(EvenOdd { odd, inverse });
        Ok(true)
    }

    /// The operator as it is stored: the diagonal blocks `[site][row][column]` and the links
    /// `[direction][site][row][column]`, complex numbers as (re, im).
    pub fn blocks(&self) -> (&[T], &[T]) {
        (&self.diag, &self.links)
    }

    /// The inverses of the diagonal blocks, with the even-odd form.
    pub fn block_inverses(&self) -> Option<&[T]> {
        self.even_odd.as_ref().map(|eo| eo.inverse.as_slice())
    }

    /// The sites the eight hops of site `x` come from, in the order of the links; on one
    /// process.
    pub fn neighbours(&self, x: usize) -> [usize; 2 * ND] {
        DIRECTIONS.map(|(mu, dir)| match self.stencil.source(x, mu, dir) {
            Source::Local { site, .. } => site,
            _ => panic!("a coarse lattice on one process"),
        })
    }

    /// Whether `prepare_even_odd` has made the even-odd form available.
    pub fn has_even_odd(&self) -> bool {
        self.even_odd.is_some()
    }

    /// One pass of the even-odd form, over the coarse sites of one parity (the others are
    /// set to zero):
    ///
    ///   out(X) = [A(X)⁻¹] ( base(X) + A(X) d(X) ± Σ_dir B_dir(X) h(X + dir) ),
    ///
    /// every part optional: `diag_of` is d, `hops_of` is h and whether its sum is
    /// subtracted, `inverse` the factor in front.
    fn pass(
        &self,
        out: &mut Field<T, Coarse>,
        odd: bool,
        base: Option<&Field<T, Coarse>>,
        diag_of: Option<&Field<T, Coarse>>,
        hops_of: Option<(&Field<T, Coarse>, bool)>,
        inverse: bool,
    ) {
        let eo = self
            .even_odd
            .as_ref()
            .expect("prepare_even_odd comes first");
        let (dim, matrix) = (self.dim, 2 * self.dim * self.dim);
        let n_coarse = self.lat.num_sites();
        let site_len = out.site_len();
        let ghost = hops_of.map(|(h, _)| self.halo(h));
        for_sites(out.slab_mut(0), site_len, |x, out| {
            if eo.odd[x] != odd {
                out.fill(T::default());
                return;
            }
            let n = out.len() / 2;
            let zero = T::Compute::ZERO;
            let (mut acc_re, mut acc_im) = (vec![zero; n], vec![zero; n]);
            let (mut in_re, mut in_im) = (vec![zero; n], vec![zero; n]);
            let mut add = |m: &[T], v: &[T], minus: bool| {
                for j in 0..n {
                    let (re, im) = (v[2 * j].load(), v[2 * j + 1].load());
                    in_re[j] = if minus { -re } else { re };
                    in_im[j] = if minus { -im } else { im };
                }
                add_block::<T>(dim, m, (&in_re, &in_im), (&mut acc_re, &mut acc_im));
            };
            if let Some(d) = diag_of {
                add(
                    &self.diag[x * matrix..(x + 1) * matrix],
                    d.site(0, x),
                    false,
                );
            }
            if let (Some((h, minus)), Some(ghost)) = (hops_of, ghost.as_ref()) {
                for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                    let v = match self.stencil.source(x, mu, dir) {
                        Source::Local { site, .. } => h.site(0, site),
                        Source::Buffer { index, .. } => {
                            &ghost[d][index * site_len..(index + 1) * site_len]
                        }
                        Source::Zero => unreachable!("the coarse stencil is periodic"),
                    };
                    let at = (d * n_coarse + x) * matrix;
                    add(&self.links[at..at + matrix], v, minus);
                }
            }
            if let Some(b) = base {
                let b = b.site(0, x);
                for j in 0..n {
                    acc_re[j] += b[2 * j].load();
                    acc_im[j] += b[2 * j + 1].load();
                }
            }
            if inverse {
                in_re.copy_from_slice(&acc_re);
                in_im.copy_from_slice(&acc_im);
                acc_re.fill(zero);
                acc_im.fill(zero);
                let m = &eo.inverse[x * matrix..(x + 1) * matrix];
                add_block::<T>(dim, m, (&in_re, &in_im), (&mut acc_re, &mut acc_im));
            }
            for j in 0..n {
                out[2 * j] = T::store(acc_re[j]);
                out[2 * j + 1] = T::store(acc_im[j]);
            }
        });
    }

    /// `out = S inp`, the Schur complement on the even coarse sites (see the type); `inp` is
    /// zero on the odd sites and so is `out`. `odd` is scratch.
    pub fn schur(
        &self,
        out: &mut Field<T, Coarse>,
        inp: &Field<T, Coarse>,
        odd: &mut Field<T, Coarse>,
    ) {
        fit(odd, inp);
        self.pass(odd, true, None, None, Some((inp, false)), true);
        self.pass(out, false, None, Some(inp), Some((odd, true)), false);
    }

    /// The right-hand side of the Schur system for `D_c c = g`: `out = g_e − B_eo A_oo⁻¹ g_o`.
    pub fn schur_source(
        &self,
        out: &mut Field<T, Coarse>,
        g: &Field<T, Coarse>,
        odd: &mut Field<T, Coarse>,
    ) {
        fit(odd, g);
        self.pass(odd, true, Some(g), None, None, true);
        self.pass(out, false, Some(g), None, Some((odd, true)), false);
    }

    /// The solution of `D_c c = g` on all sites from that of the Schur system: `c` holds c_e
    /// (zero on the odd sites) and gets `c_o = A_oo⁻¹ (g_o − B_oe c_e)` added.
    pub fn schur_solution(
        &self,
        c: &mut Field<T, Coarse>,
        g: &Field<T, Coarse>,
        odd: &mut Field<T, Coarse>,
    ) {
        fit(odd, g);
        self.pass(odd, true, Some(g), None, Some((c, true)), true);
        c.axpy(&vec![C64::new(1.0, 0.0); c.num_vectors()], odd);
    }

    /// `out = inp` on the even sites, zero on the odd ones.
    pub fn even_part(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.pass(out, false, Some(inp), None, None, false);
    }

    /// `out = A_ee⁻¹ inp` on the even sites, zero on the odd ones: the block-Jacobi
    /// preconditioner of the Schur complement.
    pub fn even_block_inverse(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.pass(out, false, Some(inp), None, None, true);
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

        let ghost = self.halo(inp);

        for_sites(out.slab_mut(0), site_len, |x, out| {
            // (matrix, the field it multiplies)
            let mut terms: [(&[T], &[T]); 2 * ND + 1] =
                [(&self.diag[x * matrix..(x + 1) * matrix], inp.site(0, x)); 2 * ND + 1];
            for (d, (mu, dir)) in DIRECTIONS.into_iter().enumerate() {
                let at = (d * n_coarse + x) * matrix;
                let v = match self.stencil.source(x, mu, dir) {
                    Source::Local { site, .. } => inp.site(0, site),
                    Source::Buffer { index, .. } => {
                        &ghost[d][index * site_len..(index + 1) * site_len]
                    }
                    Source::Zero => unreachable!("the coarse stencil is periodic"),
                };
                terms[d + 1] = (&self.links[at..at + matrix], v);
            }

            // The sums of the nine terms for every vector, in the compute type, real and imaginary
            // parts apart; the inputs are loaded once per term with Γ5 already applied, so the
            // inner loop is a plain multiply-add the compiler can vectorise.
            let nv = out.len() / (2 * dim);
            let zero = T::Compute::ZERO;
            let (mut acc_re, mut acc_im) = (vec![zero; nv * dim], vec![zero; nv * dim]);
            let (mut in_re, mut in_im) = (vec![zero; nv * dim], vec![zero; nv * dim]);
            for (m, v) in terms {
                for j in 0..nv * dim {
                    let (re, im) = (v[2 * j].load(), v[2 * j + 1].load());
                    let negate = gamma5 && j % dim >= half;
                    in_re[j] = if negate { -re } else { re };
                    in_im[j] = if negate { -im } else { im };
                }
                add_block::<T>(dim, m, (&in_re, &in_im), (&mut acc_re, &mut acc_im));
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
