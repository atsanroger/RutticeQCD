//! Levels below the coarse lattice of a multigrid, as a preconditioner of a coarse
//! operator A_c (the working log (branch `notes`): the signed multilevel V-cycle, `sm:`).

use super::{CoarseOp, MultigridError};
use crate::kernel::{Number, Real};
use crate::numerics::{hermitian_eigen, invert_complex};
use crate::{Coarse, Field, LinearFn, Storage, Vector, C64, ND};
use std::cell::{Cell, RefCell};

/// `out = m v` for a dense row-major n × n m.
fn mat_vec(m: &[C64], n: usize, v: &[C64]) -> Vec<C64> {
    (0..n)
        .map(|i| (0..n).fold(C64::ZERO, |s, k| s + m[i * n + k] * v[k]))
        .collect()
}

/// The inverse of a dense row-major n × n matrix; zero where it is singular.
fn inverse(m: &[C64], n: usize) -> Result<Vec<C64>, MultigridError> {
    let mut a: Vec<f64> = m.iter().flat_map(|z| [z.re, z.im]).collect();
    let mut inv = vec![0.0; 2 * n * n];
    invert_complex(n, &mut a, &mut inv)
        .map_err(|_| MultigridError::SingularCoarseDiagonal { coarse_site: 0 })?;
    Ok(inv
        .as_chunks::<2>()
        .0
        .iter()
        .map(|z| C64::new(z[0], z[1]))
        .collect())
}

fn site_coords(s: usize, ext: &[usize; ND]) -> [usize; ND] {
    let mut c = [0; ND];
    let mut r = s;
    for mu in 0..ND {
        c[mu] = r % ext[mu];
        r /= ext[mu];
    }
    c
}

fn site_index(c: &[usize; ND], ext: &[usize; ND]) -> usize {
    (0..ND).rev().fold(0, |s, mu| s * ext[mu] + c[mu])
}

/// The global site of local site `x` of a coarse field, numbered like a lattice.
fn global_site<T: Storage>(f: &Field<T, Coarse>, x: usize) -> usize {
    let lat = f.lattice();
    site_index(&lat.global_coords(x), &lat.global)
}

/// Vector `j` of a coarse field on the whole global lattice, as plain complex numbers site
/// after site, on every process (one global sum).
fn get_vector<T: Storage>(f: &Field<T, Coarse>, j: usize) -> Vec<C64> {
    let per = f.site_len() / f.num_vectors();
    let mut reals = vec![0.0; f.lattice().num_global_sites() * per];
    for x in 0..f.num_sites() {
        let at = global_site(f, x) * per;
        for (o, v) in reals[at..at + per]
            .iter_mut()
            .zip(&f.site(0, x)[j * per..(j + 1) * per])
        {
            *o = v.load().to_f64();
        }
    }
    f.lattice().comm.sum(&mut reals);
    reals
        .as_chunks::<2>()
        .0
        .iter()
        .map(|z| C64::new(z[0], z[1]))
        .collect()
}

/// Vector `j` of a coarse field set from a vector on the whole global lattice: the sites of
/// this process.
fn set_vector<T: Storage>(f: &mut Field<T, Coarse>, j: usize, v: &[C64]) {
    let per = f.site_len() / f.num_vectors();
    for x in 0..f.num_sites() {
        let from = global_site(f, x) * per / 2;
        let s = &mut f.site_mut(0, x)[j * per..(j + 1) * per];
        for (o, z) in s.as_chunks_mut::<2>().0.iter_mut().zip(&v[from..]) {
            o[0] = T::store(Number::from_f64(z.re));
            o[1] = T::store(Number::from_f64(z.im));
        }
    }
}

/// The super-blocks and local vectors of one level.
struct Level {
    /// Complex components per site of the level above.
    parent_dim: usize,
    /// The sites of the level above in every super-block.
    sites: Vec<Vec<usize>>,
    /// Per super-block, `nq` local vectors one after the other.
    q: Vec<Vec<C64>>,
    nq: usize,
}

impl Level {
    fn restrict(&self, v: &[C64]) -> Vec<C64> {
        let (nq, pd) = (self.nq, self.parent_dim);
        let mut out = vec![C64::ZERO; self.sites.len() * nq];
        for (s, sv) in self.sites.iter().enumerate() {
            let lt = sv.len() * pd;
            for b in 0..nq {
                let col = &self.q[s][b * lt..(b + 1) * lt];
                let mut z = C64::ZERO;
                for (k, &x) in sv.iter().enumerate() {
                    for i in 0..pd {
                        z += col[k * pd + i].conj() * v[x * pd + i];
                    }
                }
                out[s * nq + b] = z;
            }
        }
        out
    }

    fn prolong_add(&self, y: &[C64], v: &mut [C64]) {
        let (nq, pd) = (self.nq, self.parent_dim);
        for (s, sv) in self.sites.iter().enumerate() {
            let lt = sv.len() * pd;
            for b in 0..nq {
                let c = y[s * nq + b];
                let col = &self.q[s][b * lt..(b + 1) * lt];
                for (k, &x) in sv.iter().enumerate() {
                    for i in 0..pd {
                        v[x * pd + i] += col[k * pd + i] * c;
                    }
                }
            }
        }
    }

    /// The level from the sites of the level above (extent `parent_ext`, `parent_dim`
    /// components, the upper half of which have Γ5 = -1) and test vectors on it.
    fn build(
        parent_ext: [usize; ND],
        parent_dim: usize,
        super_block: usize,
        tests: &[Vec<C64>],
    ) -> (Self, [usize; ND]) {
        assert!(
            parent_ext.iter().all(|e| e % super_block == 0),
            "super-blocks of {super_block} do not divide the extent {parent_ext:?}"
        );
        let ext = parent_ext.map(|e| e / super_block);
        let total: usize = ext.iter().product();
        let mut sites = vec![Vec::new(); total];
        for x in 0..parent_ext.iter().product() {
            let c = site_coords(x, &parent_ext);
            sites[site_index(&c.map(|v| v / super_block), &ext)].push(x);
        }
        let nq = 2 * tests.len();
        let q = sites
            .iter()
            .map(|sv| {
                let lt = sv.len() * parent_dim;
                let mut cols: Vec<Vec<C64>> = Vec::with_capacity(nq);
                for upper in [false, true] {
                    let first = cols.len();
                    for t in tests {
                        let mut c: Vec<C64> = (0..lt)
                            .map(|j| {
                                let (k, i) = (j / parent_dim, j % parent_dim);
                                if (i >= parent_dim / 2) == upper {
                                    t[sv[k] * parent_dim + i]
                                } else {
                                    C64::ZERO
                                }
                            })
                            .collect();
                        for _ in 0..2 {
                            for b in &cols[first..] {
                                let d = b
                                    .iter()
                                    .zip(&c)
                                    .fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y);
                                for (ci, bi) in c.iter_mut().zip(b) {
                                    *ci = *ci - d * *bi;
                                }
                            }
                        }
                        // A dependent vector is left as zero.
                        let n = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                        let scale = if n > 1e-8 { 1.0 / n } else { 0.0 };
                        cols.push(c.iter().map(|z| z.scale(scale)).collect());
                    }
                }
                cols.concat()
            })
            .collect();
        (
            Self {
                parent_dim,
                sites,
                q,
                nq,
            },
            ext,
        )
    }
}

/// A nearest-neighbour block operator on a level's lattice: per site the diagonal block and one
/// block per direction (0..4 forward, 4..8 backward), `nq × nq` row-major.
struct LevelOp {
    nq: usize,
    nbr: Vec<[usize; 2 * ND]>,
    blocks: Vec<Vec<Vec<C64>>>,
    diag_inv: Vec<Vec<C64>>,
}

impl LevelOp {
    fn mult(&self, v: &[C64]) -> Vec<C64> {
        let nq = self.nq;
        let mut out = vec![C64::ZERO; v.len()];
        for (t, b) in self.blocks.iter().enumerate() {
            let mut acc = mat_vec(&b[0], nq, &v[t * nq..(t + 1) * nq]);
            for d in 0..2 * ND {
                if b[1 + d].is_empty() {
                    continue;
                }
                let s = self.nbr[t][d];
                for (a, z) in acc
                    .iter_mut()
                    .zip(mat_vec(&b[1 + d], nq, &v[s * nq..(s + 1) * nq]))
                {
                    *a += z;
                }
            }
            out[t * nq..(t + 1) * nq].copy_from_slice(&acc);
        }
        out
    }

    /// The operator of `level` (lattice `ext`) from that of the level above (`apply`, on vectors of
    /// `parent_len` components), by probing: sources c_μ apart, c_μ the smallest divisor of the
    /// extent from 3 up (or the extent), share no neighbour.
    fn probe(
        level: &Level,
        ext: [usize; ND],
        parent_len: usize,
        apply: &mut dyn FnMut(&[C64]) -> Vec<C64>,
    ) -> Result<(Self, usize), MultigridError> {
        let nq = level.nq;
        let total: usize = ext.iter().product();
        let nbr: Vec<[usize; 2 * ND]> = (0..total)
            .map(|s| {
                let c = site_coords(s, &ext);
                let mut n = [0; 2 * ND];
                for mu in 0..ND {
                    let (mut f, mut b) = (c, c);
                    f[mu] = (c[mu] + 1) % ext[mu];
                    b[mu] = (c[mu] + ext[mu] - 1) % ext[mu];
                    n[mu] = site_index(&f, &ext);
                    n[ND + mu] = site_index(&b, &ext);
                }
                n
            })
            .collect();
        let used = |d: usize| -> bool {
            let e = ext[d % ND];
            !(e == 1 || (e == 2 && d >= ND))
        };
        let mut blocks: Vec<Vec<Vec<C64>>> = (0..total)
            .map(|_| {
                (0..=2 * ND)
                    .map(|k| {
                        if k == 0 || used(k - 1) {
                            vec![C64::ZERO; nq * nq]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect()
            })
            .collect();
        let colours: [usize; ND] = ext.map(|e| (3..=e).find(|k| e % k == 0).unwrap_or(e).max(1));
        let n_colours: usize = colours.iter().product();
        let mut applications = 0;
        for colour in 0..n_colours {
            let mut rest = colour;
            let want = colours.map(|k| {
                let w = rest % k;
                rest /= k;
                w
            });
            let sources: Vec<usize> = (0..total)
                .filter(|&s| {
                    let c = site_coords(s, &ext);
                    (0..ND).all(|mu| c[mu] % colours[mu] == want[mu])
                })
                .collect();
            for j in 0..nq {
                let mut e = vec![C64::ZERO; total * nq];
                for &s in &sources {
                    e[s * nq + j] = C64::new(1.0, 0.0);
                }
                let mut p = vec![C64::ZERO; parent_len];
                level.prolong_add(&e, &mut p);
                let r = level.restrict(&apply(&p));
                applications += 1;
                for &s in &sources {
                    for i in 0..nq {
                        blocks[s][0][i * nq + j] = r[s * nq + i];
                    }
                    // A(t, s) with s = t + dir is stored at t = s - dir in direction dir.
                    for d in 0..2 * ND {
                        if !used(d) {
                            continue;
                        }
                        let opposite = if d < ND { d + ND } else { d - ND };
                        let t = nbr[s][opposite];
                        for i in 0..nq {
                            blocks[t][1 + d][i * nq + j] = r[t * nq + i];
                        }
                    }
                }
            }
        }
        let diag_inv = blocks
            .iter()
            .map(|b| inverse(&b[0], nq))
            .collect::<Result<_, _>>()?;
        Ok((
            Self {
                nq,
                nbr,
                blocks,
                diag_inv,
            },
            applications,
        ))
    }
}

/// The levels below the coarse lattice as a V-cycle for a coarse operator A_c (see the
/// top of the file). `apply` adds Q₂ M₂ Q₂† r to `out`.
pub struct SignedMultilevel {
    levels: Vec<Level>,
    ops: Vec<LevelOp>,
    /// The lattice of each level's operator.
    exts: Vec<[usize; ND]>,
    coarsest_inv: Vec<C64>,
    /// Applications of A_c to build it.
    pub setup_applications: usize,
}

/// Refinement of the test vectors on the lattice of one level before the level below it is built
/// from them: `rounds` times, every vector t is replaced by an approximation of A_ℓ⁻¹ t
/// (`iterations` steps of GCR on the level's operator, preconditioned by its V-cycle with the
/// levels built so far), normalised, and the levels below are built again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LevelBootstrap {
    pub rounds: usize,
    pub iterations: usize,
}

fn cdot(a: &[C64], b: &[C64]) -> C64 {
    a.iter()
        .zip(b)
        .fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y)
}

/// `iterations` steps of GCR on A x = b, preconditioned by `m`, from x = 0.
fn gcr(a: &LevelOp, m: &dyn Fn(&[C64]) -> Vec<C64>, b: &[C64], iterations: usize) -> Vec<C64> {
    let mut x = vec![C64::ZERO; b.len()];
    let mut r = b.to_vec();
    let (mut ps, mut aps): (Vec<Vec<C64>>, Vec<Vec<C64>>) = (Vec::new(), Vec::new());
    for _ in 0..iterations {
        let mut z = m(&r);
        let mut az = a.mult(&z);
        for (p, ap) in ps.iter().zip(&aps) {
            let c = cdot(ap, &az);
            for ((zi, pi), (azi, api)) in z.iter_mut().zip(p).zip(az.iter_mut().zip(ap)) {
                *zi = *zi - c * *pi;
                *azi = *azi - c * *api;
            }
        }
        let n = cdot(&az, &az).re.sqrt();
        if n == 0.0 {
            break;
        }
        for (zi, azi) in z.iter_mut().zip(az.iter_mut()) {
            *zi = zi.scale(1.0 / n);
            *azi = azi.scale(1.0 / n);
        }
        let beta = cdot(&az, &r);
        for ((xi, zi), (ri, azi)) in x.iter_mut().zip(&z).zip(r.iter_mut().zip(&az)) {
            *xi += beta * *zi;
            *ri = *ri - beta * *azi;
        }
        ps.push(z);
        aps.push(az);
    }
    x
}

impl SignedMultilevel {
    /// The levels for `a` from test vectors on the coarse lattice (each one vector of
    /// `vectors`), with super-blocks of `super_blocks[ℓ]` sites per direction on level
    /// ℓ + 2; the last level is solved exactly.
    pub fn build<T: Storage>(
        a: &mut CoarseOp<T>,
        vectors: &[Field<f64, Coarse>],
        super_blocks: &[usize],
    ) -> Result<Self, MultigridError> {
        Self::build_with(a, vectors, super_blocks, &[])
    }

    /// `build`, with the test vectors on the lattice of level ℓ + 2 refined by
    /// `bootstrap[ℓ]` (missing entries: none) before level ℓ + 3 is built from them.
    pub fn build_with<T: Storage>(
        a: &mut CoarseOp<T>,
        vectors: &[Field<f64, Coarse>],
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<Self, MultigridError> {
        assert!(!super_blocks.is_empty() && !vectors.is_empty());
        let lat = a.lattice().clone();
        let dim = a.dim();
        let like_t: Field<T, Coarse> = Field::with_shape(&lat, 2 * dim, 1, 1, 1);
        let tests: Vec<Vec<C64>> = vectors.iter().map(|v| get_vector(v, 0)).collect();
        // The first level: its operator from A_c by probing.
        let (level, ext) = Level::build(lat.global, dim, super_blocks[0], &tests);
        let parent_len = lat.global.iter().product::<usize>() * dim;
        let mut apply = |v: &[C64]| -> Vec<C64> {
            let (mut f, mut out) = (like_t.zeros_like(), like_t.zeros_like());
            set_vector(&mut f, 0, v);
            a.mult(&mut out, &f);
            get_vector(&out, 0)
        };
        let (op, applications) = LevelOp::probe(&level, ext, parent_len, &mut apply)?;
        let below: Vec<Vec<C64>> = tests.iter().map(|t| level.restrict(t)).collect();
        let mut me = Self {
            levels: vec![level],
            ops: vec![op],
            exts: vec![ext],
            coarsest_inv: Vec::new(),
            setup_applications: applications,
        };
        me.rebuild(1, &below, super_blocks)?;
        me.extend(1, below, super_blocks, bootstrap)?;
        Ok(me)
    }

    /// Refine the test vectors `tests` on the lattice of level `from` − 1's operator, from
    /// which levels `from`.. have been built, as `bootstrap[from − 1]` says, building those
    /// levels again after every round; then the same further down.
    fn extend(
        &mut self,
        from: usize,
        mut tests: Vec<Vec<C64>>,
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<(), MultigridError> {
        if from >= super_blocks.len() {
            return Ok(());
        }
        let boot = bootstrap.get(from - 1).copied().unwrap_or_default();
        for _ in 0..boot.rounds {
            let op = &self.ops[from - 1];
            let m = |r: &[C64]| self.vcycle(from - 1, r);
            tests = tests
                .iter()
                .map(|t| {
                    let x = gcr(op, &m, t, boot.iterations.max(1));
                    let n = cdot(&x, &x).re.sqrt();
                    x.iter().map(|z| z.scale(1.0 / n)).collect()
                })
                .collect();
            self.rebuild(from, &tests, super_blocks)?;
        }
        // The levels further down, from these vectors, refined in turn.
        let level = &self.levels[from];
        let below: Vec<Vec<C64>> = tests.iter().map(|t| level.restrict(t)).collect();
        self.extend(from + 1, below, super_blocks, bootstrap)
    }

    /// Levels `from`.. built again from `tests` (on the lattice of level `from` − 1's
    /// operator), without refinement; the coarsest inverse again.
    fn rebuild(
        &mut self,
        from: usize,
        tests: &[Vec<C64>],
        super_blocks: &[usize],
    ) -> Result<(), MultigridError> {
        self.levels.truncate(from);
        self.ops.truncate(from);
        self.exts.truncate(from);
        let mut tests = tests.to_vec();
        for &sb in &super_blocks[from..] {
            let ext = *self.exts.last().expect("a level above");
            let parent_dim = self.ops.last().expect("a level above").nq;
            let (level, level_ext) = Level::build(ext, parent_dim, sb, &tests);
            let parent_len = ext.iter().product::<usize>() * parent_dim;
            let above = self.ops.last().expect("a level above");
            let mut apply = |v: &[C64]| above.mult(v);
            let (op, _) = LevelOp::probe(&level, level_ext, parent_len, &mut apply)?;
            tests = tests.iter().map(|t| level.restrict(t)).collect();
            self.levels.push(level);
            self.ops.push(op);
            self.exts.push(level_ext);
        }
        // The coarsest level dense.
        let last = self.ops.last().expect("at least one level");
        let n = self.exts.last().unwrap().iter().product::<usize>() * last.nq;
        let mut dense = vec![C64::ZERO; n * n];
        for j in 0..n {
            let mut e = vec![C64::ZERO; n];
            e[j] = C64::new(1.0, 0.0);
            for (i, z) in last.mult(&e).into_iter().enumerate() {
                dense[i * n + j] = z;
            }
        }
        self.coarsest_inv = inverse(&dense, n)?;
        Ok(())
    }
    /// Complex components of the coarsest level, solved densely.
    pub fn coarsest_dim(&self) -> usize {
        (self.coarsest_inv.len() as f64).sqrt() as usize
    }

    fn vcycle(&self, l: usize, r: &[C64]) -> Vec<C64> {
        if l + 1 == self.levels.len() {
            return mat_vec(&self.coarsest_inv, r.len(), r);
        }
        let below = &self.levels[l + 1];
        let zc = self.vcycle(l + 1, &below.restrict(r));
        let mut z = vec![C64::ZERO; r.len()];
        below.prolong_add(&zc, &mut z);
        let az = self.ops[l].mult(&z);
        let nq = self.ops[l].nq;
        for (t, dinv) in self.ops[l].diag_inv.iter().enumerate() {
            let r2: Vec<C64> = (0..nq).map(|i| r[t * nq + i] - az[t * nq + i]).collect();
            for (i, c) in mat_vec(dinv, nq, &r2).into_iter().enumerate() {
                z[t * nq + i] += c;
            }
        }
        z
    }

    /// `out = Q₂ M₂ Q₂† inp`, every vector of the field by itself (on several processes, the
    /// same V-cycle on every process, on the gathered vector).
    pub fn apply<T: Storage>(&self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        let first = &self.levels[0];
        for j in 0..inp.num_vectors() {
            let z = self.vcycle(0, &first.restrict(&get_vector(inp, j)));
            let mut v =
                vec![C64::ZERO; first.sites.iter().map(Vec::len).sum::<usize>() * first.parent_dim];
            first.prolong_add(&z, &mut v);
            set_vector(out, j, &v);
        }
    }
}

/// Dense matrices on the coarse sites, applied site by site.
pub(super) struct BlockMap {
    dim: usize,
    /// Per coarse site, row-major.
    blocks: Vec<Vec<C64>>,
}

impl BlockMap {
    /// The exact inverses of the diagonal blocks of `a`.
    pub(super) fn inverse_diagonal<T: Storage>(a: &CoarseOp<T>) -> Result<Self, MultigridError> {
        let dim = a.dim();
        let blocks = (0..a.lattice().num_sites())
            .map(|x| {
                let mut m: Vec<f64> = a
                    .diagonal_block(x)
                    .iter()
                    .flat_map(|z| [z.re, z.im])
                    .collect();
                let mut inv = vec![0.0; 2 * dim * dim];
                invert_complex(dim, &mut m, &mut inv)
                    .map_err(|_| MultigridError::SingularCoarseDiagonal { coarse_site: x })?;
                Ok(inv
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|z| C64::new(z[0], z[1]))
                    .collect())
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { dim, blocks })
    }

    /// The inverses of the diagonal blocks of the exact adjoint of `a`: (A_xx†)⁻¹ = (A_xx⁻¹)†.
    pub(super) fn inverse_diagonal_adjoint(a: &CoarseOp<f64>) -> Result<Self, MultigridError> {
        let mut m = Self::inverse_diagonal(a)?;
        let dim = m.dim;
        for b in m.blocks.iter_mut() {
            *b = (0..dim * dim)
                .map(|ik| b[(ik % dim) * dim + ik / dim].conj())
                .collect();
        }
        Ok(m)
    }

    /// |H_xx|⁻¹ = V |Λ|⁻¹ V† for H = Γ5 (Γ5 A Γ5) = A Γ5, the operator MINRES works on
    /// when `a` holds P†D_PV P and is applied as its Γ5-adjoint (Γ5 = +1 on the first half
    /// of the components). Hermitian positive definite, as MINRES needs of a preconditioner.
    pub(super) fn abs_inverse_of_a_gamma5(a: &CoarseOp<f64>) -> Self {
        let dim = a.dim();
        let blocks = (0..a.lattice().num_sites())
            .map(|x| {
                let mut h = a.diagonal_block(x);
                for row in 0..dim {
                    for col in dim / 2..dim {
                        h[row * dim + col] = -h[row * dim + col];
                    }
                }
                let (lam, v) = hermitian_eigen(dim, &h);
                let mut m = vec![C64::ZERO; dim * dim];
                for i in 0..dim {
                    for k in 0..dim {
                        let mut z = C64::ZERO;
                        for (j, l) in lam.iter().enumerate() {
                            z += v[i * dim + j].scale(1.0 / l.abs().max(1e-300))
                                * v[k * dim + j].conj();
                        }
                        m[i * dim + k] = z;
                    }
                }
                m
            })
            .collect();
        Self { dim, blocks }
    }
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for BlockMap {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        let dim = self.dim;
        let site_len = inp.site_len();
        crate::for_sites(out.slab_mut(0), site_len, |x, dst| {
            let m = &self.blocks[x];
            let src = inp.site(0, x);
            for (dst, src) in dst.chunks_exact_mut(2 * dim).zip(src.chunks_exact(2 * dim)) {
                for i in 0..dim {
                    let mut z = C64::ZERO;
                    for k in 0..dim {
                        let v =
                            C64::new(src[2 * k].load().to_f64(), src[2 * k + 1].load().to_f64());
                        z += m[i * dim + k] * v;
                    }
                    dst[2 * i] = T::store(Number::from_f64(z.re));
                    dst[2 * i + 1] = T::store(Number::from_f64(z.im));
                }
            }
        });
    }
}

/// A coarse operator shared between its solver and its preconditioner, counting its
/// applications.
pub(super) struct CountedShared<'a, T: Storage> {
    pub(super) a: &'a RefCell<CoarseOp<T>>,
    pub(super) n: &'a Cell<usize>,
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for CountedShared<'_, T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.borrow_mut().mult(out, inp);
    }
}

/// The preconditioner of the quark factor: the inverses of its diagonal blocks B, or with
/// levels below the coarse lattice multiplicatively z₁ = Q M Q† r, z = z₁ + B⁻¹ (r − A z₁)
/// (one more application of A, counted); an additive combination made BiCGStab worse.
pub(super) struct LevelsAndBlocks<'a, T: Storage> {
    /// The inverses of the diagonal blocks; none for an operator preconditioned by them,
    /// whose diagonal blocks are the identity.
    pub(super) blocks: Option<&'a mut BlockMap>,
    pub(super) levels: Option<&'a SignedMultilevel>,
    pub(super) a: &'a RefCell<CoarseOp<T>>,
    pub(super) n: &'a Cell<usize>,
}

impl<T: Storage> LinearFn<Field<T, Coarse>> for LevelsAndBlocks<'_, T> {
    fn mult(&mut self, out: &mut Field<T, Coarse>, inp: &Field<T, Coarse>) {
        let Some(levels) = self.levels else {
            match self.blocks.as_mut() {
                Some(b) => b.mult(out, inp),
                None => out.copy_from(inp),
            }
            return;
        };
        let nv = inp.num_vectors();
        let mut z1 = inp.zeros_like();
        levels.apply(&mut z1, inp);
        let mut r = inp.zeros_like();
        self.a.borrow_mut().mult(&mut r, &z1);
        self.n.set(self.n.get() + nv);
        r.aypx(&vec![C64::new(-1.0, 0.0); nv], inp);
        match self.blocks.as_mut() {
            Some(b) => b.mult(out, &r),
            None => out.copy_from(&r),
        }
        out.axpy(&vec![C64::new(1.0, 0.0); nv], &z1);
    }
}
