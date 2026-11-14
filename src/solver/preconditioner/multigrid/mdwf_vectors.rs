//! The test vectors of the even-odd domain-wall multigrid (`mdwf_even_odd`): the
//! Chebyshev-filtered subspace iteration for the low modes, and the rounds of the bootstrap.

use super::mdwf_cycle::PvEoCycle;
use super::mdwf_even_odd::{
    pv_eo_basis_stored, pv_eo_coarse_vectors, BasisStorage, CoarseKrylov, PvEoBasis, PvEoParams,
};
use super::{Aggregation, LevelBootstrap, MultigridError};
use crate::fermion::{EoForm, Mdwf, PvEvenOdd};
use crate::kernel::{GammaRepre, Number, Real};
use crate::numerics::hermitian_eigen;
use crate::solver::{largest_magnitude, Solver, Stop};
use crate::{
    Aos, EvenOddIndex, Field, Half, Lattice, LinearFn, Parity, Spinor, Storage, Subspace, TestRng,
    Vector, C64,
};
use rayon::prelude::*;
use std::sync::Arc;

type HalfField<L = Aos, F = f64> = Field<F, Spinor, Half, L>;

/// Settings of the Chebyshev-filtered subspace iteration for the test vectors.
#[derive(Clone, Copy, Debug)]
pub struct LowModeParams {
    /// Vectors iterated (M).
    pub vectors: usize,
    /// Rounds of filtering and Rayleigh-Ritz.
    pub rounds: usize,
    /// Vectors filtered at a time (memory: the work fields of Ã are this large).
    pub chunk: usize,
    pub seed: u64,
}

impl Default for LowModeParams {
    /// The cheap setup of the log: M = 16, two rounds (with a bootstrap after them).
    fn default() -> Self {
        Self {
            vectors: 16,
            rounds: 2,
            chunk: 4,
            seed: 21,
        }
    }
}

/// `out = Σ_i x_i c[i n + j]` for every j, in place, site by site (no copy of x).
fn rotate<F: Storage>(x: &mut HalfField<Aos, F>, c: &[C64]) {
    let n = x.num_vectors();
    let site_len = x.site_len();
    let reals = site_len / n;
    crate::for_sites(x.slab_mut(0), site_len, |_, s| {
        let old: Vec<C64> = s
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| C64::new(z[0].load().to_f64(), z[1].load().to_f64()))
            .collect();
        for j in 0..n {
            for d in 0..reals / 2 {
                let mut z = C64::ZERO;
                for i in 0..n {
                    z += old[i * reals / 2 + d] * c[i * n + j];
                }
                s[j * reals + 2 * d] = F::store(Number::from_f64(z.re));
                s[j * reals + 2 * d + 1] = F::store(Number::from_f64(z.im));
            }
        }
    });
}

/// ⟨x_i, y_j⟩ for all i, j (row-major, x rows) in one sweep over the sites: a site holds its
/// part of every vector of a set, so each vector is read once (instead of a copy and a dot
/// product for every pair).
fn gram<F: Storage>(x: &HalfField<Aos, F>, y: &HalfField<Aos, F>) -> Vec<C64> {
    let (nx, ny) = (x.num_vectors(), y.num_vectors());
    let reals = x.site_len() / nx;
    assert!(x.num_sites() == y.num_sites() && y.site_len() / ny == reals);
    assert!(x.outer() == 1 && y.outer() == 1);
    let mut sums = crate::reduce_sites_dyn(x.num_sites(), 2 * nx * ny, |site, acc| {
        let (a, b) = (x.site(0, site), y.site(0, site));
        for (i, ai) in a.chunks_exact(reals).enumerate() {
            for (j, bj) in b.chunks_exact(reals).enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (p, q) in ai.as_chunks::<2>().0.iter().zip(bj.as_chunks::<2>().0) {
                    let (pr, pi) = (p[0].load().to_f64(), p[1].load().to_f64());
                    let (qr, qi) = (q[0].load().to_f64(), q[1].load().to_f64());
                    re += pr * qr + pi * qi;
                    im += pr * qi - pi * qr;
                }
                acc[2 * (i * ny + j)] += re;
                acc[2 * (i * ny + j) + 1] += im;
            }
        }
    });
    x.lattice().comm.sum(&mut sums);
    sums.as_chunks::<2>()
        .0
        .iter()
        .map(|[re, im]| C64::new(*re, *im))
        .collect()
}

impl<F: Storage> Subspace for HalfField<Aos, F> {
    fn rotate(&mut self, c: &[C64]) {
        rotate(self, c);
    }

    fn gram(&self, y: &Self) -> Vec<C64> {
        gram(self, y)
    }
}

/// Eigenvalues ascending, with the eigenvectors (columns) in the same order.
fn sorted_eigen(n: usize, a: &[C64]) -> (Vec<f64>, Vec<C64>) {
    let (values, v) = hermitian_eigen(n, a);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|i, j| values[*i].total_cmp(&values[*j]));
    let mut vectors = vec![C64::ZERO; n * n];
    for (j, o) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + j] = v[i * n + o];
        }
    }
    (order.iter().map(|o| values[*o]).collect(), vectors)
}

/// Orthonormalise the vectors of x through their Gram matrix (Löwdin, x ← x G^-1/2 V).
pub fn orthonormalise_all<V: Subspace>(x: &mut V) {
    let n = x.num_vectors();
    let (values, mut c) = sorted_eigen(n, &x.gram(x));
    for (j, l) in values.iter().enumerate() {
        assert!(*l > 1e-13 * values[n - 1], "the subspace lost rank");
        for i in 0..n {
            c[i * n + j] = c[i * n + j].scale(1.0 / l.sqrt());
        }
    }
    x.rotate(&c);
}

/// `vectors` random vectors with `ls` spinors per site on the even sites, from `seed`: the
/// even sites of `random_spinor(lat, ls * vectors, seed)`, each site from its own draw of the
/// generator, without that field on the whole lattice.
pub fn random_even<F: Storage>(
    lat: &Arc<Lattice>,
    index: &EvenOddIndex,
    ls: usize,
    vectors: usize,
    seed: u64,
) -> HalfField<Aos, F> {
    let mut x = HalfField::<Aos, F>::with_shape(lat, 24, ls, vectors, 1);
    x.set_parity(Parity::Even);
    let n = x.site_len();
    let globals: Vec<usize> = (0..x.num_sites())
        .map(|half| lat.global_site_index(lat.global_coords(index.full_site(Parity::Even, half))))
        .collect();
    x.as_mut_slice()
        .par_chunks_mut(n)
        .zip(&globals)
        .for_each(|(d, &global)| {
            let mut rng = TestRng::skipped(seed, (global * n) as u64);
            for d in d {
                *d = F::store(Number::from_f64(rng.uniform()));
            }
        });
    x
}

/// The random vectors on the even sites that `pv_eo_low_modes` starts from.
pub fn pv_eo_start<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> HalfField<Aos, F> {
    let index = op.even_odd_index().expect("an even-odd decomposition");
    random_even(lat, index, op.params().ls, params.vectors, params.seed)
}

/// Approximate eigenvectors of the smallest eigenvalues of a Hermitian positive operator, from
/// the vectors `x`: Chebyshev-filtered subspace iteration with Rayleigh-Ritz, the degree
/// following √(hi/lo) (at least 40), lo the largest Ritz value of the round before (hi/1000 at
/// first). `normal(like)` is the operator for sets of as many vectors as `like`. With the
/// vectors, their Ritz values.
pub fn low_modes<V: Subspace, N: LinearFn<V>>(
    mut normal: impl FnMut(&V) -> N,
    mut x: V,
    params: LowModeParams,
) -> (V, Vec<f64>) {
    let m = x.num_vectors();
    let like = x.select(&[0]);
    let mut start = x.select(&[0]);
    let hi = 1.1 * largest_magnitude(&mut normal(&like), &mut start, 50);
    let mut lo = hi / 1000.0;
    let mut theta = vec![0.0; m];
    let chunks: Vec<Vec<usize>> = (0..m)
        .collect::<Vec<_>>()
        .chunks(params.chunk.max(1))
        .map(<[usize]>::to_vec)
        .collect();
    for _ in 0..params.rounds {
        let degree = ((1.5 * (hi / lo).sqrt()).ceil() as usize).clamp(40, 1000);
        for c in &chunks {
            let xc = x.select(c);
            let mut y = xc.zeros_like();
            crate::solver::ChebyshevFilter::new(normal(&xc), degree, lo, hi, &xc).mult(&mut y, &xc);
            x.assign(c, &y);
        }
        orthonormalise_all(&mut x);
        // G = X† A X, a chunk of columns at a time.
        let mut g = vec![C64::ZERO; m * m];
        for c in &chunks {
            let xc = x.select(c);
            let mut w = xc.zeros_like();
            normal(&xc).mult(&mut w, &xc);
            let part = x.gram(&w);
            for i in 0..m {
                for (k, j) in c.iter().enumerate() {
                    g[i * m + j] = part[i * c.len() + k];
                }
            }
        }
        let (values, v) = sorted_eigen(m, &g);
        x.rotate(&v);
        theta = values;
        lo = theta[m - 1];
    }
    (x, theta)
}

/// Approximate low modes of Ã (the right singular vectors of its smallest singular values,
/// eigenvectors of Ã†Ã) for the basis: `low_modes` on `params.vectors` random vectors on the
/// even sites.
pub fn pv_eo_low_modes<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    lat: &Arc<Lattice>,
    params: LowModeParams,
) -> (HalfField<Aos, F>, Vec<f64>) {
    low_modes(
        |c: &HalfField<Aos, F>| crate::Normal::new(PvEvenOdd::new(op, pv, EoForm::Right, c), c),
        pv_eo_start(op, lat, params),
        params,
    )
}

/// Settings of the multigrid-accelerated improvement of the test vectors.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapParams {
    /// The rounds: exactly this many without `min_gain`, at most this many with it.
    pub rounds: usize,
    /// The rounds end once the cycle of the next would lower the lowest Ritz value of Ã†Ã on the
    /// vectors by less than this factor (`bootstrap_rounds`): the vectors have reached the
    /// lowest modes. `None`: all `rounds` are done.
    pub min_gain: Option<f64>,
    /// Rounds done before `min_gain` can end them. HMC sets 1 when it bootstraps again after a
    /// stalled solve: the stall shows that the vectors are not good enough, whatever the probe
    /// on the lowest Ritz vector says.
    pub min_rounds: usize,
    /// Steps of FGMRES on Ã x = y with the cycle, per vector and round.
    pub steps: usize,
    /// Settings of the cycle used (its coarse tolerance, smoother).
    pub cycle: PvEoParams,
}

impl Default for BootstrapParams {
    /// Two rounds of four steps, without the gain; the parameter files use `automatic`.
    fn default() -> Self {
        Self {
            rounds: 2,
            min_gain: None,
            min_rounds: 0,
            steps: 4,
            cycle: PvEoParams::default(),
        }
    }
}

impl BootstrapParams {
    /// Rounds of four steps until the next would gain less than a factor 4, at least 2 and at
    /// most 8: what a parameter file without `bootstrap_number_of_rounds` asks. Two rounds leave
    /// the vectors a factor 60 above the lowest modes on 24⁴ at m = 0.001, where the solves with D
    /// of HMC then stall; the third round gains 20-31, a fourth less than 2. After the first round
    /// a weak cycle (few vectors or smoother steps) can gain less than 4 on vectors still far
    /// from the lowest modes, so the probe begins before the third.
    pub fn automatic() -> Self {
        Self {
            rounds: 8,
            min_gain: Some(4.0),
            min_rounds: 2,
            ..Self::default()
        }
    }
}

/// A test vector improved by the multigrid built on the test vectors, `cycle`: `steps` steps of
/// FGMRES on A x = y_j from x = 0, preconditioned by the cycle.
pub fn bootstrap_vector<V: Vector>(
    a: &mut dyn LinearFn<V>,
    cycle: &mut dyn LinearFn<V>,
    yj: &V,
    steps: usize,
) -> V {
    let mut fgmres = crate::solver::Fgmres::new(Stop {
        tolerance_sqr: 1e-30,
        max_iterations: steps,
        max_restarts: 0,
    });
    let mut x = yj.zeros_like();
    let _ = fgmres.solve(a, Some(cycle), &mut x, yj);
    x
}

/// The lowest Ritz pair of A†A on the vectors y, which are orthonormal: the lowest eigenvalue θ
/// of (A y)†(A y) and its vector u in the span of y. θ measures how near the vectors are to the
/// lowest modes of A; it costs one application of A per vector. (The residual |A†A u − θ u|
/// does not measure it: on 24⁴ at m = 0.001, θ is 1.4e-8 of the largest eigenvalue, below
/// single precision, and the residual stays at 400 θ from the third round on.)
pub(crate) fn lowest_ritz_pair<H: Subspace>(a: &mut dyn LinearFn<H>, y: &H) -> (f64, H) {
    let n = y.num_vectors();
    let mut ay = y.zeros_like();
    for j in 0..n {
        let yj = y.select(&[j]);
        let mut t = yj.zeros_like();
        a.mult(&mut t, &yj);
        ay.assign(&[j], &t);
    }
    let (theta, v) = sorted_eigen(n, &ay.gram(&ay));
    let mut u = y.select(&[0]).zeros_like();
    for j in 0..n {
        u.axpy(&[v[j * n]], &y.select(&[j]));
    }
    (theta[0], u)
}

/// |A x|²/|x|² of one vector.
fn rayleigh_quotient<H: Vector>(a: &mut dyn LinearFn<H>, x: &H) -> f64 {
    let mut ax = x.zeros_like();
    a.mult(&mut ax, x);
    let (mut above, mut below) = ([0.0], [0.0]);
    ax.norm_sqr(&mut above);
    x.norm_sqr(&mut below);
    above[0] / below[0]
}

/// The levels below the coarse lattice of a cycle: the super-blocks of each level, how each is
/// bootstrapped, and whether the solves of the Pauli-Villars factor have them too.
pub type LevelLists<'l> = (&'l [usize], &'l [LevelBootstrap], bool);

/// Where the test vectors wait while a round of the bootstrap improves them one at a time: on
/// the CPU where they are (`InPlace`), on the device on the host (`DeviceSpinor::stash`), so that
/// the device holds only the vector worked on besides the cycle.
pub(crate) trait Waiting<H, E>: Sized {
    fn wait(y: H) -> Result<Self, E>;
    fn num_vectors(&self) -> usize;
    fn vector(&self, j: usize) -> Result<H, E>;
    fn set_vector(&mut self, j: usize, x: &H) -> Result<(), E>;
    /// All the vectors as one set again.
    fn back(self) -> Result<H, E>;
}

/// Test vectors waiting where they are.
pub(crate) struct InPlace<H>(H);

impl<H: Vector, E> Waiting<H, E> for InPlace<H> {
    fn wait(y: H) -> Result<Self, E> {
        Ok(Self(y))
    }
    fn num_vectors(&self) -> usize {
        self.0.num_vectors()
    }
    fn vector(&self, j: usize) -> Result<H, E> {
        Ok(self.0.select(&[j]))
    }
    fn set_vector(&mut self, j: usize, x: &H) -> Result<(), E> {
        self.0.assign(&[j], x);
        Ok(())
    }
    fn back(self) -> Result<H, E> {
        Ok(self.0)
    }
}

/// What a round of the bootstrap does with Ã and the cycle of the current vectors, given the
/// vectors: `bootstrap_vector` for each, the vectors waiting in `W` meanwhile.
pub(crate) type RoundWork<'r, H, W, E> =
    dyn FnMut(&mut dyn LinearFn<H>, &mut dyn LinearFn<H>, H) -> Result<(W, bool), E> + 'r;

/// How a backend makes the cycle of a round of the bootstrap: `with_cycle(y, levels, work)`
/// makes P and the coarse correction of the vectors y (with the levels below the coarse lattice
/// of its settings when `levels`) and the smoother, hands them with Ã and y to `work`, and gives
/// the vectors back once its cycle is gone. `work` also says whether the bootstrap ends on this
/// cycle (the probe found the round not needed): its P and coarse operators are then those of
/// the vectors returned, and a backend may keep them for the solve.
pub(crate) type WithCycle<'w, H, W, E> =
    dyn FnMut(H, bool, &mut RoundWork<'_, H, W, E>) -> Result<H, E> + 'w;

/// The bootstrap, the same on every backend: rounds, each improving every test vector with the
/// cycle made of the vectors of the round before (`with_cycle`), then the vectors
/// orthonormalized. `lowest(y)` is the lowest Ritz pair of Ã†Ã on them (`lowest_ritz_pair` with
/// the backend's Ã).
///
/// Without `params.min_gain` all `params.rounds` are done. With it a round begins with a probe:
/// its cycle improves the lowest Ritz vector alone, and when that lowers |Ã x|²/|x|² by less
/// than the gain against the Ritz value, the vectors have reached the lowest modes and the
/// bootstrap ends with them unchanged; the first `params.min_rounds` rounds have no probe. The probe needs the cycle of the round, that of the
/// current vectors: the cycle of the round before, made of the vectors before it, does not tell
/// (24⁴, m 0.001: after two rounds it gains 1.0-1.5 where the next round gains 20-31). So ending
/// costs one cycle made and one vector improved, about half a round.
///
/// `on_vector(round, j, x)`, `on_round(round, y, lowest)` and, when the probe ends the rounds,
/// `on_end(round, lowest, probed)` report.
///
/// The cycles have the levels below the coarse lattice from the second round on. In the first,
/// on the filtered vectors, the coarse solves were as fast without them (24⁴, m 0.001, on the
/// device: 19.8 s with, 19.7 s without, and 7 s to make them); in the second they took 26 s
/// against 45 s.
pub(crate) fn bootstrap_rounds<H: Subspace, W: Waiting<H, E>, E>(
    mut y: H,
    params: BootstrapParams,
    with_cycle: &mut WithCycle<'_, H, W, E>,
    lowest: &mut dyn FnMut(&H) -> (f64, H),
    on_vector: &mut dyn FnMut(usize, usize, &H),
    on_round: &mut dyn FnMut(usize, &H, f64),
    on_end: &mut dyn FnMut(usize, f64, f64),
) -> Result<H, E> {
    let (mut value, mut vector) = lowest(&y);
    for round in 0..params.rounds {
        let mut ended = None;
        {
            let mut work = |a: &mut dyn LinearFn<H>, cycle: &mut dyn LinearFn<H>, vectors: H| {
                // The vectors wait first: on the device the probe then has the room of a round.
                let mut waiting = W::wait(vectors)?;
                if let Some(gain) = params.min_gain.filter(|_| round >= params.min_rounds) {
                    let x = bootstrap_vector(&mut *a, &mut *cycle, &vector, params.steps);
                    let probed = rayleigh_quotient(&mut *a, &x);
                    if value < gain * probed {
                        ended = Some(probed);
                        return Ok((waiting, true));
                    }
                }
                for j in 0..waiting.num_vectors() {
                    let x =
                        bootstrap_vector(&mut *a, &mut *cycle, &waiting.vector(j)?, params.steps);
                    on_vector(round, j, &x);
                    waiting.set_vector(j, &x)?;
                }
                Ok((waiting, false))
            };
            y = with_cycle(y, round > 0, &mut work)?;
        }
        if let Some(probed) = ended {
            on_end(round, value, probed);
            break;
        }
        orthonormalise_all(&mut y);
        (value, vector) = lowest(&y);
        on_round(round, &y, value);
    }
    Ok(y)
}

/// Improve the test vectors y with the multigrid built on them: every round, the basis and the
/// cycle (right form) from the current y, then every vector by `bootstrap_vector`.
pub fn pv_eo_bootstrap<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    agg: &Arc<Aggregation>,
    y: &mut HalfField<Aos, F>,
    params: BootstrapParams,
) -> Result<PvEoBasis<F>, MultigridError> {
    pv_eo_bootstrap_stored(
        op,
        pv,
        agg,
        y,
        params,
        BasisStorage::Half,
        (&[], &[], false),
    )
}

/// `pv_eo_bootstrap` with P kept as `storage` says, in its rounds too, and the cycles of the
/// rounds after the first with the levels below the coarse lattice `levels`.
pub fn pv_eo_bootstrap_stored<F: Storage, G: GammaRepre>(
    op: &Mdwf<F, G>,
    pv: &Mdwf<F, G>,
    agg: &Arc<Aggregation>,
    y: &mut HalfField<Aos, F>,
    params: BootstrapParams,
    storage: BasisStorage,
    (super_blocks, refine, for_pv): LevelLists<'_>,
) -> Result<PvEoBasis<F>, MultigridError> {
    let like = y.select(&[0]);
    // The vectors go through the rounds by value; one vector stands in for them meanwhile.
    let taken = std::mem::replace(y, like.zeros_like());
    *y = bootstrap_rounds::<_, InPlace<_>, _>(
        taken,
        params,
        &mut |y, with_levels, work| {
            let basis = pv_eo_basis_stored(op, pv, agg, &y, storage)?;
            let mut cycle = PvEoCycle::new(op, pv, (op, pv, EoForm::Right), &basis, params.cycle)?;
            if with_levels && !super_blocks.is_empty() {
                let v = pv_eo_coarse_vectors(op, pv, &basis, &y);
                cycle.add_levels_with(&v, super_blocks, refine)?;
                if for_pv {
                    if params.cycle.pv_solver != CoarseKrylov::Minres {
                        return Err(MultigridError::Backend("MultiGrid.levels_for_pauli_villars needs MINRES for the Pauli-Villars factor (Coarse_Solver_PV)".into()));
                    }
                    cycle.add_pv_levels(&v, super_blocks)?;
                }
            }
            let mut a = PvEvenOdd::new(op, pv, EoForm::Right, &like);
            let (waiting, _) = work(&mut a, &mut cycle, y)?;
            drop(cycle);
            waiting.back()
        },
        &mut |y| lowest_ritz_pair(&mut PvEvenOdd::new(op, pv, EoForm::Right, &like), y),
        &mut |_, _, _| {},
        &mut |_, _, _| {},
        &mut |_, _, _| {},
    )?;
    pv_eo_basis_stored(op, pv, agg, y, storage)
}
