use crate::comm::{opposite_face, FACES};
use crate::field::storage::Storage;
use crate::field::{Aos, Cpu, Field, Kind, Sites, CACHE_LINE};
use crate::lattice::stencil::{Direction, Source, Stencil};
use rayon::prelude::*;

/// Sites per partial sum in `reduce_sites`. Fixed, so results do not depend on thread count.
pub const REDUCE_CHUNK: usize = 1024;

/// How much a loop does at each site, which decides how few sites are worth a task.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Weight {
    /// Real work: a stencil, a product of matrices.
    Heavy,
    /// A few arithmetic operations per real, such as the updates of a vector.
    Light,
}

/// Tasks a loop aims at for every thread of the pool, so that threads that finish early
/// find work to take over.
const TASKS_PER_THREAD: usize = 4;
/// Fewest spinors (reals, divided by 24) in a task of a heavy loop and of a light one. A
/// loop whose sites hold several spinors, or a block of sites of the SIMD layout, takes
/// fewer sites in proportion.
const MIN_SPINORS_HEAVY: usize = 512;
const MIN_SPINORS_LIGHT: usize = 1024;
const SPINOR_REALS: usize = 24;

/// The value of `RUTTICE_MIN_SITES_PER_TASK`, if it is set.
fn fixed_task_sites() -> Option<usize> {
    static VALUE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("RUTTICE_MIN_SITES_PER_TASK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
    })
}

/// The smallest task a heavy loop over sites of one spinor may be cut into.
pub fn min_sites_per_task() -> usize {
    fixed_task_sites().unwrap_or(MIN_SPINORS_HEAVY)
}

/// Fewest sites handed to one task of the thread pool by a loop over `num_sites` sites of
/// `site_len` reals each. A loop over no more sites than this runs on the calling thread
/// without touching the pool.
///
/// The rule: a loop goes to the pool only if every thread can be given a task of the
/// smallest size, and is then cut into `TASKS_PER_THREAD` tasks per thread, none smaller
/// than that size. It replaces a fixed 1024 sites per task, which failed at both ends:
///
/// - Too large for a loop of middling length: the Wilson operator on blocks of four sites
///   on a 16^3 x 8 lattice was cut into eight tasks for twelve threads and took 1.9 ms,
///   against 1.0 ms with tasks of 128 or 16 sites; each part of a sweep in two parts
///   (latency hiding) got fewer tasks still.
/// - Too small is worse, and how small is too small depends on the number of threads. With
///   tasks of 72 spinors on 32 threads, the end-to-end tests on a 4x4x4x8 lattice with
///   twelve right-hand sides (6144 spinors per loop) took 62 s, against 4.7 s with those
///   loops on the calling thread. On twelve cores, loops of 8192 spinors gain from the
///   pool: keeping them on the calling thread made the operator on two processes four
///   times slower.
///
/// Measured on those two machines only. The environment variable
/// `RUTTICE_MIN_SITES_PER_TASK` replaces the rule by a fixed number of sites, for measuring.
fn task_sites(num_sites: usize, site_len: usize, weight: Weight) -> usize {
    let (scale, spinors) = match weight {
        Weight::Heavy => (1, MIN_SPINORS_HEAVY),
        Weight::Light => (8, MIN_SPINORS_LIGHT),
    };
    if let Some(fixed) = fixed_task_sites() {
        return scale * fixed;
    }
    // The smallest task, in sites of this loop.
    let smallest = (spinors * SPINOR_REALS).div_ceil(site_len.max(1));
    let threads = rayon::current_num_threads();
    if num_sites < smallest * threads {
        num_sites
    } else {
        (num_sites / (TASKS_PER_THREAD * threads)).max(smallest)
    }
}

/// Call `f(site, reals_of_that_site)` for every site of `out`, in parallel. For loops that
/// do real work at each site: a stencil, a product of matrices.
pub fn for_sites<T, F>(out: &mut [T], site_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let grain = task_sites(out.len() / site_len, site_len, Weight::Heavy);
    for_sites_from(out, 0, site_len, grain, &f, &|| ());
}

/// `for_sites` for loops that do little at each site, such as the updates of a vector:
/// their tasks hold more sites.
pub fn for_light_sites<T, F>(out: &mut [T], site_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let grain = task_sites(out.len() / site_len, site_len, Weight::Light);
    for_sites_from(out, 0, site_len, grain, &f, &|| ());
}

/// Smallest output, in bytes, that a sweep writes with stores that bypass the cache
/// (`Storage::stream`). Such a store only pays when the data would not have stayed in the
/// cache until it is read again; below this size an ordinary store is used. The
/// environment variable `RUTTICE_STREAM_MIN_BYTES` overrides it, for measuring.
pub fn stream_min_bytes() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("RUTTICE_STREAM_MIN_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(STREAM_MIN_BYTES)
    })
}

/// The default of `stream_min_bytes`: about the last-level cache of one processor of a
/// current server. Timing the Wilson operator alone, streaming was no slower at any size
/// tried, from one thread to a whole machine; the threshold is there for the solvers, which
/// read the result of the operator straight away. That case has not been measured.
const STREAM_MIN_BYTES: usize = 32 << 20;

/// Whether a sweep that fills `out` in pieces of `piece_len` values should use
/// `Storage::stream`: the output is large, and every piece covers whole cache lines.
pub fn streams<T>(out: &[T], piece_len: usize) -> bool {
    size_of_val(out) >= stream_min_bytes()
        && out.as_ptr().align_offset(CACHE_LINE) == 0
        && (piece_len * size_of::<T>()).is_multiple_of(CACHE_LINE)
}

/// `for_sites`, with `finish()` called on each thread after the last site of every run of
/// sites it was handed. For work that has to be closed once per task rather than once per
/// site, such as `stream_fence` after `Storage::stream`.
pub fn for_sites_then<T, F, E>(out: &mut [T], site_len: usize, f: F, finish: E)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
    E: Fn() + Sync,
{
    let grain = task_sites(out.len() / site_len, site_len, Weight::Heavy);
    for_sites_from(out, 0, site_len, grain, &f, &finish);
}

/// `for_sites_then` for a piece of a field: `out` holds the sites from `first` on.
fn for_sites_from<T, F, E>(
    out: &mut [T],
    first: usize,
    site_len: usize,
    grain: usize,
    f: &F,
    finish: &E,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
    E: Fn() + Sync,
{
    if out.len() <= site_len * grain {
        for (i, v) in out.chunks_mut(site_len).enumerate() {
            f(first + i, v);
        }
        finish();
    } else {
        // A fold runs over one run of sites on one thread and hands its result on from
        // that thread when the run is finished.
        out.par_chunks_mut(site_len)
            .with_min_len(grain)
            .enumerate()
            .fold(|| (), |(), (i, v)| f(first + i, v))
            .for_each(|()| finish());
    }
}

/// Some of the sites of a field, as runs of consecutive sites in increasing order. A sweep
/// over them (`for_site_runs`) visits those sites and no others; nothing is tested per
/// site, and the work is divided by the number of sites in the runs.
pub struct SiteRuns {
    /// (first site, number of sites) of each run.
    runs: Vec<(u32, u32)>,
    /// `before[i]`: sites in the runs before run `i`; one entry more than there are runs.
    before: Vec<u32>,
}

impl SiteRuns {
    /// The sites below `num_sites` for which `member` holds.
    pub fn collect(num_sites: usize, member: impl Fn(usize) -> bool) -> Self {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for site in (0..num_sites).filter(|site| member(*site)) {
            let site = u32::try_from(site).expect("site numbers fit in 32 bits");
            match runs.last_mut() {
                Some((first, len)) if *first + *len == site => *len += 1,
                _ => runs.push((site, 1)),
            }
        }
        let mut before = Vec::with_capacity(runs.len() + 1);
        let mut total = 0;
        before.push(total);
        for (_, len) in &runs {
            total += len;
            before.push(total);
        }
        Self { runs, before }
    }

    pub fn num_sites(&self) -> usize {
        self.before[self.runs.len()] as usize
    }
}

/// What every task of `for_site_runs` needs.
struct RunTask<'a, F, E> {
    site_len: usize,
    grain: usize,
    f: &'a F,
    finish: &'a E,
}

/// Call `f(site, reals_of_that_site)` for the sites of `runs`, in parallel, and `finish()`
/// as `for_sites_then` does. The other sites of `out` are not touched.
pub fn for_site_runs<T, F, E>(out: &mut [T], site_len: usize, runs: &SiteRuns, f: F, finish: E)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
    E: Fn() + Sync,
{
    let task = RunTask {
        site_len,
        grain: task_sites(runs.num_sites(), site_len, Weight::Heavy),
        f: &f,
        finish: &finish,
    };
    over_runs(out, 0, &runs.runs, &runs.before, &task);
}

/// `out` holds the sites from `first` on and contains all of `runs`; `before` are the
/// entries of `SiteRuns::before` for these runs and the one after them. Halves the runs by
/// the number of sites in them until a piece is small enough for one task.
fn over_runs<T, F, E>(
    out: &mut [T],
    first: usize,
    runs: &[(u32, u32)],
    before: &[u32],
    task: &RunTask<'_, F, E>,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
    E: Fn() + Sync,
{
    let n = task.site_len;
    // The first site of a run and where its sites are in `out`.
    let piece = |run: (u32, u32)| {
        let (start, len) = (run.0 as usize, run.1 as usize);
        (start, (start - first) * n..(start - first + len) * n)
    };
    match runs {
        [] => {}
        // One run: a piece of the field like any other.
        [run] => {
            let (start, range) = piece(*run);
            for_sites_from(&mut out[range], start, n, task.grain, task.f, task.finish);
        }
        _ if (before[runs.len()] - before[0]) as usize <= task.grain => {
            for run in runs {
                let (start, range) = piece(*run);
                for (i, v) in out[range].chunks_mut(n).enumerate() {
                    (task.f)(start + i, v);
                }
            }
            (task.finish)();
        }
        _ => {
            // The run in which half of the sites is reached starts the second half.
            let half = before[0] + (before[runs.len()] - before[0]) / 2;
            let m = before[..runs.len()]
                .partition_point(|b| *b < half)
                .clamp(1, runs.len() - 1);
            let second = runs[m].0 as usize;
            let (lo, hi) = out.split_at_mut((second - first) * n);
            rayon::join(
                || over_runs(lo, first, &runs[..m], &before[..=m], task),
                || over_runs(hi, second, &runs[m..], &before[m..], task),
            );
        }
    }
}

/// Sum `f(site)` over all sites.
///
/// Sites are summed in fixed-size chunks and the chunk sums are added in order, so the
/// result is the same on every run and for every thread count.
pub fn reduce_sites<F>(num_sites: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync,
{
    let [s] = reduce_sites_n(num_sites, |site| [f(site)]);
    s
}

/// `reduce_sites` for several sums at once, in one sweep.
pub fn reduce_sites_n<const N: usize, F>(num_sites: usize, f: F) -> [f64; N]
where
    F: Fn(usize) -> [f64; N] + Sync,
{
    let chunk_sum = |chunk: usize| {
        let lo = chunk * REDUCE_CHUNK;
        let hi = (lo + REDUCE_CHUNK).min(num_sites);
        let mut sum = [0.0; N];
        for site in lo..hi {
            let v = f(site);
            for (s, x) in sum.iter_mut().zip(v) {
                *s += x;
            }
        }
        sum
    };
    let chunks = num_sites.div_ceil(REDUCE_CHUNK);
    // The same partial sums either way, so the result does not depend on the path taken.
    let partial: Vec<[f64; N]> = if num_sites <= REDUCE_CHUNK {
        (0..chunks).map(chunk_sum).collect()
    } else {
        (0..chunks).into_par_iter().map(chunk_sum).collect()
    };
    let mut total = [0.0; N];
    for p in &partial {
        for (t, x) in total.iter_mut().zip(p) {
            *t += x;
        }
    }
    total
}

/// `reduce_sites` when the number of sums is only known at run time. `f(site, acc)` adds
/// the contribution of one site into `acc`, which has length `n`.
pub fn reduce_sites_dyn<F>(num_sites: usize, n: usize, f: F) -> Vec<f64>
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    let mut total = vec![0.0; n];
    if n == 0 || num_sites == 0 {
        return total;
    }
    let mut partial = vec![0.0; num_sites.div_ceil(REDUCE_CHUNK) * n];
    let chunk_sum = |(chunk, acc): (usize, &mut [f64])| {
        let lo = chunk * REDUCE_CHUNK;
        let hi = (lo + REDUCE_CHUNK).min(num_sites);
        for site in lo..hi {
            f(site, acc);
        }
    };
    // The same partial sums either way, so the result does not depend on the path taken.
    if num_sites <= REDUCE_CHUNK {
        partial.chunks_mut(n).enumerate().for_each(chunk_sum);
    } else {
        partial.par_chunks_mut(n).enumerate().for_each(chunk_sum);
    }
    for p in partial.chunks_exact(n) {
        for (t, x) in total.iter_mut().zip(p) {
            *t += x;
        }
    }
    total
}

/// `reduce_sites_dyn` for a sweep that also writes: `f(site, reals_of_that_site, acc)`
/// updates the site of `out` and adds its contribution to the `n` sums. The partial sums
/// are those of `reduce_sites_dyn`, so a sum made here has the bits of the same sum made
/// there in a sweep of its own.
pub fn reduce_sites_mut<T, F>(out: &mut [T], site_len: usize, n: usize, f: F) -> Vec<f64>
where
    T: Send,
    F: Fn(usize, &mut [T], &mut [f64]) + Sync,
{
    let num_sites = out.len() / site_len;
    let mut total = vec![0.0; n];
    if n == 0 || num_sites == 0 {
        return total;
    }
    let mut partial = vec![0.0; num_sites.div_ceil(REDUCE_CHUNK) * n];
    let chunk_sum = |(chunk, (sites, acc)): (usize, (&mut [T], &mut [f64]))| {
        for (i, v) in sites.chunks_mut(site_len).enumerate() {
            f(chunk * REDUCE_CHUNK + i, v, acc);
        }
    };
    let chunk_len = site_len * REDUCE_CHUNK;
    if num_sites <= REDUCE_CHUNK {
        out.chunks_mut(chunk_len)
            .zip(partial.chunks_mut(n))
            .enumerate()
            .for_each(chunk_sum);
    } else {
        out.par_chunks_mut(chunk_len)
            .zip(partial.par_chunks_mut(n))
            .enumerate()
            .for_each(chunk_sum);
    }
    for p in partial.chunks_exact(n) {
        for (t, x) in total.iter_mut().zip(p) {
            *t += x;
        }
    }
    total
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

    /// A sweep over runs visits the listed sites once each, with their own numbers and
    /// reals, leaves the others alone, and closes every task.
    #[test]
    fn runs_cover_exactly_their_sites() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (num_sites, site_len) = (50_000, 3);
        for member in [
            (|site| site % 7 == 3) as fn(usize) -> bool,
            |site| !(10_000..40_000).contains(&site),
            |site| (123..45_678).contains(&site),
            |_| false,
            |_| true,
        ] {
            let runs = SiteRuns::collect(num_sites, member);
            assert_eq!(
                runs.num_sites(),
                (0..num_sites).filter(|s| member(*s)).count()
            );
            let mut out = vec![0u64; num_sites * site_len];
            let closed = AtomicUsize::new(0);
            for_site_runs(
                &mut out,
                site_len,
                &runs,
                |site, v| {
                    assert_eq!(v.len(), site_len);
                    for x in v {
                        *x += 1 + site as u64;
                    }
                },
                || {
                    closed.fetch_add(1, Ordering::Relaxed);
                },
            );
            for (site, v) in out.chunks(site_len).enumerate() {
                let want = if member(site) { 1 + site as u64 } else { 0 };
                assert!(v.iter().all(|x| *x == want), "site {site}");
            }
            assert!(closed.load(Ordering::Relaxed) >= usize::from(runs.num_sites() > 0));
        }
    }
    use crate::field::Gauge;
    use crate::lattice::stencil::Boundary;
    use crate::lattice::{Lattice, ND};

    #[test]
    fn reduce_is_exactly_the_chunked_sum() {
        let n = 5 * REDUCE_CHUNK + 17;
        let f = |i: usize| 1.0 / (i as f64 + 1.0);
        let mut expect = 0.0;
        for c in 0..n.div_ceil(REDUCE_CHUNK) {
            let mut s = 0.0;
            for i in c * REDUCE_CHUNK..((c + 1) * REDUCE_CHUNK).min(n) {
                s += f(i);
            }
            expect += s;
        }
        for _ in 0..20 {
            assert_eq!(reduce_sites(n, f), expect);
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
