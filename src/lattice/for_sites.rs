//! Loops over the sites of a field, in parallel: `for_sites` and its variants, the sums over
//! sites (`reduce_sites`), how many sites make a task, and when a loop writes its output
//! past the cache.

use crate::field::CACHE_LINE;
use crate::lattice::pool::{each_thread, schedule, static_threads, Schedule};
use rayon::prelude::*;

/// Sites per partial sum in `reduce_sites`. Fixed, so results do not depend on thread count.
pub(crate) const REDUCE_CHUNK: usize = 1024;

/// How much a loop does at each site, which decides how few sites are worth a task.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Weight {
    /// Real work: a stencil, a product of matrices.
    Heavy,
    /// A few arithmetic operations per real, such as the updates of a vector.
    Light,
}

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
        crate::env::parsed::<std::num::NonZeroUsize>(
            "RUTTICE_MIN_SITES_PER_TASK",
            "a number of sites above zero",
        )
        .map(usize::from)
    })
}

/// The smallest task a heavy loop over sites of one spinor may be cut into.
pub fn min_sites_per_task() -> usize {
    fixed_task_sites().unwrap_or(MIN_SPINORS_HEAVY)
}

/// The number of sites up to which a loop over `num_sites` sites of `site_len` reals stays
/// on the calling thread; a longer loop is dealt out to the threads (`for_sites_from`).
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
    if num_sites < 2 * smallest {
        num_sites
    } else {
        2 * smallest - 1
    }
}

/// Call `f(site, reals_of_that_site)` for every site of `out`, in parallel. For loops that
/// do real work at each site: a stencil, a product of matrices.
pub(crate) fn for_sites<T, F>(out: &mut [T], site_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let grain = task_sites(out.len() / site_len, site_len, Weight::Heavy);
    for_sites_from(out, 0, site_len, grain, &f, &|| ());
}

/// `for_sites` for a loop whose work per site is not what `site_len` suggests: it is
/// counted as that of `work_len` reals. Filling a face buffer writes half a spinor per site
/// but projects a whole one and multiplies it by a link.
pub(crate) fn for_sites_as<T, F>(out: &mut [T], site_len: usize, work_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let grain = task_sites(out.len() / site_len, work_len, Weight::Heavy);
    for_sites_from(out, 0, site_len, grain, &f, &|| ());
}

/// `for_sites` for loops that do little at each site, such as the updates of a vector:
/// their tasks hold more sites.
pub(crate) fn for_light_sites<T, F>(out: &mut [T], site_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let grain = task_sites(out.len() / site_len, site_len, Weight::Light);
    for_sites_from(out, 0, site_len, grain, &f, &|| ());
}

/// Smallest output, in bytes, that a sweep writes with stores that bypass the cache
/// (`Storage::stream`).
pub fn stream_min_bytes() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        crate::env::parsed("RUTTICE_STREAM_MIN_BYTES", "a number of bytes")
            .unwrap_or(STREAM_MIN_BYTES)
    })
}

/// The default of `stream_min_bytes`: about the last-level cache of one processor of a current
/// server.
const STREAM_MIN_BYTES: usize = 32 << 20;

/// Whether a sweep that fills `out` in pieces of `piece_len` values should use
/// `Storage::stream`: the output is large, and every piece covers whole cache lines.
pub(crate) fn streams<T>(out: &[T], piece_len: usize) -> bool {
    size_of_val(out) >= stream_min_bytes()
        && out.as_ptr().align_offset(CACHE_LINE) == 0
        && (piece_len * size_of::<T>()).is_multiple_of(CACHE_LINE)
}

/// `for_sites`, with `finish()` called on each thread after the last site of every run of
/// sites it was handed. For work that has to be closed once per task rather than once per
/// site, such as `stream_fence` after `Storage::stream`.
pub(crate) fn for_sites_then<T, F, E>(out: &mut [T], site_len: usize, f: F, finish: E)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
    E: Fn() + Sync,
{
    let grain = task_sites(out.len() / site_len, site_len, Weight::Heavy);
    for_sites_from(out, 0, site_len, grain, &f, &finish);
}

/// One piece of `out` per thread, for the static schedules: thread `k` gets the units from `cut(k)`
/// to `cut(k + 1)`, as (first unit, piece).
fn pieces<T>(
    out: &mut [T],
    unit: usize,
    threads: usize,
    cut: impl Fn(usize) -> usize,
) -> Vec<(usize, &mut [T])> {
    let mut pieces = Vec::with_capacity(threads);
    let (mut rest, mut at) = (out, 0);
    for k in 0..threads {
        let end = cut(k + 1).max(at);
        let (piece, after) = rest.split_at_mut(((end - at) * unit).min(rest.len()));
        pieces.push((at, piece));
        (rest, at) = (after, end);
    }
    pieces
}

/// `for_sites_then` for a piece of a field: `out` holds the sites from `first` on. A loop
/// of no more than `grain` sites runs on the calling thread.
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
    let num_sites = out.len() / site_len;
    let serial = |from: usize, piece: &mut [T]| {
        for (i, v) in piece.chunks_mut(site_len).enumerate() {
            f(first + from + i, v);
        }
        finish();
    };
    if num_sites <= grain {
        return serial(0, out);
    }
    match static_threads() {
        Some(threads) => each_thread(
            pieces(out, site_len, threads, |k| k * num_sites / threads),
            |_, (from, piece)| {
                if !piece.is_empty() {
                    serial(from, piece);
                }
            },
        ),
        // Tasks that the threads take from each other, none smaller than half of what
        // is worth the pool. A fold runs over one run of sites on one thread and hands
        // its result on from that thread when the run is finished.
        None if schedule() == Schedule::Steal => out
            .par_chunks_mut(site_len)
            .with_min_len(grain.div_ceil(2))
            .enumerate()
            .fold(|| (), |(), (i, v)| f(first + i, v))
            .for_each(|()| finish()),
        None => serial(0, out),
    }
}

/// Some of the sites of a field, as runs of consecutive sites in increasing order. A sweep
/// over them (`for_site_runs`) visits those sites and no others; nothing is tested per
/// site, and the work is divided by the number of sites in the runs.
pub(crate) struct SiteRuns {
    /// (first site, number of sites) of each run.
    runs: Vec<(u32, u32)>,
    /// `before[i]`: sites in the runs before run `i`; one entry more than there are runs.
    before: Vec<u32>,
    /// The runs dealt out to the threads of the pool, made on first use.
    plan: std::sync::OnceLock<Plan>,
}

/// The runs of a `SiteRuns` cut so that every thread gets the same number of sites.
struct Plan {
    threads: usize,
    /// The runs, with those that straddle two threads cut in two.
    runs: Vec<(u32, u32)>,
    /// `first_run[k]`: the first run of thread `k`; one entry more than there are threads.
    first_run: Vec<usize>,
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
        Self {
            runs,
            before,
            plan: std::sync::OnceLock::new(),
        }
    }

    fn plan(&self, threads: usize) -> Plan {
        let total = self.num_sites();
        let (mut runs, mut first_run) = (Vec::with_capacity(self.runs.len() + threads), vec![0]);
        let mut done = 0;
        for &(mut start, mut len) in &self.runs {
            while len > 0 {
                // Sites still to go to the thread being filled.
                let quota = first_run.len() * total / threads - done;
                if quota == 0 {
                    first_run.push(runs.len());
                    continue;
                }
                let take = len.min(u32::try_from(quota).unwrap_or(u32::MAX));
                runs.push((start, take));
                (start, len, done) = (start + take, len - take, done + take as usize);
            }
        }
        first_run.resize(threads + 1, runs.len());
        Plan {
            threads,
            runs,
            first_run,
        }
    }

    pub fn num_sites(&self) -> usize {
        self.before[self.runs.len()] as usize
    }
}

/// Call `f(site, reals_of_that_site)` for the sites of `runs`, in parallel, and `finish()`
/// as `for_sites_then` does. The other sites of `out` are not touched.
pub(crate) fn for_site_runs<T, F, E>(
    out: &mut [T],
    site_len: usize,
    runs: &SiteRuns,
    f: F,
    finish: E,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
    E: Fn() + Sync,
{
    // The runs `some`, all of them inside `piece`, which holds the sites from `first` on.
    let sweep = |some: &[(u32, u32)], first: usize, piece: &mut [T]| {
        for &(start, len) in some {
            let (start, len) = (start as usize, len as usize);
            let sites = &mut piece[(start - first) * site_len..(start - first + len) * site_len];
            for (i, v) in sites.chunks_mut(site_len).enumerate() {
                f(start + i, v);
            }
        }
        finish();
    };
    let grain = task_sites(runs.num_sites(), site_len, Weight::Heavy);
    if runs.num_sites() <= grain {
        return sweep(&runs.runs, 0, out);
    }
    match static_threads() {
        // Every thread gets the same number of sites.
        Some(threads) => {
            let plan = runs.plan.get_or_init(|| runs.plan(threads));
            assert_eq!(plan.threads, threads, "the pool has changed its size");
            let end = out.len() / site_len;
            // A thread's piece of the field begins at its first run.
            let cut = |k: usize| match plan.runs.get(plan.first_run[k]) {
                Some((start, _)) if k < threads => *start as usize,
                _ => end,
            };
            each_thread(pieces(out, site_len, threads, cut), |k, (first, piece)| {
                let mine = &plan.runs[plan.first_run[k]..plan.first_run[k + 1]];
                if !mine.is_empty() {
                    sweep(mine, first, piece);
                }
            });
        }
        None if schedule() == Schedule::Steal => {
            let task = RunTask {
                site_len,
                grain,
                f: &f,
                finish: &finish,
            };
            over_runs(out, 0, &runs.runs, &runs.before, &task);
        }
        None => sweep(&runs.runs, 0, out),
    }
}

/// What every task of the work-stealing sweep over runs needs.
struct RunTask<'a, F, E> {
    site_len: usize,
    grain: usize,
    f: &'a F,
    finish: &'a E,
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
        _ if (before[runs.len()] - before[0]) as usize <= task.grain.div_ceil(2) => {
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
pub fn reduce_sites<F>(num_sites: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync,
{
    let [s] = reduce_sites_n(num_sites, |site| [f(site)]);
    s
}

/// `reduce_sites` for several sums at once, in one sweep.
pub(crate) fn reduce_sites_n<const N: usize, F>(num_sites: usize, f: F) -> [f64; N]
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
    } else if let Some(threads) = static_threads() {
        let mut partial = vec![[0.0; N]; chunks];
        each_thread(
            pieces(&mut partial, 1, threads, |k| k * chunks / threads),
            |_, (chunk, sums)| {
                for (i, sum) in sums.iter_mut().enumerate() {
                    *sum = chunk_sum(chunk + i);
                }
            },
        );
        partial
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
pub(crate) fn reduce_sites_dyn<F>(num_sites: usize, n: usize, f: F) -> Vec<f64>
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
    let chunks = partial.len() / n;
    if num_sites <= REDUCE_CHUNK {
        partial.chunks_mut(n).enumerate().for_each(chunk_sum);
    } else if let Some(threads) = static_threads() {
        each_thread(
            pieces(&mut partial, n, threads, |k| k * chunks / threads),
            |_, (chunk, sums)| {
                for (i, acc) in sums.chunks_mut(n).enumerate() {
                    chunk_sum((chunk + i, acc));
                }
            },
        );
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

/// `reduce_sites_dyn` for a sweep that also writes: `f(site, reals_of_that_site, acc)` updates the
/// site of `out` and adds its contribution to the `n` sums.
pub(crate) fn reduce_sites_mut<T, F>(out: &mut [T], site_len: usize, n: usize, f: F) -> Vec<f64>
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
    let chunks = partial.len() / n;
    if num_sites <= REDUCE_CHUNK {
        out.chunks_mut(chunk_len)
            .zip(partial.chunks_mut(n))
            .enumerate()
            .for_each(chunk_sum);
    } else if let Some(threads) = static_threads() {
        let cut = |k: usize| k * chunks / threads;
        let both: Vec<_> = pieces(out, chunk_len, threads, cut)
            .into_iter()
            .zip(pieces(&mut partial, n, threads, cut))
            .collect();
        each_thread(both, |_, ((chunk, sites), (_, sums))| {
            for (i, pair) in sites
                .chunks_mut(chunk_len)
                .zip(sums.chunks_mut(n))
                .enumerate()
            {
                chunk_sum((chunk + i, pair));
            }
        });
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
}
