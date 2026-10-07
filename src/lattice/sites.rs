use crate::comm::{opposite_face, FACES};
use crate::field::storage::Storage;
use crate::field::{Aos, Cpu, Field, Kind, Sites, CACHE_LINE};
use crate::lattice::stencil::{Direction, Source, Stencil};
use rayon::prelude::*;

/// Sites per partial sum in `reduce_sites`. Fixed, so results do not depend on thread count.
pub const REDUCE_CHUNK: usize = 1024;

/// Fewest sites handed to one task of the thread pool. Loops over no more sites than this
/// run on the calling thread without touching the pool.
///
/// Handing work to the pool has a cost per call that does not depend on the amount of
/// work, so very small loops are slower in parallel than serially: with one site per task
/// and 32 threads, the gauge-fixing tests on a 4x4x4x8 lattice took 207 s, against 11 s
/// with the loops kept on the calling thread. The default is the chunk of the reductions;
/// it has not been tuned on large lattices. The environment variable
/// `RUTTICE_MIN_SITES_PER_TASK` overrides it, for measuring.
pub fn min_sites_per_task() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("RUTTICE_MIN_SITES_PER_TASK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(REDUCE_CHUNK)
    })
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

/// Call `f(site, reals_of_that_site)` for every site of `out`, in parallel.
pub fn for_sites<T, F>(out: &mut [T], site_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let grain = min_sites_per_task();
    if out.len() <= site_len * grain {
        for (site, v) in out.chunks_mut(site_len).enumerate() {
            f(site, v);
        }
    } else {
        out.par_chunks_mut(site_len)
            .with_min_len(grain)
            .enumerate()
            .for_each(|(site, v)| f(site, v));
    }
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
    let grain = min_sites_per_task();
    if out.len() <= site_len * grain {
        for (site, v) in out.chunks_mut(site_len).enumerate() {
            f(site, v);
        }
        finish();
    } else {
        // A fold runs over one run of sites on one thread and hands its result on from
        // that thread when the run is finished.
        out.par_chunks_mut(site_len)
            .with_min_len(grain)
            .enumerate()
            .fold(|| (), |(), (site, v)| f(site, v))
            .for_each(|()| finish());
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
    let partial: Vec<[f64; N]> = if num_sites <= min_sites_per_task() {
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
    if num_sites <= min_sites_per_task() {
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
