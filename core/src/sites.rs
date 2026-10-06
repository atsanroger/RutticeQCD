use crate::field::{Aos, Cpu, Field, Kind, Sites};
use crate::stencil::{Direction, Source, Stencil};
use crate::storage::Storage;
use rayon::prelude::*;

/// Sites per partial sum in `reduce_sites`. Fixed, so results do not depend on thread count.
pub const REDUCE_CHUNK: usize = 1024;

/// Call `f(site, reals_of_that_site)` for every site of `out`, in parallel.
pub fn for_sites<T, F>(out: &mut [T], site_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    out.par_chunks_mut(site_len)
        .enumerate()
        .for_each(|(site, v)| f(site, v));
}

/// Sum `f(site)` over all sites.
///
/// Sites are summed in fixed-size chunks and the chunk sums are added in order, so the
/// result is the same on every run and for every thread count (DESIGN.md §6.5).
pub fn reduce_sites<F>(num_sites: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync,
{
    let partial: Vec<f64> = (0..num_sites.div_ceil(REDUCE_CHUNK))
        .into_par_iter()
        .map(|chunk| {
            let lo = chunk * REDUCE_CHUNK;
            let hi = (lo + REDUCE_CHUNK).min(num_sites);
            (lo..hi).map(&f).sum::<f64>()
        })
        .collect();
    partial.iter().sum()
}

/// `out(x) = inp(x ± mu)` for every outer slab. Sites whose neighbour is `Zero` are set to
/// the default value. No boundary phase is applied.
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
    for o in 0..inp.outer() {
        for_sites(out.slab_mut(o), site_len, |site, v| {
            match st.source(site, mu, dir) {
                Source::Local { site: n, .. } => v.copy_from_slice(inp.site(o, n)),
                Source::Zero => v.fill(T::default()),
                Source::Buffer { .. } => unreachable!("no communication buffers on a single node"),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Gauge;
    use crate::lattice::{Lattice, ND};
    use crate::stencil::Boundary;

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
