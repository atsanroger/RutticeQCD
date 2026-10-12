//! Whole-field reductions and fills. The per-vector `Vector` interface
//! is built on these when the solvers arrive.

use crate::field::storage::Storage;
use crate::field::{Aos, Cpu, Field, Kind, Sites};
use crate::kernel::Real;
use crate::lattice::for_sites::reduce_sites_n;

/// Sum of squares of every real in the field, over all processes.
pub fn norm_sqr<T, K, S>(a: &Field<T, K, S, Aos, Cpu>) -> f64
where
    T: Storage,
    K: Kind,
    S: Sites,
{
    let [s] = reduce_sites_n(a.num_sites(), |site| {
        let mut s = 0.0;
        for o in 0..a.outer() {
            for x in a.site(o, site) {
                let x = x.load().to_f64();
                s += x * x;
            }
        }
        [s]
    });
    let mut s = [s];
    a.lattice().comm.sum(&mut s);
    s[0]
}

/// `<a|b> = Σ conj(a) b`, treating consecutive reals as (re, im). Returns (re, im).
pub fn dot<T, K, S>(a: &Field<T, K, S, Aos, Cpu>, b: &Field<T, K, S, Aos, Cpu>) -> (f64, f64)
where
    T: Storage,
    K: Kind,
    S: Sites,
{
    assert_eq!(a.as_slice().len(), b.as_slice().len());
    assert_eq!(a.site_len(), b.site_len());
    let [re, im] = reduce_sites_n(a.num_sites(), |site| {
        let (mut re, mut im) = (0.0, 0.0);
        for o in 0..a.outer() {
            let (x, y) = (
                a.site(o, site).as_chunks::<2>().0,
                b.site(o, site).as_chunks::<2>().0,
            );
            for (x, y) in x.iter().zip(y) {
                let (ar, ai) = (x[0].load().to_f64(), x[1].load().to_f64());
                let (br, bi) = (y[0].load().to_f64(), y[1].load().to_f64());
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
            }
        }
        [re, im]
    });
    let mut s = [re, im];
    a.lattice().comm.sum(&mut s);
    (s[0], s[1])
}

/// Set every real to `value`.
pub fn fill<T, K, S>(a: &mut Field<T, K, S, Aos, Cpu>, value: T)
where
    T: Storage,
    K: Kind,
    S: Sites,
{
    a.as_mut_slice().fill(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Spinor;
    use crate::lattice::Lattice;

    #[test]
    fn norm_and_dot_agree_with_a_plain_loop() {
        let lat = Lattice::single_node([4, 2, 2, 4]);
        let mut a = Field::<f64, Spinor>::with_shape(&lat, 24, 1, 2, 1);
        let mut b = Field::<f64, Spinor>::with_shape(&lat, 24, 1, 2, 1);
        for (i, x) in a.as_mut_slice().iter_mut().enumerate() {
            *x = ((i * 13 + 5) % 17) as f64 - 8.0;
        }
        for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
            *x = ((i * 7 + 2) % 19) as f64 - 9.0;
        }
        let n: f64 = a.as_slice().iter().map(|x| x * x).sum();
        assert_eq!(norm_sqr(&a), n);
        assert_eq!(dot(&a, &a), (n, 0.0));

        let (mut re, mut im) = (0.0, 0.0);
        for (x, y) in a
            .as_slice()
            .as_chunks::<2>()
            .0
            .iter()
            .zip(b.as_slice().as_chunks::<2>().0)
        {
            re += x[0] * y[0] + x[1] * y[1];
            im += x[0] * y[1] - x[1] * y[0];
        }
        assert_eq!(dot(&a, &b), (re, im));
        let (re2, im2) = dot(&b, &a);
        assert_eq!((re2, im2), (re, -im));
    }
}
