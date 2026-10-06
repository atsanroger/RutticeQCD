//! The small least-squares problem inside GMRES.

use ruttice_kernel::Cplx;

type C64 = Cplx<f64>;

/// Minimises `|| beta e1 - H y ||` for an upper Hessenberg matrix H that grows by one
/// column per Arnoldi step, by Givens rotations. After `k` columns H is (k+1) × k.
pub struct HessenbergLsq {
    capacity: usize,
    /// Upper triangular factor, column by column: entry (i, k) at `k * capacity + i`.
    r: Vec<C64>,
    /// Rotation k is [[c, s], [-conj(s), c]] with real c.
    cos: Vec<f64>,
    sin: Vec<C64>,
    /// The rotated right-hand side; `|g[k]|` is the residual norm after k columns.
    g: Vec<C64>,
    len: usize,
}

impl HessenbergLsq {
    /// Room for `capacity` columns.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            r: vec![C64::ZERO; capacity * capacity],
            cos: vec![1.0; capacity],
            sin: vec![C64::ZERO; capacity],
            g: vec![C64::ZERO; capacity + 1],
            len: 0,
        }
    }

    /// Start a new problem with right-hand side `beta e1`.
    pub fn reset(&mut self, beta: f64) {
        self.len = 0;
        self.g.fill(C64::ZERO);
        self.g[0] = C64::new(beta, 0.0);
    }

    /// Number of columns so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Residual norm of the current least-squares solution.
    pub fn residual(&self) -> f64 {
        self.g[self.len].norm_sqr().sqrt()
    }

    /// Append column `k = len()` of H, given as its entries 0..=k+1 (`column` has length
    /// k + 2 and is used as scratch). Returns the new residual norm.
    pub fn push(&mut self, column: &mut [C64]) -> f64 {
        let k = self.len;
        assert!(k < self.capacity, "HessenbergLsq is full");
        assert_eq!(column.len(), k + 2);

        for i in 0..k {
            let (a, b) = (column[i], column[i + 1]);
            column[i] = a.scale(self.cos[i]) + self.sin[i] * b;
            column[i + 1] = b.scale(self.cos[i]) - self.sin[i].conj() * a;
        }

        // The rotation that zeroes column[k + 1].
        let (a, b) = (column[k], column[k + 1]);
        let (abs_a, t) = (a.norm_sqr().sqrt(), (a.norm_sqr() + b.norm_sqr()).sqrt());
        let (c, s, diagonal) = if t == 0.0 {
            (1.0, C64::ZERO, C64::ZERO)
        } else if abs_a == 0.0 {
            (0.0, b.conj().scale(1.0 / t), C64::new(t, 0.0))
        } else {
            let phase = a.scale(1.0 / abs_a);
            (abs_a / t, phase * b.conj().scale(1.0 / t), phase.scale(t))
        };
        self.cos[k] = c;
        self.sin[k] = s;
        column[k] = diagonal;

        self.r[k * self.capacity..k * self.capacity + k + 1].copy_from_slice(&column[..k + 1]);
        let gk = self.g[k];
        self.g[k] = gk.scale(c);
        self.g[k + 1] = -(s.conj() * gk);
        self.len = k + 1;
        self.residual()
    }

    /// The minimiser `y` (length `len()`). A direction whose diagonal entry vanished gets
    /// coefficient zero.
    pub fn solve(&self, y: &mut [C64]) {
        let k = self.len;
        assert_eq!(y.len(), k);
        for i in (0..k).rev() {
            let mut s = self.g[i];
            for j in i + 1..k {
                s = s - self.r[j * self.capacity + i] * y[j];
            }
            let d = self.r[i * self.capacity + i];
            y[i] = if d.norm_sqr() > 0.0 { s / d } else { C64::ZERO };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A (k+1) × k upper Hessenberg matrix with arbitrary entries.
    fn hessenberg(k: usize) -> Vec<Vec<C64>> {
        (0..k)
            .map(|col| {
                (0..col + 2)
                    .map(|row| {
                        let x = (row * 7 + col * 13 + 3) as f64;
                        C64::new(
                            (x * 0.37).sin() + if row == col { 2.0 } else { 0.0 },
                            (x * 0.61).cos(),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn solution_satisfies_the_normal_equations() {
        let (k, beta) = (6, 1.7);
        let h = hessenberg(k);
        let mut lsq = HessenbergLsq::new(k);
        lsq.reset(beta);
        let mut reported = 0.0;
        for col in &h {
            reported = lsq.push(&mut col.clone());
        }
        let mut y = vec![C64::ZERO; k];
        lsq.solve(&mut y);

        // rho = beta e1 - H y
        let mut rho = vec![C64::ZERO; k + 1];
        rho[0] = C64::new(beta, 0.0);
        for (col, yc) in h.iter().zip(&y) {
            for (row, hrc) in col.iter().enumerate() {
                rho[row] = rho[row] - *hrc * *yc;
            }
        }
        let norm = rho.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - reported).abs() < 1e-13, "{norm} vs {reported}");
        assert!(
            norm > 1e-3 && norm < beta,
            "a genuine least-squares residual"
        );
        // H^† rho = 0
        for col in &h {
            let mut s = C64::ZERO;
            for (row, hrc) in col.iter().enumerate() {
                s += hrc.conj() * rho[row];
            }
            assert!(s.norm_sqr().sqrt() < 1e-13);
        }
    }

    #[test]
    fn residual_never_increases_and_reset_starts_over() {
        let h = hessenberg(5);
        let mut lsq = HessenbergLsq::new(5);
        lsq.reset(3.0);
        assert_eq!(lsq.residual(), 3.0);
        let mut last = 3.0;
        for col in &h {
            let r = lsq.push(&mut col.clone());
            assert!(r <= last + 1e-15);
            last = r;
        }
        lsq.reset(1.0);
        assert!(lsq.is_empty());
        assert_eq!(lsq.residual(), 1.0);
    }

    #[test]
    fn zero_columns_do_not_divide_by_zero() {
        let mut lsq = HessenbergLsq::new(3);
        lsq.reset(0.0);
        for k in 0..3 {
            assert_eq!(lsq.push(&mut vec![C64::ZERO; k + 2]), 0.0);
        }
        let mut y = [C64::new(1.0, 1.0); 3];
        lsq.solve(&mut y);
        assert_eq!(y, [C64::ZERO; 3]);
    }
}
