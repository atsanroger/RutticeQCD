//! Rational approximations of x^(p/q) for RHMC: r(x) = a0 + Σ_k res_k / (x + pole_k) on
//! [x_min, x_max] with the smallest largest relative error |r(x)/x^(p/q) − 1| (minimax), the
//! approximation AlgRemez computes and Bridge++ reads from its files
//! `parameter_rational.<p>_<q>[inv]_<poles>_<x_min>_<x_max>` (`Math_Rational`).

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::path::Path;

/// A double-double number hi + lo, |lo| ≤ ulp(hi)/2: about 32 significant digits from pairs
/// of doubles (Dekker, Knuth; the algorithms of the QD library).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Dd {
    hi: f64,
    lo: f64,
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

impl Dd {
    fn new(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    fn from_pair((hi, lo): (f64, f64)) -> Self {
        Self { hi, lo }
    }

    fn powi(self, n: u32) -> Self {
        let mut r = Dd::new(1.0);
        for _ in 0..n {
            r = r * self;
        }
        r
    }

    fn abs(self) -> Self {
        if self.hi < 0.0 {
            -self
        } else {
            self
        }
    }
}

impl From<f64> for Dd {
    fn from(x: f64) -> Self {
        Dd::new(x)
    }
}

impl Add for Dd {
    type Output = Dd;
    fn add(self, b: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, b.hi);
        let (t, f) = two_sum(self.lo, b.lo);
        let (s, e) = quick_two_sum(s, e + t);
        Dd::from_pair(quick_two_sum(s, e + f))
    }
}

impl Neg for Dd {
    type Output = Dd;
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for Dd {
    type Output = Dd;
    fn sub(self, b: Dd) -> Dd {
        self + (-b)
    }
}

impl Mul for Dd {
    type Output = Dd;
    fn mul(self, b: Dd) -> Dd {
        let p = self.hi * b.hi;
        let e = self.hi.mul_add(b.hi, -p);
        Dd::from_pair(quick_two_sum(p, e + (self.hi * b.lo + self.lo * b.hi)))
    }
}

impl Div for Dd {
    type Output = Dd;
    fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        let r = self - b * Dd::new(q1);
        let q2 = r.hi / b.hi;
        let r = r - b * Dd::new(q2);
        let q3 = r.hi / b.hi;
        Dd::from_pair(quick_two_sum(q1, q2)) + Dd::new(q3)
    }
}

impl Dd {
    /// e^a: a = k ln 2 + r, e^r = (e^(r/1024))^1024 with e^(r/1024) by its Taylor series
    /// (|r/1024| < 3.4e-4: 12 terms are beyond 32 digits).
    fn exp(self) -> Dd {
        const LN2: Dd = Dd {
            hi: std::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        };
        let k = (self.hi / LN2.hi).round();
        let s = (self - LN2 * Dd::new(k)) * Dd::new(1.0 / 1024.0);
        let mut term = Dd::new(1.0);
        let mut sum = Dd::new(1.0);
        for i in 1..=12 {
            term = term * s / Dd::new(i as f64);
            sum = sum + term;
        }
        for _ in 0..10 {
            sum = sum * sum;
        }
        let scale = 2.0_f64.powi(k as i32);
        Dd {
            hi: sum.hi * scale,
            lo: sum.lo * scale,
        }
    }

    /// ln x by Newton's method on e^y = x from the double-precision logarithm.
    fn ln(x: f64) -> Dd {
        let mut y = Dd::new(x.ln());
        for _ in 0..2 {
            y = y + Dd::new(x) * (-y).exp() - Dd::new(1.0);
        }
        y
    }
}

/// The function approximated: x^(p/q), or x^β for the intermediate powers of the
/// continuation (by e^(β ln x)), both to double-double precision.
#[derive(Clone, Copy, Debug)]
enum Power {
    Exact { p: i32, q: i32 },
    Real(f64),
}

impl Power {
    fn eval(self, x: f64) -> Dd {
        match self {
            Power::Real(b) => (Dd::new(b) * Dd::ln(x)).exp(),
            Power::Exact { p, q } => {
                // y = x^(1/q) by Newton on y^q = x from the double-precision root.
                let q_dd = Dd::new(q as f64);
                let mut y = Dd::new(x.powf(1.0 / q as f64));
                for _ in 0..2 {
                    let yq1 = y.powi(q as u32 - 1);
                    y = y - (yq1 * y - Dd::new(x)) / (yq1 * q_dd);
                }
                let yp = y.powi(p.unsigned_abs());
                if p < 0 {
                    Dd::new(1.0) / yp
                } else {
                    yp
                }
            }
        }
    }

    fn exponent(self) -> f64 {
        match self {
            Power::Real(b) => b,
            Power::Exact { p, q } => p as f64 / q as f64,
        }
    }
}

/// Which approximation: x^(p/q) with `poles` poles on [x_min, x_max].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalSpec {
    pub poles: usize,
    pub p: i32,
    pub q: i32,
    pub x_min: f64,
    pub x_max: f64,
}

impl RationalSpec {
    /// Bridge++'s file name for the approximation; there x_min and x_max are the range of
    /// D†D, the squares of the parameters `lower_bound` and `upper_bound`.
    pub fn bridge_file_name(&self) -> String {
        format!(
            "parameter_rational.{}_{}{}_{:02}_{:010.8}_{:07.3}",
            self.p.abs(),
            self.q,
            if self.p < 0 { "inv" } else { "" },
            self.poles,
            self.x_min,
            self.x_max
        )
    }

    /// The same number of poles and interval for the power p/q.
    pub fn with_power(&self, p: i32, q: i32) -> Self {
        Self { p, q, ..*self }
    }
}

/// a0 + Σ_k res_k / (x + pole_k) with its largest relative error on [x_min, x_max].
#[derive(Clone, Debug, PartialEq)]
pub struct RationalApprox {
    pub norm: f64,
    pub(crate) residues: Vec<f64>,
    pub poles: Vec<f64>,
    /// max |r(x)/f(x) − 1| on the interval.
    pub error: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalError(pub String);

impl fmt::Display for RationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RationalError {}

fn fail<T>(m: impl Into<String>) -> Result<T, RationalError> {
    Err(RationalError(m.into()))
}

impl RationalApprox {
    /// r(x).
    pub fn eval(&self, x: f64) -> f64 {
        self.norm
            + self
                .residues
                .iter()
                .zip(&self.poles)
                .map(|(r, p)| r / (x + p))
                .sum::<f64>()
    }

    /// The minimax approximation `spec` (0 < x_min < x_max, 0 < |p/q| < 1).
    pub fn minimax(spec: &RationalSpec) -> Result<Self, RationalError> {
        let RationalSpec {
            poles,
            p,
            q,
            x_min,
            x_max,
        } = *spec;
        if poles == 0 || q <= 0 || p == 0 || p.abs() >= q {
            return fail(format!(
                "a rational approximation of x^({p}/{q}) with {poles} poles: need poles > 0 and 0 < |p/q| < 1"
            ));
        }
        if !(x_min > 0.0 && x_max > x_min && x_max.is_finite()) {
            return fail(format!(
                "the interval [{x_min}, {x_max}]: need 0 < x_min < x_max"
            ));
        }
        let target = Power::Exact { p, q };
        let beta = target.exponent();
        // The negative power −|β| by continuation from −1/2.
        let mut r = Remez::zolotarev(poles, x_min, x_max);
        r.solve(Power::Exact { p: -1, q: 2 })?;
        let mut at = -0.5;
        let goal = -beta.abs();
        let mut step = 1.0 / 32.0;
        while at != goal {
            let next = if (goal - at).abs() <= step {
                goal
            } else {
                at + step * (goal - at).signum()
            };
            let mut trial = r.clone();
            match trial.solve(Power::Real(next)) {
                Ok(()) => {
                    r = trial;
                    at = next;
                }
                Err(e) if step < 1e-4 => return Err(e),
                Err(_) => step /= 2.0,
            }
        }
        if beta > 0.0 {
            r = r.reciprocal()?;
        }
        r.solve(target)?;
        Ok(r.result())
    }

    /// The approximation in the format of Bridge++'s files: the number of poles, p, q; x_min,
    /// x_max; the error; a0; then res_k, pole_k per line. Returns it with what it is.
    pub(crate) fn parse_bridge(text: &str) -> Result<(Self, RationalSpec), RationalError> {
        let mut words = text.split_whitespace();
        let mut next = |what: &str| {
            words
                .next()
                .ok_or_else(|| RationalError(format!("the file ends before {what}")))
        };
        let int = |s: &str| {
            s.parse::<i32>()
                .map_err(|_| RationalError(format!("not an integer: {s}")))
        };
        let real = |s: &str| {
            s.parse::<f64>()
                .map_err(|_| RationalError(format!("not a number: {s}")))
        };
        let n = int(next("the number of poles")?)?;
        let p = int(next("the numerator")?)?;
        let q = int(next("the denominator")?)?;
        let x_min = real(next("x_min")?)?;
        let x_max = real(next("x_max")?)?;
        let error = real(next("the error")?)?;
        let norm = real(next("a0")?)?;
        if n <= 0 {
            return fail(format!("{n} poles"));
        }
        let mut residues = Vec::new();
        let mut poles = Vec::new();
        for _ in 0..n {
            residues.push(real(next("a residue")?)?);
            poles.push(real(next("a pole")?)?);
        }
        // The multi-shift solve stops on the residual of the first pole (`MultiShiftCg`).
        if poles.iter().any(|pole| *pole < poles[0]) {
            return fail("the first pole must be the smallest");
        }
        Ok((
            Self {
                norm,
                residues,
                poles,
                error,
            },
            RationalSpec {
                poles: n as usize,
                p,
                q,
                x_min,
                x_max,
            },
        ))
    }

    /// `parse_bridge` of a file, checked against what it should hold (the bounds to 1e-8,
    /// as Bridge++ asserts them to sqrt(ε)).
    pub fn read_bridge(path: &Path, spec: &RationalSpec) -> Result<Self, RationalError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RationalError(format!("{}: {e}", path.display())))?;
        let (r, got) = Self::parse_bridge(&text)?;
        let close = |a: f64, b: f64| ((a - b) / b).abs() < 1e-8;
        if got.poles != spec.poles
            || got.p != spec.p
            || got.q != spec.q
            || !close(got.x_min, spec.x_min)
            || !close(got.x_max, spec.x_max)
        {
            return fail(format!("{}: holds {got:?}, not {spec:?}", path.display()));
        }
        Ok(r)
    }

    /// The approximation as Bridge++ writes it, 17 significant digits.
    pub fn to_bridge(&self, spec: &RationalSpec) -> String {
        let mut s = format!(
            "{:6}{:6}{:6}\n  {:.16e}  {:.16e}\n  {:.6e}\n  {:.16e}\n",
            self.poles.len(),
            spec.p,
            spec.q,
            spec.x_min,
            spec.x_max,
            self.error,
            self.norm
        );
        for (r, p) in self.residues.iter().zip(&self.poles) {
            s += &format!("  {r:.16e}  {p:.16e}\n");
        }
        s
    }
}

/// The state of the Remez iteration: the approximation in double-double, the reference and
/// the level E of the error on it.
#[derive(Clone, Debug)]
struct Remez {
    a: f64,
    b: f64,
    norm: Dd,
    res: Vec<Dd>,
    pole: Vec<Dd>,
    level: Dd,
    reference: Vec<f64>,
    power: Power,
}

/// sn, cn of u for the parameter m = k² (the descending Landen transformation by the
/// arithmetic-geometric mean, as GSL and Bridge++'s `Math_Sign_Zolotarev` do it).
fn jacobi_sn_cn(u: f64, m: f64) -> (f64, f64) {
    let mut mu = vec![1.0];
    let mut nu = vec![(1.0 - m).sqrt()];
    while (mu[mu.len() - 1] - nu[nu.len() - 1]).abs()
        > f64::EPSILON * (mu[mu.len() - 1] + nu[nu.len() - 1])
        && mu.len() < 30
    {
        let (a, b) = (mu[mu.len() - 1], nu[nu.len() - 1]);
        mu.push(0.5 * (a + b));
        nu.push((a * b).sqrt());
    }
    let mut n = mu.len() - 1;
    let (s, c) = (u * mu[n]).sin_cos();
    if s.abs() < c.abs() {
        let mut cs = mu[n] * s / c;
        let mut dn = 1.0;
        while n > 0 {
            n -= 1;
            let r = cs * cs / mu[n + 1];
            cs *= dn;
            dn = (r + nu[n]) / (r + mu[n]);
        }
        let k1 = (1.0 - m).sqrt();
        let dn = k1 / dn;
        let cn = dn * c.signum() / (1.0 + cs * cs).sqrt();
        (cn * cs / k1, cn)
    } else {
        let mut cs = mu[n] * c / s;
        let mut dn = 1.0;
        while n > 0 {
            n -= 1;
            let r = cs * cs / mu[n + 1];
            cs *= dn;
            dn = (r + nu[n]) / (r + mu[n]);
        }
        let sn = s.signum() / (1.0 + cs * cs).sqrt();
        (sn, cs * sn)
    }
}

/// The complete elliptic integral K(m) = π / (2 AGM(1, √(1 − m))).
fn elliptic_k(m: f64) -> f64 {
    let (mut a, mut b) = (1.0_f64, (1.0 - m).sqrt());
    while (a - b).abs() > f64::EPSILON * a {
        (a, b) = (0.5 * (a + b), (a * b).sqrt());
    }
    std::f64::consts::FRAC_PI_2 / a
}

/// Solve the dense system A x = y (row-major n × n) by Gaussian elimination with partial
/// pivoting; None if a pivot vanishes.
fn solve_dense(n: usize, mut a: Vec<f64>, mut y: Vec<f64>) -> Option<Vec<f64>> {
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))?;
        if a[p * n + c] == 0.0 || !a[p * n + c].is_finite() {
            return None;
        }
        if p != c {
            for k in 0..n {
                a.swap(p * n + k, c * n + k);
            }
            y.swap(p, c);
        }
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    a[r * n + k] -= f * a[c * n + k];
                }
                y[r] -= f * y[c];
            }
        }
    }
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| a[c * n + k] * y[k]).sum();
        y[c] = (y[c] - s) / a[c * n + c];
    }
    y.iter().all(|v| v.is_finite()).then_some(y)
}

impl Remez {
    /// Zolotarev's minimax approximation of x^(−1/2) on [a, b]: for y = x/a in [1, κ], κ = b/a,
    /// R(y) = D Π_l (y + c_2l)/(y + c_2l−1) with c_j = sn²/cn² of j K/(2n + 1) for the parameter m
    /// = 1 − 1/κ, D such that the error is level.
    fn zolotarev(n: usize, a: f64, b: f64) -> Self {
        let m = 1.0 - a / b;
        let k = elliptic_k(m);
        let c: Vec<f64> = (1..=2 * n)
            .map(|j| {
                let (sn, cn) = jacobi_sn_cn(j as f64 * k / (2 * n + 1) as f64, m);
                sn * sn / (cn * cn)
            })
            .collect();
        // Poles at −c_1, −c_3, …, zeros at −c_2, −c_4, … (1-based).
        let poles: Vec<f64> = (0..n).map(|l| c[2 * l]).collect();
        let zeros: Vec<f64> = (0..n).map(|l| c[2 * l + 1]).collect();
        let res: Vec<f64> = (0..n)
            .map(|l| {
                let mut r = 1.0;
                for i in 0..n {
                    r *= zeros[i] - poles[l];
                    if i != l {
                        r /= poles[i] - poles[l];
                    }
                }
                r
            })
            .collect();
        // R(y) √y on a grid of y, for D.
        let ratio = |y: f64| {
            (1.0 + res
                .iter()
                .zip(&poles)
                .map(|(r, p)| r / (y + p))
                .sum::<f64>())
                * y.sqrt()
        };
        let (mut lo, mut hi) = (f64::INFINITY, 0.0_f64);
        let grid = 200 * (2 * n + 2);
        for i in 0..=grid {
            let y = (b / a).powf(i as f64 / grid as f64);
            let v = ratio(y);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let d = 2.0 / (lo + hi);
        // Back to x: R(x/a)/√a = d/√a [1 + Σ a res_l/(x + a c_l)].
        let sa = a.sqrt();
        Remez {
            a,
            b,
            norm: Dd::new(d / sa),
            res: res.iter().map(|r| Dd::new(d * sa * r)).collect(),
            pole: poles.iter().map(|p| Dd::new(a * p)).collect(),
            level: Dd::new((hi - lo) / (hi + lo)),
            reference: Vec::new(),
            power: Power::Real(-0.5),
        }
    }

    fn approx(&self, x: f64) -> Dd {
        let xd = Dd::new(x);
        let mut s = self.norm;
        for (r, p) in self.res.iter().zip(&self.pole) {
            s = s + *r / (xd + *p);
        }
        s
    }

    /// r(x)/f(x) − 1.
    fn error(&self, x: f64) -> Dd {
        self.approx(x) / self.power.eval(x) - Dd::new(1.0)
    }

    /// The extrema of the error (2n + 2 of them, alternating in sign) or None if the error
    /// does not have 2n + 1 zeros.
    fn extrema(&self) -> Option<Vec<f64>> {
        let n = self.pole.len();
        let (la, lb) = (self.a.ln(), self.b.ln());
        let at = |t: f64| {
            if t <= la {
                self.a
            } else if t >= lb {
                self.b
            } else {
                t.exp()
            }
        };
        let e = |t: f64| self.error(at(t));
        let grid = 100 * (2 * n + 2);
        let ts: Vec<f64> = (0..=grid)
            .map(|i| la + (lb - la) * i as f64 / grid as f64)
            .collect();
        let es: Vec<f64> = ts.iter().map(|&t| e(t).hi).collect();
        let mut bounds = vec![la];
        for i in 0..grid {
            if (es[i] > 0.0) != (es[i + 1] > 0.0) {
                let (mut l, mut h) = (ts[i], ts[i + 1]);
                let positive_left = es[i] > 0.0;
                for _ in 0..80 {
                    let mid = 0.5 * (l + h);
                    if mid <= l || mid >= h {
                        break;
                    }
                    if (e(mid).hi > 0.0) == positive_left {
                        l = mid;
                    } else {
                        h = mid;
                    }
                }
                bounds.push(0.5 * (l + h));
            }
        }
        bounds.push(lb);
        if bounds.len() != 2 * n + 3 {
            return None;
        }
        // On each piece the error has one sign: the largest |error|, from the largest on the
        // grid refined by golden section between its neighbours (a piece may hold more than
        // one bump, the endpoint of the interval and one inside), or an endpoint.
        let g = 0.5 * (5.0_f64.sqrt() - 1.0);
        let mut points = Vec::with_capacity(2 * n + 2);
        for j in 0..2 * n + 2 {
            let (lo, hi) = (bounds[j], bounds[j + 1]);
            let inside = (0..=grid).filter(|&i| ts[i] > lo && ts[i] < hi);
            let (mut l, mut h) = match inside.max_by(|&i, &k| es[i].abs().total_cmp(&es[k].abs())) {
                Some(i) => (
                    if i > 0 { ts[i - 1].max(lo) } else { lo },
                    if i < grid { ts[i + 1].min(hi) } else { hi },
                ),
                None => (lo, hi),
            };
            let size = |t: f64| e(t).abs();
            let bigger = |x: Dd, y: Dd| (x - y).hi > 0.0;
            let mut t1 = h - g * (h - l);
            let mut t2 = l + g * (h - l);
            let (mut f1, mut f2) = (size(t1), size(t2));
            for _ in 0..200 {
                if h - l <= 4.0 * f64::EPSILON * h.abs().max(1.0) {
                    break;
                }
                if bigger(f1, f2) {
                    h = t2;
                    t2 = t1;
                    f2 = f1;
                    t1 = h - g * (h - l);
                    f1 = size(t1);
                } else {
                    l = t1;
                    t1 = t2;
                    f1 = f2;
                    t2 = l + g * (h - l);
                    f2 = size(t2);
                }
            }
            let mut best = (0.5 * (l + h), size(0.5 * (l + h)));
            for end in [la, lb] {
                if (j == 0 && end == la) || (j == 2 * n + 1 && end == lb) {
                    let v = size(end);
                    if bigger(v, best.1) {
                        best = (end, v);
                    }
                }
            }
            points.push(at(best.0));
        }
        Some(points)
    }

    /// Newton's method for the equations on the reference: a0, res, poles and E.
    fn newton(&mut self) -> Result<(), RationalError> {
        let n = self.pole.len();
        let size = 2 * n + 2;
        let sign = |i: usize| if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        let residual = |s: &Remez| -> Vec<Dd> {
            s.reference
                .iter()
                .enumerate()
                .map(|(i, &x)| s.error(x) - s.level * Dd::new(sign(i)))
                .collect()
        };
        let worst = |g: &[Dd]| g.iter().fold(0.0_f64, |m, v| m.max(v.hi.abs()));
        let mut g = residual(self);
        for _ in 0..60 {
            let g_max = worst(&g);
            if g_max <= 1e-30 {
                return Ok(());
            }
            // Columns: a0, res_k, pole_k (each per unit of its own size), E.
            let mut jac = vec![0.0; size * size];
            for (i, &x) in self.reference.iter().enumerate() {
                let f = self.power.eval(x).hi;
                let row = &mut jac[i * size..(i + 1) * size];
                row[0] = self.norm.hi.abs() / f;
                for k in 0..n {
                    let d = x + self.pole[k].hi;
                    row[1 + k] = self.res[k].hi.abs() / (d * f);
                    row[1 + n + k] = -self.res[k].hi * self.pole[k].hi / (d * d * f);
                }
                row[size - 1] = -sign(i);
            }
            let rhs: Vec<f64> = g.iter().map(|v| -v.hi).collect();
            let Some(delta) = solve_dense(size, jac, rhs) else {
                return fail("Remez: the Newton system is singular");
            };
            // Damped if the step makes things worse.
            let mut damp = 1.0;
            loop {
                let mut trial = self.clone();
                trial.norm = trial.norm + Dd::new(damp * delta[0] * self.norm.hi.abs());
                for k in 0..n {
                    trial.res[k] =
                        trial.res[k] + Dd::new(damp * delta[1 + k] * self.res[k].hi.abs());
                    trial.pole[k] =
                        trial.pole[k] + Dd::new(damp * delta[1 + n + k] * self.pole[k].hi);
                }
                trial.level = trial.level + Dd::new(damp * delta[size - 1]);
                let ok = trial.pole.iter().all(|p| p.hi > 0.0);
                let g_trial = residual(&trial);
                if ok && worst(&g_trial) < g_max {
                    *self = trial;
                    g = g_trial;
                    break;
                }
                damp *= 0.5;
                if damp < 1e-3 {
                    // No progress: at the rounding of the equations.
                    return if g_max < 1e-24 {
                        Ok(())
                    } else {
                        fail("Remez: Newton's method does not converge")
                    };
                }
            }
        }
        fail("Remez: Newton's method does not converge")
    }

    /// The Remez iteration for `power`, from the current approximation.
    fn solve(&mut self, power: Power) -> Result<(), RationalError> {
        self.power = power;
        if self.reference.is_empty() {
            self.reference = match self.extrema() {
                Some(r) => r,
                None => return fail("Remez: the start does not alternate"),
            };
        }
        for _ in 0..50 {
            self.newton()?;
            let Some(next) = self.extrema() else {
                return fail("Remez: the error lost an alternation");
            };
            let levels: Vec<f64> = next.iter().map(|&x| self.error(x).hi.abs()).collect();
            let (lo, hi) = levels
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(l, h), v| (l.min(*v), h.max(*v)));
            self.reference = next;
            // An intermediate power is only the start of the next step.
            let tolerance = match power {
                Power::Exact { .. } => 1e-15,
                Power::Real(_) => 1e-6,
            };
            if hi - lo <= tolerance * hi {
                return self.newton();
            }
        }
        fail("Remez: the extrema do not level")
    }

    /// The approximation of the reciprocal function: 1/r, whose poles are the zeros of r
    /// (one between each pair of poles of r and one below the lowest), residues 1/r'(z).
    fn reciprocal(&self) -> Result<Self, RationalError> {
        let n = self.pole.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| self.pole[i].hi.total_cmp(&self.pole[j].hi));
        if self.norm.hi <= 0.0 || self.res.iter().any(|r| r.hi <= 0.0) {
            return fail("reciprocal: needs a0 > 0 and positive residues");
        }
        let r_at = |x: Dd| {
            let mut s = self.norm;
            let mut ds = Dd::new(0.0);
            for (r, p) in self.res.iter().zip(&self.pole) {
                let d = x + *p;
                s = s + *r / d;
                ds = ds - *r / (d * d);
            }
            (s, ds)
        };
        let mut zeros = Vec::with_capacity(n);
        for j in 0..n {
            // Between −pole[order[j+1]] and −pole[order[j]]; below −pole_max for the last.
            let right = -self.pole[order[j]].hi;
            let left = if j + 1 < n {
                -self.pole[order[j + 1]].hi
            } else {
                // r → a0 > 0 for x → −∞: go left until r is positive.
                let mut l = 2.0 * right;
                while r_at(Dd::new(l)).0.hi < 0.0 {
                    l *= 2.0;
                }
                l
            };
            // r is decreasing between its poles: + at the left end, − at the right end.
            let (mut l, mut h) = (left, right);
            for _ in 0..200 {
                let mid = 0.5 * (l + h);
                if mid <= l || mid >= h {
                    break;
                }
                if r_at(Dd::new(mid)).0.hi > 0.0 {
                    l = mid;
                } else {
                    h = mid;
                }
            }
            let mut z = Dd::new(0.5 * (l + h));
            for _ in 0..3 {
                let (s, ds) = r_at(z);
                z = z - s / ds;
            }
            zeros.push(z);
        }
        let one = Dd::new(1.0);
        Ok(Remez {
            norm: one / self.norm,
            res: zeros.iter().map(|&z| one / r_at(z).1).collect(),
            pole: zeros.iter().map(|&z| -z).collect(),
            level: -self.level,
            reference: self.reference.clone(),
            power: self.power,
            ..*self
        })
    }

    /// In double precision, the poles in increasing order.
    fn result(&self) -> RationalApprox {
        let mut order: Vec<usize> = (0..self.pole.len()).collect();
        order.sort_by(|&i, &j| self.pole[i].hi.total_cmp(&self.pole[j].hi));
        RationalApprox {
            norm: self.norm.hi + self.norm.lo,
            residues: order
                .iter()
                .map(|&k| self.res[k].hi + self.res[k].lo)
                .collect(),
            poles: order
                .iter()
                .map(|&k| self.pole[k].hi + self.pole[k].lo)
                .collect(),
            error: self.level.hi.abs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bridge++'s `parameter_rational.1_2inv_16_0.00010000_010.000` (from AlgRemez).
    const BRIDGE_INV_SQRT: &str = "    16    -1     2
  1.0000000000000000e-04  1.0000000000000000e+01
  5.018002e-10
  4.3574558749020234e-02
  2.8207065027470754e-03  4.7585806214468772e-06
  3.3576170610312745e-03  4.8434288539162890e-05
  4.5336562376041200e-03  1.7062242193342412e-04
  6.5727578492637551e-03  4.6878602726160390e-04
  9.8633580925257692e-03  1.1807735101203912e-03
  1.5032930251237260e-02  2.8746540912613244e-03
  2.3069615896426390e-02  6.9025314874015462e-03
  3.5518368574668308e-02  1.6483063771751346e-02
  5.4804635061805919e-02  3.9291771924812806e-02
  8.4805236232489006e-02  9.3714149929442006e-02
  1.3199271589603401e-01  2.2425829465375471e-01
  2.0821196594717534e-01  5.4138651391596759e-01
  3.3910302840678047e-01  1.3355557335196442e+00
  5.9661735791448511e-01  3.4773524221096599e+00
  1.2742913597630039e+00  1.0453149187763142e+01
  4.7865033663018295e+00  5.0150208755014667e+01
";

    /// `parameter_rational.1_4_16_0.00010000_010.000`.
    const BRIDGE_FOURTH_ROOT: &str = "    16     1     4
  1.0000000000000000e-04  1.0000000000000000e+01
  3.508166e-10
  4.5794184532095272e+00
  -5.0983818507532463e-07  1.5419456988555205e-05
  -2.2656536772451175e-06  8.2224295863668042e-05
  -7.3758791628595034e-06  2.5423469902379386e-04
  -2.2443796123670103e-05  6.6851974673265999e-04
  -6.6962514264947182e-05  1.6552278588407006e-03
  -1.9845177735413046e-04  4.0008847126649250e-03
  -5.8680212916489751e-04  9.5764406655993121e-03
  -1.7347612684128319e-03  2.2835748685122160e-02
  -5.1366992421452203e-03  5.4408265099237088e-02
  -1.5281264043141419e-02  1.2981835017351209e-01
  -4.5986187740439430e-02  3.1125427782991649e-01
  -1.4226362717753535e-01  7.5547621578472246e-01
  -4.7044826563222830e-01  1.8898139755447529e+00
  -1.8339198498529854e+00  5.1022744934971094e+00
  -1.1054829241864152e+01  1.7037885479133106e+01
  -3.0412794449872860e+02  1.2753317210340374e+02
";

    /// `parameter_rational.1_4_20_0.00010000_064.000` and `…1_2inv_20_0.00010000_064.000`.
    const BRIDGE_FOURTH_ROOT_20: &str = "    20     1     4
  1.0000000000000000e-04  6.4000000000000000e+01
  3.634011e-11
  7.6375808086656951e+00
  -3.9484270303845542e-07  1.2648140158919201e-05
  -1.6574022896929608e-06  6.5563791142342741e-05
  -4.9828153539974881e-06  1.9374981256587803e-04
  -1.3859645006957346e-05  4.8081786136660638e-04
  -3.7635405907721851e-05  1.1140014654650403e-03
  -1.0130792779759583e-04  2.5062895737597672e-03
  -2.7174607866007743e-04  5.5658431197768768e-03
  -7.2786289221330391e-04  1.2288598747521206e-02
  -1.9485169691427380e-03  2.7061447529443537e-02
  -5.2162901833816434e-03  5.9530210624301715e-02
  -1.3971899793095287e-02  1.3092359757446051e-01
  -3.7477630366890086e-02  2.8805770802975400e-01
  -1.0085690635754967e-01  6.3464060823809931e-01
  -2.7338761109950410e-01  1.4026694255777608e+00
  -7.5298429250519650e-01  3.1222868007000542e+00
  -2.1486307018703799e+00  7.0615535476882876e+00
  -6.6370948261714879e+00  1.6559100651433088e+01
  -2.4615194437744993e+01  4.2309081800884478e+01
  -1.4426526734435208e+02  1.3555003080730415e+02
  -3.9292637741811795e+03  9.9107845949364935e+02
";
    const BRIDGE_INV_SQRT_20: &str = "    20    -1     2
  1.0000000000000000e-04  6.4000000000000000e+01
  5.186495e-11
  1.5664939498279951e-02
  2.5551093418116136e-03  3.9253615145053759e-06
  2.9562993505007941e-03  3.9123066414117205e-05
  3.8216738577967449e-03  1.3249249328767257e-04
  5.2871158676491715e-03  3.4497735091833915e-04
  7.5827431232750710e-03  8.1527109328783038e-04
  1.1069074857685631e-02  1.8503505804290952e-03
  1.6293752492733300e-02  4.1258767276386374e-03
  2.4077898706154909e-02  9.1273676461270988e-03
  3.5646256132195568e-02  2.0120594673097389e-02
  5.2823435829273915e-02  4.4286966138172026e-02
  7.8333278541017612e-02  9.7429177420189861e-02
  1.1626857211424549e-01  2.1437416642453702e-01
  1.7287135618287258e-01  4.7213329334886839e-01
  2.5796681301796709e-01  1.0422526327094901e+00
  3.8802259307593967e-01  2.3130180530977635e+00
  5.9395823687045579e-01  5.1941739920881638e+00
  9.4519997694882385e-01  1.1979290828214856e+01
  1.6415995177610463e+00  2.9395948724991356e+01
  3.5029303722449123e+00  8.4297071169235565e+01
  1.3252584611766377e+01  3.9221007887064968e+02
";

    /// The largest relative difference of the coefficients.
    fn difference(a: &RationalApprox, b: &RationalApprox) -> f64 {
        let rel = |x: f64, y: f64| ((x - y) / y).abs();
        let mut d = rel(a.norm, b.norm);
        for k in 0..a.poles.len() {
            d = d
                .max(rel(a.poles[k], b.poles[k]))
                .max(rel(a.residues[k], b.residues[k]));
        }
        d
    }

    /// Max |r/f − 1| on a fine grid, r with its coefficients in double precision summed in
    /// double-double; and how much of it the rounding of the coefficients to double precision
    /// may account for: ε (|a0| + Σ |res/(x + pole)|)/f, largest on the grid.
    fn measured_error(r: &RationalApprox, p: i32, q: i32, a: f64, b: f64) -> (f64, f64) {
        let n = 20000;
        (0..=n)
            .map(|i| {
                let x = a * (b / a).powf(i as f64 / n as f64);
                let f = Power::Exact { p, q }.eval(x);
                let mut s = Dd::new(r.norm);
                let mut size = r.norm.abs();
                for (res, pole) in r.residues.iter().zip(&r.poles) {
                    let t = Dd::new(*res) / (Dd::new(x) + Dd::new(*pole));
                    s = s + t;
                    size += t.hi.abs();
                }
                ((s / f - Dd::new(1.0)).hi.abs(), f64::EPSILON * size / f.hi)
            })
            .fold((0.0, 0.0), |(e, r), (e1, r1)| (e.max(e1), r.max(r1)))
    }

    #[test]
    fn double_double_arithmetic() {
        let third = Dd::new(1.0) / Dd::new(3.0);
        let back = third * Dd::new(3.0) - Dd::new(1.0);
        assert!(back.hi.abs() < 1e-31, "{back:?}");
        // 2^(1/4) to 32 digits.
        let r = Power::Exact { p: 1, q: 4 }.eval(2.0);
        let r4 = r.powi(4) - Dd::new(2.0);
        assert!(r4.hi.abs() < 1e-30, "{r4:?}");
        let s = Power::Exact { p: -1, q: 2 }.eval(10.0) * Power::Exact { p: 1, q: 2 }.eval(10.0);
        assert!((s - Dd::new(1.0)).hi.abs() < 1e-31);
    }

    /// The multi-shift solve stops on the first pole: a file whose first pole is not the
    /// smallest is refused when it is read.
    #[test]
    fn a_file_with_the_smallest_pole_later_is_refused() {
        let mut lines: Vec<&str> = BRIDGE_INV_SQRT.lines().collect();
        // Lines 4 and 5 hold the first two (residue, pole) pairs.
        lines.swap(4, 5);
        let swapped = lines.join("\n");
        let e = RationalApprox::parse_bridge(&swapped).unwrap_err();
        assert!(e.0.contains("smallest"), "{}", e.0);
        RationalApprox::parse_bridge(BRIDGE_INV_SQRT).unwrap();
    }

    #[test]
    fn bridge_files_are_read_and_named() {
        let (r, spec) = RationalApprox::parse_bridge(BRIDGE_INV_SQRT).unwrap();
        assert_eq!(
            (spec.poles, spec.p, spec.q, spec.x_min, spec.x_max),
            (16, -1, 2, 1e-4, 10.0)
        );
        assert_eq!(r.poles.len(), 16);
        assert_eq!(r.residues[15], 4.7865033663018295e+00);
        // Bridge++'s parameters of the test HMC.Clover.RHMC_Nf2p1: bounds of |D|, squared.
        let bridge = RationalSpec {
            poles: 16,
            p: -1,
            q: 2,
            x_min: 0.01 * 0.01,
            x_max: 3.16227766 * 3.16227766,
        };
        assert_eq!(
            bridge.bridge_file_name(),
            "parameter_rational.1_2inv_16_0.00010000_010.000"
        );
        assert_eq!(
            bridge.with_power(1, 4).bridge_file_name(),
            "parameter_rational.1_4_16_0.00010000_010.000"
        );
        // Written and read back.
        let again = RationalApprox::parse_bridge(&r.to_bridge(&spec)).unwrap().0;
        assert_eq!(again.poles, r.poles);
        assert_eq!(again.residues, r.residues);
        assert_eq!(again.norm, r.norm);
    }

    /// The Remez algorithm gives Bridge++'s (AlgRemez's) coefficients for x^(−1/2) and x^(1/4)
    /// with 16 poles on [1e-4, 10] and with 20 on [1e-4, 64], and the error it states.
    #[test]
    fn minimax_is_bridges_approximation() {
        for text in [
            BRIDGE_INV_SQRT,
            BRIDGE_FOURTH_ROOT,
            BRIDGE_INV_SQRT_20,
            BRIDGE_FOURTH_ROOT_20,
        ] {
            let (bridge, spec) = RationalApprox::parse_bridge(text).unwrap();
            let (n, p, q, a, b) = (spec.poles, spec.p, spec.q, spec.x_min, spec.x_max);
            let ours = RationalApprox::minimax(&spec).unwrap();
            let d = difference(&ours, &bridge);
            eprintln!(
                "x^({p}/{q}), {n} poles on [{a}, {b}]: error {:.6e} (Bridge++ {:.6e}), coefficients differ by {d:.2e}",
                ours.error, bridge.error
            );
            assert!(((ours.error - bridge.error) / bridge.error).abs() < 1e-6);
            assert!(d < 1e-13, "coefficients differ by {d:e}");
            let (measured, rounding) = measured_error(&ours, p, q, a, b);
            assert!(measured < ours.error + rounding, "{measured:e}");
        }
    }

    /// Other powers, pole counts and intervals: the error is level and as stated.
    #[test]
    fn minimax_for_other_powers() {
        for (n, p, q, a, b) in [
            (8, -1, 4, 1e-3, 5.0),
            (12, 3, 4, 1e-4, 64.0),
            (10, -3, 4, 1e-2, 1.0),
            (20, 1, 2, 1e-4, 64.0),
        ] {
            let r = RationalApprox::minimax(&RationalSpec {
                poles: n,
                p,
                q,
                x_min: a,
                x_max: b,
            })
            .unwrap();
            let (measured, rounding) = measured_error(&r, p, q, a, b);
            eprintln!(
                "x^({p}/{q}), {n} poles on [{a}, {b}]: error {:.3e}, measured {measured:.3e}, rounding of the coefficients {rounding:.1e}",
                r.error
            );
            assert!(r.error < 1e-5);
            // The error the coefficients have in double precision: E, give or take their
            // rounding (largest for positive powers, whose terms cancel at small x).
            assert!(
                measured <= r.error + rounding && measured > 0.99 * r.error,
                "measured {measured:e} against {:e}",
                r.error
            );
            assert!(r.poles.iter().all(|p| *p > 0.0));
        }
    }
}
