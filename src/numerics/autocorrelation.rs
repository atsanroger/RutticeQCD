//! The error of the mean of a Monte Carlo time series and its integrated autocorrelation time,
//! by the Γ-method of U. Wolff, "Monte Carlo errors with less errors", Comput. Phys. Commun.
//! 156 (2004) 143 (`UWerr`), for one primary observable.

/// What the Γ-method finds for a series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaAnalysis {
    pub mean: f64,
    /// The error of the mean, autocorrelations included.
    pub error: f64,
    /// τ_int in units of the spacing of the series (½ for independent measurements).
    pub(crate) tau_int: f64,
    /// Its statistical error, Wolff (42).
    pub(crate) tau_int_error: f64,
    /// The summation window W.
    pub(crate) window: usize,
    /// Whether the automatic choice of W found its window before the end of the series (if
    /// not, W is the largest it tried and the series is too short for the result).
    pub(crate) window_found: bool,
}

/// The Γ-method for `series`, with the automatic window of Wolff with parameter `s_tau`
/// (1.5 is the usual choice; larger for slowly decaying autocorrelations). `None` for fewer
/// than two measurements.
pub fn gamma_method(series: &[f64], s_tau: f64) -> Option<GammaAnalysis> {
    let n = series.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mean = series.iter().sum::<f64>() / nf;
    let delta: Vec<f64> = series.iter().map(|a| a - mean).collect();
    // Γ(t) = 1/(N − t) Σ_i δ_i δ_(i+t), up to t = N/2.
    let t_max = n / 2;
    let mut gamma: Vec<f64> = (0..=t_max)
        .map(|t| {
            let s: f64 = delta[..n - t]
                .iter()
                .zip(&delta[t..])
                .map(|(a, b)| a * b)
                .sum();
            s / (n - t) as f64
        })
        .collect();
    if gamma[0] <= 0.0 {
        // A constant series.
        return Some(GammaAnalysis {
            mean,
            error: 0.0,
            tau_int: 0.5,
            tau_int_error: 0.0,
            window: 0,
            window_found: true,
        });
    }
    // The window: the first W at which g(W) = exp(−W/τ) − τ/√(W N) turns negative, with τ from
    // τ_int(W) = ½ + Σ_(t=1)^W Γ(t)/Γ(0) through τ = S_τ / ln((2τ_int + 1)/(2τ_int − 1)).
    let (mut window, mut window_found) = (t_max, false);
    let mut tau_w = 0.5;
    for w in 1..=t_max {
        tau_w += gamma[w] / gamma[0];
        let tau = if tau_w <= 0.5 {
            f64::MIN_POSITIVE
        } else {
            s_tau / ((2.0 * tau_w + 1.0) / (2.0 * tau_w - 1.0)).ln()
        };
        let g = (-(w as f64) / tau).exp() - tau / ((w as f64) * nf).sqrt();
        if g < 0.0 {
            (window, window_found) = (w, true);
            break;
        }
    }
    // C_F(W) = Γ(0) + 2 Σ_(t=1)^W Γ(t), and the bias of Γ removed: Γ(t) += C_F/N, Wolff (31).
    let c_f = |g: &[f64]| g[0] + 2.0 * g[1..=window].iter().sum::<f64>();
    let shift = c_f(&gamma) / nf;
    for g in gamma.iter_mut() {
        *g += shift;
    }
    let c = c_f(&gamma);
    let tau_int = c / (2.0 * gamma[0]);
    let w = window as f64;
    Some(GammaAnalysis {
        mean,
        error: (c / nf).max(0.0).sqrt(),
        tau_int,
        tau_int_error: 2.0 * tau_int * ((w - tau_int + 0.5).max(0.0) / nf).sqrt(),
        window,
        window_found,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestRng;

    /// A normal number from two uniform ones (Box–Muller).
    fn normal(rng: &mut TestRng) -> f64 {
        let (u1, u2) = (0.5 * (rng.uniform() + 1.0), rng.uniform());
        (-2.0 * u1.max(1e-300).ln()).sqrt() * (std::f64::consts::PI * u2).cos()
    }

    /// Independent measurements: τ_int near ½ and the error of the mean σ/√N.
    #[test]
    fn independent_measurements() {
        let mut rng = TestRng::new(3);
        let x: Vec<f64> = (0..20000).map(|_| normal(&mut rng)).collect();
        let r = gamma_method(&x, 1.5).unwrap();
        assert!(r.window_found);
        assert!(
            (r.tau_int - 0.5).abs() < 3.0 * r.tau_int_error + 0.02,
            "{r:?}"
        );
        let sigma = (x.iter().map(|a| (a - r.mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        let naive = sigma / (x.len() as f64).sqrt();
        assert!((r.error / naive - 1.0).abs() < 0.1, "{r:?} against {naive}");
    }

    /// An AR(1) process x_(i+1) = ρ x_i + √(1 − ρ²) ξ_i has τ_int = (1 + ρ)/(2 (1 − ρ)).
    #[test]
    fn a_process_of_known_autocorrelation_time() {
        let mut rng = TestRng::new(11);
        for rho in [0.5_f64, 0.9] {
            let mut x = vec![0.0; 200000];
            for i in 1..x.len() {
                x[i] = rho * x[i - 1] + (1.0 - rho * rho).sqrt() * normal(&mut rng);
            }
            let r = gamma_method(&x, 1.5).unwrap();
            let exact = (1.0 + rho) / (2.0 * (1.0 - rho));
            assert!(r.window_found);
            assert!(
                (r.tau_int - exact).abs() < 4.0 * r.tau_int_error,
                "ρ {rho}: {r:?}, exact {exact}"
            );
        }
    }

    /// A constant series has no error, and fewer than two numbers no analysis.
    #[test]
    fn degenerate_series() {
        assert!(gamma_method(&[1.0], 1.5).is_none());
        let r = gamma_method(&[2.0; 10], 1.5).unwrap();
        assert_eq!((r.mean, r.error, r.tau_int), (2.0, 0.0, 0.5));
    }
}
