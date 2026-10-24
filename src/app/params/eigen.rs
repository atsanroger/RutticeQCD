//! Eigensolver files: `Main`, `Gauge`, `GaugeSmearing`, `Fopr` as in a propagator file, and an
//! `Eigensolver` section in place of `Solver`, with the keys of Bridge++'s `Eigensolver`
//! (`eigensolver_type`, `fermion_mult_mode`, `eigensolver_mode`,
//! `number_of_wanted_eigenvectors`, `number_of_working_eigenvectors`,
//! `convergence_criterion_squared`, `maximum_number_of_iteration`, `threshold_value`) and its
//! Chebyshev acceleration (`Fopr_Chebyshev`: `degree_of_polynomial`, `threshold_value`,
//! `upper_bound`) as a sub-section.
//!
//! An optional `Pseudospectrum` section adds σ_min(z − A) = 1/‖(z − A)⁻¹‖ at points z, for the
//! same operator A, by Lanczos on (z − A)†(z − A) with the Lanczos settings of `Eigensolver`
//! (`PseudospectrumBlock`).
//!
//! ```yaml
//! Main:
//!   lattice_size : [4, 4, 4, 8]
//! Gauge:
//!   config_filename_input : ./conf_04040408.txt
//! Fopr:
//!   fermion_type       : Clover
//!   hopping_parameter  : 0.12
//!   clover_coefficient : 1.0
//! Eigensolver:
//!   eigensolver_type               : IRLanczos
//!   fermion_mult_mode              : H
//!   eigensolver_mode               : abs_ascending
//!   number_of_wanted_eigenvectors  : 20
//!   number_of_working_eigenvectors : 50
//!   convergence_criterion_squared  : 1.0e-24
//!   maximum_number_of_iteration    : 500
//!   threshold_value                : 0.15
//! ```

use super::blocks::{GaugeBlock, GaugeSmearingBlock, MainBlock};
use super::*;
use crate::eigen::{ArnoldiParams, Chebyshev, LanczosParams, Order, Tolerance, Which};
use crate::C64;

keyword_enum! {
    pub enum EigensolverType {
        /// Thick-restart Lanczos, for Hermitian operators.
        Lanczos => ["IRLanczos", "Lanczos", "Thick_restart_Lanczos"],
        /// Krylov-Schur, for any operator.
        Arnoldi => ["IRArnoldi", "Arnoldi", "Krylov_Schur"],
    }
}

keyword_enum! {
    /// What the eigenvalues are of, as Bridge++'s `fermion_mult_mode`, made of the operator
    /// that `operator_type` names.
    pub enum MultMode {
        D => ["D"],
        Ddag => ["Ddag", "Dagger"],
        DdagD => ["DdagD"],
        DDdag => ["DDdag"],
        /// γ5 D (domain-wall: Γ5 D with Γ5 = γ5 R, Hermitian for c = 0 only).
        H => ["H", "Gamma5D", "g5D"],
    }
}

keyword_enum! {
    /// Which eigenvalues first (Bridge++'s `eigensolver_mode`): for Arnoldi, `ascending` and
    /// `descending` go by the real part.
    pub enum EigensolverMode {
        Ascending => ["ascending"],
        Descending => ["descending"],
        AbsAscending => ["abs_ascending"],
        AbsDescending => ["abs_descending"],
    }
}

/// The whole file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EigenParams {
    #[serde(rename = "Main")]
    pub main: MainBlock,
    #[serde(rename = "Gauge")]
    pub gauge: GaugeBlock,
    #[serde(rename = "GaugeSmearing")]
    pub gauge_smearing: Option<GaugeSmearingBlock>,
    #[serde(rename = "Fopr")]
    pub fopr: FoprBlock,
    #[serde(rename = "Eigensolver")]
    pub eigensolver: EigensolverBlock,
    /// σ_min(z − A) at points z as well.
    #[serde(rename = "Pseudospectrum")]
    pub pseudospectrum: Option<PseudospectrumBlock>,
    /// Also write the table of eigenvalues (and of σ_min(z − A)) to `file`.
    #[serde(rename = "Output", default)]
    pub output: OutputConfig,
    #[serde(rename = "Smear", default)]
    bridge_smear: Option<serde::de::IgnoredAny>,
    #[serde(rename = "Director_Smear", default)]
    bridge_director_smear: Option<serde::de::IgnoredAny>,
    #[serde(rename = "Projection", default)]
    bridge_projection: Option<serde::de::IgnoredAny>,
    #[serde(skip)]
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EigensolverBlock {
    pub eigensolver_type: EigensolverType,
    pub fermion_mult_mode: MultMode,
    /// The operator the mode is made of: `Lexical` (D on the whole lattice, five-dimensional
    /// for domain-wall fermions; the default), or for domain-wall fermions `PV_right` or
    /// `PV_left`, the even-odd Pauli-Villars operator Ã on the even sites.
    pub operator_type: Option<OperatorType>,
    pub eigensolver_mode: EigensolverMode,
    pub number_of_wanted_eigenvectors: usize,
    /// Vectors beyond the wanted ones before a restart (Bridge++'s Np: the Krylov space has
    /// wanted + working vectors).
    pub number_of_working_eigenvectors: usize,
    /// Vectors kept at a restart: wanted + working / 2 by default.
    pub number_of_kept_vectors: Option<usize>,
    /// |A x − λ x|² below this.
    pub convergence_criterion_squared: f64,
    /// Restarts.
    pub maximum_number_of_iteration: usize,
    /// Stop once the eigenvalues before this one (in `eigensolver_mode`) have converged.
    pub threshold_value: Option<f64>,
    /// Of the random start vector.
    #[serde(default = "start_seed")]
    pub seed: u64,
    #[serde(rename = "Chebyshev")]
    pub chebyshev: Option<EigenChebyshevBlock>,
    /// Accepted for Bridge++ files; `Main.verbose_level` is what counts.
    pub verbose_level: Option<VerboseLevel>,
}

/// Lanczos on a Chebyshev polynomial of the operator, which is small on
/// [threshold_value, upper_bound] (on the scale of |λ|: [t², u²] of A² for the orders by
/// magnitude, as Bridge++'s `Fopr_Chebyshev` on D†D).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EigenChebyshevBlock {
    pub degree_of_polynomial: usize,
    pub threshold_value: f64,
    pub upper_bound: f64,
    pub verbose_level: Option<VerboseLevel>,
}

/// The points z of σ_min(z − A), the operator A that of `Eigensolver`: the ε-pseudospectrum of A
/// is where σ_min(z − A) < ε. Each point is one smallest eigenvalue of (z − A)†(z − A) by
/// Lanczos, with the sizes, `convergence_criterion_squared`, `maximum_number_of_iteration` and
/// `seed` of `Eigensolver` (no Chebyshev polynomial), from the same start vector.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PseudospectrumBlock {
    /// [Re z, Im z] each.
    #[serde(default)]
    pub points: Vec<[f64; 2]>,
    /// [Re c, Im c, r, n] each: n points z = c + r e^(2πik/n), k = 0..n−1.
    #[serde(default)]
    pub circles: Vec<[f64; 4]>,
    /// [Re z0, Im z0, Re z1, Im z1, n] each: n points from z0 to z1, both included.
    #[serde(default)]
    pub lines: Vec<[f64; 5]>,
}

/// A count given as a real in a list (`circles`, `lines`).
fn count(x: f64) -> Option<usize> {
    (x >= 1.0 && x.fract() == 0.0 && x < 1e7).then_some(x as usize)
}

impl PseudospectrumBlock {
    /// Every point, in the order of `points`, `circles`, `lines`.
    pub fn all_points(&self) -> Vec<C64> {
        let mut z: Vec<C64> = self.points.iter().map(|p| C64::new(p[0], p[1])).collect();
        for &[cx, cy, r, n] in &self.circles {
            let n = count(n).expect("checked");
            z.extend((0..n).map(|k| {
                let phi = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                C64::new(cx + r * phi.cos(), cy + r * phi.sin())
            }));
        }
        for &[x0, y0, x1, y1, n] in &self.lines {
            let n = count(n).expect("checked");
            z.extend((0..n).map(|k| {
                let t = if n == 1 {
                    0.0
                } else {
                    k as f64 / (n - 1) as f64
                };
                C64::new(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            }));
        }
        z
    }

    fn check(&self) -> Result<(), CliError> {
        if let Some(c) = self
            .circles
            .iter()
            .find(|c| count(c[3]).is_none() || c[2] <= 0.0)
        {
            return Err(invalid(format!(
                "Pseudospectrum.circles: [Re c, Im c, r, n] with r > 0 and n a whole number ≥ 1, not {c:?}"
            )));
        }
        if let Some(l) = self.lines.iter().find(|l| count(l[4]).is_none()) {
            return Err(invalid(format!(
                "Pseudospectrum.lines: [Re z0, Im z0, Re z1, Im z1, n] with n a whole number ≥ 1, not {l:?}"
            )));
        }
        if self.points.is_empty() && self.circles.is_empty() && self.lines.is_empty() {
            return Err(invalid(
                "Pseudospectrum: no points (points, circles or lines)".into(),
            ));
        }
        Ok(())
    }
}

fn start_seed() -> u64 {
    1234567
}

impl EigensolverBlock {
    /// The operator the mode is made of.
    pub fn operator(&self) -> OperatorType {
        self.operator_type.unwrap_or(OperatorType::Lexical)
    }

    fn sizes(&self) -> (usize, usize, usize) {
        let (wanted, working) = (
            self.number_of_wanted_eigenvectors,
            self.number_of_working_eigenvectors,
        );
        let keep = self.number_of_kept_vectors.unwrap_or(wanted + working / 2);
        (wanted, keep, wanted + working)
    }

    /// The settings of Lanczos for σ_min(z − A): the lowest eigenvalue of (z − A)†(z − A), on
    /// the operator itself.
    pub fn lanczos_singular(&self) -> LanczosParams {
        let (_, keep, max_dim) = self.sizes();
        LanczosParams {
            wanted: 1,
            keep,
            max_dim,
            tolerance: Tolerance::AbsoluteSquared(self.convergence_criterion_squared),
            max_restarts: self.maximum_number_of_iteration,
            chebyshev: None,
            order: Order::Ascending,
            threshold: None,
        }
    }

    /// The settings of Lanczos.
    pub fn lanczos(&self) -> LanczosParams {
        let (wanted, keep, max_dim) = self.sizes();
        let order = match self.eigensolver_mode {
            EigensolverMode::Ascending => Order::Ascending,
            EigensolverMode::Descending => Order::Descending,
            EigensolverMode::AbsAscending => Order::AbsAscending,
            EigensolverMode::AbsDescending => Order::AbsDescending,
        };
        let squared = matches!(order, Order::AbsAscending | Order::AbsDescending);
        LanczosParams {
            wanted,
            keep,
            max_dim,
            tolerance: Tolerance::AbsoluteSquared(self.convergence_criterion_squared),
            max_restarts: self.maximum_number_of_iteration,
            chebyshev: self.chebyshev.map(|c| {
                let scale = |x: f64| if squared { x * x } else { x };
                Chebyshev {
                    degree: c.degree_of_polynomial,
                    lo: scale(c.threshold_value),
                    hi: scale(c.upper_bound),
                }
            }),
            order,
            threshold: self.threshold_value,
        }
    }

    /// The settings of Arnoldi.
    pub fn arnoldi(&self) -> ArnoldiParams {
        let (wanted, keep, max_dim) = self.sizes();
        ArnoldiParams {
            wanted,
            keep,
            max_dim,
            tolerance: Tolerance::AbsoluteSquared(self.convergence_criterion_squared),
            max_restarts: self.maximum_number_of_iteration,
            which: match self.eigensolver_mode {
                EigensolverMode::Ascending => Which::SmallestReal,
                EigensolverMode::Descending => Which::LargestReal,
                EigensolverMode::AbsAscending => Which::SmallestMagnitude,
                EigensolverMode::AbsDescending => Which::LargestMagnitude,
            },
            threshold: self.threshold_value,
        }
    }

    /// Whether the operator of the mode is Hermitian.
    pub fn hermitian(&self, fopr: &FoprBlock) -> bool {
        match self.fermion_mult_mode {
            MultMode::DdagD | MultMode::DDdag => true,
            MultMode::D | MultMode::Ddag => false,
            MultMode::H => !fopr.fermion_type.is_domain_wall() || fopr.coefficient_c == 0.0,
        }
    }
}

impl EigenParams {
    /// Whether a file is an eigensolver file: one with an `Eigensolver` section.
    pub fn is_eigen_file(path: &Path) -> Result<bool, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => text
                .parse::<toml::Table>()
                .is_ok_and(|t| t.contains_key("Eigensolver")),
            _ => serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text)
                .ok()
                .and_then(|v| v.as_mapping().map(|m| m.contains_key("Eigensolver")))
                .unwrap_or(false),
        })
    }

    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        let mut p: Self =
            serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        p.validate()?;
        Ok(p)
    }

    /// Read and check a file; the format is chosen by the extension.
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut p: Self = match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => toml::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?,
            Some("yaml" | "yml") => {
                serde_yaml_ng::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?
            }
            other => {
                return Err(invalid(format!(
                    "parameter file must end in .toml, .yaml or .yml, not {other:?}"
                )))
            }
        };
        p.validate()?;
        Ok(p)
    }

    /// What was given but is not used.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn validate(&mut self) -> Result<(), CliError> {
        let mut notes = Vec::new();
        self.check(&mut notes)?;
        self.notes = notes;
        Ok(())
    }

    fn check(&self, notes: &mut Vec<String>) -> Result<(), CliError> {
        if self.bridge_smear.is_some()
            || self.bridge_director_smear.is_some()
            || self.bridge_projection.is_some()
        {
            return Err(invalid(
                "Smear / Director_Smear / Projection: gauge (link) smearing is the block GaugeSmearing (smear_type: Stout or APE, rho_uniform, number_of_smearing)".into(),
            ));
        }
        let f = &self.fopr;
        let dwf = f.fermion_type.is_domain_wall();
        if dwf {
            Params::check_domain_wall_fopr(f, "Fopr", notes)?;
        } else {
            Params::check_wilson_fopr(f, "Fopr", notes)?;
        }
        match self.main.backend {
            Backend::Cpu => {}
            Backend::CudaBridge => self.main.check_cuda_bridge()?,
            b => {
                return Err(invalid(format!(
                    "Main.backend {b}: the eigensolvers run on the cpu and cuda_bridge backends"
                )))
            }
        }
        let e = &self.eigensolver;
        let (wanted, keep, max_dim) = e.sizes();
        if wanted == 0 {
            return Err(invalid(
                "Eigensolver.number_of_wanted_eigenvectors: at least 1".into(),
            ));
        }
        let room = match e.eigensolver_type {
            EigensolverType::Lanczos => 1,
            EigensolverType::Arnoldi => 2,
        };
        if keep < wanted || keep + room > max_dim {
            return Err(invalid(format!(
                "Eigensolver: number_of_kept_vectors {keep} must be at least the {wanted} wanted and at most wanted + working − {room} = {}",
                max_dim as isize - room as isize
            )));
        }
        match e.operator() {
            OperatorType::Lexical => {}
            OperatorType::PvRight | OperatorType::PvLeft if dwf => {
                if e.fermion_mult_mode == MultMode::H {
                    return Err(invalid(format!(
                        "Eigensolver: fermion_mult_mode H is not defined for {} (no Γ5 makes it Hermitian); use DdagD, or D with IRArnoldi",
                        e.operator()
                    )));
                }
            }
            other => {
                return Err(invalid(format!(
                    "Eigensolver.operator_type {other}: Lexical{} so far",
                    if dwf { ", PV_right or PV_left" } else { "" }
                )))
            }
        }
        let hermitian = e.hermitian(f);
        if e.eigensolver_type == EigensolverType::Lanczos && !hermitian {
            return Err(invalid(format!(
                "Eigensolver: IRLanczos needs a Hermitian operator, which {} is not{}; use IRArnoldi, or DdagD",
                e.fermion_mult_mode,
                if e.fermion_mult_mode == MultMode::H {
                    " for domain-wall fermions with coefficient_c ≠ 0 (Γ5 D Γ5 ≠ D†)"
                } else {
                    ""
                }
            )));
        }
        if let Some(c) = e.chebyshev {
            if e.eigensolver_type != EigensolverType::Lanczos {
                return Err(invalid("Eigensolver.Chebyshev: for IRLanczos only".into()));
            }
            if !matches!(
                e.eigensolver_mode,
                EigensolverMode::Ascending | EigensolverMode::AbsAscending
            ) {
                return Err(invalid(
                    "Eigensolver.Chebyshev: for the low end (ascending, abs_ascending) only".into(),
                ));
            }
            if c.degree_of_polynomial < 2 || c.degree_of_polynomial % 2 != 0 {
                return Err(invalid(
                    "Eigensolver.Chebyshev.degree_of_polynomial: even and at least 2".into(),
                ));
            }
            if !(0.0 <= c.threshold_value && c.threshold_value < c.upper_bound) {
                return Err(invalid(
                    "Eigensolver.Chebyshev: 0 ≤ threshold_value < upper_bound".into(),
                ));
            }
            if c.verbose_level.is_some() {
                notes
                    .push("Eigensolver.Chebyshev.verbose_level: Main.verbose_level is used".into());
            }
        }
        if e.verbose_level.is_some() {
            notes.push("Eigensolver.verbose_level: Main.verbose_level is used".into());
        }
        if let Some(ps) = &self.pseudospectrum {
            ps.check()?;
        }
        Ok(())
    }
}

impl fmt::Display for EigenParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (m, fo, e) = (&self.main, &self.fopr, &self.eigensolver);
        writeln!(f, "lattice {:?} on {}", m.lattice_size, m.backend)?;
        match fo.fermion_type {
            FermionType::DomainWall => writeln!(
                f,
                "{}: Ls = {}, M5 = {}, m = {}, m_PV = {}, b = {}, c = {}, α = {} (D_PV: {})",
                fo.fermion_type,
                fo.ls(),
                fo.domain_wall_height.unwrap_or(f64::NAN),
                fo.mass(),
                fo.quark_mass_pauli_villars,
                fo.coefficient_b,
                fo.coefficient_c,
                fo.neff_alpha,
                fo.neff_alpha_pauli_villars
            )?,
            _ => writeln!(
                f,
                "{}: κ = {}{}",
                fo.fermion_type,
                fo.hopping_parameter.unwrap_or(f64::NAN),
                fo.clover_coefficient
                    .map_or(String::new(), |c| format!(", c_SW = {c}"))
            )?,
        }
        let (wanted, keep, max_dim) = e.sizes();
        writeln!(
            f,
            "{} of {} of {}, {}: {wanted} wanted, {keep} kept of {max_dim}, |A x − λ x|² < {:e}, at most {} restarts{}{}",
            e.eigensolver_type,
            e.fermion_mult_mode,
            e.operator(),
            e.eigensolver_mode,
            e.convergence_criterion_squared,
            e.maximum_number_of_iteration,
            e.threshold_value
                .map_or(String::new(), |t| format!(", threshold {t}")),
            e.chebyshev.map_or(String::new(), |c| format!(
                ", Chebyshev degree {} on [{}, {}]",
                c.degree_of_polynomial, c.threshold_value, c.upper_bound
            ))
        )?;
        if let Some(ps) = &self.pseudospectrum {
            writeln!(
                f,
                "pseudospectrum: σ_min(z − A) at {} points, Lanczos on (z − A)†(z − A)",
                ps.all_points().len()
            )?;
        }
        for n in &self.notes {
            writeln!(f, "note: {n}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOVER: &str = "
Main:
  lattice_size : [4, 4, 4, 8]
Gauge:
  config_filename_input : conf.txt
Fopr:
  fermion_type       : Clover
  hopping_parameter  : 0.12
  clover_coefficient : 1.0
Eigensolver:
  eigensolver_type               : IRLanczos
  fermion_mult_mode              : H
  eigensolver_mode               : abs_ascending
  number_of_wanted_eigenvectors  : 20
  number_of_working_eigenvectors : 50
  convergence_criterion_squared  : 1.0e-24
  maximum_number_of_iteration    : 500
  threshold_value                : 0.15
  verbose_level                  : General
";

    #[test]
    fn bridge_keys_and_what_is_refused() {
        let p = EigenParams::from_yaml(CLOVER).unwrap();
        let l = p.eigensolver.lanczos();
        assert_eq!((l.wanted, l.keep, l.max_dim), (20, 45, 70));
        assert_eq!(l.order, Order::AbsAscending);
        assert_eq!(l.tolerance, Tolerance::AbsoluteSquared(1e-24));
        assert!(p.to_string().contains("IRLanczos of H of Lexical"), "{p}");
        let err = |text: String| EigenParams::from_yaml(&text).unwrap_err().to_string();
        let e = err(CLOVER.replace(
            "fermion_mult_mode              : H",
            "fermion_mult_mode : D",
        ));
        assert!(e.contains("use IRArnoldi"), "{e}");
        let arnoldi = CLOVER.replace("IRLanczos", "IRArnoldi").replace(
            "fermion_mult_mode              : H",
            "fermion_mult_mode : D",
        );
        let p = EigenParams::from_yaml(&arnoldi).unwrap();
        assert_eq!(p.eigensolver.arnoldi().which, Which::SmallestMagnitude);
        let e = err(CLOVER.replace(
            "  verbose_level                  : General\n",
            "  Chebyshev:\n    degree_of_polynomial : 41\n    threshold_value : 0.16\n    upper_bound : 2.5\n",
        ));
        assert!(e.contains("even"), "{e}");
        let p = EigenParams::from_yaml(&CLOVER.replace(
            "  verbose_level                  : General\n",
            "  Chebyshev:\n    degree_of_polynomial : 40\n    threshold_value : 0.16\n    upper_bound : 2.5\n",
        ))
        .unwrap();
        let c = p.eigensolver.lanczos().chebyshev.unwrap();
        assert!((c.lo - 0.0256).abs() < 1e-15 && (c.hi - 6.25).abs() < 1e-15);
        let mobius = CLOVER.replace(
            "  fermion_type       : Clover\n  hopping_parameter  : 0.12\n  clover_coefficient : 1.0\n",
            "  fermion_type : Mobius\n  quark_mass : 0.1\n  domain_wall_height : 1.6\n  extent_of_5th_dimension : 8\n",
        );
        let e = err(mobius.clone());
        assert!(e.contains("coefficient_c"), "{e}");
        let e = err(mobius.replace(
            "fermion_mult_mode              : H",
            "fermion_mult_mode : DdagD\n  operator_type : Even_odd",
        ));
        assert!(e.contains("PV_right or PV_left"), "{e}");
        assert!(EigenParams::from_yaml(&mobius.replace(
            "fermion_mult_mode              : H",
            "fermion_mult_mode : DdagD\n  operator_type : PV_right"
        ))
        .is_ok());
    }

    #[test]
    fn pseudospectrum_points() {
        let with = |section: &str| format!("{CLOVER}Pseudospectrum:\n{section}");
        let p = EigenParams::from_yaml(&with(
            "  points : [[0.1, 0.2]]\n  circles : [[1.0, 0.0, 0.5, 4]]\n  lines : [[0.0, 0.0, 1.0, 0.0, 3]]\n",
        ))
        .unwrap();
        let z = p.pseudospectrum.as_ref().unwrap().all_points();
        let want = [
            (0.1, 0.2),
            (1.5, 0.0),
            (1.0, 0.5),
            (0.5, 0.0),
            (1.0, -0.5),
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
        ];
        assert_eq!(z.len(), want.len());
        for (z, (re, im)) in z.iter().zip(want) {
            assert!(
                (z.re - re).abs() < 1e-15 && (z.im - im).abs() < 1e-15,
                "{z:?}"
            );
        }
        assert!(p.to_string().contains("at 8 points"), "{p}");
        let l = p.eigensolver.lanczos_singular();
        assert_eq!(
            (l.wanted, l.order, l.chebyshev.is_none()),
            (1, Order::Ascending, true)
        );
        let err = |text: String| EigenParams::from_yaml(&text).unwrap_err().to_string();
        assert!(err(with("  circles : [[0.0, 0.0, 0.5, 2.5]]\n")).contains("whole number"));
        assert!(err(with("  circles : [[0.0, 0.0, 0.0, 4]]\n")).contains("r > 0"));
        assert!(err(with("  lines : [[0.0, 0.0, 1.0, 0.0, 0]]\n")).contains("whole number"));
        assert!(err(with("  points : []\n")).contains("no points"));
    }
}
