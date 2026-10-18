//! Eigensolver files: `Main`, `Gauge`, `GaugeSmearing`, `Fopr` as in a propagator file, and an
//! `Eigensolver` section in place of `Solver`, with the keys of Bridge++'s `Eigensolver`
//! (`eigensolver_type`, `fermion_mult_mode`, `eigensolver_mode`,
//! `number_of_wanted_eigenvectors`, `number_of_working_eigenvectors`,
//! `convergence_criterion_squared`, `maximum_number_of_iteration`, `threshold_value`) and its
//! Chebyshev acceleration (`Fopr_Chebyshev`: `degree_of_polynomial`, `threshold_value`,
//! `upper_bound`) as a sub-section.
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
    /// Also write the table of eigenvalues to `file`.
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
        if self.main.backend != Backend::Cpu {
            return Err(invalid(format!(
                "Main.backend {}: the eigensolvers run on the cpu backend so far",
                self.main.backend
            )));
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
}
