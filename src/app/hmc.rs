//! Hybrid Monte Carlo from a parameter file: the start configuration, the actions on the
//! levels of the integrator, the trajectories, and where the configurations go.
//!
//! ```yaml
//! lattice: [4, 4, 4, 8]
//! start: {type: file, file: conf_04040408.txt}     # or {type: cold}
//! gauge_smearing: {type: stout, rho: 0.1, steps: 2} # the links of `smeared` fermions
//! hmc:
//!   trajectories: 10
//!   first_trajectory: 200000      # the random numbers are Bridge++'s M-series of this number
//!   trajectory_length: 1.0
//!   integrator: {scheme: omelyan, steps: [5, 4]}
//! actions:
//!   - {type: rectangle, beta: 2.3, c_plaq: 3.648, c_rect: -0.331, level: 1}
//!   - type: two_flavor_ratio
//!     fermion: {type: domain_wall, ls: 8, m5: 1.6, mass: 0.1, b: 1.0, c: 0.0,
//!               boundary: [periodic, periodic, periodic, antiperiodic]}
//!     level: 0
//!   - type: rational              # det(D†D)^(1/2): one flavour (the strange quark), RHMC
//!     fermion: {type: clover, kappa: 0.12, csw: 1.0, boundary: [periodic, periodic, periodic, antiperiodic]}
//!     poles: 16
//!     bounds: [1.0e-2, 3.16227766] # of |D|: D†D in [lower², upper²], Bridge++'s lower_bound, upper_bound
//!     level: 1
//! output: {save_every: 5, file: "conf_{n}.txt"}
//! ```
//!
//! A `two_flavor` action takes `even_odd: true` for the action of the Schur complement on the
//! even sites (`hmc::TwoFlavorEo`, Wilson, clover or domain-wall fermions), and its `solver`
//! `initial_guess: n` to start the force solves from the last n solutions (`hmc::Chrono`):
//!
//! ```yaml
//!   - type: two_flavor
//!     fermion: {type: clover, kappa: 0.12, csw: 1.0, boundary: [periodic, periodic, periodic, antiperiodic]}
//!     even_odd: true
//!     solver: {tolerance_squared: 1.0e-24, initial_guess: 4}
//!     level: 0
//! ```
//!
//! The conventions are those of Bridge++'s `HMC_General` (momenta, pseudofermions and the
//! Metropolis test from one M-series generator; the integrator levels of its
//! `Builder_Integrator`), so a run can be repeated there. The gauge smearing of the fermion
//! links is `gauge_smearing`, as in the other parameter files; it is not the smearing of
//! sources.

use super::config::{BoundaryConfig, GaugeFormat, GaugeSmearingConfig, AMBIGUOUS_SMEARING};
use super::CliError;
use crate::fermion::{Clover, CloverParams, Mdwf, MdwfParams, Wilson, WilsonParams};
use crate::gauge::{plaquette, random_gauge};
use crate::hmc::{
    cold_start, Hmc, MdAction, Mseries, PlaquetteAction, Rational, RectangleAction, Scheme, Term,
    TwoFlavor, TwoFlavorEo, TwoFlavorRatio,
};
use crate::io::{BridgeText, GaugeFile, Ildg, Scidac};
use crate::numerics::{RationalApprox, RationalSpec};
use crate::solver::Stop;
use crate::{Boundary, Comm, Lattice, Stencil, ND};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmcConfig {
    pub lattice: [usize; ND],
    #[serde(default)]
    pub process_grid: Option<[usize; ND]>,
    pub start: StartConfig,
    /// The smearing of the links of the fermion actions that ask for it (`smeared`).
    #[serde(default)]
    pub gauge_smearing: Option<GaugeSmearingConfig>,
    #[serde(default)]
    smearing: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    smear: Option<serde::de::IgnoredAny>,
    pub hmc: HmcBlock,
    pub actions: Vec<ActionConfig>,
    #[serde(default)]
    pub output: HmcOutput,
}

/// The configuration the run starts from.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum StartConfig {
    /// Read from a file; relative paths from the parameter file's directory.
    File {
        file: PathBuf,
        #[serde(default)]
        format: GaugeFormat,
    },
    /// Every link the identity.
    Cold,
    /// Random links.
    Hot { seed: u64 },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmcBlock {
    pub trajectories: usize,
    /// The number of the first trajectory: the random numbers of the run are those of the
    /// M-series generator made from it (Bridge++'s `trajectory_number`).
    #[serde(default)]
    pub first_trajectory: u32,
    pub trajectory_length: f64,
    pub integrator: IntegratorConfig,
    #[serde(default = "yes")]
    pub metropolis: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegratorConfig {
    pub scheme: SchemeConfig,
    /// Steps of each level, the outermost first.
    pub steps: Vec<usize>,
    /// λ of the Omelyan scheme.
    #[serde(default = "omelyan_lambda")]
    pub lambda: f64,
    /// Order of the Taylor series of exp(ε iP).
    #[serde(default = "exp_order")]
    pub exp_order: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemeConfig {
    Leapfrog,
    Omelyan,
}

/// An action and the level of the integrator it is on.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ActionConfig {
    Plaquette {
        beta: f64,
        level: usize,
    },
    /// Plaquettes and 1 × 2 rectangles (Iwasaki: c_plaq = 3.648, c_rect = −0.331).
    Rectangle {
        beta: f64,
        c_plaq: f64,
        c_rect: f64,
        level: usize,
    },
    /// Two flavours, S = φ† (D†D)⁻¹ φ.
    TwoFlavor {
        fermion: HmcFermion,
        level: usize,
        /// On the links smeared by `gauge_smearing`.
        #[serde(default = "yes")]
        smeared: bool,
        /// Even-odd preconditioned: φ on the even sites, S = φ_e† (Ŝ†Ŝ)⁻¹ φ_e − 2 ln |det T_oo|
        /// with Ŝ the Schur complement (`hmc::TwoFlavorEo`; Bridge++'s
        /// `Action_F_alt_Standard_eo` for Wilson fermions). The same distribution of links,
        /// about a third of the operator applications per trajectory. Off by default.
        #[serde(default)]
        even_odd: bool,
        #[serde(default)]
        solver: HmcSolver,
    },
    /// Two flavours over two of `denominator` (Hasenbusch), S = φ† H_d (H†H)⁻¹ H_d† φ; for
    /// domain-wall fermions without `denominator`, over two Pauli-Villars flavours.
    TwoFlavorRatio {
        fermion: HmcFermion,
        #[serde(default)]
        denominator: Option<HmcFermion>,
        level: usize,
        #[serde(default = "yes")]
        smeared: bool,
        #[serde(default)]
        solver: HmcSolver,
    },
    /// det(D†D)^(p/q) by rational approximation (RHMC), p/q = `power` (one flavour: [1, 2]):
    /// S = φ† r(D†D) φ, r ≈ x^(−p/q), φ = r'(D†D) ξ, r' ≈ x^(p/2q), minimax with `poles`
    /// poles on [lower², upper²], `bounds` = [lower, upper] those of the eigenvalues of
    /// |D| = √(D†D), as Bridge++'s `lower_bound` and `upper_bound`. The coefficients are
    /// computed (Remez), or read from Bridge++'s files `parameter_rational.*` in the
    /// directory `coefficients` (relative to the parameter file's directory).
    Rational {
        fermion: HmcFermion,
        level: usize,
        poles: usize,
        bounds: [f64; 2],
        #[serde(default = "one_flavor")]
        power: [i32; 2],
        #[serde(default)]
        coefficients: Option<PathBuf>,
        #[serde(default = "yes")]
        smeared: bool,
        #[serde(default)]
        solver: HmcSolver,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HmcFermion {
    Wilson {
        kappa: f64,
        boundary: [BoundaryConfig; ND],
    },
    Clover {
        kappa: f64,
        csw: f64,
        boundary: [BoundaryConfig; ND],
    },
    /// Möbius domain-wall fermions (b = 1, c = 0: Shamir's), the normalisation of Bridge++.
    DomainWall {
        ls: usize,
        m5: f64,
        mass: f64,
        #[serde(default = "one_f")]
        b: f64,
        #[serde(default)]
        c: f64,
        boundary: [BoundaryConfig; ND],
    },
}

/// CG on D†D for the forces and the energies.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmcSolver {
    #[serde(default = "md_tolerance")]
    pub tolerance_squared: f64,
    /// The energies at the start and the end of a trajectory: as precise as the force or more.
    #[serde(default = "md_tolerance")]
    pub energy_tolerance_squared: f64,
    #[serde(default = "md_iterations")]
    pub max_iterations: usize,
    /// Start each force solve from the best combination of the solutions of the last this
    /// many force solves of the trajectory (`hmc::Chrono`); 0 (the default): from zero, as
    /// Bridge++ does, so that its runs are repeated number by number. Two flavours only.
    #[serde(default)]
    pub initial_guess: usize,
}

impl Default for HmcSolver {
    fn default() -> Self {
        Self {
            tolerance_squared: md_tolerance(),
            energy_tolerance_squared: md_tolerance(),
            max_iterations: md_iterations(),
            initial_guess: 0,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmcOutput {
    /// Write the configuration after every this many trajectories (and after the last).
    #[serde(default)]
    pub save_every: Option<usize>,
    /// The file name, `{n}` replaced by the trajectory number; relative to the parameter
    /// file's directory. On several processes rank 0 writes it.
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub format: GaugeFormat,
}

fn yes() -> bool {
    true
}
fn one_flavor() -> [i32; 2] {
    [1, 2]
}
fn one_f() -> f64 {
    1.0
}
fn omelyan_lambda() -> f64 {
    0.1931833275
}
fn exp_order() -> usize {
    8
}
fn md_tolerance() -> f64 {
    1e-24
}
fn md_iterations() -> usize {
    10000
}

fn boundary(b: &[BoundaryConfig; ND]) -> [Boundary; ND] {
    b.map(Boundary::from)
}

fn invalid<T>(m: String) -> Result<T, CliError> {
    Err(CliError::Invalid(m))
}

/// Whether a parameter file is an HMC file (it has an `hmc` section).
pub fn is_hmc_file(path: &Path) -> Result<bool, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(match path.extension().and_then(|e| e.to_str()) {
        Some("toml") => text
            .parse::<toml::Table>()
            .is_ok_and(|t| t.contains_key("hmc")),
        _ => serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text)
            .ok()
            .and_then(|v| v.as_mapping().map(|m| m.contains_key("hmc")))
            .unwrap_or(false),
    })
}

impl HmcConfig {
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let c: Self = match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => toml::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?,
            _ => serde_yaml_ng::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?,
        };
        c.check()?;
        Ok(c)
    }

    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        let c: Self = serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<(), CliError> {
        if self.smearing.is_some() || self.smear.is_some() {
            return invalid(AMBIGUOUS_SMEARING.into());
        }
        let levels = self.hmc.integrator.steps.len();
        if levels == 0 {
            return invalid("hmc.integrator.steps: at least one level".into());
        }
        if self.actions.is_empty() {
            return invalid("actions: at least one".into());
        }
        for (i, a) in self.actions.iter().enumerate() {
            let (level, smeared, ratio) = match a {
                ActionConfig::Plaquette { level, .. } | ActionConfig::Rectangle { level, .. } => {
                    (*level, false, None)
                }
                ActionConfig::TwoFlavor { level, smeared, .. } => (*level, *smeared, None),
                ActionConfig::Rational {
                    level,
                    smeared,
                    fermion,
                    poles,
                    bounds,
                    power,
                    ..
                } => {
                    if matches!(fermion, HmcFermion::DomainWall { .. }) {
                        return invalid(format!(
                            "actions[{i}]: rational: Wilson or clover fermions (domain-wall fermions need the ratio over Pauli-Villars fields)"
                        ));
                    }
                    let [p, q] = *power;
                    if !(p > 0 && q > p) {
                        return invalid(format!(
                            "actions[{i}]: rational: power [{p}, {q}] must be a fraction p/q in (0, 1)"
                        ));
                    }
                    if *poles == 0 || !(bounds[0] > 0.0 && bounds[1] > bounds[0]) {
                        return invalid(format!(
                            "actions[{i}]: rational: poles > 0 and 0 < bounds[0] < bounds[1]"
                        ));
                    }
                    (*level, *smeared, None)
                }
                ActionConfig::TwoFlavorRatio {
                    level,
                    smeared,
                    fermion,
                    denominator,
                    ..
                } => (*level, *smeared, Some((fermion, denominator))),
            };
            if level >= levels {
                return invalid(format!(
                    "actions[{i}].level {level}: the integrator has {levels} levels"
                ));
            }
            let _ = smeared;
            if let ActionConfig::TwoFlavorRatio { solver, .. } = a {
                if solver.initial_guess > 0 {
                    return invalid(format!(
                        "actions[{i}].solver.initial_guess: for two_flavor actions only"
                    ));
                }
            }
            if let Some((f, d)) = ratio {
                match (f, d) {
                    (HmcFermion::DomainWall { .. }, None) => {}
                    (_, None) => {
                        return invalid(format!(
                            "actions[{i}]: a ratio of Wilson or clover fermions needs a denominator"
                        ))
                    }
                    (a, Some(b)) if std::mem::discriminant(a) != std::mem::discriminant(b) => {
                        return invalid(format!(
                            "actions[{i}]: the denominator must be the same kind of fermion"
                        ))
                    }
                    _ => {}
                }
            }
        }
        if self.output.save_every.is_some() != self.output.file.is_some() {
            return invalid("output: save_every and file go together".into());
        }
        Ok(())
    }

    /// The actions on the lattice `lat`; files named relative to `base`.
    fn terms(&self, lat: &Arc<Lattice>, base: &Path) -> Result<Vec<Term<'static>>, CliError> {
        let smearing = self.gauge_smearing.map(|g| g.smearing());
        let stops = |s: &HmcSolver| {
            let stop = |t: f64| Stop {
                tolerance_sqr: t,
                max_iterations: s.max_iterations,
                max_restarts: 10,
            };
            (stop(s.tolerance_squared), stop(s.energy_tolerance_squared))
        };
        self.actions
            .iter()
            .map(|a| -> Result<Term<'static>, CliError> {
                let (level, action): (usize, Box<dyn MdAction>) = match a {
                    ActionConfig::Plaquette { beta, level } => {
                        (*level, Box::new(PlaquetteAction { beta: *beta }))
                    }
                    ActionConfig::Rectangle {
                        beta,
                        c_plaq,
                        c_rect,
                        level,
                    } => (
                        *level,
                        Box::new(RectangleAction {
                            beta: *beta,
                            c_plaq: *c_plaq,
                            c_rect: *c_rect,
                        }),
                    ),
                    ActionConfig::TwoFlavor {
                        fermion,
                        level,
                        smeared,
                        even_odd,
                        solver,
                    } => {
                        let sm = smearing.filter(|_| *smeared);
                        let (md, h) = stops(solver);
                        let g = solver.initial_guess;
                        // The same operator in either action; the clover term adds its force
                        // (and, even-odd, its det term).
                        macro_rules! two_flavor {
                            ($op:expr, $clover:expr) => {{
                                let op = $op;
                                let clover: Option<f64> = $clover;
                                if *even_odd {
                                    let a = TwoFlavorEo::new(op, sm, md, h).with_guess(g);
                                    match clover {
                                        Some(kc) => {
                                            Box::new(a.with_clover(kc)) as Box<dyn MdAction>
                                        }
                                        None => Box::new(a),
                                    }
                                } else {
                                    let a = TwoFlavor::new(op, sm, md, h).with_guess(g);
                                    match clover {
                                        Some(kc) => {
                                            Box::new(a.with_clover(kc)) as Box<dyn MdAction>
                                        }
                                        None => Box::new(a),
                                    }
                                }
                            }};
                        }
                        let action: Box<dyn MdAction> = match *fermion {
                            HmcFermion::Wilson { kappa, boundary: b } => two_flavor!(
                                Wilson::<f64>::new(
                                    lat,
                                    WilsonParams {
                                        kappa,
                                        boundary: boundary(&b),
                                    },
                                ),
                                None
                            ),
                            HmcFermion::Clover {
                                kappa,
                                csw,
                                boundary: b,
                            } => two_flavor!(
                                Clover::<f64>::new(
                                    lat,
                                    CloverParams {
                                        kappa,
                                        csw,
                                        boundary: boundary(&b),
                                    },
                                ),
                                Some(kappa * csw)
                            ),
                            HmcFermion::DomainWall { .. } => {
                                two_flavor!(Mdwf::<f64>::new(lat, mdwf_params(fermion)), None)
                            }
                        };
                        (*level, action)
                    }
                    ActionConfig::TwoFlavorRatio {
                        fermion,
                        denominator,
                        level,
                        smeared,
                        solver,
                    } => {
                        let sm = smearing.filter(|_| *smeared);
                        let (md, h) = stops(solver);
                        let action: Box<dyn MdAction> = match (*fermion, *denominator) {
                            (HmcFermion::DomainWall { .. }, d) => {
                                let p = mdwf_params(fermion);
                                let pd = d.map_or(p.pauli_villars(), |d| mdwf_params(&d));
                                Box::new(TwoFlavorRatio::new(
                                    Mdwf::<f64>::new(lat, p),
                                    Mdwf::<f64>::new(lat, pd),
                                    sm,
                                    md,
                                    h,
                                ))
                            }
                            (
                                HmcFermion::Wilson { kappa, boundary: b },
                                Some(HmcFermion::Wilson {
                                    kappa: kd,
                                    boundary: bd,
                                }),
                            ) => {
                                let w = |kappa, b: &[BoundaryConfig; ND]| {
                                    Wilson::<f64>::new(
                                        lat,
                                        WilsonParams {
                                            kappa,
                                            boundary: boundary(b),
                                        },
                                    )
                                };
                                Box::new(TwoFlavorRatio::new(w(kappa, &b), w(kd, &bd), sm, md, h))
                            }
                            (
                                HmcFermion::Clover {
                                    kappa,
                                    csw,
                                    boundary: b,
                                },
                                Some(HmcFermion::Clover {
                                    kappa: kd,
                                    csw: cd,
                                    boundary: bd,
                                }),
                            ) => Box::new(
                                TwoFlavorRatio::new(
                                    clover(lat, kappa, csw, &b),
                                    clover(lat, kd, cd, &bd),
                                    sm,
                                    md,
                                    h,
                                )
                                .with_clover(kappa * csw, kd * cd),
                            ),
                            _ => unreachable!("checked when the file is read"),
                        };
                        (*level, action)
                    }
                    ActionConfig::Rational {
                        fermion,
                        level,
                        poles,
                        bounds,
                        power: [p, q],
                        coefficients,
                        smeared,
                        solver,
                    } => {
                        let sm = smearing.filter(|_| *smeared);
                        let (md_stop, h_stop) = stops(solver);
                        // S = φ† (D†D)^(−p/q) φ, φ = (D†D)^(p/2q) ξ.
                        let spec = RationalSpec {
                            poles: *poles,
                            p: -p,
                            q: *q,
                            x_min: bounds[0] * bounds[0],
                            x_max: bounds[1] * bounds[1],
                        };
                        let g = gcd(*p, 2 * q);
                        let (md, hb) = (
                            approximation(&spec, coefficients.as_deref(), base)?,
                            approximation(
                                &spec.with_power(p / g, 2 * q / g),
                                coefficients.as_deref(),
                                base,
                            )?,
                        );
                        let action: Box<dyn MdAction> = match *fermion {
                            HmcFermion::Wilson { kappa, boundary: b } => Box::new(Rational::new(
                                Wilson::<f64>::new(
                                    lat,
                                    WilsonParams {
                                        kappa,
                                        boundary: boundary(&b),
                                    },
                                ),
                                sm,
                                md,
                                hb,
                                md_stop,
                                h_stop,
                            )),
                            HmcFermion::Clover {
                                kappa,
                                csw,
                                boundary: b,
                            } => Box::new(
                                Rational::new(
                                    clover(lat, kappa, csw, &b),
                                    sm,
                                    md,
                                    hb,
                                    md_stop,
                                    h_stop,
                                )
                                .with_clover(kappa * csw),
                            ),
                            HmcFermion::DomainWall { .. } => {
                                unreachable!("checked when the file is read")
                            }
                        };
                        (*level, action)
                    }
                };
                Ok(Term { level, action })
            })
            .collect()
    }
}

fn clover(lat: &Arc<Lattice>, kappa: f64, csw: f64, b: &[BoundaryConfig; ND]) -> Clover<f64> {
    Clover::<f64>::new(
        lat,
        CloverParams {
            kappa,
            csw,
            boundary: boundary(b),
        },
    )
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// The rational approximation `spec`: read from Bridge++'s file in `dir` (relative to `base`)
/// if a directory is given, computed otherwise.
fn approximation(
    spec: &RationalSpec,
    dir: Option<&Path>,
    base: &Path,
) -> Result<RationalApprox, CliError> {
    match dir {
        Some(dir) => {
            RationalApprox::read_bridge(&base.join(dir).join(spec.bridge_file_name()), spec)
        }
        None => RationalApprox::minimax(spec),
    }
    .map_err(|e| CliError::Invalid(format!("rational approximation: {e}")))
}

impl HmcConfig {}

fn mdwf_params(f: &HmcFermion) -> MdwfParams {
    match *f {
        HmcFermion::DomainWall {
            ls,
            m5,
            mass,
            b,
            c,
            boundary: bd,
        } => MdwfParams {
            ls,
            m5,
            mass,
            b,
            c,
            boundary: boundary(&bd),
        },
        _ => unreachable!("a domain-wall fermion"),
    }
}

fn gauge_format(f: GaugeFormat) -> Result<Box<dyn GaugeFile>, CliError> {
    Ok(match f {
        GaugeFormat::BridgeText => Box::new(BridgeText),
        GaugeFormat::Ildg => Box::new(Ildg::default()),
        GaugeFormat::Scidac => Box::new(Scidac::default()),
        #[cfg(feature = "hdf5")]
        GaugeFormat::Hdf5 => Box::new(crate::io::Hdf5::default()),
        #[cfg(not(feature = "hdf5"))]
        GaugeFormat::Hdf5 => {
            return invalid("format hdf5 needs a program built with the feature hdf5".into())
        }
    })
}

/// What a run did, trajectory by trajectory.
#[derive(Debug, Clone)]
pub struct HmcReport {
    pub plaquette_start: f64,
    /// (trajectory number, ΔH, accepted, plaquette after it).
    pub trajectories: Vec<(u32, f64, bool, f64)>,
}

/// Run the HMC the file describes on the processes of `comm`; rank 0 prints a line per
/// trajectory to `out`.
pub fn run_on(
    config: &HmcConfig,
    base: &Path,
    out: &mut dyn Write,
    comm: Arc<dyn Comm>,
) -> Result<HmcReport, CliError> {
    let grid = super::grid_for(config.lattice, config.process_grid, comm.size())?;
    let lat = Lattice::new(config.lattice, grid, comm);
    let root = lat.comm.rank() == 0;
    let mut u = match &config.start {
        StartConfig::File { file, format } => gauge_format(*format)?
            .read(&base.join(file), &lat)
            .map_err(CliError::Gauge)?,
        StartConfig::Cold => cold_start(&lat),
        StartConfig::Hot { seed } => random_gauge(&lat, *seed),
    };
    let st = Stencil::nearest(&lat, [Boundary::Periodic; ND]);
    let plaquette_start = plaquette(&u, &st);
    let io = |e| CliError::Io {
        path: PathBuf::from("<stdout>"),
        source: e,
    };
    let ig = &config.hmc.integrator;
    let scheme = match ig.scheme {
        SchemeConfig::Leapfrog => Scheme::Leapfrog,
        SchemeConfig::Omelyan => Scheme::Omelyan { lambda: ig.lambda },
    };
    let mut hmc = Hmc::new(
        config.terms(&lat, base)?,
        scheme,
        ig.steps.clone(),
        ig.exp_order,
        config.hmc.trajectory_length,
    );
    hmc.metropolis = config.hmc.metropolis;
    let first = config.hmc.first_trajectory;
    let mut rng = Mseries::new(first);
    if root {
        writeln!(
            out,
            "# HMC on {:?}, {} actions, plaquette {plaquette_start:.15}",
            config.lattice,
            config.actions.len()
        )
        .map_err(io)?;
        writeln!(
            out,
            "# trajectory  plaquette  dH  exp(-dH)  accepted  seconds"
        )
        .map_err(io)?;
    }
    let mut report = HmcReport {
        plaquette_start,
        trajectories: Vec::new(),
    };
    let (mut accepted, mut exp_sum) = (0usize, 0.0);
    for k in 0..config.hmc.trajectories {
        let n = first + k as u32 + 1;
        let start = Instant::now();
        let t = hmc.trajectory(&mut u, &mut rng);
        accepted += usize::from(t.accepted);
        exp_sum += (-t.delta_h).exp();
        if root {
            writeln!(
                out,
                "{n:8}  {:.15}  {:+.6e}  {:.6}  {}  {:.1}",
                t.plaquette,
                t.delta_h,
                (-t.delta_h).exp(),
                u8::from(t.accepted),
                start.elapsed().as_secs_f64()
            )
            .map_err(io)?;
        }
        report
            .trajectories
            .push((n, t.delta_h, t.accepted, t.plaquette));
        if let (Some(every), Some(file)) = (config.output.save_every, &config.output.file) {
            if (k + 1) % every == 0 || k + 1 == config.hmc.trajectories {
                let path = base.join(file.replace("{n}", &format!("{n:06}")));
                // Every process takes part; rank 0 writes the file.
                gauge_format(config.output.format)?
                    .write(&path, &u)
                    .map_err(CliError::Gauge)?;
            }
        }
    }
    if root {
        let n = config.hmc.trajectories.max(1) as f64;
        writeln!(
            out,
            "# acceptance {:.3}, <exp(-dH)> {:.4}",
            accepted as f64 / n,
            exp_sum / n
        )
        .map_err(io)?;
    }
    Ok(report)
}
