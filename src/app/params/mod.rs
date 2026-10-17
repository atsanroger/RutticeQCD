//! Parameter files in the style of Bridge++: sections `Main`, `Gauge`, `Fopr` and
//! `Solver`, keys spelt out (`maximum_number_of_iteration`, `convergence_criterion_squared`),
//! read from YAML (`.yaml`, `.yml`) or TOML (`.toml`). The values of choices are keywords
//! read without regard to case or `_` (see [`super::keyword`]); a misspelt key is an error.
//!
//! The multigrid is a section of its own, `MultiGrid`, next to `Solver`, which uses it with
//! `preconditioner_type: MG` and may keep it when it does not (switching is one keyword).
//! It can also come from a file of its own (`MultiGrid_file`), to share one multigrid among
//! many parameter files. `MultiGrid_Level1` is the coarse lattice; it names a solver for each
//! coarse factor, with the keys of `Solver`. `MultiGrid_Level2` to `MultiGrid_Level4`, if
//! given, are levels below it (`super_block_size`) in the solves of the quark factor, and with
//! `levels_for_pauli_villars : true` in those of the second factor too.
//!
//! ```yaml
//! Main:
//!   lattice_size : [12, 12, 12, 12]
//!   backend      : cpu
//! Gauge:
//!   gauge_config_type_input : Text
//!   config_filename_input   : ./conf.txt
//! Fopr:
//!   fermion_type            : Domainwall_5din
//!   quark_mass              : 0.005
//!   domain_wall_height      : 1.0
//!   extent_of_5th_dimension : 8
//!   neff_alpha              : 0.6
//! Solver:
//!   solver_type                   : GCR
//!   operator_type                 : PV_left
//!   convergence_criterion_squared : 1.0e-18
//!   preconditioner_type           : MG
//! MultiGrid:
//!   MultiGrid_Level1:
//!     block_size : [4, 4, 4, 4]
//! ```

mod blocks;
mod convert;
mod display;
mod keywords;

pub use blocks::*;
pub use convert::is_bridge_style;
pub use keywords::*;

use super::config::{
    BoundaryConfig, CoarseSolverKind, Config, FermionConfig, GaugeConfig, GaugeFixingConfig,
    GaugeFormat, GaugeSmearingConfig, MeasureConfig, MultigridConfig, OutputConfig, Precision,
    QuarkConfig, SmootherKind, SolverConfig, SolverKind, SourceConfig,
};
use super::CliError;
use crate::fermion::{EoForm, MdwfParams};
use crate::gauge::CoulombParams;
use crate::keyword_enum;
use crate::solver::preconditioner::multigrid::{
    BootstrapParams, CoarseKrylov, LevelBootstrap, LowModeParams, PvEoParams,
};
use crate::solver::Stop;
use crate::{Boundary, ND};
use serde::Deserialize;
use std::fmt;
use std::path::{Path, PathBuf};

fn invalid(what: String) -> CliError {
    CliError::Invalid(what)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "
Main:
  lattice_size : [12, 12, 12, 12]
Gauge:
  config_filename_input : conf.txt
Fopr:
  fermion_type            : domainwall_5din
  quark_mass              : 0.005
  domain_wall_height      : 1.0
  extent_of_5th_dimension : 8
Solver:
  solver_type                   : cgne
  convergence_criterion_squared : 1.0e-18
";

    const MG: &str = "MultiGrid:\n  MultiGrid_Level1:\n    block_size : [4, 4, 4, 4]\n";

    /// MINIMAL with its solver lines replaced, and the multigrid section appended.
    fn with(solver: &str, mg: bool) -> String {
        let text = MINIMAL.replace("  solver_type                   : cgne\n", solver);
        if mg {
            text + MG
        } else {
            text
        }
    }

    fn parse(text: &str) -> Result<Params, CliError> {
        let mut p: Params =
            serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        p.validate()?;
        Ok(p)
    }

    #[test]
    fn cgne_without_multigrid_needs_no_more() {
        let p = parse(MINIMAL).unwrap();
        assert_eq!(p.solver.operator(), OperatorType::NormalLeft);
        assert_eq!(p.form(), (EoForm::Left, true));
        assert!(p.multigrid().is_none());
        assert!(p.to_string().contains("CG on Normal_left (CGNE)"), "{p}");
    }

    #[test]
    fn multigrid_block_with_defaults_and_keywords_in_any_case() {
        let text = with(
            "  solver_type : gcr\n  operator_type : pv_LEFT\n  preconditioner_type : multi_grid\n",
            true,
        );
        let p = parse(&text).unwrap();
        let (cycle, low, boot) = p.multigrid().unwrap();
        assert_eq!(cycle.d_solver, CoarseKrylov::BiCgStab);
        assert_eq!(cycle.pv_solver, CoarseKrylov::Minres);
        assert_eq!(low.vectors, 16);
        assert_eq!(boot.rounds, 2);
        assert_eq!(p.form(), (EoForm::Left, false));
        assert!(p.multigrid.as_ref().unwrap().super_blocks().is_empty());
    }

    #[test]
    fn levels_below_the_coarse_lattice() {
        let solver = "  solver_type : gcr\n  preconditioner_type : MG\n";
        // 12⁴ in blocks of 4⁴ is 3⁴ coarse sites; 16⁴ is 4⁴.
        let sixteen = |levels: &str| {
            with(solver, true).replace("[12, 12, 12, 12]", "[16, 16, 16, 16]") + levels
        };
        let p = parse(&sixteen(
            "  MultiGrid_Level2:\n    super_block_size : 2\n  MultiGrid_Level3:\n    super_block_size : 2\n",
        ))
        .unwrap();
        assert_eq!(p.multigrid.as_ref().unwrap().super_blocks(), vec![2, 2]);
        assert!(p.to_string().contains("super-blocks [2, 2]"), "{p}");
        let e = parse(&(with(solver, true) + "  MultiGrid_Level2:\n    super_block_size : 2\n"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("does not divide the extent [3, 3, 3, 3]"), "{e}");
        let e = parse(&sixteen("  MultiGrid_Level3:\n    super_block_size : 2\n"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("Level3 needs MultiGrid_Level2"), "{e}");
    }

    #[test]
    fn confusing_combinations_are_explained() {
        let mg_block_unused = with("  solver_type : gcr\n", true);
        // Not an error: switching the multigrid off is one keyword; the block is noted.
        let p = parse(&mg_block_unused).unwrap();
        assert!(p.multigrid().is_none());
        assert!(p.to_string().contains("no preconditioner"), "{p}");
        assert!(
            p.notes()[0].contains("MultiGrid block is not used"),
            "{:?}",
            p.notes()
        );
        let mg_missing = with("  solver_type : gcr\n  preconditioner_type : MG\n", false);
        let e = parse(&mg_missing).unwrap_err().to_string();
        assert!(e.contains("neither a MultiGrid block"), "{e}");
        let cg_on_a = with("  solver_type : CG\n  operator_type : PV_right\n", false);
        let e = parse(&cg_on_a).unwrap_err().to_string();
        assert!(e.contains("Hermitian positive"), "{e}");
        let normal_minres = with(
            "  solver_type : GCR\n  operator_type : normal\n  preconditioner_type : MG\n",
            true,
        );
        let e = parse(&normal_minres).unwrap_err().to_string();
        assert!(e.contains("P†D†P"), "{e}");
        let gpu = MINIMAL.replace(
            "  lattice_size : [12, 12, 12, 12]\n",
            "  lattice_size : [12, 12, 12, 12]\n  backend : GPU\n",
        );
        let e = parse(&gpu).unwrap_err().to_string();
        assert!(e.contains("not implemented"), "{e}");
        let typo = MINIMAL.replace("quark_mass  ", "quark_mas   ");
        assert!(parse(&typo).is_err());
    }

    #[test]
    fn toml_reads_the_same() {
        let text = "
[Main]
lattice_size = [12, 12, 12, 12]
[Gauge]
config_filename_input = \"conf.txt\"
[Fopr]
fermion_type = \"MDWF\"
quark_mass = 0.005
domain_wall_height = 1.0
extent_of_5th_dimension = 8
neff_alpha = 0.6
[Solver]
solver_type = \"BiCGStab\"
convergence_criterion_squared = 1e-18
";
        let mut p: Params = toml::from_str(text).unwrap();
        p.validate().unwrap();
        assert_eq!(p.solver.operator(), OperatorType::PvRight);
        assert_eq!(p.fopr.neff_alpha, 0.6);
    }

    /// One multigrid file shared by two parameter files, one of them YAML, the other TOML.
    #[test]
    fn multigrid_from_a_file_of_its_own() {
        let dir = std::env::temp_dir().join(format!("ruttice_params_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("mg.yaml"),
            "MultiGrid:\n  coarse_precision : DOUBLE\n  MultiGrid_Level1:\n    block_size : [2, 2, 2, 2]\n    Coarse_Solver_PV:\n      solver_type : gcr\n      convergence_criterion_squared : 1.0e-2\n",
        )
        .unwrap();
        let yaml = with("  solver_type : GCR\n  preconditioner_type : MG\n", false)
            + "MultiGrid_file : mg.yaml\n";
        std::fs::write(dir.join("run.yaml"), yaml).unwrap();
        let toml_text = "MultiGrid_file = \"mg.yaml\"\n[Main]\nlattice_size = [12, 12, 12, 12]\n[Gauge]\nconfig_filename_input = \"conf.txt\"\n[Fopr]\nfermion_type = \"mobius\"\nquark_mass = 0.005\ndomain_wall_height = 1.0\nextent_of_5th_dimension = 8\n[Solver]\nsolver_type = \"gcr\"\npreconditioner_type = \"mg\"\nconvergence_criterion_squared = 1e-18\n";
        std::fs::write(dir.join("run.toml"), toml_text).unwrap();
        for name in ["run.yaml", "run.toml"] {
            let p = Params::load(&dir.join(name)).unwrap();
            let (cycle, _, _) = p.multigrid().unwrap();
            assert_eq!(cycle.pv_solver, CoarseKrylov::Gcr { k: 2 }, "{name}");
            let mg = p.multigrid.as_ref().unwrap();
            assert_eq!(mg.coarse_precision(), PrecisionType::Double, "{name}");
            assert_eq!(mg.level1.block_size, [2; ND], "{name}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The precisions in any of their usual spellings and cases.
    #[test]
    fn precision_spellings() {
        use super::keywords::BasisPrecision;
        for w in [
            "single", "Single", "SINGLE", "float", "f32", "FP32", "fp32", "Float32", "SP",
        ] {
            assert_eq!(
                PrecisionType::from_keyword(w),
                Some(PrecisionType::Single),
                "{w}"
            );
        }
        for w in ["double", "Double", "f64", "FP64", "fp_64", "float64", "DP"] {
            assert_eq!(
                PrecisionType::from_keyword(w),
                Some(PrecisionType::Double),
                "{w}"
            );
        }
        for w in ["half", "Half", "f16", "FP16", "float16"] {
            assert_eq!(
                BasisPrecision::from_keyword(w),
                Some(BasisPrecision::Half),
                "{w}"
            );
        }
        assert_eq!(PrecisionType::from_keyword("half"), None);
    }

    #[test]
    fn wilson_and_clover_files() {
        let p = parse(include_str!("../../../examples/clover_mg.yaml")).unwrap();
        assert_eq!(p.operator(), OperatorType::EvenOdd);
        let c = p.to_config();
        assert_eq!(
            c.quark[0].fermion,
            FermionConfig::Clover {
                kappa: 0.12,
                csw: 1.0,
                boundary: [
                    BoundaryConfig::Periodic,
                    BoundaryConfig::Periodic,
                    BoundaryConfig::Periodic,
                    BoundaryConfig::Antiperiodic
                ]
            }
        );
        assert!(c.solver.even_odd);
        assert_eq!(c.solver.kind, SolverKind::Fgmres);
        assert_eq!((c.solver.max_iterations, c.solver.max_restarts), (30, 100));
        let mg = c.solver.multigrid.unwrap();
        assert_eq!(
            (mg.block, mg.test_vectors, mg.setup_iterations),
            ([2; ND], 8, 4)
        );
        assert_eq!(
            (mg.smoother, mg.smoother_cycles, mg.smoother_block_steps),
            (SmootherKind::Sap, 4, 6)
        );
        assert_eq!(
            (mg.coarse_tolerance_squared, mg.coarse_max_iterations),
            (2e-3, 200)
        );

        let wilson = "Main:\n  lattice_size : [4, 4, 4, 8]\nGauge:\n  config_filename_input : c.txt\nFopr:\n  fermion_type : wilson\n  hopping_parameter : 0.13\nSolver:\n  solver_type : cg\n  convergence_criterion_squared : 1.0e-20\nSource:\n  source_position : [0, 0, 0, 0]\n";
        // Gauge smearing is the block GaugeSmearing; Bridge++'s Smear blocks are refused.
        let p = parse(&format!(
            "{wilson}GaugeSmearing:\n  smear_type : stout\n  rho_uniform : 0.1\n  number_of_smearing : 3\n"
        ))
        .unwrap();
        assert_eq!(
            p.to_config().gauge_smearing,
            Some(GaugeSmearingConfig::Stout {
                rho: 0.1,
                steps: 3,
                spatial_only: false
            })
        );
        for block in [
            "Smear:\n  smear_type : APE\n",
            "Director_Smear:\n  number_of_smearing : 1\n",
            "Projection:\n  projection_type : Stout_SU3\n",
        ] {
            let e = parse(&format!("{wilson}{block}")).unwrap_err().to_string();
            assert!(
                e.contains("GaugeSmearing") && e.contains("Source.source_type"),
                "{e}"
            );
        }
        let p = parse(wilson).unwrap();
        let c = p.to_config();
        assert_eq!(c.solver.kind, SolverKind::CgNormal);
        assert!(p.to_string().contains("Wilson: κ = 0.13"), "{p}");
        assert!(p.to_string().contains("CGNE"), "{p}");
        // The Bridge++ defaults of the multigrid when the block names only the blocks.
        let p = parse(
            &(wilson.replace(
                "solver_type : cg",
                "solver_type : bicgstab\n  preconditioner_type : MG",
            ) + "MultiGrid:\n  MultiGrid_Level1:\n    block_size : [2, 2, 2, 2]\n"),
        )
        .unwrap();
        let mg = p.to_config().solver.multigrid.unwrap();
        assert_eq!(
            (
                mg.test_vectors,
                mg.setup_iterations,
                mg.coarse_max_iterations
            ),
            (16, 4, 200)
        );

        let err = |text: String| parse(&text).unwrap_err().to_string();
        let e = err(wilson.replace("  hopping_parameter : 0.13\n", ""));
        assert!(e.contains("hopping_parameter is needed"), "{e}");
        let e = err(wilson.replace("fermion_type : wilson", "fermion_type : clover"));
        assert!(e.contains("clover_coefficient is needed"), "{e}");
        let e = err(wilson.replace("solver_type : cg", "solver_type : gcr"));
        assert!(e.contains("not available for Wilson"), "{e}");
        let e = err(wilson.replace(
            "solver_type : cg",
            "solver_type : cg\n  operator_type : PV_right",
        ));
        assert!(e.contains("for domain-wall fermions"), "{e}");
        let e = err(wilson.replace("Source:\n  source_position : [0, 0, 0, 0]\n", ""));
        assert!(e.contains("add a Source block"), "{e}");
        let p = parse(&wilson.replace(
            "hopping_parameter : 0.13",
            "hopping_parameter : 0.13\n  quark_mass : 0.1",
        ))
        .unwrap();
        assert!(
            p.notes()[0].contains("quark_mass is for domain-wall"),
            "{:?}",
            p.notes()
        );
        // And the other way round.
        let e = err(MINIMAL.replace("  quark_mass              : 0.005\n", ""));
        assert!(e.contains("quark_mass is needed for domain-wall"), "{e}");
    }
}
