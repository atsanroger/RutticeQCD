//! Reading a parameter file, checking it, and what it asks the program to run.

use super::*;

/// Whether a file is a parameter file in the style of Bridge++: one with a `Main` section.
pub fn is_bridge_style(path: &Path) -> Result<bool, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(match path.extension().and_then(|e| e.to_str()) {
        Some("toml") => text
            .parse::<toml::Table>()
            .is_ok_and(|t| t.contains_key("Main")),
        _ => serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&text)
            .ok()
            .and_then(|v| v.as_mapping().map(|m| m.contains_key("Main")))
            .unwrap_or(false),
    })
}

impl Params {
    /// The parameters written as YAML. A `MultiGrid_file` is not read: that needs the
    /// directory of the file (`load`).
    pub fn from_yaml(text: &str) -> Result<Self, CliError> {
        let params = serde_yaml_ng::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        Self::checked(params)
    }

    /// The parameters written as TOML; see `from_yaml`.
    pub fn from_toml(text: &str) -> Result<Self, CliError> {
        let params = toml::from_str(text).map_err(|e| CliError::Parse(e.to_string()))?;
        Self::checked(params)
    }

    fn checked(mut params: Self) -> Result<Self, CliError> {
        if params.multigrid.is_none() {
            if let Some(file) = &params.multigrid_file {
                return Err(invalid(format!(
                    "MultiGrid_file {file:?} is read relative to the parameter file: load the file itself"
                )));
            }
        }
        params.validate()?;
        Ok(params)
    }

    /// Read a parameter file; the format is chosen by the file extension.
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(|e| CliError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let params: Self = match path.extension().and_then(|e| e.to_str()) {
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
        let mut params = params;
        if let Some(file) = params.multigrid_file.clone() {
            if params.multigrid.is_some() {
                params.notes.push(format!(
                    "both a MultiGrid block and MultiGrid_file {file:?}; the block is used"
                ));
            } else {
                let file = path.parent().map_or(file.clone(), |d| d.join(&file));
                let text = std::fs::read_to_string(&file).map_err(|e| CliError::Io {
                    path: file.clone(),
                    source: e,
                })?;
                let block: MultigridFile = match file.extension().and_then(|e| e.to_str()) {
                    Some("toml") => {
                        toml::from_str(&text).map_err(|e| CliError::Parse(e.to_string()))?
                    }
                    _ => serde_yaml_ng::from_str(&text)
                        .map_err(|e| CliError::Parse(e.to_string()))?,
                };
                params.multigrid = Some(block.multigrid);
            }
        }
        params.validate()?;
        Ok(params)
    }

    /// What was given but is not used, worth a line in the output.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// Checks what the types cannot; what is only odd goes to the notes.
    pub fn validate(&mut self) -> Result<(), CliError> {
        let mut notes = std::mem::take(&mut self.notes);
        let result = self.check(&mut notes);
        self.notes = notes;
        result
    }

    fn check(&self, notes: &mut Vec<String>) -> Result<(), CliError> {
        if self.bridge_smear.is_some()
            || self.bridge_director_smear.is_some()
            || self.bridge_projection.is_some()
        {
            return Err(invalid(
                "Smear / Director_Smear / Projection: \"smear\" alone does not say what is smeared; gauge (link) smearing is the block GaugeSmearing (smear_type: Stout or APE, rho_uniform, number_of_smearing); the smearing of sources (operator smearing) is Source.source_type".into(),
            ));
        }
        self.check_common()?;
        if !self.fopr.fermion_type.is_domain_wall() {
            return self.check_wilson(notes);
        }
        let f = &self.fopr;
        for (key, missing) in [
            ("quark_mass", f.quark_mass.is_none()),
            ("domain_wall_height", f.domain_wall_height.is_none()),
            (
                "extent_of_5th_dimension",
                f.extent_of_5th_dimension.is_none(),
            ),
        ] {
            if missing {
                return Err(invalid(format!(
                    "Fopr.{key} is needed for domain-wall fermions"
                )));
            }
        }
        for (key, given) in [
            ("hopping_parameter", f.hopping_parameter.is_some()),
            ("clover_coefficient", f.clover_coefficient.is_some()),
        ] {
            if given {
                notes.push(format!(
                    "Fopr.{key} is for Wilson and clover fermions; not used"
                ));
            }
        }
        match self.main.backend {
            Backend::Cpu => {}
            Backend::Simd => {
                let lanes = self.main.simd_lanes;
                if ![2, 4, 8].contains(&lanes) {
                    return Err(invalid(format!("Main.simd_lanes: 2, 4 or 8, not {lanes}")));
                }
                let cut = crate::Blocks::new(self.main.lattice_size, lanes);
                if cut.as_ref().and_then(|b| b.even_odd_index()).is_none() {
                    return Err(invalid(format!(
                        "Main.simd_lanes: the lattice {:?} cannot be cut into {lanes} sub-lattices whose lattice of blocks has even extents (needed for even-odd); use fewer lanes",
                        self.main.lattice_size
                    )));
                }
            }
            Backend::Gpu => {
                return Err(invalid(
                    "Main.backend: gpu is not implemented yet; use cpu".into(),
                ))
            }
        }
        let s = &self.solver;
        s.check_preconditioner("Solver", self.multigrid.is_some(), notes)?;
        let operator = s.operator();
        if matches!(operator, OperatorType::EvenOdd | OperatorType::Lexical) {
            return Err(invalid(format!(
                "Solver.operator_type {operator} is for Wilson and clover fermions; use PV_right, PV_left or Normal_left"
            )));
        }
        match (s.solver_type, operator) {
            (SolverType::Cg, OperatorType::PvRight | OperatorType::PvLeft) => {
                return Err(invalid(format!(
                    "Solver: CG needs a Hermitian positive operator; {operator} is not (use operator_type: Normal_left, which is CGNE)"
                )))
            }
            (SolverType::Minres, _) => {
                return Err(invalid(
                    "Solver: MINRES is for the second coarse factor only".into(),
                ))
            }
            (SolverType::Cg, _) if s.preconditioner_type == PreconditionerType::Multigrid => {
                return Err(invalid(
                    "Solver: the multigrid cycle is not a fixed symmetric map, which CG needs; use GCR on Normal_left".into(),
                ))
            }
            _ => {}
        }
        if let Some(mg) = self
            .multigrid
            .as_ref()
            .filter(|_| s.preconditioner_type == PreconditionerType::Multigrid)
        {
            let normal = operator == OperatorType::NormalLeft;
            let level = &mg.level1;
            if let Some(t) = mg
                .smoother
                .smoother_type
                .filter(|t| *t != SmootherType::Chebyshev)
            {
                return Err(invalid(format!(
                    "MultiGrid.Smoother: {t} is for Wilson and clover fermions; the domain-wall smoother is Chebyshev"
                )));
            }
            level.solver_d(true).check_coarse(
                "MultiGrid.MultiGrid_Level1.Coarse_Solver_D",
                false,
                normal,
            )?;
            level.solver_pv().check_coarse(
                "MultiGrid.MultiGrid_Level1.Coarse_Solver_PV",
                true,
                normal,
            )?;
            let n = level.test_vectors(true);
            if !n.is_multiple_of(2) || n / 2 > mg.setup.filter_number_of_vectors {
                return Err(invalid(
                    "MultiGrid.MultiGrid_Level1: number_of_test_vectors must be even and at most twice Setup.filter_number_of_vectors".into(),
                ));
            }
            let levels = mg.super_blocks();
            let given = [&mg.level2, &mg.level3, &mg.level4]
                .iter()
                .filter(|l| l.is_some())
                .count();
            if given != levels.len() {
                return Err(invalid(
                    "MultiGrid: MultiGrid_Level3 needs MultiGrid_Level2, and MultiGrid_Level4 needs MultiGrid_Level3".into(),
                ));
            }
            let mut ext: Vec<usize> = (0..ND)
                .map(|mu| self.main.lattice_size[mu] / level.block_size[mu])
                .collect();
            for (k, &sb) in levels.iter().enumerate() {
                if sb == 0 || ext.iter().any(|e| e % sb != 0) {
                    return Err(invalid(format!(
                        "MultiGrid.MultiGrid_Level{}: super_block_size {sb} does not divide the extent {ext:?} of level {}",
                        k + 2,
                        k + 1
                    )));
                }
                ext.iter_mut().for_each(|e| *e /= sb);
            }
        }
        Ok(())
    }

    /// What every fermion needs: the gamma matrices, the boundary, the source.
    fn check_common(&self) -> Result<(), CliError> {
        if self.fopr.gamma_matrix_type != GammaType::Dirac {
            return Err(invalid(
                "Fopr.gamma_matrix_type: only Dirac is implemented".into(),
            ));
        }
        let domain_wall = self.fopr.fermion_type.is_domain_wall();
        if let Some(b) = self
            .fopr
            .boundary_condition
            .iter()
            .find(|b| b.abs() > 1 || (domain_wall && **b == 0))
        {
            return Err(invalid(format!(
                "Fopr.boundary_condition: 1 (periodic) or -1 (antiperiodic){}, not {b}",
                if domain_wall { "" } else { " or 0 (Dirichlet)" }
            )));
        }
        if domain_wall {
            if self.quark_2.is_some() || self.quark_3.is_some() || self.quark_4.is_some() {
                return Err(invalid(
                    "Quark_2 … Quark_4 are for Wilson and clover runs; a domain-wall run has one quark".into(),
                ));
            }
            if self.gauge_fixing.is_some() {
                return Err(invalid("GaugeFixing is for Wilson and clover runs".into()));
            }
            if self
                .source
                .is_some_and(|s| s.source_type != SourceType::Local)
            {
                return Err(invalid(
                    "Source.source_type: domain-wall runs take a Local source".into(),
                ));
            }
        }
        if self.main.backend == Backend::Gpu {
            return Err(invalid(
                "Main.backend: gpu is not implemented yet; use cpu".into(),
            ));
        }
        if let Some(src) = &self.source {
            self.check_source(src, "Source")?;
        }
        Ok(())
    }

    /// A source: inside the lattice, with what its type needs.
    fn check_source(&self, src: &SourceBlock, name: &str) -> Result<(), CliError> {
        let n = src.number_of_rhs_per_solve;
        if !(1..=12).contains(&n) {
            return Err(invalid(format!(
                "{name}.number_of_rhs_per_solve: 1 to 12, not {n}"
            )));
        }
        if src
            .source_position
            .iter()
            .zip(&self.main.lattice_size)
            .any(|(x, l)| x >= l)
        {
            return Err(invalid(format!(
                "{name}.source_position {:?} is outside the lattice {:?}",
                src.source_position, self.main.lattice_size
            )));
        }
        if src.source_type == SourceType::Exponential
            && (src.slope.is_none() || src.power.is_none())
        {
            return Err(invalid(format!(
                "{name}: an Exponential source needs slope and power (exp(-slope r^power))"
            )));
        }
        Ok(())
    }

    /// What a Wilson or clover quark needs of its `Fopr`.
    fn check_wilson_fopr(f: &FoprBlock, at: &str, notes: &mut Vec<String>) -> Result<(), CliError> {
        let name = f.fermion_type;
        if name.is_domain_wall() {
            return Err(invalid(format!(
                "{at}.fermion_type: the quarks of one run are Wilson or clover fermions, not {name}"
            )));
        }
        if f.gamma_matrix_type != GammaType::Dirac {
            return Err(invalid(format!(
                "{at}.gamma_matrix_type: only Dirac is implemented"
            )));
        }
        if let Some(b) = f.boundary_condition.iter().find(|b| b.abs() > 1) {
            return Err(invalid(format!(
                "{at}.boundary_condition: 1 (periodic), -1 (antiperiodic) or 0 (Dirichlet), not {b}"
            )));
        }
        if f.hopping_parameter.is_none() {
            return Err(invalid(format!(
                "{at}.hopping_parameter is needed for {name} fermions"
            )));
        }
        match (name, f.clover_coefficient.is_some()) {
            (FermionType::Clover, false) => {
                return Err(invalid(format!(
                    "{at}.clover_coefficient is needed for clover fermions"
                )))
            }
            (FermionType::Wilson, true) => notes.push(format!(
                "{at}.clover_coefficient is not used by Wilson fermions (fermion_type: Clover uses it)"
            )),
            _ => {}
        }
        for (key, given) in [
            ("quark_mass", f.quark_mass.is_some()),
            ("domain_wall_height", f.domain_wall_height.is_some()),
            (
                "extent_of_5th_dimension",
                f.extent_of_5th_dimension.is_some(),
            ),
            ("neff_alpha", f.neff_alpha != 1.0),
        ] {
            if given {
                notes.push(format!("{at}.{key} is for domain-wall fermions; not used"));
            }
        }
        Ok(())
    }

    /// Wilson and clover fermions: what the program of `ruttice::app::run` needs, which
    /// is then checked again there (lattice, blocks, SIMD widths, processes).
    fn check_wilson(&self, notes: &mut Vec<String>) -> Result<(), CliError> {
        let name = self.fopr.fermion_type;
        Self::check_wilson_fopr(&self.fopr, "Fopr", notes)?;
        if (self.quark_3.is_some() && self.quark_2.is_none())
            || (self.quark_4.is_some() && self.quark_3.is_none())
        {
            return Err(invalid(
                "Quark_3 needs Quark_2, and Quark_4 needs Quark_3".into(),
            ));
        }
        for (k, q) in self.more_quarks().into_iter().enumerate() {
            let at = format!("Quark_{}", k + 2);
            Self::check_wilson_fopr(&q.fopr, &format!("{at}.Fopr"), notes)?;
            self.check_source(&q.source, &format!("{at}.Source"))?;
        }
        if self.source.is_none() {
            return Err(invalid(format!(
                "Source: {name} runs solve for the propagator of a source; add a Source block"
            )));
        }
        let s = &self.solver;
        s.check_preconditioner("Solver", self.multigrid.is_some(), notes)?;
        match s.solver_type {
            SolverType::Cg | SolverType::BiCgStab | SolverType::Fgmres => {}
            other => {
                return Err(invalid(format!(
                    "Solver: {other} is not available for {name} fermions; use CG (on the normal equations), BiCGStab or FGMRES"
                )))
            }
        }
        if let Some(o) = s
            .operator_type
            .filter(|o| !matches!(o, OperatorType::EvenOdd | OperatorType::Lexical))
        {
            return Err(invalid(format!(
                "Solver.operator_type {o} is for domain-wall fermions; use Even_odd or Lexical"
            )));
        }
        match s.preconditioner_type {
            PreconditionerType::Chebyshev => {
                return Err(invalid(format!(
                    "Solver: the Chebyshev preconditioner is for domain-wall fermions; use MG or None for {name}"
                )))
            }
            PreconditionerType::Multigrid if s.solver_type == SolverType::Cg => {
                return Err(invalid(
                    "Solver: the multigrid needs a flexible solver: BiCGStab or FGMRES".into(),
                ))
            }
            _ => {}
        }
        if let Some(mg) = self
            .multigrid
            .as_ref()
            .filter(|_| s.preconditioner_type == PreconditionerType::Multigrid)
        {
            if mg.smoother.smoother_type == Some(SmootherType::Chebyshev) {
                return Err(invalid(format!(
                    "MultiGrid.Smoother: Chebyshev is the domain-wall smoother; use SAP or GMRES for {name}"
                )));
            }
            match mg.level1.solver_d(false).solver_type {
                SolverType::BiCgStab | SolverType::Fgmres => {}
                other => {
                    return Err(invalid(format!(
                        "MultiGrid.MultiGrid_Level1.Coarse_Solver_D: {other} is not available for {name}; use BiCGStab or GMRES"
                    )))
                }
            }
            if mg.level1.coarse_pv.is_some() {
                notes.push(format!(
                    "MultiGrid.MultiGrid_Level1.Coarse_Solver_PV is for domain-wall fermions; not used for {name}"
                ));
            }
            if mg.level4.is_some() {
                return Err(invalid(format!(
                    "MultiGrid.MultiGrid_Level4: {name} fermions take levels down to MultiGrid_Level3"
                )));
            }
            if mg.level3.is_some() && mg.level2.is_none() {
                return Err(invalid(
                    "MultiGrid: MultiGrid_Level3 needs MultiGrid_Level2".into(),
                ));
            }
            let mut ext: Vec<usize> = (0..ND)
                .map(|mu| self.main.lattice_size[mu] / mg.level1.block_size[mu].max(1))
                .collect();
            for (k, sb) in mg.super_blocks().into_iter().enumerate() {
                if sb == 0 || ext.iter().any(|e| e % sb != 0) {
                    return Err(invalid(format!(
                        "MultiGrid.MultiGrid_Level{}: super_block_size {sb} does not divide the extent {ext:?} of level {}",
                        k + 2,
                        k + 1
                    )));
                }
                ext.iter_mut().for_each(|e| *e /= sb);
            }
        }
        Ok(())
    }

    /// Wilson and clover fermions: the parameter file of `ruttice::app::run` this one
    /// describes (the propagator of the source, all meson channels).
    pub fn to_config(&self) -> Config {
        let quark = |f: &FoprBlock, src: &SourceBlock| {
            let boundary = f.boundary_config();
            let kappa = f.hopping_parameter.expect("checked");
            let position = src.source_position;
            QuarkConfig {
                fermion: match f.fermion_type {
                    FermionType::Wilson => FermionConfig::Wilson { kappa, boundary },
                    FermionType::Clover => FermionConfig::Clover {
                        kappa,
                        csw: f.clover_coefficient.expect("checked"),
                        boundary,
                    },
                    FermionType::DomainWall => {
                        unreachable!("domain-wall runs do not go through Config")
                    }
                },
                source: match src.source_type {
                    SourceType::Local => SourceConfig::Point { position },
                    SourceType::Exponential => SourceConfig::Exponential {
                        position,
                        slope: src.slope.expect("checked"),
                        power: src.power.expect("checked"),
                    },
                },
            }
        };
        let s = &self.solver;
        let multigrid = self
            .multigrid
            .as_ref()
            .filter(|_| s.preconditioner_type == PreconditionerType::Multigrid)
            .map(|mg| {
                let (l, sm) = (&mg.level1, &mg.smoother);
                let coarse = l.solver_d(false);
                let gmres = sm.smoother_type == Some(SmootherType::Gmres);
                MultigridConfig {
                    block: l.block_size,
                    test_vectors: l.test_vectors(false),
                    setup_iterations: mg.setup.rounds(false),
                    coarse_solver: match coarse.solver_type {
                        SolverType::Fgmres => CoarseSolverKind::Gmres,
                        _ => CoarseSolverKind::Bicgstab,
                    },
                    coarse_tolerance_squared: coarse.convergence_criterion_squared,
                    coarse_max_iterations: coarse.maximum_number_of_iteration,
                    precondition_coarse_diagonal: l.precondition_diagonal,
                    smoother: if gmres {
                        SmootherKind::Gmres
                    } else {
                        SmootherKind::Sap
                    },
                    pre_smoothing: sm.pre_smoothing,
                    smoother_cycles: sm.number_of_cycles.unwrap_or(4),
                    smoother_block_steps: sm.number_of_steps.unwrap_or(6),
                    smoother_iterations: sm.number_of_steps.unwrap_or(4),
                    smoother_tolerance_squared: sm.convergence_criterion_squared.unwrap_or(1e-3),
                    precision: match mg.coarse_precision {
                        PrecisionType::Single => Precision::Single,
                        PrecisionType::Double => Precision::Double,
                    },
                    seed: mg.setup.seed,
                    level2_super_block: mg.level2.map(|l| l.super_block_size),
                    level3_super_block: mg.level3.map(|l| l.super_block_size),
                }
            });
        let first = quark(&self.fopr, self.source.as_ref().expect("checked"));
        let more = self
            .more_quarks()
            .into_iter()
            .map(|q| quark(&q.fopr, &q.source));
        Config {
            lattice: self.main.lattice_size,
            process_grid: (self.main.grid_size != [1; ND]).then_some(self.main.grid_size),
            gauge: GaugeConfig {
                file: self.gauge.config_filename_input.clone(),
                format: GaugeFormat::from(self.gauge.gauge_config_type_input),
                fixing: self.gauge_fixing.map(|g| g.to_config()),
            },
            gauge_smearing: self.gauge_smearing.map(|g| g.to_config()),
            solver: SolverConfig {
                kind: match s.solver_type {
                    SolverType::Cg => SolverKind::CgNormal,
                    SolverType::BiCgStab => SolverKind::Bicgstab,
                    _ => SolverKind::Fgmres,
                },
                even_odd: s.operator_type != Some(OperatorType::Lexical),
                tolerance_squared: s.convergence_criterion_squared,
                // FGMRES keeps its Krylov space of number_of_kept_directions vectors and
                // restarts until maximum_number_of_iteration iterations are done.
                max_iterations: if s.solver_type == SolverType::Fgmres {
                    s.number_of_kept_directions
                } else {
                    s.maximum_number_of_iteration
                },
                max_restarts: if s.solver_type == SolverType::Fgmres {
                    (s.maximum_number_of_iteration / s.number_of_kept_directions.max(1)).max(1)
                } else {
                    s.maximum_number_of_restart.max(1)
                },
                simd: match self.main.backend {
                    Backend::Simd => self.main.simd_lanes,
                    _ => 1,
                },
                multigrid,
            },
            quark: std::iter::once(first).chain(more).collect(),
            measure: self.measure.clone(),
            output: self.output.clone(),
        }
    }

    /// The quarks after the first (`Quark_2` …), in order.
    pub fn more_quarks(&self) -> Vec<&QuarkBlock> {
        [&self.quark_2, &self.quark_3, &self.quark_4]
            .into_iter()
            .map_while(|q| q.as_ref())
            .collect()
    }

    /// The domain-wall operator of `Fopr` (α = 1; set α with `Mdwf::set_alpha`).
    pub fn mdwf_params(&self, mass: f64) -> MdwfParams {
        let f = &self.fopr;
        let boundary = f.boundary();
        MdwfParams {
            ls: f.ls(),
            m5: f
                .domain_wall_height
                .expect("checked for domain-wall fermions"),
            mass,
            b: f.coefficient_b,
            c: f.coefficient_c,
            boundary,
        }
    }

    /// The operator the outer solver works on: by default `Normal_left` for CG and
    /// `PV_right` otherwise for domain-wall fermions, `Even_odd` for Wilson and clover.
    pub fn operator(&self) -> OperatorType {
        if self.fopr.fermion_type.is_domain_wall() {
            self.solver.operator()
        } else {
            self.solver.operator_type.unwrap_or(OperatorType::EvenOdd)
        }
    }

    /// The even-odd form of the domain-wall operator, and whether D_PV is replaced by D
    /// (CGNE).
    pub fn form(&self) -> (EoForm, bool) {
        match self.solver.operator() {
            OperatorType::PvRight => (EoForm::Right, false),
            OperatorType::PvLeft => (EoForm::Left, false),
            OperatorType::NormalLeft => (EoForm::Left, true),
            other => unreachable!("{other} was refused for domain-wall fermions"),
        }
    }

    /// The settings of the multigrid cycle, setup and bootstrap, when it is the
    /// preconditioner.
    pub fn multigrid(&self) -> Option<(PvEoParams, LowModeParams, BootstrapParams)> {
        if self.solver.preconditioner_type != PreconditionerType::Multigrid {
            return None;
        }
        let mg = self.multigrid.as_ref()?;
        let level = &mg.level1;
        let (coarse_d, coarse_pv) = (level.solver_d(true), level.solver_pv());
        let cycle = PvEoParams {
            chebyshev_steps: mg.smoother.number_of_steps.unwrap_or(cheb_steps()),
            chebyshev_ratio: mg.smoother.interval_ratio,
            coarse_tolerance_sqr: coarse_d.convergence_criterion_squared,
            coarse_max_iterations: coarse_d.maximum_number_of_iteration,
            pv_tolerance_sqr: coarse_pv.convergence_criterion_squared,
            pv_max_iterations: coarse_pv.maximum_number_of_iteration,
            d_solver: coarse_d.krylov(),
            pv_solver: coarse_pv.krylov(),
            ..PvEoParams::default()
        };
        let low = LowModeParams {
            vectors: mg.setup.filter_number_of_vectors,
            rounds: mg.setup.filter_number_of_rounds,
            chunk: mg.setup.filter_number_in_block,
            seed: mg.setup.seed,
        };
        let boot = BootstrapParams {
            rounds: mg.setup.rounds(true),
            steps: mg.setup.bootstrap_number_of_steps,
            cycle,
        };
        Some((cycle, low, boot))
    }
}
