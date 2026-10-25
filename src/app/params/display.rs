//! What will run, in a few lines.

use super::*;

/// What will run, in a few lines: printed at the start so that the effect of the defaults
/// and of `operator_type` is visible.
impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (m, fo, s) = (&self.main, &self.fopr, &self.solver);
        writeln!(
            f,
            "lattice {:?} on {}{}",
            m.lattice_size,
            m.backend,
            if m.backend == Backend::Simd {
                format!(" ({} lanes)", m.simd_lanes)
            } else {
                String::new()
            }
        )?;
        let shown = |x: Option<f64>| x.map_or("?".to_string(), |x| x.to_string());
        match fo.fermion_type {
            FermionType::DomainWall => writeln!(
                f,
                "{}: Ls = {}, M5 = {}, m = {}, m_PV = {}, b = {}, c = {}, α = {} (D_PV: {})",
                fo.fermion_type,
                fo.extent_of_5th_dimension
                    .map_or("?".to_string(), |l| l.to_string()),
                shown(fo.domain_wall_height),
                shown(fo.quark_mass),
                fo.quark_mass_pauli_villars,
                fo.coefficient_b,
                fo.coefficient_c,
                fo.neff_alpha,
                fo.neff_alpha_pauli_villars
            )?,
            FermionType::Wilson => writeln!(
                f,
                "{}: κ = {}",
                fo.fermion_type,
                shown(fo.hopping_parameter)
            )?,
            FermionType::Clover => writeln!(
                f,
                "{}: κ = {}, c_SW = {}",
                fo.fermion_type,
                shown(fo.hopping_parameter),
                shown(fo.clover_coefficient)
            )?,
        }
        for (k, q) in self.more_quarks().into_iter().enumerate() {
            writeln!(
                f,
                "quark {}: {}, κ = {}{}; {} source at {:?}",
                k + 2,
                q.fopr.fermion_type,
                shown(q.fopr.hopping_parameter),
                q.fopr
                    .clover_coefficient
                    .map_or(String::new(), |c| format!(", c_SW = {c}")),
                q.source.source_type,
                q.source.source_position
            )?;
        }
        if let Some(g) = &self.gauge_fixing {
            writeln!(
                f,
                "{} gauge fixing to {:e}",
                g.gauge_fixing_type, g.convergence_criterion_squared
            )?;
        }
        let operator = self.operator();
        let cg = s.solver_type == SolverType::Cg;
        write!(
            f,
            "{} on {}{}, to |r|²/|b|² < {:e}",
            s.solver_type,
            operator,
            match operator {
                OperatorType::PvRight => " (A′, α cancels)",
                OperatorType::PvLeft => " (left-preconditioned A′)",
                OperatorType::NormalLeft => " (CGNE)",
                OperatorType::EvenOdd if cg => " (the Schur complement on the even sites, CGNE)",
                OperatorType::EvenOdd => " (the Schur complement on the even sites)",
                OperatorType::Lexical if cg => " (D on the whole lattice, CGNE)",
                OperatorType::Lexical => " (D on the whole lattice)",
            },
            s.convergence_criterion_squared
        )?;
        if let Some(drop) = s.mixed() {
            write!(
                f,
                ", in mixed precision (CG in single precision, reliable updates in double precision at each fall of |r|² by {drop:e})"
            )?;
        }
        match (s.preconditioner_type, &self.multigrid) {
            (PreconditionerType::Multigrid, Some(mg)) if !fo.fermion_type.is_domain_wall() => {
                let (l, sm) = (&mg.level1, &mg.smoother);
                let coarse = l.solver_d(false);
                let smoother = match sm.smoother_type {
                    Some(SmootherType::Gmres) => {
                        format!("GMRES({})", sm.number_of_steps.unwrap_or(4))
                    }
                    _ => format!(
                        "SAP {} cycles × {} steps",
                        sm.number_of_cycles.unwrap_or(4),
                        sm.number_of_steps.unwrap_or(6)
                    ),
                };
                write!(
                    f,
                    "\npreconditioned by the multigrid of the whole lattice: {:?} blocks, {} test vectors, {} setup iterations; coarse {} to {:e} in {}; smoother {smoother}",
                    l.block_size,
                    l.test_vectors(false),
                    mg.setup.rounds(false),
                    coarse.solver_type,
                    coarse.convergence_criterion_squared,
                    mg.coarse_precision(),
                )
            }
            (PreconditionerType::Multigrid, Some(mg)) => {
                let l = &mg.level1;
                write!(
                    f,
                    "\npreconditioned by the multigrid: {:?} blocks, {} test vectors; coarse {} (P†DP) to {:e} and {} ({}) to {:e} in {}, above the coarse lattice in {}; smoother Chebyshev {}/{}",
                    l.block_size,
                    l.test_vectors(true),
                    l.solver_d(true).solver_type,
                    l.solver_d(true).convergence_criterion_squared,
                    l.solver_pv().solver_type,
                    if operator == OperatorType::NormalLeft { "P†D†P" } else { "P†D_PV†P" },
                    l.solver_pv().convergence_criterion_squared,
                    mg.coarse_precision(),
                    mg.fine_precision(),
                    mg.smoother.number_of_steps.unwrap_or(cheb_steps()),
                    mg.smoother.interval_ratio
                )?;
                let levels = mg.super_blocks();
                if !levels.is_empty() {
                    write!(
                        f,
                        "\nlevels below the coarse lattice for P†DP: super-blocks {levels:?} (V-cycle, the last solved exactly)"
                    )?;
                    if mg.levels_for_pauli_villars {
                        write!(f, "; for P†D_PV†P too (|H|⁻¹, added to the blocks)")?;
                    }
                }
                Ok(())
            }
            (PreconditionerType::Chebyshev, _) => {
                let c = s.chebyshev_settings();
                write!(
                    f,
                    "\npreconditioned by Chebyshev {}/{}",
                    c.number_of_steps, c.interval_ratio
                )
            }
            _ => write!(f, "\nno preconditioner"),
        }?;
        if let Some(src) = &self.source {
            write!(
                f,
                "\n{} source at {:?}, {} columns per solve: meson correlators{}",
                src.source_type,
                src.source_position,
                src.number_of_rhs_per_solve,
                if fo.fermion_type.is_domain_wall() {
                    " and the residual mass"
                } else {
                    ""
                }
            )?;
            let m = &self.measure;
            if !m.momenta.is_empty() {
                write!(f, ", also at the momenta n = {:?}", m.momenta)?;
            }
            if m.proton {
                write!(f, "; the proton")?;
            }
            if let Some(t) = &m.three_point {
                write!(
                    f,
                    "; a three-point function: sink {} {} time slices after the source, currents {:?} on {} at the momenta {:?}",
                    t.sink,
                    t.t_sink,
                    t.currents,
                    match t.current_on {
                        crate::app::config::CurrentOn::U => "u",
                        crate::app::config::CurrentOn::D => "d",
                    },
                    t.momenta
                )?;
            }
        }
        for note in &self.notes {
            write!(f, "\nnote: {note}")?;
        }
        Ok(())
    }
}
