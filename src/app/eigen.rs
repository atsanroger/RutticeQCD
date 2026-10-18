//! Eigenvalues of fermion operators from an eigensolver file (`params::EigenParams`): Lanczos
//! (`eigen::lanczos`) for the Hermitian modes, Krylov-Schur Arnoldi (`eigen::arnoldi`) for any,
//! on Wilson, clover and domain-wall fermions, printed as a table.

use super::params::{EigenParams, EigensolverType, MultMode};
use super::{gauge_file, CliError};
use crate::eigen::{arnoldi, lanczos};
use crate::fermion::{
    Bound, Clover, CloverParams, EoForm, FermionOp, Mdwf, PvEvenOdd, Variant, Wilson, WilsonParams,
};
use crate::field::random::TestRng;
use crate::kernel::Dirac;
use crate::solver::preconditioner::multigrid::random_even;
use crate::{
    random_spinor, Comm, Field, Gauge, Lattice, Lexical, LinearFn, LinearOp, Normal, Parity,
    Spinor, Vector, C64, ND,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// An operator with its adjoint behind a pointer.
struct Dyn<'a, V>(Box<dyn LinearOp<V> + 'a>);

impl<V: Vector> LinearFn<V> for Dyn<'_, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.0.mult(out, inp);
    }
}

impl<V: Vector> LinearOp<V> for Dyn<'_, V> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.0.mult_dag(out, inp);
    }
}

/// A†.
struct Adjoint<A>(A);

impl<V: Vector, A: LinearOp<V>> LinearFn<V> for Adjoint<A> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.0.mult_dag(out, inp);
    }
}

impl<V: Vector, A: LinearOp<V>> LinearOp<V> for Adjoint<A> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.0.mult(out, inp);
    }
}

/// The operator of the mode made of `d` (`h`: γ5 D, when the mode is H).
fn of_mode<'a, V: Vector + 'a>(
    mode: MultMode,
    d: impl LinearOp<V> + 'a,
    h: Option<Box<dyn LinearOp<V> + 'a>>,
    like: &V,
) -> Dyn<'a, V> {
    Dyn(match mode {
        MultMode::D => Box::new(d),
        MultMode::Ddag => Box::new(Adjoint(d)),
        MultMode::DdagD => Box::new(Normal::new(d, like)),
        MultMode::DDdag => Box::new(Normal::new(Adjoint(d), like)),
        MultMode::H => h.expect("H is refused where it is not defined"),
    })
}

/// One eigenpair as printed.
struct Row {
    value: C64,
    residual: f64,
}

/// What a run found.
struct Found {
    rows: Vec<Row>,
    within: usize,
    converged: bool,
    restarts: usize,
    applications: usize,
}

fn solve<V: Vector>(p: &EigenParams, a: &mut Dyn<'_, V>, start: &V) -> Found {
    let e = &p.eigensolver;
    match e.eigensolver_type {
        EigensolverType::Lanczos => {
            let r = lanczos(a, start, e.lanczos());
            Found {
                rows: r
                    .values
                    .iter()
                    .zip(&r.residuals)
                    .map(|(l, res)| Row {
                        value: C64::new(*l, 0.0),
                        residual: *res,
                    })
                    .collect(),
                within: r.within_threshold,
                converged: r.converged,
                restarts: r.restarts,
                applications: r.applications,
            }
        }
        EigensolverType::Arnoldi => {
            let r = arnoldi(a, start, e.arnoldi());
            Found {
                rows: r
                    .values
                    .iter()
                    .zip(&r.residuals)
                    .map(|(l, res)| Row {
                        value: *l,
                        residual: *res,
                    })
                    .collect(),
                within: r.within_threshold,
                converged: r.converged,
                restarts: r.restarts,
                applications: r.applications,
            }
        }
    }
}

/// Five-dimensional random spinors: the spinors of `ls` slices on every site drawn in the
/// order of the whole lattice (`random_spinor` with `ls` reals × 24 per site).
fn random_five(lat: &Arc<Lattice>, ls: usize, seed: u64) -> Field<f64, Spinor, Lexical> {
    let mut x = Field::<f64, Spinor, Lexical>::with_shape(lat, 24, ls, 1, 1);
    let mut rng = TestRng::new(seed);
    let mut reals = vec![0.0; x.site_len()];
    for global in 0..lat.num_global_sites() {
        for r in reals.iter_mut() {
            *r = rng.uniform();
        }
        if let Some(site) = lat.local_site(lat.global_site_coords(global)) {
            x.site_mut(0, site).copy_from_slice(&reals);
        }
    }
    x
}

/// Run an eigensolver file on the processes of `comm`; only rank 0 should be given a real
/// `out`.
pub fn run_on(
    p: &EigenParams,
    file: &Path,
    out: &mut dyn Write,
    comm: Arc<dyn Comm>,
) -> Result<(), CliError> {
    let run = |e: std::io::Error| CliError::Run(e.to_string());
    let operator_error = |e| CliError::Operator {
        quark: 1,
        source: e,
    };
    let relative = |q: &PathBuf| file.parent().map_or(q.clone(), |d| d.join(q));
    let size = p.main.lattice_size;
    let requested = (p.main.grid_size != [1; ND]).then_some(p.main.grid_size);
    let grid = super::grid_for(size, requested, comm.size())?;
    let lat = Lattice::new(size, grid, comm);
    let format = super::config::GaugeFormat::from(p.gauge.gauge_config_type_input);
    let u: Field<f64, Gauge> = gauge_file(format)?
        .read(&relative(&p.gauge.config_filename_input), &lat)
        .map_err(CliError::Gauge)?;
    let u = match &p.gauge_smearing {
        None => u,
        Some(g) => {
            let st = crate::Stencil::nearest(&lat, [crate::Boundary::Periodic; ND]);
            let v = g.to_config().smearing().apply(&u);
            writeln!(
                out,
                "gauge smearing {:?}: plaquette {:.15} → {:.15}",
                g.to_config(),
                crate::gauge::plaquette(&u, &st),
                crate::gauge::plaquette(&v, &st)
            )
            .map_err(run)?;
            v
        }
    };
    let e = &p.eigensolver;
    let f = &p.fopr;
    let mode = e.fermion_mult_mode;
    let start = Instant::now();
    let found = match f.fermion_type {
        crate::app::params::FermionType::DomainWall => {
            let make = |mass: f64, alpha: f64| -> Result<Mdwf<f64, Dirac>, CliError> {
                let mut op = Mdwf::<f64, Dirac>::new(&lat, f.mdwf(mass));
                op.set_gauge(&u).map_err(operator_error)?;
                op.set_alpha(alpha);
                Ok(op)
            };
            let d = make(f.mass(), f.neff_alpha)?;
            match e.operator() {
                crate::app::params::OperatorType::Lexical => {
                    let x0 = random_five(&lat, f.ls(), e.seed);
                    let h: Box<dyn LinearOp<_>> = Box::new(Bound::new(&d, Variant::Gamma5D));
                    let mut a = of_mode(mode, Bound::new(&d, Variant::D), Some(h), &x0);
                    solve(p, &mut a, &x0)
                }
                o => {
                    let pv = make(f.quark_mass_pauli_villars, f.neff_alpha_pauli_villars)?;
                    let form = if o == crate::app::params::OperatorType::PvLeft {
                        EoForm::Left
                    } else {
                        EoForm::Right
                    };
                    let index = d
                        .even_odd_index()
                        .ok_or_else(|| CliError::Run("no even-odd decomposition".into()))?
                        .clone();
                    let mut x0 = random_even::<f64>(&lat, &index, f.ls(), 1, e.seed);
                    x0.set_parity(Parity::Even);
                    let mut a = of_mode(mode, PvEvenOdd::new(&d, &pv, form, &x0), None, &x0);
                    solve(p, &mut a, &x0)
                }
            }
        }
        crate::app::params::FermionType::Wilson | crate::app::params::FermionType::Clover => {
            let boundary = f.boundary_config().map(Into::into);
            let kappa = f.hopping_parameter.expect("checked");
            let x0 = random_spinor(&lat, 1, e.seed);
            match f
                .clover_coefficient
                .filter(|_| f.fermion_type == crate::app::params::FermionType::Clover)
            {
                Some(csw) => {
                    let mut op = Clover::<f64>::new(
                        &lat,
                        CloverParams {
                            kappa,
                            csw,
                            boundary,
                        },
                    );
                    op.set_gauge(&u).map_err(operator_error)?;
                    let h: Box<dyn LinearOp<_>> = Box::new(Bound::new(&op, Variant::Gamma5D));
                    let mut a = of_mode(mode, Bound::new(&op, Variant::D), Some(h), &x0);
                    solve(p, &mut a, &x0)
                }
                None => {
                    let mut op = Wilson::<f64>::new(&lat, WilsonParams { kappa, boundary });
                    op.set_gauge(&u).map_err(operator_error)?;
                    let h: Box<dyn LinearOp<_>> = Box::new(Bound::new(&op, Variant::Gamma5D));
                    let mut a = of_mode(mode, Bound::new(&op, Variant::D), Some(h), &x0);
                    solve(p, &mut a, &x0)
                }
            }
        }
    };
    let seconds = start.elapsed().as_secs_f64();
    let mut table = String::new();
    let complex = e.eigensolver_type == EigensolverType::Arnoldi;
    table.push_str(if complex {
        "#    k  Re lambda               Im lambda               |A x - lambda x|^2\n"
    } else {
        "#    k  lambda                  |A x - lambda x|^2\n"
    });
    for (k, r) in found.rows.iter().enumerate() {
        let mark = match (
            k == found.within,
            r.residual * r.residual < e.convergence_criterion_squared,
        ) {
            (true, true) => "  (first at the threshold)",
            (true, false) => "  (first at the threshold; not converged)",
            (false, true) => "",
            (false, false) => "  (not converged)",
        };
        if complex {
            table.push_str(&format!(
                "  {k:4}  {:+.16e}  {:+.16e}  {:.6e}{mark}\n",
                r.value.re,
                r.value.im,
                r.residual * r.residual
            ));
        } else {
            table.push_str(&format!(
                "  {k:4}  {:+.16e}  {:.6e}{mark}\n",
                r.value.re,
                r.residual * r.residual
            ));
        }
    }
    write!(out, "{table}").map_err(run)?;
    writeln!(
        out,
        "{} {} restarts, {} applications of the operator, {:.1} s{}",
        if found.converged {
            "converged in"
        } else {
            "NOT converged after"
        },
        found.restarts,
        found.applications,
        seconds,
        e.threshold_value.map_or(String::new(), |t| format!(
            "; {} eigenvalues before the threshold {t}",
            found.within
        ))
    )
    .map_err(run)?;
    if let Some(path) = &p.output.file {
        let path = relative(path);
        std::fs::write(&path, &table).map_err(|e| CliError::Io { path, source: e })?;
    }
    if !found.converged {
        return Err(CliError::Run(format!(
            "the eigensolver did not converge in {} restarts (Eigensolver.maximum_number_of_iteration)",
            found.restarts
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SingleComm;

    /// Bridge++'s test `Eigensolver` (clover, κ 0.12, c_SW 1, H = γ5 D, abs_ascending on the
    /// 4³×8 configuration): its `expected_result`, the eigenvalue of smallest magnitude, is
    /// 0.15203194923405. Fewer vectors than Bridge++'s 20 + 50 keep the test short.
    #[test]
    fn bridge_eigensolver_test() {
        let conf = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/conf_04040408.txt");
        let text = format!(
            "Main:\n  lattice_size : [4, 4, 4, 8]\nGauge:\n  config_filename_input : {}\nFopr:\n  fermion_type : Clover\n  hopping_parameter : 0.12\n  clover_coefficient : 1.0\nEigensolver:\n  eigensolver_type : IRLanczos\n  fermion_mult_mode : H\n  eigensolver_mode : abs_ascending\n  number_of_wanted_eigenvectors : 4\n  number_of_working_eigenvectors : 24\n  convergence_criterion_squared : 1.0e-24\n  maximum_number_of_iteration : 500\n",
            conf.display()
        );
        let p = EigenParams::from_yaml(&text).unwrap();
        let mut out = Vec::new();
        run_on(&p, &conf, &mut out, Arc::new(SingleComm)).unwrap();
        let out = String::from_utf8(out).unwrap();
        let first: f64 = out
            .lines()
            .find(|l| l.trim_start().starts_with("0 "))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no eigenvalue in\n{out}"));
        assert!((first - 0.15203194923405).abs() < 1e-12, "{out}");
    }
}
