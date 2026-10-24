//! Eigenvalues of fermion operators from an eigensolver file (`params::EigenParams`): Lanczos
//! (`eigen::lanczos`) for the Hermitian modes, Krylov-Schur Arnoldi (`eigen::arnoldi`) for any,
//! on Wilson, clover and domain-wall fermions, printed as a table.

use super::params::{Backend, EigenParams, EigensolverType, FermionType, MultMode, OperatorType};
use super::CliError;
use crate::eigen::{arnoldi, lanczos, resolvent_norms, Resolvent};
use crate::fermion::{
    Bound, Clover, CloverParams, EoForm, FermionOp, Mdwf, MdwfParams, PvEvenOdd, Variant, Wilson,
    WilsonParams,
};
use crate::kernel::Dirac;
use crate::solver::preconditioner::multigrid::random_even;
use crate::{
    random_fifth, random_spinor, Adjoint, Comm, Field, Gauge, Half, Lattice, LinearOp, Normal,
    Parity, Spinor, Vector, C64, ND,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// An operator with its adjoint.
type Dyn<'a, V> = Box<dyn LinearOp<V> + 'a>;

/// The operator of the mode made of `d` (`h`: γ5 D, when the mode is H).
fn of_mode<'a, V: Vector + 'a>(
    mode: MultMode,
    d: impl LinearOp<V> + 'a,
    h: Option<Box<dyn LinearOp<V> + 'a>>,
    like: &V,
) -> Dyn<'a, V> {
    match mode {
        MultMode::D => Box::new(d),
        MultMode::Ddag => Box::new(Adjoint(d)),
        MultMode::DdagD => Box::new(Normal::new(d, like)),
        MultMode::DDdag => Box::new(Normal::new(Adjoint(d), like)),
        MultMode::H => h.expect("H is refused where it is not defined"),
    }
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
    /// σ_min(z − A) at the points of `Pseudospectrum`.
    resolvents: Vec<Resolvent>,
    /// The seconds of the eigenpairs; those of the resolvents.
    seconds: [f64; 2],
}

fn solve<V: Vector>(p: &EigenParams, a: &mut Dyn<'_, V>, start: &V) -> Found {
    let mut a: &mut dyn LinearOp<V> = a.as_mut();
    let e = &p.eigensolver;
    let start_eigen = Instant::now();
    let found = match e.eigensolver_type {
        EigensolverType::Lanczos => {
            let r = lanczos(&mut a, start, e.lanczos());
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
                resolvents: Vec::new(),
                seconds: [0.0; 2],
            }
        }
        EigensolverType::Arnoldi => {
            let r = arnoldi(&mut a, start, e.arnoldi());
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
                resolvents: Vec::new(),
                seconds: [0.0; 2],
            }
        }
    };
    let eigen_seconds = start_eigen.elapsed().as_secs_f64();
    let start_resolvents = Instant::now();
    let resolvents = p.pseudospectrum.as_ref().map_or(Vec::new(), |ps| {
        resolvent_norms(&mut a, &ps.all_points(), start, e.lanczos_singular())
    });
    Found {
        resolvents,
        seconds: [eigen_seconds, start_resolvents.elapsed().as_secs_f64()],
        ..found
    }
}

/// What a backend supplies to an eigensolver run: its operators, and the starting vectors (made
/// on the host, the same on every backend) moved to it. The choice of the operator of the mode
/// and the solve (`found`) are the same for all.
trait EigenBackend {
    /// Fields on all sites, four- or five-dimensional.
    type Whole: Vector;
    /// Fields on the even sites.
    type Half: Vector;
    type Wilson: FermionOp<Self::Whole>;
    type Clover: FermionOp<Self::Whole>;
    type Mdwf: FermionOp<Self::Whole>;

    fn lattice(&self) -> &Arc<Lattice>;
    fn wilson(&self, params: WilsonParams) -> Result<Self::Wilson, CliError>;
    fn clover(&self, params: CloverParams) -> Result<Self::Clover, CliError>;
    /// The domain-wall operator with Neff's `alpha`.
    fn mdwf(&self, params: MdwfParams, alpha: f64) -> Result<Self::Mdwf, CliError>;
    fn whole(&self, x: Field<f64, Spinor>) -> Result<Self::Whole, CliError>;
    /// A random field on the even sites of `d`.
    fn even(&self, d: &Self::Mdwf, seed: u64) -> Result<Self::Half, CliError>;
    /// The Pauli-Villars preconditioned operator of `d` and `pv` on the even sites.
    fn pv_even<'a>(
        &self,
        d: &'a Self::Mdwf,
        pv: &'a Self::Mdwf,
        form: EoForm,
        like: &Self::Half,
    ) -> Dyn<'a, Self::Half>;
}

/// The operator the file names, on the backend `b`, its eigenpairs.
fn found<B: EigenBackend>(p: &EigenParams, b: &B) -> Result<Found, CliError> {
    let e = &p.eigensolver;
    let f = &p.fopr;
    let mode = e.fermion_mult_mode;
    Ok(match f.fermion_type {
        FermionType::DomainWall => {
            let d = b.mdwf(f.mdwf(f.mass()), f.neff_alpha)?;
            match e.operator() {
                OperatorType::Lexical => {
                    let x0 = b.whole(random_fifth(b.lattice(), f.ls(), e.seed))?;
                    let h: Dyn<'_, _> = Box::new(Bound::new(&d, Variant::Gamma5D));
                    let mut a = of_mode(mode, Bound::new(&d, Variant::D), Some(h), &x0);
                    solve(p, &mut a, &x0)
                }
                o => {
                    let pv = b.mdwf(
                        f.mdwf(f.quark_mass_pauli_villars),
                        f.neff_alpha_pauli_villars,
                    )?;
                    let form = if o == OperatorType::PvLeft {
                        EoForm::Left
                    } else {
                        EoForm::Right
                    };
                    let x0 = b.even(&d, e.seed)?;
                    let mut a = of_mode(mode, b.pv_even(&d, &pv, form, &x0), None, &x0);
                    solve(p, &mut a, &x0)
                }
            }
        }
        FermionType::Wilson | FermionType::Clover => {
            let boundary = f.boundary_config().map(Into::into);
            let kappa = f.hopping_parameter.expect("checked");
            let x0 = b.whole(random_spinor(b.lattice(), 1, e.seed))?;
            match f
                .clover_coefficient
                .filter(|_| f.fermion_type == FermionType::Clover)
            {
                Some(csw) => {
                    let op = b.clover(CloverParams {
                        kappa,
                        csw,
                        boundary,
                    })?;
                    let h: Dyn<'_, _> = Box::new(Bound::new(&op, Variant::Gamma5D));
                    let mut a = of_mode(mode, Bound::new(&op, Variant::D), Some(h), &x0);
                    solve(p, &mut a, &x0)
                }
                None => {
                    let op = b.wilson(WilsonParams { kappa, boundary })?;
                    let h: Dyn<'_, _> = Box::new(Bound::new(&op, Variant::Gamma5D));
                    let mut a = of_mode(mode, Bound::new(&op, Variant::D), Some(h), &x0);
                    solve(p, &mut a, &x0)
                }
            }
        }
    })
}
/// The CPU: the fields in the usual layout, the Pauli-Villars operator with its blocks fused
/// (`PvEvenOdd`).
struct OnCpu<'a> {
    lat: &'a Arc<Lattice>,
    u: &'a Field<f64, Gauge>,
}

fn operator_error(e: crate::fermion::GaugeError) -> CliError {
    CliError::Operator {
        quark: 1,
        source: e,
    }
}

impl EigenBackend for OnCpu<'_> {
    type Whole = Field<f64, Spinor>;
    type Half = Field<f64, Spinor, Half>;
    type Wilson = Wilson<f64>;
    type Clover = Clover<f64>;
    type Mdwf = Mdwf<f64, Dirac>;

    fn lattice(&self) -> &Arc<Lattice> {
        self.lat
    }

    fn wilson(&self, params: WilsonParams) -> Result<Wilson<f64>, CliError> {
        let mut op = Wilson::<f64>::new(self.lat, params);
        op.set_gauge(self.u).map_err(operator_error)?;
        Ok(op)
    }

    fn clover(&self, params: CloverParams) -> Result<Clover<f64>, CliError> {
        let mut op = Clover::<f64>::new(self.lat, params);
        op.set_gauge(self.u).map_err(operator_error)?;
        Ok(op)
    }

    fn mdwf(&self, params: MdwfParams, alpha: f64) -> Result<Mdwf<f64, Dirac>, CliError> {
        let mut op = Mdwf::<f64, Dirac>::new(self.lat, params);
        op.set_gauge(self.u).map_err(operator_error)?;
        op.set_alpha(alpha);
        Ok(op)
    }

    fn whole(&self, x: Field<f64, Spinor>) -> Result<Field<f64, Spinor>, CliError> {
        Ok(x)
    }

    fn even(&self, d: &Mdwf<f64, Dirac>, seed: u64) -> Result<Field<f64, Spinor, Half>, CliError> {
        let index = d
            .even_odd_index()
            .ok_or_else(|| CliError::Run("no even-odd decomposition".into()))?;
        let mut x = random_even::<f64>(self.lat, index, d.params().ls, 1, seed);
        x.set_parity(Parity::Even);
        Ok(x)
    }

    fn pv_even<'a>(
        &self,
        d: &'a Mdwf<f64, Dirac>,
        pv: &'a Mdwf<f64, Dirac>,
        form: EoForm,
        like: &Field<f64, Spinor, Half>,
    ) -> Dyn<'a, Field<f64, Spinor, Half>> {
        Box::new(PvEvenOdd::new(d, pv, form, like))
    }
}

/// The device (`Main.backend : cuda_bridge`): every field on it, the Pauli-Villars operator made
/// of the even-odd blocks (`PvRatio`).
#[cfg(feature = "cuda-bridge")]
mod device {
    use super::{CliError, Dyn, EigenBackend};
    use crate::backend::cuda_bridge::{CudaClover, CudaError, CudaMdwf, CudaWilson, DeviceSpinor};
    use crate::fermion::{CloverParams, EoForm, FermionOp, MdwfParams, PvRatio, WilsonParams};
    use crate::solver::preconditioner::multigrid::random_even;
    use crate::{EvenOddIndex, Field, Gauge, Lattice, Parity, Spinor};
    use std::sync::Arc;

    fn device(e: CudaError) -> CliError {
        CliError::Run(format!("Main.backend cuda_bridge: {e}"))
    }

    pub(super) struct OnDevice<'a> {
        lat: &'a Arc<Lattice>,
        u: &'a Field<f64, Gauge>,
    }

    impl<'a> OnDevice<'a> {
        pub(super) fn new(lat: &'a Arc<Lattice>, u: &'a Field<f64, Gauge>) -> Self {
            Self { lat, u }
        }

        fn gauged<O: FermionOp<DeviceSpinor>>(&self, mut op: O) -> Result<O, CliError> {
            op.set_gauge(self.u).map_err(super::operator_error)?;
            Ok(op)
        }
    }

    impl EigenBackend for OnDevice<'_> {
        type Whole = DeviceSpinor;
        type Half = DeviceSpinor;
        type Wilson = CudaWilson;
        type Clover = CudaClover;
        type Mdwf = CudaMdwf;

        fn lattice(&self) -> &Arc<Lattice> {
            self.lat
        }

        fn wilson(&self, params: WilsonParams) -> Result<CudaWilson, CliError> {
            self.gauged(CudaWilson::new(self.lat, params).map_err(device)?)
        }

        fn clover(&self, params: CloverParams) -> Result<CudaClover, CliError> {
            self.gauged(CudaClover::new(self.lat, params).map_err(device)?)
        }

        fn mdwf(&self, params: MdwfParams, alpha: f64) -> Result<CudaMdwf, CliError> {
            let mut op = self.gauged(CudaMdwf::new(self.lat, params).map_err(device)?)?;
            op.set_alpha(alpha);
            Ok(op)
        }

        fn whole(&self, x: Field<f64, Spinor>) -> Result<DeviceSpinor, CliError> {
            DeviceSpinor::from_host(&x).map_err(device)
        }

        fn even(&self, d: &CudaMdwf, seed: u64) -> Result<DeviceSpinor, CliError> {
            let index = EvenOddIndex::new(self.lat);
            let mut x = random_even::<f64>(self.lat, &index, d.params().ls, 1, seed);
            x.set_parity(Parity::Even);
            DeviceSpinor::from_host_half(&x, &index).map_err(device)
        }

        fn pv_even<'b>(
            &self,
            d: &'b CudaMdwf,
            pv: &'b CudaMdwf,
            form: EoForm,
            like: &DeviceSpinor,
        ) -> Dyn<'b, DeviceSpinor> {
            let odd = DeviceSpinor::zeros_half(self.lat, d.params().ls, Parity::Odd)
                .expect("fields on the device");
            Box::new(PvRatio::new(d, pv, form, like, &odd))
        }
    }
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
    let relative = |q: &PathBuf| file.parent().map_or(q.clone(), |d| d.join(q));
    let size = p.main.lattice_size;
    let requested = (p.main.grid_size != [1; ND]).then_some(p.main.grid_size);
    let grid = super::grid_for(size, requested, comm.size())?;
    let lat = Lattice::new(size, grid, comm);
    // Stout smearing on the device for `cuda_bridge`, as for the propagators.
    let smearing = p.gauge_smearing.as_ref();
    let u = match p.main.backend {
        #[cfg(feature = "cuda-bridge")]
        Backend::CudaBridge => super::load_links_on_device(&lat, &p.gauge, smearing, file, out)
            .map_err(|e| CliError::Run(e.to_string()))?,
        #[cfg(not(feature = "cuda-bridge"))]
        Backend::CudaBridge => unreachable!("refused when the file is read"),
        _ => super::load_links(&lat, &p.gauge, smearing, file, out)?,
    };
    let e = &p.eigensolver;
    let found = match p.main.backend {
        #[cfg(feature = "cuda-bridge")]
        Backend::CudaBridge => found(p, &device::OnDevice::new(&lat, &u)),
        #[cfg(not(feature = "cuda-bridge"))]
        Backend::CudaBridge => unreachable!("refused when the file is read"),
        _ => found(p, &OnCpu { lat: &lat, u: &u }),
    }?;
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
        found.seconds[0],
        e.threshold_value.map_or(String::new(), |t| format!(
            "; {} eigenvalues before the threshold {t}",
            found.within
        ))
    )
    .map_err(run)?;
    if !found.resolvents.is_empty() {
        let mut ps = String::from(
            "#    Re z                     Im z                     sigma_min(z - A)         applications\n",
        );
        for r in &found.resolvents {
            ps.push_str(&format!(
                "  z  {:+.16e}  {:+.16e}  {:.16e}  {:6}{}\n",
                r.z.re,
                r.z.im,
                r.sigma_min,
                r.applications,
                if r.converged { "" } else { "  (not converged)" }
            ));
        }
        write!(out, "{ps}").map_err(run)?;
        writeln!(
            out,
            "pseudospectrum: {} points, {} applications of (z - A)^dag (z - A), {:.1} s",
            found.resolvents.len(),
            found
                .resolvents
                .iter()
                .map(|r| r.applications)
                .sum::<usize>(),
            found.seconds[1]
        )
        .map_err(run)?;
        table.push_str(&ps);
    }
    if let Some(path) = &p.output.file {
        let path = relative(path);
        std::fs::write(&path, &table).map_err(|e| CliError::Io { path, source: e })?;
    }
    if let Some(r) = found.resolvents.iter().find(|r| !r.converged) {
        return Err(CliError::Run(format!(
            "sigma_min(z - A) at z = {:+e} {:+e} i did not converge in {} restarts (Eigensolver.maximum_number_of_iteration)",
            r.z.re, r.z.im, e.maximum_number_of_iteration
        )));
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

    fn conf() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/conf_04040408.txt")
    }

    /// The eigenvalues the run of `text` prints.
    fn eigenvalues(text: &str) -> Vec<f64> {
        let p = EigenParams::from_yaml(text).unwrap();
        let mut out = Vec::new();
        run_on(&p, &conf(), &mut out, Arc::new(SingleComm)).unwrap();
        let out = String::from_utf8(out).unwrap();
        let values: Vec<f64> = out
            .lines()
            .filter(|l| {
                l.split_whitespace()
                    .next()
                    .is_some_and(|k| k.parse::<usize>().is_ok())
            })
            .filter_map(|l| l.split_whitespace().nth(1)?.parse().ok())
            .collect();
        assert!(!values.is_empty(), "no eigenvalue in\n{out}");
        values
    }

    /// Bridge++'s test `Eigensolver` (clover, κ 0.12, c_SW 1, H = γ5 D, abs_ascending on the
    /// 4³×8 configuration) on `backend`: its `expected_result`, the eigenvalue of smallest
    /// magnitude, is 0.15203194923405. Fewer vectors than Bridge++'s 20 + 50 keep the test short.
    /// With `Pseudospectrum`, σ_min(0 − H) is the same number.
    fn bridge_eigensolver_test_on(backend: &str) {
        let text = format!(
            "Main:\n  lattice_size : [4, 4, 4, 8]\n  backend : {backend}\nGauge:\n  config_filename_input : {}\nFopr:\n  fermion_type : Clover\n  hopping_parameter : 0.12\n  clover_coefficient : 1.0\nEigensolver:\n  eigensolver_type : IRLanczos\n  fermion_mult_mode : H\n  eigensolver_mode : abs_ascending\n  number_of_wanted_eigenvectors : 4\n  number_of_working_eigenvectors : 24\n  convergence_criterion_squared : 1.0e-24\n  maximum_number_of_iteration : 500\n",
            conf().display()
        );
        let first = eigenvalues(&text)[0];
        assert!((first - 0.15203194923405).abs() < 1e-12, "{first}");
        // H is Hermitian: σ_min(z − H) is the distance from z to its spectrum, at z = 0 the
        // eigenvalue of smallest magnitude.
        let text = format!("{text}Pseudospectrum:\n  points : [[0.0, 0.0]]\n");
        let p = EigenParams::from_yaml(&text).unwrap();
        let mut out = Vec::new();
        run_on(&p, &conf(), &mut out, Arc::new(SingleComm)).unwrap();
        let out = String::from_utf8(out).unwrap();
        let sigma: f64 = out
            .lines()
            .find(|l| l.trim_start().starts_with("z "))
            .and_then(|l| l.split_whitespace().nth(3)?.parse().ok())
            .unwrap_or_else(|| panic!("no sigma_min in\n{out}"));
        assert!((sigma - 0.15203194923405).abs() < 1e-12, "{out}");
    }

    #[test]
    fn bridge_eigensolver_test() {
        bridge_eigensolver_test_on("cpu");
    }

    #[cfg(feature = "cuda-bridge")]
    #[test]
    fn bridge_eigensolver_test_on_the_device() {
        bridge_eigensolver_test_on("cuda_bridge");
    }

    /// The lowest singular values squared of the Pauli-Villars operator in the left form, with
    /// Neff's α, the same on the device as on the CPU.
    #[cfg(feature = "cuda-bridge")]
    #[test]
    fn pauli_villars_modes_on_the_device_are_those_on_the_cpu() {
        let text = |backend: &str| {
            format!(
                "Main:\n  lattice_size : [4, 4, 4, 8]\n  backend : {backend}\nGauge:\n  config_filename_input : {}\nFopr:\n  fermion_type : Mobius\n  quark_mass : 0.01\n  domain_wall_height : 1.8\n  extent_of_5th_dimension : 8\n  coefficient_b : 1.5\n  coefficient_c : 0.5\n  boundary_condition : [1, 1, 1, -1]\n  neff_alpha : 1.3\n  neff_alpha_pauli_villars : 1.3\nEigensolver:\n  eigensolver_type : IRLanczos\n  fermion_mult_mode : DdagD\n  operator_type : PV_left\n  eigensolver_mode : ascending\n  number_of_wanted_eigenvectors : 2\n  number_of_working_eigenvectors : 16\n  convergence_criterion_squared : 1.0e-20\n  maximum_number_of_iteration : 1000\n",
                conf().display()
            )
        };
        let (cpu, device) = (eigenvalues(&text("cpu")), eigenvalues(&text("cuda_bridge")));
        assert_eq!(cpu.len(), device.len());
        for (c, d) in cpu.iter().zip(&device) {
            assert!(
                (c - d).abs() < 1e-12 * c.abs(),
                "cpu {cpu:?}, device {device:?}"
            );
        }
    }
}
