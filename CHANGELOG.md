# Changelog

## Unreleased

- The gradient flow (`gauge::flow`): the Wilson flow by Lüscher's third-order Runge-Kutta
  scheme, the energy density (plaquette and clover) and the topological charge (clover and
  improved), as Bridge++ and checked against its tests; `ruttice-flow` and `ruttice-flow-mpi`
  print t0 and the charge of saved configurations.
- HMC measures t0 and the charge by the flow every `GradientFlow.measurement_interval`
  trajectories; `skip_trajectories` (`ruttice-flow --skip`) leaves the thermalization out of
  the analysis.
- The Γ-method of Wolff (`numerics::gamma_method`): the error of a mean and the integrated
  autocorrelation time, printed for t0 and the charges at the end of an HMC run and of
  `ruttice-flow`.
- The force of the plaquette action walks its staples site by site: the same numbers, faster.
- The Chebyshev filter of the test vectors of the domain-wall multigrid no longer overflows in
  single precision: on stout-smeared links its setup panicked ("the subspace lost rank").
- The levels below the coarse lattice of the domain-wall multigrid (`MultiGrid_Level2` and on,
  and `levels_for_pauli_villars`) are on the device as well, the last level of any size
  (cuSOLVER); the solves print the applications of the coarse operators they took. So is
  the exact adjoint of the Pauli-Villars coarse operator (`Coarse_Solver_PV` other than
  MINRES): the whole domain-wall multigrid of the CPU is on the device. On both, that
  adjoint is now an operator of its own (`CoarseOp::adjoint`), solved in its even-odd form on
  the coarse lattice as the quark factor is, not on the whole coarse lattice.
- Stout smearing and the chain rule of its force on the device (`DeviceLinks`,
  `GaugeSmearing::levels_on_device`, `chain_on_device`).
- HMC with `Main.backend : cuda_bridge`: the domain-wall Ratio action with its solves and the
  derivative by the links on the device (`ForceOp` for `CudaMdwf`); the trajectories of Bridge++'s
  test_HMC_Domainwall_Nf2_PV are those of the CPU.
- HMC is written once for links on the host or on a device (`hmc::LinkField`): with
  `Main.backend : cuda_bridge` the links, the momenta, the integrator, the gauge actions and the
  smearing are on the device too (`gauge/links.cuh`), and the links cross over only for the
  flow and to be saved.
- `eigen::lowest`: the lowest eigenpairs of a Hermitian positive operator by thick-restart
  Lanczos, optionally on a Chebyshev polynomial, for fields on the host or on a device;
  `measure::chirality`; `ruttice-low-modes` (feature `cuda-bridge`): the lowest singular
  values of the even-odd Pauli-Villars domain-wall operator on the GPU and the chirality of
  their vectors, the topological near-zero modes. Bridge++ has no counterpart to compare with.
- The even-odd ratio of two flavours over two Pauli-Villars (or Hasenbusch) flavours,
  `hmc::TwoFlavorRatioEo`, as Bridge++'s `Action_F_alt_Ratio_eo`: H = Γ5 K on the even sites,
  K = 1 − T_ee⁻¹ M_eo T_oo⁻¹ M_oe (`fermion::SchurFactor`, the left form). Wilson and domain-wall
  fermions (not clover), with CG or Chebyshev and the initial guess of the force solves; on the
  CPU and, with `Main.backend : cuda_bridge`, on the device (`hmc::EvenOddHalves` for
  `CudaMdwf`): `action_type : Ratio` with `operator_type : Even_odd`. Checked: the force is the
  derivative of the action and S = |ξ_e|² after the heat bath (domain-wall and Wilson); K, K† and
  Γ5 on the even sites against the blocks and the whole lattice; Bridge++'s
  test_HMC_Wilson_Nf2_alt_eo_ratio (8⁴) to 1e-11; the trajectories of the device are those of
  the CPU. **Not agreeing**: Bridge++'s domain-wall tests test_HMC_Domainwall_Nf2_PV_alt_eo and
  _nosmr (8⁴) end 7e-6 and 1e-5 from their expected plaquettes, while the lexical ratio of the
  same operator on the same lattice agrees to 1e-11. Not understood: Bridge++'s numbers inside
  the trajectory are not at hand (its even-odd domain-wall HMC runs only on QXS fields, which
  neither Bridge++ tree here builds). The two tests are `#[ignore]` with the numbers.
- The Bridge++ tests of its alternative code with domain-wall fermions over Pauli-Villars
  fields on all sites (test_HMC_Domainwall_Nf2_PV_alt and _nosmr, 8⁴) agree to 1e-11.
- The CUDA kernels are compiled for the architectures 80 and 86 by default (A100 and A30, RTX
  30 series); `RUTTICE_CUDA_ARCH` takes a cmake list. It was 86 only.
- The README, `docs/DEVELOPMENT.md` and the 0.1.0 entry below no longer say that everything is
  checked against Bridge++: the comparisons are ours, where Bridge++ computes the same thing,
  and its developers have not reviewed Ruttice.

## 0.2.0 (2026-10-14)

Known issue: on an A30, the domain-wall multigrid on large lattices (28^4, 24^3x48) gives
different last digits from run to run of the same program, from the first round of the
bootstrap on; not on 12^4, and not on an RTX 3080. Not understood yet.

- `backend : cuda_bridge`: Wilson, clover and domain-wall propagators on an NVIDIA GPU with
  the CUDA kernels extracted from Bridge++ (feature `cuda-bridge`), in double, single and
  mixed precision, several right-hand sides in one launch, optionally TF32 on the tensor cores.
  The domain-wall multigrid is on the device with its setup (the filter, the bootstrap, P, the
  coarse operators and their solves). Not on the device: the multigrid of Wilson and clover
  fermions, HMC. On an A30, large lattices gave different last digits from run to run: not
  understood yet.
- The domain-wall multigrid of the CPU: single precision above the coarse lattice
  (`MultiGrid.fine_precision`), P in half precision (`F16`, with F16C where the processor has
  it), the coarse operators built from the operator itself a few columns at a time; less
  memory (start vectors drawn on the even sites, scratch fields made when needed, test vectors
  dropped once the cycle is made). One coarse correction and one cycle for the CPU and the
  device (`Correction`, `TwoGrid`).
- `app::mdwf` reads the gauge field in the format `gauge_config_type_input` names; it read
  Bridge++ text only.
- `fermion::PvRatio`: the Pauli-Villars preconditioned operator on the even sites for any
  operator with even-odd blocks; `fermion::solve_even_odd` for any kind of field.

## 0.1.0 (2026-10-13)

The first tagged version.

### What is in it

- **Fermions.** Wilson, clover and Möbius domain-wall operators, on all sites and in
  even-odd form; fields in the usual layout or in blocks of 2, 4 or 8 sites for SIMD; on one
  process or on several (MPI).
- **Solvers.** CG, BiCGStab, FGMRES, GCR, MINRES, multi-shift CG; mixed precision.
- **Multigrid.** For Wilson and clover fermions (SAP or GMRES smoother), and for domain-wall
  fermions on the even sites (the Pauli-Villars preconditioned operator, a factorized
  coarse correction, Chebyshev smoother, levels below the coarse lattice). The coarse
  systems are solved on the even coarse sites.
- **Measurements.** Meson two-point functions in fourteen channels for any pair of quarks;
  the pion correlator and the residual mass of domain-wall fermions; point and smeared
  sources; Coulomb gauge fixing; stout and APE smearing of the links.
- **Generating configurations.** HMC with the plaquette and rectangle gauge actions; two
  flavours of Wilson, clover or domain-wall fermions, on all sites or on the even sites;
  ratios to Pauli-Villars fermions; RHMC with rational approximations computed by a Remez
  iteration; leapfrog and Omelyan integrators on several time scales; initial guesses from
  earlier solutions; saving configurations on several processes.
- **Input and output.** Gauge configurations as Bridge++ text and binary, NERSC, ILDG, SciDAC and HDF5.
- **Solvers of HMC.** CG, BiCGStab, FGMRES and GCR for the two-flavour actions, multi-shift CG
  or CG pole by pole for RHMC; Chebyshev and (Wilson, clover) multigrid preconditioners; Neff's
  α for domain-wall fermions.
- **Block GCR** (`Block_GCR`) for domain-wall fermions: the right-hand sides of a solve share
  their directions. On the lattices tried it takes as many steps as GCR and more time.
- **Threads.** `cpu_thread` in a parameter file, or `RAYON_NUM_THREADS`.
- **Compared with the results of Bridge++** (our comparisons; not reviewed by its
  developers): operators, solver iteration counts, correlators, and one trajectory of its
  HMC tests of the actions Ruttice has. The domain-wall multigrid has no counterpart there.

### Changed since the last state of `main` (0.0.1, 2026-10-10)

- **Parameter files.** Solves are described in the sections of Bridge++ only (`Main`,
  `Gauge`, `Fopr`, `Solver`, `Source`, `MultiGrid`, …). The earlier format (`lattice`,
  `[solver]`, `[[quark]]`) is no longer read; what it could say that the Bridge++ sections
  could not has been added to them: `Quark_2` … `Quark_4`, `GaugeFixing`, `Measure`,
  `Output`, the `Exponential` source, Dirichlet boundaries, the `SciDAC` and `HDF5` formats.
  HMC files have the sections of Bridge++'s (`HMC_General`, `Builder_Integrator`, `Action_G`)
  and list their fermion actions as `Action_F_1`, `Action_F_2`, ….
- **Even-odd or lexical is one choice** (`Solver.operator_type : Even_odd | Lexical`), for
  the fine lattice and the coarse lattice of the multigrid alike. `Full` is still read for
  `Lexical`.
- **Names.** In the library the marker of a field on all sites is `Lexical` (it was
  `Full`). Files are named by their role: under `fermion`, every fermion has `mod.rs`,
  `site.rs`, `lexical.rs`, `even_odd.rs`; `lattice/for_sites.rs`; the polynomial
  preconditioners are under `solver/preconditioner`; the tests are named by subject
  (`tests/wilson.rs`, `tests/mdwf.rs`, …) instead of `stage` numbers.
- **Speed of the domain-wall operator.** The even-odd Pauli-Villars operator is five passes
  over the field instead of ten: the factor in the fifth dimension, the inverse of the
  site-local part and the difference with the input are done in the sweeps of the hopping
  term, and large outputs are written past the cache.
- **Speed of the multigrid.** Coarse systems are solved on the even coarse sites (the Schur
  complement of the coarse operator); inner solves no longer compute a first and a last
  residual.
- **HMC.** One driver (`Hmc`); the earlier quenched-only one is gone.
- **Repository.** The study programs of the domain-wall multigrid left `examples/` (tag
  `mdwf-mg-studies`); the working notes are on the branch `notes`.

### Known limits

- No GPU backend.
- The multigrids were developed and measured on 12⁴ and 16⁴ lattices.
- The LIME and HDF5 files have not been exchanged with other programs.
