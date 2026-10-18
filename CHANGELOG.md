# Changelog

## Unreleased

- `eigen::arnoldi`: eigenpairs of a non-Hermitian operator by the Krylov-Schur method
  (Stewart 2001) on any `Vector`, at the end of the spectrum asked for (`Which`: smallest or
  largest real part, largest or smallest magnitude); `eigen::schur`: the complex Schur form of a
  small dense matrix (Householder to Hessenberg, the shifted QR algorithm), its reordering and
  the eigenvectors of the triangular factor.
- `preconditioned_fov_device --arnoldi`: the eigenvalues of Ã M of smallest real part with their
  condition numbers 1/|y†x| (the left eigenvectors from M† Ã†), and of largest magnitude. On the
  v2 12⁴ configuration at m 0.001 every eigenvalue lies at Re ≥ 0.868 (stout, M5 1.0) and
  Re ≥ 0.831 (M5 1.8), with condition numbers 6–20 and |λ| at most 5.9 and 1.75: the spectrum
  is far from the origin although the field of values is not.
- `Main.verbose_level : Detailed` (or `Paranoiac`) prints the residual history of GCR in the
  propagator runs, on the CPU and on the device: |r|²/|b|² before every step (`Gcr::recording`).
- `preconditioned_fov_device --parts`: the smoother S and the coarse correction E of the cycle
  apart, the field of values and ‖I − Ã X‖₂ of Ã S, Ã E and Ã M, and the vector at λ_min of the
  Hermitian part of Ã M dissected. On the v2 12⁴ configuration at m 0.001, with the coarse
  factors solved exactly, ‖I − Ã E‖₂ = 1.9e4 (stout, M5 1.0) and 2.9e3 (M5 1.8) against 18.5 on
  4³×8, the smoother's ‖I − Ã S‖₂ about 1: the coarse correction is the factor that amplifies.
- HMC with the multigrid on the device: `MultiGrid.fine_precision : single` puts the cycle
  (smoother, residuals, P and the corrections) on the operators in single precision, made from
  the links of the double ones on the device (`CudaMdwf::converted_from`), the outer GCR in double
  precision; the coarse parts (and the cache) are those of the double operators.
- `examples/preconditioned_fov` and `preconditioned_fov_device`: the field of values of Ã and
  of Ã M (M the two-grid cycle) by Lanczos on their Hermitian parts, M† the cycle of Ã† on the
  same P with the same top of the smoother's interval (`CoarseParts::with_top`; with tops of
  their own ⟨z, Ã M w⟩ and ⟨M† Ã† z, w⟩ differed by 3e-6, now by 1e-11) and both coarse
  factors solved on their exact adjoints. The two agree on 4³×8 (Ã M in [0.772, 1.222]). On
  the v2 12⁴ configuration at m 0.001 the field of values of Ã M reaches far into the left half
  plane: [−23.9, 30.8] with stout links at M5 1.0, [−1.49, 4.17] at M5 1.8 (Ã itself:
  [−0.0046, 1.98] and [−0.050, 4.46]).
- HMC runs print what the solves of each fermion action cost in every trajectory
  (`# cost  n  Action_F_1: solves, iterations, seconds; multigrid setup, rebuild, rebuilds,
  reuses`), from `MdAction::take_cost` (`SolveCost`): the numbers of a comparison of solvers.
- HMC: mixed precision on the device too (`backend::cuda_bridge::hmc_mixed::SingleCuda`, the
  operator in single precision on the links of the device; on all sites and for K of the
  even-odd Ratio action). Two trajectories of the Bridge++ example, the Ratio action on all
  sites and even-odd and the Standard action: the plaquettes of CG in double precision on the
  CPU to the last digit, ΔH to 1e-9.
- HMC with the domain-wall multigrid on the device (`backend : cuda_bridge`, the Ratio action on
  all sites and even-odd): the test vectors, P (half precision, shared by the cycles of Ã and
  Ã†), the coarse operators and both cycles on the device (`backend::cuda_bridge::hmc_multigrid`,
  over `backend::cuda_bridge::multigrid`); the sweeps on D†D and K†K are those of the CPU,
  written once for both (`EvenSolves`). Two trajectories of the Bridge++ example (stout, 4³×8):
  the plaquettes of CG on the CPU to the last digit, ΔH to 1e-9. The cache
  (`rebuild_iteration_growth`) is on the device as well: `DeviceParts` (P, the coarse operators,
  |H_xx|⁻¹), shared, make the corrections for new links (`DeviceCorrection::from_parts`), the
  tops of the smoothers kept with them; the cached trajectories are those of CG too.
- HMC: the domain-wall multigrid as a cache (`MultiGrid_HMC.rebuild_iteration_growth`): P and
  the coarse parts of the cycles of Ã and Ã† (`CoarseParts`: the coarse operators, the inverses
  of their blocks, the top of the smoother) are kept from solve to solve and the cycles made
  from them for the links of each solve (`PvEoCycle::from_parts`, `into_parts`), and built again
  only after a bootstrap or when the GCR iterations per solve have grown by the factor given.
  The outer GCR and the smoothers work with the operators of the current links, so the solves
  are those of CG still (4³×8, links moved by MD steps: 1e-14 and 2e-11 relative, one rebuild
  in eight solves). What it saves on large lattices is not measured yet.
- HMC: mixed precision for domain-wall fermions (`Solver.precision : mixed`, the Standard
  (Lexical) and Ratio actions, on all sites or even-odd, CG without a preconditioner, on the
  CPU): defect correction on the residual of D†D (K†K) in double precision, each correction by
  CG in single precision on the same operator on the same links (`hmc::SingleNormal`,
  `SingleMdwf`, `SingleMdwfEo`) to `Solver.convergence_criterion_squared_inner` (1e-10). It
  gives the energy and the force of CG in double precision.
- The even-odd Ratio action of domain-wall fermions takes the multigrid
  (`operator_type : Even_odd` with `Solver.preconditioner_type : MG`): (K†K)⁻¹ on the even sites,
  K = T_ee⁻¹ Ŝ, through the same two solves on the even sites as the action on all sites
  (Ŝ⁻¹ u is the even part of the solve of D x = (u, 0)), repeated on the residual of K†K. It
  gives the energy and the force of CG (4³×8: 6e-15 and 1e-11 relative).
- One way to the solves of the pseudofermion actions: the domain-wall multigrid is a
  preconditioner of `normal_solve` like the others, given the Pauli-Villars operator, instead
  of a case of its own in the Ratio action. The denominator's solve in the heat bath is CG with
  the multigrid as without it.
- HMC files: the multigrid of an action is the `MultiGrid` section of the propagator files,
  beside its `Solver` (the same keys and defaults: `MultiGrid_Level1`, `Setup`, `Smoother`, the
  coarse solvers), for Wilson and clover fermions and for domain-wall fermions alike; what
  only HMC has is in `MultiGrid_HMC` (`refresh_interval`, `reuse_vectors`,
  `number_of_reuse_iterations`, and `convergence_criterion_squared_inner` of the domain-wall
  solves on the even sites). **The old place, `Solver.MultiGrid`, is refused** with where its
  keys went (`number_of_setup_iterations` is `Setup.bootstrap_number_of_rounds`; the Wilson
  multigrid's seed is `Setup.seed`, 21 by default, 1 before). The domain-wall smoother is
  `MultiGrid.Smoother`, no longer `Solver.Chebyshev`, and `number_of_test_vectors` counts the
  columns of P as in the propagator files (twice the vectors kept). What HMC does not have yet
  is refused rather than ignored: `fine_precision : single`, levels below the coarse lattice,
  `basis_precision`, `Setup.checkpoint`. The coarse solves of the domain-wall multigrid are in
  single precision by default as they always were, `coarse_precision : double` for double.
- HMC files: `Solver.operator_type`, as in the propagator files, accepting what the action
  solves (`Lexical`, `Even_odd`, `Normal_left`, `PV_right`).
- HMC files: the messages about an action name its section and keys in the file
  (`Action_F_1.Solver.solver_type`), not the internal index, which counted `Action_G` too.
- The checks and the conversions of the `MultiGrid` section are methods of the section
  (`MultigridBlock::check_domain_wall`, `check_wilson`, `domain_wall`, `wilson`), shared by the
  propagator and HMC files; the Wilson check now also refuses blocks that do not divide the
  lattice and notes the domain-wall keys it does not use.
- Precisions are read in their usual spellings, in any case: `single`, `float`, `f32`, `FP32`,
  `float32`; `double`, `f64`, `FP64`, `float64`; `half`, `f16`, `FP16`.
- `ruttice` with `backend : cuda_bridge` no longer has a path that made the coarse correction of
  the multigrid on the CPU: no file could reach it (the coarse solvers it was for are refused
  when the file is read), and everything of the multigrid is on the device.
- The setup and the cycle of the domain-wall multigrid on the device are in the library
  (`backend::cuda_bridge::multigrid`: the filtered test vectors, the bootstrap, the top of the
  smoother, the two-grid cycle), for any program; `ruttice` uses them and prints the same
  numbers to the bit.
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
  fields on all sites (test_HMC_Domainwall_Nf2_PV_alt and _nosmr, 8⁴) agree to 1e-11, and so
  does test_HMC_Domainwall_Moebius_Nf2_PV of Bridge++ 2.1.3 (Möbius b = 1.5, c = 0.5). Bridge++
  2.1.3 itself has no even-odd ratio any more (`Action_F_Ratio_eo` and `test_alt_hmc` are gone).
- D† x = b through Ã†, the Pauli-Villars operator with the roles of D and D_PV exchanged
  (`fermion::adjoint_right_hand_side`, `adjoint_solution`; Ã† = `PvRatio::new(pv, op, ..)`, its
  cycle M† = `PvEoCycle::new(pv, op, ..)` on the P of Ã), on the CPU and on the device: what the
  multigrid needs in the force solves of HMC. Checked: D† x = b to 4e-27 on both.
- `Hmc::reversibility` and `HMC_General.reversibility_checks`: forward, P → −P, back. On the
  device for the 8⁴ domain-wall ratio on all sites and even-odd, |ΔU| ≤ 5e-13 and H back − H ≤
  9e-10 of H ≈ 2.5e5.
- `ruttice-low-modes`: `--coherence` (Lüscher's test of local coherence with the prolongation
  of the multigrid, right, left and cross), `--fov` (the distance of Ã and D from
  Γ5-Hermiticity, and [λ_min, λ_max] of (Ã + Ã†)/2), and its memory printed first.
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
