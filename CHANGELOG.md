# Changelog

## Unreleased

- HMC with the domain-wall multigrid on the device takes the levels below the coarse lattice
  (`MultiGrid_Level2` and on, `levels_for_pauli_villars`), made with P and kept with the
  coarse parts (`DeviceParts`). On 24⁴ with stout links at m 0.001 (24 test vectors, levels
  with super-blocks 2 and 3) a correction takes 46–67 applications of the coarse operator
  instead of 2000–2900, and the solves of a short trajectory 61 s instead of 840 s. With 16
  test vectors the solves of D there stagnate, levels or not: P lacks the lowest modes.
- The solves with D† take less device memory: the preconditioner of `adjoint_by_gamma5`
  applies X_PV† to its fields directly (the right-hand side of Ã for a source with nothing on
  the odd sites) and makes its fields once, and the sweeps of `two_solves_half` reuse one work
  field. On 24⁴ the peak of HMC went from 22.9 to 20.7 GB (24.1 to 21.9 GB with what the
  memory pool keeps); iterations and ΔH unchanged.
- HMC with the domain-wall multigrid solves with D† by the multigrid of D: flexible GCR on K†
  (K = C_e⁻¹ S) preconditioned by C_e† Γ5 M Γ5, M the cycle of Ã (`fermion::adjoint_by_gamma5`).
  For Möbius fermions D† Γ5 D⁻¹ Γ5 = Γ5 (1 − c [Y, W] D⁻¹) Γ5 (the MG note, "Solving with
  D†"). The cycle of Ã† on the P of Ã, its coarse operators and the top of its smoother are
  gone: GCR(2) now converges where the old dual cycle needed GCR(8) or stagnated (16⁴, stout,
  m 0.01, one short trajectory: 313 iterations in 60 s, against 306 iterations of GCR(8) in
  73 s; the rebuild 24 → 15 s; ΔH the same), and on 12⁴ with stout links at m 0.001, where
  the old dual cycle did not converge, a solve takes 10–20 iterations of GCR(8). On 24⁴
  (m 0.01, GCR(2), 16 single-precision test vectors) a short trajectory takes 174 s and
  20.5 GB on the A30.
- HMC with the domain-wall multigrid on the device takes less memory: the test vectors are
  kept on the host between the solves (`DeviceSpinor::stash`) and on the device only while P
  is made or the bootstrap runs, and the cycle is made for each solve on the even sites and
  dropped after it. On 24⁴ (stout, m 0.01) HMC went from out of memory on the A30 to fitting.
- The coarse correction can solve its first factor by MINRES on the Γ5-Hermitian form
  (`CoarseKrylov::Minres` as `d_solver`, for a cycle whose first operator is D_PV), and
  `low_modes_pair` / `filtered_test_vector_pair` give the left singular vectors U = Ã V Σ⁻¹
  with the right ones from one filter (one application of Ã a vector in the Rayleigh-Ritz
  step). Neither is used by default; they are kept for a two-sided setup.

## 0.3.2 (2026-10-21)

Faster device kernels for the MDWF multigrid; no new features and no change to parameter
files. Some kernels add in another order, so the last digits, and at times an iteration or
two, move. On the A30 (24³×48, m 0.001, a point-source
propagator of 12 columns, setup included): 584 → 401 s, the setup 311 → 207 s, the solves
251 → 170 s, the same iterations.

- `cuda_bridge` multigrid: the restriction P† reads P coalesced (`fold_restrict_tiled_kernel`):
  a block for each coarse site reads the P of two fine sites at a time in 16-byte words into
  shared memory, and the fine spinors folded by chirality, where a thread for each output read
  its own row of P one number at a time (11% of the bandwidth on the A30). RTX 3080, 16⁴:
  4.64 → 1.31 ms a launch, setup 25 → 21 s, solve 0.6 → 0.5 s, the same iterations; the sums
  are in another order, so the last digits move. The old kernel stays for where the new one's
  shared memory does not fit.
- `cuda_bridge` coarse operator (`coarse_pass_kernel`): the fields at a coarse site and its
  eight neighbours, the same for every row, are read once into shared memory instead of once a
  row, and the matrices two columns at a time in 16-byte words, the eight links of a row read
  before they are used. RTX 3080, 16⁴: 82.5 → 62.6 µs a pass with the hops, DRAM 62 → 82%,
  instructions −42%; the coarse dimension must be even (it is twice the test vectors).
  A pass without the hops is a kernel of its own, without the registers of the unrolled links
  (on the A30 those passes had become 18% slower): RTX 3080, 70 → 44 registers, 19 → 16 µs.
- `cuda_bridge`: `hop_split_kernel` in single precision runs two blocks an SM (128 registers,
  no spill, where nvcc chose 168): RTX 3080, 16⁴, 1.9–2.1 → 1.5–1.6 ms a launch, DRAM 66 → 88%.
- `cuda_bridge`: |H|⁻¹ of the coarse blocks (`blocks_abs_inverse_gamma5_rounds_kernel`) by the
  cyclic Jacobi method in rounds of disjoint pairs at once, with H in shared memory, where the
  pairs went one at a time with H and V in global memory (3.8 s a launch on the A30, 1.15 TB of
  DRAM traffic). RTX 3080, 16⁴: 0.70 → 0.10 s a launch. The old kernel stays where the shared
  memory does not fit. Where H and V both fit (the A30 and A100 at a coarse dimension of 64,
  not the RTX 3080), V is in shared memory too: on the A30 the kernel with V in global memory
  had 6% of its warps active, waiting on V.
- `cuda_bridge`: the coarse operator with the hops at most 85 registers (`__launch_bounds__(256,
  3)`; CUDA 13 gave 128 for sm_80, 24% warps active on the A30), and the restriction copies P to
  shared memory with `cp.async`, its copies in flight while the fine fields are folded.
- `cuda_bridge`: the kernels of P read the entries of their geometry (γ5's columns and signs,
  the blocks) without an index known only at run time, which had put the whole kernel argument
  on the stack (a 96-byte frame and 0.5 GB of local loads a launch of the restriction on the
  A30); the prolongation reads P a complex number at a time. RTX 3080, 16⁴: restriction 1.29 →
  1.15 ms, prolongation 2.93 → 2.85 ms; the results the same bit for bit.
- `cuda_bridge` multigrid: the restriction to the levels below the coarse lattice takes a warp
  for each output instead of a thread (`level_restrict_kernel`); on the last level that was a
  single block. RTX 3080, 16⁴ with two levels below: 187 → 11 µs a launch on that level, the
  setup 32 → 25 s, the solve 0.7 → 0.6 s, the same iterations; the sums are in another order,
  so the last digits move.
- Corrections to 0.3.0: the double-precision `hop_gemm` at Ls 8 runs blocks of 128 threads
  (192 is the bound of its launch), three an SM with 168 registers where two fitted with 255.
  On the A30 (24³×48) it is 18% faster, a double CG iteration 31.0 → 28.0 ms. The stout
  smearing on the device saves nothing there yet: the CPU smeared in about 2 s (the 11.7 s
  that was the reason came from a run under nsys).
- Example `correction_exactness` (CPU): how exact the factorized coarse correction is on the
  vectors its prolongation is made from, for the right form, the left form and the dual
  correction, and GCR(2) on Ã_R with the cycle against Ã_R† with the dual cycle.
  `preconditioned_fov_device` prints, for the leftmost vector of the field of values, what the
  cycle does to it at several coarse tolerances.
- MG note: the timings again at 6d5bf18 on the A30 (mass scan, point-source propagators), the
  results in the order of the abstract, the implementation in an appendix.

### Known issues

- The dual multigrid cycle for Ã_R† (used by the HMC multigrid for the normal equations) did
  not converge in GCR(2) on the 12⁴ test configuration, setup S, m 0.001, where the cycle for
  Ã_R needs 23 iterations (`examples/correction_exactness`); being investigated. Propagators
  use the cycle for Ã_R or Ã_L and are not affected.

## 0.3.1 (2026-10-21)

- ILDG files are written with the record `ildg-data-lfn` (the logical file name, here the
  file's name), which the ILDG format asks for: Grid's reader refused our files without it.
  Files written before can still be read by Ruttice and QUDA; for Grid, write them again.

## 0.3.0 (2026-10-20)

**Fixed: the known issue of 0.2.0.** The domain-wall multigrid on an A30 gave different last
digits from run to run on large lattices: a race in the Jacobi kernel of |H_xx|⁻¹ (threads of
different warps for blocks of dimension above 32, without a barrier before a_pq was
overwritten). Runs of the same program now agree to the bit on the A30 too.

### Changes that affect existing parameter files or results

- HMC files: the multigrid of an action is the `MultiGrid` section, as in the propagator files;
  what only HMC has is in `MultiGrid_HMC`. **`Solver.MultiGrid` is refused** with where its keys
  went (`number_of_setup_iterations` is `Setup.bootstrap_number_of_rounds`; the Wilson
  multigrid's `Setup.seed` is 21 by default, 1 before). The domain-wall smoother is
  `MultiGrid.Smoother`, not `Solver.Chebyshev`; `number_of_test_vectors` counts the columns of P
  as in the propagator files. What HMC does not have yet is refused rather than ignored.
- Propagator files, `Solver.precision : mixed` (CGNE on `cuda_bridge`): one CG in single
  precision with reliable updates in double precision. `convergence_criterion_squared_inner` is
  now the fall of |r|² after which the solution and residual are made again in double precision
  (1e-2), no longer the tolerance of an inner CG (1e-10).
- The default smoother of the domain-wall multigrid is Chebyshev 8 steps on [hi/10, hi] (was 5
  on [hi/30, hi], with which stout-smeared 12⁴ solves at m = 0.001 stagnated). M5 = 1.8 without
  smearing takes 12 steps on [hi/30, hi]; the example files say so.
- `cuda_bridge`: stout smearing of the propagator runs is done on the device; the links agree
  with those of the CPU to rounding, not to the bit. The coarse operator kernel sums in
  another order than in 0.2.0, so the multigrid's results move in the last digits (the
  iterations the same or within one).
- The CUDA kernels are compiled for the architectures 80 and 86 by default (A100, A30, RTX 30
  series); `RUTTICE_CUDA_ARCH` takes a cmake list. It was 86 only.
- Precisions are read in their usual spellings, in any case (`single`, `float`, `f32`, `FP32`,
  `double`, `f64`, `FP64`, `half`, `f16`, …).

### HMC

- HMC on the device (`Main.backend : cuda_bridge`): links, momenta, integrator, gauge actions,
  stout smearing and its chain rule, and the domain-wall Ratio action with its solves and force;
  the links cross over only for the flow and to be saved. Bridge++'s
  test_HMC_Domainwall_Nf2_PV gives the trajectories of the CPU.
- The even-odd ratio of two flavours over two Pauli-Villars (or Hasenbusch) flavours
  (`hmc::TwoFlavorRatioEo`, as Bridge++'s `Action_F_alt_Ratio_eo`), Wilson and domain-wall, on
  the CPU and the device. Bridge++'s test_HMC_Wilson_Nf2_alt_eo_ratio agrees to 1e-11.
- The domain-wall multigrid in HMC, on the CPU and on the device, for the Ratio action on all
  sites and even-odd; kept as a cache from solve to solve and rebuilt when the iterations grow
  (`MultiGrid_HMC.rebuild_iteration_growth`); the cycle in single precision on the device
  (`fine_precision : single`). D† x = b through Ã† for the force solves.
- Mixed precision for the domain-wall actions (`Solver.precision : mixed`), on the CPU and the
  device: the energy and force of CG in double precision.
- Every trajectory prints what the solves of each fermion action cost (solves, iterations,
  seconds, multigrid setups and reuses). `HMC_General.reversibility_checks`.
- The gradient flow (`gauge::flow`, Lüscher's RK3): t0, the energy density and the
  topological charge, as Bridge++ and checked against its tests; measured during HMC
  (`GradientFlow`) and by `ruttice-flow` / `ruttice-flow-mpi`, with the error by Wolff's
  Γ-method (`numerics::gamma_method`).
- The force of the plaquette action is faster (staples walked site by site), same numbers.

### Propagators and solvers

- Eigensolver files: an `Eigensolver` section prints eigenvalues of the fermion operator with
  the keys of Bridge++ (IRLanczos, IRArnoldi; D, D†, D†D, DD†, H; the four orders; Chebyshev).
  Bridge++'s `Eigensolver` test agrees to 1e-13. CPU only.
- The whole domain-wall multigrid is on the device, levels below the coarse lattice and the
  exact adjoint of the Pauli-Villars coarse operator included, and in the library
  (`backend::cuda_bridge::multigrid`) for any program.
- `Main.verbose_level : Detailed` prints the residual history of GCR.
- The Chebyshev filter of the test vectors no longer overflows in single precision (it
  panicked on stout-smeared links).

### Performance

On an A30, 24³×48, Möbius m = 0.001 with stout links, a point-source two-point function (12
columns): the multigrid 584 s with its setup, mixed CGNE 804 s, double CGNE 1333 s.

- Device hop kernel: two blocks of 256 threads an SM in single precision (128 registers; sm_80
  gave 168), the field it is subtracted from read early, the matrix of Neff's α folded into
  the next hop. A30 mixed CGNE at α = 0.6: 83.1 → 59.1 s. In double precision blocks of 192
  threads with the legs one at a time (168 registers instead of 254).
- Device multigrid: the coarse operator kernel reads a row per warp (3.6 → 1.1 ms an
  application on an RTX 3080); work fields from the stream-ordered memory pool; the Gram
  matrix in one pass; the coarse operators made on the device; reductions finished on the
  device; the Chebyshev filter and smoother steps and the filter's last hop fused.
- CPU: Neff's α inside the even-odd sweeps, CGNE on 12⁴ 87 → 42 ms an iteration (Bridge++'s
  core code: 82 ms on the same i9-12900, the same iterations and correlator); the coarse
  operator build 6.0 → 3.7 s (8⁴ benchmark).

### Analysis tools

- `eigen`: Krylov-Schur Arnoldi for non-Hermitian operators, the complex Schur form,
  shifted and shift-invert transforms, smallest singular values and resolvent norms
  (ε-pseudospectra); the orders, tolerances and thresholds of Bridge++ for Lanczos.
- `preconditioned_fov` and `preconditioned_fov_device`: the field of values of Ã and of Ã M,
  their eigenvalues with condition numbers (`--arnoldi`), pseudospectra (`--pseudo`), the
  smoother and coarse correction apart (`--parts`).
- `ruttice-low-modes` (feature `cuda-bridge`): the lowest singular values of the even-odd
  Pauli-Villars operator and their chirality; `--coherence` (Lüscher's local coherence),
  `--fov`.
- `examples/smeared_ildg`: a configuration stout-smeared as a run does it, written as ILDG.

### Documentation

- `docs/mg_background/MDWF_MULTIGRID_NOTE`: the domain-wall multigrid, its construction and
  results. `docs/CONVENTIONS.md`: the notation against Brower-Neff-Orginos, Kanamori et al. and
  Bridge++. `docs/TODO.md`: what is known and not done.
- The README asks users to cite Bridge++; it and `docs/DEVELOPMENT.md` say the comparisons with
  Bridge++ are ours. `cargo test` finds the Bridge++ test configuration in `examples/`.

### Known issues

- Mixed CGNE on 24³×48 at m = 0.001 (A30): |Dx−b|² ends at 5e-17–3e-16 for a target of 1e-18;
  the correlator is 2e-7 from that of the multigrid. Use double precision where the true
  residual matters. QUDA's sloppy CG shows the same.
- Bridge++'s even-odd domain-wall HMC tests test_HMC_Domainwall_Nf2_PV_alt_eo and _nosmr (8⁴)
  end 7e-6 and 1e-5 from their plaquettes while the lexical ratio agrees to 1e-11; not
  understood (`#[ignore]` with the numbers).
- Running out of device memory panics.

## 0.2.0 (2026-10-14)

Known issue: on an A30, the domain-wall multigrid on large lattices (28^4, 24^3x48) gives
different last digits from run to run of the same program, from the first round of the
bootstrap on; not on 12^4, and not on an RTX 3080. Not understood yet. (Fixed in 0.3.0: a race
in the Jacobi kernel of |H_xx|⁻¹.)

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
