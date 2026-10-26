# Changelog

## Unreleased

- A breakdown of the single-precision CG of mixed precision (a NaN since 5cc9799) is returned to
  the outer defect correction as `Breakdown`, as for any other solver; it panicked
  (`hmc::mixed`, `a_nan_in_single_precision_is_a_breakdown`).
- The clover blocks and their inverses of the CPU operator are made by the threads of the pool
  (`set_gauge` on 16⁴: 0.51 s to 0.40 s; the rest is the field strength, in `docs/TODO.md`).
  The same blocks; the first singular site is still the one reported.
- `src/fermion`: the message of `GaugeError::NotSpecialUnitary` lost a run of spaces; the
  documentation of the SIMD Wilson and domain-wall operators no longer says they run on one
  process only (they run on several, `tests/simd.rs`); two comments put back where they belong.
- The setup of the domain-wall multigrid has fewer copies: the vectors s and t̂ that P is made
  of (`lifted_vectors`) and the rounds of the bootstrap (`bootstrap_rounds`) are written once
  for the CPU and the device (they were three and two). The device gives the same bits as
  before at every stage (`RUTTICE_CHECKSUM`); on the CPU s is now scaled before t̂ is made of
  it, as on the device, which changes P at rounding (4³×8: the propagators with the same
  iterations and coarse applications, correlators within 3e-10; of five HMC runs, one with an
  iteration more or fewer in a trajectory, ΔH within 1e-10).
- The defect correction on the normal equations (A†A + σ) x = b is written once
  (`refine_normal`): the two-step solves of HMC, mixed precision, and the two solves of the
  domain-wall multigrid had four copies of the loop. The operations are the same, in the same
  order.
- `multigrid/mdwf_even_odd.rs` (1362 lines) is three files: P and the operator, the cycle
  (`mdwf_cycle.rs`), and the test vectors, low modes and bootstrap (`mdwf_vectors.rs`). The code
  is moved, not changed.
- The multigrid of Wilson and clover fermions runs the cycle of the domain-wall one (`TwoGrid`)
  instead of a copy of it: `TwoGrid` takes its smoothing as a trait (`Smooths`: the Chebyshev
  polynomial of the domain-wall multigrid; x = b or a pre-smoother, then a post-smoother, for
  Wilson), and the Wilson coarse correction is a `Corrects`. The same operations in the same
  order: the propagator runs of `examples/clover_mg.yaml` (even-odd, lexical, double coarse,
  SIMD, a level below, GMRES smoothing with pre-smoothing, BiCGStab, Wilson; on the device
  even-odd, lexical, double coarse) print the same iterations, residuals, work and correlators,
  digit for digit, before and after.
- The iterations of the clover multigrid on the Bridge++ test configuration are pinned by a
  test (`tests/run.rs`, `tests/cuda_bridge.rs`), with the final residual within a factor 1.5:
  the comparisons with Bridge++ only asked that the multigrid help. Started from zero instead of
  x = b, the cycle fails it.

### Fixed

- A propagator run stops with an error when a solve reported convergence but the solution's
  residual |Dx−b|²/|b|² is more than 1000 times the tolerance (and above 1e-20): the solver was
  misled by a broken operator or exchange. Sound runs end within a few times the tolerance
  (Wilson and clover at most 0.4 of it, domain-wall 2); a run of the domain-wall multigrid on
  two processes with the NCCL receives in the wrong order ended at 8e-2 against 1e-22 and
  exited 0 before, and now fails. Wilson, clover and domain-wall, CPU and device.
- The NCCL halo (feature `nccl`): `RUTTICE_HALO` must be the same on every process, and only
  `host`, `nccl` or unset (a process with `host` skipped the collective calls the others waited
  in, and a misspelt value was taken for unset); the communicator's stream and events are
  destroyed with it; the ranks of a `LocalComm` on one GPU go through the host instead of
  trying NCCL (with `NCCL_MULTI_RANK_GPU_ENABLE=1` NCCL accepted them, and the exchange never
  completed on the default stream they share); an error NCCL reports from an earlier exchange
  stops the run at the next one. Two processes on one GPU by `mpirun` give the same results
  through the host and NCCL, as before.
- On MPI the global sums add the processes' parts in rank order (`MPI_Allgather`, then the
  sum on every process), as `LocalComm` does: `MPI_Allreduce` added them in an order of its
  own (Open MPI 5.0 on 4 processes: 1, 1e16, −1e16, 1 summed to 0, in rank order to 1), so a
  run on MPI was not the run the `LocalComm` tests check, and the MPI standard does not even
  require every process to get the same bits. `ruttice-mpi-check` checks it, and also the
  gather and a single-precision halo.
- `LocalComm`'s halo exchanges refuse their buffers between `start` and `wait`, as the MPI
  ones do, and `wait` without `start` does nothing: a test on `LocalComm` finds an access that
  would race under MPI. The contract is written on `HaloExchange`.
- A NaN or infinity from the operator or the preconditioner is a `Breakdown` in every solver,
  found at the first residual that has it. Before, a NaN residual counted as neither converged
  nor active: CG and reliable CG applied the operator to the end of their iterations (6000 in
  the test) and FGMRES to the end of every cycle, and reported `NotConverged`, which HMC takes
  for a slow solve (the two-step solves go on, the domain-wall multigrid makes P again and
  falls back to CG); MINRES returned `Ok` with a NaN solution (`max` dropped the NaN, γ became
  zero and the solve looked converged). MINRES outside an inner solve now returns `Ok` only on
  the true residual (the recurrence's is in the norm of the preconditioner), and CG with a
  preconditioner refuses ⟨r, M r⟩ ≤ 0. Finite solves are unchanged, bit for bit
  (`tests/solver.rs`, `a_nan_from_the_operator_is_a_breakdown`).
- The `# cost` lines of an HMC run leave out no action: the Rational action and the Standard
  action on the even sites now report their solves and iterations too (a multi-shift solve is
  one solve, a pole solved on its own one).
- `the_work_of_a_trajectory_is_counted_alike_on_both` failed on an RTX 3080 since 0.9.0: the
  CPU and the device took 5 iterations and some CG restarts more or fewer (11 applications of
  D†D, 1.26e-3 of the fine work, above the test's 1e-3). The test now asks what does not depend
  on the solvers: the gauge work and the forces equal, the fine work equal up to whole
  applications of D†D, the vectors to 1e-3. A force counted twice on the device fails it.
- With levels below the coarse lattice, the bootstrap of the CPU now takes them from its second
  round on, as the device has done since 0.4.1 in HMC and since 0.5.0 in the propagator runs: the
  test vectors of the two backends had drifted apart from the third round on (4³×8, three rounds
  and `MultiGrid_Level2`: 188 against 192 coarse applications for a right-hand side).
  `tests/cuda_bridge.rs` compares the two with three rounds and a level.
- The Rational action refuses `Solver.precision : mixed`, `Solver.number_of_initial_guess` and
  `Fopr.neff_alpha` when the file is read, and every action a `Solver.Chebyshev` section
  without `preconditioner_type : Chebyshev`; they were ignored (the Rational action since
  0.1.0: the same iterations and plaquette with and without them).
- The `# cost` lines of HMC with the domain-wall multigrid count the GCR iterations of a solve
  that stalled and of the attempts after it (P made again, a bootstrap, CG); a stalled
  attempt counted none, on the CPU and the device.
- A coarse operator built with a right factor (P† D C_PV⁻¹ P, the lexical domain-wall
  multigrid's `RightPreconditioned`) refuses `mult_dag`: its adjoint is not Γ5 D_c Γ5, and
  `mult_dag` gave a wrong one (13% off in ⟨y, D_c x⟩ against ⟨D_c† y, x⟩ at c = 0). No program
  applied it; with levels or CGNE on that coarse operator it would have been wrong.
- A Bridge++ coefficient file whose first pole is not the smallest is refused when it is read:
  the multi-shift CG stops on the residual of the first pole, so the others would have been
  left unconverged. Bridge++'s files and the computed coefficients have the poles in
  increasing order.

### Corrections to earlier releases

- 0.9.0 said the CPU and the device count every kind of a trajectory and of a multigrid solve
  alike. `tests/cuda_bridge.rs` compares a clover RHMC trajectory and a clover multigrid solve
  without its setup; domain-wall fermions are counted differently by design (the 0.9.0 bullet
  on the fifth dimension), and the Wilson operator, domain-wall trajectories and the setups of
  the multigrids are not compared.
- 0.9.0 called the work report complete. Open: the halo exchanges are not counted, and the
  Jacobi sweeps of |H|⁻¹ are counted as six, an estimate (TODO 22).
- 0.9.0: the momenta, the proton and the three-point functions are for Wilson and clover
  fermions; a domain-wall file with them is refused when it is read.
- 0.7.0: of the Wilson and clover HMC on the device, `Even_odd` is for the Standard action and
  the Ratio of Wilson fermions; Rational is `Lexical` only, and the even-odd Ratio of clover
  fermions is not written (refused on the CPU too).
- 0.5.0 said every option of the domain-wall files does the same on both backends; the
  bootstrap with levels below the coarse lattice did not (fixed above).
- 0.3.0 said every trajectory prints the cost of each fermion action; the Rational action and
  the even-odd Standard action printed none (fixed above).

## 0.9.0 (2026-10-25)

Measurements beyond the zero-momentum mesons, and a work report that counts everything. Meson
correlators at momenta and the proton correlator give Bridge++'s numbers (1008 values that no
Bridge++ test checks, printed by a driver linked to Bridge++); connected three-point functions by a
sequential source through the sink, for mesons and the proton, satisfy the Ward identity of the
conserved vector current to 1e-15. All of it in the parameter files (`Measure.momenta`, `proton`,
`three_point`; examples/clover_measurements.yaml), on the CPU and the device. The work report now
counts the gauge field and the fermion forces (a clover RHMC trajectory: 3.6e9 Flop of `Gauge`,
2.0e9 of `Force`, 2.8e10 of fine operators) and the setup of the multigrids, and the CPU and the
device count every kind of a trajectory and of a multigrid solve alike.

- Fixed (found by a code review): domain-wall parameter files accepted `Measure.momenta`,
  `proton` and `three_point` without computing them, and the header said they ran; they are now
  refused when the file is read. `Measure.momenta` took the phase origin of quark 1 for every
  pair; quarks with their sources elsewhere are now refused for it. With `[0, 0, 0]` among the
  momenta the proton at zero momentum was printed twice. The device's domain-wall force counted
  its two γ5 passes, which the CPU's does not.

- Fixed: the work report counted different Flop for the same solve on the CPU and the device
  (a clover even-odd solve: 1.36e9 on the CPU, 2.97e9 on the device). The CPU did not count the
  site blocks T and T⁻¹ of the even-odd solves (now `count_site_block`, Bridge++'s
  Nc Nd (2 + 48 Nc) as the device); the device counted its vector operations over the padding
  of the last block of 32 sites (now the lattice reals only). The CPU, the SIMD layout and the
  device give the same count; `tests/cuda_bridge.rs` checks the Schur complement of the Wilson
  and clover operators to be counted alike.
- The work report of the CPU counts the fifth dimension of domain-wall fermions (the factor
  b − c Y, which the hops recompute on each of their eight legs, C⁻¹ by substitution, Neff's Q,
  `combine`) and the copies of `select` and `assign`. A domain-wall solve on the CPU now reports
  its arithmetic (4.5e9 Flop for a column of a 4³×8×8 CG solve, 1.7e9 before); the device counts
  its dense Ls × Ls products instead (2.8e9), so the two differ by design (USAGE).
- The work report is complete: two kinds of work more, `Gauge` (plaquettes, the gauge actions
  and forces, stout and APE smearing and the chain rule, the algebra of links and momenta in HMC,
  the clover blocks) and `Force` (the hop force of the three fermions, the clover insertion, its
  leaves and the log det insertion), and the setup of the multigrids (block and dense inverses,
  |H|⁻¹, the bases and the levels), the CPU's level V-cycle, SAP's block operations and the
  even-odd restriction on the CPU, copies, casts and γ5 counted. One formula for each operation
  where both backends do the same arithmetic (`gauge::count`, `hmc::count`,
  `multigrid::count`). A clover HMC trajectory now counts 3.6e9 Flop of `Gauge` and 2.0e9 of
  `Force` beside 2.8e10 of fine operators (none before). The device's clover operator counts the
  2 Nc Nd additions of its term and the bytes of its blocks, as the CPU. `tests/cuda_bridge.rs`
  compares every kind of a trajectory and of a multigrid solve on the two backends; the test
  fails with one count taken out.

- Meson correlators with momentum (`measure::meson_correlator_with_momentum`): the sink weighted by
  e^{+i p·(x − x0)}, p = 2π n / L, as Bridge++'s `meson_momentum_correlator`. Checked against
  Bridge++'s numbers (PS, V1–V3 at ten momenta, three flavour pairs of clover propagators:
  960 values, which no Bridge++ test checks, printed by a driver linked to Bridge++). The
  `measure` module is split into `source` and `meson`, its correlators unchanged to the last
  bit.
- The proton correlator (`measure::proton_correlator`): ε_abc (u_a^T Cγ5 d_b) u_c with
  C = γ4γ2 and a projection Γ, as Bridge++'s `proton_correlator`, with an optional momentum.
  Checked against Bridge++'s numbers (Γ = 1, γ4 and (1 + γ4)/2; u and d of different masses and
  of one: 48 values, which no Bridge++ test checks).
- Connected three-point functions by a sequential source through the sink
  (`measure::sequential_source`, `current_insertion`, `conserved_charge`), for any sink that
  gives its contraction and the derivative of it by the propagator the current acts on (trait
  `Sink`; `MesonSink`, `ProtonSink` with a projection Σ w Γ and the current on u or d). Bridge++
  has none, so they are checked by the conserved vector current of the Wilson and clover
  operators: its charge does not change between source and sink and jumps there by the number of
  quarks times the two-point function (1 for the meson, 2 for u and 1 for d in the proton), to
  1e-15 on a random configuration with u and d of different masses; and by the sequential source
  through the current with momenta at the sink and the current (to 1e-15).
- In the parameter files, `Measure` takes `momenta`, `proton` and `three_point` (`t_sink`,
  `sink`, `current_on`, `currents`, `momenta`); see USAGE and
  `examples/clover_measurements.yaml`. Through the program the mesons at momenta and the proton
  give Bridge++'s numbers, and the conserved charge of the three-point functions its Ward
  identity, for a meson sink with the current on u and a proton sink with it on d. The
  sequential source is solved by the solver, multigrid and backend of the file.

## 0.8.0 (2026-10-25)

The device does every fermion on one GPU or several, now with the Wilson and clover multigrid.
A clover propagator with the multigrid (examples/clover_mg.yaml, 4³×8) takes the CPU's 7
iterations and gives its correlators, in 0.26 s on an A30 against 2.7 s on two CPU threads; HMC
with the multigrid gives the CPU's plaquettes and ΔH. Wilson and clover operators and their
multigrid on several processes give the result of one (P and the test vectors bit for bit, the
same iterations). Parameter files are unchanged; on the device the Wilson multigrid refuses its
`levels` and `precondition_diagonal` when the file is read. In the library, the Wilson
multigrid, SAP and `OnParity` are generic over a backend (`WilsonMultigrid<B>`, `Sap<B, V>`,
`OnParity<.., P>`); `Multigrid` stays the CPU's type.

- The Wilson and clover multigrid on the device, for propagators and HMC
  (`preconditioner_type : MG` with `backend : cuda_bridge`), on one GPU or several: P made from
  the test vectors on the host by the CPU's code and kept on the device folded in spin (a vector
  of one chirality has spins 2, 3 equal to ± spins 0, 1, so the domain-wall folding serves with
  Ls = 1), the coarse operator made on the device by one function for all three operators
  (`coarse::coarse_blocks`), SAP on the device (the block operator an instance of the hop kernel
  with a `BlockCut`, the block minimal-residual steps with sums by a fixed tree), the cycle in
  single or double precision, on the even sites for an even-odd solve. The setup and the cycle
  are written once over a backend (`WilsonMultigrid`, `MultigridBackend`: `WilsonOnCpu`,
  `WilsonOnDevice`), and so are SAP (`Sap`, `SapBlocks`), the cycle on one parity (`OnParity`
  over `SplitParity`) and the kept test vectors of HMC (`MultigridSetup::cycle_on`); the CPU's
  results are those of before to the last digit. Checked: the device takes the CPU's iterations
  (clover_mg.yaml: 7, 0.26 s against 2.7 s on two CPU threads) and gives its correlators; HMC
  with the multigrid gives the CPU's trajectory; on several processes P and the test vectors are
  those of one bit for bit and FGMRES takes the same iterations. Refused on the device so far:
  the levels below the coarse lattice and `precondition_diagonal` of the Wilson multigrid.
- Fixed: the Wilson and clover multigrid with even-odd solves (`coarse_even_odd`, set by an
  even-odd solver) prepared its coarse operator for solves on the even coarse sites only
  before the setup iterations, so with any (the default is 4) the coarse solves ran on the
  whole coarse lattice. Now the coarse operator of the last setup iteration is prepared too;
  even-odd multigrid runs on the CPU change at the level of the coarse solves (the outer
  iterations stayed the same in the tests).

- Wilson and clover fermions on several GPUs: the operators, their even-odd blocks, the solves
  and HMC (Standard, Ratio, Rational; the clover term and its force on extended links), each
  process with its GPU, the result of one GPU bit for bit (checked up to 16 ranks with
  `LocalComm`). The Wilson and clover operators are now the hop kernel of the domain-wall
  operator, specialised (`hop_kernel` with `FourD`, `Clover`), so they take the interior and
  surface parts of several processes; `wilson_interior_kernel` and `clover_dslash_kernel` are
  removed. On the A30 at 16⁴ the time is that of the old kernels within 2%, the results the same
  to the last bit.
- The sums of a leg of the hops are explicit fused multiply-adds in a fixed order (`c_macc`,
  `add_leg`, the third row of a link, the clover blocks): nvcc chooses which `a b + c` to fuse
  kernel by kernel, which made the surface of the Wilson operator differ from one process in the
  last bit. Results change at the rounding level; the CG on several processes may take a few
  iterations in a thousand more or fewer than on one (its sums go in another order).
- `su3_reconstruct_row3` and `c_mult`, unused (and the former wrong in one term), are removed.

## 0.7.0 (2026-10-24)

Wilson and clover HMC on the device, and every run reporting the work it did. Bridge++'s
test_HMC_Clover_RHMC_Nf2p1 (Ratio, Standard and Rational clover actions on stout links) now runs
on the GPU and gives the trajectory of the CPU and of Bridge++ (plaquette 0.627668186553636;
Bridge++ 0.627668186553629). Each solve and trajectory prints Flop, GFlop/s and Flop/byte by kind
of work, for a roofline estimate: a 16⁴ domain-wall trajectory on an A30 runs its fine operator
in single precision at 2.38 Flop/byte, 1570 GFlop/s in all. Parameter files are unchanged; the
library's public interface is narrower (a breaking change for code using its internals).

- Every solve and trajectory reports the work it did: Flop, GFlop/s and Flop/byte, split into the
  fine operators, the coarse operators, restriction and prolongation, and vector operations, in
  single and double precision (`src/meter.rs`). Each operator application and vector operation
  counts itself where it runs, as QUDA counts its kernels; the bytes follow QUDA's convention, so
  the intensity is an estimate for a roofline. A count costs about 3 ns on the host. The ad hoc
  GFlop/s of the domain-wall solve lines is replaced by it.

- HMC with Wilson and clover fermions on the device (`Main.backend : cuda_bridge`): the Standard,
  Ratio and Rational actions, Lexical or Even_odd, on one process. The clover term, its inverse and
  ln det T are made from the links on the device (`fermion/clover_links.cuh`; the clover operator
  no longer builds them on the host), and its force too. The clover force sums the insertions of
  all pairs before walking the leaves once, so a rational action walks them once instead of once
  a pole, on the CPU as on the device (`ForceOp::clover_derivative` takes weighted pairs; a
  breaking change for code implementing `ForceOp`). `Rational` is generic over the links.

- The public interface of the library narrowed to what programs and users are meant to use:
  1151 public items and 608 public fields became 893 and 478; the rest is `pub(crate)`, so the
  compiler reports what nothing uses. Internal pieces (site arithmetic helpers, the parallel
  loops, the force kernels of HMC, the steps of the multigrid setup, the kernel argument
  structures of the device) are no longer exported; parameter files and programs are unchanged.
  This is a breaking change for code that used those internals.
- What the narrowing showed unused, removed: `gauge::paths::path_product`, the chiral split and
  join of spinors, two pairs of `len`/`is_empty`, `HalfSpinor`'s conversions to and from reals;
  earlier the leftovers of the refactors (`on_several_processes`, `ColorMatrix::REALS`, ...).
- Every CUDA grid-stride loop goes through `GridStride` (in `size_t`); the same time as the
  hand-written loops.

## 0.6.0 (2026-10-24)

Domain-wall fermions on several GPUs: the propagators (with the multigrid, its levels and every
coarse solver), the eigensolvers and HMC (with stout smearing and the multigrid), each process
with its GPU, the operators giving the result of one GPU bit for bit. The eigensolvers and
pseudospectra on the device. Parameter files are unchanged; `Main.grid_size` now applies to
`backend : cuda_bridge` for domain-wall fermions. New feature `nccl`.

- Several GPUs for domain-wall fermions (`ruttice-mpi` with `cuda-bridge`, one GPU a process):
  the hops pack the faces of the local lattice into half spinors (`fermion/hops.cuh`, `Halo`),
  which travel while the interior is computed, the surface summed in the order of one process,
  so the operators give one process's result bit for bit; the reductions of the fields on the
  device are summed over the processes. Stout smearing, its chain rule, the gauge force and
  action and the fermion force read an extended field of links (`gauge/stout.cuh`, a halo of one
  or two layers, corners included, as `gauge::paths::Extended`). The faces travel by NCCL
  (feature `nccl`) or through the host (`backend::cuda_bridge::halo`), NCCL where every process
  can have it. Propagator solves without the multigrid, eigensolvers, pseudospectra and HMC
  without the multigrid on several processes; checked against one process with `LocalComm`
  (up to 16 ranks on one GPU) and with two MPI processes (CG, mixed CG, eigenvalues, the HMC of
  a smeared Ratio action with α: the same iterations and ΔH).
- The multigrid of domain-wall fermions on several GPUs, for the propagators and HMC: the
  aggregates within a process, the coarse operator probed across the cuts (the faces of the
  probing vectors exchanged), its hops reading ghost sites filled by an exchange of the faces of
  the coarse field before each application (`coarse_pass_kernel` with a ghost field), the coarse
  fields summed over the processes. P, the coarse operator and its applications are those of one
  process bit for bit; with two MPI processes the propagator and HMC take the same iterations.
  The levels below the coarse lattice too: each level a lattice cut as the one above (a
  super-block within a process), its operator probed through ghosts with colours by the global
  coordinates, the last level gathered over the processes and inverted whole on every GPU; bit
  for bit those of one process. The exact adjoint of the second factor (a `Coarse_Solver_PV`
  other than MINRES) too: the links of the ghost sites exchanged once at setup, their conjugate
  transposes made on the host. Checked with two MPI processes on the GPU and on the CPU, whose
  multigrid of domain-wall fermions (levels included) also runs on several processes: the same
  iterations as one process.

- The eigensolver files on the device (`Main.backend : cuda_bridge`): Lanczos and Arnoldi on
  Wilson, clover and domain-wall fermions, every `fermion_mult_mode` and `operator_type` of the
  CPU (`PV_right`, `PV_left`, Neff's α). The run is written once (`app::eigen`), a backend
  supplying its operators and the start vector moved to it; on 4³×8 the CPU and the device take
  the same restarts and agree to 1e-14. The checks of `Main` every device run needs are one
  function (`MainBlock::check_cuda_bridge`).
- Pseudospectra from the eigensolver files: a `Pseudospectrum` section (`points`, `circles`,
  `lines`) adds σ_min(z − A) at each z for the operator of `Eigensolver`, on the CPU and the
  device (`eigen::resolvent_norms`, until now reached only from
  `examples/preconditioned_fov_device.rs`). `examples/pseudospectrum_mdwf.yaml`.
- The CUDA code extracted from Bridge++ but never called (domain decomposition, an older
  transfer and coarse operator, multi-right-hand-side Wilson, APE and plaquette on the device,
  and others; 460 lines) is removed.
- `docs/implementation/`: the architecture, how to write a GPU kernel, every CUDA kernel derived
  line by line, and several GPUs with an evaluation of NVSHMEM.

Not checked: NCCL itself and several real GPUs (the test machine has one GPU, so the faces went
through the host), and the speed of the exchanges.

## 0.5.0 (2026-10-24)

The domain-wall fermions the same on the CPU and on the device: each algorithm written once,
against traits, a backend supplying operators, fields and kernels (docs/DEVELOPMENT.md, "One
algorithm, two backends"). Every option of the propagator and HMC files for domain-wall
fermions now does the same on both. Parameter files are unchanged; the library's interfaces of
the mixed solvers and of the HMC multigrid setup changed (`DwfMultigridSetup::new` takes an
`MdwfMultigridSettings`).

- The multigrid of HMC has one driver (`hmc::dwf_multigrid::with_solves`): when to set up,
  bootstrap or rebuild, the cache, the retries before CG, and the solves with D (Neff's α by the
  left form and Q) and with D† (Γ5 M Γ5). A backend (`Backend`: `OnCpu`, the device's
  `OnDevice`) supplies the test vectors, P and the coarse parts, and the cycles made of them.
  The CPU gains what only the device had: Neff's α, the levels below the coarse lattice (kept in
  the cache with the coarse operators: `CoarseParts` shared, `PvEoCycle::parts`), and
  `fine_precision : single` (the operators converted with `Mdwf::converted_from`).
- `basis_precision` on both, in the propagator runs and in HMC: `PvEoBasis` keeps P in half or
  single precision (`BasisStorage`); before, the CPU always kept it in half and HMC refused the
  key.
- Mixed precision written once (`fermion::SingleAll`, `fermion::SingleEven` over a backend's
  `SingleOperator`): the CPU gains the mixed CGNE of the propagators (with reliable updates, in
  the SIMD layouts too) and mixed CG with Neff's α on the even-odd Ratio; `SingleMdwf`,
  `SingleMdwfEo` and the device's `SingleCuda` are gone.
- The setup of the multigrid is one for the propagator runs and HMC: the settings of the file
  (`MdwfMultigridSettings`, `MultigridBlock::mdwf_settings`), and the test vectors (from a
  checkpoint or the filter, the bootstrap), P, the levels, the coarse parts and the tops of the
  smoothers by the backend (`MdwfBackend::rebuild`, `cycles`), the same steps on the CPU and the
  device. The propagator runs give the same numbers as before (bit for bit on the device).
- The device's multigrid of HMC with `fine_precision : single` probes the coarse operators and
  takes the tops of the smoothers in single precision, as the propagator runs and the CPU did
  (it probed in double before): on 16⁴ the same iterations, ΔH within 1e-7, the rebuild
  2.75 → 1.64 s. With the cycle's operators converted to single precision, Neff's α is kept
  (`CudaMdwf::converted_from` takes the parameters only).
- The device gains the even-odd Standard action of domain-wall fermions (`TwoFlavorEo` over any
  links; `EvenOddForceOp` without the host links).
- An HMC file with `Main.backend : cuda_bridge` is refused by a program built without the
  feature cuda-bridge, as the propagator files are. Before, it passed the checks of the device
  and ran on the CPU without Neff's α and the levels.
- HMC with APE smearing on a smeared fermion action is refused when the file is read; the force
  needs the derivative of the smearing, which stout has. Before, both builds panicked at the
  first force.

## 0.4.1 (2026-10-23)

Faster setup and prolongation of the domain-wall multigrid on the device. HMC on 24⁴ with stout
links at m 0.001 (one short trajectory, 20 test vectors, levels below, A30): 242 → 203 s, the
setup 141 → 113 s.

- The prolongation P for one vector (`fold_prolong_tiled_kernel`): a block for 32 fine sites, a
  site's threads reading P in 32-byte sectors, the field written whole through shared memory;
  bitwise the same results. `examples/fold_bench`, 24⁴, Ls 8, 20 test vectors: 15.9 → 7.3 ms.
- The bootstrap of HMC takes the levels below the coarse lattice for its coarse solves, from its
  second round on (`device_correction_with_levels`): the second round's vectors 45 → 26 s. In
  the first round they gained nothing and cost 7 s to make.
- The coarse operators are probed with as many columns at a time as a quarter of the free device
  memory holds (between 0.5 and 3 GB; `device_room`), so that P is read once for several: on
  24⁴, P†DP in single precision 6.5 → 4.4 s.
- P made orthonormal on its aggregates by CholeskyQR2 (`fold_orthonormalise_cholesky_kernel`:
  twice the Gram matrix, its Cholesky factor and W R⁻¹, two sweeps over the vectors a pass
  instead of one for every pair), with as many coarse sites a launch as an eighth of the free
  device memory holds (up to 2 GB): the same Q as modified Gram-Schmidt, and P, kept in half
  precision, bitwise the same on 24⁴. Where a pivot is below 1e-12 of its diagonal (nearly
  dependent vectors), Gram-Schmidt does those sites. 24⁴, 20 test vectors (40 columns):
  making P 4.0 → 1.8 s.
- HMC with the domain-wall multigrid does not stop where the multigrid stalls: the solve is
  done again with P and the coarse parts made for the links of now, then after a bootstrap as
  well, each from the solution so far; if it still stalls, CG finishes it (at least 50000
  iterations, whatever the file's limit for the GCR of the multigrid), with a warning. Found on
  a trajectory of length 1 from a quenched 24⁴ configuration at m 0.001 (ΔH 4e9, the links far
  from where P was made), where the sweeps stalled at |r|²/|b|² 7e-11 and the run panicked.
  The same on the CPU.
- CG with Neff's α on the even-odd Ratio on the CPU too, in double precision (0.4.0 had it on
  the device only).
- The note: why Neff's α helps CG more than the multigrid (K^(α) = Q_e⁻¹ K Q_e), over setups S
  and B.

## 0.4.0 (2026-10-23)

The multigrid in HMC. The domain-wall Ratio action over Pauli-Villars fields solves with the
multigrid on the device: D† by the multigrid of D (no second setup), the levels below the
coarse lattice, Neff's α, and the memory to run 24⁴ on the A30. On 24⁴ with stout links at
m 0.001 (one short trajectory, 5 solves, 20 test vectors, levels below) the multigrid takes
242 s with its setup (141 s), mixed CG on the even-odd Ratio 264 s at α 0.6 and 301 s at α 1;
the solves alone 63 s against 264 s. ΔH the same.

- CG on the even-odd Ratio takes `Fopr.neff_alpha` on the device: with K^(α) = Q_e⁻¹ K Q_e,
  (K†K)⁻¹ = Q_e ((K Q_e)†(K Q_e))⁻¹ Q_e, CG on the normal equations of K Q_e
  (`fermion::AfterQ`, `TwoFlavorRatioEo::with_neff`), mixed precision too; the action and its
  force are those of K. 24⁴, stout, m 0.001, mixed CG: 40912 → 31996 iterations, 301 → 264 s
  at α 0.6, ΔH the same to six digits. On the CPU refused.
- HMC with the domain-wall multigrid on the device takes `Fopr.neff_alpha`: the solves with D
  are those of D^(α) = D Q in the left form (Ã_L^(α), as in the propagator runs), x = Q x^(α),
  with the setup, P and the coarse operators of D and the smoother on the range of the left
  form; the solves with D† are unchanged (D^(α)† = Q D†). Also for the even-odd Ratio. On 24⁴ (stout, m 0.001, 20 test vectors, levels below) α = 0.6 gives the
  same ΔH as α = 1 and about the same iterations (362 against 356), as α does in the
  propagator runs. 20 test vectors are the fewest that converge there (16 stagnate, also
  filtered from 24 or with the left vectors Ã y in P): setup 141 s, solves 63 s, 19.1 GB.
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
- The examples on the device only (`dminus_cost`, `dual_basis`) need the feature
  `cuda-bridge`, so that `cargo check --all-targets` passes without it.
- `examples/hop_gemm_bench`: the even-odd hop of the Möbius operator on the device alone, its
  time and checksums.

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
