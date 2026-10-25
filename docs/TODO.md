# To do

Work that is known and not done, so that the next person (or session) can pick it up without
rediscovering it. Each item says why, where, how, how much, and how to check it. Nothing here
changes results; items marked *performance* change speed, the others only the structure.

Priorities: finishing the multigrid paper (`docs/mg_background/MDWF_MULTIGRID_NOTE.tex`) comes
first; the structural items after it, in the order given. Last reviewed 2026-10-21 (the
paper's numbers at 6d5bf18, released as 0.3.2).

## The aim of 1.0 (decided 2026-10-21)

Production MDWF propagators and HMC, on one GPU and on several, that people outside can use
with confidence:

- **Parameter files**: the format frozen and documented in full (`docs/USAGE.md`); after 1.0
  an incompatible change only in a major version.
- **Correct**: propagators, HMC (ΔH, reversibility) and the ILDG I/O checked against Bridge++,
  Grid and QUDA; the multigrid reliable for D and for D† (the dual cycle of "Known, not
  understood" solved).
- **Multi-GPU**: the device backend over several GPUs (halo exchange of the fine operator, the
  coarse operators and the levels below), with the single-GPU results reproduced.
- **Fast**: the multigrid clearly faster than mixed CGNE on one GPU, and on an A100 at least
  comparable with Grid's HDCG, or a documented reason why not.
- **Engineering**: CI green on the CPU paths; running out of device memory a readable error,
  not a panic; published on crates.io.
- **The paper**: the MG note on arXiv as the reference for the method.

Not needed for 1.0: autotuning, CUDA graphs, a QUDA backend.

## Release 0.4.0: the multigrid in HMC (decided 2026-10-22)

0.4.0 is released when HMC with the domain-wall multigrid is usable and compared fairly with CG:

1. **Runs**: HMC with the multigrid on 24⁴ on the A30 with room to spare (20.5 GB with GCR(2)
   since the solves with D† use Γ5 M Γ5, 2026-10-22).
2. **Neff's α in the multigrid of HMC** (CG has it, so the comparison is not fair without it).
   The right form Ã_R, which HMC uses, does not depend on α (Sec. V of the note); the solves
   must move to the left form Ã_L^(α): for D the correction of the propagator runs with the
   factors of Q (Table "factors"), for D† the dual correction of Ã_L^(α)† has to be derived
   (lifts, factors, coarse solves) and implemented. Then the refusal in `hmc_params.rs`
   ("without Fopr.neff_alpha") goes.
3. **Faster**: on 24⁴, m 0.001, a trajectory with the multigrid (setup and rebuilds included)
   faster than with mixed CG, both at α 0.6, measured. The setup (87 s on 24⁴) and the rebuilds
   are part of that.
4. **Correct**: the HMC tests with the multigrid give the trajectories of CG (plaquette, ΔH),
   with α too.
5. **Documented**: USAGE says how to set the multigrid of HMC (GCR(8) or what replaces it,
   the cache, `fine_precision`, α); the note has the HMC results.

**HMC cost, measured (2026-10-23):** on a dynamical 16⁴ configuration (thermalized from the
quenched one: 60 trajectories at m 0.05, 40 at m 0.01, `therm16/` on as13) CG needs about 300
iterations a solve at m 0.01 and at m 0.001 (finite volume), and the multigrid is 3.4 (m 0.01) and
4.2 (m 0.001) times slower a trajectory; tuning (smoother 4/6/8, coarse 1e-1, inner 1e-14,
bootstrap 1 round, GCR(4)) gains at most 6%. The 24⁴ quenched configuration, where CG needs 6400,
blows up in a trajectory of length 1. So there are no data yet that the multigrid is faster in
HMC: the volumes thermalized here are finite-volume limited (the note, "In HMC"). Next: a volume
where CG needs more than about 1300 iterations a solve.

**Released 2026-10-23** on the user's call, with: 1 done (19.1 GB live with 20 test vectors);
2 done (the left form for D, D† unchanged; CG on the even-odd Ratio takes α too, on K Q_e);
3 measured on one short trajectory only (24⁴, m 0.001, length 0.01: multigrid 242 s with its
setup against mixed CG at α 0.6 264 s) — a trajectory of length 1 is still to be run; 4 by ΔH
on the device runs only (the HMC tests with the multigrid and the CPU path were not run: as13's
CPU is shared); 5 USAGE yes, the note's HMC results not yet. After 0.4.0: optimization (item 20).

**0.4.1 (2026-10-23)**: the setup and prolongation faster (24⁴ short trajectory 242 → 203 s),
the retries before CG where the multigrid stalls (device and CPU), CG with α on the even-odd Ratio
on the CPU, the note's α discussion and HMC results, the walkthrough. Checked: both builds compile
clippy-clean, the device HMC on 16⁴ gives ΔH and iterations of before bit for bit. Not run: the
CPU paths (the retries and α on the CPU compile only), the HMC tests.

**0.5.0 (2026-10-24)**: the domain-wall fermions the same on the CPU and the device
(DEVELOPMENT.md, "One algorithm, two backends"). Checked on 4³×8 (`conf_04040408.txt`, Ls 4,
4 CPU threads): the propagators (CGNE, mixed CGNE, the multigrid with P in half and in single
precision) give the same iterations, coarse applications and residuals on both; HMC (CG, mixed CG
with α on the even-odd Ratio, the even-odd Standard action, the multigrid with α, levels,
`fine_precision : single`, `basis_precision : single` and the cache, Lexical and Even_odd) the
same ΔH and iterations on both, the multigrid's ΔH that of CG to 1e-10. The propagator runs as
before (the device's checksums of every stage bit for bit). The device on 16⁴ (`therm16/tune`):
CG bit for bit; the multigrid with `fine_precision : single` the same iterations, ΔH within 1e-7
(its coarse operators now probed in single precision). Tests run: CPU lib, hmc, mdwf,
multigrid, run, simd; device cuda_bridge — all pass.

**Eigensolvers on the device (2026-10-24)**: `app::eigen` written once over `EigenBackend`
(CPU, device). Checked on 4³×8, CPU and device: clover H (Lanczos, Bridge++'s 0.152031949234),
Wilson D†D, clover D (Arnoldi), domain-wall D† (Arnoldi), `PV_right` D (Arnoldi), `PV_left` D†D
with α 1.3, `PV_right` DD† after stout — the same restarts and applications, eigenvalues within
1e-14. Still CPU only: the gradient flow, the Wilson multigrid, MPI. (The Wilson and clover HMC
and RHMC came to the device on 2026-10-24, below.)

**Wilson and clover HMC on the device (2026-10-24)**: DESIGN §7.8 step 1. The clover term and its
inverse made from the links on the device (`fermion/clover_links.cuh`), the clover force by an
insertion summed over the pairs and one leaf walk (a rational action walks the leaves once, not
once a pole; on the CPU too), ln det T for the even-odd action. Checked: the derivatives of the
device against the CPU to 1e-16 relative, ln det T to 1e-15; Bridge++'s test_HMC_Clover_RHMC_Nf2p1
on the device gives the CPU's trajectory (plaquette 0.627668186553636, Bridge++
0.627668186553629) and so do its even-odd clover Standard and a Wilson even-odd Ratio variant.
Next (§7.8): Wilson and clover on several GPUs, then their multigrid on the device.

**Wilson and clover on several GPUs (2026-10-25)**: DESIGN §7.8 step 2. The operators are the
hop kernel (`FourD`, `Clover`), with its interior and surface parts and `Halos`; the clover
blocks and force read extended links. Checked with `LocalComm` on 1×1×1×2, 2×1×1×2 and 2×2×2×2
against one process: D, D†, γ5 D, the hops, T, T⁻¹ and the HMC derivatives bit for bit, ln det T
to 1e-12, CGNE within 0.2% of the iterations; broken once on purpose (surface skipped). The
fused multiply-adds of a leg had to be fixed (CHANGELOG). Not run yet: two MPI processes for
Wilson and clover HMC (as was done for domain-wall fermions). Next: their multigrid on the device.

**Measurements (2026-10-25)**: DESIGN §7.10. Meson correlators with momentum and the proton
correlator, checked against Bridge++'s numbers; connected three-point functions by a sequential
source through the sink, for mesons and the proton, checked by the conserved vector current
(exact Ward identity) and by the sequential source through the current; in the parameter files
as `Measure.momenta`, `proton`, `three_point`. Open: the current on the antiquark line of a meson (swap the two
propagators of the sink; not tested); three-point functions on several processes (written for
them, no `LocalComm` test); the conserved current of domain-wall fermions (five-dimensional);
g_A and other charges against published numbers.

**The Wilson and clover multigrid on the device (2026-10-25)**: DESIGN §7.8 step 3; the device
now does every fermion, solver and multigrid of the CPU on one GPU or several, but these:
the levels below the coarse lattice and `precondition_diagonal` of the Wilson multigrid
(refused when the file is read), the gradient flow, boundary phases other than ±1 and 0. Checked
against the CPU (iterations, correlators, HMC trajectories) and several processes against one
(P bit for bit, the same iterations). Not checked: NCCL and several real GPUs, two MPI processes
for the Wilson and clover multigrid (LocalComm only), the speed of the device multigrid beyond
4³×8 (its P is made on the host: a setup on a large lattice spends its time there).

**Several GPUs (2026-10-24)**: domain-wall fermions without the multigrid on several GPUs
(CHANGELOG). Not checked: NCCL itself and several real GPUs (as13 has one A30; there NCCL is
refused for ranks sharing a device and the host path runs), the speed of the exchanges (on one
GPU shared by two processes every exchange pays a context switch: HMC with the multigrid 48 s
against 1.6 s). The multigrid, its levels and the exact adjoint of the second factor run on several
GPUs, and the CPU's multigrid on several processes (checked with two MPI processes: the same
iterations). NVSHMEM: evaluated, not implemented (docs/implementation/MULTI_GPU_NOTE, Sec. 4.2: the coarse
ghosts are latency-bound, an estimated 1.5-3x faster coarse solve; nothing to gain on the fine
operator of a large local lattice); to be measured on a machine with several GPUs, after checking
whether NCCL's own symmetric memory does the same. Notes:
`docs/implementation/` (the layout of the faces, coalescing, the transports, Bridge++, every kernel).

## Structure

### 1. One propagator driver for the CPU and the device (done 2026-10-24)

What follows the setup is written once (`solve_sources`, 2026-10-19), and since 2026-10-24 the
setup too: the settings (`MultigridBlock::mdwf_settings`), the test vectors, P, the levels and
the coarse parts by the backend (`MdwfBackend::rebuild`, `OnCpu`, `OnDevice`), the same as HMC.
The drivers keep their reports and, on the CPU, the smoother in the SIMD layouts. Checked: the
propagators on 4³×8 (CGNE, mixed, the multigrid in PV_left with α, Normal_left, the levels,
P in half and single, the cycle in single, SIMD) the same iterations, coarse applications and
residuals on the CPU and the same checksums of every stage on the device as before.

### 2. One solver dispatch (done 2026-10-19)

`SolverBlock::solve` (`src/app/params/blocks.rs`) is the one `match` on `SolverType`; the CPU and
device propagator drivers call it, and item 1 should too. `src/app/wilson.rs` keeps its own: it
reads another configuration (`SolverConfig`, `SolverKind`) with other solvers and the even-odd
split inside the dispatch.

### 3. Split `src/app/hmc.rs`

- **Why.** 1600 lines holding three things: the configuration types, their validation
  (`HmcConfig::check`, about 250 lines) and the construction of the actions (`terms`, about 210;
  `device_terms`). The HMC parameters also exist twice: `app/hmc_params.rs` reads the file into
  its blocks, which are then converted into `HmcConfig` (the same two-layer pattern as
  `params::Params` to `config::Config` for the Wilson propagators).
- **How.** `app/hmc/{config,check,build,run}.rs`; whether the two layers can become one is a
  separate decision.
- **Effort.** Medium. Mechanical apart from the second point.
- **Check.** `cargo test --release --test hmc`, and `--features cuda-bridge` for the device terms.

### 4. Move the CUDA FFI declarations out of `src/backend/cuda_bridge/mod.rs` (done 2026-10-19)

`cuda_bridge/ffi.rs` holds the `extern` functions, the `#[repr(C)]` structures and `DeviceReal`;
`mod.rs` (450 lines) keeps the buffers, geometry and helpers.

### 5. Argument parsing of the examples

- **Why.** Every example parses its command line with its own `value` closure and positional
  arguments (`examples/preconditioned_fov_device.rs` has twenty options this way).
- **How.** A small shared module for the examples (`#[path]` include), or the examples that are
  used as tools turned into `apps/` with a real parser.
- **Effort.** Small to medium.

### 6. Parameter structures instead of argument tuples

- **Why.** 27 `#[allow(clippy::too_many_arguments)]` in `src/`; arguments are grouped as tuples
  (`(d, pv)`, `(even, odd)`, `(exact, hi)`) where a named structure would say what they are.
- **Effort.** Small each; do it when touching a function anyway.

Not an issue: `src/hmc/mod.rs` is 1600 lines, but about 130 of them are code and the rest its
tests.

## Performance

### 7. CPU multigrid: the coarse operator build

- **Why.** `CoarseOp::build` is most of the CPU setup (about 134 s per `PvEoCycle::new` on 12⁴,
  three times with two bootstrap rounds: about 400 of 681 s; the device does the whole setup
  in about 14 s). It builds one component of a coarse site at a time with the unfused
  `HopTerms`, and sorts the hops in a serial loop over the fine sites
  (`src/solver/preconditioner/multigrid/coarse.rs`).
- **How.** Several components at once as the vectors of the fields, the sorting in parallel
  (`for_sites`), the restriction of a crossing hop reading only the faces of the aggregates (as
  the device does).
- **Partly done 2026-10-20.** Blocks of more than two sites took D of φ kept on one face for
  each of the eight hops (nine applications of D a batch); now the hops of `HopTerms` like the
  smaller blocks, through `HopTerms::each_hop`, which `Mdwf` overrides to make w = b ψ − c Y ψ
  once for the eight and to compute each hop on the face it is restricted from only
  (`WilsonOp::hop_term_where`). `examples/coarse_build_bench.rs` (8⁴, 4⁴ blocks, Ls 8, dim 64,
  24 threads): 6.0 s against 3.7 s a build, D_c the same bit for bit. Left (perf): the one
  application of D (15%), Q and the fifth dimension of w (21%), the hops (11%), the
  restrictions (10%).
- **Check.** Coarse operators equal to rounding (`--test multigrid`); setup time of
  `examples/mdwf_pv_eo_mg.yaml`; `cargo run --release --example coarse_build_bench`.

### 8. CPU multigrid in single precision above the coarse lattice

- **Why.** `PvEoCycle` and the CPU's setup fix the fine fields to `f64`. On the device single
  precision kept the outer iterations and the correlators; on the CPU it would double the SIMD
  width and halve the memory traffic.
- **How.** Make `mdwf_even_odd.rs` generic over the fine storage (`low_modes`,
  `bootstrap_rounds`, `lifted_vectors`, `TwoGrid`, `coarse_solve` already are, and serve both
  backends since 2026-10-25). While there: `PvEoCycle` has its own cycle and
  `coarse_correction`, the device uses the generic `TwoGrid` and `DeviceCorrection`; one
  implementation needs the coarse operator behind a trait (`schur`, `schur_source`,
  `schur_solution`, `even_block_inverse`, `apply`, `gamma5`, which `CoarseOp` and
  `DeviceCoarseOp` both have). It is the last part of the domain-wall setup and cycle still
  written twice.

Not worth doing on the CPU: the prolongation in half precision; reading it once for several
vectors (the CPU already loops over the vectors inside a site).

### 9. Device: the coarse operator kernel (done 2026-10-20)

`coarse_pass_kernel` had a thread for each row of a site's matrices and read the rows of a warp
2 dim reals apart; it now has a warp for each row (`cuda_bridge/README.md`): 3.59 ms against
1.13 ms per application on 2592 sites of dim 64 in single precision on an RTX 3080. Setup of
16⁴ with Level2 + Level3 of 2: 45 s against 35 s; 12⁴: 18 s against 17 s, the solve 1.2 s
(15 iterations) against 0.9 s (14). The checksums differ from earlier runs from the first
bootstrap vector on (another order of the sums); runs agree with each other. On the A30
(24³×48 setup S, 7009fbb = items 9, 10 and the filter step of 15, without `RUTTICE_CHECKSUM`,
one run each, 2026-10-20): setup 391 s against 368 s (filter 85 s either way; bootstrap rounds
106 and 125 s against 101 and 111 s; levels 15 against 11 s), solve 25.4 s (27 iterations)
against 21.5 s (27).

### 10. Device: the work fields from a memory pool (done 2026-10-20)

Every `DeviceBuffer` was a `cudaMalloc`, and every drop a `cudaDeviceSynchronize` and `cudaFree`;
the cycles make and drop several for every application of a preconditioner (`solve_shared`,
`DeviceMultilevel::vcycle`, a solver made for every coarse solve). They now come from the
stream-ordered pool of the device (`src/backend/cuda_bridge/mod.rs`, `pool`). 12⁴ setup and
one solve: 5340 `cudaMalloc` (2.6 s) and 5357 `cudaDeviceSynchronize` against 5319
`cudaMallocAsync` (0.19 s) and 21; setup of 16⁴ with levels 35 s against 30 s, of 12⁴ 17 s
against 15 s; the checksums the same bit for bit. Not seen to run: the path where the device
refuses and the pool gives back what it keeps (WSL pages device memory to the host instead).

### 11. Device: the bootstrap one vector at a time

- **Why.** `bootstrap` (`src/backend/cuda_bridge/multigrid.rs`) runs FGMRES with the cycle on
  one test vector after the other, so P (8 GB in half precision on 24³×48), the coarse matrices
  and the links are read once per vector; the kernels take several vectors in one launch. The
  bootstrap is about two thirds of the setup (12⁴: rounds of 7–8 s in 18 s).
- **How.** Two to four vectors at a time (the solvers already take sets with a coefficient per
  vector). Device memory on 24³×48 (about 22 of 24 GB) limits how many; the cycle in single
  precision helps.
- **Check.** Setup time; the test vectors to the tolerance of the solves.

### 12. The Gram matrix one pair at a time (done 2026-10-20)

`gram` copied a vector with `select` for every pair and reduced each pair alone (1024 dot
products and 1024 copies of fine vectors for 32 vectors). It is now `Subspace::gram`: on the
CPU one sweep over the sites (a site holds every vector of a set), on the device
`blas_gram_kernel`, tiles of 4 × 4 pairs in double precision in a fixed order (a vector read
once for every four of the other set). 16⁴ with levels: setup 30 s against 29 s, the same
iterations; tests `gram_matrix_in_one_sweep` (`tests/multigrid.rs`) and in
`tests/cuda_bridge.rs`. Not measured on the A30 (estimated about 2 s for each of the six Gram
matrices of a setup before).

### 13. Device: the coarse operator made on the host (done 2026-10-20)

`CudaMdwf::coarse_operator` brought P† of each of the eight hops and of the block to the host
for every batch of components (`DeviceCoarse::values`), put them into the matrices there, and
`DeviceCoarseOp::new` sent them back (in double precision, then converted); the unit vectors
of a batch were made on the host too. Now `coarse_operator_on_device`: the matrices are written
on the device as they are made (`coarse_columns_kernel`, the links directly in the precision of
the coarse solves; `coarse_units_kernel`), and only the tables of the coarse lattice (its
neighbours and parities) go up from the host. The checksums of P†DP and of the whole run the
same bit for bit; setup of 16⁴ with levels 29 s against 27 s, of 12⁴ 15 s against 14 s. The
device holds the links while the fine fields of the build are there: peak of the bootstrap on
16⁴ 1.76 GB against 1.68 GB (estimated about 0.3 GB more on 24³×48). `coarse_operator` (on the
host, for the tests) is the same matrices brought down once.

What the setup and the solves still move between host and device, on purpose: the test vectors
wait on the host during each bootstrap round (`Stashed`, to leave the device to P and the cycle:
about 16 GB a round on 24³×48 over pageable memory; page-locked memory would make it faster);
the right-hand side of a solve waits on the host during it (one field); the results of every
reduction (a few numbers); the tables of the coarse lattices and the coefficients of a change of
basis (kilobytes); sources and propagators.

### 14. Device: CGNE

Mixed CGNE on 16⁴ (setup S, m = 0.01, RTX 3080): 355 iterations, 0.7 s, about 2.0 ms an
iteration; in double 349 iterations, 4.7 s (the RTX 3080 has little FP64). What an iteration
should move (four passes of the Schur factor and the vectors of CG) is about 0.5 GB, 0.66 ms
at the bandwidth of the card: it runs at about a third of it (on the A30, 24³×48: about 21–24 ms
an iteration against about 5.5 ms). Nsight Compute (`ncu` works under WSL; nsys does not trace
kernels there), one iteration:

- `hop_gemm_kernel` four times, 215 µs each, 112 MB from DRAM where about 62 MB is needed
  (input, output, links): the neighbours read again from DRAM.
- `fifth_gemm_columns_kernel` about three times, 85 µs: `B Q` before the hops of K (twice) and
  `(C Q)⁻†` before K†, passes of their own. **Done 2026-10-20:** two of them. `B Q (C Q)⁻¹ =
  B C⁻¹`, so the matrix of the second hops of K goes into the product after the first
  (`EvenOdd::inverse_hops_twice`, overridden by `CudaMdwf` with α; `Matrices::pre_inv`): two
  passes fewer a K†K, 355 iterations as before, about 85 µs an iteration less (16⁴ mixed 0.7 s against
  0.6 s). Without α there was no such pass. Every Schur factor of the left form with α on the
  device has it (the outer operator of the multigrid too).
- `blas_caxpy`, `blas_caypx`, `reduce<AxpyNorm>` at 84–88% of the bandwidth;
  `reduce<Norm>` (|A p|² of `Normal::mult_dot`) at 30%: 114 µs for 25 MB. Probably the
  RTX 3080's FP64 (1/64 of FP32: two conversions and two FMAs in double for every complex
  number), as `reduce<AxpyNorm>` takes as long for three times the bytes; not expected on the
  A30.
- The kernels make about 1.7 of the 2.0 ms; the rest is launches and the waits of the
  reductions.

`hop_gemm_kernel` itself (`ncu`, 16⁴): 128 registers a thread, 256 threads and 25.6 KB of shared
memory a block, so two blocks on an SM: 16 warps, a third of the RTX 3080's 48 (a quarter of the
A30's 64), 70% of the bandwidth; 74–104 MB read where 38–63 MB is needed (L2 hits 53–63%), 5.3
sectors a request (coalesced). Fewer registers by `-maxrregcount` spill and are slower (96: 2.5
times, 80: 5.6 times the time): for a kernel that waits on memory, more warps bought with local
or shared memory move more bytes and gain little. What helps is fewer bytes (the neighbours
read again from DRAM: an order of the sites that keeps them in L2; the fields in half precision
in the inner solve; the links of several right-hand sides read once) and more loads in flight
from each thread (all the neighbour loads of a site issued before the arithmetic, wider loads).

On the A30 (24³×48 setup S, m = 0.001, 702c471, 2026-10-20; `ncu` locks the clocks and flushes
the caches, so its times are longer than those of a plain run): 21.7 ms an iteration in a plain
run (about the RTX 3080's 2.0 ms on 16⁴ times the volume). `hop_gemm_kernel` 5.4 ms a launch at
27% of the DRAM bandwidth, 12% active warps, 1.30 GB a launch, L2 hits 55%: four of them are
about three quarters of an iteration. `fifth_gemm_columns_kernel` 1.12 ms at 48%; the BLAS
kernels at 73–85% (`reduce<Norm>` too: its 30% on the RTX 3080 was the FP64 of that card). So
on the A30 the hop kernel is the whole question: it waits on memory, at 27% of the bandwidth
and with 1.30 GB a launch where about 0.64–0.89 GB is needed (input, output, links, and the
field it is subtracted from where there is one). At the bandwidth of the BLAS kernels and with
the bytes it needs it would take about 1–1.5 ms instead of 5.4, and an iteration about half its
time. That is the next step for CGNE (and for the smoother and outer operator of the multigrid,
which use the same kernel); see above for the means.

On the A30 at 4826b93 (24³×48 setup S, point source, 12 columns, 2026-10-20): CGNE in double at
m = 0.001 3613–3619 iterations a column, 109 s each (30.2 ms an iteration), 1333 s in all; the
multigrid 637 s (setup 329 s, 23.8 s a column), so 2.1 times faster with its setup. PS of the
two agree to about 1e-8; CGNE at t = 0 is Bridge++ 2.1.3's to the digits printed.

**Done 2026-10-20, the hop kernel on the A30.** Its stalls (`ncu`, A30, 24³×48 mixed CGNE): nvcc
gave `hop_gemm_kernel<float>` 168 registers for sm_80 (128 for sm_86, by chance), so one block of
256 threads on an SM, 8 warps of 64; and the second hop of K, the one with `d_from`, took 6.79
ms against 4.13 for the first (long scoreboard 4.0 against 1.0): `d_from` was read when the
result was written, its latency after everything else. Now: `__launch_bounds__(256, 2)` for
the float kernel (128 registers, no spill for sm_80 and sm_86 by `ptxas -v`; in double 128
spills 4 KB, so none there), `d_from` read before the product of the fifth dimension, and
without α `B C⁻¹` as the matrix of the first hops of K (`Matrices::pre_inv`), so the second
read one spinor of a neighbour and not three. A30, mixed CGNE, one right-hand side: α = 0.6
83.1 → 59.1 s (4012 iterations both, 14.7 ms an iteration), α = 1 130.3 → 89.6 s (6237 → 6157
iterations: B in the matrix rounds differently in single precision). A launch 4.13/6.79 → 3.18/
3.37 ms, active warps 12 → 25%. It is no longer waiting on DRAM (39–46% of it): it issues
(math pipe throttle 1.35, not selected 1.55), 367 million instructions a launch. Next: fewer
instructions (every one of the Ls threads of a site rebuilds the third row of the same links;
the index arithmetic), then the fields in half precision.

Tried and not kept (A30):
- The blocks in chunks of time slices, so that the neighbours in t are still in L2: DRAM read
  of a launch 894 → 464 MB, L2 hits 60 → 74%, the time the same. Worth trying again once the
  kernel waits on DRAM.
- Q made by the hop kernel as it reads (`(b − c Y) Q ψ`), instead of the `B Q` pass before the
  first hops of K with α: that hop 3.18 → 4.52 ms against the 1.11 ms of the pass saved, so
  2.6–3.8% slower in all (mixed α 0.6 and 1, double α 0.6), though 1.2 GB less device memory.

**Done 2026-10-20, mixed precision by reliable updates.** The defect correction restarted CG
from zero at each correction: on 24³×48 at m = 0.001 (point source, a5df341) 5261–5951
iterations a column against 3613–3619 in double precision, and two of the 12 columns stopped at
|Dx−b|² 4e-15 for a target of 1e-18 (the rule is on the residual of K†K; an error in the low
modes meets it while |Dx−b| does not), their PS 3e-6 off. Now one CG in single precision with
the solution and the residual made again in double precision each time |r|² falls by 1e-2
(`solver::reliable_cg`, `SingleCuda::reliable_normal`; `convergence_criterion_squared_inner`).
16⁴ point source, RTX 3080: at m = 0.001 365–374 iterations a column, the same as double
precision (404–495 before), |Dx−b|² 4e-18–8e-18 as in double (8e-20–7e-17 before), PS 2e-11
from double (2.5e-8 before), 0.9 s a column against 5.6 in double. HMC keeps the defect
correction (`hmc::mixed`).

On the A30 (24³×48, m = 0.001, 85fea29) only half of it holds. The point source at α = 0.6:
4220–4574 iterations a column (5261–5951 before, 3613–3619 in double), 65 s (80.7), 804 s for
the 12 columns (994), PS 2e-7 from the multigrid (3e-6); but |Dx−b|² ends at 5e-17–3e-16, not
1e-18. One random right-hand side: α = 1 6802 iterations, 98.9 s (old mixed 6157, 89.6 s;
double 5576), α = 0.6 4091 (4012; double 3602). On this condition number CG in single
precision loses the conjugacy of its directions, and the rule on the residual of K†K lets an
error in the low modes through. QUDA does the same: CG with sloppy single 6360 iterations
against 5580 in double, its own check of the true residual |r|/|b| 1.8e-8 against 3.4e-9
(ours: |Dx−b|² 4e-16, |r|/|b| 2e-8). What to try (item 17).

What to try in CG: the norm of `mult_dot` in the last pass of the operator; x of CG updated in
the pass that updates p (one read of p fewer).

### 15. CPU: Neff's α in the sweeps (done 2026-10-20)

With α ≠ 1 the even-odd blocks of `Mdwf` were made of passes of their own: Q on the input of
every hop, the hop without C⁻¹ and the subtraction in its sweep (`hop_then_of_blocks`), and
C^(α)⁻¹ through a temporary field. CGNE (Normal_left) took 87 ms an iteration on 12⁴ against 40
without α. Now Q goes into the factor B of the hop's input (`Fifth::q`, `neff_q`), Q or Q⁻¹ on
its output and C⁻¹ into the same sweep, the Q⁻¹ and Q between the two hops of K cancel
(`inverse_hops_twice`, as on the device), and C⁻¹ takes Q⁻¹ in its pass: 42 ms an iteration.

Against Bridge++ (with_alpha 5540a26, core code: the QXS code has no α; -O3, OpenMP) on the
same CPU (i9-12900), 12⁴ setup S stout 0.1×3, m = 0.01, point source, 12 columns, 1e-18, even-odd
CG on the normal equation: the plaquette after smearing, the iterations of every column and the
PS correlator (to about 1e-11) are the same.

| α | Bridge++ (16 threads, its best) | RutticeQCD cpu (16 rayon threads) |
|---|---|---|
| 1.0 | 3974 iterations, 323.6 s, 81.4 ms | 3974, 158.9 s, 40.0 ms |
| 0.6 | 2597 iterations, 214.6 s, 82.6 ms | 2597, 109.9 s, 42.3 ms (227.0 s before) |

Bridge++ with 24 threads (with the E-cores) took 3253 ms an iteration: a barrier after every
four-dimensional operation of each slice. The `simd` backend is 10% slower than `cpu` here
(both built without `-C target-cpu`).

The lexical operator (`lexical.rs`) and `mult_site` (C^(α), used by `Schur`) still apply Q in a
pass of its own.

### 16. Smaller ones

- `DeviceBasis::pv_eo_coarse_vectors` computes s and t̂ again that `pv_eo` has just made; they
  could be restricted there.
- (Done 2026-10-20.) A step of `ChebyshevFilter` was the operator and five passes over the
  vectors (scale, axpy, scale, axpy, norm: 11 reads and writes of a field); it is now the
  operator and `Vector::scale_axpy2_norm_sqr` (4), on the CPU and the device. Below the one
  second the setup is reported in on 12⁴ and 16⁴; the Ritz values the same to the digits
  printed. The smoother (`Chebyshev`) has the same kind of passes (8 a step besides the
  operator) and is in every cycle: not done.
- `RUTTICE_CHECKSUM`: the checksums are taken inside the timed setup, by downloading the whole
  buffer and an FNV-1a over its bytes (about 0.9 GB/s); about 56 GB on 24³×48. Timings of runs
  with it are not those of the setup.

### 17. Where to look next (collected 2026-10-20)

From the A30 runs (24³×48 setup S, m = 0.001, 12 columns of a point source) and the comparison
with QUDA develop e75ee88 on the same links. The final benchmarks are for the A100 of the
collaborators; these say where the time goes.

End to end, the 12 columns: multigrid 584 s (setup 311, solves 251, the rest 22), mixed CGNE
α 0.6 804 s, double CGNE 1333 s. QUDA, one random right-hand side, α 1 (it has no α): CG double
5580 iterations 133.5 s (23.9 ms an iteration; ours 30.7), sloppy single 6360 76.6 s (12.0
ms; ours 14.5), sloppy half 11074 96.6 s (8.7 ms).

Where the multigrid's time goes (nsys, one solve at m = 0.001, 85fea29). The solve, 18.4 s
busy of 18.7: the coarse operator `coarse_pass_kernel` 30% (370 launches a GCR iteration,
558 µs each), restriction and prolongation 26% (`fold_restrict<float,half>` only 2 launches an
iteration but 32.5 ms each, `fold_prolong` 15 ms, `level_restrict` 11%), the fine hops 25%
(`hop_gemm` float 22%, double 3%), BLAS 12%. The setup, 314 s: `fold_restrict` about 82 s
(3800 launches of 21.7 ms; 1280 in each bootstrap round), `hop_split` 29 s, `coarse_pass`
28 s, the unfused `hop_kernel<float, many>` 15.8 s of the filter's 46 s (34%),
`fold_orthonormalise` about 12 s; about 19 s of host work before the first kernel and about
40 s of the GPU idle between the phases.

**Priorities (2026-10-20): the multigrid first.** CGNE gets the fix of the double hop kernel
(one block an SM, item b) and is then left as it is.

The multigrid, in this order:
1. *Restriction and prolongation* (`fold_restrict`, `fold_prolong`): 82 s of the setup and 14%
   of the solve. A launch takes 21–32 ms where reading P once in half precision should take a
   few: why is in the ncu asked of the A30. Then the bootstrap's 1280 restrictions a round one
   vector at a time (item 11): several vectors for one read of P.
2. *The coarse operator* (`coarse_pass_kernel`): 30% of the solve, 370 launches a GCR
   iteration (about 107 coarse applications), 558 µs each; how far from the bandwidth, and
   whether fewer, larger launches (several vectors) are possible.
3. *The setup's host time and idle GPU*: 19 s before the first kernel and about 40 s between
   the phases (the nsys trace says which calls).
4. *The filter's unfused hops* (done 2026-10-20): `hop_kernel<float, many>` (34% of the
   filter) was the last hop of `hop_then` without C⁻¹, which now goes through `hop_gemm` with
   the unit matrix (bit for bit the same; the 16⁴ filter on the RTX 3080 7 → 5 s).
5. *Fewer instructions in the hop kernel* (item 14): 735 warp-instructions a 5D site and CG
   iteration against QUDA's 343 (ncu, A30); `hop_gemm` 366–401 million a launch, DRAM 39–46%.
   Every one of the Ls threads of a site rebuilds the third row of the same links and repeats
   the neighbours' index arithmetic: links of three rows (the bytes are no longer the limit),
   or links and indices once a site. It is 22% of the multigrid's solve, and its filter and
   bootstrap. Then the chunked order of the blocks again (DRAM read −48% on the A30, no time
   gained while the kernel issues).
6. *Half precision* where the result is approximate anyway: the smoother and the coarse
   correction (QUDA's half is 27% faster an iteration than its single).
7. *`level_restrict`* (11% of the solve).
8. *Launch parameters by measurement* (QUDA autotunes every kernel): nvcc gave the same kernel
   128 or 168 registers by architecture and by small changes.
9. *Multi-GPU*, later: the CPU side has the communicator and the halo exchange; the device
   kernels need an interior and a boundary part and their own streams (QUDA's pattern),
   splitting in t first.

CGNE:
- (b) *The double hop kernel* (done 2026-10-20, not yet timed on the A30):
  `hop_gemm_kernel<double>` used 255 registers; at Ls 8 its blocks are 128 threads (16 sites:
  32 would not fit the shared memory), two an SM by the registers, 8 warps; a bound of 128
  registers spilled 4 KB. Now the legs one at a time in double and `__launch_bounds__(192, 2)`:
  168 registers, no spill, so three blocks of 128 an SM, 12 warps. The last change to CGNE.
  ncu of a double CG iteration on the A30 at 85fea29 (24³×48): 40.1 ms, the four hops 30.7 ms
  (77%; 9.5, 7.8, 5.7, 7.7 ms) at 12% warps active with the FP64 pipe 39–52% busy, the
  fifth-dimension product on the DMMA tensor cores (also inside the hops), the BLAS at 86–89% of
  the DRAM bandwidth; 6.7 kB of DRAM traffic and 757 warp-instructions a 5D site.
  At 7b2c1fc (A30, one run each against 85fea29): the four hops 30.7 → 25.1 ms (−18%; warps
  active 12.4 → 18.5%, FP64 pipe 39–52 → 49–60%, the same instructions), a double CG
  iteration 31.0 → 28.0 ms (−9.7%). The multigrid, one column (24³×48, m 0.001): setup 315 →
  299 s (filter 64 → 59 s, bootstrap rounds 90/99 → 84/94 s), solve 18.6 → 18.1 s, the
  program 338.6 → 322.3 s (−4.8%).
- Not pursued: the mixed CG at this condition number (iterations 13–22% above double,
  |Dx−b|² 1e-16; stop on |ρ − K x|, or the direction in double), the `B Q` pass with α (1.11 ms
  an iteration), the BLAS of CG (4 kernels, caxpy and caypx each reading and writing 0.8 GB).

More directions, decided 2026-10-20:
- *Several right-hand sides a solve*: a YAML/TOML parameter, never fixed in the code (a fixed
  number may not fit the memory of a larger lattice); it is one already
  (`Source.number_of_rhs_per_solve`). On the A30 (24³×48 MG, 85fea29) 2 does not fit: the
  setup peaks at 16.4 GB, a solve of one right-hand side at 21.8 GB of the 24, and the first
  solve of two dies with `cudaErrorMemoryAllocation` (a panic: "the CUDA runtime returned
  error 2"). To measure on an A100 (40/80 GB). The message should say what ran out and that
  fewer right-hand sides a solve would fit.
- *The Chebyshev smoother* (done 2026-10-20): `out += d`, `r −= A d` and `d = c d + a r` one
  pass over the vectors (`Vector::chebyshev_step`), bit for bit the same.
- *Stout smearing on the device* (done 2026-10-20): `load_links_on_device` for the
  `cuda_bridge` backend (it was on the device only for HMC); agrees with the CPU to rounding,
  not bit for bit. APE stays on the CPU. On the A30 (24³×48) it saves nothing: the CPU smeared
  in 1.9 s in a plain run; the 11.7 s idle GPU that was the reason was the nsys run's own
  start-up. Kept for larger volumes and more steps.
- *Bootstrap and filter parameters* (rounds, steps, the interval): tune on the A30.
- *`hop_split`* (29 s of the setup): look at it with the hop kernel.
- *`blocks_abs_inverse_gamma5_kernel`*: one launch a setup (Jacobi with many block
  synchronisations); low priority.
- Not pursued: the setup reused across masses (not how the propagators are used), CUDA graphs
  (not now).

What QUDA has for domain-wall fermions (its source, develop e75ee88): CG with reliable updates in
mixed precision, MSPCG as PCG with an MR preconditioner in half precision and additive Schwarz
(no domain splitting on one GPU), the MADWF accelerator (reduced Ls 4 with a trained transfer),
fused 4d dslash with M5⁻¹ and M5 on tensor cores, EOFA, eigCG; no multigrid for domain-wall
fermions in develop, but an unmerged branch `feature/dwf-pv-operator` (E. Weinberg, 2024-10 to
2026-01: a two-level `dwf_4d / PV† dwf_4d` multigrid, the line of Brower, Clark, Howarth and
Weinberg). Grid (Boyle) has HDCG in production.

### 18. Closing this phase (decided 2026-10-20)

After the Grid comparison and the multigrid items of 17 (at least `fold_restrict`), on the
A30 at one commit, without `RUTTICE_CHECKSUM`, the numbers of the MG note that are timings
run again and put in `docs/mg_background/results.tex`; then this phase ends.

- Timings (change with the code): the mass scan (Table `tab:massscan`, Fig. `fig:massscan`,
  `data/massscan_24.tsv`; seven masses, setups S and B; about 1.5 h MG and 40 min CGNE on
  the A30, plus the mixed CGNE of the figure, now by reliable updates); the point-source
  propagators (Table `tab:pointsource`, MG against mixed CGNE, 12 columns); the costs of the
  parameter changes on 12⁴ (Table `tab:costs`, at 8ce4c14 now). The commit in each caption.
- Iterations and spectra (the field of values, Arnoldi, pseudospectra, GCR(k), the residual
  histories, Hermiticity, MINRES, sensitivity) do not depend on the speed of the code: kept,
  unless the iterations of a rerun differ.
- New: the comparison with Grid's HDCG (and QUDA's CG) on the same links, each code with its
  best method (our Neff α included: it is a right preconditioner, the physics is the same),
  as a subsection of the results. On the A30 (2026-10-21, Grid develop 3232b43,
  `examples/Example_hdcg_multigrid.cc` with Boyle's parameters, Ls 8, m 0.001, block
  {4,4,6,4}): **Grid's HDCG does not fit in 24 GB**. NBASIS 24 and 16 (production: 62) run out
  of memory building the 81-point coarse operator; NBASIS 8 spends 877 s making the subspace
  (Chebyshev order 500) and then dies in the same build. Ruttice's multigrid does the whole
  setup and solve in 19.9 GB. Grid's plain CG on the same links: plaquette equal to all digits,
  double 6102 iterations at 38.0 ms (ours 30.7, QUDA 23.9). The comparison of the multigrids
  waits for an A100 80 GB. Grid's PVdagM multigrid (Boyle, Lattice 2026) is in develop
  (`Grid/algorithms/multigrid/PVdagM*`); not run, same memory constraints.
- The figures `fig_massscan.pdf` and `fig_history.pdf` are regenerated then; until now they
  are kept out of the commits.
- With the A30 numbers (decided 2026-10-21): the results section in the order of the
  abstract, the analysis of why it converges first (field of values, spectrum,
  pseudospectra, truncated GCR), then the comparison with CGNE (mass scan, point source).
  **Done 2026-10-21 at 6d5bf18**: mass scan (MG 13–28 / 7–26 iterations, at m 0.001 7.5× (S)
  and 9.0× (B) faster than double CGNE, 4.5× and 4.8× than mixed; setup 200–237 s, paid back
  after 2.4/2.1 rhs against double, 4.4/4.3 against mixed), point source (wall 396/409 s
  against 810/918 s at m 0.001; 305/331 against 264/324 at m 0.01), the abstract; the
  implementation and the reproduction in Appendix B, the precision row in Table setups. The
  regenerated `fig_massscan.pdf` committed with the OK (5dfa821). Fewer test vectors on
  24³×48 (S, m 0.001, point source, A30): 16 stagnate (910–940 iterations or the 2000 cap),
  24 need 162 for column 0; 32 stay (29–32). In the sensitivity text.
- Readability, with the same pass (from a review of the PDF, checked against the text; done
  2026-10-21: Fig. 4, Table 2, "Why both forms", the roadmap of Sec. XI.C, and the figures
  placed `[tbp]` — Fig. 3 had drifted to page 26):
  (a) a figure of one factorized coarse correction, Eq. (5.2) read from the right: the even
  sites, the whole lattice and the coarse lattice as three lanes, the first coarse solve
  returning t̂, whose even part a is the right-hand side of the second (Proposition 1); drawn
  from the equation, in TikZ. (b) The abstract's long sentence split, and a table in Sec. I or
  V.A: outer system, operator coarsened, coarse stencil (HDCG: the squared Schur complement,
  81 points; HDCR: nearest neighbor on the whole lattice; D_PV†D: two hops, 1 + 8 + 24 = 33
  points for blocks of at least 2, to be derived in the text; ours: D and D_PV apart, 9 points
  each), with device memory once the A100 comparison is there. (c) The end of Sec. IV: why
  both forms — Ã_R gives the exact factorized correction on the lifted basis, only Ã_L keeps
  Neff's α (Q cancels in D Q (C Q)⁻¹). (d) A roadmap at the start of Sec. XI.C: the coarse
  correction amplifies by 10³–10⁴ and the field of values leaves the right half-plane, yet the
  eigenvalues stay at Re λ ≥ 0.78 and the pseudospectra bound full GMRES, and truncated GCR(2)
  converges because the coarse solves stop at 10⁻²; subsection titles that say so.

**The aim and the end of the optimization.** The measure is the whole two-point function on
the A30: 24³×48, m = 0.001, point source, 12 columns, setup included; 584 s at 85fea29 (setup
311, solves 251). The aim is about 400 s, from the bandwidth floors of the ncu profiles
(estimates, each to be measured):

| | now | near the bandwidth (estimate) | saved |
|---|---|---|---|
| `fold_restrict` (setup 3800 launches, solve 2 an iteration) | 32.5 ms, DRAM 11% | about 5 ms (done: P read coalesced through shared memory; RTX 3080 16⁴ 4.64 → 1.31 ms, DRAM 31% with 256 blocks for 68 SMs; A30 to measure) | setup −65 s, solves −17 s |
| `coarse_pass` (30% of a solve, 28 s of the setup) | DRAM 22–37% | DRAM 60–70% (done: fields in shared memory, 16-byte matrix loads; RTX 3080 16⁴ 82.5 → 62.6 µs, DRAM 62 → 82%; A30 to measure) | solves −38 s, setup −14 s |
| setup host time and idle GPU | 19 + 40 s under nsys; the 19 s before the first kernel is nsys's own (plain run: smearing done at 1.9 s) | to be measured in a plain run | unknown |
| `level_restrict` (11% of a solve) | 47 single-block launches an iteration, 1.2 ms each (8.4%) | a warp an output (done; RTX 3080: 187 → 11 µs, 16⁴ setup 32 → 25 s) | A30, 6269f81: solves 251.2 → 215.5 s with 7b2c1fc's hop, setup unchanged |

`level_restrict` on the A30 at 6269f81 (ncu): the last level 1.2 ms → 133 µs (16 blocks, still
latency-bound, but under 1% of a solve now), the grid-41 level 392 → 140 µs at 67% of the DRAM
bandwidth: done.

Measured on the A30 (12 columns, one run): 85fea29 584 s; 6269f81 535.9 s (setup 298 s,
solves 215.5 s, the rest 22 s; 362 GCR iterations as before, PS within 7.3e-9); f63b8ad
(`fold_restrict` coalesced, `coarse_pass` fields in shared memory) **460.9 s** (setup 244 s:
bootstrap rounds 83 → 67 and 94 → 75 s; solves 191.6 s; the rest 25 s; 365 GCR iterations;
worst |Dx−b|² 2e-18 in one column, the others ≤ 1e-18; PS within 1.85e-8 of 6269f81); 68b76e7
(|H|⁻¹ by rounds, `hop_split` two blocks an SM, `coarse_pass` without hops apart) **433.8 s**
(filter 61 s, bootstrap rounds 59 and 68 s, setup 225 s; solves 186.5 s; the rest 22 s; 364
GCR iterations; one column at 2e-18; PS within 1.17e-8 of f63b8ad). 34 s from the aim.
6ccca76 (`coarse_pass` with hops ≤ 85 registers — 62 with CUDA 13, no spill —, `fold_restrict`
by `cp.async`, |H|⁻¹ with V in shared memory) **404.6 s** (filter 60 s, bootstrap rounds 56 and
63 s, levels 6 s, setup 212 s; solves 171.1 s; the rest 21 s; every column's iterations,
coarse applications and residual as at 68b76e7, the PS bit for bit the same). 4.6 s from the
aim. Its ncu: `coarse_pass` with hops 810/884–916 → 540/590–603 µs (DRAM 68–70%, warps 47%:
done), without 140/242 → 129/223 µs; `fold_restrict` −15% everywhere (long scoreboard 2.6 →
0.7, DRAM 34%; a 96-byte stack frame, 0.5 GB of local loads a solve launch, mostly L1 hits);
|H|⁻¹ 1.2 → 0.37 s a launch.
6d5bf18 (no run-time indexing of the fold kernels' geometry: `fold_restrict` without its stack
frame; the prolongation by pairs) **401.1 s** (filter 58 s, setup 207 s, solves 170.1 s, the
rest 24 s against 21 before; iterations, residuals and PS bit for bit those of 6ccca76). From
85fea29: −31%, setup 311 → 207 s, solves 251 → 170 s. Its ncu: `fold_restrict` without local
memory (0.26–0.5 GB of loads before), 3% faster on the A30, now held up by wait and short
scoreboard at 35% of the DRAM bandwidth; `fold_prolong` 17.4 ms a solve launch (19.9 at
85fea29), still long scoreboard 14. **The kernel work of this phase ends here (2026-10-21,
the aim taken as met).** Left for later: `fold_restrict` and `fold_prolong` at 27–35% of the
bandwidth, the filter (58 s, its cost the number of operator applications).
The ncu of 68b76e7 (A30, ncu times): `hop_split` 13.5 → 7.9–8.6 ms (DRAM 39–41%, warps 24%,
long scoreboard); |H|⁻¹ 4.8 → 1.2 s a launch, now latency-bound (2 blocks of 64 an SM, 6% warps,
waiting on V in global memory: V in shared memory too fits the A30's 164 kB, not the RTX
3080's); `coarse_pass` without hops 219/397 → 140/242 µs, with hops unchanged at 810–916 µs:
128 registers there with CUDA 13 (96 with 12.5), warps 24% — `__launch_bounds__(256, 3)` gives
62 registers without spills.
| `fold_prolong` | DRAM 31% | | −6 s |
| `blocks_abs_inverse_gamma5` | 3 launches a setup, 3.8 s each, 1.15 TB of DRAM traffic | H in shared memory, rounds of disjoint pairs (done; RTX 3080 0.70 → 0.10 s; A30 to measure) | −11 s |
| `hop_split<float>` (2 × 1024 launches, 9.4 ms) | 168 registers, one block an SM | two blocks an SM, 128 registers (done; RTX 3080 −20%, DRAM 88%; A30 to measure) | about −8 s |
| `fold_orthonormalise` | 42 blocks on 56 SMs, 12% warps active | more blocks | a few s |

`fold_restrict_tiled_kernel` on the A30 (f63b8ad): 11.6–12.2 ms in a solve (was 32.5), 5.8 ms in
the bootstrap (was 21.7), DRAM 29–34%, warps active 30%: shared memory (28 kB a block) allows 5
blocks an SM, and it waits on its loads (long scoreboard, wait). Tried on the RTX 3080 and not
kept: four loads of a thread issued before their stores (72 registers instead of 56, 10–15%
slower); P read straight into registers, no tiles (31% warps active, but the half-used sectors
are fetched again from DRAM: 307 → 510 MB a launch, the heavy shape 5.0 → 5.4 ms). Left:
copies to shared memory that do not go through registers (`cp.async`, sm_80), double-buffered.
`coarse_pass_kernel` on the A30 (f63b8ad): the passes with the hops 19–21% faster (DRAM 37 →
46%), those without 18% slower (128 registers for the unrolled links, warps 35 → 24%); now a
kernel of its own without the hops (RTX 3080: 70 → 44 registers, 19 → 16 µs).

The ncu of the setup (A30, 85fea29): the filter's `hop_kernel<float, many>` (728 launches, 15.8 s)
was the Chebyshev step's last hop, which 7b2c1fc sends through `hop_gemm`. `fold_restrict` in the
bootstrap has the shape of the solve's (DRAM 11–12%, L1 hits 92%, long scoreboard and LSU
throttle): a thread for each (coarse site, component) reads its part of P one half at a time,
neighbouring threads 192 B apart, so nothing is coalesced.

Rules:
1. A kernel is done at 60–70% of the DRAM bandwidth, or within 1.5 times its floor. Beyond
   that only fewer bytes help (half precision, several right-hand sides): the next phase.
2. A change that gains less than 5% end to end on the A30 is reverted.
3. The phase ends when no part is above 10% of the total and far from its floor.
4. Against Grid's HDCG: if we are clearly faster (1.5 times or more), finish the table and
   stop. If not, find where: more iterations or coarse applications is the algorithm (next
   phase, noted here), more time an application is this phase.
5. A30 budget: one ncu and one plain run a change; the final numbers of the note about 3 h.

Not in this phase: multi-GPU, autotuning, CUDA graphs, half precision, a QUDA backend,
cuda-rust, CGNE (the double `hop_gemm` was the last change). Several right-hand sides a solve
as a parameter (item 17) only if the A30 memory test allows it.

### 19. CPU multigrid: the next optimization phase (decided 2026-10-21)

After the dual cycle ("Known, not understood") is settled: that work runs long CPU jobs (12⁴
runs of 87 min on about 8 cores), and CPU timings taken next to it are not to be trusted.
The device phase (items 9–18) is the model: measure, change one thing, keep it only if it
gains, results the same to rounding.

**The measure.** 12⁴, setup S (stout 0.1×3, M5 1.0), m 0.001 and 0.01, a point source of 12
columns, setup included, i9-12900 with 16 rayon threads on the P-cores (Bridge++ with 24 threads,
the E-cores included, took 40 times longer an iteration: check our scaling at 16 and 24 too).
`examples/mdwf_pv_eo_mg.yaml` as the run. The reference is our CPU even-odd CGNE on the same
propagator (m 0.01: 158.9 s, α 1; 109.9 s, α 0.6; item 15) — Bridge++ has no domain-wall
multigrid. **Aim:** the multigrid with its setup faster than that CGNE at m 0.01, and clearly
faster at m 0.001. First step: a `perf` profile of setup and solve, as the ncu profiles were
for the device, to put the items below in order.

**Directions,** roughly in the order expected to pay:
1. **Build flags.** `-C target-cpu=native` (AVX2 and FMA on the i9-12900; Alder Lake has no
   AVX-512) for the `cpu` and the `simd` backend; `simd` was 10% slower than `cpu` without it
   (item 15). A proposed session, "CPU SIMD backend with target-cpu=native vs Bridge++", has
   the brief. Decide whether release builds of `ruttice` should default to it (portability of
   the binary).
2. **The coarse operator build** (item 7): most of the CPU setup. Left after the partial fix:
   the one application of D (15%), Q and the fifth dimension of w (21%), the hops (11%), the
   restrictions (10%); the hop sorting is a serial loop over the fine sites.
3. **Fused vector passes.** The CPU `Field` takes the default `Vector::chebyshev_step` (axpy,
   axpy, scale, axpy: four passes over the vectors), where the device does one (2026-10-20);
   the smoother is in every cycle. Same for any other default `Vector` method that `Field`
   does not override: list them, fuse those in the cycle and the setup. Memory-bound on the
   CPU as on the GPU, so the gain is the passes saved.
4. **Single precision above the coarse lattice** (item 8): the fine fields of `PvEoCycle`, the
   filter and the bootstrap in f32 doubles the SIMD width and halves the traffic; on the device
   it kept the outer iterations and the correlators. Making `mdwf_even_odd.rs` generic over the
   fine storage also brings the CPU and the device onto one cycle (`TwoGrid`).
5. **Restriction and prolongation** (`basis.rs`, `restrict_site`, `prolong_site`): the device
   gained most from reading P in the order it is stored (`fold_restrict`, 32.5 → 12 ms).
   On the CPU check that a thread walks P of one aggregate contiguously and that all vectors
   of a site are done in one pass; P in f32 (not half: item 8's note).
6. **The coarse operator apply** (`coarse.rs`): on the device, reading a coarse site's and
   its neighbours' fields once and the matrices in wide words gave 25–35%. On the CPU: the
   nine 2n_v × 2n_v blocks as a small complex GEMV per neighbour, vectorized (or a small GEMM
   when several right-hand sides are solved together), the neighbour fields kept in cache by
   the site order.
7. **The levels below and the small coarse solves.** On the device the last level was a single
   block until each output got a warp; on the CPU the analogue is rayon overhead on tiny
   levels: below a size threshold, serial. The dense inverse of the last level and the |H|⁻¹
   of the blocks (setup, once) only if the profile shows them.

**Rules** (as item 18): a change that gains less than 5% end to end is reverted; iterations
and the PS correlator as before, to rounding; the commit and the numbers in this item.

### 20. Device: the HMC multigrid profiled (2026-10-23)

One short trajectory on 24⁴ (stout, m 0.001, 20 test vectors, levels below, A30), nsys and ncu
(`/home/chen/red_gpu/ruttice_a30_20261016/prof/` on as13). 242 s wall, 211 s of it in kernels
(87%): not launch latency, the kernels themselves. By GPU time: the fine hops 41%
(`hop_gemm<float>` alone 27%), the coarse operator 29% (`coarse_pass<float,1>` 24%), P
(restrict 12%, prolong 8%, orthonormalise 5%) 24%.

| kernel | DRAM of peak | occupancy | what limits it |
|---|---|---|---|
| `hop_gemm<float>` | 28–44% | 25% (128 registers) | the FMA pipe (70%), issue; 5900 instructions a thread |
| `coarse_pass<float,1>` | 55–62% | 47–50% | near enough to bandwidth |
| `fold_prolong` | 27% | 49% | the access pattern, 10–20 ms a call |
| `fold_restrict_tiled` | 35% | 34% | the same |
| `coarse_pass` of the levels below | 13–25% | 18% | 81 blocks: too little to fill the device; small in total |

**`hop_gemm<float>`: tried and reverted** (`examples/hop_gemm_bench`, 24⁴ Ls 8, D 1.53 ms, D†
1.15 ms; the gap is c Y applied at the neighbours, three spinors read for each):
- three blocks an SM (`__launch_bounds__(256, 3)`, 80 registers): 3.13 ms, a variant spills;
- (b − c Y) with one sum per projector and multiply-adds instead of the projected spinors:
  1.60 ms, double 2.94 ms (the compiler had folded the multiplications already);
- the neighbours' Ls spinors through the shared memory, each read once (16 barriers): 1.98 ms,
  bitwise the same results; the triple reads hit L1 (68%), and the barriers stop two blocks
  from hiding the loads.
What is left for it is a redesign (more sites a thread for independent loads, or the loads of
the next leg in flight across the barrier), not a local change.

**P (2026-10-23, `examples/fold_bench`, 24⁴ Ls 8, 20 test vectors):** prolong 15.9 → 7.3 ms
(`fold_prolong_tiled_kernel`: a block for 32 sites, a site's threads together reading P in
32-byte sectors, the field written whole through shared memory; bitwise the same; HMC on 24⁴
242 → 230 s, the same iterations and ΔH). A site to each lane was slower (21 ms: 32 places of P
7.7 kB apart). Restrict (12.8 ms) tried and reverted: six sites a round 11.5 ms (sums in another
order, 10%), double-buffered rounds 29.6 ms (52 kB of shared memory, three blocks an SM).

**The setup (2026-10-23, 24⁴ m 0.001, 20 test vectors, levels below):** the filter 34.5 s, the
bootstrap 101 s (two rounds: making the correction 18 s each — P 3.9, P†DP 6.5, P†D_PV P 6.0,
|H|⁻¹ 0.8, the top 1.3 —, and the vectors 20 s and 45 s). Done: the levels below for the
bootstrap from its second round (45 → 26 s; in the first they gained nothing for 7 s of making),
the probing of the coarse operators four columns at a time (6.5 → 4.4 s each). Batching the
bootstrap's vectors (four at a time through FGMRES) gained nothing (136 → 138 s, +3.6 GB):
`coarse_pass` reads the matrices again for every vector. Setup 141 → 117 s. Left: making the
levels (7 s a round), P (4 s), the filter (35 s), and a `coarse_pass` that applies its matrices
to several vectors in one pass (then batching pays, and the tensor cores fit). The filter cannot
lose its second round: with one, HMC on 24⁴ at m 0.001 did not finish in 1500 s (the vectors too
poor). P by CholeskyQR2 instead of Gram-Schmidt (800 sweeps over W for 40 columns): 4.0 → 1.8 s,
P bitwise the same; HMC 210 → 203 s.

**Next, by the profile:** P (`fold_prolong`/`fold_restrict` at 27–35% of the bandwidth: read
it in the order it is stored, several columns a block; 2–2.5× on 20% of the time); the coarse
matrices in half precision for `coarse_pass` (bytes halved, about 1.6×); the bootstrap in
batches of 4–5 vectors, which makes `coarse_pass` a block product (the tensor cores) and saves
about a third of the setup (item 11).

### 21. CPU domain-wall: the factor b − c Y once, not on eight legs (found 2026-10-25)

The even-odd hops of the CPU apply B = (4 − M5)(b − c Y) to the input as the sweep reads it
(`Pre::Fifth`): every leg reads a neighbour and makes its B again, 8 × 312 Flop a spinor beside
the hops' 1320, more than the hops themselves (counted since 607d923; the device does one dense
Ls × Ls product, 768 at Ls 8). B made once on the input field, then plain hops, does an eighth of
that work, for one more field of memory and one more pass over it. Whether it is faster depends
on whether the sweep is bound by memory or by arithmetic there: measure on 12⁴ and 16⁴ first
(rules of item 19).

### 22. The work report: complete (2026-10-25)

Every kernel and CPU operation counts itself (DESIGN §7.11); where the backends differ by design
is listed in USAGE (domain-wall fifth dimension, the CPU's ln det of even-odd clover, the setup
of the Wilson and clover multigrid, bytes of links and P). `tests/cuda_bridge.rs` compares every
kind of a clover RHMC trajectory and of a clover multigrid solve on the two backends. Open: the
Jacobi sweeps of |H|⁻¹ are an estimate (six); nothing compares the domain-wall counts or the
setups across the backends; the halo exchanges are not counted (communication).

Planned (agreed 2026-10-25, after the work on the structure and the bugs): the report against
Nsight Compute on 16⁴ or 24⁴, clover and domain-wall solves: the bandwidth the report implies
(GFlop/s over its Flop/byte) against the DRAM GB/s ncu measures, kernel by kernel, to see how
far the model (no caches) is from the traffic.

## Ideas from the literature (to evaluate later)

- **Test vectors from a renormalization-group blocked lattice** (Eick and Mawhinney,
  PoS LATTICE2025 048): their coarse low modes, interpolated, as the starting vectors of the
  filter and bootstrap, to cut the setup (207 s on 24³×48 at 6d5bf18), above all for HMC.
- **Chiral SVD truncation of a larger set of test vectors** (Whyte, Stathopoulos, Romero,
  arXiv:2505.14399): more vectors truncated per aggregate and Γ5 chirality to the present
  coarse size, streamed to save device memory. It breaks the exactness of the correction on
  {s, t̂}; measure whether the gain outweighs it.
- **The fifth-dimensional structure of the low modes** (from Eick and Mawhinney: their coarse
  low modes fall off more slowly in s and are more jagged than the fine ones). Two steps:
  (a) look first — the s-profile of our filtered low modes per aggregate, and how many
  different profiles a block sees (the singular values of a test vector's restriction to one
  aggregate, arranged as s × (sites, spin, colour)); (b) if there are several, split each
  test vector in s as it is split in chirality now (the walls apart from the bulk), which
  enlarges the coarse space without more setup vectors and keeps s and t̂ in the span
  (Proposition 1 still exact); the coarse operator grows as the square of its dimension.
  Motivation: on 24³×48 (S, m 0.001) 16 test vectors stagnate and 24 need 162 iterations
  against 29–32 with 32 (A30, 6d5bf18), while 16 suffice on 12⁴.

## Continuous integration

There is none (`.github/` does not exist; note that `.gitignore` ignores every top-level entry
not listed, so `!/.github/` must be added). Evaluated, not set up:

- **Every push:** `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test --release` on the CPU (about 10–15 minutes, mostly `tests/hmc.rs`).
- **Separate job:** `cargo clippy --all-targets --features cuda-bridge` with the CUDA toolkit and
  cmake installed (`RUTTICE_CUDA_ARCH=80` to shorten the nvcc build): compile check only.
- **Not on hosted runners:** the GPU tests (`tests/cuda_bridge.rs`) need a self-hosted runner with
  an NVIDIA GPU. The `mpi` feature needs OpenMPI and libclang (`tests/mpirun.rs` skips without
  `mpirun`); `hdf5` 0.8 has known problems with HDF5 1.14.
- A green CI would cover the CPU paths only; every number of the multigrid paper is from the
  device.

## Multigrid paper (`docs/mg_background/`)

- Not measured, listed in the note: more configurations (the main request to whoever has GPUs:
  see `reproduce.tex`), Lüscher's coherence test for the
  mass-scan basis, a rigorous lower bound on `sigma_min` (Kato–Temple), the spectrum on
  `24^3x48` (does not fit in 24 GB).

## Device robustness

- **The 24³×48 setup on the A30 was about 30% slower than in the mass scan** (505–511 s at
  cbe655b against 389–393 s at 6c7ff42, setup S and B; the solve is unchanged within an
  iteration). Not the code: the runs at cbe655b were those with `RUTTICE_CHECKSUM=1`
  (29 checksums), and the checksums are taken inside the timed setup (item 16). Measured on the
  A30 on 2026-10-20: cbe655b without the variable 391 s, a fresh build of 6c7ff42 395 s; with it
  7376a0f, 0c13b18, 192657f, 1371927 and cbe655b all 509–512 s (+118 s, all in the two
  bootstrap rounds). The barrier of 0c13b18 costs nothing measurable (2592 blocks of dim 64:
  12 s either way on the RTX 3080). The run-to-run differences are gone: two runs at cbe655b
  gave the same checksums (`cuda_bridge/README.md`).
- **Running out of device memory panics.** `Vector` has no `Result`, so `zeros_like`,
  `zeros_with` and `select` of the device fields panic (`must`) when the device refuses; the
  message is the CUDA error. A solver that narrows its vectors (`Narrowed::narrow`) holds the
  old set and at most three quarters of it at once. Making `Vector` fallible would touch every
  solver; not planned. (`Subspace::rotate` of more than 32 vectors used to make n full copies
  one after the other; since 2026-10-19 it needs one more set only.)

## Known, not understood

- **The filter is weak** (2026-10-23, 24⁴ m 0.001, 20 test vectors). Its degree is
  1.5 √(hi/lo) between 40 and 1000, lo = hi/1000 in the first round and the largest Ritz value
  of the block after it. After the first round (degree 48) that Ritz value is still 0.34 hi, so
  the second round has degree 40 and damps only the top two thirds of the spectrum: the filter
  hardly separates the low modes, and the bootstrap does the work. To try: keep lo at hi/1000 (or
  at the m-th Ritz value of a larger block) in the later rounds, more rounds; then whether 16
  test vectors converge on 24⁴ (P 25% smaller) and the bootstrap can be shorter. With one round
  instead of two, HMC there did not finish (1500 s).
- **Where α acts on Ã_L^(α)** (2026-10-23). In the mass scan on 24³×48 α 0.6 saves the multigrid
  2–3 iterations at every mass in setup B, but in S only at heavy masses (none from m 0.01 down;
  HMC on 24⁴ at m 0.001: 362 against 356), while CG gains about 20%. The note reads it as α
  improving only what the smoother sees (the coarse grid takes the low modes); not checked. To
  check: the extreme eigenvalues (Arnoldi) and the field of values of Ã_L with and without α, and
  of Ã_L M, in S and B.

- **The dual cycle on 12⁴** (superseded 2026-10-22 by Γ5 M Γ5 below; the dual cycle is gone
  from HMC) (`examples/correction_exactness`, CPU, 2026-10-21): setup S,
  m 0.001, GCR(2) on Ã_R with M converges in 23 iterations, on Ã_R† with the dual cycle not
  in 1000. The HMC multigrid (`src/hmc/dwf_multigrid.rs`) used this dual cycle for the
  normal equations; its comment reports 17 (S) and 57 (B) iterations on 16⁴. To check: the
  dual on 16⁴ with the example, and a basis for D† (low modes of Ã_R Ã_R†, or closed under Γ5).
  Later (2026-10-22, a lost uncommitted fix of the example: the bootstrap with the run's
  smoother, MINRES on the Pauli-Villars factor): primal 12 (S) and 15 (B), dual on the same P
  27 (B, GCR(2); 23 GCR(8)), S stalls near 1e-3 also with MINRES or double coarse solves;
  u_j = Ã y_j as the basis is worse.
- **Why Γ5 does not give D† for Möbius (found 2026-10-22).** D = D+ − D- Y with
  D+ = 1 + b W, D- = c W − 1 (Brower-Neff-Orginos) standing *left* of the fifth-dimension hop
  Y. With Γ5 = γ5 R5 (`mult_gamma5`): Γ5 D Γ5 = D+† − D-† Y†, but D† = D+† − Y† D-†; W has γ_μ, which do not
  commute with the P± in Y, so the two differ unless c = 0. The exact relation is
  **D- Γ5 D† = D Γ5 D-†**, i.e. D† = Γ5 D-⁻¹ D Γ5 D-†, and a near-null vector v of D gives
  the near-null vector (D-†)⁻¹ Γ5 v of D†. Checked on the 12⁴ v2 configuration, stout 0.1×3,
  M5 1.0 (`examples/mobius_gamma5_check`): |Γ5DΓ5 b − D†b|/|D†b| =
  0.147 (b 1.5, c 0.5), 0.22 (b 2, c 1), 0 (Shamir); the relation holds to 3e-16.
  **D- cannot be inverted in practice:** D- = c (W − 1/c) is c times the Wilson operator at mass
  −M5 − 1/c (−3 in S, −3.8 in B), deep among the doublers; its spectrum surrounds the origin.
  On the A30 (12⁴, `examples/dminus_cost`), D- and D-† by GCR(8) stall at
  |r|² 0.74 (S) and 0.96 (B) after 20000 iterations, BiCGStab diverges. So the exact map of
  the basis, or solving D† through D, costs more than the solve itself. On smooth modes D- is
  about −(1 + c M5) (a number), so Γ5 v is close to a dual near-null vector up to its high
  modes: a possible reason why the dual on P converges in B but slowly, and stalls with stout
  links (not checked). For the note.
- **Solved (2026-10-22): D† by Γ5 M Γ5.** D† Γ5 D⁻¹ Γ5 = Γ5 (1 − c [Y, W] D⁻¹) Γ5, so the
  cycle of D between two Γ5 preconditions D†; on the even sites C_e† Γ5 M Γ5 for K† (MG note,
  "Solving with D†"; `fermion::adjoint_by_gamma5`, `examples/dagger_precond`). 2–3 iterations
  more than the solve with D, in S and B, on random and HMC-like right-hand sides; HMC uses it
  with GCR(2). Open: the 12⁴ S configuration with Q = +1 at m 0.001 needs GCR(8) on the
  residual of a sweep (topological mode?); a bound on ‖N²‖. The CPU path of HMC with it has
  not been run (as13's CPU is shared). The items below record what was tried before.
- **A P for D† from the test vectors of D** (2026-10-22, A30, `examples/dual_basis`;
  m 0.001, 32 vectors, blocks 4⁴, GCR(8) to 1e-12). The Galerkin P†ÃP needs both sides: for Ã† the left near-null vectors are y (those of D), the right ones close
  to Γ5 y. span(Γ5 y) is almost orthogonal to span(y) (0.001 of its norm in it). Iterations
  (Ã with P(y): 7 S, 9 B on 12⁴; 7, 9 on 16⁴; 9, 13 on 24⁴):

  | P of Ã† | S 12⁴ | B 12⁴ | S 16⁴ | B 16⁴ | S 24⁴ | B 24⁴ |
  |---|---|---|---|---|---|---|
  | P(y), shared | stall (1e-6) | 14 | 10 | 12 | 21 | 20 |
  | P(Γ5 y) | 14 | 50 | 15 | 60 | | |
  | P(y and Γ5 y), 64 vectors | 10 | 9 | 8 | 9 | out of memory | out of memory |

  Half of y and half of Γ5 y (32) fails; a full setup of its own for Ã† stalls in S 12⁴ too
  (only the right side). The 64 vectors cost 2–4× a solve of the shared P (four times the
  coarse operators), so they were an option (`MultiGrid_HMC.dual_with_gamma5`, removed with the
  dual cycle), not the default. Not tried: Petrov-Galerkin with P† as restriction and Γ5 P as prolongation (no
  more memory than the shared P). The first lexical MDWF multigrid (6830b9c) had a Petrov-
  Galerkin restriction, but no coarse operator helped there (random test vectors), so it says
  nothing about this one.
- **Device memory of HMC with the multigrid** (2026-10-22, 24⁴, stout, m 0.01, 16 test vectors,
  `fine_precision : single`, GCR(8)). Measured with the allocation counter (`device_memory`):
  links, smeared levels, operators in f64 and f32, pseudofermion 3.5 GB; the test vectors
  (f32) 2.2 GB, on the host between the solves since 19b8af9; P (f16, folded on the fifth
  dimension) 4.1 GB; the coarse parts of Ã and Ã† 0.4 GB; the fields of the sweeps on D†D
  (`two_solves_half`, f64) 3.3 GB; GCR(8), 19 vectors on the even sites in f64, 4.8 GB; the
  work fields of one cycle about 1.5 GB (the cycles of Ã and Ã† made in turn since 19b8af9).
  Solves 19.5 GB, peak 21.3 GB (the filter of the setup 12.3 GB). Per volume twice a propagator
  run: two systems (D and D†), GCR(8) against GCR(2), the sweeps on D†D, the fields of HMC.
  **2026-10-23** (m 0.001, 20 test vectors, levels below, GCR(2), Γ5 M Γ5 for D†, after the
  savings of 2d07414): HMC 3.3 GB, the multigrid kept between the solves 6.6 GB (P 4.75 GB),
  the sweeps 2.6 GB, the solve with D† on top 6.4 GB (GCR 1.7, the two Schur factors' work,
  the preconditioner, the cycle); peak 19.1 GB live, 20.1 GB with the pool. Left to save:
  one set of work fields for K, X_PV and the K of the residual (about 1.5 GB), the whole-
  lattice fields of the sweeps (about 1 GB).
  Since then the solves with D† use Γ5 M Γ5 and GCR(2) (no cycle or coarse operator of Ã†):
  20.5 GB on 24⁴ (nvidia-smi). Left: the sweep fields in f64. 24³×48 needs about twice all of
  it: not on the A30.
- **The example's GCR(2) does not converge in setup B on 12⁴**, with either cycle, even with
  B's smoother (`--steps 12 --ratio 30`), where the solver of the note needs 15 (Ã_R, Table
  gcrk). Its differences from the solver: both coarse factors by BiCGStab (the note's B with
  BiCGStab on the Pauli-Villars factor: 19, on Ã_L), the cycle in double precision, no levels
  below, Ã_R, `PvEoParams::default()` otherwise.

- With the multigrid, BiCGStab on the exact adjoint for the Pauli-Villars coarse factor is far
  worse than MINRES on `H_c`, also for `c = 0` (373 against 14 outer iterations, stout 12⁴).
- The Bridge++ `alt_eo` reference for the domain-wall even-odd Pauli-Villars ratio differs from
  Ruttice by 1e-5; external checks agree with Ruttice.
