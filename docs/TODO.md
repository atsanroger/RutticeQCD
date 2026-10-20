# To do

Work that is known and not done, so that the next person (or session) can pick it up without
rediscovering it. Each item says why, where, how, how much, and how to check it. Nothing here
changes results; items marked *performance* change speed, the others only the structure.

Priorities: finishing the multigrid paper (`docs/mg_background/MDWF_MULTIGRID_NOTE.tex`) comes
first; the structural items after it, in the order given. Last reviewed 2026-10-20.

## Structure

### 1. One propagator driver for the CPU and the device (partly done 2026-10-19)

- **Done.** What follows the setup is written once: `solve_sources` (`src/app/mdwf.rs`) runs the
  random right-hand sides or the point-source propagator and the measurements; each backend
  gives the fields through `SourceFields` (`CpuFields`, `DeviceFields`) and its own `solve`
  closure. The solver dispatch is shared (item 2).
- **Left.** The setup (operators, test vectors, the multigrid cycle or Chebyshev) and the
  `solve` closures are still two: `src/app/mdwf.rs::run_in` and `src/app/mdwf_cuda.rs::run`.
  Their data differ (layouts and an f32 `Convert` on the CPU; device bases, levels and
  counters on the GPU), so a common trait there would carry most of both; do it only when a
  feature needs it on both sides. Mixed-precision CGNE is still on the device only.
- **Check.** `cargo test --release --features cuda-bridge --test run --test mdwf --test cuda_bridge`;
  the iteration counts of `examples/mdwf_pv_eo_mg.yaml` on the CPU and the device unchanged.

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

- **Why.** `PvEoCycle`, `pv_eo_low_modes`, `pv_eo_bootstrap`, `lifted_vectors` fix the fine fields
  to `f64`. On the device single precision kept the outer iterations and the correlators; on
  the CPU it would double the SIMD width and halve the memory traffic.
- **How.** Make `mdwf_even_odd.rs` generic over the fine storage (`low_modes`, `bootstrap_round`,
  `TwoGrid`, `coarse_solve` already are). While there: `PvEoCycle` has its own cycle and
  `coarse_correction`, the device uses the generic `TwoGrid` and `DeviceCorrection`; one
  implementation needs the coarse operator behind a trait (`schur`, `schur_source`,
  `schur_solution`, `even_block_inverse`, `apply`, `gamma5`, which `CoarseOp` and
  `DeviceCoarseOp` both have).

Not worth doing on the CPU: the prolongation in half precision; reading it once for several
vectors (the CPU already loops over the vectors inside a site).

### 9. Device: the coarse operator kernel (done 2026-10-20)

`coarse_pass_kernel` had a thread for each row of a site's matrices and read the rows of a warp
2 dim reals apart; it now has a warp for each row (`cuda_bridge/README.md`): 3.59 ms against
1.13 ms per application on 2592 sites of dim 64 in single precision on an RTX 3080. Setup of
16⁴ with Level2 + Level3 of 2: 45 s against 35 s; 12⁴: 18 s against 17 s, the solve 1.2 s
(15 iterations) against 0.9 s (14). The checksums differ from earlier runs from the first
bootstrap vector on (another order of the sums); runs agree with each other. On the A30
(24³×48 setup S, 14359e1 = items 9, 10 and the filter step of 15, without `RUTTICE_CHECKSUM`,
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

On the A30 (24³×48 setup S, m = 0.001, 1a58b1f, 2026-10-20; `ncu` locks the clocks and flushes
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

On the A30 at bc32f92 (24³×48 setup S, point source, 12 columns, 2026-10-20): CGNE in double at
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

Careful when timing it: with `precision : mixed`, `Solver.maximum_number_of_iteration` caps
each inner single-precision CG, and the defect correction repeats it up to
`maximum_number_of_restart.max(50)` times; 400 gave 20000 iterations, not converged.

What to try: the norm of `mult_dot` in the last pass of the operator; x of CG updated in the
pass that updates p (one read of p fewer); mixed precision by reliable updates instead of
restarts (each inner CG of the defect correction starts from zero; 355 against 349 iterations
in double on 16⁴, so little to gain there).

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

## Ideas from the literature (to evaluate later)

- **Test vectors from a renormalization-group blocked lattice** (Eick and Mawhinney,
  PoS LATTICE2025 048): their coarse low modes, interpolated, as the starting vectors of the
  filter and bootstrap, to cut the setup (about 390 s on 24³×48), above all for HMC.
- **Chiral SVD truncation of a larger set of test vectors** (Whyte, Stathopoulos, Romero,
  arXiv:2505.14399): more vectors truncated per aggregate and Γ5 chirality to the present
  coarse size, streamed to save device memory. It breaks the exactness of the correction on
  {s, t̂}; measure whether the gain outweighs it.

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
  462a423 against 389–393 s at 535c9ef, setup S and B; the solve is unchanged within an
  iteration). Not the code: the runs at 462a423 were those with `RUTTICE_CHECKSUM=1`
  (29 checksums), and the checksums are taken inside the timed setup (item 16). Measured on the
  A30 on 2026-10-20: 462a423 without the variable 391 s, a fresh build of 535c9ef 395 s; with it
  d665b50, b72bb75, dd484b3, a5dc425 and 462a423 all 509–512 s (+118 s, all in the two
  bootstrap rounds). The barrier of b72bb75 costs nothing measurable (2592 blocks of dim 64:
  12 s either way on the RTX 3080). The run-to-run differences are gone: two runs at 462a423
  gave the same checksums (`cuda_bridge/README.md`).
- **Running out of device memory panics.** `Vector` has no `Result`, so `zeros_like`,
  `zeros_with` and `select` of the device fields panic (`must`) when the device refuses; the
  message is the CUDA error. A solver that narrows its vectors (`Narrowed::narrow`) holds the
  old set and at most three quarters of it at once. Making `Vector` fallible would touch every
  solver; not planned. (`Subspace::rotate` of more than 32 vectors used to make n full copies
  one after the other; since 2026-10-19 it needs one more set only.)

## Known, not understood

- With the multigrid, BiCGStab on the exact adjoint for the Pauli-Villars coarse factor is far
  worse than MINRES on `H_c`, also for `c = 0` (373 against 14 outer iterations, stout 12⁴).
- The Bridge++ `alt_eo` reference for the domain-wall even-odd Pauli-Villars ratio differs from
  Ruttice by 1e-5; external checks agree with Ruttice.
