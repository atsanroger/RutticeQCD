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

**Done 2026-10-20, mixed precision by reliable updates.** The defect correction restarted CG
from zero at each correction: on 24³×48 at m = 0.001 (point source, 8f4fd7e) 5261–5951
iterations a column against 3613–3619 in double precision, and two of the 12 columns stopped at
|Dx−b|² 4e-15 for a target of 1e-18 (the rule is on the residual of K†K; an error in the low
modes meets it while |Dx−b| does not), their PS 3e-6 off. Now one CG in single precision with
the solution and the residual made again in double precision each time |r|² falls by 1e-2
(`solver::reliable_cg`, `SingleCuda::reliable_normal`; `convergence_criterion_squared_inner`).
16⁴ point source, RTX 3080: at m = 0.001 365–374 iterations a column, the same as double
precision (404–495 before), |Dx−b|² 4e-18–8e-18 as in double (8e-20–7e-17 before), PS 2e-11
from double (2.5e-8 before), 0.9 s a column against 5.6 in double. HMC keeps the defect
correction (`hmc::mixed`).

On the A30 (24³×48, m = 0.001, 6bbfa74) only half of it holds. The point source at α = 0.6:
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

Where the multigrid's time goes (nsys, one solve at m = 0.001, 6bbfa74). The solve, 18.4 s
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
  ncu of a double CG iteration on the A30 at 6bbfa74 (24³×48): 40.1 ms, the four hops 30.7 ms
  (77%; 9.5, 7.8, 5.7, 7.7 ms) at 12% warps active with the FP64 pipe 39–52% busy, the
  fifth-dimension product on the DMMA tensor cores (also inside the hops), the BLAS at 86–89% of
  the DRAM bandwidth; 6.7 kB of DRAM traffic and 757 warp-instructions a 5D site.
  At a22f8c5 (A30, one run each against 6bbfa74): the four hops 30.7 → 25.1 ms (−18%; warps
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
  (`Source.number_of_rhs_per_solve`). On the A30 (24³×48 MG, 6bbfa74) 2 does not fit: the
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
  parameter changes on 12⁴ (Table `tab:costs`, at a147871 now). The commit in each caption.
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
- Readability, with the same pass (from a review of the PDF, checked against the text):
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
the A30: 24³×48, m = 0.001, point source, 12 columns, setup included; 584 s at 6bbfa74 (setup
311, solves 251). The aim is about 400 s, from the bandwidth floors of the ncu profiles
(estimates, each to be measured):

| | now | near the bandwidth (estimate) | saved |
|---|---|---|---|
| `fold_restrict` (setup 3800 launches, solve 2 an iteration) | 32.5 ms, DRAM 11% | about 5 ms (done: P read coalesced through shared memory; RTX 3080 16⁴ 4.64 → 1.31 ms, DRAM 31% with 256 blocks for 68 SMs; A30 to measure) | setup −65 s, solves −17 s |
| `coarse_pass` (30% of a solve, 28 s of the setup) | DRAM 22–37% | DRAM 60–70% (done: fields in shared memory, 16-byte matrix loads; RTX 3080 16⁴ 82.5 → 62.6 µs, DRAM 62 → 82%; A30 to measure) | solves −38 s, setup −14 s |
| setup host time and idle GPU | 19 + 40 s under nsys; the 19 s before the first kernel is nsys's own (plain run: smearing done at 1.9 s) | to be measured in a plain run | unknown |
| `level_restrict` (11% of a solve) | 47 single-block launches an iteration, 1.2 ms each (8.4%) | a warp an output (done; RTX 3080: 187 → 11 µs, 16⁴ setup 32 → 25 s) | A30, 046567d: solves 251.2 → 215.5 s with a22f8c5's hop, setup unchanged |

`level_restrict` on the A30 at 046567d (ncu): the last level 1.2 ms → 133 µs (16 blocks, still
latency-bound, but under 1% of a solve now), the grid-41 level 392 → 140 µs at 67% of the DRAM
bandwidth: done.

Measured on the A30 (12 columns, one run): 6bbfa74 584 s; 046567d 535.9 s (setup 298 s,
solves 215.5 s, the rest 22 s; 362 GCR iterations as before, PS within 7.3e-9); d01d2bb
(`fold_restrict` coalesced, `coarse_pass` fields in shared memory) **460.9 s** (setup 244 s:
bootstrap rounds 83 → 67 and 94 → 75 s; solves 191.6 s; the rest 25 s; 365 GCR iterations;
worst |Dx−b|² 2e-18 in one column, the others ≤ 1e-18; PS within 1.85e-8 of 046567d). 61 s
from the aim.
| `fold_prolong` | DRAM 31% | | −6 s |
| `blocks_abs_inverse_gamma5` | 3 launches a setup, 3.8 s each, 1.15 TB of DRAM traffic | H in shared memory, rounds of disjoint pairs (done; RTX 3080 0.70 → 0.10 s; A30 to measure) | −11 s |
| `hop_split<float>` (2 × 1024 launches, 9.4 ms) | 168 registers, one block an SM | two blocks an SM, 128 registers (done; RTX 3080 −20%, DRAM 88%; A30 to measure) | about −8 s |
| `fold_orthonormalise` | 42 blocks on 56 SMs, 12% warps active | more blocks | a few s |

`fold_restrict_tiled_kernel` on the A30 (d01d2bb): 11.6–12.2 ms in a solve (was 32.5), 5.8 ms in
the bootstrap (was 21.7), DRAM 29–34%, warps active 30%: shared memory (28 kB a block) allows 5
blocks an SM, and it waits on its loads (long scoreboard, wait). Tried on the RTX 3080 and not
kept: four loads of a thread issued before their stores (72 registers instead of 56, 10–15%
slower); P read straight into registers, no tiles (31% warps active, but the half-used sectors
are fetched again from DRAM: 307 → 510 MB a launch, the heavy shape 5.0 → 5.4 ms). Left:
copies to shared memory that do not go through registers (`cp.async`, sm_80), double-buffered.
`coarse_pass_kernel` on the A30 (d01d2bb): the passes with the hops 19–21% faster (DRAM 37 →
46%), those without 18% slower (128 registers for the unrolled links, warps 35 → 24%); now a
kernel of its own without the hops (RTX 3080: 70 → 44 registers, 19 → 16 µs).

The ncu of the setup (A30, 6bbfa74): the filter's `hop_kernel<float, many>` (728 launches, 15.8 s)
was the Chebyshev step's last hop, which a22f8c5 sends through `hop_gemm`. `fold_restrict` in the
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
