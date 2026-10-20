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
- **Check.** Coarse operators equal to rounding (`--test multigrid`); setup time of
  `examples/mdwf_pv_eo_mg.yaml`.

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
(15 iterations) against 0.9 s (14). **Not measured on the A30**: rerun `mg_24.yaml` there.
The checksums differ from earlier runs from the first bootstrap vector on (another order of the
sums); runs agree with each other.

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

### 12. The Gram matrix one pair at a time

- **Why.** `gram` (`src/solver/preconditioner/multigrid/mdwf_even_odd.rs`) copies a vector with
  `select` for every pair and reduces each pair alone: 1024 dot products of two fine vectors and
  1024 copies for 32 vectors, about 2 s on the A30 for each of the six Gram matrices of a setup
  (estimated, not measured). The CPU does the same.
- **How.** ⟨x_i, y_j⟩ for all i at once (`x.dot` against y_j repeated, or an X†Y kernel that reads
  every vector once).

### 13. Smaller ones

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
  iteration). Most likely not the code: the runs at 462a423 were those with
  `RUTTICE_CHECKSUM=1` (29 checksums), and the checksums are taken inside the timed setup
  (item 13; about 56 GB hashed at about 0.9 GB/s on 24³×48; 12⁴ on an RTX 3080 with them 24 s,
  without 22 s). The barrier of b72bb75 costs nothing measurable (2592 blocks of dim 64: 12 s
  either way on the RTX 3080). **Check:** 24³×48 on the A30 without `RUTTICE_CHECKSUM`. The
  run-to-run differences are gone: two runs at 462a423 gave the same checksums
  (`cuda_bridge/README.md`).
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
