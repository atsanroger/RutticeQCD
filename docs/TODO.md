# To do

Work that is known and not done, so that the next person (or session) can pick it up without
rediscovering it. Each item says why, where, how, how much, and how to check it. Nothing here
changes results; items marked *performance* change speed, the others only the structure.

Priorities: finishing the multigrid paper (`docs/mg_background/MDWF_MULTIGRID_NOTE.tex`) comes
first; the structural items after it, in the order given. Last reviewed 2026-10-18.

## Structure

### 1. One propagator driver for the CPU and the device

- **Why.** `src/app/mdwf.rs::run_in` (about 290 lines) and `src/app/mdwf_cuda.rs::run` (about 270)
  are parallel copies: build the operators, set up the multigrid or the Chebyshev
  preconditioner, a `solve` closure that dispatches the solver and prints one line, then the
  random right-hand sides or the point-source propagator and `write_measurements`. A feature
  added to one is missing from the other (mixed-precision CGNE exists on the device only).
- **How.** A backend trait for what differs (field types, operators, the multigrid cycle, host
  and device copies of the source and solution); the driver written once on it. The device
  types are already `Vector`, `FermionOp`, `EvenOdd`.
- **Effort.** Medium to large.
- **Check.** `cargo test --release --features cuda-bridge --test run --test mdwf --test cuda_bridge`;
  the iteration counts of `examples/mdwf_pv_eo_mg.yaml` on the CPU and the device unchanged.

### 2. One solver dispatch

- **Why.** The `match` on `SolverType` that builds GCR, Block GCR, BiCGStab, FGMRES or CG is
  written three times (`src/app/mdwf.rs`, `src/app/mdwf_cuda.rs`, `src/app/wilson.rs`), including
  details such as how FGMRES's `number_of_kept_directions` becomes its restart length. A change
  made in one place is easily missed in the others.
- **How.** One function in `src/solver/` (or `app/`) from the solver block to a solve on any
  `Vector`, used by the three.
- **Effort.** Small. Do it before item 1, which then uses it.
- **Check.** `cargo test --release --test run --test wilson --test mdwf`.

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

### 4. Move the CUDA FFI declarations out of `src/backend/cuda_bridge/mod.rs`

- **Why.** About 750 of its 1290 lines are the `DeviceReal` trait with the `extern` functions of
  the kernels; the rest is buffers, geometry and helpers.
- **How.** `cuda_bridge/ffi.rs`. Purely moving code.
- **Effort.** Small.
- **Check.** `cargo clippy --all-targets --features cuda-bridge`; `--test cuda_bridge`.

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

- The dense contour of setup B on 12⁴ (`--contour 1.4,0,0.93,96`) for the Lipschitz lower bound;
  then update `fig_spectrum`, Table `tab:pseudo` and remove the last `\todo` in `results.tex`.
- Not measured, listed in the note: more configurations (the main request to whoever has GPUs:
  see `reproduce.tex`), A30 timings of the sensitivity table, Lüscher's coherence test for the
  mass-scan basis, a rigorous lower bound on `sigma_min` (Kato–Temple), the spectrum on
  `24^3x48` (does not fit in 24 GB).

## Known, not understood

- Two builds (`535c9ef`, `ca1e892`) give up to two iterations difference for the same
  parameter file on the A30 in three of eight runs: a code change or non-deterministic device
  reductions.
- With the multigrid, BiCGStab on the exact adjoint for the Pauli-Villars coarse factor is far
  worse than MINRES on `H_c`, also for `c = 0` (373 against 14 outer iterations, stout 12⁴).
- The Bridge++ `alt_eo` reference for the domain-wall even-odd Pauli-Villars ratio differs from
  Ruttice by 1e-5; external checks agree with Ruttice.
