# CLAUDE.md

Guidance for Claude Code (and anyone) working in this repository. Read it first; the user should
not have to repeat any of this.

## What this is

RutticeQCD: lattice QCD in Rust — Wilson, clover and Möbius domain-wall fermions, Krylov solvers,
the multigrids, eigensolvers, gradient flow and HMC — on the CPU (threads, SIMD, MPI) and on NVIDIA
GPUs through CUDA kernels extracted from Bridge++ (`cuda_bridge/`, Rust side
`src/backend/cuda_bridge/`). One Cargo package. Start with:

- `docs/implementation/ARCHITECTURE.md` — the layers, the traits between them, the backends
- `docs/implementation/GPU_KERNELS.md` — how a CUDA kernel is written and bound to Rust
- `docs/implementation/KERNELS_NOTE.pdf`, `MULTI_GPU_NOTE.pdf` — every kernel; several GPUs
- `docs/DEVELOPMENT.md` (layout, tests), `docs/USAGE.md` (parameter files), `docs/TODO.md` (open items)

## Principles

1. **The architecture is easy to understand; the complexity goes into the kernels.** Clear
   structure and high performance; kernel code may be hard. Performance tricks (fusion, swizzles,
   special layouts, overlap) stay inside a kernel or a backend and are explained in
   `docs/implementation/KERNELS_NOTE`; they do not leak into interfaces. If a speed-up needs an
   architectural complication, contain it.
2. **One algorithm, two backends.** Each algorithm is written once against traits; a backend
   supplies only what touches its memory. A feature added to the algorithm is on the CPU and the
   device at once, and the two offer the same features (table in `docs/DEVELOPMENT.md`).
3. **Exact and testable.** Operators on several processes give the result of one process bit for
   bit; reductions have a fixed order, so runs repeat bit for bit.
4. **Design from references first.** For a new algorithm or kernel: read Bridge++ (and any CUDA
   the user points to), write down the design, get the user's agreement, then code.
5. **Report faithfully.** Say what was checked and what was not; give estimates as estimates.

## Rust

**Structure**
- Algorithms (solvers, multigrid, eigensolvers, HMC) are generic over the traits of the layer below:
  `Vector`, `LinearFn`/`LinearOp`, `FermionOp`, `EvenOdd`, `SplitParity`, `CoarseOperator`,
  `Transfer`, `MdwfBackend`, `LinkField`, `ForceOp`. A solver never knows the storage, the backend
  or the number of processes.
- A backend implements those traits for its types (`Field`, `DeviceSpinor`, `DeviceCoarse`,
  `DeviceLinks`). Backend-specific code stays in its module (`src/backend/cuda_bridge`,
  `src/backend/mpi.rs`) behind its feature (`cuda-bridge`, `mpi`, `nccl`); the rest of the crate
  compiles without it.
- Before writing a second version of an algorithm for a backend, find the trait that lets one
  version serve both.
- Reductions in a `Vector` impl are summed over the processes of its lattice (`comm.sum`), so the
  layers above never see processes.

**Visibility**
- `pub` only for what outside code (apps, other users of the library) is meant to use; internal
  items are `pub(crate)` / `pub(super)`. The compiler then reports dead code.
- Do not export implementation details of a backend (buffers, kernel argument structs).
- Remove what a refactor leaves unused.

**Errors**
- Building something that can fail (an operator, a field, a parameter file) returns a `Result`
  (`CliError` for files and runs, `CudaError` for the device, `MultigridError`, ...). Operations of
  `Vector`/`FermionOp` have no error to return and panic on a broken invariant (`must`).
- Parameter files are checked when read: refuse what would run wrongly with a message naming the
  key and the fix; never silently ignore an option.

**Unsafe**
- Every `unsafe` block has a `// SAFETY:` comment saying why the pointers, sizes and aliasing are
  right. Check the shapes with `assert!` before calling a kernel.

**Style**
- Names and comments say what a thing is or does, in plain words; comments explain why. Match the
  surrounding code's density.
- Tests that need nothing external go in the module (`#[cfg(test)]`) or `tests/<subject>.rs`; the
  device against the CPU in `tests/cuda_bridge.rs`; several processes against one with
  `LocalComm::run`, bit for bit, broken once on purpose to see the test fail.

## CUDA

The kernels live in `cuda_bridge/include/<area>/*.cuh` (templates), their launchers in
`cuda_bridge/src/<area>/*.cu`, declarations in `cuda_bridge/include/bridge_cuda.h`. Full guide:
`docs/implementation/GPU_KERNELS.md`.

**Layout and threads**
- Fields are blocks of 32 sites, the same component of the 32 sites consecutive
  (`base/aosoa.cuh`). Give consecutive threads consecutive sites (`threadIdx.x`) so a warp reads
  one transaction; loop over a site's components inside the thread.
- Fermion kernels use a 2D grid: `threadIdx.x` the site, `threadIdx.y` the fibre (one s of one
  vector); the fibres of a site share its links. At most 256 threads a block.
- A field on one parity numbers sites by x/2; several vectors follow one another, each padded to
  whole blocks (`vector_reals`).
- A kernel over n independent elements (BLAS, casts, per-site operations of one thread a site)
  uses the grid-stride loop `for (size_t i : GridStride(n))` (`base/aosoa.cuh`) with a capped grid
  (`get_grid_size`): it covers any n, and consecutive threads take consecutive indices. Do not
  hand-write the loop. Kernels with a 2D (site, fibre) grid or a block per site keep one thread
  per element and return early past the end.

**Writing a kernel**
- A template on the precision (`Float`), shared pieces as `__device__ __forceinline__` functions
  (`fifth_term`, `load_link`, a leg of the hop).
- A launcher per precision inside `extern "C"` (`..._f32`, `..._f64`), taking a `cudaStream_t`,
  declared and commented in `bridge_cuda.h`.
- Rust binds it in a trait over the precisions (`DeviceReal` in `ffi.rs`, or a module trait such
  as `SplitReal`, `LevelReal`) and wraps it in a method that checks shapes.
- Kernels are asynchronous; do not synchronise in a wrapper. Memory comes from the stream-ordered
  pool (`DeviceBuffer`), never `cudaMalloc` in a loop.
- The wrapper counts the launch (`meter::add`, `count_fine`): its flops (Bridge++'s count) and
  bytes (QUDA's convention: every value a site reads from memory). The same for a new CPU
  operator or `Vector` operation, so the run reports stay complete.

**Performance rules**
- Most kernels are bound by memory: read each value once, fuse passes where it saves a read, keep
  the order of reads coalesced.
- Watch registers (`-Xptxas -v`): above ~128 a thread fewer blocks fit; use `__launch_bounds__`
  and check for spills. Double precision may need the work split (the hops take one leg at a time).
- Shared memory has 32 banks of 4 bytes: when a warp reads a column whose rows are 128 bytes apart,
  permute the columns (`k ^ f(row)`, see `fifth_gemm.cuh`) and write down the bank arithmetic.
- Single precision for what is corrected afterwards (cycles, smoothers, inner solves); TF32 only
  for products whose error is corrected.

**Determinism**
- No atomic additions in sums: a fixed tree per block, then one block over the block sums. Any
  variant of a kernel (interior/surface, split/whole) adds terms in the same order.

**Several processes**
- A kernel that reads neighbours handles a cut lattice: fermion hops through `Halo` and the
  interior/surface parts; gauge paths through an extended field; coarse operators through ghost
  sites in the neighbour table. A kernel that reads only its own site needs nothing.

**Documentation**
- Every kernel and every trick in it is explained in `docs/implementation/KERNELS_NOTE`
  (derivation line by line, thread layout, what limits its speed); update it with the kernel.

## Writing

- Comments and docs: plain, short sentences, the point first (Hsiang-nan Li's advice).
- Internal notes (LaTeX, `docs/implementation/`, `docs/mg_background/`): derivations written out
  line by line at the level of a new graduate student, every symbol defined; never labelled "for
  beginners". Prose meant for a future wiki in Markdown.
- Keep `docs/DEVELOPMENT.md`, `USAGE.md`, `CHANGELOG.md` and `TODO.md` in step with features.

## Building and testing

```bash
cargo build --release --features cuda-bridge        # + CUDA (cmake); RUTTICE_CUDA_ARCH picks GPUs
cargo test --release --features cuda-bridge --test cuda_bridge -- --test-threads=1
cargo clippy --release --all-targets [--features cuda-bridge]   # both clean
cargo fmt
```

- Features: `cuda-bridge` (CUDA kernels), `mpi` (needs MPI and libclang), `nccl` (`NCCL_HOME`),
  `hdf5`.
- `RUTTICE_CHECKSUM=1` prints checksums of the stages of a device run, to compare builds bit by bit.
- The small checks use the Bridge++ test configuration `examples/conf_04040408.txt` (4³×8).
