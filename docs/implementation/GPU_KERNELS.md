# Writing a GPU kernel in RutticeQCD

How a CUDA kernel is written here and how it reaches Rust, with the rules the existing kernels
follow and why. What each existing kernel computes is in [KERNELS_NOTE.pdf](KERNELS_NOTE.pdf).

## 1. The layout decides everything

Every device field is stored in blocks of 32 sites, the same component of the 32 sites
consecutive (`base/aosoa.cuh`):

```
a(c, n) = (⌊n/32⌋ · N_c + c) · 32 + (n mod 32)
```

A warp is 32 threads. If thread j of a warp handles site 32k + j and reads component c, the 32
addresses are consecutive: one memory transaction. So:

- **give consecutive threads consecutive sites** (`threadIdx.x` the site);
- loop over the components of a site inside a thread;
- a field on one parity numbers its sites by x/2; several vectors follow one another, each padded
  to whole blocks (`vector_reals`).

Most kernels are limited by memory, not arithmetic. A kernel that reads in another order loses
most of the bandwidth.

## 2. Threads: (site, fibre)

The fermion kernels launch a 2D grid (`site_grid`, `site_block` in `fermion/hops.cuh`):

- `threadIdx.x`: the site;
- `threadIdx.y`: the fibre — one fifth coordinate s of one vector.

The fibres of a site read the same links, which come from the cache after the first. At most 256
threads a block: a thread may use 255 registers and a block has 65536.

## 3. Where the code goes

1. **The kernel**, as a template on the precision, in `cuda_bridge/include/<area>/<name>.cuh`
   (`fermion/`, `gauge/`, `multigrid/`, `linalg/`). Shared device functions (a leg of the hop,
   `fifth_term`, `load_link`) are `__device__ __forceinline__` there too.
2. **The launcher**, in `cuda_bridge/src/<area>/<name>.cu`, inside `extern "C"`, one entry per
   precision (`..._f32`, `..._f64`), taking a `cudaStream_t`. It computes the grid and launches.
3. **The declaration** in `cuda_bridge/include/bridge_cuda.h`, with a comment saying what it
   computes.
4. **The Rust binding**: an `extern "C"` declaration, then a method of a trait over the
   precisions — `DeviceReal` (`ffi.rs`, the `device_real!` macro) for the core kernels, or a trait
   of its own module (`SplitReal` in `halo.rs`, `LevelReal` in `levels.rs`) for a family. Generic
   code calls `T::kernel(...)` for `T: DeviceReal`.
5. **The safe wrapper**: a method of the type it belongs to (`CudaMdwf`, `DeviceCoarseOp`, ...),
   which checks the shapes with `assert!`, and calls the kernel in an `unsafe` block with a
   `// SAFETY:` comment saying why the pointers and sizes are right.

New `.cu` files go into `BRIDGE_CUDA_SOURCES` in `cuda_bridge/CMakeLists.txt`.

## 4. Memory and streams

- Allocate with `DeviceBuffer` (`mod.rs`): it takes memory from a stream-ordered pool and frees it
  in stream order. A `cudaMalloc` and a synchronising `cudaFree` per temporary field cost about
  0.65 ms each; the pool about 17 µs.
- Launch on `STREAM` (the default stream) unless the kernel is part of an overlap (the halo code
  has its own stream for NCCL, joined by events).
- Do not synchronise in a wrapper. Kernels are asynchronous; what needs their result on the host
  waits (a reduction, a transfer). `finish()` exists for the places that must wait.

## 5. Determinism

Runs repeat bit for bit; keep it so.

- **No atomic additions in sums.** A reduction is two kernels in a fixed order: a tree inside each
  block, then one block adding the block sums (`linalg/blas.cuh`). The coarse operator adds the
  lanes' parts by a fixed butterfly.
- Keep the order of a sum the same in every variant of a kernel (the interior and surface kernels
  of several processes sum the legs in the order of one process, which is why they agree bit for
  bit).

## 6. Registers, shared memory, precision

- Check the registers of a new kernel (`nvcc -Xptxas -v`, or Nsight Compute). Above about 128 a
  thread, fewer blocks fit on a multiprocessor. `__launch_bounds__` caps them; check for spills.
- Double precision has twice the registers per value: some kernels take the legs one after the
  other in double precision for that reason (`hops` in `hops.cuh`).
- Shared memory has 32 banks of 4 bytes. When the threads of a warp read a column of a tile whose
  rows are a multiple of 128 bytes apart, they hit the same banks; storing column k of row r at
  `k ^ f(r)` spreads them (`fifth_gemm.cuh`, `hop_gemm_kernel`; the derivation is in the kernel
  note).
- Single precision is for what is corrected afterwards (smoothers, inner solves, the cycle);
  TF32 only for products whose error is corrected.

## 7. Several processes

A kernel that reads neighbours must handle a cut lattice (see
[MULTI_GPU_NOTE.pdf](MULTI_GPU_NOTE.pdf)):

- fermion hops: template on `Part` (whole, interior, surface) and read remote legs from a `Halo`
  (`hops<Float, Split>`);
- gauge paths: read an extended field (`StoutGeometry` with local extents and depths);
- coarse operators: neighbours through a table that may point at ghost sites.

A kernel that does not read neighbours needs nothing.

## 8. Counting the work

Every kernel launch adds its floating-point operations and the bytes it moves to the counter of
`src/meter.rs` (as QUDA's `Tunable::flops`, `bytes`): in the Rust wrapper, from the shapes, one
call of `meter::add` (or the helpers `count_fine` in `mod.rs`, `count` in `columns.rs`). The
bytes are counted as QUDA counts them: every value a site reads from memory (eight neighbouring
spinors and eight links for a hop), the caches left out, so the intensity can be compared with
the literature. A new kernel gets its count in its wrapper; the runs print the totals (Flop,
GFlop/s, Flop/byte by kind of work and precision) after each solve and trajectory. A count costs
about 3 ns on the host and nothing on the device.

## 9. Testing

- Compare with the CPU in `tests/cuda_bridge.rs` (`same_operator`, `same_blocks`, ...): the device
  and the CPU agree to rounding.
- For several processes compare with one process **bit for bit** (`LocalComm::run`, threads as
  processes on one GPU), and break the kernel on purpose once to see the test fail.
- `RUTTICE_CHECKSUM=1` prints checksums of the stages of a run, to compare two builds bit by bit.

## 10. Building

`cargo build --release --features cuda-bridge` runs cmake on `cuda_bridge/` (`build.rs`).
`RUTTICE_CUDA_ARCH` chooses the architectures (default `80;86`); one architecture compiles faster.
`--features nccl` links NCCL (`NCCL_HOME`).
