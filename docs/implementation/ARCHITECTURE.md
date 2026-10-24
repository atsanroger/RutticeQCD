# Architecture

How RutticeQCD is put together: the layers, the traits that join them, how the CPU, the
device (`cuda_bridge`) and several processes plug in, and what happens in a run. The layout of
the directories is in [DEVELOPMENT.md](../DEVELOPMENT.md); the CUDA kernels one by one in
[KERNELS_NOTE.pdf](KERNELS_NOTE.pdf); writing a new kernel in [GPU_KERNELS.md](GPU_KERNELS.md);
several GPUs in [MULTI_GPU_NOTE.pdf](MULTI_GPU_NOTE.pdf).

## The layers

Each layer knows only the traits of the one below it.

| Layer | Directory | What it provides | Its traits |
|---|---|---|---|
| site arithmetic | `src/kernel` | complex numbers, colour matrices, spinors, gamma matrices, SIMD lanes | `Number`, `Real`, `GammaRepre`, `Lane` |
| fields | `src/field` | `Field<T, Kind, Sites, Layout>`; vectors and linear maps | `Vector`, `LinearFn`, `LinearOp`, `Subspace`, `Storage` |
| lattice and processes | `src/lattice`, `src/comm` | geometry, neighbours, even-odd numbering; global sums and halo exchanges | `Comm`, `HaloExchange` |
| operators | `src/fermion` | Wilson, clover, Möbius domain-wall; their even-odd blocks | `FermionOp`, `EvenOdd`, `HopTerms`, `SplitParity`, `SingleOperator` |
| solvers | `src/solver` | CG, BiCGStab, GCR, FGMRES, MINRES, multishift, mixed precision | `Solver` |
| preconditioners | `src/solver/preconditioner` | Chebyshev, the multigrids (Wilson, domain-wall), the levels below the coarse lattice | `CoarseOperator`, `Transfer`, `Corrects`, `MdwfBackend`, `Cycles` |
| eigensolvers | `src/eigen` | Lanczos, Krylov-Schur, singular values, pseudospectra | (on `Vector`, `LinearOp`) |
| HMC | `src/hmc` | gauge and fermion actions, their forces, integrators, the multigrid of HMC | `LinkField`, `ForceOp`, `MdAction`, `EvenOddHalves` |
| programs | `src/app`, `apps/` | parameter files, building what they name, running and reporting | — |

The rule that makes the rest possible: **a solver sees only `Vector` and `LinearFn`**. It does
not know whether the vector is a CPU field, a device field, a coarse field, or several of them;
nor whether the map is an operator, a Schur complement or a preconditioned product. So each
algorithm is written once.

## Vectors: one interface, many storages

`Vector` is what a Krylov solver needs: `zeros_like`, `axpy`, `dot`, `norm_sqr`, and fused forms
(`axpy_norm_sqr`, `dot_norms`, `xpay_norm_sqr_dot`, ...) that save passes over memory. A field
holds several vectors at once (right-hand sides, test vectors), and every operation takes one
coefficient per vector.

| Type | Where | Used for |
|---|---|---|
| `Field<T, Spinor, Sites, Layout>` | CPU, `Aos` or SIMD `Aosoa<W>` | fine fields |
| `Field<T, Coarse>` | CPU | coarse fields of the multigrid |
| `DeviceSpinor<T>` | device | fine fields |
| `DeviceCoarse<T>` | device | coarse fields and the levels below |

Reductions are summed over the processes inside `dot` and `norm_sqr` (each type holds its
communicator), so nothing above has to know about processes.

## Operators

`FermionOp<V>` applies D, D† or γ5 D to vectors of type V. `EvenOdd<H>` gives the blocks of the
even-odd decomposition (hops to a parity, the block within a site, its inverse, fused products),
from which `fermion::even_odd` builds the operators of the even sites once for every backend:
`SchurFactor`, `PvRatio` (the Pauli-Villars preconditioned operator), `AfterQ` (Neff's α).

| Operator | CPU | device |
|---|---|---|
| Wilson | `Wilson<T>` | `CudaWilson<T>` |
| clover | `Clover<T>` | `CudaClover<T>` |
| Möbius domain-wall | `Mdwf<T, G, L>` | `CudaMdwf<T>` |

`SingleOperator` adds what mixed precision needs (the operator in single precision on the links
of the double one); `fermion::SingleAll` and `SingleEven` are the mixed solvers written once on
it.

## One algorithm, two backends

The CPU and the device do the same things because each algorithm is written once against
traits, and a backend supplies only the parts that touch its memory. The table is in
[DEVELOPMENT.md](../DEVELOPMENT.md) ("One algorithm, two backends"); in short:

- the multigrid setup of domain-wall fermions (test vectors, bootstrap, P, levels, coarse parts)
  is written once in `MdwfBackend::rebuild`; `OnCpu` and `cuda_bridge::mdwf_backend::OnDevice`
  supply the fields and kernels;
- the multigrid of HMC (cache, rebuilds, retries, Neff's α) is `hmc::dwf_multigrid::with_solves`;
- the propagator runs are `app::mdwf::solve_sources`, the eigensolver runs `app::eigen::found`
  over an `EigenBackend`.

## The device backend (`src/backend/cuda_bridge`)

| File | Role |
|---|---|
| `ffi.rs` | the C functions of `cuda_bridge/include/bridge_cuda.h`, the CUDA runtime, `DeviceReal` (the kernels of one precision as a trait) |
| `mod.rs` | device memory (`DeviceBuffer`, a stream-ordered pool), the geometry the kernels take, choosing the GPU of a process |
| `field.rs` | `DeviceSpinor`, its `Vector` impl, parities |
| `blas.rs`, `columns.rs` | the linear algebra of fields of several vectors |
| `mdwf.rs`, `wilson.rs` | the operators |
| `gauge.rs` | links on the device, stout smearing, HMC's link operations (`LinkField`) |
| `coarse.rs` | P (`DeviceBasis`), coarse fields, the coarse operator (`DeviceCoarseOp`) and its ghosts |
| `levels.rs` | the levels below the coarse lattice |
| `correction.rs`, `multigrid.rs`, `mdwf_backend.rs` | the coarse correction, the setup, `OnDevice` |
| `halo.rs` | several processes: face exchanges (NCCL or through the host), `SplitReal` |
| `hmc_mixed.rs`, `hmc_multigrid.rs` | mixed precision and the multigrid of HMC on the device |

The C/CUDA side is `cuda_bridge/` (built by `build.rs` with cmake into a static library):
`include/` holds the kernels as templates (`fermion/`, `gauge/`, `multigrid/`, `linalg/`, `base/`),
`src/` the launchers with C entry points for each precision.

Errors: making a field or an operator returns `CudaError`; the operations of `Vector` and
`FermionOp` panic if the runtime fails. Kernels are asynchronous; a failing kernel shows at the
next transfer or reduction.

## Several processes

`Lattice` holds the local extents, the process grid and a `Comm`: `SingleComm` (one process),
`LocalComm` (threads as processes, for tests) or `MpiComm` (feature `mpi`).

- CPU: the hops overlap their halo exchange with the interior (`fermion::wilson::sweep`); gauge
  paths read an extended field (`gauge::paths::Extended`).
- Device: the same structure, with the faces as half spinors, interior and surface kernels, an
  extended field of links for the gauge field, and ghost sites for the coarse lattice
  (`halo.rs`, `coarse.rs`, [MULTI_GPU_NOTE.pdf](MULTI_GPU_NOTE.pdf)).

## A propagator run

1. `app::run_on` reads the file (`app::params`), checks it, and makes the `Lattice`.
2. The links are read and smeared (`app::load_links`, on the device for stout with `cuda_bridge`).
3. The operator is built (`Mdwf` or `CudaMdwf`), with Neff's α.
4. If asked, the multigrid is set up (`MdwfBackend::rebuild`).
5. `solve_sources` makes each source, solves on the even sites through `PvRatio` (or the left
   form), reconstructs the odd sites, checks the residual with the whole operator, and makes the
   four-dimensional fields.
6. The contractions are done on the host and written.

## An HMC trajectory

1. Momenta drawn on the host in Bridge++'s global order (`hmc::draw_momenta`), pseudofermions the
   same way.
2. The integrator (`hmc::integrator`) steps the links: each `MdAction` gives its force; the gauge
   actions from staples, the fermion actions from solves (`hmc::solve`, the multigrid through
   `with_solves`) and the derivative of the operator (`ForceOp`), through the chain rule of
   stout smearing where the action is smeared.
3. The Metropolis test with ΔH summed over the processes.

`LinkField` is implemented by `Field<f64, Gauge>` (CPU) and `DeviceLinks` (device), so the
integrator and the actions are written once.

## Where to look for what

| Question | Place |
|---|---|
| how a kernel works | [KERNELS_NOTE.pdf](KERNELS_NOTE.pdf) |
| how to write one | [GPU_KERNELS.md](GPU_KERNELS.md) |
| several GPUs | [MULTI_GPU_NOTE.pdf](MULTI_GPU_NOTE.pdf) |
| the multigrid | `docs/mg_background/` |
| what the parameter files take | [USAGE.md](../USAGE.md) |
| what is not done | [TODO.md](../TODO.md) |
