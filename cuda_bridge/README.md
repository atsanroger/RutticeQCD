# bridge_cuda: High-Performance GPU Lattice QCD Library

`bridge_cuda` is a modernized, standalone CUDA/C++ kernel library for Lattice QCD, originally refactored and extracted from the Bridge++ CUDA codebase (`test_alt_amg_dw_GEMM`, Wei-Lun Chen, Issaku Kanamori, Hideo Matsufuru).

It holds GPU kernels for Wilson, Clover, and Möbius Domain Wall Fermions (MDWF), Domain Decomposition (SAP / Block DD), Adaptive Multigrid (AMG), Multi-RHS Dirac operators, gauge actions & smearing, and linear algebra routines. Section 2 says which of them are checked against the CPU operators of RutticeQCD; the others are not.

---

## 1. Architectural Layout (RutticeQCD Hierarchy)

The directory structure and namespace strictly mirror the modular architecture of RutticeQCD:

```
cuda_bridge/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── bridge_cuda.h             # The C interface Rust binds (every entry commented)
│   ├── base/
│   │   ├── alloc.cuh             # cudaMalloc that gives back what the memory pool keeps if refused
│   │   ├── constant.h            # Nc, Nd, the warp size
│   │   ├── types.h               # Geometry, the halo of several processes, parameter structs
│   │   ├── aosoa.cuh             # Blocks of 32 sites; GridStride
│   │   ├── su3.cuh               # SU(3) products and the third row from the first two
│   │   └── gamma.cuh             # gamma matrices, chiral projections
│   ├── fermion/
│   │   ├── hops.cuh              # The hops of all three operators (Wilson and clover D too), parts for several processes
│   │   ├── clover.cuh            # T psi, the block of the clover operator within a site
│   │   ├── clover_links.cuh      # The clover term and its force made of the links
│   │   ├── mdwf.cuh              # Möbius DWF: D in one pass, the fifth dimension
│   │   ├── fifth_gemm.cuh        # What acts within a site of the DWF operator, as a matrix product (mma)
│   │   └── force.cuh             # The derivative of the hops by the links
│   ├── gauge/
│   │   ├── stout.cuh             # Stout smearing and its chain rule, extended fields
│   │   └── links.cuh             # HMC's operations on the links, the gauge force and action
│   ├── multigrid/
│   │   ├── fold.cuh              # P, restriction and prolongation, the coarse operator
│   │   └── levels.cuh            # The levels below the coarse lattice
│   └── linalg/
│       ├── blas.cuh              # The linear algebra of sets of vectors, deterministic reductions
│       └── cast.cuh              # f64 <-> f32
├── src/                          # The launchers, extern "C", one per precision
│   ├── fermion/                  # clover.cu, force.cu, mdwf.cu (the hops of all three operators)
│   ├── gauge/stout.cu            # every kernel that walks paths of links (clover term included)
│   ├── multigrid/                # fold.cu, levels.cu, dense.cu (cuSOLVER for the last level)
│   └── linalg/blas.cu
└── tests/test_mdwf_1pass.cu
```

---

## 2. What is checked

Checked against the CPU operators of RutticeQCD to 1e-13 (`tests/cuda_bridge.rs` there, feature
`cuda-bridge`) in double precision, and to what single precision allows in single precision,
for periodic, antiperiodic and Dirichlet boundaries:

- `bridge_cuda_hop` (and its parts on several processes): the Wilson operator D and D^dag
  (`from` = in at -kappa), the clover operator (with its blocks), and the hops of all three.
- `bridge_cuda_clover_term`, `bridge_cuda_clover_blocks`: the block of the clover operator within
  a site (or its inverse), and the blocks made of the links.
- `bridge_cuda_mdwf_dslash_1pass`: the Möbius domain-wall operator D.
- `bridge_cuda_hop`, `bridge_cuda_mdwf_fifth`, `bridge_cuda_fifth_gemm`: its parts, of which
  D^dag and the even-odd blocks D_pq, D_pq^dag, C, C^-1 and their adjoints are made, with
  Neff's alpha. `bridge_cuda_fifth_gemm` is a matrix product in the fifth dimension, in double
  precision on the tensor cores (hand-written `mma` PTX).
- `bridge_cuda_hop` with Ls = 1 and `bridge_cuda_clover_term` on one parity: the even-odd blocks
  of the Wilson and clover operators. `bridge_cuda_parity_copy`.
- `bridge_cuda_gamma5`; `bridge_cuda_blas_*`: axpy, scal, complex axpy and aypx, norm and
  inner product, and the rotation of a set of vectors (`bridge_cuda_blas_rotate_*`), for any
  number of vectors in one call; the Gram matrix of two sets (`bridge_cuda_blas_gram_*`, 37 × 6
  and 6 × 6 vectors) against the inner products of the pairs.
- `bridge_cuda_fifth_gemm_tf32_f32`: the product in the fifth dimension in TF32, to what TF32
  allows.
- The domain-wall multigrid (`multigrid/fold.cuh`): P folded on the fifth dimension, made from
  test vectors and orthonormalised on the device (`bridge_cuda_fold_fill_*`,
  `bridge_cuda_fold_orthonormalise`), the restriction and prolongation
  (`bridge_cuda_fold_restrict_*`, `bridge_cuda_fold_prolong_*`), with P in single or half
  precision; the coarse operators built on the device (`bridge_cuda_hop_split_*`, put together
  there by `bridge_cuda_coarse_units_*` and `bridge_cuda_coarse_columns_*`), applied
  (`bridge_cuda_coarse_pass_*`, `bridge_cuda_coarse_gamma5_*`), the inverses of their blocks
  and |H_xx|^-1 (`bridge_cuda_blocks_inverse`, `bridge_cuda_blocks_abs_inverse_gamma5`); the
  four-dimensional fields of a five-dimensional one (`bridge_cuda_walls_*`). The coarse
  correction on the device agrees with that of the CPU to 1e-9.
- The levels below the coarse lattice (`multigrid/levels.cuh`): their local vectors made
  orthonormal on the device (`bridge_cuda_level_basis_*`), the restriction and prolongation
  between levels (`bridge_cuda_level_restrict_*`, `bridge_cuda_level_prolong_*`), the operators
  of the levels by probing (`bridge_cuda_level_sources_*`, `bridge_cuda_level_probe_*`), then
  applied by `bridge_cuda_coarse_pass_*`, |B|^-1 of their blocks for the Pauli-Villars factor
  (`bridge_cuda_blocks_abs_inverse`), and the last level as a dense matrix
  (`bridge_cuda_level_dense`) inverted, or |H|^-1 of it, by cuSOLVER and cuBLAS
  (`multigrid/dense.cu`: `bridge_cuda_dense_inverse`, `bridge_cuda_dense_abs_inverse`, of any
  size) and applied by `bridge_cuda_dense_apply_*`. Their test vectors, the V-cycle of the
  quark factor with the bootstrap of a level and the preconditioner of the Pauli-Villars
  factor agree with those of the CPU to 1e-12, with a last level of up to 1536 complex numbers
  in the test.
- The exact adjoint of a coarse operator (`bridge_cuda_coarse_adjoint_*`): its blocks and links
  conjugate-transposed and gathered from the sites the hops go to, an operator of its own; the
  coarse correction with the Pauli-Villars factor solved on it agrees with that of the CPU to
  1e-9.

The operators are in the Dirac representation (gamma_4 diagonal). `alpha` of
`BridgeCudaMdwfParams` is not used: Neff's Q is one of the matrices of `bridge_cuda_fifth_gemm`.

Each kernel is derived in `docs/implementation/KERNELS_NOTE.pdf` of RutticeQCD.

Conventions:

- All public symbols are `extern "C"` with the prefix `bridge_cuda_*` (`include/bridge_cuda.h`);
  the kernels are templates in `namespace bridge_cuda`.
- Fields are in blocks of 32 sites (`base/aosoa.cuh`); a field on one parity has site
  (x, y, z, t) at x/2 + (nx/2)(y + ny (z + nz t)).
- Reductions (plaquette, norms, dot products) are deterministic tree reductions, and no kernel
  uses atomics. The linear algebra of sets of vectors (`linalg/blas.cu`) reduces in two stages
  on the device, in double precision for either field precision, and brings only the results
  to the host (page-locked, one wait for the stream); the fused operations of `Vector`
  (`axpy_norm_sqr`, `xpay_norm_sqr_dot`, `dot_norms`, `xpay`, `axpy2`, `axpy_aypx`,
  `scale_axpy2_norm_sqr`) are one pass each.
- On an A30 the domain-wall multigrid on 28^4 and 24^3x48 gave different last digits from run
  to run, from the first round of the bootstrap on. A data race is the likely cause, fixed on
  2026-10-19: in the Jacobi kernel of `|H_xx|^-1` (`multigrid/fold.cuh`,
  `blocks_abs_inverse_gamma5_kernel`) thread p overwrote a_pq while the other warps of the
  block (dim > 32) could still be reading it. It never showed in the runs on an RTX 3080 (12^4,
  81 coarse sites: the same bit for bit with and without the fix), but it does there on 2592
  random blocks of dim 64 (up to 6e-2 apart, 2026-10-20). With the fix (cbe655b) two runs of
  24^3x48 on the A30 gave the same 29 checksums, for both domain-wall setups (2026-10-19).
- The coarse operator (`coarse_pass_kernel`, `multigrid/fold.cuh`) has a warp for each row of
  a site's matrices, its lanes the columns, so that a warp reads a row as one stretch of
  memory; the matrices stay row-major. A thread for each row (until 2026-10-20) read the rows
  of a warp 2 dim reals apart: on an RTX 3080, 2592 sites of dim 64 (24^3x48 in blocks of 4^4),
  diagonal block and eight links, 3.59 ms against 1.13 ms now in single precision (1.13 ms is
  675 GB/s of matrices), 14.4 ms against 3.10 ms in double. The lanes' parts of a row are added
  in a fixed order, so runs still agree bit for bit with each other, but not with runs before
  that change from the first bootstrap vector on (the coarse sums are added in another order);
  12^4 then needed 14 outer iterations instead of 15.
- The fields of the Rust side (`DeviceBuffer`) come from the stream-ordered memory pool of the
  device (`cudaMallocAsync`, freed by `cudaFreeAsync` on the default stream, on which every
  kernel runs), set to keep what is freed; until 2026-10-20 every field was a `cudaMalloc` of
  its own and freeing one waited for the device. The scratch of the library itself
  (`linalg/blas.cu`, `multigrid/dense.cu`) is still `cudaMalloc`, by
  `bridge_cuda::malloc_or_trim` (`base/alloc.cuh`): if the device refuses, what the pool keeps
  unused is given back and the allocation tried once more, as the Rust side does for its own.
  Under WSL the device takes more than it has (it pages to the host), so that path has not
  been seen to run.
- The coarse operator of the domain-wall multigrid is made and kept on the device: P† of each
  hop and of the block goes into the matrices there (`coarse_columns_kernel`, the links in the
  precision of the coarse solves), and only the tables of the coarse lattice come from the host
  (until 2026-10-20 the matrices went to the host for every batch of components and back).
- The Gram matrix of two sets of vectors (`blas_gram_kernel`, `linalg/blas.cuh`) is made by
  tiles of 4 × 4 pairs, in double precision and in a fixed order, so a vector is read once for
  every four of the other set.

---

## 3. Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
make -j
```

Outputs:
- Shared library: `libbridge_cuda.so`
- Static library: `libbridge_cuda_static.a`
