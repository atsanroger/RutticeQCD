# bridge_cuda: High-Performance GPU Lattice QCD Library

`bridge_cuda` is a modernized, standalone CUDA/C++ kernel library for Lattice QCD, originally refactored and extracted from the Bridge++ CUDA codebase (`test_alt_amg_dw_GEMM`, Wei-Lun Chen, Issaku Kanamori, Hideo Matsufuru).

It holds GPU kernels for Wilson, Clover, and Möbius Domain Wall Fermions (MDWF), Domain Decomposition (SAP / Block DD), Adaptive Multigrid (AMG), Multi-RHS Dirac operators, gauge actions & smearing, and linear algebra routines. Section 2 says which of them are checked against the CPU operators of RutticeQCD; the others are not.

---

## 1. Architectural Layout (RutticeQCD Hierarchy)

The directory structure and namespace strictly mirror the modular architecture of RutticeQCD:

```
gpu_fer/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── bridge_cuda.h             # Unified C-ABI public interface for FFI / bindgen
│   ├── base/                     # Foundation layer
│   │   ├── types.h               # Geometry, param structs, and constants
│   │   ├── aosoa.cuh             # AoSoA coalesced memory layout (SIMD width W=32)
│   │   ├── su3.cuh               # SU(3) complex matrix math & 3rd-row reconstruction
│   │   └── gamma.cuh             # Spin projection, gamma5, Dirac <-> Chiral basis
│   ├── fermion/                  # Fermion Dirac operators
│   │   ├── hops.cuh              # The hops all three operators share, on the lattice or one parity
│   │   ├── wilson.cuh            # Wilson D, boundary packing
│   │   ├── clover.cuh            # Clover D and the block within a site (6x6 Hermitian blocks)
│   │   ├── mdwf.cuh              # Möbius DWF: D, its hops on the lattice or one parity, C^-1
│   │   └── dd.cuh                # Block Domain Decomposition (SAP)
│   ├── multigrid/                # Adaptive Multigrid (AMG)
│   │   ├── coarse.cuh            # Coarse grid Dirac operator
│   │   ├── transfer.cuh          # Restriction, prolongation, block Gram-Schmidt
│   │   └── mrhs.cuh              # Multi-RHS batched Dslash
│   ├── gauge/                    # Gauge configurations & smearing
│   │   └── gauge.cuh             # Deterministic plaquette & APE smearing
│   └── linalg/                   # GPU Linear algebra
│       ├── blas.cuh              # axpy, scal, complex axpy/aypx, norm and inner product
│       └── cast.cuh              # On-device FP64 <-> FP32 precision conversion
└── src/
    ├── fermion/
    │   ├── wilson.cu             # Host C-ABI wrappers for Wilson
    │   ├── clover.cu             # Host C-ABI wrappers for Clover
    │   ├── mdwf.cu               # Host C-ABI wrappers for MDWF
    │   └── dd.cu                 # Host C-ABI wrappers for SAP
    ├── multigrid/
    │   ├── multigrid.cu          # Host C-ABI wrappers for coarse operator & transfers
    │   └── mrhs.cu               # Host C-ABI wrappers for MRHS
    ├── gauge/
    │   └── gauge.cu              # Host C-ABI wrappers for Plaquette & Smearing
    └── linalg/
        └── blas.cu               # Host C-ABI wrappers for BLAS, Cast, and Basis conversion
```

---

## 2. What is checked

Checked against the CPU operators of RutticeQCD to 1e-13 (`tests/cuda_bridge.rs` there, feature
`cuda-bridge`) in double precision, and to what single precision allows in single precision,
for periodic, antiperiodic and Dirichlet boundaries:

- `bridge_cuda_wilson_dslash`: the Wilson operator D and D^dag.
- `bridge_cuda_clover_dslash`, `bridge_cuda_clover_term`: the clover operator D and D^dag, and
  the block of it within a site (or its inverse), with the blocks made on the CPU.
- `bridge_cuda_mdwf_dslash_1pass`: the Möbius domain-wall operator D.
- `bridge_cuda_hop`, `bridge_cuda_mdwf_fifth`, `bridge_cuda_mdwf_site_inv`: its parts, of
  which D^dag and the even-odd blocks D_pq, D_pq^dag, C, C^-1 and their adjoints are made.
- `bridge_cuda_hop` with Ls = 1 and `bridge_cuda_clover_term` on one parity: the even-odd blocks
  of the Wilson and clover operators. `bridge_cuda_parity_copy`.
- `bridge_cuda_gamma5`; `bridge_cuda_blas_*`: axpy, scal, complex axpy and aypx, norm and
  inner product.

The operators are in the Dirac representation (gamma_4 diagonal). Neff's D^(alpha) = D Q is made
of the plain kernels and `bridge_cuda_mdwf_q`; `alpha` of `BridgeCudaMdwfParams` is not used.

Not there: the clover term from the links on the device. Not checked: the multi-RHS kernels (no gamma matrices yet), SAP, the multigrid transfers and coarse operator, the gauge
kernels. `tests/test_clover_mg_transfer` fails with a CUDA error.

Conventions:

- All public symbols are `extern "C"` with the prefix `bridge_cuda_*` (`include/bridge_cuda.h`);
  the kernels are templates in `namespace bridge_cuda`.
- Fields are in blocks of 32 sites (`base/aosoa.cuh`); a field on one parity has site
  (x, y, z, t) at x/2 + (nx/2)(y + ny (z + nz t)).
- Reductions (plaquette, norm) are deterministic tree reductions.

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
