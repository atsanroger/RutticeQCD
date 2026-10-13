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
│   │   ├── wilson.cuh            # Wilson Dslash (interior, EO hopping, boundary packing)
│   │   ├── clover.cuh            # Clover term (6x6 hermitian blocks) & EO Clover Dslash
│   │   ├── mdwf.cuh              # Möbius DWF: D in one pass
│   │   ├── mdwf_eo.cuh           # Möbius DWF on one parity: the hop D_pq and C^-1
│   │   └── dd.cuh                # Block Domain Decomposition (SAP)
│   ├── multigrid/                # Adaptive Multigrid (AMG)
│   │   ├── coarse.cuh            # Coarse grid Dirac operator
│   │   ├── transfer.cuh          # Restriction, prolongation, block Gram-Schmidt
│   │   └── mrhs.cuh              # Multi-RHS batched Dslash
│   ├── gauge/                    # Gauge configurations & smearing
│   │   └── gauge.cuh             # Deterministic plaquette & APE smearing
│   └── linalg/                   # GPU Linear algebra
│       ├── blas.cuh              # Deterministic AXPY, SCAL, NORM2 tree reduction
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
`cuda-bridge`), in double precision, for periodic, antiperiodic and Dirichlet boundaries:

- `bridge_cuda_wilson_dslash`: the Wilson operator.
- `bridge_cuda_mdwf_dslash_1pass`: the Möbius domain-wall operator D.
- `bridge_cuda_mdwf_eo_hop`, `bridge_cuda_mdwf_site_inv`: its even-odd hop D_pq and the inverse
  of its part within a site, C^-1 and C^dag^-1, on fields of one parity.

These are the plain operators in the Dirac representation (gamma_4 diagonal); `alpha` of
`BridgeCudaMdwfParams` (Neff) is not used.

Not checked: the even-odd Wilson hop, the clover kernels, the multi-RHS kernels (these three
have no gamma matrices yet), SAP, the multigrid transfers and coarse operator, the gauge
kernels, single precision. `tests/test_clover_mg_transfer` fails with a CUDA error.

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
