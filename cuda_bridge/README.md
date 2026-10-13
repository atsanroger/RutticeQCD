# bridge_cuda: High-Performance GPU Lattice QCD Library

`bridge_cuda` is a modernized, standalone CUDA/C++ kernel library for Lattice QCD, originally refactored and extracted from the Bridge++ CUDA codebase (`test_alt_amg_dw_GEMM`, Wei-Lun Chen, Issaku Kanamori, Hideo Matsufuru).

It provides zero-overhead GPU kernels for Wilson, Clover, and Möbius Domain Wall Fermions (MDWF), Domain Decomposition (SAP / Block DD), Pauli-Villars preconditioned operators, Adaptive Multigrid (AMG), Multi-RHS Dirac operators, gauge actions & smearing, and linear algebra routines.

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
│   │   ├── mdwf.cuh              # 5D Möbius DWF (D5 hopping, Ls-parallel, register LU)
│   │   └── dd.cuh                # Block Domain Decomposition (SAP) & Pauli-Villars operator
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
    │   └── dd.cu                 # Host C-ABI wrappers for SAP & Pauli-Villars
    ├── multigrid/
    │   ├── multigrid.cu          # Host C-ABI wrappers for coarse operator & transfers
    │   └── mrhs.cu               # Host C-ABI wrappers for MRHS
    ├── gauge/
    │   └── gauge.cu              # Host C-ABI wrappers for Plaquette & Smearing
    └── linalg/
        └── blas.cu               # Host C-ABI wrappers for BLAS, Cast, and Basis conversion
```

---

## 2. Technical Highlights

- **Pure C-ABI Interface (`include/bridge_cuda.h`)**:
  - All public symbols are exported with `extern "C"` and prefixed with `bridge_cuda_*`.
  - Fully compatible with C, C++, and Rust (`bindgen` / FFI).
- **C++ Scope**:
  - All template kernels and device math reside cleanly in `namespace bridge_cuda`.
- **Möbius Domain Wall Fermion (MDWF)**:
  - Supports generalized parameters $b_5, c_5$. Standard Shamir DWF corresponds to $b_5 - c_5 = 1$.
  - Includes $L_s$-parallel odd-even hopping and on-chip register LU-inversion for $(b_5 D_4 + 1)^{-1}$.
- **Pauli-Villars Preconditioned AMG Support**:
  - Directly reproduces the algorithm in `pvprec_amg.tex`: solves $A = (D_{PV} C_{PV}^{-1})^\dagger C^{-1} D$ using Domain Decomposition (SAP) smoothing on blocks with parity `ieo`.
- **Numerical Determinism**:
  - Reductions (Plaquette, Norm2) use deterministic block-shared-memory tree reductions rather than floating-point non-deterministic `atomicAdd`.
- **Dual Precision**:
  - Native templates instantiated for FP32 (`float`) and FP64 (`double`).

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
