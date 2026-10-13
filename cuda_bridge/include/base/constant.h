#ifndef BRIDGE_CUDA_BASE_CONSTANT_H
#define BRIDGE_CUDA_BASE_CONSTANT_H

#ifdef __cplusplus
namespace bridge_cuda {
    // Fundamental Lattice QCD physical degrees of freedom
    constexpr int Nc       = 3; // Number of colors SU(3)
    constexpr int Ncolor   = 3; // Long alias for Nc
    constexpr int Nspin    = 4; // Number of Dirac spin components (4-spinor)
    constexpr int Ns       = 4; // Short alias for Nspin
    constexpr int Ndim     = 4; // Spacetime dimensions (4D Euclidean spacetime)
    constexpr int Nd       = 4; // Short alias for Ndim
    constexpr int Ncomplex = 2; // Real & Imaginary components (2)

    // Derived degrees of freedom
    constexpr int Nspin2   = 2; // Half-spinor (2-spinor, used for chiral projections)
    constexpr int Nvc      = Ncomplex * Nc;               // 6: Real elements per color vector
    constexpr int Nspinor  = Ncomplex * Nc * Nspin;       // 24: Real elements per Dirac 4-spinor
    constexpr int Ndf      = Ncomplex * Nc * Nc;          // 18: Real elements per SU(3) gauge matrix
    constexpr int Nclover  = 2 * (Nc * Nspin / 2) * (Nc * Nspin / 2); // 72: Real elements for clover hermitian blocks

    // Architecture & Memory Layout constants
    constexpr int WarpSize = 32;                          // AoSoA lane width W=32 for coalesced GPU access
    constexpr int DefaultBlockSize = 256;                 // Standard CUDA thread block size
}
#endif

// C macros for C-ABI and pure C compatibility
#define BRIDGE_CUDA_NC          3
#define BRIDGE_CUDA_NCOLOR      3
#define BRIDGE_CUDA_NSPIN       4
#define BRIDGE_CUDA_NS          4
#define BRIDGE_CUDA_NDIM        4
#define BRIDGE_CUDA_ND          4
#define BRIDGE_CUDA_NCOMPLEX    2

#define BRIDGE_CUDA_NVC         (BRIDGE_CUDA_NCOMPLEX * BRIDGE_CUDA_NC)
#define BRIDGE_CUDA_NSP         (BRIDGE_CUDA_NCOMPLEX * BRIDGE_CUDA_NC * BRIDGE_CUDA_NSPIN)
#define BRIDGE_CUDA_NSPINOR     (BRIDGE_CUDA_NCOMPLEX * BRIDGE_CUDA_NC * BRIDGE_CUDA_NSPIN)
#define BRIDGE_CUDA_NDF         (BRIDGE_CUDA_NCOMPLEX * BRIDGE_CUDA_NC * BRIDGE_CUDA_NC)
#define BRIDGE_CUDA_WARP_SIZE   32

#endif // BRIDGE_CUDA_BASE_CONSTANT_H
