#ifndef BRIDGE_CUDA_BASE_TYPES_H
#define BRIDGE_CUDA_BASE_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include "constant.h"

#ifdef __cplusplus
extern "C" {
#endif

// Precision enum for C-API dispatch
typedef enum {
    BRIDGE_CUDA_PRECISION_FLOAT = 0,
    BRIDGE_CUDA_PRECISION_DOUBLE = 1
} BridgeCudaPrecision;

// Gamma matrix representation - aligned with Ruttice GammaRepre
typedef enum {
    BRIDGE_CUDA_GAMMA_DIRAC = 0,   // Standard Dirac-Pauli basis (gamma_4 diagonal)
    BRIDGE_CUDA_GAMMA_CHIRAL = 1   // Chiral / Weyl basis (gamma_5 diagonal)
} BridgeCudaGammaRep;

// 4D / 5D Lattice geometry descriptor - aligned with Ruttice dims: [usize; 4]
typedef struct {
    int32_t dims[4];  // [nx, ny, nz, nt]
    int32_t Ls;       // 5th dimension size for Mobius Domain Wall (1 for 4D Wilson/Clover)
    int32_t bc[4];    // Boundary conditions: +1 for periodic, -1 for anti-periodic, 0 for open
    int32_t vol;      // Local 4D volume = dims[0] * dims[1] * dims[2] * dims[3]
    int32_t vol5d;    // Local 5D volume = vol * Ls
    int32_t vol_pad;  // Volume padded to multiple of BRIDGE_CUDA_WARP_SIZE (W=32)
    int32_t gamma_rep;// BridgeCudaGammaRep (Dirac or Chiral)
} BridgeCudaGeometry;

// Wilson operator parameters - aligned with Ruttice WilsonParams
typedef struct {
    double kappa;     // Hopping parameter
} BridgeCudaWilsonParams;

// Clover operator parameters - aligned with Ruttice CloverParams
typedef struct {
    double kappa;     // Hopping parameter
    double csw;       // Clover coefficient
} BridgeCudaCloverParams;

// Mobius Domain Wall (MDWF) operator parameters - aligned with Ruttice / MDWF standards
typedef struct {
    double M5;        // Domain wall height (typically 1.8)
    double mass;      // Bare quark mass mq
    double alpha;     // 5D bulk-vs-boundary scaling factor (Neff alpha)
    double b5;        // Mobius b parameter
    double c5;        // Mobius c parameter
} BridgeCudaMdwfParams;

// Multigrid coarse grid descriptor
typedef struct {
    int32_t num_colors;    // Number of coarse degrees of freedom per site
    int32_t block_size[4]; // Aggregation block dimensions [bx, by, bz, bt]
} BridgeCudaMgParams;

// Gauge smearing parameters
typedef struct {
    double rho;       // Stout smearing weight parameter (typically ~0.1)
    double alpha_ape; // APE smearing weight parameter
    int32_t steps;    // Number of smearing iterations
} BridgeCudaSmearParams;

// Domain Decomposition (SAP / Block DD) parameters for Multigrid smoother
typedef struct {
    int32_t block_size[4]; // [bx, by, bz, bt] aggregate block dimensions
    int32_t ieo;           // Block parity: 0 for even blocks, 1 for odd blocks
} BridgeCudaDdParams;

// =========================================================================
// Backward Compatibility Aliases
// =========================================================================
#define GPU_FER_NC         BRIDGE_CUDA_NC
#define GPU_FER_ND         BRIDGE_CUDA_ND
#define GPU_FER_NVC        BRIDGE_CUDA_NVC
#define GPU_FER_NSP        BRIDGE_CUDA_NSP
#define GPU_FER_NDF        BRIDGE_CUDA_NDF
#define GPU_FER_WARP_SIZE  BRIDGE_CUDA_WARP_SIZE

typedef BridgeCudaPrecision    GpuFerPrecision;
typedef BridgeCudaGammaRep     GpuFerGammaRep;
typedef BridgeCudaGeometry     GpuFerGeometry;
typedef BridgeCudaWilsonParams GpuFerWilsonParams;
typedef BridgeCudaCloverParams GpuFerCloverParams;
typedef BridgeCudaMdwfParams   GpuFerMdwfParams;
typedef BridgeCudaMdwfParams   GpuFerDwfParams;
typedef BridgeCudaMgParams     GpuFerMgParams;
typedef BridgeCudaSmearParams  GpuFerSmearParams;
typedef BridgeCudaDdParams     GpuFerDdParams;

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_CUDA_BASE_TYPES_H
