#ifndef BRIDGE_CUDA_BASE_GAMMA_CUH
#define BRIDGE_CUDA_BASE_GAMMA_CUH

#include "types.h"
#include "aosoa.cuh"
#include <cuda_runtime.h>

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Spin projection operators for Dirac and Chiral (Weyl) representations
// -------------------------------------------------------------------------

// Chiral projection: P_R = (1 + gamma_5)/2
template <typename Float, int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL>
__device__ __forceinline__ void proj_pr(
    Float out[Nspin][Nc][Ncomplex],
    const Float in[Nspin][Nc][Ncomplex])
{
    if constexpr (GammaRep == BRIDGE_CUDA_GAMMA_CHIRAL) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[0][c][0] = in[0][c][0]; out[0][c][1] = in[0][c][1];
            out[1][c][0] = in[1][c][0]; out[1][c][1] = in[1][c][1];
            out[2][c][0] = 0;           out[2][c][1] = 0;
            out[3][c][0] = 0;           out[3][c][1] = 0;
        }
    } else {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            Float sum0_r = static_cast<Float>(0.5) * (in[0][c][0] + in[2][c][0]);
            Float sum0_i = static_cast<Float>(0.5) * (in[0][c][1] + in[2][c][1]);
            Float sum1_r = static_cast<Float>(0.5) * (in[1][c][0] + in[3][c][0]);
            Float sum1_i = static_cast<Float>(0.5) * (in[1][c][1] + in[3][c][1]);

            out[0][c][0] = sum0_r; out[0][c][1] = sum0_i;
            out[1][c][0] = sum1_r; out[1][c][1] = sum1_i;
            out[2][c][0] = sum0_r; out[2][c][1] = sum0_i;
            out[3][c][0] = sum1_r; out[3][c][1] = sum1_i;
        }
    }
}

// Chiral projection: P_L = (1 - gamma_5)/2
template <typename Float, int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL>
__device__ __forceinline__ void proj_pl(
    Float out[Nspin][Nc][Ncomplex],
    const Float in[Nspin][Nc][Ncomplex])
{
    if constexpr (GammaRep == BRIDGE_CUDA_GAMMA_CHIRAL) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[0][c][0] = 0;           out[0][c][1] = 0;
            out[1][c][0] = 0;           out[1][c][1] = 0;
            out[2][c][0] = in[2][c][0]; out[2][c][1] = in[2][c][1];
            out[3][c][0] = in[3][c][0]; out[3][c][1] = in[3][c][1];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            Float diff0_r = static_cast<Float>(0.5) * (in[0][c][0] - in[2][c][0]);
            Float diff0_i = static_cast<Float>(0.5) * (in[0][c][1] - in[2][c][1]);
            Float diff1_r = static_cast<Float>(0.5) * (in[1][c][0] - in[3][c][0]);
            Float diff1_i = static_cast<Float>(0.5) * (in[1][c][1] - in[3][c][1]);

            out[0][c][0] = diff0_r;  out[0][c][1] = diff0_i;
            out[1][c][0] = diff1_r;  out[1][c][1] = diff1_i;
            out[2][c][0] = -diff0_r; out[2][c][1] = -diff0_i;
            out[3][c][0] = -diff1_r; out[3][c][1] = -diff1_i;
        }
    }
}

// Gamma_5 operator: psi' = gamma_5 * psi
template <typename Float, int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL>
__device__ __forceinline__ void mult_gamma5(
    Float out[Nspin][Nc][Ncomplex],
    const Float in[Nspin][Nc][Ncomplex])
{
    if constexpr (GammaRep == BRIDGE_CUDA_GAMMA_CHIRAL) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[0][c][0] = in[0][c][0];  out[0][c][1] = in[0][c][1];
            out[1][c][0] = in[1][c][0];  out[1][c][1] = in[1][c][1];
            out[2][c][0] = -in[2][c][0]; out[2][c][1] = -in[2][c][1];
            out[3][c][0] = -in[3][c][0]; out[3][c][1] = -in[3][c][1];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[0][c][0] = in[2][c][0]; out[0][c][1] = in[2][c][1];
            out[1][c][0] = in[3][c][0]; out[1][c][1] = in[3][c][1];
            out[2][c][0] = in[0][c][0]; out[2][c][1] = in[0][c][1];
            out[3][c][0] = in[1][c][0]; out[3][c][1] = in[1][c][1];
        }
    }
}

// Unitary transformation between Dirac-Pauli and Chiral (Weyl) representation
// U = 1/sqrt(2) * [[I, I], [I, -I]] = U^dag = U^{-1}
template <typename Float>
__device__ __forceinline__ void convert_dirac_chiral_spinor(
    Float out[Nspin][Nc][Ncomplex],
    const Float in[Nspin][Nc][Ncomplex])
{
    const Float inv_sqrt2 = static_cast<Float>(0.70710678118654752440);
    #pragma unroll
    for (int c = 0; c < Nc; ++c) {
        #pragma unroll
        for (int ri = 0; ri < 2; ++ri) {
            out[0][c][ri] = inv_sqrt2 * (in[0][c][ri] + in[2][c][ri]);
            out[1][c][ri] = inv_sqrt2 * (in[1][c][ri] + in[3][c][ri]);
            out[2][c][ri] = inv_sqrt2 * (in[0][c][ri] - in[2][c][ri]);
            out[3][c][ri] = inv_sqrt2 * (in[1][c][ri] - in[3][c][ri]);
        }
    }
}

// Kernel for whole-field basis transformation
template <typename Float>
__global__ void convert_dirac_chiral_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int total_sites)
{
    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < total_sites; site += blockDim.x * gridDim.x) {
        Float sp_in[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                sp_in[s][c][0] = d_in[aosoa_spinor_idx(c, s, 0, site)];
                sp_in[s][c][1] = d_in[aosoa_spinor_idx(c, s, 1, site)];
            }
        }
        Float sp_out[Nspin][Nc][Ncomplex];
        convert_dirac_chiral_spinor<Float>(sp_out, sp_in);
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                d_out[aosoa_spinor_idx(c, s, 0, site)] = sp_out[s][c][0];
                d_out[aosoa_spinor_idx(c, s, 1, site)] = sp_out[s][c][1];
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_GAMMA_CUH
