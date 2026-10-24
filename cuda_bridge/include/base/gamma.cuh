#ifndef BRIDGE_CUDA_BASE_GAMMA_CUH
#define BRIDGE_CUDA_BASE_GAMMA_CUH

#include "types.h"
#include "aosoa.cuh"
#include <cuda_runtime.h>

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Spin projection operators for Dirac and Chiral (Weyl) representations
// -------------------------------------------------------------------------

// out = gamma_mu in, in the Dirac representation of Ruttice and Bridge++ (gamma_4 diagonal):
//   (gamma_mu psi)[row] = val[row] * psi[col[row]],  val one of +1, -1, +i, -i.
template <typename Float>
__device__ __forceinline__ void mult_gamma_dirac(
    Float out[Nspin][Nc][Ncomplex],
    const Float in[Nspin][Nc][Ncomplex],
    int mu)
{
    // col[mu][row]; val as (re, im)
    const int col[4][4] = {{3, 2, 1, 0}, {3, 2, 1, 0}, {2, 3, 0, 1}, {0, 1, 2, 3}};
    const int re[4][4]  = {{0, 0, 0, 0}, {-1, 1, 1, -1}, {0, 0, 0, 0}, {1, 1, -1, -1}};
    const int im[4][4]  = {{-1, -1, 1, 1}, {0, 0, 0, 0}, {-1, 1, 1, -1}, {0, 0, 0, 0}};
    #pragma unroll
    for (int row = 0; row < Nspin; ++row) {
        const int k = col[mu][row];
        const Float a = static_cast<Float>(re[mu][row]);
        const Float b = static_cast<Float>(im[mu][row]);
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[row][c][0] = a * in[k][c][0] - b * in[k][c][1];
            out[row][c][1] = a * in[k][c][1] + b * in[k][c][0];
        }
    }
}

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


} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_GAMMA_CUH
