#ifndef BRIDGE_CUDA_BASE_GAMMA_CUH
#define BRIDGE_CUDA_BASE_GAMMA_CUH

#include "types.h"
#include "aosoa.cuh"
#include <cuda_runtime.h>

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Spin projection operators for Dirac and Chiral (Weyl) representations
// -------------------------------------------------------------------------

// gamma_mu in the Dirac representation of Ruttice and Bridge++ (gamma_4 diagonal), the one
// table every kernel reads: row `row` of gamma_mu has one entry, in column
// dirac_col(mu, row), of value dirac_re(mu, row) + i dirac_im(mu, row) (+-1 or +-i). As
// kernel::gamma's Dirac on the CPU.
__host__ __device__ constexpr int dirac_col(int mu, int row) {
    constexpr int col[Ndim][Nspin] = {{3, 2, 1, 0}, {3, 2, 1, 0}, {2, 3, 0, 1}, {0, 1, 2, 3}};
    return col[mu][row];
}
__host__ __device__ constexpr int dirac_re(int mu, int row) {
    constexpr int re[Ndim][Nspin] = {{0, 0, 0, 0}, {-1, 1, 1, -1}, {0, 0, 0, 0}, {1, 1, -1, -1}};
    return re[mu][row];
}
__host__ __device__ constexpr int dirac_im(int mu, int row) {
    constexpr int im[Ndim][Nspin] = {{-1, -1, 1, 1}, {0, 0, 0, 0}, {-1, 1, 1, -1}, {0, 0, 0, 0}};
    return im[mu][row];
}
// The table checked where it is written: gamma_4 diagonal, gamma_1 = -i on its first row.
static_assert(dirac_col(3, 0) == 0 && dirac_re(3, 2) == -1 && dirac_im(0, 0) == -1);

// out = gamma_mu in, in the Dirac representation:
//   (gamma_mu psi)[row] = val[row] * psi[col[row]],  val one of +1, -1, +i, -i.
template <typename Float>
__device__ __forceinline__ void mult_gamma_dirac(
    Float out[Nspin][Nc][Ncomplex],
    const Float in[Nspin][Nc][Ncomplex],
    int mu)
{
    #pragma unroll
    for (int row = 0; row < Nspin; ++row) {
        const int k = dirac_col(mu, row);
        const Float a = static_cast<Float>(dirac_re(mu, row));
        const Float b = static_cast<Float>(dirac_im(mu, row));
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
