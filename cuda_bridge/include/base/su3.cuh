#ifndef BRIDGE_CUDA_BASE_SU3_CUH
#define BRIDGE_CUDA_BASE_SU3_CUH

#include "types.h"
#include <cuda_runtime.h>

namespace bridge_cuda {

// Complex multiply: c = a * b
template <typename Float>
__device__ __forceinline__ void c_mult(Float &cr, Float &ci, Float ar, Float ai, Float br, Float bi) {
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

// Complex multiply-accumulate: c += a * b
template <typename Float>
__device__ __forceinline__ void c_macc(Float &cr, Float &ci, Float ar, Float ai, Float br, Float bi) {
    cr += ar * br - ai * bi;
    ci += ar * bi + ai * br;
}

// Complex conjugate-multiply-accumulate: c += conj(a) * b
template <typename Float>
__device__ __forceinline__ void c_conj_macc(Float &cr, Float &ci, Float ar, Float ai, Float br, Float bi) {
    cr += ar * br + ai * bi;
    ci += ar * bi - ai * br;
}

// SU(3) matrix times color vector: out = U * in
template <typename Float>
__device__ __forceinline__ void su3_mult_vec(
    Float out_r[Nc], Float out_i[Nc],
    const Float u_r[Nc][Nc], const Float u_i[Nc][Nc],
    const Float in_r[Nc], const Float in_i[Nc])
{
    #pragma unroll
    for (int r = 0; r < Nc; ++r) {
        out_r[r] = 0;
        out_i[r] = 0;
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            c_macc(out_r[r], out_i[r], u_r[r][c], u_i[r][c], in_r[c], in_i[c]);
        }
    }
}

// SU(3) dagger matrix times color vector: out = U^dag * in
template <typename Float>
__device__ __forceinline__ void su3_dagger_mult_vec(
    Float out_r[Nc], Float out_i[Nc],
    const Float u_r[Nc][Nc], const Float u_i[Nc][Nc],
    const Float in_r[Nc], const Float in_i[Nc])
{
    #pragma unroll
    for (int r = 0; r < Nc; ++r) {
        out_r[r] = 0;
        out_i[r] = 0;
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            c_conj_macc(out_r[r], out_i[r], u_r[c][r], u_i[c][r], in_r[c], in_i[c]);
        }
    }
}

// SU(3) 3rd-row reconstruction (cross product of first two rows)
template <typename Float>
__device__ __forceinline__ void su3_reconstruct_row3(
    Float u_r[Nc][Nc], Float u_i[Nc][Nc])
{
    // row2 = conj(row0 x row1)
    Float r0_r = u_r[0][1]*u_r[1][2] - u_i[0][1]*u_i[1][2] - (u_r[0][2]*u_r[1][1] - u_i[0][2]*u_i[1][1]);
    Float r0_i = -(u_r[0][1]*u_i[1][2] + u_i[0][1]*u_r[1][2] - (u_r[0][2]*u_i[1][1] + u_i[0][2]*u_r[1][1]));

    Float r1_r = u_r[0][2]*u_r[1][0] - u_i[0][2]*u_i[1][0] - (u_r[0][0]*u_r[1][2] - u_i[0][0]*u_i[1][2]);
    Float r1_i = -(u_r[0][2]*u_i[1][0] + u_i[0][2]*u_r[1][0] - (u_r[0][0]*u_i[1][2] + u_i[0][0]*u_r[1][2]));

    Float r2_r = u_r[0][0]*u_r[1][1] - u_i[0][0]*u_i[1][1] - (u_r[0][1]*u_r[1][0] - u_i[0][1]*u_r[1][0]);
    Float r2_i = -(u_r[0][0]*u_i[1][1] + u_i[0][0]*u_r[1][1] - (u_r[0][1]*u_i[1][0] + u_i[0][1]*u_r[1][0]));

    u_r[2][0] = r0_r; u_i[2][0] = r0_i;
    u_r[2][1] = r1_r; u_i[2][1] = r1_i;
    u_r[2][2] = r2_r; u_i[2][2] = r2_i;
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_SU3_CUH
