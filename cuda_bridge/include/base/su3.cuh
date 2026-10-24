#ifndef BRIDGE_CUDA_BASE_SU3_CUH
#define BRIDGE_CUDA_BASE_SU3_CUH

#include "types.h"
#include <cuda_runtime.h>

namespace bridge_cuda {

// A product rounded on its own: nvcc never contracts it into an fma. With -fmad (the default)
// the compiler chooses per kernel which a * b + c become fused, so the same sum could round
// differently in two kernels; the sums below fix the choice, so the kernels of several
// processes (interior, surface, the packing of the faces) agree with one process bit for bit.
__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double mul_rn(double a, double b) { return __dmul_rn(a, b); }

// Complex multiply-accumulate c += a b, four fused multiply-adds in a fixed order.
template <typename Float>
__device__ __forceinline__ void c_macc(Float &cr, Float &ci, Float ar, Float ai, Float br, Float bi) {
    cr = fma(ar, br, cr);
    cr = fma(-ai, bi, cr);
    ci = fma(ar, bi, ci);
    ci = fma(ai, br, ci);
}

// Complex conjugate-multiply-accumulate c += conj(a) b, likewise.
template <typename Float>
__device__ __forceinline__ void c_conj_macc(Float &cr, Float &ci, Float ar, Float ai, Float br, Float bi) {
    cr = fma(ar, br, cr);
    cr = fma(ai, bi, cr);
    ci = fma(ar, bi, ci);
    ci = fma(-ai, br, ci);
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

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_SU3_CUH
