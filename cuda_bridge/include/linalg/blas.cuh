#ifndef BRIDGE_CUDA_LINALG_BLAS_CUH
#define BRIDGE_CUDA_LINALG_BLAS_CUH

#include "../base/types.h"
#include <cuda_runtime.h>

namespace bridge_cuda {

template <typename Float>
__global__ void blas_axpy_kernel(Float* __restrict__ y, const Float* __restrict__ x, Float a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] += a * x[i];
    }
}

template <typename Float>
__global__ void blas_scal_kernel(Float* __restrict__ x, Float a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] *= a;
    }
}

template <typename Float, int BLOCK_SIZE = 256>
__global__ void blas_norm2_partial_kernel(Float* __restrict__ block_sums, const Float* __restrict__ x, int n) {
    __shared__ Float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    Float sum = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += x[i] * x[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// The kernels below take a field of complex numbers in blocks of W: the real parts of W
// numbers, then their imaginary parts (every field of base/aosoa.cuh is such). `pairs` is
// the number of complex numbers.
__device__ __forceinline__ int blas_real_index(int pair) {
    return (pair / BRIDGE_CUDA_WARP_SIZE) * 2 * BRIDGE_CUDA_WARP_SIZE + pair % BRIDGE_CUDA_WARP_SIZE;
}

// y += a x, a complex
template <typename Float>
__global__ void blas_caxpy_kernel(Float* __restrict__ y, const Float* __restrict__ x, Float ar, Float ai, int pairs) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
        const int kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        const Float xr = x[kr], xi = x[ki];
        y[kr] += ar * xr - ai * xi;
        y[ki] += ar * xi + ai * xr;
    }
}

// y = a y + x, a complex
template <typename Float>
__global__ void blas_caypx_kernel(Float* __restrict__ y, const Float* __restrict__ x, Float ar, Float ai, int pairs) {
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
        const int kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        const Float yr = y[kr], yi = y[ki];
        y[kr] = ar * yr - ai * yi + x[kr];
        y[ki] = ar * yi + ai * yr + x[ki];
    }
}

// <x|y> = sum conj(x) y: the real part of the sum of CUDA block b in block_sums[b], the
// imaginary part in block_sums[gridDim.x + b].
template <typename Float, int BLOCK_SIZE = 256>
__global__ void blas_dotc_partial_kernel(Float* __restrict__ block_sums, const Float* __restrict__ x, const Float* __restrict__ y, int pairs) {
    __shared__ Float sr[BLOCK_SIZE];
    __shared__ Float si[BLOCK_SIZE];
    int tid = threadIdx.x;
    Float re = 0, im = 0;

    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
        const int kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        re += x[kr] * y[kr] + x[ki] * y[ki];
        im += x[kr] * y[ki] - x[ki] * y[kr];
    }
    sr[tid] = re;
    si[tid] = im;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sr[tid] += sr[tid + s];
            si[tid] += si[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sr[0];
        block_sums[gridDim.x + blockIdx.x] = si[0];
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_LINALG_BLAS_CUH
