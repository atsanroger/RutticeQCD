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

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_LINALG_BLAS_CUH
