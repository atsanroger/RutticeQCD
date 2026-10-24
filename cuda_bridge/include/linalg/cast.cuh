#ifndef BRIDGE_CUDA_LINALG_CAST_CUH
#define BRIDGE_CUDA_LINALG_CAST_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include <cuda_runtime.h>

namespace bridge_cuda {

__global__ void cast_f64_to_f32_kernel(float* __restrict__ dst, const double* __restrict__ src, int total_elements) {
    for (size_t idx : GridStride(total_elements)) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

__global__ void cast_f32_to_f64_kernel(double* __restrict__ dst, const float* __restrict__ src, int total_elements) {
    for (size_t idx : GridStride(total_elements)) {
        dst[idx] = static_cast<double>(src[idx]);
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_LINALG_CAST_CUH
