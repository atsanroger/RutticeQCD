#ifndef BRIDGE_CUDA_BASE_ALLOC_CUH
#define BRIDGE_CUDA_BASE_ALLOC_CUH

#include <cuda_runtime.h>

namespace bridge_cuda {

// cudaMalloc for the scratch of the library itself. The fields of the Rust side come from the
// stream-ordered pool of the device, which keeps what is freed for the next allocation
// (src/backend/cuda_bridge/mod.rs, `pool`): if the device refuses, what the pool keeps unused
// is given back and the allocation tried once more.
template <typename T>
cudaError_t malloc_or_trim(T** ptr, size_t bytes)
{
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e != cudaErrorMemoryAllocation) return e;
    cudaGetLastError();  // the refusal is not an error of the kernels
    int device = 0;
    cudaMemPool_t pool;
    if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) {
        cudaGetLastError();
        return e;
    }
    cudaDeviceSynchronize();
    cudaMemPoolTrimTo(pool, 0);
    return cudaMalloc(ptr, bytes);
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_ALLOC_CUH
