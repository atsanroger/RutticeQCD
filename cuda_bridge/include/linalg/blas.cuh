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

// The fields below hold `vectors` vectors of n reals one after the other, with a coefficient
// for each: the coefficients are an argument of the kernel, which takes up to BlasVectors
// vectors.
constexpr int BlasVectors = 16;

template <typename Float>
struct BlasCoefficients {
    Float re[BlasVectors];
    Float im[BlasVectors];
};

// x_v *= a_v, a real
template <typename Float>
__global__ void blas_scal_kernel(Float* __restrict__ x, BlasCoefficients<Float> a, int n, int vectors) {
    const size_t all = static_cast<size_t>(n) * vectors;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < all; i += blockDim.x * gridDim.x) {
        x[i] *= a.re[i / n];
    }
}

// |x_v|^2: gridDim.x / vectors CUDA blocks for each vector, the sum of CUDA block b in
// block_sums[b].
template <typename Float, int BLOCK_SIZE = 256>
__global__ void blas_norm2_partial_kernel(Float* __restrict__ block_sums, const Float* __restrict__ x, int n, int vectors) {
    __shared__ Float sdata[BLOCK_SIZE];
    const int tid = threadIdx.x, per = gridDim.x / vectors;
    x += static_cast<size_t>(blockIdx.x / per) * n;
    Float sum = 0;

    for (int i = (blockIdx.x % per) * blockDim.x + tid; i < n; i += blockDim.x * per) {
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

// The kernels below take fields of complex numbers in blocks of W: the real parts of W
// numbers, then their imaginary parts (every field of base/aosoa.cuh is such). `pairs` is
// the number of complex numbers of a vector, a multiple of W.
__device__ __forceinline__ size_t blas_real_index(size_t pair) {
    return (pair / BRIDGE_CUDA_WARP_SIZE) * 2 * BRIDGE_CUDA_WARP_SIZE + pair % BRIDGE_CUDA_WARP_SIZE;
}

// y_v += a_v x_v, a complex
template <typename Float>
__global__ void blas_caxpy_kernel(Float* __restrict__ y, const Float* __restrict__ x, BlasCoefficients<Float> a, int pairs, int vectors) {
    const size_t all = static_cast<size_t>(pairs) * vectors;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < all; p += blockDim.x * gridDim.x) {
        const size_t kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        const Float ar = a.re[p / pairs], ai = a.im[p / pairs];
        const Float xr = x[kr], xi = x[ki];
        y[kr] += ar * xr - ai * xi;
        y[ki] += ar * xi + ai * xr;
    }
}

// y_v = a_v y_v + x_v, a complex
template <typename Float>
__global__ void blas_caypx_kernel(Float* __restrict__ y, const Float* __restrict__ x, BlasCoefficients<Float> a, int pairs, int vectors) {
    const size_t all = static_cast<size_t>(pairs) * vectors;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < all; p += blockDim.x * gridDim.x) {
        const size_t kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        const Float ar = a.re[p / pairs], ai = a.im[p / pairs];
        const Float yr = y[kr], yi = y[ki];
        y[kr] = ar * yr - ai * yi + x[kr];
        y[ki] = ar * yi + ai * yr + x[ki];
    }
}

// <x_v|y_v> = sum conj(x_v) y_v: gridDim.x / vectors CUDA blocks for each vector, the real part
// of the sum of CUDA block b in block_sums[b], the imaginary part in block_sums[gridDim.x + b].
template <typename Float, int BLOCK_SIZE = 256>
__global__ void blas_dotc_partial_kernel(Float* __restrict__ block_sums, const Float* __restrict__ x, const Float* __restrict__ y, int pairs, int vectors) {
    __shared__ Float sr[BLOCK_SIZE];
    __shared__ Float si[BLOCK_SIZE];
    const int tid = threadIdx.x, per = gridDim.x / vectors;
    x += static_cast<size_t>(blockIdx.x / per) * 2 * pairs;
    y += static_cast<size_t>(blockIdx.x / per) * 2 * pairs;
    Float re = 0, im = 0;

    for (int p = (blockIdx.x % per) * blockDim.x + tid; p < pairs; p += blockDim.x * per) {
        const size_t kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
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

// x_j <- sum_i x_i c[i n + j] for the n vectors of a set, in place: a thread for each complex
// number, which reads it of every vector, then writes it. c: n^2 complex numbers (re, im),
// row-major. n at most RotateVectors.
constexpr int RotateVectors = 32;

template <typename Float>
__global__ void blas_rotate_kernel(Float* __restrict__ x, const double* __restrict__ c, int pairs, int n) {
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < static_cast<size_t>(pairs);
         p += blockDim.x * gridDim.x) {
        const size_t kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        Float re[RotateVectors], im[RotateVectors];
        for (int i = 0; i < n; ++i) {
            re[i] = x[static_cast<size_t>(i) * 2 * pairs + kr];
            im[i] = x[static_cast<size_t>(i) * 2 * pairs + ki];
        }
        for (int j = 0; j < n; ++j) {
            Float zr = 0, zi = 0;
            for (int i = 0; i < n; ++i) {
                const Float cr = static_cast<Float>(c[2 * (i * n + j)]), ci = static_cast<Float>(c[2 * (i * n + j) + 1]);
                zr += re[i] * cr - im[i] * ci;
                zi += re[i] * ci + im[i] * cr;
            }
            x[static_cast<size_t>(j) * 2 * pairs + kr] = zr;
            x[static_cast<size_t>(j) * 2 * pairs + ki] = zi;
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_LINALG_BLAS_CUH
