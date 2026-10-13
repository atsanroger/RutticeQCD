#include "../../include/bridge_cuda.h"
#include "../../include/linalg/blas.cuh"
#include "../../include/linalg/cast.cuh"
#include "../../include/base/gamma.cuh"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, 1024);
    }
}

extern "C" {

void bridge_cuda_blas_axpy_f32(float* y, const float* x, float a, int n, cudaStream_t stream) {
    int grid = get_grid_size(n);
    bridge_cuda::blas_axpy_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(y, x, a, n);
}

void bridge_cuda_blas_axpy_f64(double* y, const double* x, double a, int n, cudaStream_t stream) {
    int grid = get_grid_size(n);
    bridge_cuda::blas_axpy_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(y, x, a, n);
}

void bridge_cuda_blas_scal_f32(float* x, float a, int n, cudaStream_t stream) {
    int grid = get_grid_size(n);
    bridge_cuda::blas_scal_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(x, a, n);
}

void bridge_cuda_blas_scal_f64(double* x, double a, int n, cudaStream_t stream) {
    int grid = get_grid_size(n);
    bridge_cuda::blas_scal_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(x, a, n);
}

void bridge_cuda_blas_norm2_partial_f32(float* block_sums, const float* x, int n, int grid_size, cudaStream_t stream) {
    bridge_cuda::blas_norm2_partial_kernel<float><<<grid_size, BLOCK_SIZE, 0, stream>>>(block_sums, x, n);
}

void bridge_cuda_blas_norm2_partial_f64(double* block_sums, const double* x, int n, int grid_size, cudaStream_t stream) {
    bridge_cuda::blas_norm2_partial_kernel<double><<<grid_size, BLOCK_SIZE, 0, stream>>>(block_sums, x, n);
}

void bridge_cuda_cast_f64_to_f32(float* dst, const double* src, int total_elements, cudaStream_t stream) {
    int grid = get_grid_size(total_elements);
    bridge_cuda::cast_f64_to_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(dst, src, total_elements);
}

void bridge_cuda_cast_f32_to_f64(double* dst, const float* src, int total_elements, cudaStream_t stream) {
    int grid = get_grid_size(total_elements);
    bridge_cuda::cast_f32_to_f64_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(dst, src, total_elements);
}

void bridge_cuda_convert_dirac_chiral_f32(float* d_out, const float* d_in, int total_sites, cudaStream_t stream) {
    int grid = get_grid_size(total_sites);
    bridge_cuda::convert_dirac_chiral_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(d_out, d_in, total_sites);
}

void bridge_cuda_convert_dirac_chiral_f64(double* d_out, const double* d_in, int total_sites, cudaStream_t stream) {
    int grid = get_grid_size(total_sites);
    bridge_cuda::convert_dirac_chiral_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_out, d_in, total_sites);
}

} // extern "C"
