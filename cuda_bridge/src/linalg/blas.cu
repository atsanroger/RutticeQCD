#include "../../include/bridge_cuda.h"
#include "../../include/linalg/blas.cuh"
#include "../../include/linalg/cast.cuh"
#include "../../include/base/gamma.cuh"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;

    constexpr int MAX_GRID = 1024;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, MAX_GRID);
    }

    // Device memory for the sums of the CUDA blocks of a reduction, kept between calls.
    // One thread uses the device at a time.
    template <typename Float>
    Float* block_sums() {
        static Float* sums = nullptr;
        if (sums == nullptr) cudaMalloc(&sums, 2 * MAX_GRID * sizeof(Float));
        return sums;
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

void bridge_cuda_blas_caxpy_f32(float* y, const float* x, float ar, float ai, int n, cudaStream_t stream) {
    bridge_cuda::blas_caxpy_kernel<float><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(y, x, ar, ai, n / 2);
}

void bridge_cuda_blas_caypx_f32(float* y, const float* x, float ar, float ai, int n, cudaStream_t stream) {
    bridge_cuda::blas_caypx_kernel<float><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(y, x, ar, ai, n / 2);
}

double bridge_cuda_blas_norm2_f32(const float* x, int n, cudaStream_t stream) {
    int grid = get_grid_size(n);
    float* sums = block_sums<float>();
    bridge_cuda::blas_norm2_partial_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(sums, x, n);
    float host[MAX_GRID];
    cudaMemcpy(host, sums, grid * sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (int b = 0; b < grid; ++b) total += host[b];
    return total;
}

void bridge_cuda_blas_dotc_f32(double* re, double* im, const float* x, const float* y, int n, cudaStream_t stream) {
    int grid = get_grid_size(n / 2);
    float* sums = block_sums<float>();
    bridge_cuda::blas_dotc_partial_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(sums, x, y, n / 2);
    float host[2 * MAX_GRID];
    cudaMemcpy(host, sums, 2 * grid * sizeof(float), cudaMemcpyDeviceToHost);
    double tr = 0.0, ti = 0.0;
    for (int b = 0; b < grid; ++b) {
        tr += host[b];
        ti += host[grid + b];
    }
    *re = tr;
    *im = ti;
}

void bridge_cuda_blas_caxpy_f64(double* y, const double* x, double ar, double ai, int n, cudaStream_t stream) {
    bridge_cuda::blas_caxpy_kernel<double><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(y, x, ar, ai, n / 2);
}

void bridge_cuda_blas_caypx_f64(double* y, const double* x, double ar, double ai, int n, cudaStream_t stream) {
    bridge_cuda::blas_caypx_kernel<double><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(y, x, ar, ai, n / 2);
}

double bridge_cuda_blas_norm2_f64(const double* x, int n, cudaStream_t stream) {
    int grid = get_grid_size(n);
    double* sums = block_sums<double>();
    bridge_cuda::blas_norm2_partial_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(sums, x, n);
    double host[MAX_GRID];
    cudaMemcpy(host, sums, grid * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0.0;
    for (int b = 0; b < grid; ++b) total += host[b];
    return total;
}

void bridge_cuda_blas_dotc_f64(double* re, double* im, const double* x, const double* y, int n, cudaStream_t stream) {
    int grid = get_grid_size(n / 2);
    double* sums = block_sums<double>();
    bridge_cuda::blas_dotc_partial_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(sums, x, y, n / 2);
    double host[2 * MAX_GRID];
    cudaMemcpy(host, sums, 2 * grid * sizeof(double), cudaMemcpyDeviceToHost);
    double tr = 0.0, ti = 0.0;
    for (int b = 0; b < grid; ++b) {
        tr += host[b];
        ti += host[grid + b];
    }
    *re = tr;
    *im = ti;
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
