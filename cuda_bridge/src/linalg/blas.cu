#include "../../include/bridge_cuda.h"
#include "../../include/linalg/blas.cuh"
#include "../../include/linalg/cast.cuh"
#include "../../include/base/gamma.cuh"
#include <algorithm>
#include <vector>

namespace {
    constexpr int BLOCK_SIZE = 256;

    constexpr int MAX_GRID = 1024;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, MAX_GRID);
    }

    // For `vectors` vectors of n elements each.
    inline int get_grid_size(int n, int vectors) {
        size_t grid = (static_cast<size_t>(n) * vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return static_cast<int>(std::min<size_t>(grid, MAX_GRID));
    }

    // Device memory for the sums of the CUDA blocks of the reductions of `vectors` vectors,
    // kept between calls: one for each thread, so that threads may reduce at the same time.
    template <typename Float>
    Float* block_sums(int vectors) {
        static thread_local Float* sums = nullptr;
        static thread_local int held = 0;
        if (vectors > held) {
            if (sums != nullptr) cudaFree(sums);
            cudaMalloc(&sums, static_cast<size_t>(vectors) * 2 * MAX_GRID * sizeof(Float));
            held = vectors;
        }
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

void bridge_cuda_blas_norm2_partial_f32(float* block_sums, const float* x, int n, int grid_size, cudaStream_t stream) {
    bridge_cuda::blas_norm2_partial_kernel<float><<<grid_size, BLOCK_SIZE, 0, stream>>>(block_sums, x, n, 1);
}

void bridge_cuda_blas_norm2_partial_f64(double* block_sums, const double* x, int n, int grid_size, cudaStream_t stream) {
    bridge_cuda::blas_norm2_partial_kernel<double><<<grid_size, BLOCK_SIZE, 0, stream>>>(block_sums, x, n, 1);
}

void bridge_cuda_blas_norm2_f32(double* out, const float* x, int n, int vectors, cudaStream_t stream) {
    int grid = get_grid_size(n);
    float* sums = block_sums<float>(vectors);
    bridge_cuda::blas_norm2_partial_kernel<float><<<vectors * grid, BLOCK_SIZE, 0, stream>>>(sums, x, n, vectors);
    std::vector<float> host(static_cast<size_t>(vectors) * grid);
    cudaMemcpy(host.data(), sums, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int v = 0; v < vectors; ++v) {
        double total = 0.0;
        for (int b = 0; b < grid; ++b) total += host[static_cast<size_t>(v) * grid + b];
        out[v] = total;
    }
}

void bridge_cuda_blas_dotc_f32(double* re, double* im, const float* x, const float* y, int n, int vectors, cudaStream_t stream) {
    int grid = get_grid_size(n / 2);
    float* sums = block_sums<float>(vectors);
    bridge_cuda::blas_dotc_partial_kernel<float><<<vectors * grid, BLOCK_SIZE, 0, stream>>>(sums, x, y, n / 2, vectors);
    std::vector<float> host(static_cast<size_t>(vectors) * 2 * grid);
    cudaMemcpy(host.data(), sums, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int v = 0; v < vectors; ++v) {
        double tr = 0.0, ti = 0.0;
        for (int b = 0; b < grid; ++b) {
            tr += host[static_cast<size_t>(v) * grid + b];
            ti += host[static_cast<size_t>(vectors + v) * grid + b];
        }
        re[v] = tr;
        im[v] = ti;
    }
}

void bridge_cuda_blas_norm2_f64(double* out, const double* x, int n, int vectors, cudaStream_t stream) {
    int grid = get_grid_size(n);
    double* sums = block_sums<double>(vectors);
    bridge_cuda::blas_norm2_partial_kernel<double><<<vectors * grid, BLOCK_SIZE, 0, stream>>>(sums, x, n, vectors);
    std::vector<double> host(static_cast<size_t>(vectors) * grid);
    cudaMemcpy(host.data(), sums, host.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int v = 0; v < vectors; ++v) {
        double total = 0.0;
        for (int b = 0; b < grid; ++b) total += host[static_cast<size_t>(v) * grid + b];
        out[v] = total;
    }
}

void bridge_cuda_blas_dotc_f64(double* re, double* im, const double* x, const double* y, int n, int vectors, cudaStream_t stream) {
    int grid = get_grid_size(n / 2);
    double* sums = block_sums<double>(vectors);
    bridge_cuda::blas_dotc_partial_kernel<double><<<vectors * grid, BLOCK_SIZE, 0, stream>>>(sums, x, y, n / 2, vectors);
    std::vector<double> host(static_cast<size_t>(vectors) * 2 * grid);
    cudaMemcpy(host.data(), sums, host.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int v = 0; v < vectors; ++v) {
        double tr = 0.0, ti = 0.0;
        for (int b = 0; b < grid; ++b) {
            tr += host[static_cast<size_t>(v) * grid + b];
            ti += host[static_cast<size_t>(vectors + v) * grid + b];
        }
        re[v] = tr;
        im[v] = ti;
    }
}

void bridge_cuda_blas_scal_f32(float* x, const double* a, int n, int vectors, cudaStream_t stream) {
    for (int v = 0; v < vectors; v += bridge_cuda::BlasVectors) {
        const int count = std::min(bridge_cuda::BlasVectors, vectors - v);
        bridge_cuda::BlasCoefficients<float> c{};
        for (int k = 0; k < count; ++k) c.re[k] = static_cast<float>(a[v + k]);
        bridge_cuda::blas_scal_kernel<float><<<get_grid_size(n, count), BLOCK_SIZE, 0, stream>>>(
            x + static_cast<size_t>(v) * n, c, n, count);
    }
}

void bridge_cuda_blas_caxpy_f32(float* y, const float* x, const double* a, int n, int vectors, cudaStream_t stream) {
    for (int v = 0; v < vectors; v += bridge_cuda::BlasVectors) {
        const int count = std::min(bridge_cuda::BlasVectors, vectors - v);
        bridge_cuda::BlasCoefficients<float> c{};
        for (int k = 0; k < count; ++k) {
            c.re[k] = static_cast<float>(a[2 * (v + k)]);
            c.im[k] = static_cast<float>(a[2 * (v + k) + 1]);
        }
        bridge_cuda::blas_caxpy_kernel<float><<<get_grid_size(n / 2, count), BLOCK_SIZE, 0, stream>>>(
            y + static_cast<size_t>(v) * n, x + static_cast<size_t>(v) * n, c, n / 2, count);
    }
}

void bridge_cuda_blas_caypx_f32(float* y, const float* x, const double* a, int n, int vectors, cudaStream_t stream) {
    for (int v = 0; v < vectors; v += bridge_cuda::BlasVectors) {
        const int count = std::min(bridge_cuda::BlasVectors, vectors - v);
        bridge_cuda::BlasCoefficients<float> c{};
        for (int k = 0; k < count; ++k) {
            c.re[k] = static_cast<float>(a[2 * (v + k)]);
            c.im[k] = static_cast<float>(a[2 * (v + k) + 1]);
        }
        bridge_cuda::blas_caypx_kernel<float><<<get_grid_size(n / 2, count), BLOCK_SIZE, 0, stream>>>(
            y + static_cast<size_t>(v) * n, x + static_cast<size_t>(v) * n, c, n / 2, count);
    }
}

void bridge_cuda_blas_scal_f64(double* x, const double* a, int n, int vectors, cudaStream_t stream) {
    for (int v = 0; v < vectors; v += bridge_cuda::BlasVectors) {
        const int count = std::min(bridge_cuda::BlasVectors, vectors - v);
        bridge_cuda::BlasCoefficients<double> c{};
        for (int k = 0; k < count; ++k) c.re[k] = static_cast<double>(a[v + k]);
        bridge_cuda::blas_scal_kernel<double><<<get_grid_size(n, count), BLOCK_SIZE, 0, stream>>>(
            x + static_cast<size_t>(v) * n, c, n, count);
    }
}

void bridge_cuda_blas_caxpy_f64(double* y, const double* x, const double* a, int n, int vectors, cudaStream_t stream) {
    for (int v = 0; v < vectors; v += bridge_cuda::BlasVectors) {
        const int count = std::min(bridge_cuda::BlasVectors, vectors - v);
        bridge_cuda::BlasCoefficients<double> c{};
        for (int k = 0; k < count; ++k) {
            c.re[k] = static_cast<double>(a[2 * (v + k)]);
            c.im[k] = static_cast<double>(a[2 * (v + k) + 1]);
        }
        bridge_cuda::blas_caxpy_kernel<double><<<get_grid_size(n / 2, count), BLOCK_SIZE, 0, stream>>>(
            y + static_cast<size_t>(v) * n, x + static_cast<size_t>(v) * n, c, n / 2, count);
    }
}

void bridge_cuda_blas_caypx_f64(double* y, const double* x, const double* a, int n, int vectors, cudaStream_t stream) {
    for (int v = 0; v < vectors; v += bridge_cuda::BlasVectors) {
        const int count = std::min(bridge_cuda::BlasVectors, vectors - v);
        bridge_cuda::BlasCoefficients<double> c{};
        for (int k = 0; k < count; ++k) {
            c.re[k] = static_cast<double>(a[2 * (v + k)]);
            c.im[k] = static_cast<double>(a[2 * (v + k) + 1]);
        }
        bridge_cuda::blas_caypx_kernel<double><<<get_grid_size(n / 2, count), BLOCK_SIZE, 0, stream>>>(
            y + static_cast<size_t>(v) * n, x + static_cast<size_t>(v) * n, c, n / 2, count);
    }
}

void bridge_cuda_blas_rotate_f32(float* x, const double* c, int n, int vectors, cudaStream_t stream) {
    bridge_cuda::blas_rotate_kernel<float><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(x, c, n / 2, vectors);
}

void bridge_cuda_blas_rotate_f64(double* x, const double* c, int n, int vectors, cudaStream_t stream) {
    bridge_cuda::blas_rotate_kernel<double><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(x, c, n / 2, vectors);
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
