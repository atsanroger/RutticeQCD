#include "../../include/bridge_cuda.h"
#include "../../include/linalg/blas.cuh"
#include "../../include/linalg/cast.cuh"
#include "../../include/base/gamma.cuh"
#include "../../include/base/alloc.cuh"
#include <algorithm>

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

    // Device memory for the sums of the CUDA blocks of a reduction and for its results, and
    // page-locked host memory the results are copied to, kept between calls and grown as
    // needed: one set for each thread, so that threads may reduce at the same time.
    struct Reduction {
        double* sums = nullptr;
        size_t sums_held = 0;
        double* results = nullptr;
        double* host = nullptr;
        size_t results_held = 0;
    };
    thread_local Reduction reduction;

    // Room for `sums` and `results` doubles; a cudaError_t.
    int reserve(size_t sums, size_t results) {
        Reduction& r = reduction;
        if (sums > r.sums_held) {
            if (r.sums != nullptr) cudaFree(r.sums);
            r.sums = nullptr;
            r.sums_held = 0;
            const cudaError_t e = bridge_cuda::malloc_or_trim(&r.sums, sums * sizeof(double));
            if (e != cudaSuccess) {
                r.sums = nullptr;
                return e;
            }
            r.sums_held = sums;
        }
        if (results > r.results_held) {
            if (r.results != nullptr) cudaFree(r.results);
            if (r.host != nullptr) cudaFreeHost(r.host);
            r.results = r.host = nullptr;
            r.results_held = 0;
            cudaError_t e = bridge_cuda::malloc_or_trim(&r.results, results * sizeof(double));
            if (e == cudaSuccess) e = cudaMallocHost(&r.host, results * sizeof(double));
            if (e != cudaSuccess) {
                if (r.results != nullptr) cudaFree(r.results);
                r.results = r.host = nullptr;
                return e;
            }
            r.results_held = results;
        }
        return cudaSuccess;
    }

    // Coefficients `first` .. `first + count` of the complex numbers a (re, im).
    template <typename Float>
    bridge_cuda::BlasCoefficients<Float> coefficients(const double* a, int first, int count) {
        bridge_cuda::BlasCoefficients<Float> c{};
        for (int k = 0; k < count; ++k) {
            c.re[k] = static_cast<Float>(a[2 * (first + k)]);
            c.im[k] = static_cast<Float>(a[2 * (first + k) + 1]);
        }
        return c;
    }

    // A reduction of NQ quantities of each of `vectors` vectors of `pairs` complex numbers:
    // make(first, count) is the operation on the vectors first .. first + count (at most
    // BlasVectors, for the coefficients of a launch). On success `*results` is host memory
    // with quantity q of vector v at q * vectors + v, until the next reduction of the thread.
    template <int NQ, typename Make>
    int reduce(int pairs, int vectors, cudaStream_t stream, Make make, const double** results) {
        const int per = get_grid_size(pairs);
        const size_t count_all = static_cast<size_t>(NQ) * vectors;
        int e = reserve(count_all * per, count_all);
        if (e != cudaSuccess) return e;
        for (int first = 0; first < vectors; first += bridge_cuda::BlasVectors) {
            const int count = std::min(bridge_cuda::BlasVectors, vectors - first);
            bridge_cuda::blas_reduce_kernel<NQ><<<count * per, BLOCK_SIZE, 0, stream>>>(
                make(first, count), reduction.sums, pairs, count, first, vectors);
        }
        bridge_cuda::blas_finish_kernel<<<static_cast<int>(count_all), BLOCK_SIZE, 0, stream>>>(
            reduction.results, reduction.sums, per);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(reduction.host, reduction.results, count_all * sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) return e;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;
        *results = reduction.host;
        return cudaSuccess;
    }

    // An operation without a result on `vectors` vectors, BlasVectors at a time.
    template <typename Make>
    void map(int pairs, int vectors, cudaStream_t stream, Make make) {
        for (int first = 0; first < vectors; first += bridge_cuda::BlasVectors) {
            const int count = std::min(bridge_cuda::BlasVectors, vectors - first);
            bridge_cuda::blas_map_kernel<<<get_grid_size(pairs, count), BLOCK_SIZE, 0, stream>>>(
                make(first, count), pairs, count);
        }
    }

    template <typename Float>
    int norm2(double* out, const Float* x, int n, int vectors, cudaStream_t stream) {
        const double* r = nullptr;
        const int e = reduce<1>(n / 2, vectors, stream, [=](int first, int) {
            return bridge_cuda::BlasNorm<Float>{x + static_cast<size_t>(first) * n};
        }, &r);
        if (e != cudaSuccess) return e;
        std::copy(r, r + vectors, out);
        return cudaSuccess;
    }

    template <typename Float>
    int dotc(double* re, double* im, const Float* x, const Float* y, int n, int vectors, cudaStream_t stream) {
        const double* r = nullptr;
        const int e = reduce<2>(n / 2, vectors, stream, [=](int first, int) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasDot<Float>{x + at, y + at};
        }, &r);
        if (e != cudaSuccess) return e;
        std::copy(r, r + vectors, re);
        std::copy(r + vectors, r + 2 * vectors, im);
        return cudaSuccess;
    }

    template <typename Float>
    int axpy_norm2(Float* y, const Float* x, const double* a, double* norm, int n, int vectors, cudaStream_t stream) {
        const double* r = nullptr;
        const int e = reduce<1>(n / 2, vectors, stream, [=](int first, int count) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasAxpyNorm<Float>{y + at, x + at, coefficients<Float>(a, first, count)};
        }, &r);
        if (e != cudaSuccess) return e;
        std::copy(r, r + vectors, norm);
        return cudaSuccess;
    }

    // Coefficients `first` .. `first + count` of the reals c, as complex numbers.
    template <typename Float>
    bridge_cuda::BlasCoefficients<Float> real_coefficients(const double* c, int first, int count) {
        bridge_cuda::BlasCoefficients<Float> r{};
        for (int k = 0; k < count; ++k) r.re[k] = static_cast<Float>(c[first + k]);
        return r;
    }

    template <typename Float>
    int scale_axpy2_norm2(Float* s, const double* c, const double* a, const Float* x, const double* b,
                          const Float* y, double* norm, int n, int vectors, cudaStream_t stream) {
        const double* r = nullptr;
        const int e = reduce<1>(n / 2, vectors, stream, [=](int first, int count) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasScaleAxpy2Norm<Float>{s + at, x + at, y + at,
                                                          real_coefficients<Float>(c, first, count),
                                                          coefficients<Float>(a, first, count),
                                                          coefficients<Float>(b, first, count)};
        }, &r);
        if (e != cudaSuccess) return e;
        std::copy(r, r + vectors, norm);
        return cudaSuccess;
    }

    // <x_i|y_j> into re[i ny + j], im[i ny + j]: blas_gram_kernel, then the sums of its CUDA
    // blocks added on the device; one wait for the stream.
    template <typename Float>
    int gram(double* re, double* im, const Float* x, const Float* y, int n, int nx, int ny, cudaStream_t stream) {
        constexpr int T = bridge_cuda::GramTile, NQ = 2 * T * T;
        const int pairs = n / 2, tiles_y = (ny + T - 1) / T;
        const int tiles = ((nx + T - 1) / T) * tiles_y;
        // Enough CUDA blocks over all the tiles to fill the device, at least one for each.
        const int per = std::max(1, std::min(get_grid_size(pairs), 2048 / tiles));
        const size_t count_all = static_cast<size_t>(tiles) * NQ;
        int e = reserve(count_all * per, count_all);
        if (e != cudaSuccess) return e;
        bridge_cuda::blas_gram_kernel<Float><<<dim3(per, tiles), BLOCK_SIZE, 0, stream>>>(
            reduction.sums, x, y, pairs, nx, ny);
        bridge_cuda::blas_finish_kernel<<<static_cast<int>(count_all), BLOCK_SIZE, 0, stream>>>(
            reduction.results, reduction.sums, per);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(reduction.host, reduction.results, count_all * sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) return e;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;
        for (int tile = 0; tile < tiles; ++tile) {
            const int i0 = (tile / tiles_y) * T, j0 = (tile % tiles_y) * T;
            for (int a = 0; a < T && i0 + a < nx; ++a) {
                for (int c = 0; c < T && j0 + c < ny; ++c) {
                    const size_t q = static_cast<size_t>(tile) * NQ + 2 * (a * T + c);
                    const size_t at = static_cast<size_t>(i0 + a) * ny + j0 + c;
                    re[at] = reduction.host[q];
                    im[at] = reduction.host[q + 1];
                }
            }
        }
        return cudaSuccess;
    }

    template <typename Float>
    int xpay_norm2_dot(Float* s, const Float* x, const double* a, const Float* y, const Float* z,
                       double* norm, double* re, double* im, int n, int vectors, cudaStream_t stream) {
        const double* r = nullptr;
        const int e = reduce<3>(n / 2, vectors, stream, [=](int first, int count) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasXpayNormDot<Float>{s + at, x + at, y + at, z + at,
                                                       coefficients<Float>(a, first, count)};
        }, &r);
        if (e != cudaSuccess) return e;
        std::copy(r, r + vectors, norm);
        std::copy(r + vectors, r + 2 * vectors, re);
        std::copy(r + 2 * vectors, r + 3 * vectors, im);
        return cudaSuccess;
    }

    template <typename Float>
    int dot_norms(const Float* s, const Float* x, double* re, double* im, double* norm_s, double* norm_x,
                  int n, int vectors, cudaStream_t stream) {
        const double* r = nullptr;
        const int e = reduce<4>(n / 2, vectors, stream, [=](int first, int) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasDotNorms<Float>{s + at, x + at};
        }, &r);
        if (e != cudaSuccess) return e;
        std::copy(r, r + vectors, re);
        std::copy(r + vectors, r + 2 * vectors, im);
        std::copy(r + 2 * vectors, r + 3 * vectors, norm_s);
        std::copy(r + 3 * vectors, r + 4 * vectors, norm_x);
        return cudaSuccess;
    }

    template <typename Float>
    void xpay(Float* s, const Float* x, const double* a, const Float* y, int n, int vectors, cudaStream_t stream) {
        map(n / 2, vectors, stream, [=](int first, int count) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasXpay<Float>{s + at, x + at, y + at, coefficients<Float>(a, first, count)};
        });
    }

    template <typename Float>
    void axpy2(Float* s, const double* a, const Float* x, const double* b, const Float* y, int n, int vectors, cudaStream_t stream) {
        map(n / 2, vectors, stream, [=](int first, int count) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasAxpy2<Float>{s + at, x + at, y + at, coefficients<Float>(a, first, count),
                                                 coefficients<Float>(b, first, count)};
        });
    }

    template <typename Float>
    void axpy_aypx(Float* s, const double* a, const Float* x, const double* b, const Float* y, int n, int vectors, cudaStream_t stream) {
        map(n / 2, vectors, stream, [=](int first, int count) {
            const size_t at = static_cast<size_t>(first) * n;
            return bridge_cuda::BlasAxpyAypx<Float>{s + at, x + at, y + at, coefficients<Float>(a, first, count),
                                                    coefficients<Float>(b, first, count)};
        });
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

int bridge_cuda_blas_norm2_f32(double* out, const float* x, int n, int vectors, cudaStream_t stream) {
    return norm2(out, x, n, vectors, stream);
}
int bridge_cuda_blas_norm2_f64(double* out, const double* x, int n, int vectors, cudaStream_t stream) {
    return norm2(out, x, n, vectors, stream);
}
int bridge_cuda_blas_dotc_f32(double* re, double* im, const float* x, const float* y, int n, int vectors, cudaStream_t stream) {
    return dotc(re, im, x, y, n, vectors, stream);
}
int bridge_cuda_blas_dotc_f64(double* re, double* im, const double* x, const double* y, int n, int vectors, cudaStream_t stream) {
    return dotc(re, im, x, y, n, vectors, stream);
}
int bridge_cuda_blas_axpy_norm2_f32(float* y, const float* x, const double* a, double* norm, int n, int vectors, cudaStream_t stream) {
    return axpy_norm2(y, x, a, norm, n, vectors, stream);
}
int bridge_cuda_blas_axpy_norm2_f64(double* y, const double* x, const double* a, double* norm, int n, int vectors, cudaStream_t stream) {
    return axpy_norm2(y, x, a, norm, n, vectors, stream);
}
int bridge_cuda_blas_xpay_norm2_dot_f32(float* s, const float* x, const double* a, const float* y, const float* z, double* norm, double* re, double* im, int n, int vectors, cudaStream_t stream) {
    return xpay_norm2_dot(s, x, a, y, z, norm, re, im, n, vectors, stream);
}
int bridge_cuda_blas_xpay_norm2_dot_f64(double* s, const double* x, const double* a, const double* y, const double* z, double* norm, double* re, double* im, int n, int vectors, cudaStream_t stream) {
    return xpay_norm2_dot(s, x, a, y, z, norm, re, im, n, vectors, stream);
}
int bridge_cuda_blas_dot_norms_f32(const float* s, const float* x, double* re, double* im, double* norm_s, double* norm_x, int n, int vectors, cudaStream_t stream) {
    return dot_norms(s, x, re, im, norm_s, norm_x, n, vectors, stream);
}
int bridge_cuda_blas_dot_norms_f64(const double* s, const double* x, double* re, double* im, double* norm_s, double* norm_x, int n, int vectors, cudaStream_t stream) {
    return dot_norms(s, x, re, im, norm_s, norm_x, n, vectors, stream);
}
int bridge_cuda_blas_gram_f32(double* re, double* im, const float* x, const float* y, int n, int nx, int ny, cudaStream_t stream) {
    return gram(re, im, x, y, n, nx, ny, stream);
}
int bridge_cuda_blas_gram_f64(double* re, double* im, const double* x, const double* y, int n, int nx, int ny, cudaStream_t stream) {
    return gram(re, im, x, y, n, nx, ny, stream);
}
int bridge_cuda_blas_scale_axpy2_norm2_f32(float* s, const double* c, const double* a, const float* x, const double* b, const float* y, double* norm, int n, int vectors, cudaStream_t stream) {
    return scale_axpy2_norm2(s, c, a, x, b, y, norm, n, vectors, stream);
}
int bridge_cuda_blas_scale_axpy2_norm2_f64(double* s, const double* c, const double* a, const double* x, const double* b, const double* y, double* norm, int n, int vectors, cudaStream_t stream) {
    return scale_axpy2_norm2(s, c, a, x, b, y, norm, n, vectors, stream);
}
void bridge_cuda_blas_xpay_f32(float* s, const float* x, const double* a, const float* y, int n, int vectors, cudaStream_t stream) {
    xpay(s, x, a, y, n, vectors, stream);
}
void bridge_cuda_blas_xpay_f64(double* s, const double* x, const double* a, const double* y, int n, int vectors, cudaStream_t stream) {
    xpay(s, x, a, y, n, vectors, stream);
}
void bridge_cuda_blas_axpy2_f32(float* s, const double* a, const float* x, const double* b, const float* y, int n, int vectors, cudaStream_t stream) {
    axpy2(s, a, x, b, y, n, vectors, stream);
}
void bridge_cuda_blas_axpy2_f64(double* s, const double* a, const double* x, const double* b, const double* y, int n, int vectors, cudaStream_t stream) {
    axpy2(s, a, x, b, y, n, vectors, stream);
}
void bridge_cuda_blas_axpy_aypx_f32(float* s, const double* a, const float* x, const double* b, const float* y, int n, int vectors, cudaStream_t stream) {
    axpy_aypx(s, a, x, b, y, n, vectors, stream);
}
void bridge_cuda_blas_axpy_aypx_f64(double* s, const double* a, const double* x, const double* b, const double* y, int n, int vectors, cudaStream_t stream) {
    axpy_aypx(s, a, x, b, y, n, vectors, stream);
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

void bridge_cuda_blas_combine_f32(float* out, const float* x, const double* c, int n, int n_in, int n_out, cudaStream_t stream) {
    for (int j0 = 0; j0 < n_out; j0 += bridge_cuda::RotateVectors) {
        const int count = std::min(bridge_cuda::RotateVectors, n_out - j0);
        bridge_cuda::blas_combine_kernel<float><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(out, x, c, n / 2, n_in, n_out, j0, count);
    }
}

void bridge_cuda_blas_combine_f64(double* out, const double* x, const double* c, int n, int n_in, int n_out, cudaStream_t stream) {
    for (int j0 = 0; j0 < n_out; j0 += bridge_cuda::RotateVectors) {
        const int count = std::min(bridge_cuda::RotateVectors, n_out - j0);
        bridge_cuda::blas_combine_kernel<double><<<get_grid_size(n / 2), BLOCK_SIZE, 0, stream>>>(out, x, c, n / 2, n_in, n_out, j0, count);
    }
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
