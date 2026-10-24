#ifndef BRIDGE_CUDA_LINALG_BLAS_CUH
#define BRIDGE_CUDA_LINALG_BLAS_CUH

#include "../base/types.h"
#include <cuda_runtime.h>

namespace bridge_cuda {

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

// -------------------------------------------------------------------------
// Fused operations on sets of vectors, in one pass over the fields. `Op` has what is done to
// complex number `p` of vector `v` of the set (reals at kr and ki), and in a reduction adds
// its NQ quantities to `acc`. A reduction is in two stages on the device, both in a fixed
// order (no atomics), so that the result does not depend on timing: `blas_reduce_kernel`
// leaves the sum of every CUDA block of every (quantity, vector), in double precision
// whatever the field stores, and `blas_finish_kernel` adds those up; only the NQ numbers of
// each vector go to the host.
// -------------------------------------------------------------------------
template <typename Op>
__global__ void blas_map_kernel(Op op, int pairs, int vectors) {
    const size_t all = static_cast<size_t>(pairs) * vectors;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < all; p += blockDim.x * gridDim.x) {
        const size_t kr = blas_real_index(p);
        op(static_cast<int>(p / pairs), kr, kr + BRIDGE_CUDA_WARP_SIZE);
    }
}

// gridDim.x = vectors * per: CUDA block b of vector v is block v * per + b; its sum of
// quantity q goes to block_sums[((q * total + first + v) * per + b], `total` the vectors of
// the whole reduction and `first` those before this launch.
template <int NQ, typename Op, int BLOCK_SIZE = 256>
__global__ void blas_reduce_kernel(Op op, double* __restrict__ block_sums, int pairs, int vectors, int first, int total) {
    __shared__ double s[NQ][BLOCK_SIZE];
    const int tid = threadIdx.x, per = gridDim.x / vectors;
    const int v = blockIdx.x / per, b = blockIdx.x % per;
    double acc[NQ];
    #pragma unroll
    for (int q = 0; q < NQ; ++q) acc[q] = 0.0;
    for (int p = b * blockDim.x + tid; p < pairs; p += blockDim.x * per) {
        const size_t kr = static_cast<size_t>(v) * 2 * pairs + blas_real_index(p);
        op(v, kr, kr + BRIDGE_CUDA_WARP_SIZE, acc);
    }
    #pragma unroll
    for (int q = 0; q < NQ; ++q) s[q][tid] = acc[q];
    __syncthreads();
    #pragma unroll
    for (int step = BLOCK_SIZE / 2; step > 0; step >>= 1) {
        if (tid < step) {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) s[q][tid] += s[q][tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            block_sums[(static_cast<size_t>(q) * total + first + v) * per + b] = s[q][0];
        }
    }
}

// out[i] = sum_b block_sums[i * per + b], a CUDA block for each i.
template <int BLOCK_SIZE = 256>
__global__ void blas_finish_kernel(double* __restrict__ out, const double* __restrict__ block_sums, int per) {
    __shared__ double s[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const double* from = block_sums + static_cast<size_t>(blockIdx.x) * per;
    double acc = 0.0;
    for (int b = tid; b < per; b += BLOCK_SIZE) acc += from[b];
    s[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int step = BLOCK_SIZE / 2; step > 0; step >>= 1) {
        if (tid < step) s[tid] += s[tid + step];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// <x_i|y_j> for every vector x_i of one set and y_j of another, by tiles of GramTile x GramTile
// pairs, blockIdx.y the tile (those of y fastest): a thread reads the GramTile numbers of
// either side at its places once for all the pairs of the tile, so a vector is read once for
// every GramTile vectors of the other set (one pair at a time, it was read for each, and
// copied). gridDim.x CUDA blocks for each tile; the sum of block b of quantity q of the tile
// (2 (a GramTile + c), + 1 for the imaginary part, of x_(i0 + a) and y_(j0 + c)) goes to
// block_sums[(tile 2 GramTile^2 + q) gridDim.x + b], for blas_finish_kernel. In double
// precision, the lanes and then the warps added in a fixed order.
constexpr int GramTile = 4;

template <typename Float, int BLOCK_SIZE = 256>
__global__ void blas_gram_kernel(double* __restrict__ block_sums, const Float* __restrict__ x,
                                 const Float* __restrict__ y, int pairs, int nx, int ny) {
    constexpr int NQ = 2 * GramTile * GramTile, WARPS = BLOCK_SIZE / BRIDGE_CUDA_WARP_SIZE;
    __shared__ double s[WARPS][NQ];
    const int per = gridDim.x, b = blockIdx.x, tile = blockIdx.y;
    const int tiles_y = (ny + GramTile - 1) / GramTile;
    const int i0 = (tile / tiles_y) * GramTile, j0 = (tile % tiles_y) * GramTile;
    const size_t n = 2 * static_cast<size_t>(pairs);
    double acc[NQ];
    #pragma unroll
    for (int q = 0; q < NQ; ++q) acc[q] = 0.0;
    for (int p = b * BLOCK_SIZE + threadIdx.x; p < pairs; p += BLOCK_SIZE * per) {
        const size_t kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        double xr[GramTile], xi[GramTile];
        #pragma unroll
        for (int a = 0; a < GramTile; ++a) {
            const bool in = i0 + a < nx;
            xr[a] = in ? static_cast<double>(x[(i0 + a) * n + kr]) : 0.0;
            xi[a] = in ? static_cast<double>(x[(i0 + a) * n + ki]) : 0.0;
        }
        #pragma unroll
        for (int c = 0; c < GramTile; ++c) {
            const bool in = j0 + c < ny;
            const double yr = in ? static_cast<double>(y[(j0 + c) * n + kr]) : 0.0;
            const double yi = in ? static_cast<double>(y[(j0 + c) * n + ki]) : 0.0;
            #pragma unroll
            for (int a = 0; a < GramTile; ++a) {
                acc[2 * (a * GramTile + c)] += xr[a] * yr + xi[a] * yi;
                acc[2 * (a * GramTile + c) + 1] += xr[a] * yi - xi[a] * yr;
            }
        }
    }
    const int lane = threadIdx.x % BRIDGE_CUDA_WARP_SIZE, warp = threadIdx.x / BRIDGE_CUDA_WARP_SIZE;
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        double v = acc[q];
        #pragma unroll
        for (int step = BRIDGE_CUDA_WARP_SIZE / 2; step > 0; step >>= 1) v += __shfl_xor_sync(0xffffffffu, v, step);
        if (lane == 0) s[warp][q] = v;
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
        double t = 0.0;
        for (int w = 0; w < WARPS; ++w) t += s[w][threadIdx.x];
        block_sums[(static_cast<size_t>(tile) * NQ + threadIdx.x) * per + b] = t;
    }
}

// The operations. Coefficients a, b: one complex number for each vector of the launch.
template <typename Float>
struct BlasNorm {  // |x_v|^2
    const Float* x;
    __device__ void operator()(int, size_t r, size_t i, double* acc) const {
        const double xr = x[r], xi = x[i];
        acc[0] += xr * xr + xi * xi;
    }
};

template <typename Float>
struct BlasDot {  // <x_v|y_v>
    const Float* x;
    const Float* y;
    __device__ void operator()(int, size_t r, size_t i, double* acc) const {
        const double xr = x[r], xi = x[i], yr = y[r], yi = y[i];
        acc[0] += xr * yr + xi * yi;
        acc[1] += xr * yi - xi * yr;
    }
};

template <typename Float>
struct BlasDotNorms {  // <s_v|x_v>, |s_v|^2, |x_v|^2
    const Float* s;
    const Float* x;
    __device__ void operator()(int, size_t r, size_t i, double* acc) const {
        const double sr = s[r], si = s[i], xr = x[r], xi = x[i];
        acc[0] += sr * xr + si * xi;
        acc[1] += sr * xi - si * xr;
        acc[2] += sr * sr + si * si;
        acc[3] += xr * xr + xi * xi;
    }
};

template <typename Float>
struct BlasAxpyNorm {  // y_v += a_v x_v, then |y_v|^2
    Float* y;
    const Float* x;
    BlasCoefficients<Float> a;
    __device__ void operator()(int v, size_t r, size_t i, double* acc) const {
        const Float xr = x[r], xi = x[i];
        const Float yr = y[r] + a.re[v] * xr - a.im[v] * xi;
        const Float yi = y[i] + a.re[v] * xi + a.im[v] * xr;
        y[r] = yr;
        y[i] = yi;
        acc[0] += static_cast<double>(yr) * yr + static_cast<double>(yi) * yi;
    }
};

template <typename Float>
struct BlasXpayNormDot {  // s_v = x_v + a_v y_v, then |s_v|^2 and <z_v|s_v>
    Float* s;
    const Float* x;
    const Float* y;
    const Float* z;
    BlasCoefficients<Float> a;
    __device__ void operator()(int v, size_t r, size_t i, double* acc) const {
        const Float yr = y[r], yi = y[i];
        const Float sr = x[r] + a.re[v] * yr - a.im[v] * yi;
        const Float si = x[i] + a.re[v] * yi + a.im[v] * yr;
        s[r] = sr;
        s[i] = si;
        const double zr = z[r], zi = z[i];
        acc[0] += static_cast<double>(sr) * sr + static_cast<double>(si) * si;
        acc[1] += zr * sr + zi * si;
        acc[2] += zr * si - zi * sr;
    }
};

template <typename Float>
struct BlasScaleAxpy2Norm {  // s_v = c_v s_v + a_v x_v + b_v y_v (c_v real: c.re), then |s_v|^2
    Float* s;
    const Float* x;
    const Float* y;
    BlasCoefficients<Float> c, a, b;
    __device__ void operator()(int v, size_t r, size_t i, double* acc) const {
        const Float xr = x[r], xi = x[i], yr = y[r], yi = y[i];
        const Float sr = c.re[v] * s[r] + a.re[v] * xr - a.im[v] * xi + b.re[v] * yr - b.im[v] * yi;
        const Float si = c.re[v] * s[i] + a.re[v] * xi + a.im[v] * xr + b.re[v] * yi + b.im[v] * yr;
        s[r] = sr;
        s[i] = si;
        acc[0] += static_cast<double>(sr) * sr + static_cast<double>(si) * si;
    }
};

template <typename Float>
struct BlasXpay {  // s_v = x_v + a_v y_v
    Float* s;
    const Float* x;
    const Float* y;
    BlasCoefficients<Float> a;
    __device__ void operator()(int v, size_t r, size_t i) const {
        const Float yr = y[r], yi = y[i];
        s[r] = x[r] + a.re[v] * yr - a.im[v] * yi;
        s[i] = x[i] + a.re[v] * yi + a.im[v] * yr;
    }
};

template <typename Float>
struct BlasAxpy2 {  // s_v += a_v x_v + b_v y_v
    Float* s;
    const Float* x;
    const Float* y;
    BlasCoefficients<Float> a, b;
    __device__ void operator()(int v, size_t r, size_t i) const {
        const Float xr = x[r], xi = x[i], yr = y[r], yi = y[i];
        s[r] += a.re[v] * xr - a.im[v] * xi + b.re[v] * yr - b.im[v] * yi;
        s[i] += a.re[v] * xi + a.im[v] * xr + b.re[v] * yi + b.im[v] * yr;
    }
};

template <typename Float>
struct BlasAxpyAypx {  // s_v = y_v + b_v (s_v + a_v x_v)
    Float* s;
    const Float* x;
    const Float* y;
    BlasCoefficients<Float> a, b;
    __device__ void operator()(int v, size_t r, size_t i) const {
        const Float xr = x[r], xi = x[i];
        const Float tr = s[r] + a.re[v] * xr - a.im[v] * xi;
        const Float ti = s[i] + a.re[v] * xi + a.im[v] * xr;
        s[r] = y[r] + b.re[v] * tr - b.im[v] * ti;
        s[i] = y[i] + b.re[v] * ti + b.im[v] * tr;
    }
};

// A step of Chebyshev iteration (solver Chebyshev): out_v += d_v, r_v -= ad_v, then
// d_v = c_v d_v + a_v r_v (c_v real: c.re), in one pass where the four kernels it replaces
// (caxpy, caxpy, scal, caxpy) made four: 7 reads and writes of a field against 11. The
// arithmetic is theirs, with `one` = 1 and `minus_one` = -1 as coefficients as they had them,
// so that the result is the same to the bit. c d is rounded as the store of scal rounded it:
// a plain product would be contracted with the sum after it into an FMA.
__device__ __forceinline__ float blas_mul_rounded(float x, float y) { return __fmul_rn(x, y); }
__device__ __forceinline__ double blas_mul_rounded(double x, double y) { return __dmul_rn(x, y); }

template <typename Float>
struct BlasChebyshevStep {
    Float* out;
    Float* d;
    Float* r;
    const Float* ad;
    BlasCoefficients<Float> one, minus_one, c, a;
    __device__ void operator()(int v, size_t kr, size_t ki) const {
        Float dr = d[kr], di = d[ki];
        out[kr] += one.re[v] * dr - one.im[v] * di;
        out[ki] += one.re[v] * di + one.im[v] * dr;
        const Float adr = ad[kr], adi = ad[ki];
        Float rr = r[kr], ri = r[ki];
        rr += minus_one.re[v] * adr - minus_one.im[v] * adi;
        ri += minus_one.re[v] * adi + minus_one.im[v] * adr;
        r[kr] = rr;
        r[ki] = ri;
        dr = blas_mul_rounded(dr, c.re[v]);
        di = blas_mul_rounded(di, c.re[v]);
        dr += a.re[v] * rr - a.im[v] * ri;
        di += a.re[v] * ri + a.im[v] * rr;
        d[kr] = dr;
        d[ki] = di;
    }
};

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

// out_(j0 + j) = sum_i x_i c[i n_out + j0 + j] for j < count (at most RotateVectors), the n_in
// vectors of x and the n_out of out sets of their own: the outputs of a chunk in registers,
// the inputs read once per chunk.
template <typename Float>
__global__ void blas_combine_kernel(Float* __restrict__ out, const Float* __restrict__ x, const double* __restrict__ c,
                                    int pairs, int n_in, int n_out, int j0, int count) {
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < static_cast<size_t>(pairs);
         p += blockDim.x * gridDim.x) {
        const size_t kr = blas_real_index(p), ki = kr + BRIDGE_CUDA_WARP_SIZE;
        Float zr[RotateVectors], zi[RotateVectors];
        for (int j = 0; j < count; ++j) zr[j] = zi[j] = 0;
        for (int i = 0; i < n_in; ++i) {
            const Float xr = x[static_cast<size_t>(i) * 2 * pairs + kr], xi = x[static_cast<size_t>(i) * 2 * pairs + ki];
            for (int j = 0; j < count; ++j) {
                const size_t at = 2 * (static_cast<size_t>(i) * n_out + j0 + j);
                const Float cr = static_cast<Float>(c[at]), ci = static_cast<Float>(c[at + 1]);
                zr[j] += xr * cr - xi * ci;
                zi[j] += xr * ci + xi * cr;
            }
        }
        for (int j = 0; j < count; ++j) {
            out[static_cast<size_t>(j0 + j) * 2 * pairs + kr] = zr[j];
            out[static_cast<size_t>(j0 + j) * 2 * pairs + ki] = zi[j];
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_LINALG_BLAS_CUH
