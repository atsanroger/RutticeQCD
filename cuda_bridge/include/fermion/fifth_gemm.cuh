#ifndef BRIDGE_CUDA_FERMION_FIFTH_GEMM_CUH
#define BRIDGE_CUDA_FERMION_FIFTH_GEMM_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "hops.cuh"
#include <cuda_runtime.h>

// What acts within a site of the four-dimensional lattice, on the fifth dimension and the
// chirality only, as a matrix product.
//
// A field with Ls spinors per site is, in every block of 32 sites, 2 Ls rows of 384 reals,
// one after the other: row 2 s + h holds the upper (h = 0) or lower (h = 1) two spin
// components of the spinors at s of the 32 sites (base/aosoa.cuh). In the Dirac
// representation gamma_5 exchanges the upper and the lower components, so P+ and P- are
// combinations of the two rows of an s, and every operator of the form
//
//   sum over s' of (A_ss' P+ + B_ss' P-)
//
// (Y, C = d+ + d- Y and its inverse, b - c Y, Neff's Q, their adjoints and their products)
// is one real (2 Ls) x (2 Ls) matrix M on the rows:
//
//   out (2 Ls x 384) = M (2 Ls x 2 Ls) in (2 Ls x 384)     in every block.
//
// Products of such operators are multiplied on the host, and applied in one pass.
//
// M is given with its rows padded to a multiple of 8 and its columns to a multiple of 4 with
// zeros, row after row: 8 NT rows of K4 = 4 ceil(2 Ls / 4) reals.

namespace bridge_cuda {

constexpr int FifthColumns = 384;   // reals in a row of a block
constexpr int FifthStrips = FifthColumns / 8;

// -------------------------------------------------------------------------
// The product in double precision on the tensor cores:
//
//   D (8 x 8) += A (8 x 4) B (4 x 8),    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64,
//
// in which thread t of a warp holds A[t/4][t%4], B[t%4][t/4] and D[t/4][2 (t%4) + i], i = 0, 1.
// A warp has a strip of 8 columns of one block: for every 4 rows of the input it reads its
// 4 x 8 reals once (B), and with the 8 x 4 tiles of M (A) adds to the NT tiles of 8 rows of
// the result, which stay in its registers until they are written.
//
// M is in the shared memory of the block of threads. A tile of it is read by the 32 threads
// of a warp at once, 8 rows of 4 columns: with the rows one under the other they would fall
// on the same banks of the shared memory (a row is a multiple of 16 reals, and 16 reals of 8
// bytes are the 32 banks), eight threads to a bank. So real (row, k) is kept at column
// k ^ (4 (row % 4)) of its row: the four columns of a tile are then four different groups of
// four in rows that differ in row % 4, and two threads share a bank, the least there can be
// when 32 threads read 8 bytes each. The rows of the copy have K16 = 16 ceil(2 Ls / 16) reals.
//
// out = M in, or from - M in if d_from is not null. out must not be in.
// Shared memory: 8 NT * K16 doubles.
// -------------------------------------------------------------------------
template <int NT>
__global__ void fifth_gemm_mma_kernel(
    double* __restrict__ d_out,
    const double* __restrict__ d_in,
    const double* __restrict__ d_from,
    const double* __restrict__ d_m,
    int blocks,
    int Ls)
{
    extern __shared__ double m_shared[];
    const int rows = 2 * Ls;
    const int k4 = (rows + 3) / 4 * 4;
    const int k16 = (rows + 15) / 16 * 16;
    for (int i = threadIdx.x; i < 8 * NT * k4; i += blockDim.x) {
        const int row = i / k4, k = i % k4;
        m_shared[row * k16 + (k ^ (4 * (row % 4)))] = d_m[i];
    }
    __syncthreads();

    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    // A whole warp leaves, or none of it: every thread of a warp takes part in an mma.
    if (warp >= blocks * FifthStrips) return;
    const size_t base = static_cast<size_t>(warp / FifthStrips) * rows * FifthColumns;
    const int n0 = (warp % FifthStrips) * 8;

    double d[NT][2];
    #pragma unroll
    for (int it = 0; it < NT; ++it) { d[it][0] = 0.0; d[it][1] = 0.0; }

    for (int kk = 0; kk < k4; kk += 4) {
        const int j = kk + lane % 4;
        // The rows past the last one are zero; no thread branches before the mma.
        const int j_in = (j < rows) ? j : rows - 1;
        const double read = d_in[base + static_cast<size_t>(j_in) * FifthColumns + n0 + lane / 4];
        const double fb = (j < rows) ? read : 0.0;
        #pragma unroll
        for (int it = 0; it < NT; ++it) {
            const int row = it * 8 + lane / 4;
            const double fa = m_shared[row * k16 + (j ^ (4 * (row % 4)))];
            const double c0 = d[it][0], c1 = d[it][1];
            double d0, d1;
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
                : "=d"(d0), "=d"(d1)
                : "d"(fa), "d"(fb), "d"(c0), "d"(c1));
            d[it][0] = d0;
            d[it][1] = d1;
        }
    }

    #pragma unroll
    for (int it = 0; it < NT; ++it) {
        const int row = it * 8 + lane / 4;
        if (row < rows) {
            const size_t k = base + static_cast<size_t>(row) * FifthColumns + n0 + (lane % 4) * 2;
            if (d_from != nullptr) {
                d_out[k] = d_from[k] - d[it][0];
                d_out[k + 1] = d_from[k + 1] - d[it][1];
            } else {
                d_out[k] = d[it][0];
                d_out[k + 1] = d[it][1];
            }
        }
    }
}

// -------------------------------------------------------------------------
// The product in single precision on the tensor cores, which is TF32: the reals of M and of
// the input are rounded to ten bits of mantissa, and the sums are single precision. Not for
// an operator whose result is kept: for what is corrected afterwards (a smoother, an inner
// solve), and only where it is asked for.
//
//   D (16 x 8) += A (16 x 8) B (8 x 8),    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32,
//
// in which thread t = 4 g + i of a warp holds A[g][i], A[g+8][i], A[g][i+4], A[g+8][i+4],
// B[i][g], B[i+4][g], and D[g][2i], D[g][2i+1], D[g+8][2i], D[g+8][2i+1]. A warp has a strip
// of 8 columns of one block, as in fifth_gemm_mma_kernel.
//
// M is in the shared memory, rounded already, 16 NT rows of K32 = 32 ceil(2 Ls / 32) reals,
// real (row, k) at column k ^ (4 (row % 8)): the 32 threads of a warp read 8 rows of 4
// columns at once, which then are the 32 banks.
// Shared memory: 16 NT * K32 reals.
// -------------------------------------------------------------------------
__device__ __forceinline__ unsigned to_tf32(float x) {
    unsigned rounded;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(rounded) : "f"(x));
    return rounded;
}

template <int NT>
__global__ void fifth_gemm_tf32_kernel(
    float* __restrict__ d_out,
    const float* __restrict__ d_in,
    const float* __restrict__ d_from,
    const float* __restrict__ d_m,
    int blocks,
    int Ls)
{
    extern __shared__ unsigned m_tf32[];
    const int rows = 2 * Ls;
    // M as it is given: rows8 rows of k4 reals.
    const int rows8 = (rows + 7) / 8 * 8, k4 = (rows + 3) / 4 * 4;
    const int k32 = (rows + 31) / 32 * 32;
    for (int i = threadIdx.x; i < 16 * NT * rows8; i += blockDim.x) {
        const int row = i / rows8, k = i % rows8;
        const float m = (row < rows8 && k < k4) ? d_m[row * k4 + k] : 0.0f;
        m_tf32[row * k32 + (k ^ (4 * (row % 8)))] = to_tf32(m);
    }
    __syncthreads();

    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int g = (threadIdx.x % 32) / 4, i = threadIdx.x % 4;
    // A whole warp leaves, or none of it.
    if (warp >= blocks * FifthStrips) return;
    const size_t base = static_cast<size_t>(warp / FifthStrips) * rows * FifthColumns;
    const int n0 = (warp % FifthStrips) * 8;

    float d[NT][4];
    #pragma unroll
    for (int it = 0; it < NT; ++it) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) d[it][q] = 0.0f;
    }

    for (int kk = 0; kk < rows8; kk += 8) {
        unsigned b[2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            // The rows past the last one are zero; no thread branches before the mma.
            const int j = kk + i + 4 * h;
            const int j_in = (j < rows) ? j : rows - 1;
            const float read = d_in[base + static_cast<size_t>(j_in) * FifthColumns + n0 + g];
            b[h] = to_tf32((j < rows) ? read : 0.0f);
        }
        #pragma unroll
        for (int it = 0; it < NT; ++it) {
            const int upper = (it * 16 + g) * k32, lower = upper + 8 * k32;
            const int j0 = (kk + i) ^ (4 * g), j1 = (kk + i + 4) ^ (4 * g);
            const unsigned a0 = m_tf32[upper + j0], a1 = m_tf32[lower + j0];
            const unsigned a2 = m_tf32[upper + j1], a3 = m_tf32[lower + j1];
            float e0, e1, e2, e3;
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(e0), "=f"(e1), "=f"(e2), "=f"(e3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[0]), "r"(b[1]),
                  "f"(d[it][0]), "f"(d[it][1]), "f"(d[it][2]), "f"(d[it][3]));
            d[it][0] = e0;
            d[it][1] = e1;
            d[it][2] = e2;
            d[it][3] = e3;
        }
    }

    #pragma unroll
    for (int it = 0; it < NT; ++it) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int row = it * 16 + g + 8 * (q / 2);
            if (row < rows) {
                const size_t k = base + static_cast<size_t>(row) * FifthColumns + n0 + 2 * i + q % 2;
                d_out[k] = (d_from != nullptr) ? d_from[k] - d[it][q] : d[it][q];
            }
        }
    }
}

// -------------------------------------------------------------------------
// The same product in single precision without the tensor cores, exact to single precision. A thread has a column of a block: it reads the
// 2 Ls reals of the column once each and adds them to the 8 NT reals of the result, which
// stay in its registers until they are written. M is in the shared memory of the block of
// threads; its threads read the same real of it at the same time, which is one access.
// Shared memory: 8 NT * K4 reals.
// -------------------------------------------------------------------------
template <typename Float, int NT>
__global__ void fifth_gemm_columns_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_from,
    const Float* __restrict__ d_m,
    int blocks,
    int Ls)
{
    extern __shared__ unsigned char shared_bytes[];
    Float* m_shared = reinterpret_cast<Float*>(shared_bytes);
    const int rows = 2 * Ls;
    const int k4 = (rows + 3) / 4 * 4;
    for (int i = threadIdx.x; i < 8 * NT * k4; i += blockDim.x) m_shared[i] = d_m[i];
    __syncthreads();

    const size_t columns = static_cast<size_t>(blocks) * FifthColumns;
    for (size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; column < columns;
         column += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t base = column / FifthColumns * rows * FifthColumns + column % FifthColumns;
        Float acc[8 * NT];
        #pragma unroll
        for (int r = 0; r < 8 * NT; ++r) acc[r] = 0;
        for (int j = 0; j < rows; ++j) {
            const Float x = d_in[base + static_cast<size_t>(j) * FifthColumns];
            #pragma unroll
            for (int r = 0; r < 8 * NT; ++r) acc[r] += m_shared[r * k4 + j] * x;
        }
        #pragma unroll
        for (int r = 0; r < 8 * NT; ++r) {
            if (r < rows) {
                const size_t k = base + static_cast<size_t>(r) * FifthColumns;
                d_out[k] = (d_from != nullptr) ? d_from[k] - acc[r] : acc[r];
            }
        }
    }
}

// -------------------------------------------------------------------------
// The hops and the product in one pass: out = M H (b - c Y) in, or from - that, with
// H^dag for `sign` -1; out on the sites of parity target_parity and in on the others, or
// both on the whole lattice (target_parity < 0).
//
// A block of threads has `lanes` = blockDim.x sites and their Ls spinors, a thread for each
// (site, s) as in hop_kernel. The threads sum their hops in their registers and put the
// result into the shared memory of the block, as the 2 Ls rows of 12 lanes reals that the
// product takes; nothing of it passes through the memory of the device. Then
//
// - in double precision the warps of the block take the strips of 8 columns of these rows in
//   turn and multiply as fifth_gemm_mma_kernel does, on the tensor cores;
// - in single precision every thread sums the two rows of its s over the rows of its site.
//
// The rows in the shared memory: real (row, column) is at column ^ (8 (row & 1)) of its row
// (12 lanes reals, a multiple of 16). A warp reads 4 rows of 8 columns for an mma, and the
// rows one under the other would fall on the same banks, four threads to a bank; with odd
// rows moved by 8 it is two, the least for 8 bytes each. The same holds for the threads that
// write: a warp of them writes rows of one parity.
//
// The vectors of the fields: blockIdx.y.
//
// lanes: a multiple of 4, and lanes * Ls a multiple of 32 in double precision (whole warps).
// Shared memory: 2 Ls * 12 lanes reals, then the matrix as the kernel of the precision keeps
// it (8 NT * K16 doubles, or 8 NT * K4 floats).
// -------------------------------------------------------------------------
template <typename Float, int NT>
__global__ void hop_gemm_kernel(
    Float* d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    const Float* d_from,
    const Float* __restrict__ d_m,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls,
    int target_parity,
    Float sign,
    Float mass, Float b5, Float c5)
{
    extern __shared__ unsigned char shared_bytes[];
    Float* tile = reinterpret_cast<Float*>(shared_bytes);
    const int lanes = blockDim.x;
    const int rows = 2 * Ls, width = 12 * lanes;
    const int k4 = (rows + 3) / 4 * 4;
    const int k16 = (rows + 15) / 16 * 16;
    Float* m_shared = tile + rows * width;
    const int tid = threadIdx.y * lanes + threadIdx.x;
    const bool tensor_cores = (sizeof(Float) == 8);
    for (int i = tid; i < 8 * NT * k4; i += lanes * Ls) {
        const int row = i / k4, k = i % k4;
        m_shared[tensor_cores ? row * k16 + (k ^ (4 * (row % 4))) : i] = d_m[i];
    }

    const int lane = threadIdx.x, is = threadIdx.y;
    const bool half = (target_parity >= 0);
    const int sites = (half ? nx / 2 : nx) * ny * nz * nt;
    const int at = blockIdx.x * lanes + lane;
    // No thread leaves before the barrier, and none of a warp before its mma.
    const bool live = (at < sites);
    const size_t first = (gridDim.y == 1) ? 0 : blockIdx.y * vector_reals(sites, Ls);
    d_out += first;
    d_in += first;
    if (d_from != nullptr) d_from += first;

    {
        Float acc[Nspin][Nc][Ncomplex] = {0};
        if (live) {
            const int dims[4] = {nx, ny, nz, nt};
            const Float bc[4] = {bcx, bcy, bcz, bct};
            int xc[4];
            if (half) {
                half_site_to_coords(at, target_parity, xc, nx, ny, nz);
            } else {
                site_to_coords(at, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
            }
            hops<Float>(acc, d_in, d_gauge, link_rows, xc, is, Ls, dims, bc, half, sign,
                        static_cast<Float>(0.5), mass, b5, c5);
        }
        // The upper two spin components are row 2 s, the lower two row 2 s + 1.
        #pragma unroll
        for (int k = 0; k < Nspin * Nc * Ncomplex; ++k) {
            const int row = 2 * is + k / 12, column = (k % 12) * lanes + lane;
            tile[row * width + (column ^ (8 * (row & 1)))] = (&acc[0][0][0])[k];
        }
    }
    __syncthreads();

    // Where real `column` of row `row` of the block goes in the fields.
    auto in_field = [&](int row, int column) {
        const int site = blockIdx.x * lanes + column % lanes;
        const int comp = column / lanes + 12 * (row % 2);
        return ((site / WarpSize) * Ls + row / 2) * (Nspin * Nc * Ncomplex) * WarpSize
               + comp * WarpSize + site % WarpSize;
    };

    if constexpr (sizeof(Float) == 8) {
        const int warp = tid / 32, t = tid % 32, warps = lanes * Ls / 32;
        for (int strip = warp; strip < width / 8; strip += warps) {
            const int n0 = strip * 8;
            double d[NT][2];
            #pragma unroll
            for (int it = 0; it < NT; ++it) { d[it][0] = 0.0; d[it][1] = 0.0; }
            for (int kk = 0; kk < k4; kk += 4) {
                const int j = kk + t % 4;
                const int j_in = (j < rows) ? j : rows - 1;
                const double read = tile[j_in * width + ((n0 + t / 4) ^ (8 * (j_in & 1)))];
                const double fb = (j < rows) ? read : 0.0;
                #pragma unroll
                for (int it = 0; it < NT; ++it) {
                    const int row = it * 8 + t / 4;
                    const double fa = m_shared[row * k16 + (j ^ (4 * (row % 4)))];
                    const double c0 = d[it][0], c1 = d[it][1];
                    double d0, d1;
                    asm volatile(
                        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
                        : "=d"(d0), "=d"(d1)
                        : "d"(fa), "d"(fb), "d"(c0), "d"(c1));
                    d[it][0] = d0;
                    d[it][1] = d1;
                }
            }
            #pragma unroll
            for (int it = 0; it < NT; ++it) {
                const int row = it * 8 + t / 4;
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int column = n0 + (t % 4) * 2 + i;
                    if (row < rows && blockIdx.x * lanes + column % lanes < sites) {
                        const int k = in_field(row, column);
                        d_out[k] = (d_from != nullptr) ? d_from[k] - d[it][i] : d[it][i];
                    }
                }
            }
        }
    } else {
        Float upper[12], lower[12];
        #pragma unroll
        for (int c = 0; c < 12; ++c) { upper[c] = 0; lower[c] = 0; }
        for (int j = 0; j < rows; ++j) {
            const Float m_upper = m_shared[(2 * is) * k4 + j], m_lower = m_shared[(2 * is + 1) * k4 + j];
            #pragma unroll
            for (int c = 0; c < 12; ++c) {
                const Float x = tile[j * width + ((c * lanes + lane) ^ (8 * (j & 1)))];
                upper[c] += m_upper * x;
                lower[c] += m_lower * x;
            }
        }
        if (live) {
            #pragma unroll
            for (int c = 0; c < 12; ++c) {
                const int ku = in_field(2 * is, c * lanes + lane), kl = in_field(2 * is + 1, c * lanes + lane);
                d_out[ku] = (d_from != nullptr) ? d_from[ku] - upper[c] : upper[c];
                d_out[kl] = (d_from != nullptr) ? d_from[kl] - lower[c] : lower[c];
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_FIFTH_GEMM_CUH
