#ifndef BRIDGE_CUDA_FERMION_FIFTH_GEMM_CUH
#define BRIDGE_CUDA_FERMION_FIFTH_GEMM_CUH

#include "../base/types.h"
#include "../base/constant.h"
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
// out = M in, or from - M in if d_from is not null. out must not be in.
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
    const int rows = 2 * Ls;
    const int k4 = (rows + 3) / 4 * 4;
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
            const double fa = d_m[(it * 8 + lane / 4) * k4 + j];
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
// The same product without the tensor cores, one thread for a real of the result: single
// precision (whose tensor cores are TF32, ten bits of mantissa: not for an operator).
// -------------------------------------------------------------------------
template <typename Float>
__global__ void fifth_gemm_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_from,
    const Float* __restrict__ d_m,
    int blocks,
    int Ls)
{
    const int rows = 2 * Ls;
    const int k4 = (rows + 3) / 4 * 4;
    const size_t total = static_cast<size_t>(blocks) * rows * FifthColumns;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t base = idx / (rows * FifthColumns) * (rows * FifthColumns);
        const int row = (idx - base) / FifthColumns, n = (idx - base) % FifthColumns;
        Float sum = 0;
        for (int j = 0; j < rows; ++j) {
            sum += d_m[row * k4 + j] * d_in[base + static_cast<size_t>(j) * FifthColumns + n];
        }
        d_out[idx] = (d_from != nullptr) ? d_from[idx] - sum : sum;
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_FIFTH_GEMM_CUH
