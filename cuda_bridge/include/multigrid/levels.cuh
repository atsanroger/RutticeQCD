#ifndef BRIDGE_CUDA_MULTIGRID_LEVELS_CUH
#define BRIDGE_CUDA_MULTIGRID_LEVELS_CUH

#include "../base/constant.h"
#include <cuda_runtime.h>

// The levels below the coarse lattice of a multigrid (the signed multilevel V-cycle of
// solver/preconditioner/multigrid/levels.rs): every level a lattice of super-blocks of the
// sites of the level above, with nq local vectors on each, two per test vector, one for each
// half of the components of a site (its chirality).
//
// The fields of every level are coarse fields (fold.cuh): complex number q = X dim + k at
// level_at(q) (coarse_index there) and that + W of its vector. The local vectors, Q, are complex numbers
// (re, im) one after the other:
//
//   d_q[((S nq + b) bs + k) pd + i],
//
// for super-block S, local vector b, the k-th of its bs sites (d_members[S bs + k], a site of
// the level above, in the order of the global index there) and component i of the pd of a site
// above; d_owner[x] = S bs + k for the site x above.

namespace bridge_cuda {

// coarse_index and coarse_reals of fold.cuh, which is not included here: its kernels are
// defined in fold.cu.
__host__ __device__ __forceinline__ size_t level_at(size_t q) {
    return (q / WarpSize) * 2 * WarpSize + q % WarpSize;
}

__host__ __device__ __forceinline__ size_t level_reals(int n_sites, int dim) {
    return (static_cast<size_t>(n_sites) * dim + WarpSize - 1) / WarpSize * 2 * WarpSize;
}

constexpr int LevelThreads = 256;

// -------------------------------------------------------------------------
// out = Q^dag v: a thread for each (super-block, local vector), blockIdx.y the vector.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void level_restrict_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_v,
    const Float* __restrict__ d_q,
    const unsigned* __restrict__ d_members,
    int n_blocks, int nq, int bs, int n_parent, int pd)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_blocks * nq) return;
    const int S = t / nq;
    const Float* v = d_v + blockIdx.y * level_reals(n_parent, pd);
    const Float* q = d_q + 2 * static_cast<size_t>(t) * bs * pd;
    Float re = 0, im = 0;
    for (int k = 0; k < bs; ++k) {
        const size_t x = d_members[static_cast<size_t>(S) * bs + k];
        for (int i = 0; i < pd; ++i) {
            const Float a = q[2 * (k * pd + i)], b = q[2 * (k * pd + i) + 1];
            const size_t at = level_at(x * pd + i);
            const Float xr = v[at], xi = v[at + WarpSize];
            re += a * xr + b * xi;
            im += a * xi - b * xr;
        }
    }
    const size_t at = blockIdx.y * level_reals(n_blocks, nq) + level_at(t);
    d_out[at] = re;
    d_out[at + WarpSize] = im;
}

// -------------------------------------------------------------------------
// out = Q y, or out += Q y with `add`: a thread for each (site above, component), blockIdx.y
// the vector.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void level_prolong_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_y,
    const Float* __restrict__ d_q,
    const unsigned* __restrict__ d_owner,
    int n_blocks, int nq, int bs, int n_parent, int pd, int add)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_parent * pd) return;
    const int x = p / pd, i = p % pd;
    const unsigned o = d_owner[x];
    const int S = o / bs, k = o % bs;
    const Float* y = d_y + blockIdx.y * level_reals(n_blocks, nq);
    Float re = 0, im = 0;
    for (int b = 0; b < nq; ++b) {
        const Float* q = d_q + 2 * ((static_cast<size_t>(S) * nq + b) * bs * pd + k * pd + i);
        const size_t at = level_at(static_cast<size_t>(S) * nq + b);
        const Float yr = y[at], yi = y[at + WarpSize];
        re += q[0] * yr - q[1] * yi;
        im += q[0] * yi + q[1] * yr;
    }
    const size_t at = blockIdx.y * level_reals(n_parent, pd) + level_at(p);
    if (add) {
        d_out[at] += re;
        d_out[at + WarpSize] += im;
    } else {
        d_out[at] = re;
        d_out[at + WarpSize] = im;
    }
}

// -------------------------------------------------------------------------
// Q from n_t test vectors on the level above (d_tests, n_t vectors of a coarse field there), in
// double precision: a block of LevelThreads threads for each (super-block, half h); local vector
// h n_t + j is the part of test vector j on the sites of the super-block and on the components
// of half h, made orthogonal to the earlier ones of that half by Gram-Schmidt twice and
// normalised, or zero if its norm is not above 1e-8 then (as levels.rs).
// -------------------------------------------------------------------------
template <typename Float>
__global__ void level_basis_kernel(
    double* __restrict__ d_q,
    const Float* __restrict__ d_tests,
    const unsigned* __restrict__ d_members,
    int n_t, int bs, int n_parent, int pd)
{
    __shared__ double sum_r[LevelThreads];
    __shared__ double sum_i[LevelThreads];
    const int tid = threadIdx.x;
    const int S = blockIdx.x / 2, h = blockIdx.x % 2;
    const int nq = 2 * n_t, len = bs * pd;
    auto column = [&](int b) { return d_q + 2 * (static_cast<size_t>(S) * nq + b) * len; };
    auto total = [&](double& re, double& im) {
        sum_r[tid] = re;
        sum_i[tid] = im;
        __syncthreads();
        #pragma unroll
        for (int step = LevelThreads / 2; step > 0; step >>= 1) {
            if (tid < step) {
                sum_r[tid] += sum_r[tid + step];
                sum_i[tid] += sum_i[tid + step];
            }
            __syncthreads();
        }
        re = sum_r[0];
        im = sum_i[0];
        __syncthreads();
    };
    for (int j = 0; j < n_t; ++j) {
        double* c = column(h * n_t + j);
        const Float* t = d_tests + j * level_reals(n_parent, pd);
        for (int e = tid; e < len; e += LevelThreads) {
            const int k = e / pd, i = e % pd;
            double re = 0, im = 0;
            if ((i >= pd / 2) == (h == 1)) {
                const size_t at = level_at(static_cast<size_t>(d_members[static_cast<size_t>(S) * bs + k]) * pd + i);
                re = t[at];
                im = t[at + WarpSize];
            }
            c[2 * e] = re;
            c[2 * e + 1] = im;
        }
        __syncthreads();
        for (int pass = 0; pass < 2; ++pass) {
            for (int b = 0; b < j; ++b) {
                const double* w = column(h * n_t + b);
                double re = 0, im = 0;
                for (int e = tid; e < len; e += LevelThreads) {
                    re += w[2 * e] * c[2 * e] + w[2 * e + 1] * c[2 * e + 1];
                    im += w[2 * e] * c[2 * e + 1] - w[2 * e + 1] * c[2 * e];
                }
                total(re, im);
                for (int e = tid; e < len; e += LevelThreads) {
                    c[2 * e] -= re * w[2 * e] - im * w[2 * e + 1];
                    c[2 * e + 1] -= re * w[2 * e + 1] + im * w[2 * e];
                }
                __syncthreads();
            }
        }
        double nn = 0, unused = 0;
        for (int e = tid; e < len; e += LevelThreads) nn += c[2 * e] * c[2 * e] + c[2 * e + 1] * c[2 * e + 1];
        total(nn, unused);
        const double n = sqrt(nn);
        const double scale = (n > 1e-8) ? 1.0 / n : 0.0;
        for (int e = tid; e < len; e += LevelThreads) {
            c[2 * e] *= scale;
            c[2 * e + 1] *= scale;
        }
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// The sources of colour `colour` of the probing of a level's operator: vector j is 1 in
// component j of the sites of that colour (d_colour) and zero elsewhere.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void level_sources_kernel(
    Float* __restrict__ d_e, const unsigned* __restrict__ d_colour, unsigned colour,
    int n_sites, int nq)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_sites * nq) return;
    const size_t at = blockIdx.y * level_reals(n_sites, nq) + level_at(q);
    const int t = q / nq, i = q % nq;
    d_e[at] = (d_colour[t] == colour && i == static_cast<int>(blockIdx.y)) ? 1 : 0;
    d_e[at + WarpSize] = 0;
}

// -------------------------------------------------------------------------
// The blocks of a level's operator from the response d_r (nq vectors, vector j that to the
// sources of `colour`, level_sources_kernel), in double precision, in the layout of
// coarse_pass_kernel: the block of site t if t has the colour, and its link in direction dir
// (bit dir of `used`) if the site it comes from (d_neighbours[dir n + t]) has it. A thread for
// each (site, row), blockIdx.y the column.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void level_probe_kernel(
    double* __restrict__ d_diag, double* __restrict__ d_links,
    const Float* __restrict__ d_r, const unsigned* __restrict__ d_colour,
    const unsigned* __restrict__ d_neighbours, unsigned colour, unsigned used,
    int n_sites, int nq)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_sites * nq) return;
    const int t = q / nq, i = q % nq, j = blockIdx.y;
    const size_t at = j * level_reals(n_sites, nq) + level_at(q);
    const double re = d_r[at], im = d_r[at + WarpSize];
    const size_t entry = (static_cast<size_t>(i) * nq + j) * 2;
    const size_t matrix = static_cast<size_t>(nq) * nq * 2;
    if (d_colour[t] == colour) {
        d_diag[matrix * t + entry] = re;
        d_diag[matrix * t + entry + 1] = im;
    }
    for (int dir = 0; dir < 8; ++dir) {
        if (!((used >> dir) & 1u)) continue;
        const size_t link = static_cast<size_t>(dir) * n_sites + t;
        if (d_colour[d_neighbours[link]] == colour) {
            d_links[matrix * link + entry] = re;
            d_links[matrix * link + entry + 1] = im;
        }
    }
}

// -------------------------------------------------------------------------
// The operator of the last level as one dense n x n matrix, n = n_sites nq, row-major
// (re, im), in double precision: a thread for each row.
// -------------------------------------------------------------------------
__global__ void level_dense_kernel(
    double* __restrict__ d_dense, const double* __restrict__ d_diag,
    const double* __restrict__ d_links, const unsigned* __restrict__ d_neighbours,
    unsigned used, int n_sites, int nq)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_sites * nq;
    if (row >= n) return;
    const int t = row / nq, i = row % nq;
    const size_t matrix = static_cast<size_t>(nq) * nq * 2;
    double* out = d_dense + 2 * static_cast<size_t>(row) * n;
    for (int c = 0; c < 2 * n; ++c) out[c] = 0;
    auto add = [&](const double* m, int s) {
        for (int j = 0; j < nq; ++j) {
            out[2 * (s * nq + j)] += m[2 * (i * nq + j)];
            out[2 * (s * nq + j) + 1] += m[2 * (i * nq + j) + 1];
        }
    };
    add(d_diag + matrix * t, t);
    for (int dir = 0; dir < 8; ++dir) {
        if (!((used >> dir) & 1u)) continue;
        const size_t link = static_cast<size_t>(dir) * n_sites + t;
        add(d_links + matrix * link, d_neighbours[link]);
    }
}

// -------------------------------------------------------------------------
// out = M r for a dense n x n matrix M, row-major (re, im), and fields of n complex numbers: a
// thread for each row, blockIdx.y the vector.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void dense_apply_kernel(
    Float* __restrict__ d_out, const Float* __restrict__ d_r, const Float* __restrict__ d_m, int n)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const Float* r = d_r + blockIdx.y * level_reals(1, n);
    const Float* m = d_m + 2 * static_cast<size_t>(row) * n;
    Float re = 0, im = 0;
    for (int c = 0; c < n; ++c) {
        const size_t at = level_at(c);
        const Float a = m[2 * c], b = m[2 * c + 1], xr = r[at], xi = r[at + WarpSize];
        re += a * xr - b * xi;
        im += a * xi + b * xr;
    }
    const size_t at = blockIdx.y * level_reals(1, n) + level_at(row);
    d_out[at] = re;
    d_out[at + WarpSize] = im;
}

} // namespace bridge_cuda

#endif
