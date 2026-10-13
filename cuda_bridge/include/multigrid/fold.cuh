#ifndef BRIDGE_CUDA_MULTIGRID_FOLD_CUH
#define BRIDGE_CUDA_MULTIGRID_FOLD_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../fermion/hops.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// The prolongation P of the multigrid of domain-wall fermions and its adjoint.
//
// The aggregates keep the whole fifth dimension and are split by the chirality of
// Gamma_5 = gamma_5 R (R reverses the fifth dimension): with n test vectors a coarse site has
// 2 n complex numbers, number c n + i for test vector i and chirality c (0 positive). P is
// stored folded: of test vector i and chirality c at a fine site the spinors at s < Ls/2 only
// (12 Ls reals), those at Ls - 1 - s following from Gamma_5:
//
//   d_p[((x n + i) 2 + c) 12 Ls + 24 s + 2 (colour + 3 spin) + ri],   x a site of the whole lattice,
//
// in single precision, or in half precision (`Stored`: float or __half), which halves what is
// the largest thing a multigrid keeps; the coarse operator is built from P as it is stored, so
// the correction stays the Galerkin one. gamma_5 has one entry in a row:
// (gamma_5 w)_a = sign[a] w_(col[a]).
//
// A coarse field has complex number q = X dim + k of coarse site X at coarse_index(q) (real
// part) and that + W (imaginary part) of its vector: blocks of W real parts and W imaginary
// parts, as linalg/blas.cuh takes them; coarse_reals reals to a vector, the vectors one after
// the other. The fine fields are those of fermion/hops.cuh, on the whole lattice (parity < 0)
// or on the sites of one parity: then only those sites of an aggregate take part.

namespace bridge_cuda {

struct FoldGeometry {
    int nx, ny, nz, nt;
    int Ls;
    int parity;
    int col[Nspin];
    int sign[Nspin];
    // For a field that is zero but where leg `leg` of the hops (fermion/hops.cuh) comes from
    // another aggregate of `block` sites: only those sites are read. leg < 0: all sites.
    int leg;
    int block[4];
};

__device__ __forceinline__ float from_stored(float x) { return x; }
__device__ __forceinline__ float from_stored(__half x) { return __half2float(x); }
__device__ __forceinline__ void to_stored(float& slot, float x) { slot = x; }
__device__ __forceinline__ void to_stored(__half& slot, float x) { slot = __float2half(x); }

__host__ __device__ __forceinline__ size_t coarse_index(size_t q) {
    return (q / WarpSize) * 2 * WarpSize + q % WarpSize;
}

__host__ __device__ __forceinline__ size_t coarse_reals(int n_coarse, int dim) {
    return (static_cast<size_t>(n_coarse) * dim + WarpSize - 1) / WarpSize * 2 * WarpSize;
}

// Reading P is most of what the two kernels below cost: a thread reads its part of P once
// for up to FoldVectors vectors of the fields, blockIdx.y the group of vectors.
constexpr int FoldVectors = 8;

// -------------------------------------------------------------------------
// coarse = P^dag fine. A thread for each (coarse site, component).
// d_sites: the block_volume sites (of the whole lattice) of every coarse site.
// A field of g.leg takes the sites it is not zero on.
// -------------------------------------------------------------------------
template <typename Float, typename Stored>
__global__ void fold_restrict_kernel(
    Float* __restrict__ d_coarse,
    const Float* __restrict__ d_fine,
    const Stored* __restrict__ d_p,
    const unsigned* __restrict__ d_sites,
    int n_coarse, int block_volume, int n, int vectors,
    FoldGeometry g)
{
    const int dim = 2 * n;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_coarse * dim) return;
    const int X = t / dim, k = t % dim, c = k / n, i = k % n;
    const int Ls = g.Ls, kept = 12 * Ls;
    const bool half = (g.parity >= 0);
    const int sites = (half ? g.nx / 2 : g.nx) * g.ny * g.nz * g.nt;
    const int first = blockIdx.y * FoldVectors;
    const int count = (vectors - first < FoldVectors) ? vectors - first : FoldVectors;
    const size_t fine_reals = vector_reals(sites, Ls);
    const Float* fine = d_fine + first * fine_reals;
    const Float chirality = (c == 0) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);

    Float zr[FoldVectors], zi[FoldVectors];
    #pragma unroll
    for (int v = 0; v < FoldVectors; ++v) {
        zr[v] = 0;
        zi[v] = 0;
    }
    for (int j = 0; j < block_volume; ++j) {
        const int x = d_sites[static_cast<size_t>(X) * block_volume + j];
        int at = x;
        if (half || g.leg >= 0) {
            int xc[4];
            site_to_coords(x, xc[0], xc[1], xc[2], xc[3], g.nx, g.ny, g.nz, g.nt);
            if (half) {
                if (((xc[0] + xc[1] + xc[2] + xc[3]) & 1) != g.parity) continue;
                at = xc[0] / 2 + (g.nx / 2) * (x / g.nx);
            }
            if (g.leg >= 0) {
                const int mu = g.leg % 4, in_block = xc[mu] % g.block[mu];
                if (in_block != (g.leg < 4 ? g.block[mu] - 1 : 0)) continue;
            }
        }
        const Stored* b = d_p + ((static_cast<size_t>(x) * n + i) * 2 + c) * kept;
        for (int s = 0; s < Ls / 2; ++s) {
            #pragma unroll
            for (int a = 0; a < Nspin; ++a) {
                const Float sign = chirality * g.sign[a];
                #pragma unroll
                for (int colour = 0; colour < Nc; ++colour) {
                    const int e = 24 * s + 2 * (colour + Nc * a);
                    const Float br = from_stored(b[e]), bi = from_stored(b[e + 1]);
                    const int near_r = aosoa_5d_spinor_idx(colour, a, 0, s, Ls, at);
                    const int near_i = aosoa_5d_spinor_idx(colour, a, 1, s, Ls, at);
                    const int far_r = aosoa_5d_spinor_idx(colour, g.col[a], 0, Ls - 1 - s, Ls, at);
                    const int far_i = aosoa_5d_spinor_idx(colour, g.col[a], 1, Ls - 1 - s, Ls, at);
                    for (int v = 0; v < count; ++v) {
                        // f_s + or - gamma_5 f_(Ls-1-s)
                        const Float* f = fine + v * fine_reals;
                        const Float fr = f[near_r] + sign * f[far_r];
                        const Float fi = f[near_i] + sign * f[far_i];
                        zr[v] += br * fr + bi * fi;
                        zi[v] += br * fi - bi * fr;
                    }
                }
            }
        }
    }
    const size_t q = coarse_index(static_cast<size_t>(X) * dim + k);
    for (int v = 0; v < count; ++v) {
        Float* coarse = d_coarse + (first + v) * coarse_reals(n_coarse, dim);
        coarse[q] = zr[v];
        coarse[q + WarpSize] = zi[v];
    }
}

// -------------------------------------------------------------------------
// fine = P coarse. A thread for each (fine site, s < Ls/2, spin, colour), which writes the
// spinor at s and its mirror at Ls - 1 - s.
// d_coarse_of: the coarse site of every site of the whole lattice.
// -------------------------------------------------------------------------
template <typename Float, typename Stored>
__global__ void fold_prolong_kernel(
    Float* __restrict__ d_fine,
    const Float* __restrict__ d_coarse,
    const Stored* __restrict__ d_p,
    const unsigned* __restrict__ d_coarse_of,
    int n_coarse, int n, int vectors,
    FoldGeometry g)
{
    const int dim = 2 * n;
    const int Ls = g.Ls, kept = 12 * Ls, entries = 6 * Ls;
    const bool half = (g.parity >= 0);
    const int sites = (half ? g.nx / 2 : g.nx) * g.ny * g.nz * g.nt;
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(sites) * entries) return;
    const int at = t / entries, e = t % entries;
    int x = at;
    if (half) {
        int xc[4];
        half_site_to_coords(at, g.parity, xc, g.nx, g.ny, g.nz);
        x = coords_to_site(xc[0], xc[1], xc[2], xc[3], g.nx, g.ny, g.nz, g.nt);
    }
    const int first_vector = blockIdx.y * FoldVectors;
    const int count = (vectors - first_vector < FoldVectors) ? vectors - first_vector : FoldVectors;
    const size_t reals = coarse_reals(n_coarse, dim);
    const Float* coarse = d_coarse + first_vector * reals;
    const size_t first = static_cast<size_t>(d_coarse_of[x]) * dim;

    // a_c = sum_i coefficient(c, i) b(i, c), for the two chiralities.
    Float ar[2][FoldVectors], ai[2][FoldVectors];
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int v = 0; v < FoldVectors; ++v) {
            ar[c][v] = 0;
            ai[c][v] = 0;
        }
        for (int i = 0; i < n; ++i) {
            const size_t q = coarse_index(first + c * n + i);
            const Stored* b = d_p + ((static_cast<size_t>(x) * n + i) * 2 + c) * kept + 2 * e;
            const Float br = from_stored(b[0]), bi = from_stored(b[1]);
            for (int v = 0; v < count; ++v) {
                const Float cr = coarse[v * reals + q], ci = coarse[v * reals + q + WarpSize];
                ar[c][v] += cr * br - ci * bi;
                ai[c][v] += cr * bi + ci * br;
            }
        }
    }
    const int s = e / 12, a = (e % 12) / Nc, colour = e % Nc;
    // The mirror: gamma_5 (a_+ - a_-), whose row `to` takes this spin.
    int to = 0;
    #pragma unroll
    for (int row = 0; row < Nspin; ++row) {
        if (g.col[row] == a) to = row;
    }
    const Float sign = g.sign[to];
    for (int v = 0; v < count; ++v) {
        Float* fine = d_fine + (first_vector + v) * vector_reals(sites, Ls);
        fine[aosoa_5d_spinor_idx(colour, a, 0, s, Ls, at)] = ar[0][v] + ar[1][v];
        fine[aosoa_5d_spinor_idx(colour, a, 1, s, Ls, at)] = ai[0][v] + ai[1][v];
        fine[aosoa_5d_spinor_idx(colour, to, 0, Ls - 1 - s, Ls, at)] = sign * (ar[0][v] - ar[1][v]);
        fine[aosoa_5d_spinor_idx(colour, to, 1, Ls - 1 - s, Ls, at)] = sign * (ai[0][v] - ai[1][v]);
    }
}

// -------------------------------------------------------------------------
// Making P. First the chirality parts (1 +- Gamma_5) v / 2 of the test vectors, as they are
// kept: vector blockIdx.y of the field d_fine on the whole lattice is test vector
// first + blockIdx.y. A thread for each (site, s < Ls/2, spin, colour).
// -------------------------------------------------------------------------
template <typename Float, typename Stored>
__global__ void fold_fill_kernel(
    Stored* __restrict__ d_p,
    const Float* __restrict__ d_fine,
    int n, int first,
    FoldGeometry g)
{
    const int Ls = g.Ls, kept = 12 * Ls, entries = 6 * Ls;
    const int sites = g.nx * g.ny * g.nz * g.nt;
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(sites) * entries) return;
    const int x = t / entries, e = t % entries;
    const int s = e / 12, a = (e % 12) / Nc, colour = e % Nc;
    const Float* fine = d_fine + blockIdx.y * vector_reals(sites, Ls);
    #pragma unroll
    for (int ri = 0; ri < Ncomplex; ++ri) {
        // (Gamma_5 v)_s = gamma_5 v_(Ls-1-s)
        const Float near = fine[aosoa_5d_spinor_idx(colour, a, ri, s, Ls, x)];
        const Float far = g.sign[a] * fine[aosoa_5d_spinor_idx(colour, g.col[a], ri, Ls - 1 - s, Ls, x)];
        Stored* b = d_p + (static_cast<size_t>(x) * n + first + blockIdx.y) * 2 * kept + 2 * e + ri;
        to_stored(b[0], static_cast<float>(static_cast<Float>(0.5) * (near + far)));
        to_stored(b[kept], static_cast<float>(static_cast<Float>(0.5) * (near - far)));
    }
}

// -------------------------------------------------------------------------
// Then the parts of either chirality made orthonormal over every aggregate, by the modified
// Gram-Schmidt procedure in double precision, and scaled (P keeps half of the fifth
// dimension, which carries half of every inner product: scale = 1/sqrt(2)).
//
// A block of FoldThreads threads for each (coarse site first_site + blockIdx.x / 2,
// chirality blockIdx.x % 2). d_w: the vectors of the block in double precision, n vectors of
// block_volume * kept / 2 complex numbers for every block of threads. d_fail[blockIdx.x]: 0,
// or 1 + the first vector that is in the span of the earlier ones (the block then stops).
// -------------------------------------------------------------------------
constexpr int FoldThreads = 256;

template <typename Stored>
__global__ void fold_orthonormalise_kernel(
    Stored* __restrict__ d_p,
    double* __restrict__ d_w,
    const unsigned* __restrict__ d_sites,
    unsigned* __restrict__ d_fail,
    int first_site, int block_volume, int n, int kept,
    double scale)
{
    __shared__ double sum_r[FoldThreads];
    __shared__ double sum_i[FoldThreads];
    const int tid = threadIdx.x;
    const int X = first_site + blockIdx.x / 2, c = blockIdx.x % 2;
    const int per_site = kept / 2;
    const int dim = block_volume * per_site;
    double* w = d_w + static_cast<size_t>(blockIdx.x) * n * dim * 2;
    // Where complex number e of vector i of this block is in P.
    auto in_p = [&](int i, int e) {
        const size_t x = d_sites[static_cast<size_t>(X) * block_volume + e / per_site];
        return ((x * n + i) * 2 + c) * kept + 2 * (e % per_site);
    };
    // The sum over the threads of (re, im), for all of them.
    auto total = [&](double& re, double& im) {
        sum_r[tid] = re;
        sum_i[tid] = im;
        __syncthreads();
        #pragma unroll
        for (int step = FoldThreads / 2; step > 0; step >>= 1) {
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
    auto norm = [&](const double* v) {
        double re = 0, im = 0;
        for (int e = tid; e < dim; e += FoldThreads) re += v[2 * e] * v[2 * e] + v[2 * e + 1] * v[2 * e + 1];
        total(re, im);
        return re;
    };

    for (int k = tid; k < n * dim; k += FoldThreads) {
        const size_t at = in_p(k / dim, k % dim);
        w[2 * k] = from_stored(d_p[at]);
        w[2 * k + 1] = from_stored(d_p[at + 1]);
    }
    if (tid == 0) d_fail[blockIdx.x] = 0;
    __syncthreads();

    for (int i = 0; i < n; ++i) {
        double* wi = w + static_cast<size_t>(i) * dim * 2;
        const double before = norm(wi);
        for (int j = 0; j < i; ++j) {
            const double* wj = w + static_cast<size_t>(j) * dim * 2;
            double re = 0, im = 0;
            for (int e = tid; e < dim; e += FoldThreads) {
                re += wj[2 * e] * wi[2 * e] + wj[2 * e + 1] * wi[2 * e + 1];
                im += wj[2 * e] * wi[2 * e + 1] - wj[2 * e + 1] * wi[2 * e];
            }
            total(re, im);
            for (int e = tid; e < dim; e += FoldThreads) {
                wi[2 * e] -= re * wj[2 * e] - im * wj[2 * e + 1];
                wi[2 * e + 1] -= re * wj[2 * e + 1] + im * wj[2 * e];
            }
            __syncthreads();
        }
        const double after = norm(wi);
        if (!(after > 1e-24 * before)) {
            if (tid == 0) d_fail[blockIdx.x] = i + 1;
            return;
        }
        const double factor = rsqrt(after);
        for (int e = tid; e < dim; e += FoldThreads) {
            wi[2 * e] *= factor;
            wi[2 * e + 1] *= factor;
        }
        __syncthreads();
    }

    for (int k = tid; k < n * dim; k += FoldThreads) {
        const size_t at = in_p(k / dim, k % dim);
        to_stored(d_p[at], static_cast<float>(scale * w[2 * k]));
        to_stored(d_p[at + 1], static_cast<float>(scale * w[2 * k + 1]));
    }
}

// -------------------------------------------------------------------------
// The coarse operator D_c = P^dag D P: at every coarse site X a block A(X) and a link
// B_dir(X) for each of the eight hops, dim x dim complex matrices, row-major (re, im):
//
//   d_diag[(X dim + row) dim + col],   d_links[((dir n_coarse + X) dim + row) dim + col],
//
// with d_neighbours[dir n_coarse + X] the site the hop dir of X comes from. Everything that is
// done with it is one pass over the coarse sites of one parity (the others are set to zero;
// parity < 0: all of them),
//
//   out(X) = [M(X)] ( base(X) + A(X) d(X) + sign sum_dir B_dir(X) h(X + dir) ),
//
// every part optional (a null pointer): the fields base, d (d_diag_of) and h (d_hops_of),
// and the matrices M (d_inverse: the inverses of the blocks, or any others). With `gamma5`,
// Gamma_5 (+1 on the first half of the components of a site, -1 on the second) acts on d and
// h and on the result: Gamma_5 D_c Gamma_5, the adjoint of D_c.
//
// A block of threads for each (coarse site, vector), a thread for each row; blockIdx.y the
// vector. out must be none of the fields. Shared memory: 2 dim reals.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void coarse_pass_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_base,
    const Float* __restrict__ d_diag_of,
    const Float* __restrict__ d_hops_of,
    const Float* __restrict__ d_diag,
    const Float* __restrict__ d_links,
    const Float* __restrict__ d_inverse,
    const unsigned* __restrict__ d_neighbours,
    const unsigned* __restrict__ d_odd,
    int n_coarse, int dim,
    int parity, Float sign, int gamma5)
{
    extern __shared__ unsigned char shared_bytes[];
    Float* sum = reinterpret_cast<Float*>(shared_bytes);
    const int X = blockIdx.x, row = threadIdx.x;
    const size_t first = blockIdx.y * coarse_reals(n_coarse, dim);
    const size_t matrix = static_cast<size_t>(dim) * dim;
    const size_t at = first + coarse_index(static_cast<size_t>(X) * dim + row);
    // The whole block of threads leaves, or none of it.
    if (parity >= 0 && static_cast<int>(d_odd[X]) != parity) {
        d_out[at] = 0;
        d_out[at + WarpSize] = 0;
        return;
    }

    // (m v)_row for the matrix m and the site Y of the field v.
    auto times = [&](const Float* m, const Float* v, int Y, Float& re, Float& im) {
        const Float* m_row = m + 2 * static_cast<size_t>(dim) * row;
        for (int col = 0; col < dim; ++col) {
            const size_t q = first + coarse_index(static_cast<size_t>(Y) * dim + col);
            const Float flip = (gamma5 && col >= dim / 2) ? static_cast<Float>(-1.0) : static_cast<Float>(1.0);
            const Float xr = flip * v[q], xi = flip * v[q + WarpSize];
            const Float a = m_row[2 * col], b = m_row[2 * col + 1];
            re += a * xr - b * xi;
            im += a * xi + b * xr;
        }
    };

    Float re = 0, im = 0;
    if (d_diag_of != nullptr) {
        times(d_diag + 2 * matrix * X, d_diag_of, X, re, im);
    }
    if (d_hops_of != nullptr) {
        Float hr = 0, hi = 0;
        for (int dir = 0; dir < 8; ++dir) {
            const size_t link = static_cast<size_t>(dir) * n_coarse + X;
            times(d_links + 2 * matrix * link, d_hops_of, d_neighbours[link], hr, hi);
        }
        re += sign * hr;
        im += sign * hi;
    }
    if (d_base != nullptr) {
        re += d_base[at];
        im += d_base[at + WarpSize];
    }
    if (d_inverse != nullptr) {
        sum[row] = re;
        sum[dim + row] = im;
        __syncthreads();
        re = 0;
        im = 0;
        const Float* m_row = d_inverse + 2 * (matrix * X + static_cast<size_t>(dim) * row);
        for (int col = 0; col < dim; ++col) {
            const Float a = m_row[2 * col], b = m_row[2 * col + 1];
            re += a * sum[col] - b * sum[dim + col];
            im += a * sum[dim + col] + b * sum[col];
        }
    }
    const Float flip = (gamma5 && row >= dim / 2) ? static_cast<Float>(-1.0) : static_cast<Float>(1.0);
    d_out[at] = flip * re;
    d_out[at + WarpSize] = flip * im;
}

// v = Gamma_5 v on the coarse lattice, every vector.
template <typename Float>
__global__ void coarse_gamma5_kernel(Float* __restrict__ d_v, int n_coarse, int dim)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_coarse * dim || q % dim < dim / 2) return;
    const size_t at = blockIdx.y * coarse_reals(n_coarse, dim) + coarse_index(q);
    d_v[at] = -d_v[at];
    d_v[at + WarpSize] = -d_v[at + WarpSize];
}

// -------------------------------------------------------------------------
// Dense matrices on the coarse sites, dim x dim complex numbers each, row-major (re, im), in
// double precision. A block of dim threads for each site (dim at most 1024).
//
// The inverses, by Gauss-Jordan elimination with a pivot in the column: thread r has row r of
// [A | 1] (d_scratch: 4 dim^2 reals a site), and in the steps along a row column r and
// dim + r. d_fail[site]: 1 where a pivot is zero.
// -------------------------------------------------------------------------
__global__ void blocks_inverse_kernel(
    double* __restrict__ d_out,
    const double* __restrict__ d_a,
    double* __restrict__ d_scratch,
    unsigned* __restrict__ d_fail,
    int dim)
{
    __shared__ double size[1024];
    __shared__ double pivot_value[2];
    __shared__ int pivot_row;
    const int r = threadIdx.x, wide = 2 * dim;
    const size_t site = blockIdx.x;
    double* m = d_scratch + site * dim * wide * 2;
    const double* a = d_a + site * dim * dim * 2;
    for (int c = 0; c < dim; ++c) {
        m[2 * (r * wide + c)] = a[2 * (r * dim + c)];
        m[2 * (r * wide + c) + 1] = a[2 * (r * dim + c) + 1];
        m[2 * (r * wide + dim + c)] = (r == c) ? 1.0 : 0.0;
        m[2 * (r * wide + dim + c) + 1] = 0.0;
    }
    if (r == 0) d_fail[site] = 0;
    __syncthreads();

    for (int k = 0; k < dim; ++k) {
        const double re = m[2 * (r * wide + k)], im = m[2 * (r * wide + k) + 1];
        size[r] = (r >= k) ? re * re + im * im : -1.0;
        __syncthreads();
        if (r == 0) {
            int best = k;
            for (int i = k + 1; i < dim; ++i) {
                if (size[i] > size[best]) best = i;
            }
            pivot_row = (size[best] > 0.0) ? best : -1;
        }
        __syncthreads();
        const int p = pivot_row;
        if (p < 0) {
            if (r == 0) d_fail[site] = 1;
            return;
        }
        // Rows k and p change places; thread r has columns r and dim + r of them.
        if (p != k) {
            #pragma unroll
            for (int c = r; c < wide; c += dim) {
                #pragma unroll
                for (int ri = 0; ri < 2; ++ri) {
                    const double t = m[2 * (k * wide + c) + ri];
                    m[2 * (k * wide + c) + ri] = m[2 * (p * wide + c) + ri];
                    m[2 * (p * wide + c) + ri] = t;
                }
            }
        }
        __syncthreads();
        if (r == 0) {
            pivot_value[0] = m[2 * (k * wide + k)];
            pivot_value[1] = m[2 * (k * wide + k) + 1];
        }
        __syncthreads();
        // Row k over the pivot.
        {
            const double pr = pivot_value[0], pi = pivot_value[1];
            const double inverse = 1.0 / (pr * pr + pi * pi);
            #pragma unroll
            for (int c = r; c < wide; c += dim) {
                const double x = m[2 * (k * wide + c)], y = m[2 * (k * wide + c) + 1];
                m[2 * (k * wide + c)] = (x * pr + y * pi) * inverse;
                m[2 * (k * wide + c) + 1] = (y * pr - x * pi) * inverse;
            }
        }
        __syncthreads();
        // The other rows: column k away.
        if (r != k) {
            const double fr = m[2 * (r * wide + k)], fi = m[2 * (r * wide + k) + 1];
            for (int c = 0; c < wide; ++c) {
                const double x = m[2 * (k * wide + c)], y = m[2 * (k * wide + c) + 1];
                m[2 * (r * wide + c)] -= fr * x - fi * y;
                m[2 * (r * wide + c) + 1] -= fr * y + fi * x;
            }
        }
        __syncthreads();
    }

    double* out = d_out + site * dim * dim * 2;
    for (int c = 0; c < dim; ++c) {
        out[2 * (r * dim + c)] = m[2 * (r * wide + dim + c)];
        out[2 * (r * dim + c) + 1] = m[2 * (r * wide + dim + c) + 1];
    }
}

// -------------------------------------------------------------------------
// |H|^-1 = V |Lambda|^-1 V^dag for H the Hermitian part of A Gamma_5 (Gamma_5: -1 on the
// second half of the columns), by the cyclic Jacobi method as numerics::hermitian_eigen has
// it on the CPU: for every pair (p, q), a phase on column and row q that makes a_pq real,
// then a rotation. Thread k has what the loops over k of that method do. d_scratch: H and V,
// 4 dim^2 reals a site.
// -------------------------------------------------------------------------
__global__ void blocks_abs_inverse_gamma5_kernel(
    double* __restrict__ d_out,
    const double* __restrict__ d_a,
    double* __restrict__ d_scratch,
    int dim)
{
    __shared__ double part[1024];
    __shared__ double sums[2];
    const int k = threadIdx.x, n = dim;
    const size_t site = blockIdx.x;
    double* h = d_scratch + site * n * n * 4;
    double* v = h + static_cast<size_t>(n) * n * 2;
    const double* a = d_a + site * n * n * 2;
    auto flip = [&](int col) { return (col >= n / 2) ? -1.0 : 1.0; };
    for (int c = 0; c < n; ++c) {
        // ((A Gamma_5) + (A Gamma_5)^dag) / 2
        h[2 * (k * n + c)] = 0.5 * (flip(c) * a[2 * (k * n + c)] + flip(k) * a[2 * (c * n + k)]);
        h[2 * (k * n + c) + 1] = 0.5 * (flip(c) * a[2 * (k * n + c) + 1] - flip(k) * a[2 * (c * n + k) + 1]);
        v[2 * (k * n + c)] = (k == c) ? 1.0 : 0.0;
        v[2 * (k * n + c) + 1] = 0.0;
    }
    __syncthreads();
    // The sum over the threads of two numbers, for all of them.
    auto total = [&](double x, double y, double& sx, double& sy) {
        part[k] = x;
        __syncthreads();
        if (k == 0) {
            double t = 0;
            for (int i = 0; i < n; ++i) t += part[i];
            sums[0] = t;
        }
        __syncthreads();
        part[k] = y;
        __syncthreads();
        if (k == 0) {
            double t = 0;
            for (int i = 0; i < n; ++i) t += part[i];
            sums[1] = t;
        }
        __syncthreads();
        sx = sums[0];
        sy = sums[1];
        __syncthreads();
    };
    double all = 0, off = 0;
    {
        double x = 0;
        for (int c = 0; c < n; ++c) x += h[2 * (k * n + c)] * h[2 * (k * n + c)] + h[2 * (k * n + c) + 1] * h[2 * (k * n + c) + 1];
        total(x, 0.0, all, off);
    }

    for (int sweep = 0; sweep < 100; ++sweep) {
        {
            double x = 0, unused;
            for (int c = k + 1; c < n; ++c) x += h[2 * (k * n + c)] * h[2 * (k * n + c)] + h[2 * (k * n + c) + 1] * h[2 * (k * n + c) + 1];
            total(x, 0.0, off, unused);
        }
        if (sqrt(off) <= 1e-15 * sqrt(all)) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double gr = h[2 * (p * n + q)], gi = h[2 * (p * n + q) + 1];
                const double g = sqrt(gr * gr + gi * gi);
                if (g <= 1e-300) continue;
                // d = conj(a_pq) / |a_pq|: column q of H and V times d, then row q of H times conj(d).
                const double dr = gr / g, di = -gi / g;
                {
                    double* x = h + 2 * (k * n + q);
                    const double xr = x[0], xi = x[1];
                    x[0] = xr * dr - xi * di;
                    x[1] = xr * di + xi * dr;
                    double* y = v + 2 * (k * n + q);
                    const double yr = y[0], yi = y[1];
                    y[0] = yr * dr - yi * di;
                    y[1] = yr * di + yi * dr;
                }
                __syncthreads();
                {
                    double* x = h + 2 * (q * n + k);
                    const double xr = x[0], xi = x[1];
                    x[0] = xr * dr + xi * di;
                    x[1] = xi * dr - xr * di;
                }
                __syncthreads();
                const double theta = (h[2 * (q * n + q)] - h[2 * (p * n + p)]) / (2.0 * g);
                const double t = ((theta >= 0.0) ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                __syncthreads();
                #pragma unroll
                for (int which = 0; which < 2; ++which) {
                    double* m = which == 0 ? h : v;
                    #pragma unroll
                    for (int ri = 0; ri < 2; ++ri) {
                        const double kp = m[2 * (k * n + p) + ri], kq = m[2 * (k * n + q) + ri];
                        m[2 * (k * n + p) + ri] = c * kp - s * kq;
                        m[2 * (k * n + q) + ri] = s * kp + c * kq;
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int ri = 0; ri < 2; ++ri) {
                    const double pk = h[2 * (p * n + k) + ri], qk = h[2 * (q * n + k) + ri];
                    h[2 * (p * n + k) + ri] = c * pk - s * qk;
                    h[2 * (q * n + k) + ri] = s * pk + c * qk;
                }
                __syncthreads();
                if (k == 0) {
                    h[2 * (p * n + q)] = 0.0;
                    h[2 * (p * n + q) + 1] = 0.0;
                    h[2 * (q * n + p)] = 0.0;
                    h[2 * (q * n + p) + 1] = 0.0;
                }
                __syncthreads();
            }
        }
    }

    // out_ik = sum_j v_ij conj(v_kj) / |lambda_j|, row i = k of the thread.
    double* out = d_out + site * n * n * 2;
    for (int col = 0; col < n; ++col) {
        double re = 0, im = 0;
        for (int j = 0; j < n; ++j) {
            const double lambda = fmax(fabs(h[2 * (j * n + j)]), 1e-300);
            const double xr = v[2 * (k * n + j)], xi = v[2 * (k * n + j) + 1];
            const double yr = v[2 * (col * n + j)], yi = v[2 * (col * n + j) + 1];
            re += (xr * yr + xi * yi) / lambda;
            im += (xi * yr - xr * yi) / lambda;
        }
        out[2 * (k * n + col)] = re;
        out[2 * (k * n + col) + 1] = im;
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_FOLD_CUH
