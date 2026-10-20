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
// coarse = P^dag fine as fold_restrict_kernel, with P read coalesced: a block for each coarse
// site, `groups` = blockDim.x / dim of its fine sites at a time. The block reads the P of those
// sites (dim rows of `kept` numbers each, one after the other in d_p) in 16-byte words into
// shared memory, its rows fold_row_stride apart so that the threads of a warp, one row each,
// read different banks; and the fine spinors of the sites folded by chirality,
// f_s +- gamma_5 f_(Ls-1-s), which every thread of a group reads alike. Thread (group, k) adds
// row k against them; at the end the groups are added in order, so the result does not depend
// on the timing. (A thread for each output read its own row of P one number at a time, its
// neighbours 192 bytes away: on the A30 11% of the bandwidth, held up by the load units.)
//
// Shared memory: groups dim fold_row_stride Stored, then groups `per_site` 2 kept Float (the
// folded fine fields of up to `per_site` vectors), then the active sites (x and its index in
// the field) of the coarse site: fold_restrict_shared.
// -------------------------------------------------------------------------
template <typename Stored>
__host__ __device__ __forceinline__ int fold_row_stride(int kept)
{
    // An odd number of 4-byte words: kept is a multiple of 24.
    return sizeof(Stored) == 2 ? kept + 2 : kept + 1;
}

__host__ __device__ __forceinline__ size_t fold_align(size_t bytes) { return (bytes + 15) / 16 * 16; }

// The first part: P of `groups` sites, which at the end holds the sums of the groups.
template <typename Float, typename Stored>
__host__ __device__ __forceinline__ size_t fold_restrict_p_bytes(int groups, int dim, int kept)
{
    const size_t p = static_cast<size_t>(groups) * dim * fold_row_stride<Stored>(kept) * sizeof(Stored);
    const size_t sums = static_cast<size_t>(groups) * dim * FoldVectors * 2 * sizeof(Float);
    return fold_align(p > sums ? p : sums);
}

template <typename Float, typename Stored>
__host__ __device__ __forceinline__ size_t fold_restrict_shared(int groups, int dim, int kept, int per_site, int block_volume)
{
    return fold_restrict_p_bytes<Float, Stored>(groups, dim, kept) +
           fold_align(static_cast<size_t>(groups) * per_site * 2 * kept * sizeof(Float)) +
           static_cast<size_t>(2) * block_volume * sizeof(int);
}

template <typename Float, typename Stored>
__global__ void fold_restrict_tiled_kernel(
    Float* __restrict__ d_coarse,
    const Float* __restrict__ d_fine,
    const Stored* __restrict__ d_p,
    const unsigned* __restrict__ d_sites,
    int n_coarse, int block_volume, int n, int vectors, int per_site,
    FoldGeometry g)
{
    extern __shared__ __align__(16) unsigned char fold_shared[];
    const int dim = 2 * n, Ls = g.Ls, kept = 12 * Ls;
    const int groups = blockDim.x / dim, stride = fold_row_stride<Stored>(kept);
    const int tid = threadIdx.x, slot = tid / dim, k = tid % dim;
    const int X = blockIdx.x;
    const bool half = (g.parity >= 0);
    const int sites = (half ? g.nx / 2 : g.nx) * g.ny * g.nz * g.nt;
    const int first = blockIdx.y * FoldVectors;
    const int count = (vectors - first < FoldVectors) ? vectors - first : FoldVectors;
    const size_t fine_reals = vector_reals(sites, Ls);
    const Float* fine = d_fine + first * fine_reals;

    Stored* p_sh = reinterpret_cast<Stored*>(fold_shared);
    const size_t p_bytes = fold_restrict_p_bytes<Float, Stored>(groups, dim, kept);
    Float* f_sh = reinterpret_cast<Float*>(fold_shared + p_bytes);
    const size_t f_bytes = fold_align(static_cast<size_t>(groups) * per_site * 2 * kept * sizeof(Float));
    int* active_x = reinterpret_cast<int*>(fold_shared + p_bytes + f_bytes);
    int* active_at = active_x + block_volume;
    __shared__ int n_active;

    // The sites of the aggregate that take part, in their order: warp 0, 32 at a time.
    if (tid < WarpSize) {
        int found = 0;
        for (int base = 0; base < block_volume; base += WarpSize) {
            const int j = base + tid;
            bool take = false;
            int x = 0, at = 0;
            if (j < block_volume) {
                x = d_sites[static_cast<size_t>(X) * block_volume + j];
                at = x;
                take = true;
                if (half || g.leg >= 0) {
                    int xc[4];
                    site_to_coords(x, xc[0], xc[1], xc[2], xc[3], g.nx, g.ny, g.nz, g.nt);
                    if (half) {
                        take = ((xc[0] + xc[1] + xc[2] + xc[3]) & 1) == g.parity;
                        at = xc[0] / 2 + (g.nx / 2) * (x / g.nx);
                    }
                    if (take && g.leg >= 0) {
                        const int mu = g.leg % 4, in_block = xc[mu] % g.block[mu];
                        take = in_block == (g.leg < 4 ? g.block[mu] - 1 : 0);
                    }
                }
            }
            const unsigned mask = __ballot_sync(0xffffffffu, take);
            if (take) {
                const int to = found + __popc(mask & ((1u << tid) - 1u));
                active_x[to] = x;
                active_at[to] = at;
            }
            found += __popc(mask);
        }
        if (tid == 0) n_active = found;
    }
    __syncthreads();

    const int c = k / n, i = k % n, row = 2 * i + c;
    Float zr[FoldVectors], zi[FoldVectors];
    #pragma unroll
    for (int v = 0; v < FoldVectors; ++v) {
        zr[v] = 0;
        zi[v] = 0;
    }
    constexpr int per_word = 16 / sizeof(Stored);
    const int words = dim * kept / per_word;
    for (int a0 = 0; a0 < n_active; a0 += groups) {
        const int here = (n_active - a0 < groups) ? n_active - a0 : groups;
        // P of the sites, a 16-byte word at a time.
        for (int w = tid; w < here * words; w += blockDim.x) {
            const int s_slot = w / words, word = w % words;
            const size_t x = active_x[a0 + s_slot];
            const uint4 bits = reinterpret_cast<const uint4*>(d_p + x * dim * kept)[word];
            const int e0 = word * per_word, r = e0 / kept, col = e0 % kept;
            Stored* to = p_sh + (static_cast<size_t>(s_slot) * dim + r) * stride + col;
            if constexpr (sizeof(Stored) == 2) {
                // stride and col are even: four 4-byte words.
                unsigned* words4 = reinterpret_cast<unsigned*>(to);
                words4[0] = bits.x;
                words4[1] = bits.y;
                words4[2] = bits.z;
                words4[3] = bits.w;
            } else {
                const Stored* from = reinterpret_cast<const Stored*>(&bits);
                #pragma unroll
                for (int m = 0; m < per_word; ++m) to[m] = from[m];
            }
        }
        // The fine fields folded by chirality.
        for (int w = tid; w < here * count * kept; w += blockDim.x) {
            const int s_slot = w / (count * kept), rest = w % (count * kept);
            const int v = rest / kept, e = rest % kept;
            const int at = active_at[a0 + s_slot];
            const int s = e / 24, q = (e % 24) / 2, ri = e % 2, a = q / Nc, colour = q % Nc;
            const Float* f = fine + v * fine_reals;
            const Float near = f[aosoa_5d_spinor_idx(colour, a, ri, s, Ls, at)];
            const Float far = g.sign[a] * f[aosoa_5d_spinor_idx(colour, g.col[a], ri, Ls - 1 - s, Ls, at)];
            Float* to = f_sh + ((static_cast<size_t>(s_slot) * per_site + v) * 2) * kept + e;
            to[0] = near + far;
            to[kept] = near - far;
        }
        __syncthreads();
        if (slot < here) {
            const Stored* b = p_sh + (static_cast<size_t>(slot) * dim + row) * stride;
            const Float* fs = f_sh + (static_cast<size_t>(slot) * per_site * 2 + c) * kept;
            for (int e = 0; e < kept; e += 2) {
                const Float br = from_stored(b[e]), bi = from_stored(b[e + 1]);
                #pragma unroll
                for (int v = 0; v < FoldVectors; ++v) {
                    if (v < count) {
                        const Float fr = fs[v * 2 * kept + e], fi = fs[v * 2 * kept + e + 1];
                        zr[v] += br * fr + bi * fi;
                        zi[v] += br * fi - bi * fr;
                    }
                }
            }
        }
        __syncthreads();
    }

    // The groups added in order, through the space of P.
    Float* part = reinterpret_cast<Float*>(fold_shared);
    if (slot < groups) {
        #pragma unroll
        for (int v = 0; v < FoldVectors; ++v) {
            if (v < count) {
                part[((static_cast<size_t>(slot) * dim + k) * FoldVectors + v) * 2] = zr[v];
                part[((static_cast<size_t>(slot) * dim + k) * FoldVectors + v) * 2 + 1] = zi[v];
            }
        }
    }
    __syncthreads();
    if (slot == 0) {
        const size_t q = coarse_index(static_cast<size_t>(X) * dim + k);
        for (int v = 0; v < count; ++v) {
            Float re = 0, im = 0;
            for (int s_slot = 0; s_slot < groups; ++s_slot) {
                re += part[((static_cast<size_t>(s_slot) * dim + k) * FoldVectors + v) * 2];
                im += part[((static_cast<size_t>(s_slot) * dim + k) * FoldVectors + v) * 2 + 1];
            }
            Float* coarse = d_coarse + (first + v) * coarse_reals(n_coarse, dim);
            coarse[q] = re;
            coarse[q + WarpSize] = im;
        }
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
// A block of threads for each (coarse site, vector), blockIdx.y the vector, of
// coarse_pass_warps(dim) warps: a warp for each row (warp w the rows w, w + warps, ...), its
// lanes the pairs of columns (2 lane, 2 lane + 1), (2 lane + 64, ...), ..., so that a warp reads
// a row of a matrix as one stretch of memory in 16-byte words (a thread for each row read the
// rows 2 dim reals apart: about a third of the bandwidth on 2592 sites of dim 64). The fields
// at X and its eight neighbours, the same for every row, are read once into shared memory
// (Gamma_5 applied); the eight links of a row are read before they are used. The sum of a row
// is the lanes' parts added in a fixed order (the butterfly of coarse_pass_sum), so the result
// does not depend on timing. out must be none of the fields. dim is even.
// Shared memory: coarse_pass_shared(dim) bytes.
// -------------------------------------------------------------------------

template <typename Float>
__host__ __device__ __forceinline__ size_t coarse_pass_shared(int dim) {
    // The sums of the rows (2 dim), then the fields at X and its eight neighbours (9 x 2 dim).
    return static_cast<size_t>(20) * dim * sizeof(Float);
}

// Columns 2c and 2c + 1 of a row of a matrix (re, im, re, im), 16 bytes at a time.
__device__ __forceinline__ void coarse_pass_pair(const float* p, float& a0, float& b0, float& a1, float& b1) {
    const float4 t = *reinterpret_cast<const float4*>(p);
    a0 = t.x; b0 = t.y; a1 = t.z; b1 = t.w;
}
__device__ __forceinline__ void coarse_pass_pair(const double* p, double& a0, double& b0, double& a1, double& b1) {
    const double2 t0 = reinterpret_cast<const double2*>(p)[0], t1 = reinterpret_cast<const double2*>(p)[1];
    a0 = t0.x; b0 = t0.y; a1 = t1.x; b1 = t1.y;
}

// Warps in a block of coarse_pass_kernel: one for each row, at most 8.
__host__ __device__ __forceinline__ int coarse_pass_warps(int dim) {
    return dim < 8 ? dim : 8;
}

// The sum over the lanes of a warp, in every lane.
template <typename Float>
__device__ __forceinline__ Float coarse_pass_sum(Float x) {
    #pragma unroll
    for (int step = WarpSize / 2; step > 0; step >>= 1) x += __shfl_xor_sync(0xffffffffu, x, step);
    return x;
}

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
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    Float* sum = reinterpret_cast<Float*>(shared_bytes);
    // Site s (0: X, 1 + dir: the neighbour of hop dir): re at fields[2 s dim + col], im dim further.
    Float* fields = sum + 2 * dim;
    const int X = blockIdx.x;
    const int lane = threadIdx.x % WarpSize, warp = threadIdx.x / WarpSize, warps = blockDim.x / WarpSize;
    const size_t first = blockIdx.y * coarse_reals(n_coarse, dim);
    const size_t matrix = static_cast<size_t>(dim) * dim;
    auto at_row = [&](int row) { return first + coarse_index(static_cast<size_t>(X) * dim + row); };
    // The whole block of threads leaves, or none of it.
    if (parity >= 0 && static_cast<int>(d_odd[X]) != parity) {
        for (int row = threadIdx.x; row < dim; row += blockDim.x) {
            d_out[at_row(row)] = 0;
            d_out[at_row(row) + WarpSize] = 0;
        }
        return;
    }

    for (int e = threadIdx.x; e < 9 * dim; e += blockDim.x) {
        const int s = e / dim, col = e % dim;
        const Float* v = (s == 0) ? d_diag_of : d_hops_of;
        if (v == nullptr) continue;
        const size_t Y = (s == 0) ? X : d_neighbours[static_cast<size_t>(s - 1) * n_coarse + X];
        const size_t q = first + coarse_index(Y * dim + col);
        const Float flip = (gamma5 && col >= dim / 2) ? static_cast<Float>(-1.0) : static_cast<Float>(1.0);
        fields[2 * s * dim + col] = flip * v[q];
        fields[(2 * s + 1) * dim + col] = flip * v[q + WarpSize];
    }
    __syncthreads();

    // This lane's part of (m v)_row for the matrix m and the field of site s: its pairs of columns.
    auto times = [&](const Float* m, int s, int row, Float& re, Float& im) {
        const Float* m_row = m + 2 * static_cast<size_t>(dim) * row;
        const Float* xr = fields + 2 * s * dim;
        const Float* xi = xr + dim;
        for (int col = 2 * lane; col < dim; col += 2 * WarpSize) {
            Float a0, b0, a1, b1;
            coarse_pass_pair(m_row + 2 * col, a0, b0, a1, b1);
            re += a0 * xr[col] - b0 * xi[col];
            im += a0 * xi[col] + b0 * xr[col];
            re += a1 * xr[col + 1] - b1 * xi[col + 1];
            im += a1 * xi[col + 1] + b1 * xr[col + 1];
        }
    };
    // The row as it leaves the kernel: Gamma_5 on it with `gamma5`.
    auto store = [&](int row, Float re, Float im) {
        const Float flip = (gamma5 && row >= dim / 2) ? static_cast<Float>(-1.0) : static_cast<Float>(1.0);
        d_out[at_row(row)] = flip * re;
        d_out[at_row(row) + WarpSize] = flip * im;
    };

    for (int row = warp; row < dim; row += warps) {
        Float re = 0, im = 0;
        if (d_diag_of != nullptr) {
            times(d_diag + 2 * matrix * X, 0, row, re, im);
        }
        if (d_hops_of != nullptr) {
            Float hr = 0, hi = 0;
            #pragma unroll
            for (int dir = 0; dir < 8; ++dir) {
                const size_t link = static_cast<size_t>(dir) * n_coarse + X;
                times(d_links + 2 * matrix * link, 1 + dir, row, hr, hi);
            }
            re += sign * hr;
            im += sign * hi;
        }
        re = coarse_pass_sum(re);
        im = coarse_pass_sum(im);
        if (d_base != nullptr) {
            re += d_base[at_row(row)];
            im += d_base[at_row(row) + WarpSize];
        }
        if (d_inverse != nullptr) {
            if (lane == 0) {
                sum[row] = re;
                sum[dim + row] = im;
            }
        } else if (lane == 0) {
            store(row, re, im);
        }
    }
    if (d_inverse == nullptr) return;
    __syncthreads();
    for (int row = warp; row < dim; row += warps) {
        Float re = 0, im = 0;
        const Float* m_row = d_inverse + 2 * (matrix * X + static_cast<size_t>(dim) * row);
        for (int col = 2 * lane; col < dim; col += 2 * WarpSize) {
            Float a0, b0, a1, b1;
            coarse_pass_pair(m_row + 2 * col, a0, b0, a1, b1);
            re += a0 * sum[col] - b0 * sum[dim + col];
            im += a0 * sum[dim + col] + b0 * sum[col];
            re += a1 * sum[col + 1] - b1 * sum[dim + col + 1];
            im += a1 * sum[dim + col + 1] + b1 * sum[col + 1];
        }
        re = coarse_pass_sum(re);
        im = coarse_pass_sum(im);
        if (lane == 0) store(row, re, im);
    }
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

// The coarse unit vectors of the components first .. first + vectors: vector v (blockIdx.y) is 1
// at component first + v of every coarse site and 0 at the others. A thread for each (site,
// component); the padding of the field is left as it is (zero).
template <typename Float>
__global__ void coarse_units_kernel(Float* __restrict__ d_v, int n_coarse, int dim, int first)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_coarse * dim) return;
    const int v = blockIdx.y;
    const size_t at = static_cast<size_t>(v) * coarse_reals(n_coarse, dim) + coarse_index(q);
    d_v[at] = (q % dim == first + v) ? static_cast<Float>(1) : static_cast<Float>(0);
    d_v[at + WarpSize] = 0;
}

// Column first + v of the matrix of every coarse site (dim x dim, row-major (re, im), in the
// precision Target: the layout of coarse_pass_kernel) = vector v (blockIdx.y) of the coarse field
// d_c, whose component row at site X is entry (row, first + v) of the matrix of X. A thread for
// each (site, row): the coarse operator is put together on the device as it is made, in the
// precision it is kept in.
template <typename Float, typename Target>
__global__ void coarse_columns_kernel(Target* __restrict__ d_target, const Float* __restrict__ d_c,
                                      int n_coarse, int dim, int first)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_coarse * dim) return;
    const int v = blockIdx.y;
    const size_t at = static_cast<size_t>(v) * coarse_reals(n_coarse, dim) + coarse_index(q);
    const size_t k = 2 * (static_cast<size_t>(q) * dim + first + v);
    // Through double, as the matrices made on the host and converted on the device were.
    d_target[k] = static_cast<Target>(static_cast<double>(d_c[at]));
    d_target[k + 1] = static_cast<Target>(static_cast<double>(d_c[at + WarpSize]));
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
// second half of the columns), or of A itself without `gamma5`, by the cyclic Jacobi method
// as numerics::hermitian_eigen has it on the CPU: for every pair (p, q), a phase on column
// and row q that makes a_pq real, then a rotation. Thread k has what the loops over k of that
// method do. With cutoff > 0 the eigenvalues not above cutoff times the largest in magnitude
// are left out (abs_inverse of levels.rs); else 1 / max(|lambda|, 1e-300). d_scratch: H and
// V, 4 dim^2 reals a site.
// -------------------------------------------------------------------------
__global__ void blocks_abs_inverse_gamma5_kernel(
    double* __restrict__ d_out,
    const double* __restrict__ d_a,
    double* __restrict__ d_scratch,
    int dim, int gamma5, double cutoff)
{
    __shared__ double part[1024];
    __shared__ double sums[2];
    const int k = threadIdx.x, n = dim;
    const size_t site = blockIdx.x;
    double* h = d_scratch + site * n * n * 4;
    double* v = h + static_cast<size_t>(n) * n * 2;
    const double* a = d_a + site * n * n * 2;
    auto flip = [&](int col) { return (gamma5 && col >= n / 2) ? -1.0 : 1.0; };
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
                // Every thread has read a_pq before thread p overwrites it below (h is in global
                // memory and the block may be several warps); all of them then take the same branch.
                __syncthreads();
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
    double top = 0;
    for (int j = 0; j < n; ++j) top = fmax(top, fabs(h[2 * (j * n + j)]));
    double* out = d_out + site * n * n * 2;
    for (int col = 0; col < n; ++col) {
        double re = 0, im = 0;
        for (int j = 0; j < n; ++j) {
            const double l = fabs(h[2 * (j * n + j)]);
            const double w = (cutoff > 0) ? ((l > cutoff * top) ? 1.0 / l : 0.0) : 1.0 / fmax(l, 1e-300);
            const double xr = v[2 * (k * n + j)], xi = v[2 * (k * n + j) + 1];
            const double yr = v[2 * (col * n + j)], yi = v[2 * (col * n + j) + 1];
            re += (xr * yr + xi * yi) * w;
            im += (xi * yr - xr * yi) * w;
        }
        out[2 * (k * n + col)] = re;
        out[2 * (k * n + col) + 1] = im;
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_FOLD_CUH
