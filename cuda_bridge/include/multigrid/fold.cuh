#ifndef BRIDGE_CUDA_MULTIGRID_FOLD_CUH
#define BRIDGE_CUDA_MULTIGRID_FOLD_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../fermion/hops.cuh"
#include <cuda_runtime.h>

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
// in single precision. gamma_5 has one entry in a row: (gamma_5 w)_a = sign[a] w_(col[a]).
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
};

__host__ __device__ __forceinline__ size_t coarse_index(size_t q) {
    return (q / WarpSize) * 2 * WarpSize + q % WarpSize;
}

__host__ __device__ __forceinline__ size_t coarse_reals(int n_coarse, int dim) {
    return (static_cast<size_t>(n_coarse) * dim + WarpSize - 1) / WarpSize * 2 * WarpSize;
}

// -------------------------------------------------------------------------
// coarse = P^dag fine. A thread for each (coarse site, component); blockIdx.y the vector.
// d_sites: the block_volume sites (of the whole lattice) of every coarse site.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void fold_restrict_kernel(
    Float* __restrict__ d_coarse,
    const Float* __restrict__ d_fine,
    const float* __restrict__ d_p,
    const unsigned* __restrict__ d_sites,
    int n_coarse, int block_volume, int n,
    FoldGeometry g)
{
    const int dim = 2 * n;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_coarse * dim) return;
    const int X = t / dim, k = t % dim, c = k / n, i = k % n;
    const int Ls = g.Ls, kept = 12 * Ls;
    const bool half = (g.parity >= 0);
    const int sites = (half ? g.nx / 2 : g.nx) * g.ny * g.nz * g.nt;
    const Float* fine = d_fine + blockIdx.y * vector_reals(sites, Ls);
    const Float chirality = (c == 0) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);

    Float zr = 0, zi = 0;
    for (int j = 0; j < block_volume; ++j) {
        const int x = d_sites[static_cast<size_t>(X) * block_volume + j];
        int at = x;
        if (half) {
            int xc[4];
            site_to_coords(x, xc[0], xc[1], xc[2], xc[3], g.nx, g.ny, g.nz, g.nt);
            if (((xc[0] + xc[1] + xc[2] + xc[3]) & 1) != g.parity) continue;
            at = xc[0] / 2 + (g.nx / 2) * (x / g.nx);
        }
        const float* b = d_p + ((static_cast<size_t>(x) * n + i) * 2 + c) * kept;
        for (int s = 0; s < Ls / 2; ++s) {
            #pragma unroll
            for (int a = 0; a < Nspin; ++a) {
                const Float sign = chirality * g.sign[a];
                #pragma unroll
                for (int colour = 0; colour < Nc; ++colour) {
                    // f_s + or - gamma_5 f_(Ls-1-s)
                    const Float fr = fine[aosoa_5d_spinor_idx(colour, a, 0, s, Ls, at)]
                        + sign * fine[aosoa_5d_spinor_idx(colour, g.col[a], 0, Ls - 1 - s, Ls, at)];
                    const Float fi = fine[aosoa_5d_spinor_idx(colour, a, 1, s, Ls, at)]
                        + sign * fine[aosoa_5d_spinor_idx(colour, g.col[a], 1, Ls - 1 - s, Ls, at)];
                    const int e = 24 * s + 2 * (colour + Nc * a);
                    const Float br = b[e], bi = b[e + 1];
                    zr += br * fr + bi * fi;
                    zi += br * fi - bi * fr;
                }
            }
        }
    }
    Float* coarse = d_coarse + blockIdx.y * coarse_reals(n_coarse, dim);
    const size_t q = coarse_index(static_cast<size_t>(X) * dim + k);
    coarse[q] = zr;
    coarse[q + WarpSize] = zi;
}

// -------------------------------------------------------------------------
// fine = P coarse. A thread for each (fine site, s < Ls/2, spin, colour), which writes the
// spinor at s and its mirror at Ls - 1 - s; blockIdx.y the vector.
// d_coarse_of: the coarse site of every site of the whole lattice.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void fold_prolong_kernel(
    Float* __restrict__ d_fine,
    const Float* __restrict__ d_coarse,
    const float* __restrict__ d_p,
    const unsigned* __restrict__ d_coarse_of,
    int n_coarse, int n,
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
    const Float* coarse = d_coarse + blockIdx.y * coarse_reals(n_coarse, dim);
    const size_t first = static_cast<size_t>(d_coarse_of[x]) * dim;

    // a_c = sum_i coefficient(c, i) b(i, c), for the two chiralities.
    Float ar[2], ai[2];
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        ar[c] = 0;
        ai[c] = 0;
        for (int i = 0; i < n; ++i) {
            const size_t q = coarse_index(first + c * n + i);
            const Float cr = coarse[q], ci = coarse[q + WarpSize];
            const float* b = d_p + ((static_cast<size_t>(x) * n + i) * 2 + c) * kept + 2 * e;
            const Float br = b[0], bi = b[1];
            ar[c] += cr * br - ci * bi;
            ai[c] += cr * bi + ci * br;
        }
    }
    const int s = e / 12, a = (e % 12) / Nc, colour = e % Nc;
    Float* fine = d_fine + blockIdx.y * vector_reals(sites, Ls);
    fine[aosoa_5d_spinor_idx(colour, a, 0, s, Ls, at)] = ar[0] + ar[1];
    fine[aosoa_5d_spinor_idx(colour, a, 1, s, Ls, at)] = ai[0] + ai[1];
    // The mirror: gamma_5 (a_+ - a_-), whose row `to` takes this spin.
    int to = 0;
    #pragma unroll
    for (int row = 0; row < Nspin; ++row) {
        if (g.col[row] == a) to = row;
    }
    const Float sign = g.sign[to];
    fine[aosoa_5d_spinor_idx(colour, to, 0, Ls - 1 - s, Ls, at)] = sign * (ar[0] - ar[1]);
    fine[aosoa_5d_spinor_idx(colour, to, 1, Ls - 1 - s, Ls, at)] = sign * (ai[0] - ai[1]);
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_FOLD_CUH
