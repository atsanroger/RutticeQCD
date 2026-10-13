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
    // For a field that is zero but where leg `leg` of the hops (fermion/hops.cuh) comes from
    // another aggregate of `block` sites: only those sites are read. leg < 0: all sites.
    int leg;
    int block[4];
};

__host__ __device__ __forceinline__ size_t coarse_index(size_t q) {
    return (q / WarpSize) * 2 * WarpSize + q % WarpSize;
}

__host__ __device__ __forceinline__ size_t coarse_reals(int n_coarse, int dim) {
    return (static_cast<size_t>(n_coarse) * dim + WarpSize - 1) / WarpSize * 2 * WarpSize;
}

// -------------------------------------------------------------------------
// coarse = P^dag fine. A thread for each (coarse site, component); blockIdx.y the vector.
// All of P is read, which is most of the time this takes; a field of g.leg takes the sites
// it is not zero on.
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

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_FOLD_CUH
