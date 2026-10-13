#ifndef BRIDGE_CUDA_FERMION_CLOVER_CUH
#define BRIDGE_CUDA_FERMION_CLOVER_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include "wilson.cuh"

// The clover operator D = T - kappa hops, Dirac representation.
//
// T, the block of D within a site (1 - kappa c_SW sigma F, or its inverse), commutes with
// gamma_5 and is stored as RutticeQCD stores it: 72 reals per site, the block on gamma_5 = +1
// and then the one on gamma_5 = -1, each a packed Hermitian 6 x 6 matrix (index colour +
// 3 spin): the six real diagonal entries, then the upper triangle row by row as (re, im).
// In the Dirac representation gamma_5 = [[0, 1], [1, 0]], with eigenvectors
// (e_a +- e_(a+2)) / sqrt(2).

namespace bridge_cuda {

constexpr int CloverReals = 72;

// y = M x for the packed Hermitian 6 x 6 matrix `which` (0 or 1) of `site`.
template <typename Float>
__device__ __forceinline__ void hermitian6_mul(
    Float y[6][Ncomplex],
    const Float* __restrict__ d_clover,
    int which, int site,
    const Float x[6][Ncomplex])
{
    const int base = 36 * which;
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        const Float d = d_clover[aosoa_idx(base + r, site, CloverReals)];
        y[r][0] = d * x[r][0];
        y[r][1] = d * x[r][1];
    }
    int k = 6;
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        #pragma unroll
        for (int c = r + 1; c < 6; ++c) {
            const Float zr = d_clover[aosoa_idx(base + k, site, CloverReals)];
            const Float zi = d_clover[aosoa_idx(base + k + 1, site, CloverReals)];
            // y_r += z x_c,  y_c += conj(z) x_r
            y[r][0] += zr * x[c][0] - zi * x[c][1];
            y[r][1] += zr * x[c][1] + zi * x[c][0];
            y[c][0] += zr * x[r][0] + zi * x[r][1];
            y[c][1] += zr * x[r][1] - zi * x[r][0];
            k += 2;
        }
    }
}

// out = T psi: psi at `at` of d_in, T that of the site `site` of the lattice.
template <typename Float>
__device__ __forceinline__ void clover_site(
    Float out[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_clover,
    int at, int site)
{
    // The components along e_a + e_(a+2) and e_a - e_(a+2), without the 1/sqrt(2).
    Float plus[6][Ncomplex], minus[6][Ncomplex];
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                const Float upper = d_in[aosoa_spinor_idx(c, s, ri, at)];
                const Float lower = d_in[aosoa_spinor_idx(c, s + 2, ri, at)];
                plus[c + Nc * s][ri]  = upper + lower;
                minus[c + Nc * s][ri] = upper - lower;
            }
        }
    }
    Float p[6][Ncomplex], m[6][Ncomplex];
    hermitian6_mul<Float>(p, d_clover, 0, site, plus);
    hermitian6_mul<Float>(m, d_clover, 1, site, minus);
    // Back to spin components; the two 1/sqrt(2) make 1/2.
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                const int k = c + Nc * s;
                out[s][c][ri]     = static_cast<Float>(0.5) * (p[k][ri] + m[k][ri]);
                out[s + 2][c][ri] = static_cast<Float>(0.5) * (p[k][ri] - m[k][ri]);
            }
        }
    }
}

// -------------------------------------------------------------------------
// out = T in, with the blocks of T (or of its inverse) in d_clover: on the whole lattice
// (parity < 0), or in and out on the sites of `parity`. The blocks are those of every site of
// the lattice. Every vector of the fields; the fibres of a site (hops.cuh) are its vectors,
// which read the same block.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void clover_term_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_clover,
    int nx, int ny, int nz, int nt,
    int parity,
    int vectors)
{
    const int sites = ((parity >= 0) ? nx / 2 : nx) * ny * nz * nt;
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (at >= sites || v >= vectors) return;
    const Float* __restrict__ in = d_in + v * vector_reals(sites, 1);
    Float* __restrict__ out = d_out + v * vector_reals(sites, 1);
    int site = at;
    if (parity >= 0) {
        int xc[4];
        half_site_to_coords(at, parity, xc, nx, ny, nz);
        site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    }
    Float acc[Nspin][Nc][Ncomplex];
    clover_site<Float>(acc, in, d_clover, at, site);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[aosoa_spinor_idx(c, s, 0, at)] = acc[s][c][0];
            out[aosoa_spinor_idx(c, s, 1, at)] = acc[s][c][1];
        }
    }
}

// -------------------------------------------------------------------------
// out = D in = T in + H in (hops.cuh), or D^dag in (`sign` -1; T is Hermitian), of every
// vector of the fields, as wilson_interior_kernel.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void clover_dslash_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    const Float* __restrict__ d_clover,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    Float kappa,
    Float sign,
    int vectors)
{
    const int vol = nx * ny * nz * nt;
    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (site >= vol || v >= vectors) return;
    const Float* __restrict__ in = d_in + v * vector_reals(vol, 1);
    Float* __restrict__ out = d_out + v * vector_reals(vol, 1);
    Float acc[Nspin][Nc][Ncomplex];
    clover_site<Float>(acc, in, d_clover, site, site);
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    hops<Float>(acc, in, d_gauge, link_rows, xc, 0, 1, dims, bc, false, sign, kappa,
                static_cast<Float>(0.0), static_cast<Float>(1.0), static_cast<Float>(0.0));
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[aosoa_spinor_idx(c, s, 0, site)] = acc[s][c][0];
            out[aosoa_spinor_idx(c, s, 1, site)] = acc[s][c][1];
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_CLOVER_CUH
