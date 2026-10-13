#ifndef BRIDGE_CUDA_FERMION_HOPS_CUH
#define BRIDGE_CUDA_FERMION_HOPS_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include <cuda_runtime.h>

// What the Wilson, clover and domain-wall operators share, in the Dirac representation: the
// hops of the Wilson kernel,
//
//   H w = - kappa sum_mu [ (1 - gamma_mu) U_mu(x) w(x+mu) + (1 + gamma_mu) U_mu(x-mu)^dag w(x-mu) ],
//
// H^dag being H with the signs of gamma_mu exchanged, and the hop in the fifth dimension
// of domain-wall fermions,
//
//   (Y psi)_s = - P- psi_{s+1} - P+ psi_{s-1}     (+ m instead of - 1 across the walls),
//
// with P+- = (1 +- gamma_5)/2.
//
// A field has Ls spinors per site (1 for a four-dimensional field), on the whole lattice or
// on the sites of one parity: site (x, y, z, t) of parity p = (x + y + z + t) & 1 is site
// x/2 + (nx/2)(y + ny (z + nz t)) of the field of parity p (nx even). The links are those of
// the whole lattice. bc[mu] multiplies a spinor that arrives across the boundary in
// direction mu (+1 periodic, -1 antiperiodic, 0 Dirichlet); the links are periodic.

namespace bridge_cuda {

// The coordinates of site `site` of the field of parity `parity`.
__host__ __device__ __forceinline__ void half_site_to_coords(
    int site, int parity, int xc[4], int nx, int ny, int nz)
{
    int rem = site / (nx / 2);
    xc[1] = rem % ny; rem /= ny;
    xc[2] = rem % nz;
    xc[3] = rem / nz;
    xc[0] = 2 * (site % (nx / 2)) + ((parity + xc[1] + xc[2] + xc[3]) & 1);
}

// w = a psi_s + y (Y psi)_s, or with Y^dag, at fifth coordinate `is` of site `site`.
template <typename Float>
__device__ __forceinline__ void fifth_term(
    Float w[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    int is, int Ls, int site,
    Float mass, Float a, Float y, bool dagger)
{
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            w[s][c][0] = a * d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)];
            w[s][c][1] = a * d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)];
        }
    }
    if (y == static_cast<Float>(0.0)) return;
    // The coefficient of the projected neighbour in Y: -1, or +m across the walls.
    const int is_p = (is + 1 < Ls) ? (is + 1) : 0;
    const int is_m = (is >= 1) ? (is - 1) : (Ls - 1);
    const Float factor_p = y * ((is + 1 < Ls) ? static_cast<Float>(-1.0) : mass);
    const Float factor_m = y * ((is >= 1) ? static_cast<Float>(-1.0) : mass);
    Float above[Nspin][Nc][Ncomplex], below[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            above[s][c][0] = factor_p * d_in[aosoa_5d_spinor_idx(c, s, 0, is_p, Ls, site)];
            above[s][c][1] = factor_p * d_in[aosoa_5d_spinor_idx(c, s, 1, is_p, Ls, site)];
            below[s][c][0] = factor_m * d_in[aosoa_5d_spinor_idx(c, s, 0, is_m, Ls, site)];
            below[s][c][1] = factor_m * d_in[aosoa_5d_spinor_idx(c, s, 1, is_m, Ls, site)];
        }
    }
    // Y takes P- from above and P+ from below; Y^dag the other way round.
    Float from_above[Nspin][Nc][Ncomplex], from_below[Nspin][Nc][Ncomplex];
    if (dagger) {
        proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_above, above);
        proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_below, below);
    } else {
        proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_above, above);
        proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_below, below);
    }
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            w[s][c][0] += from_above[s][c][0] + from_below[s][c][0];
            w[s][c][1] += from_above[s][c][1] + from_below[s][c][1];
        }
    }
}

// acc += H w at the site of coordinates xc and fifth coordinate `is`, with
// w = (b - c Y) psi of d_in; `sign` -1 gives H^dag. With `half`, d_in holds the sites of the
// parity opposite to that of xc. For a four-dimensional field: Ls = 1, b = 1, c = 0.
template <typename Float>
__device__ __forceinline__ void hops(
    Float acc[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    const int xc[4], int is, int Ls,
    const int dims[4], const Float bc[4],
    bool half, Float sign, Float kappa,
    Float mass, Float b5, Float c5)
{
    const int nx = dims[0], ny = dims[1], nz = dims[2], nt = dims[3];
    const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    #pragma unroll
    for (int mu = 0; mu < Ndim; ++mu) {
        int f[4] = {xc[0], xc[1], xc[2], xc[3]};
        int b[4] = {xc[0], xc[1], xc[2], xc[3]};
        const bool wrap_f = (xc[mu] + 1 == dims[mu]);
        const bool wrap_b = (xc[mu] == 0);
        f[mu] = wrap_f ? 0 : xc[mu] + 1;
        b[mu] = wrap_b ? dims[mu] - 1 : xc[mu] - 1;
        const int site_f = coords_to_site(f[0], f[1], f[2], f[3], nx, ny, nz, nt);
        const int site_b = coords_to_site(b[0], b[1], b[2], b[3], nx, ny, nz, nt);
        // Where the neighbours are in d_in.
        const int in_f = half ? f[0] / 2 + (nx / 2) * (f[1] + ny * (f[2] + nz * f[3])) : site_f;
        const int in_b = half ? b[0] / 2 + (nx / 2) * (b[1] + ny * (b[2] + nz * b[3])) : site_b;
        const Float sign_f = wrap_f ? bc[mu] : static_cast<Float>(1.0);
        const Float sign_b = wrap_b ? bc[mu] : static_cast<Float>(1.0);

        Float w_f[Nspin][Nc][Ncomplex], w_b[Nspin][Nc][Ncomplex];
        fifth_term<Float>(w_f, d_in, is, Ls, in_f, mass, b5, -c5, false);
        fifth_term<Float>(w_b, d_in, is, Ls, in_b, mass, b5, -c5, false);
        Float g_f[Nspin][Nc][Ncomplex], g_b[Nspin][Nc][Ncomplex];
        mult_gamma_dirac<Float>(g_f, w_f, mu);
        mult_gamma_dirac<Float>(g_b, w_b, mu);

        Float u_f_r[Nc][Nc], u_f_i[Nc][Nc], u_b_r[Nc][Nc], u_b_i[Nc][Nc];
        #pragma unroll
        for (int r = 0; r < Nc; ++r) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                u_f_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site)];
                u_f_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site)];
                u_b_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site_b)];
                u_b_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site_b)];
            }
        }
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            Float v_in_r[Nc], v_in_i[Nc], v_out_r[Nc], v_out_i[Nc];
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sign_f * (w_f[s][c][0] - sign * g_f[s][c][0]);
                v_in_i[c] = sign_f * (w_f[s][c][1] - sign * g_f[s][c][1]);
            }
            su3_mult_vec(v_out_r, v_out_i, u_f_r, u_f_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                acc[s][c][0] -= kappa * v_out_r[c];
                acc[s][c][1] -= kappa * v_out_i[c];
            }
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sign_b * (w_b[s][c][0] + sign * g_b[s][c][0]);
                v_in_i[c] = sign_b * (w_b[s][c][1] + sign * g_b[s][c][1]);
            }
            su3_dagger_mult_vec(v_out_r, v_out_i, u_b_r, u_b_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                acc[s][c][0] -= kappa * v_out_r[c];
                acc[s][c][1] -= kappa * v_out_i[c];
            }
        }
    }
}

// -------------------------------------------------------------------------
// out = H (b - c Y) in, or with H^dag (`sign` -1): on the whole lattice (target_parity < 0),
// or out on the sites of parity target_parity and in on the others.
// One thread for each (site, s): threadIdx.x the site within its block of 32, threadIdx.y
// the fifth coordinate within LS_PER_BLOCK of them.
// -------------------------------------------------------------------------
template <typename Float, int LS_PER_BLOCK = 4>
__global__ void hop_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls,
    int target_parity,
    Float sign, Float kappa,
    Float mass, Float b5, Float c5)
{
    const bool half = (target_parity >= 0);
    const int sites = (half ? nx / 2 : nx) * ny * nz * nt;
    const int site  = threadIdx.x + WarpSize * blockIdx.x;
    const int is    = blockIdx.y * LS_PER_BLOCK + threadIdx.y;
    if (site >= sites || is >= Ls) return;

    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    if (half) {
        half_site_to_coords(site, target_parity, xc, nx, ny, nz);
    } else {
        site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    }

    Float acc[Nspin][Nc][Ncomplex] = {0};
    hops<Float>(acc, d_in, d_gauge, xc, is, Ls, dims, bc, half, sign, kappa, mass, b5, c5);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)] = acc[s][c][0];
            d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)] = acc[s][c][1];
        }
    }
}

// -------------------------------------------------------------------------
// A field on one parity out of the field on the whole lattice (to_half), or into it; the
// sites of the other parity are left as they are.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void parity_copy_kernel(
    Float* __restrict__ d_half,
    Float* __restrict__ d_full,
    int nx, int ny, int nz, int nt,
    int Ls,
    int parity,
    int to_half)
{
    const int sites = (nx / 2) * ny * nz * nt;
    for (int half = blockIdx.x * blockDim.x + threadIdx.x; half < sites; half += blockDim.x * gridDim.x) {
        int xc[4];
        half_site_to_coords(half, parity, xc, nx, ny, nz);
        const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
        for (int is = 0; is < Ls; ++is) {
            #pragma unroll
            for (int k = 0; k < Nspin * Nc * Ncomplex; ++k) {
                const int s = k / (Nc * Ncomplex), c = (k / Ncomplex) % Nc, ri = k % Ncomplex;
                const int at_half = aosoa_5d_spinor_idx(c, s, ri, is, Ls, half);
                const int at_full = aosoa_5d_spinor_idx(c, s, ri, is, Ls, site);
                if (to_half) {
                    d_half[at_half] = d_full[at_full];
                } else {
                    d_full[at_full] = d_half[at_half];
                }
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_HOPS_CUH
