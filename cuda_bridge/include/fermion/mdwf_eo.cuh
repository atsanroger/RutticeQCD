#ifndef BRIDGE_CUDA_FERMION_MDWF_EO_CUH
#define BRIDGE_CUDA_FERMION_MDWF_EO_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../base/gamma.cuh"
#include "../base/su3.cuh"
#include <cuda_runtime.h>

// The even-odd blocks of the plain Möbius domain-wall operator (Neff's alpha = 1, Dirac
// representation), on fields that hold the sites of one parity only:
//
//   D = [[C, D_eo], [D_oe, C]],   C = d+ + d- Y,   D_pq = -1/2 sum_mu hops_pq (b - c Y),
//
// with Y, d+ and d- as in mdwf.cuh. Site (x, y, z, t) of parity p = (x + y + z + t) & 1 is
// site x/2 + (nx/2) (y + ny (z + nz t)) of the field of parity p; nx must be even. The links
// are those of the whole lattice.

namespace bridge_cuda {

// w = (b - c Y) psi at fifth coordinate `is` of site `site` of `d_in`.
template <typename Float>
__device__ __forceinline__ void mdwf_kernel_input(
    Float w[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    int is, int Ls, int site,
    Float mass, Float b5, Float c5)
{
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            w[s][c][0] = b5 * d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)];
            w[s][c][1] = b5 * d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)];
        }
    }
    if (c5 == static_cast<Float>(0.0)) return;
    // -Y psi = P- psi_{s+1} + P+ psi_{s-1}, with -m in the place of 1 across the walls
    const int is_p = (is + 1 < Ls) ? (is + 1) : 0;
    const int is_m = (is >= 1) ? (is - 1) : (Ls - 1);
    const Float factor_p = (is + 1 < Ls) ? static_cast<Float>(1.0) : -mass;
    const Float factor_m = (is >= 1) ? static_cast<Float>(1.0) : -mass;
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
    Float pl[Nspin][Nc][Ncomplex], pr[Nspin][Nc][Ncomplex];
    proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(pl, above);
    proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(pr, below);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            w[s][c][0] += c5 * (pl[s][c][0] + pr[s][c][0]);
            w[s][c][1] += c5 * (pl[s][c][1] + pr[s][c][1]);
        }
    }
}

// -------------------------------------------------------------------------
// out = D_pq in: out on the sites of parity `target_parity`, in on the others.
// One thread for each (site, s): threadIdx.x the site within its block of 32, threadIdx.y
// the fifth coordinate within LS_PER_BLOCK of them.
// -------------------------------------------------------------------------
template <typename Float, int LS_PER_BLOCK = 4>
__global__ void mdwf_eo_hop_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls,
    int target_parity,
    Float mass, Float b5, Float c5)
{
    const int half   = (nx / 2) * ny * nz * nt;
    const int site_e = threadIdx.x + WarpSize * blockIdx.x;
    const int is     = blockIdx.y * LS_PER_BLOCK + threadIdx.y;
    if (site_e >= half || is >= Ls) return;

    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    {
        int rem = site_e / (nx / 2);
        xc[1] = rem % ny; rem /= ny;
        xc[2] = rem % nz;
        xc[3] = rem / nz;
        xc[0] = 2 * (site_e % (nx / 2)) + ((target_parity + xc[1] + xc[2] + xc[3]) & 1);
    }
    const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);

    Float acc[Nspin][Nc][Ncomplex] = {0};
    #pragma unroll
    for (int mu = 0; mu < Ndim; ++mu) {
        int f[4] = {xc[0], xc[1], xc[2], xc[3]};
        int b[4] = {xc[0], xc[1], xc[2], xc[3]};
        const bool wrap_f = (xc[mu] + 1 == dims[mu]);
        const bool wrap_b = (xc[mu] == 0);
        f[mu] = wrap_f ? 0 : xc[mu] + 1;
        b[mu] = wrap_b ? dims[mu] - 1 : xc[mu] - 1;
        // The neighbours in the field of the other parity, and the link behind in the whole lattice.
        const int half_f = f[0] / 2 + (nx / 2) * (f[1] + ny * (f[2] + nz * f[3]));
        const int half_b = b[0] / 2 + (nx / 2) * (b[1] + ny * (b[2] + nz * b[3]));
        const int site_b = coords_to_site(b[0], b[1], b[2], b[3], nx, ny, nz, nt);
        const Float sign_f = wrap_f ? bc[mu] : static_cast<Float>(1.0);
        const Float sign_b = wrap_b ? bc[mu] : static_cast<Float>(1.0);

        Float w_f[Nspin][Nc][Ncomplex], w_b[Nspin][Nc][Ncomplex];
        mdwf_kernel_input<Float>(w_f, d_in, is, Ls, half_f, mass, b5, c5);
        mdwf_kernel_input<Float>(w_b, d_in, is, Ls, half_b, mass, b5, c5);
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
                v_in_r[c] = sign_f * (w_f[s][c][0] - g_f[s][c][0]);
                v_in_i[c] = sign_f * (w_f[s][c][1] - g_f[s][c][1]);
            }
            su3_mult_vec(v_out_r, v_out_i, u_f_r, u_f_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                acc[s][c][0] -= static_cast<Float>(0.5) * v_out_r[c];
                acc[s][c][1] -= static_cast<Float>(0.5) * v_out_i[c];
            }
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sign_b * (w_b[s][c][0] + g_b[s][c][0]);
                v_in_i[c] = sign_b * (w_b[s][c][1] + g_b[s][c][1]);
            }
            su3_dagger_mult_vec(v_out_r, v_out_i, u_b_r, u_b_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                acc[s][c][0] -= static_cast<Float>(0.5) * v_out_r[c];
                acc[s][c][1] -= static_cast<Float>(0.5) * v_out_i[c];
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_e)] = acc[s][c][0];
            d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_e)] = acc[s][c][1];
        }
    }
}

// P+ (plus) or P- of the spinor at fifth coordinate `is` of site `site`.
template <typename Float>
__device__ __forceinline__ void mdwf_load_projected(
    Float r[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    int is, int Ls, int site, bool plus)
{
    Float phi[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            phi[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)];
            phi[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)];
        }
    }
    if (plus) {
        proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(r, phi);
    } else {
        proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(r, phi);
    }
}

// -------------------------------------------------------------------------
// out = C^-1 in (C^dag^-1 with `dagger`), exactly, on `sites` sites (of the whole lattice
// or of one parity). out must not be in.
//
// C x = r is, on P+ x and with rows counted from s = 0 (on P- x: from s = Ls-1; the other
// way round for C^dag),
//
//   d+ x_i - d- x_(i-1) = r_i  (i > 0),     d+ x_0 + m d- x_(Ls-1) = r_0,
//
// so x_i = p_i + q^i x_0 with p_0 = 0, p_i = (r_i + d- p_(i-1)) / d+, q = d- / d+, and
// x_0 = (r_0 - m d- p_(Ls-1)) / (d+ + m d- q^(Ls-1)). The recursion runs twice: for the
// last p, which gives x_0, and then for every x, with no buffer of a site's spinors.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void mdwf_site_inv_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int sites,
    int Ls,
    Float M5, Float mass, Float b5, Float c5,
    int dagger)
{
    const Float d_plus  = static_cast<Float>(1.0) + b5 * (static_cast<Float>(4.0) - M5);
    const Float d_minus = static_cast<Float>(1.0) - c5 * (static_cast<Float>(4.0) - M5);
    const Float q = d_minus / d_plus;
    Float q_last = static_cast<Float>(1.0);
    for (int i = 1; i < Ls; ++i) q_last *= q;
    const Float corner   = static_cast<Float>(1.0) / (d_plus + mass * d_minus * q_last);
    const Float per_step = static_cast<Float>(1.0) / d_plus;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < sites; site += blockDim.x * gridDim.x) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool plus = (pass == 0);
            const bool up = (plus != (dagger != 0));
            Float r[Nspin][Nc][Ncomplex], p[Nspin][Nc][Ncomplex], first[Nspin][Nc][Ncomplex];
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) { p[s][c][0] = 0; p[s][c][1] = 0; }
            }
            for (int i = 1; i < Ls; ++i) {
                mdwf_load_projected<Float>(r, d_in, up ? i : Ls - 1 - i, Ls, site, plus);
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        p[s][c][0] = per_step * (r[s][c][0] + d_minus * p[s][c][0]);
                        p[s][c][1] = per_step * (r[s][c][1] + d_minus * p[s][c][1]);
                    }
                }
            }
            mdwf_load_projected<Float>(r, d_in, up ? 0 : Ls - 1, Ls, site, plus);
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    first[s][c][0] = corner * (r[s][c][0] - mass * d_minus * p[s][c][0]);
                    first[s][c][1] = corner * (r[s][c][1] - mass * d_minus * p[s][c][1]);
                    p[s][c][0] = 0; p[s][c][1] = 0;
                }
            }
            Float power = static_cast<Float>(1.0);
            for (int i = 0; i < Ls; ++i) {
                const int is = up ? i : Ls - 1 - i;
                if (i > 0) {
                    mdwf_load_projected<Float>(r, d_in, is, Ls, site, plus);
                    #pragma unroll
                    for (int s = 0; s < Nspin; ++s) {
                        #pragma unroll
                        for (int c = 0; c < Nc; ++c) {
                            p[s][c][0] = per_step * (r[s][c][0] + d_minus * p[s][c][0]);
                            p[s][c][1] = per_step * (r[s][c][1] + d_minus * p[s][c][1]);
                        }
                    }
                }
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        #pragma unroll
                        for (int ri = 0; ri < Ncomplex; ++ri) {
                            const Float x = p[s][c][ri] + power * first[s][c][ri];
                            const int k = aosoa_5d_spinor_idx(c, s, ri, is, Ls, site);
                            d_out[k] = plus ? x : d_out[k] + x;
                        }
                    }
                }
                power *= q;
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_MDWF_EO_CUH
