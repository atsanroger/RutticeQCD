#ifndef BRIDGE_CUDA_FERMION_MDWF_SCHUR_CUH
#define BRIDGE_CUDA_FERMION_MDWF_SCHUR_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../base/gamma.cuh"
#include "../base/su3.cuh"
#include "mdwf.cuh"
#include <cuda_runtime.h>

namespace bridge_cuda {

// -------------------------------------------------------------------------
// 5D Möbius Even-Odd Spatial Hopping Kernel (D_eo or D_oe across all Ls)
//
// Evaluates D_4 across all Ls slices between opposite parities:
// Target parity target_parity (0 for Even, 1 for Odd), source parity 1 - target_parity.
// Links are cached in shared memory across Ls slices.
// -------------------------------------------------------------------------
template <typename Float, int LS_PER_BLOCK = 4>
__global__ void mdwf_eo_dslash_5d_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    int Ls,
    int target_parity,
    Float kappa_5)
{
    int vol4d = nx * ny * nz * nt;
    int vol4d_eo = vol4d / 2;

    const int lane     = threadIdx.x; // 0 .. 31
    const int is_local = threadIdx.y; // 0 .. LS_PER_BLOCK - 1
    const int warp_id  = blockIdx.x;
    const int is_base  = blockIdx.y * LS_PER_BLOCK;
    const int is       = is_base + is_local;
    const int site_eo  = lane + WarpSize * warp_id;

    if (site_eo >= vol4d_eo || is >= Ls) return;

    // Convert EO site to full 4D coordinates
    int x_half = site_eo % (nx / 2);
    int rem = site_eo / (nx / 2);
    int y = rem % ny; rem /= ny;
    int z = rem % nz;
    int t = rem / nz;
    int x = 2 * x_half + ((target_parity + y + z + t) & 1);
    int site = coords_to_site(x, y, z, t, nx, ny, nz, nt);

    __shared__ Float u_sh[18 * WarpSize];

    Float out_sp[Nspin][Nc][Ncomplex] = {0};

    #pragma unroll
    for (int mu = 0; mu < Ndim; ++mu) {
        int x_f = (mu == 0) ? (x + 1 < nx ? x + 1 : 0) : x;
        int y_f = (mu == 1) ? (y + 1 < ny ? y + 1 : 0) : y;
        int z_f = (mu == 2) ? (z + 1 < nz ? z + 1 : 0) : z;
        int t_f = (mu == 3) ? (t + 1 < nt ? t + 1 : 0) : t;
        int site_f = coords_to_site(x_f, y_f, z_f, t_f, nx, ny, nz, nt);

        int x_b = (mu == 0) ? (x - 1 >= 0 ? x - 1 : nx - 1) : x;
        int y_b = (mu == 1) ? (y - 1 >= 0 ? y - 1 : ny - 1) : y;
        int z_b = (mu == 2) ? (z - 1 >= 0 ? z - 1 : nz - 1) : z;
        int t_b = (mu == 3) ? (t - 1 >= 0 ? t - 1 : nt - 1) : t;
        int site_b = coords_to_site(x_b, y_b, z_b, t_b, nx, ny, nz, nt);

        if (is_local == 0) {
            #pragma unroll
            for (int r = 0; r < Nc; ++r) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    u_sh[lane + WarpSize * (2 * (c + Nc * r) + 0)] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site)];
                    u_sh[lane + WarpSize * (2 * (c + Nc * r) + 1)] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site)];
                }
            }
        }
        __syncthreads();

        Float u_f_r[Nc][Nc], u_f_i[Nc][Nc];
        #pragma unroll
        for (int r = 0; r < Nc; ++r) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                u_f_r[r][c] = u_sh[lane + WarpSize * (2 * (c + Nc * r) + 0)];
                u_f_i[r][c] = u_sh[lane + WarpSize * (2 * (c + Nc * r) + 1)];
            }
        }
        __syncthreads();

        Float sp_f[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                sp_f[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_f)];
                sp_f[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_f)];
            }
        }

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            Float v_in_r[Nc], v_in_i[Nc], v_out_r[Nc], v_out_i[Nc];
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sp_f[s][c][0];
                v_in_i[c] = sp_f[s][c][1];
            }
            su3_mult_vec(v_out_r, v_out_i, u_f_r, u_f_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                out_sp[s][c][0] -= kappa_5 * v_out_r[c];
                out_sp[s][c][1] -= kappa_5 * v_out_i[c];
            }
        }

        Float sp_b[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                sp_b[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_b)];
                sp_b[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_b)];
            }
        }

        Float u_b_r[Nc][Nc], u_b_i[Nc][Nc];
        #pragma unroll
        for (int r = 0; r < Nc; ++r) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                u_b_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site_b)];
                u_b_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site_b)];
            }
        }

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            Float v_in_r[Nc], v_in_i[Nc], v_out_r[Nc], v_out_i[Nc];
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sp_b[s][c][0];
                v_in_i[c] = sp_b[s][c][1];
            }
            su3_dagger_mult_vec(v_out_r, v_out_i, u_b_r, u_b_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                out_sp[s][c][0] -= kappa_5 * v_out_r[c];
                out_sp[s][c][1] -= kappa_5 * v_out_i[c];
            }
        }
    }

    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)] = out_sp[s][c][0];
            d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)] = out_sp[s][c][1];
        }
    }
}

// -------------------------------------------------------------------------
// Subtraction Kernel on Even sites: out = inp - sub
// -------------------------------------------------------------------------
template <typename Float>
__global__ void mdwf_eo_sub_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_inp,
    const Float* __restrict__ d_sub,
    int nx, int ny, int nz, int nt,
    int Ls,
    int target_parity)
{
    int vol4d = nx * ny * nz * nt;
    int vol4d_eo = vol4d / 2;

    int site_eo = blockIdx.x * blockDim.x + threadIdx.x;
    if (site_eo >= vol4d_eo) return;

    int x_half = site_eo % (nx / 2);
    int rem = site_eo / (nx / 2);
    int y = rem % ny; rem /= ny;
    int z = rem % nz;
    int t = rem / nz;
    int x = 2 * x_half + ((target_parity + y + z + t) & 1);
    int site = coords_to_site(x, y, z, t, nx, ny, nz, nt);

    for (int is = 0; is < Ls; ++is) {
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                int idx_r = aosoa_5d_spinor_idx(c, s, 0, is, Ls, site);
                int idx_i = aosoa_5d_spinor_idx(c, s, 1, is, Ls, site);

                d_out[idx_r] = d_inp[idx_r] - d_sub[idx_r];
                d_out[idx_i] = d_inp[idx_i] - d_sub[idx_i];
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_MDWF_SCHUR_CUH
