#ifndef BRIDGE_CUDA_FERMION_DD_CUH
#define BRIDGE_CUDA_FERMION_DD_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include "wilson.cuh"
#include "clover.cuh"
#include "mdwf.cuh"

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Wilson Domain Decomposition (Block DD / SAP) Kernel
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_dd_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    int bx, int by, int bz, int bt,
    int ieo,
    Float kappa)
{
    int vol = nx * ny * nz * nt;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        int x, y, z, t;
        site_to_coords(site, x, y, z, t, nx, ny, nz, nt);

        int ibx = x / bx;
        int iby = y / by;
        int ibz = z / bz;
        int ibt = t / bt;
        int jeo = (ibx + iby + ibz + ibt) & 1;

        if (jeo != ieo) {
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    d_out[aosoa_spinor_idx(c, s, 0, site)] = 0;
                    d_out[aosoa_spinor_idx(c, s, 1, site)] = 0;
                }
            }
            continue;
        }

        Float out_sp[Nspin][Nc][Ncomplex] = {0};

        #pragma unroll
        for (int mu = 0; mu < Ndim; ++mu) {
            int x_f = x, y_f = y, z_f = z, t_f = t;
            if (mu == 0) x_f = (x + 1 < nx) ? x + 1 : 0;
            if (mu == 1) y_f = (y + 1 < ny) ? y + 1 : 0;
            if (mu == 2) z_f = (z + 1 < nz) ? z + 1 : 0;
            if (mu == 3) t_f = (t + 1 < nt) ? t + 1 : 0;
            int site_f = coords_to_site(x_f, y_f, z_f, t_f, nx, ny, nz, nt);

            int x_b = x, y_b = y, z_b = z, t_b = t;
            if (mu == 0) x_b = (x - 1 >= 0) ? x - 1 : nx - 1;
            if (mu == 1) y_b = (y - 1 >= 0) ? y - 1 : ny - 1;
            if (mu == 2) z_b = (z - 1 >= 0) ? z - 1 : nz - 1;
            if (mu == 3) t_b = (t - 1 >= 0) ? t - 1 : nt - 1;
            int site_b = coords_to_site(x_b, y_b, z_b, t_b, nx, ny, nz, nt);

            Float sp_f[Nspin][Nc][Ncomplex], sp_b[Nspin][Nc][Ncomplex];
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    sp_f[s][c][0] = d_in[aosoa_spinor_idx(c, s, 0, site_f)];
                    sp_f[s][c][1] = d_in[aosoa_spinor_idx(c, s, 1, site_f)];
                    sp_b[s][c][0] = d_in[aosoa_spinor_idx(c, s, 0, site_b)];
                    sp_b[s][c][1] = d_in[aosoa_spinor_idx(c, s, 1, site_b)];
                }
            }

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
                    v_in_r[c] = sp_f[s][c][0];
                    v_in_i[c] = sp_f[s][c][1];
                }
                su3_mult_vec(v_out_r, v_out_i, u_f_r, u_f_i, v_in_r, v_in_i);
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    out_sp[s][c][0] -= kappa * v_out_r[c];
                    out_sp[s][c][1] -= kappa * v_out_i[c];
                }

                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    v_in_r[c] = sp_b[s][c][0];
                    v_in_i[c] = sp_b[s][c][1];
                }
                su3_dagger_mult_vec(v_out_r, v_out_i, u_b_r, u_b_i, v_in_r, v_in_i);
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    out_sp[s][c][0] -= kappa * v_out_r[c];
                    out_sp[s][c][1] -= kappa * v_out_i[c];
                }
            }
        }

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                Float self_r = d_in[aosoa_spinor_idx(c, s, 0, site)];
                Float self_i = d_in[aosoa_spinor_idx(c, s, 1, site)];
                d_out[aosoa_spinor_idx(c, s, 0, site)] = self_r + out_sp[s][c][0];
                d_out[aosoa_spinor_idx(c, s, 1, site)] = self_i + out_sp[s][c][1];
            }
        }
    }
}

// -------------------------------------------------------------------------
// MDWF 5D Domain Decomposition Kernel (5dir D_5)
// -------------------------------------------------------------------------
template <typename Float, int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL>
__global__ void mdwf_dd_5dir_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int nx, int ny, int nz, int nt,
    int bx, int by, int bz, int bt,
    int Ls,
    int ieo,
    Float M5,
    Float mass,
    Float alpha,
    Float b5,
    Float c5)
{
    int vol4d = nx * ny * nz * nt;
    int vol5d = vol4d * Ls;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vol5d; idx += blockDim.x * gridDim.x) {
        int site4d = idx % vol4d;
        int is     = idx / vol4d;

        int x, y, z, t;
        site_to_coords(site4d, x, y, z, t, nx, ny, nz, nt);

        int ibx = x / bx;
        int iby = y / by;
        int ibz = z / bz;
        int ibt = t / bt;
        int jeo = (ibx + iby + ibz + ibt) & 1;

        if (jeo != ieo) {
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site4d)] = 0;
                    d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site4d)] = 0;
                }
            }
            continue;
        }

        Float diag = b5 * (static_cast<Float>(4.0) - M5) + static_cast<Float>(1.0);

        Float psi_cur[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                psi_cur[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site4d)];
                psi_cur[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site4d)];
            }
        }

        int is_p = (is + 1 < Ls) ? (is + 1) : 0;
        int is_m = (is - 1 >= 0) ? (is - 1) : (Ls - 1);
        Float factor_p = (is + 1 < Ls) ? alpha : -mass;
        Float factor_m = (is - 1 >= 0) ? alpha : -mass;

        Float psi_p[Nspin][Nc][Ncomplex], psi_m[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                psi_p[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is_p, Ls, site4d)] * factor_p;
                psi_p[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is_p, Ls, site4d)] * factor_p;

                psi_m[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is_m, Ls, site4d)] * factor_m;
                psi_m[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is_m, Ls, site4d)] * factor_m;
            }
        }

        Float pr_m[Nspin][Nc][Ncomplex], pl_p[Nspin][Nc][Ncomplex];
        proj_pr<Float, GammaRep>(pr_m, psi_m);
        proj_pl<Float, GammaRep>(pl_p, psi_p);

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                Float res_r = diag * psi_cur[s][c][0] - pr_m[s][c][0] - pl_p[s][c][0];
                Float res_i = diag * psi_cur[s][c][1] - pr_m[s][c][1] - pl_p[s][c][1];
                d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site4d)] = res_r;
                d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site4d)] = res_i;
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_DD_CUH
