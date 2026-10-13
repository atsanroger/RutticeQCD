#ifndef BRIDGE_CUDA_FERMION_MDWF_CUH
#define BRIDGE_CUDA_FERMION_MDWF_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"

namespace bridge_cuda {

// -------------------------------------------------------------------------
// 5D Mobius Domain Wall s-direction hopping and chiral projection (D_5)
// -------------------------------------------------------------------------
template <typename Float, int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL>
__global__ void mdwf_d5_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int vol4d,
    int Ls,
    Float M5,
    Float mass,
    Float alpha,
    Float b5,
    Float c5)
{
    int vol5d = vol4d * Ls;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vol5d; idx += blockDim.x * gridDim.x) {
        int site4d = idx % vol4d;
        int is     = idx / vol4d;

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

// -------------------------------------------------------------------------
// 5D Mobius Domain Wall Even-Odd Hopping Kernel with Ls Parallelism
// -------------------------------------------------------------------------
template <typename Float, int LS_PER_BLOCK = 4>
__global__ void mdwf_eo_hopb_5d_v2_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_gauge,
    const Float* __restrict__ d_in,
    int vol4d_eo,
    int Ls,
    int nx, int ny, int nz, int nt,
    int ieo, int jeo,
    Float alpha)
{
    const int lane     = threadIdx.x;
    const int is_local = threadIdx.y;
    const int idx_out  = blockIdx.x;
    const int is_base  = blockIdx.y * LS_PER_BLOCK;
    const int is       = is_base + is_local;
    const int site_eo  = lane + W * idx_out;

    if (site_eo >= vol4d_eo || is >= Ls) return;

    __shared__ Float u_sh[18 * W];

    Float sp_in[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            sp_in[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_eo)];
            sp_in[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_eo)];
        }
    }

    Float out_sp[Nspin][Nc][Ncomplex] = {0};

    for (int mu = 0; mu < Ndim; ++mu) {
        if (is_local == 0) {
            #pragma unroll
            for (int r = 0; r < Nc; ++r) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    u_sh[lane + W * (2 * (c + 3 * r) + 0)] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site_eo)];
                    u_sh[lane + W * (2 * (c + 3 * r) + 1)] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site_eo)];
                }
            }
        }
        __syncthreads();

        Float u_r[Nc][Nc], u_i[Nc][Nc];
        #pragma unroll
        for (int r = 0; r < Nc; ++r) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                u_r[r][c] = u_sh[lane + W * (2 * (c + 3 * r) + 0)];
                u_i[r][c] = u_sh[lane + W * (2 * (c + 3 * r) + 1)];
            }
        }

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            Float v_in_r[Nc], v_in_i[Nc], v_out_r[Nc], v_out_i[Nc];
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sp_in[s][c][0];
                v_in_i[c] = sp_in[s][c][1];
            }
            su3_mult_vec(v_out_r, v_out_i, u_r, u_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                out_sp[s][c][0] -= alpha * v_out_r[c];
                out_sp[s][c][1] -= alpha * v_out_i[c];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_eo)] = out_sp[s][c][0];
            d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_eo)] = out_sp[s][c][1];
        }
    }
}

// -------------------------------------------------------------------------
// 5D Mobius Domain Wall Even-Odd LU-Inversion (L^{-1} o U^{-1})
// -------------------------------------------------------------------------
template <typename Float, int MAX_LS = 32>
__global__ void mdwf_ee_LUinv_5d_v2_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int vol4d_eo,
    int Ls,
    Float M5,
    Float mass,
    Float alpha,
    Float b5,
    Float c5)
{
    for (int site_eo = blockIdx.x * blockDim.x + threadIdx.x; site_eo < vol4d_eo; site_eo += blockDim.x * gridDim.x) {
        Float diag = b5 * (static_cast<Float>(4.0) - M5) + static_cast<Float>(1.0);
        Float diag_inv = static_cast<Float>(1.0) / diag;

        Float vt_fwd[MAX_LS][Nspin][Nc][Ncomplex];

        // Forward Substitution (L-inv)
        for (int is = 0; is < Ls; ++is) {
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    Float in_r = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_eo)];
                    Float in_i = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_eo)];

                    if (is == 0) {
                        vt_fwd[0][s][c][0] = in_r * diag_inv;
                        vt_fwd[0][s][c][1] = in_i * diag_inv;
                    } else {
                        Float prev_r = (s < 2) ? vt_fwd[is - 1][s][c][0] : static_cast<Float>(0.0);
                        Float prev_i = (s < 2) ? vt_fwd[is - 1][s][c][1] : static_cast<Float>(0.0);

                        vt_fwd[is][s][c][0] = (in_r + alpha * prev_r) * diag_inv;
                        vt_fwd[is][s][c][1] = (in_i + alpha * prev_i) * diag_inv;
                    }
                }
            }
        }

        // Backward Substitution (U-inv)
        for (int is = Ls - 1; is >= 0; --is) {
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    Float cur_r = vt_fwd[is][s][c][0];
                    Float cur_i = vt_fwd[is][s][c][1];

                    if (is < Ls - 1 && s >= 2) {
                        cur_r += alpha * vt_fwd[is + 1][s][c][0] * diag_inv;
                        cur_i += alpha * vt_fwd[is + 1][s][c][1] * diag_inv;
                    }

                    d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_eo)] = cur_r;
                    d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_eo)] = cur_i;
                }
            }
        }
    }
}


// -------------------------------------------------------------------------
// Fused 1-Pass Möbius Domain Wall Dslash Kernel
// Computes D_MDWF psi = (local_5D) + D_4(b_5, c_5) in a single pass without
// any intermediate memory writes or reads.
// -------------------------------------------------------------------------
template <typename Float, int LS_PER_BLOCK = 4, int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL>
__global__ void mdwf_dslash_1pass_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int vol4d,
    int Ls,
    int nx, int ny, int nz, int nt,
    Float M5,
    Float mass,
    Float alpha,
    Float b5,
    Float c5)
{
    const int lane     = threadIdx.x;
    const int is_local = threadIdx.y;
    const int outer    = blockIdx.x;
    const int is_base  = blockIdx.y * LS_PER_BLOCK;
    const int is       = is_base + is_local;
    const int site     = lane + WarpSize * outer;

    if (site >= vol4d || is >= Ls) return;

    int x, y, z, t;
    site_to_coords(site, x, y, z, t, nx, ny, nz, nt);

    Float diag    = b5 * (static_cast<Float>(4.0) - M5) + static_cast<Float>(1.0);
    Float d_minus = static_cast<Float>(1.0) - c5 * (static_cast<Float>(4.0) - M5);

    int is_p = (is + 1 < Ls) ? (is + 1) : 0;
    int is_m = (is - 1 >= 0) ? (is - 1) : (Ls - 1);
    Float factor_p = (is + 1 < Ls) ? alpha : -mass;
    Float factor_m = (is - 1 >= 0) ? alpha : -mass;

    Float psi_cur[Nspin][Nc][Ncomplex];
    Float psi_p[Nspin][Nc][Ncomplex], psi_m[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            psi_cur[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)];
            psi_cur[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)];
            psi_p[s][c][0]   = d_in[aosoa_5d_spinor_idx(c, s, 0, is_p, Ls, site)] * factor_p;
            psi_p[s][c][1]   = d_in[aosoa_5d_spinor_idx(c, s, 1, is_p, Ls, site)] * factor_p;
            psi_m[s][c][0]   = d_in[aosoa_5d_spinor_idx(c, s, 0, is_m, Ls, site)] * factor_m;
            psi_m[s][c][1]   = d_in[aosoa_5d_spinor_idx(c, s, 1, is_m, Ls, site)] * factor_m;
        }
    }

    Float pr_m[Nspin][Nc][Ncomplex], pl_p[Nspin][Nc][Ncomplex];
    proj_pr<Float, GammaRep>(pr_m, psi_m);
    proj_pl<Float, GammaRep>(pl_p, psi_p);

    Float out_sp[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out_sp[s][c][0] = diag * psi_cur[s][c][0] - d_minus * (pr_m[s][c][0] + pl_p[s][c][0]);
            out_sp[s][c][1] = diag * psi_cur[s][c][1] - d_minus * (pr_m[s][c][1] + pl_p[s][c][1]);
        }
    }

    __shared__ Float u_sh[18 * WarpSize];
    Float kappa_5 = static_cast<Float>(0.5);

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

        Float u_b_r[Nc][Nc], u_b_i[Nc][Nc];
        #pragma unroll
        for (int r = 0; r < Nc; ++r) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                u_b_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site_b)];
                u_b_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site_b)];
            }
        }

        Float phi_f[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                Float cur_r = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_f)];
                Float cur_i = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_f)];
                phi_f[s][c][0] = b5 * cur_r;
                phi_f[s][c][1] = b5 * cur_i;
            }
        }
        if (c5 != static_cast<Float>(0.0)) {
            Float f_p[Nspin][Nc][Ncomplex], f_m[Nspin][Nc][Ncomplex];
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    f_p[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is_p, Ls, site_f)] * factor_p;
                    f_p[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is_p, Ls, site_f)] * factor_p;
                    f_m[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is_m, Ls, site_f)] * factor_m;
                    f_m[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is_m, Ls, site_f)] * factor_m;
                }
            }
            Float p_rm[Nspin][Nc][Ncomplex], p_lp[Nspin][Nc][Ncomplex];
            proj_pr<Float, GammaRep>(p_rm, f_m);
            proj_pl<Float, GammaRep>(p_lp, f_p);
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    phi_f[s][c][0] += c5 * (p_rm[s][c][0] + p_lp[s][c][0]);
                    phi_f[s][c][1] += c5 * (p_rm[s][c][1] + p_lp[s][c][1]);
                }
            }
        }

        Float phi_b[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                Float cur_r = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site_b)];
                Float cur_i = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site_b)];
                phi_b[s][c][0] = b5 * cur_r;
                phi_b[s][c][1] = b5 * cur_i;
            }
        }
        if (c5 != static_cast<Float>(0.0)) {
            Float b_p[Nspin][Nc][Ncomplex], b_m[Nspin][Nc][Ncomplex];
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    b_p[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is_p, Ls, site_b)] * factor_p;
                    b_p[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is_p, Ls, site_b)] * factor_p;
                    b_m[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is_m, Ls, site_b)] * factor_m;
                    b_m[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is_m, Ls, site_b)] * factor_m;
                }
            }
            Float p_rm[Nspin][Nc][Ncomplex], p_lp[Nspin][Nc][Ncomplex];
            proj_pr<Float, GammaRep>(p_rm, b_m);
            proj_pl<Float, GammaRep>(p_lp, b_p);
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    phi_b[s][c][0] += c5 * (p_rm[s][c][0] + p_lp[s][c][0]);
                    phi_b[s][c][1] += c5 * (p_rm[s][c][1] + p_lp[s][c][1]);
                }
            }
        }

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            Float v_in_r[Nc], v_in_i[Nc], v_out_r[Nc], v_out_i[Nc];
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = phi_f[s][c][0];
                v_in_i[c] = phi_f[s][c][1];
            }
            su3_mult_vec(v_out_r, v_out_i, u_f_r, u_f_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                out_sp[s][c][0] -= kappa_5 * v_out_r[c];
                out_sp[s][c][1] -= kappa_5 * v_out_i[c];
            }

            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = phi_b[s][c][0];
                v_in_i[c] = phi_b[s][c][1];
            }
            su3_dagger_mult_vec(v_out_r, v_out_i, u_b_r, u_b_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                out_sp[s][c][0] -= kappa_5 * v_out_r[c];
                out_sp[s][c][1] -= kappa_5 * v_out_i[c];
            }
        }
        __syncthreads();
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
} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_MDWF_CUH
