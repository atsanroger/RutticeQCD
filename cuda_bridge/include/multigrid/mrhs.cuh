#ifndef BRIDGE_CUDA_MULTIGRID_MRHS_CUH
#define BRIDGE_CUDA_MULTIGRID_MRHS_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../fermion/wilson.cuh"

namespace bridge_cuda {

template <typename Float, int MAX_RHS = 8>
__global__ void mrhs_wilson_dslash_kernel(
    Float* const* __restrict__ d_out_ptrs,
    const Float* const* __restrict__ d_in_ptrs,
    const Float* __restrict__ d_gauge,
    int num_rhs,
    int nx, int ny, int nz, int nt,
    Float kappa)
{
    int vol = nx * ny * nz * nt;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        int x, y, z, t;
        site_to_coords(site, x, y, z, t, nx, ny, nz, nt);

        Float out_sp[MAX_RHS][Nspin][Nc][Ncomplex] = {0};

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

            for (int irhs = 0; irhs < num_rhs; ++irhs) {
                const Float* in_ptr = d_in_ptrs[irhs];

                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    Float v_f_r[Nc], v_f_i[Nc], v_b_r[Nc], v_b_i[Nc];
                    Float hop_f_r[Nc], hop_f_i[Nc], hop_b_r[Nc], hop_b_i[Nc];

                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        v_f_r[c] = in_ptr[aosoa_spinor_idx(c, s, 0, site_f)];
                        v_f_i[c] = in_ptr[aosoa_spinor_idx(c, s, 1, site_f)];
                        v_b_r[c] = in_ptr[aosoa_spinor_idx(c, s, 0, site_b)];
                        v_b_i[c] = in_ptr[aosoa_spinor_idx(c, s, 1, site_b)];
                    }

                    su3_mult_vec(hop_f_r, hop_f_i, u_f_r, u_f_i, v_f_r, v_f_i);
                    su3_dagger_mult_vec(hop_b_r, hop_b_i, u_b_r, u_b_i, v_b_r, v_b_i);

                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        out_sp[irhs][s][c][0] -= kappa * (hop_f_r[c] + hop_b_r[c]);
                        out_sp[irhs][s][c][1] -= kappa * (hop_f_i[c] + hop_b_i[c]);
                    }
                }
            }
        }

        for (int irhs = 0; irhs < num_rhs; ++irhs) {
            Float* out_ptr = d_out_ptrs[irhs];
            const Float* in_ptr = d_in_ptrs[irhs];

            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    Float self_r = in_ptr[aosoa_spinor_idx(c, s, 0, site)];
                    Float self_i = in_ptr[aosoa_spinor_idx(c, s, 1, site)];
                    out_ptr[aosoa_spinor_idx(c, s, 0, site)] = self_r + out_sp[irhs][s][c][0];
                    out_ptr[aosoa_spinor_idx(c, s, 1, site)] = self_i + out_sp[irhs][s][c][1];
                }
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_MRHS_CUH
