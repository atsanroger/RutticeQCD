#ifndef BRIDGE_CUDA_MULTIGRID_COARSE_CUH
#define BRIDGE_CUDA_MULTIGRID_COARSE_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"

namespace bridge_cuda {

template <typename Float, int MAX_COARSE_COLOR = 24>
__global__ void mg_coarse_dslash_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_clover_inv,
    const Float* __restrict__ d_coarse_links,
    int nx_c, int ny_c, int nz_c, int nt_c,
    int num_colors)
{
    int vol_c = nx_c * ny_c * nz_c * nt_c;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol_c; site += blockDim.x * gridDim.x) {
        int x = site % nx_c; int rem = site / nx_c;
        int y = rem % ny_c; rem = rem / ny_c;
        int z = rem % nz_c; int t = rem / nz_c;

        Float v_out_r[MAX_COARSE_COLOR] = {0};
        Float v_out_i[MAX_COARSE_COLOR] = {0};

        // 8-direction coarse Dirac hopping
        for (int mu = 0; mu < 4; ++mu) {
            int x_f = (mu == 0) ? (x + 1) % nx_c : x;
            int y_f = (mu == 1) ? (y + 1) % ny_c : y;
            int z_f = (mu == 2) ? (z + 1) % nz_c : z;
            int t_f = (mu == 3) ? (t + 1) % nt_c : t;
            int site_f = x_f + nx_c * (y_f + ny_c * (z_f + nz_c * t_f));

            for (int r = 0; r < num_colors; ++r) {
                for (int c = 0; c < num_colors; ++c) {
                    int link_idx = (site * 8 + mu) * num_colors * num_colors + (r * num_colors + c);
                    Float u_r = d_coarse_links[link_idx * 2 + 0];
                    Float u_i = d_coarse_links[link_idx * 2 + 1];

                    int in_idx = site_f * num_colors + c;
                    Float in_r = d_in[in_idx * 2 + 0];
                    Float in_i = d_in[in_idx * 2 + 1];

                    v_out_r[r] += u_r * in_r - u_i * in_i;
                    v_out_i[r] += u_r * in_i + u_i * in_r;
                }
            }
        }

        // Write out
        for (int c = 0; c < num_colors; ++c) {
            int out_idx = site * num_colors + c;
            d_out[out_idx * 2 + 0] = v_out_r[c];
            d_out[out_idx * 2 + 1] = v_out_i[c];
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_COARSE_CUH
