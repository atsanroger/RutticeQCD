#ifndef BRIDGE_CUDA_FERMION_WILSON_CUH
#define BRIDGE_CUDA_FERMION_WILSON_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"

namespace bridge_cuda {



// -------------------------------------------------------------------------
// Wilson Interior Dslash Kernel (Grid-Stride Loop)
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_interior_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    Float kappa)
{
    int vol = nx * ny * nz * nt;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        int x, y, z, t;
        site_to_coords(site, x, y, z, t, nx, ny, nz, nt);

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
// Wilson Even-Odd Hopping Kernel (Grid-Stride Loop over half volume V/2)
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_eo_dslash_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    int target_parity,
    Float kappa)
{
    int vol = nx * ny * nz * nt;
    int vol_eo = vol / 2;

    for (int site_eo = blockIdx.x * blockDim.x + threadIdx.x; site_eo < vol_eo; site_eo += blockDim.x * gridDim.x) {
        int x_half = site_eo % (nx / 2);
        int rem = site_eo / (nx / 2);
        int y = rem % ny; rem /= ny;
        int z = rem % nz;
        int t = rem / nz;
        int x = 2 * x_half + ((target_parity + y + z + t) & 1);
        int site_full = coords_to_site(x, y, z, t, nx, ny, nz, nt);

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
                    u_f_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site_full)];
                    u_f_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site_full)];
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
                d_out[aosoa_spinor_idx(c, s, 0, site_full)] = out_sp[s][c][0];
                d_out[aosoa_spinor_idx(c, s, 1, site_full)] = out_sp[s][c][1];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Wilson Boundary Packing Kernel (Halo Exchange)
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_pack_face_kernel(
    Float* __restrict__ d_face_buf,
    const Float* __restrict__ d_in,
    int dim, int dir,
    int nx, int ny, int nz, int nt)
{
    int face_vol = (nx * ny * nz * nt) / (dim == 0 ? nx : (dim == 1 ? ny : (dim == 2 ? nz : nt)));

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < face_vol; idx += blockDim.x * gridDim.x) {
        int x = 0, y = 0, z = 0, t = 0;
        if (dim == 0) {
            x = (dir == 1) ? (nx - 1) : 0;
            y = idx % ny; int rem = idx / ny;
            z = rem % nz; t = rem / nz;
        } else if (dim == 1) {
            x = idx % nx; int rem = idx / nx;
            y = (dir == 1) ? (ny - 1) : 0;
            z = rem % nz; t = rem / nz;
        } else if (dim == 2) {
            x = idx % nx; int rem = idx / nx;
            y = rem % ny;
            z = (dir == 1) ? (nz - 1) : 0;
            t = rem / ny;
        } else if (dim == 3) {
            x = idx % nx; int rem = idx / nx;
            y = rem % ny; z = rem / ny;
            t = (dir == 1) ? (nt - 1) : 0;
        }

        int site = coords_to_site(x, y, z, t, nx, ny, nz, nt);

        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                d_face_buf[aosoa_spinor_idx(c, s, 0, idx)] = d_in[aosoa_spinor_idx(c, s, 0, site)];
                d_face_buf[aosoa_spinor_idx(c, s, 1, idx)] = d_in[aosoa_spinor_idx(c, s, 1, site)];
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_WILSON_CUH
