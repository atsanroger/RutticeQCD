#ifndef BRIDGE_CUDA_FERMION_WILSON_CUH
#define BRIDGE_CUDA_FERMION_WILSON_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"

namespace bridge_cuda {

// -------------------------------------------------------------------------
// The hops of the Wilson operator at one site, Dirac representation:
//
//   acc -= kappa sum_mu [ (1 - gamma_mu) U_mu(x) psi(x+mu) + (1 + gamma_mu) U_mu(x-mu)^dag psi(x-mu) ]
//
// bc[mu] multiplies a spinor that arrives across the boundary in direction mu
// (+1 periodic, -1 antiperiodic, 0 Dirichlet); the links are periodic.
// -------------------------------------------------------------------------
template <typename Float>
__device__ __forceinline__ void wilson_hops(
    Float acc[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int site,
    const int dims[4],
    const Float bc[4],
    Float kappa)
{
    const int nx = dims[0], ny = dims[1], nz = dims[2], nt = dims[3];
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
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
        const Float sign_f = wrap_f ? bc[mu] : static_cast<Float>(1.0);
        const Float sign_b = wrap_b ? bc[mu] : static_cast<Float>(1.0);

        Float sp_f[Nspin][Nc][Ncomplex], sp_b[Nspin][Nc][Ncomplex];
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                sp_f[s][c][0] = sign_f * d_in[aosoa_spinor_idx(c, s, 0, site_f)];
                sp_f[s][c][1] = sign_f * d_in[aosoa_spinor_idx(c, s, 1, site_f)];
                sp_b[s][c][0] = sign_b * d_in[aosoa_spinor_idx(c, s, 0, site_b)];
                sp_b[s][c][1] = sign_b * d_in[aosoa_spinor_idx(c, s, 1, site_b)];
            }
        }
        // (1 - gamma_mu) psi(x+mu) and (1 + gamma_mu) psi(x-mu)
        Float g_f[Nspin][Nc][Ncomplex], g_b[Nspin][Nc][Ncomplex];
        mult_gamma_dirac<Float>(g_f, sp_f, mu);
        mult_gamma_dirac<Float>(g_b, sp_b, mu);

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
                v_in_r[c] = sp_f[s][c][0] - g_f[s][c][0];
                v_in_i[c] = sp_f[s][c][1] - g_f[s][c][1];
            }
            su3_mult_vec(v_out_r, v_out_i, u_f_r, u_f_i, v_in_r, v_in_i);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                acc[s][c][0] -= kappa * v_out_r[c];
                acc[s][c][1] -= kappa * v_out_i[c];
            }
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                v_in_r[c] = sp_b[s][c][0] + g_b[s][c][0];
                v_in_i[c] = sp_b[s][c][1] + g_b[s][c][1];
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
// Wilson operator (Grid-Stride Loop): D psi = psi + hops (see wilson_hops).
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_interior_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    Float kappa)
{
    const int vol = nx * ny * nz * nt;
    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        Float acc[Nspin][Nc][Ncomplex] = {0};
        wilson_hops<Float>(acc, d_in, d_gauge, site, dims, bc, kappa);
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                d_out[aosoa_spinor_idx(c, s, 0, site)] = d_in[aosoa_spinor_idx(c, s, 0, site)] + acc[s][c][0];
                d_out[aosoa_spinor_idx(c, s, 1, site)] = d_in[aosoa_spinor_idx(c, s, 1, site)] + acc[s][c][1];
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
