#ifndef BRIDGE_CUDA_GAUGE_GAUGE_CUH
#define BRIDGE_CUDA_GAUGE_GAUGE_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Plaquette Measurement Kernel (Deterministic Block-Tree Reduction, no atomicAdd)
// -------------------------------------------------------------------------
template <typename Float, int BLOCK_SIZE = 256>
__global__ void gauge_plaquette_partial_kernel(
    Float* __restrict__ block_sums,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt)
{
    __shared__ Float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int vol = nx * ny * nz * nt;
    Float local_plaq = 0;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        int x = site % nx; int rem = site / nx;
        int y = rem % ny; rem = rem / ny;
        int z = rem % nz; int t = rem / nz;

        for (int mu = 0; mu < Ndim; ++mu) {
            for (int nu = mu + 1; nu < Ndim; ++nu) {
                int x_mu = (mu == 0) ? (x + 1) % nx : x;
                int y_mu = (mu == 1) ? (y + 1) % ny : y;
                int z_mu = (mu == 2) ? (z + 1) % nz : z;
                int t_mu = (mu == 3) ? (t + 1) % nt : t;
                int site_mu = x_mu + nx * (y_mu + ny * (z_mu + nz * t_mu));

                int x_nu = (nu == 0) ? (x + 1) % nx : x;
                int y_nu = (nu == 1) ? (y + 1) % ny : y;
                int z_nu = (nu == 2) ? (z + 1) % nz : z;
                int t_nu = (nu == 3) ? (t + 1) % nt : t;
                int site_nu = x_nu + nx * (y_nu + ny * (z_nu + nz * t_nu));

                Float u1_r[Nc][Nc], u1_i[Nc][Nc];
                #pragma unroll
                for (int r = 0; r < Nc; ++r) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        u1_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site)];
                        u1_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site)];
                    }
                }

                Float u2_r[Nc][Nc], u2_i[Nc][Nc];
                #pragma unroll
                for (int r = 0; r < Nc; ++r) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        u2_r[r][c] = d_gauge[aosoa_gauge_idx(nu, r, c, 0, site_mu)];
                        u2_i[r][c] = d_gauge[aosoa_gauge_idx(nu, r, c, 1, site_mu)];
                    }
                }

                Float u3_r[Nc][Nc], u3_i[Nc][Nc];
                #pragma unroll
                for (int r = 0; r < Nc; ++r) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        u3_r[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 0, site_nu)];
                        u3_i[r][c] = d_gauge[aosoa_gauge_idx(mu, r, c, 1, site_nu)];
                    }
                }

                Float u4_r[Nc][Nc], u4_i[Nc][Nc];
                #pragma unroll
                for (int r = 0; r < Nc; ++r) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        u4_r[r][c] = d_gauge[aosoa_gauge_idx(nu, r, c, 0, site)];
                        u4_i[r][c] = d_gauge[aosoa_gauge_idx(nu, r, c, 1, site)];
                    }
                }

                Float tr_re = 0;
                #pragma unroll
                for (int a = 0; a < 3; ++a) {
                    #pragma unroll
                    for (int b = 0; b < 3; ++b) {
                        #pragma unroll
                        for (int c = 0; c < Nc; ++c) {
                            #pragma unroll
                            for (int d = 0; d < 3; ++d) {
                                Float prod1_r, prod1_i, prod2_r, prod2_i, prod_r, prod_i;
                                c_mult(prod1_r, prod1_i, u1_r[a][b], u1_i[a][b], u2_r[b][c], u2_i[b][c]);
                                c_mult(prod2_r, prod2_i, u3_r[d][c], -u3_i[d][c], u4_r[a][d], -u4_i[a][d]);
                                c_mult(prod_r, prod_i, prod1_r, prod1_i, prod2_r, prod2_i);
                                tr_re += prod_r;
                            }
                        }
                    }
                }
                local_plaq += tr_re;
            }
        }
    }

    sdata[tid] = local_plaq;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// -------------------------------------------------------------------------
// APE Smearing Kernel
// -------------------------------------------------------------------------
template <typename Float>
__global__ void gauge_ape_smear_kernel(
    Float* __restrict__ d_gauge_out,
    const Float* __restrict__ d_gauge_in,
    int nx, int ny, int nz, int nt,
    Float alpha)
{
    int vol = nx * ny * nz * nt;
    Float weight_u = static_cast<Float>(1.0) - alpha;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        for (int mu = 0; mu < Ndim; ++mu) {
            #pragma unroll
            for (int r = 0; r < Nc; ++r) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    Float u_r = d_gauge_in[aosoa_gauge_idx(mu, r, c, 0, site)];
                    Float u_i = d_gauge_in[aosoa_gauge_idx(mu, r, c, 1, site)];

                    d_gauge_out[aosoa_gauge_idx(mu, r, c, 0, site)] = weight_u * u_r;
                    d_gauge_out[aosoa_gauge_idx(mu, r, c, 1, site)] = weight_u * u_i;
                }
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_GAUGE_GAUGE_CUH
