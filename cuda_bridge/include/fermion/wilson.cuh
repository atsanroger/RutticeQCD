#ifndef BRIDGE_CUDA_FERMION_WILSON_CUH
#define BRIDGE_CUDA_FERMION_WILSON_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include "hops.cuh"

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Wilson operator (Grid-Stride Loop): D psi = psi + H psi (hops.cuh), or D^dag (`sign` -1).
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_interior_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    Float kappa,
    Float sign)
{
    const int vol = nx * ny * nz * nt;
    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol; site += blockDim.x * gridDim.x) {
        Float acc[Nspin][Nc][Ncomplex] = {0};
        int xc[4];
        site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
        hops<Float>(acc, d_in, d_gauge, link_rows, xc, 0, 1, dims, bc, false, sign, kappa,
                    static_cast<Float>(0.0), static_cast<Float>(1.0), static_cast<Float>(0.0));
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
