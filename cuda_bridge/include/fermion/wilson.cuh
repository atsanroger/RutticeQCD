#ifndef BRIDGE_CUDA_FERMION_WILSON_CUH
#define BRIDGE_CUDA_FERMION_WILSON_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include "hops.cuh"

namespace bridge_cuda {

// -------------------------------------------------------------------------
// Wilson operator: D psi = psi + H psi (hops.cuh), or D^dag (`sign` -1), of every vector of
// the fields. The fibres of a site: its vectors.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void wilson_interior_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    Float kappa,
    Float sign,
    int vectors)
{
    const int vol = nx * ny * nz * nt;
    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (site >= vol || v >= vectors) return;
    const Float* __restrict__ in = d_in + v * vector_reals(vol, 1);
    Float* __restrict__ out = d_out + v * vector_reals(vol, 1);
    Float acc[Nspin][Nc][Ncomplex] = {0};
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    hops<Float>(acc, in, d_gauge, link_rows, xc, 0, 1, dims, bc, false, sign, kappa,
                static_cast<Float>(0.0), static_cast<Float>(1.0), static_cast<Float>(0.0));
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[aosoa_spinor_idx(c, s, 0, site)] = in[aosoa_spinor_idx(c, s, 0, site)] + acc[s][c][0];
            out[aosoa_spinor_idx(c, s, 1, site)] = in[aosoa_spinor_idx(c, s, 1, site)] + acc[s][c][1];
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_WILSON_CUH
