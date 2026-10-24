#ifndef BRIDGE_CUDA_FERMION_CLOVER_CUH
#define BRIDGE_CUDA_FERMION_CLOVER_CUH

#include "../base/types.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include "hops.cuh"

// The clover operator D = T - kappa hops, Dirac representation.
//
// T, the block of D within a site (1 - kappa c_SW sigma F, or its inverse), commutes with
// gamma_5 and is stored as RutticeQCD stores it: 72 reals per site, the block on gamma_5 = +1
// and then the one on gamma_5 = -1, each a packed Hermitian 6 x 6 matrix (index colour +
// 3 spin): the six real diagonal entries, then the upper triangle row by row as (re, im).
// In the Dirac representation gamma_5 = [[0, 1], [1, 0]], with eigenvectors
// (e_a +- e_(a+2)) / sqrt(2).

namespace bridge_cuda {

// -------------------------------------------------------------------------
// out = T in, with the blocks of T (or of its inverse) in d_clover: on the whole lattice
// (parity < 0), or in and out on the sites of `parity`. The blocks are those of every site of
// the lattice. Every vector of the fields; the fibres of a site (hops.cuh) are its vectors,
// which read the same block.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void clover_term_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_clover,
    int nx, int ny, int nz, int nt,
    int parity,
    int vectors)
{
    const int sites = ((parity >= 0) ? nx / 2 : nx) * ny * nz * nt;
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (at >= sites || v >= vectors) return;
    const Float* __restrict__ in = d_in + v * vector_reals(sites, 1);
    Float* __restrict__ out = d_out + v * vector_reals(sites, 1);
    int site = at;
    if (parity >= 0) {
        int xc[4];
        half_site_to_coords(at, parity, xc, nx, ny, nz);
        site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    }
    Float acc[Nspin][Nc][Ncomplex];
    clover_site<Float>(acc, in, d_clover, at, site);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            out[aosoa_spinor_idx(c, s, 0, at)] = acc[s][c][0];
            out[aosoa_spinor_idx(c, s, 1, at)] = acc[s][c][1];
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_CLOVER_CUH
