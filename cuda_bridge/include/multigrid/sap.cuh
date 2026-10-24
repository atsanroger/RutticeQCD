#ifndef BRIDGE_CUDA_MULTIGRID_SAP_CUH
#define BRIDGE_CUDA_MULTIGRID_SAP_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../fermion/hops.cuh"
#include <cuda_runtime.h>

// The parts of the Schwarz alternating procedure (multigrid/sap.rs) that touch the fields, for
// four-dimensional fields of several vectors (fermion/hops.cuh's layout): the blocks are the
// aggregates of the multigrid, coloured by the parity of their block coordinates; the block
// operator is hop_kernel with a BlockCut. A minimal-residual step on every block of one colour
// and every vector:
//
//   alpha = <p, r>_block / <p, p>_block,   e += alpha r,   r -= alpha p,
//
// the sums in double precision. d_blocks: the colour of every block; d_sites: the block_volume
// sites of every block; d_block_of: the block of every site.

namespace bridge_cuda {

// The sums of sap_alpha_kernel: threads a block (a power of two).
constexpr int SapThreads = 128;

// v = 0 outside the blocks of colour c. Every (site, vector).
template <typename Float>
__global__ void sap_keep_colour_kernel(
    Float* __restrict__ d_v, const unsigned* __restrict__ d_blocks,
    const unsigned* __restrict__ d_block_of, int sites, int vectors, int c)
{
    for (size_t t : GridStride(static_cast<size_t>(sites) * vectors)) {
        const int site = static_cast<int>(t % sites), v = static_cast<int>(t / sites);
        if (d_blocks[d_block_of[site]] == c) continue;
        Float* f = d_v + v * vector_reals(sites, 1);
        #pragma unroll
        for (int comp = 0; comp < Nspinor; ++comp) f[aosoa_idx(comp, site, Nspinor)] = 0;
    }
}

// alpha of block blockIdx.x and vector blockIdx.y, 0 for a block of the other colour or with
// p = 0: (re, im) at 2 (block vectors + vector). Each thread adds its sites of the block in
// order, then a fixed tree over the threads, so the result does not depend on the timing.
template <typename Float>
__global__ void __launch_bounds__(SapThreads) sap_alpha_kernel(
    double* __restrict__ d_alpha, const Float* __restrict__ d_p, const Float* __restrict__ d_r,
    const unsigned* __restrict__ d_blocks, const unsigned* __restrict__ d_sites,
    int block_volume, int sites, int vectors, int c)
{
    __shared__ double pr_re[SapThreads], pr_im[SapThreads], pp[SapThreads];
    const int block = blockIdx.x, v = blockIdx.y, tid = threadIdx.x;
    const size_t out = 2 * (static_cast<size_t>(block) * vectors + v);
    if (d_blocks[block] != c) {
        if (tid == 0) d_alpha[out] = d_alpha[out + 1] = 0.0;
        return;
    }
    const Float* p = d_p + v * vector_reals(sites, 1);
    const Float* r = d_r + v * vector_reals(sites, 1);
    double re = 0.0, im = 0.0, nn = 0.0;
    for (int j = tid; j < block_volume; j += SapThreads) {
        const int site = d_sites[static_cast<size_t>(block) * block_volume + j];
        #pragma unroll
        for (int k = 0; k < Nspinor; k += Ncomplex) {
            const double ar = p[aosoa_idx(k, site, Nspinor)], ai = p[aosoa_idx(k + 1, site, Nspinor)];
            const double br = r[aosoa_idx(k, site, Nspinor)], bi = r[aosoa_idx(k + 1, site, Nspinor)];
            // conj(p) r
            re += ar * br + ai * bi;
            im += ar * bi - ai * br;
            nn += ar * ar + ai * ai;
        }
    }
    pr_re[tid] = re;
    pr_im[tid] = im;
    pp[tid] = nn;
    __syncthreads();
    for (int half = SapThreads / 2; half > 0; half /= 2) {
        if (tid < half) {
            pr_re[tid] += pr_re[tid + half];
            pr_im[tid] += pr_im[tid + half];
            pp[tid] += pp[tid + half];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const bool some = pp[0] > 0.0;
        d_alpha[out] = some ? pr_re[0] / pp[0] : 0.0;
        d_alpha[out + 1] = some ? pr_im[0] / pp[0] : 0.0;
    }
}

// e += alpha r, then r -= alpha p, alpha that of the site's block and the vector. Every
// (site, vector); a block with alpha 0 is left as it is (the same numbers).
template <typename Float>
__global__ void sap_update_kernel(
    Float* __restrict__ d_e, Float* __restrict__ d_r, const Float* __restrict__ d_p,
    const double* __restrict__ d_alpha, const unsigned* __restrict__ d_block_of,
    int sites, int vectors)
{
    for (size_t t : GridStride(static_cast<size_t>(sites) * vectors)) {
        const int site = static_cast<int>(t % sites), v = static_cast<int>(t / sites);
        const size_t at = 2 * (static_cast<size_t>(d_block_of[site]) * vectors + v);
        const Float ar = static_cast<Float>(d_alpha[at]), ai = static_cast<Float>(d_alpha[at + 1]);
        if (ar == 0 && ai == 0) continue;
        const size_t first = v * vector_reals(sites, 1);
        Float* e = d_e + first;
        Float* r = d_r + first;
        const Float* p = d_p + first;
        #pragma unroll
        for (int k = 0; k < Nspinor; k += Ncomplex) {
            const int kr = aosoa_idx(k, site, Nspinor), ki = aosoa_idx(k + 1, site, Nspinor);
            const Float rr = r[kr], ri = r[ki], pr = p[kr], pi = p[ki];
            e[kr] += ar * rr - ai * ri;
            e[ki] += ar * ri + ai * rr;
            r[kr] = rr - (ar * pr - ai * pi);
            r[ki] = ri - (ar * pi + ai * pr);
        }
    }
}

} // namespace bridge_cuda

#endif
