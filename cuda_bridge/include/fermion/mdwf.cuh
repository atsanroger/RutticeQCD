#ifndef BRIDGE_CUDA_FERMION_MDWF_CUH
#define BRIDGE_CUDA_FERMION_MDWF_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include "hops.cuh"
#include <cuda_runtime.h>

// The plain Möbius domain-wall operator (Neff's alpha = 1), Dirac representation. With
// P+- = (1 +- gamma_5)/2 and the hop in the fifth dimension
//
//   (Y psi)_s = - P- psi_{s+1} - P+ psi_{s-1}     (+ m instead of - 1 across the walls),
//
//   D psi = d+ psi + d- Y psi + H (b - c Y) psi,   d+ = 1 + b (4 - M5),   d- = 1 - c (4 - M5),
//
// and D^dag psi = d+ psi + d- Y^dag psi + (b - c Y^dag) H^dag psi, with H the hops of
// hops.cuh at kappa = 1/2. C = d+ + d- Y is the part of D within a site of the
// four-dimensional lattice. The fields are those of hops.cuh.
//
// What acts within a site (C and its inverse, Neff's Q, the factor b - c Y^dag of the adjoint
// hops, and their products) is a matrix product: fifth_gemm.cuh.

namespace bridge_cuda {

// -------------------------------------------------------------------------
// out = D in on the whole lattice, in one pass. One thread for each (site, vector, s), as in
// hop_kernel; with several processes its interior or its surface (`P`, as in hop_kernel).
// -------------------------------------------------------------------------
template <typename Float, bool Many, int P = Whole>
__global__ void mdwf_dslash_1pass_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls, int vectors,
    Float M5, Float mass, Float b5, Float c5,
    Halo<Float> halo = {})
{
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int fibre = blockIdx.y * blockDim.y + threadIdx.y;
    if (thread >= (P == Surface ? halo.surface_sites : nx * ny * nz * nt) || fibre >= Ls * vectors) return;
    const int site = (P == Surface) ? halo.surface[thread] : thread;
    const int is = Many ? fibre % Ls : fibre;
    const size_t first = Many ? (fibre / Ls) * vector_reals(nx * ny * nz * nt, Ls) : 0;
    d_out += first;
    d_in += first;

    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    if (P == Interior && on_surface(halo, xc, dims)) return;
    const Float d_plus  = static_cast<Float>(1.0) + b5 * (static_cast<Float>(4.0) - M5);
    const Float d_minus = static_cast<Float>(1.0) - c5 * (static_cast<Float>(4.0) - M5);

    Float acc[Nspin][Nc][Ncomplex];
    fifth_term<Float>(acc, d_in, is, Ls, site, mass, d_plus, d_minus, false);
    if constexpr (P == Surface) {
        hops<Float, true>(acc, d_in, d_gauge, link_rows, xc, is, Ls, dims, bc, false,
                          static_cast<Float>(1.0), static_cast<Float>(0.5), mass, b5, c5, 0xFF,
                          &halo, Many ? fibre / Ls : 0);
    } else {
        hops<Float>(acc, d_in, d_gauge, link_rows, xc, is, Ls, dims, bc, false, static_cast<Float>(1.0),
                    static_cast<Float>(0.5), mass, b5, c5);
    }
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)] = acc[s][c][0];
            d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)] = acc[s][c][1];
        }
    }
}

// -------------------------------------------------------------------------
// out = (a1 + y1 Y) x + (a2 + y2 Y) z, or with Y^dag, on `sites` sites; d_z may be null
// (no second term). out must be neither x nor z.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void mdwf_fifth_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_x,
    const Float* __restrict__ d_z,
    int sites,
    int Ls,
    Float mass,
    Float a1, Float y1, Float a2, Float y2,
    int dagger)
{
    for (size_t at : GridStride(sites)) {
        const int site = static_cast<int>(at);
        for (int is = 0; is < Ls; ++is) {
            Float acc[Nspin][Nc][Ncomplex];
            fifth_term<Float>(acc, d_x, is, Ls, site, mass, a1, y1, dagger != 0);
            if (d_z != nullptr) {
                Float second[Nspin][Nc][Ncomplex];
                fifth_term<Float>(second, d_z, is, Ls, site, mass, a2, y2, dagger != 0);
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        acc[s][c][0] += second[s][c][0];
                        acc[s][c][1] += second[s][c][1];
                    }
                }
            }
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)] = acc[s][c][0];
                    d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)] = acc[s][c][1];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// out = gamma_5 in on `sites` sites of Ls spinors (Ls = 1: a four-dimensional field); with
// `reflect` the fifth dimension is reflected as well, out_s = gamma_5 in_(Ls-1-s), which
// is the Gamma_5 of domain-wall fermions. out must not be in.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void gamma5_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int sites,
    int Ls,
    int reflect)
{
    for (size_t at : GridStride(sites)) {
        const int site = static_cast<int>(at);
        for (int is = 0; is < Ls; ++is) {
            const int from = reflect ? Ls - 1 - is : is;
            Float phi[Nspin][Nc][Ncomplex], g[Nspin][Nc][Ncomplex];
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    phi[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, from, Ls, site)];
                    phi[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, from, Ls, site)];
                }
            }
            mult_gamma5<Float, BRIDGE_CUDA_GAMMA_DIRAC>(g, phi);
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    d_out[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)] = g[s][c][0];
                    d_out[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)] = g[s][c][1];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Between a field of one spinor per site and the walls of one of Ls, on the `sites` sites of
// the lattice, for every vector (the fibres of a site, hops.cuh):
//
//   to_4d:   out = P- in_(s_minus) + P+ in_(s_plus)
//   from_4d: out_(s_minus) = P- in, out_(s_plus) = P+ in; the other s are left as they are.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void walls_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int sites,
    int Ls,
    int s_minus, int s_plus,
    int to_4d,
    int vectors)
{
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (site >= sites || v >= vectors) return;
    const size_t four = v * vector_reals(sites, 1), five = v * vector_reals(sites, Ls);
    const Float* __restrict__ in = d_in + (to_4d ? five : four);
    Float* __restrict__ out = d_out + (to_4d ? four : five);

    Float m[Nspin][Nc][Ncomplex], p[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                if (to_4d) {
                    m[s][c][ri] = in[aosoa_5d_spinor_idx(c, s, ri, s_minus, Ls, site)];
                    p[s][c][ri] = in[aosoa_5d_spinor_idx(c, s, ri, s_plus, Ls, site)];
                } else {
                    m[s][c][ri] = in[aosoa_spinor_idx(c, s, ri, site)];
                    p[s][c][ri] = m[s][c][ri];
                }
            }
        }
    }
    Float minus[Nspin][Nc][Ncomplex], plus[Nspin][Nc][Ncomplex];
    proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(minus, m);
    proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(plus, p);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                if (to_4d) {
                    out[aosoa_spinor_idx(c, s, ri, site)] = minus[s][c][ri] + plus[s][c][ri];
                } else {
                    out[aosoa_5d_spinor_idx(c, s, ri, s_minus, Ls, site)] = minus[s][c][ri];
                    out[aosoa_5d_spinor_idx(c, s, ri, s_plus, Ls, site)] = plus[s][c][ri];
                }
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_MDWF_CUH
