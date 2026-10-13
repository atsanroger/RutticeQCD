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

namespace bridge_cuda {

// -------------------------------------------------------------------------
// out = D in on the whole lattice, in one pass. One thread for each (site, s), as in hop_kernel.
// -------------------------------------------------------------------------
template <typename Float, int LS_PER_BLOCK = 4>
__global__ void mdwf_dslash_1pass_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls,
    Float M5, Float mass, Float b5, Float c5)
{
    const int site = threadIdx.x + WarpSize * blockIdx.x;
    const int is   = blockIdx.y * LS_PER_BLOCK + threadIdx.y;
    if (site >= nx * ny * nz * nt || is >= Ls) return;

    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    const Float d_plus  = static_cast<Float>(1.0) + b5 * (static_cast<Float>(4.0) - M5);
    const Float d_minus = static_cast<Float>(1.0) - c5 * (static_cast<Float>(4.0) - M5);

    Float acc[Nspin][Nc][Ncomplex];
    fifth_term<Float>(acc, d_in, is, Ls, site, mass, d_plus, d_minus, false);
    hops<Float>(acc, d_in, d_gauge, xc, is, Ls, dims, bc, false, static_cast<Float>(1.0),
                static_cast<Float>(0.5), mass, b5, c5);
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
    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < sites; site += blockDim.x * gridDim.x) {
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

// P+ (plus) or P- of the spinor at fifth coordinate `is` of site `site`.
template <typename Float>
__device__ __forceinline__ void mdwf_load_projected(
    Float r[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    int is, int Ls, int site, bool plus)
{
    Float phi[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            phi[s][c][0] = d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)];
            phi[s][c][1] = d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)];
        }
    }
    if (plus) {
        proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(r, phi);
    } else {
        proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(r, phi);
    }
}

// -------------------------------------------------------------------------
// out = C^-1 in (C^dag^-1 with `dagger`), exactly, on `sites` sites (of the whole lattice
// or of one parity). out must not be in.
//
// C x = r is, on P+ x and with rows counted from s = 0 (on P- x: from s = Ls-1; the other
// way round for C^dag),
//
//   d+ x_i - d- x_(i-1) = r_i  (i > 0),     d+ x_0 + m d- x_(Ls-1) = r_0,
//
// so x_i = p_i + q^i x_0 with p_0 = 0, p_i = (r_i + d- p_(i-1)) / d+, q = d- / d+, and
// x_0 = (r_0 - m d- p_(Ls-1)) / (d+ + m d- q^(Ls-1)). The recursion runs twice: for the
// last p, which gives x_0, and then for every x, with no buffer of a site's spinors.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void mdwf_site_inv_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_in,
    int sites,
    int Ls,
    Float M5, Float mass, Float b5, Float c5,
    int dagger)
{
    const Float d_plus  = static_cast<Float>(1.0) + b5 * (static_cast<Float>(4.0) - M5);
    const Float d_minus = static_cast<Float>(1.0) - c5 * (static_cast<Float>(4.0) - M5);
    const Float q = d_minus / d_plus;
    Float q_last = static_cast<Float>(1.0);
    for (int i = 1; i < Ls; ++i) q_last *= q;
    const Float corner   = static_cast<Float>(1.0) / (d_plus + mass * d_minus * q_last);
    const Float per_step = static_cast<Float>(1.0) / d_plus;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < sites; site += blockDim.x * gridDim.x) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool plus = (pass == 0);
            const bool up = (plus != (dagger != 0));
            Float r[Nspin][Nc][Ncomplex], p[Nspin][Nc][Ncomplex], first[Nspin][Nc][Ncomplex];
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) { p[s][c][0] = 0; p[s][c][1] = 0; }
            }
            for (int i = 1; i < Ls; ++i) {
                mdwf_load_projected<Float>(r, d_in, up ? i : Ls - 1 - i, Ls, site, plus);
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        p[s][c][0] = per_step * (r[s][c][0] + d_minus * p[s][c][0]);
                        p[s][c][1] = per_step * (r[s][c][1] + d_minus * p[s][c][1]);
                    }
                }
            }
            mdwf_load_projected<Float>(r, d_in, up ? 0 : Ls - 1, Ls, site, plus);
            #pragma unroll
            for (int s = 0; s < Nspin; ++s) {
                #pragma unroll
                for (int c = 0; c < Nc; ++c) {
                    first[s][c][0] = corner * (r[s][c][0] - mass * d_minus * p[s][c][0]);
                    first[s][c][1] = corner * (r[s][c][1] - mass * d_minus * p[s][c][1]);
                    p[s][c][0] = 0; p[s][c][1] = 0;
                }
            }
            Float power = static_cast<Float>(1.0);
            for (int i = 0; i < Ls; ++i) {
                const int is = up ? i : Ls - 1 - i;
                if (i > 0) {
                    mdwf_load_projected<Float>(r, d_in, is, Ls, site, plus);
                    #pragma unroll
                    for (int s = 0; s < Nspin; ++s) {
                        #pragma unroll
                        for (int c = 0; c < Nc; ++c) {
                            p[s][c][0] = per_step * (r[s][c][0] + d_minus * p[s][c][0]);
                            p[s][c][1] = per_step * (r[s][c][1] + d_minus * p[s][c][1]);
                        }
                    }
                }
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        #pragma unroll
                        for (int ri = 0; ri < Ncomplex; ++ri) {
                            const Float x = p[s][c][ri] + power * first[s][c][ri];
                            const int k = aosoa_5d_spinor_idx(c, s, ri, is, Ls, site);
                            d_out[k] = plus ? x : d_out[k] + x;
                        }
                    }
                }
                power *= q;
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
    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < sites; site += blockDim.x * gridDim.x) {
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

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_MDWF_CUH
