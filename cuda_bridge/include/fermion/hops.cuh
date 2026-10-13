#ifndef BRIDGE_CUDA_FERMION_HOPS_CUH
#define BRIDGE_CUDA_FERMION_HOPS_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../base/su3.cuh"
#include "../base/gamma.cuh"
#include <cuda_runtime.h>

// What the Wilson, clover and domain-wall operators share, in the Dirac representation: the
// hops of the Wilson kernel,
//
//   H w = - kappa sum_mu [ (1 - gamma_mu) U_mu(x) w(x+mu) + (1 + gamma_mu) U_mu(x-mu)^dag w(x-mu) ],
//
// H^dag being H with the signs of gamma_mu exchanged, and the hop in the fifth dimension
// of domain-wall fermions,
//
//   (Y psi)_s = - P- psi_{s+1} - P+ psi_{s-1}     (+ m instead of - 1 across the walls),
//
// with P+- = (1 +- gamma_5)/2.
//
// A field has Ls spinors per site (1 for a four-dimensional field), on the whole lattice or
// on the sites of one parity: site (x, y, z, t) of parity p = (x + y + z + t) & 1 is site
// x/2 + (nx/2)(y + ny (z + nz t)) of the field of parity p (nx even). The links are those of
// the whole lattice. bc[mu] multiplies a spinor that arrives across the boundary in
// direction mu (+1 periodic, -1 antiperiodic, 0 Dirichlet); the links are periodic.
//
// A field may hold several vectors, each as above, one after the other (vector_reals). The
// kernels take all of them in one launch: a thread for each (site, fibre), a fibre being one
// spinor of a site, of one vector at one s. threadIdx.x is the site and threadIdx.y the fibre
// within the block of threads (site_block), so the threads of a warp read neighbouring sites
// of a field, and the fibres of a site in a block the same links.

namespace bridge_cuda {

// Reals of one vector of a field of `sites` sites with Ls spinors each.
__host__ __device__ __forceinline__ size_t vector_reals(int sites, int Ls) {
    return static_cast<size_t>((sites + WarpSize - 1) / WarpSize) * WarpSize * Ls * (Nspin * Nc * Ncomplex);
}

// The limit of a block of threads: a thread of these kernels may use every register it can
// have (255), and a block has 65536 of them, so with more than 256 threads the launch can fail.
constexpr int MaxThreads = 256;
// Fibres of a site in one block of threads.
constexpr int MaxFibres = 32;

// The block of threads for fields of `vectors` vectors with Ls spinors to a site: as many
// sites as MaxThreads allow, a power of two, and of each
// - in four dimensions its vectors (up to MaxFibres of them), which read the same links;
// - in five the Ls spinors of one vector. More vectors in the block would leave it fewer
//   sites, and a warp fewer neighbouring reals of a field to read at once: that costs more
//   than the links read again.
inline dim3 site_block(int Ls, int vectors) {
    const int fibres = (Ls > 1) ? Ls : vectors;
    int per = fibres < MaxFibres ? fibres : MaxFibres;
    int lanes = 1;
    while (2 * lanes * per <= MaxThreads) lanes *= 2;
    return dim3(lanes, per);
}
inline dim3 site_grid(int sites, int Ls, int vectors) {
    dim3 block = site_block(Ls, vectors);
    return dim3((sites + block.x - 1) / block.x, (Ls * vectors + block.y - 1) / block.y);
}

// The coordinates of site `site` of the field of parity `parity`.
__host__ __device__ __forceinline__ void half_site_to_coords(
    int site, int parity, int xc[4], int nx, int ny, int nz)
{
    int rem = site / (nx / 2);
    xc[1] = rem % ny; rem /= ny;
    xc[2] = rem % nz;
    xc[3] = rem / nz;
    xc[0] = 2 * (site % (nx / 2)) + ((parity + xc[1] + xc[2] + xc[3]) & 1);
}

// w = a psi_s + y (Y psi)_s, or with Y^dag, at fifth coordinate `is` of site `site`.
template <typename Float>
__device__ __forceinline__ void fifth_term(
    Float w[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    int is, int Ls, int site,
    Float mass, Float a, Float y, bool dagger)
{
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            w[s][c][0] = a * d_in[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site)];
            w[s][c][1] = a * d_in[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site)];
        }
    }
    if (y == static_cast<Float>(0.0)) return;
    // The coefficient of the projected neighbour in Y: -1, or +m across the walls.
    const int is_p = (is + 1 < Ls) ? (is + 1) : 0;
    const int is_m = (is >= 1) ? (is - 1) : (Ls - 1);
    const Float factor_p = y * ((is + 1 < Ls) ? static_cast<Float>(-1.0) : mass);
    const Float factor_m = y * ((is >= 1) ? static_cast<Float>(-1.0) : mass);
    Float above[Nspin][Nc][Ncomplex], below[Nspin][Nc][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            above[s][c][0] = factor_p * d_in[aosoa_5d_spinor_idx(c, s, 0, is_p, Ls, site)];
            above[s][c][1] = factor_p * d_in[aosoa_5d_spinor_idx(c, s, 1, is_p, Ls, site)];
            below[s][c][0] = factor_m * d_in[aosoa_5d_spinor_idx(c, s, 0, is_m, Ls, site)];
            below[s][c][1] = factor_m * d_in[aosoa_5d_spinor_idx(c, s, 1, is_m, Ls, site)];
        }
    }
    // Y takes P- from above and P+ from below; Y^dag the other way round.
    Float from_above[Nspin][Nc][Ncomplex], from_below[Nspin][Nc][Ncomplex];
    if (dagger) {
        proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_above, above);
        proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_below, below);
    } else {
        proj_pl<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_above, above);
        proj_pr<Float, BRIDGE_CUDA_GAMMA_DIRAC>(from_below, below);
    }
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            w[s][c][0] += from_above[s][c][0] + from_below[s][c][0];
            w[s][c][1] += from_above[s][c][1] + from_below[s][c][1];
        }
    }
}

// The link in direction mu at `site`. The operators keep `rows` rows of every link: all three
// (18 reals, the links of base/aosoa.cuh), or the first two (12 reals), for links in SU(3),
// whose third row is the conjugate of the cross product of the first two. A third less to read
// for a link, which is most of what a four-dimensional operator reads.
template <typename Float>
__device__ __forceinline__ void load_link(
    Float u_r[Nc][Nc], Float u_i[Nc][Nc],
    const Float* __restrict__ d_gauge,
    int mu, int site, int rows)
{
    const int reals = 4 * 2 * Nc * rows;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            const int comp = 2 * (c + Nc * (r + rows * mu));
            u_r[r][c] = d_gauge[aosoa_idx(comp, site, reals)];
            u_i[r][c] = d_gauge[aosoa_idx(comp + 1, site, reals)];
        }
    }
    if (rows == 2) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            const int a = (c + 1) % Nc, b = (c + 2) % Nc;
            // conj(u0[a] u1[b] - u0[b] u1[a])
            u_r[2][c] = (u_r[0][a] * u_r[1][b] - u_i[0][a] * u_i[1][b])
                      - (u_r[0][b] * u_r[1][a] - u_i[0][b] * u_i[1][a]);
            u_i[2][c] = (u_r[0][b] * u_i[1][a] + u_i[0][b] * u_r[1][a])
                      - (u_r[0][a] * u_i[1][b] + u_i[0][a] * u_r[1][b]);
        }
    } else {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            const int comp = 2 * (c + Nc * (2 + rows * mu));
            u_r[2][c] = d_gauge[aosoa_idx(comp, site, reals)];
            u_i[2][c] = d_gauge[aosoa_idx(comp + 1, site, reals)];
        }
    }
}

// acc += coef U (1 - s gamma_mu) w, or with U^dag (`adjoint`), s = +-1.
//
// (1 - s gamma_mu) w has two independent spin components, so the link multiplies two colour
// vectors, not four. With (gamma_mu w)[row] = val[row] w[col[row]]: for mu < 3, col takes the
// rows 0, 1 to the rows 2, 3, the rows a = 0, 1 of (1 - s gamma_mu) w are
// h_a = w_a - s val[a] w_col[a], and its row col[a] is - s val[col[a]] h_a. gamma_4 is diagonal,
// (1, 1, -1, -1): (1 - s gamma_4) w is 2 w on the rows 2, 3 for s = +1, on the rows 0, 1 for
// s = -1, and zero on the others.
template <typename Float>
__device__ __forceinline__ void hop_leg(
    Float acc[Nspin][Nc][Ncomplex],
    const Float w[Nspin][Nc][Ncomplex],
    const Float u_r[Nc][Nc], const Float u_i[Nc][Nc],
    bool adjoint, int mu, Float s, Float coef)
{
    // col[mu][row]; val as (re, im)
    const int col[3][4] = {{3, 2, 1, 0}, {3, 2, 1, 0}, {2, 3, 0, 1}};
    const int re[3][4]  = {{0, 0, 0, 0}, {-1, 1, 1, -1}, {0, 0, 0, 0}};
    const int im[3][4]  = {{-1, -1, 1, 1}, {0, 0, 0, 0}, {-1, 1, 1, -1}};
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
        // The row of the half spinor, and the other row it gives with the factor (fr, fi).
        int row, other = -1;
        Float h_r[Nc], h_i[Nc], fr = 0, fi = 0;
        if (mu < 3) {
            row = a;
            other = col[mu][a];
            const Float vr = s * static_cast<Float>(re[mu][a]);
            const Float vi = s * static_cast<Float>(im[mu][a]);
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                h_r[c] = w[a][c][0] - (vr * w[other][c][0] - vi * w[other][c][1]);
                h_i[c] = w[a][c][1] - (vr * w[other][c][1] + vi * w[other][c][0]);
            }
            fr = -s * static_cast<Float>(re[mu][other]);
            fi = -s * static_cast<Float>(im[mu][other]);
        } else {
            row = (s > static_cast<Float>(0.0)) ? a + 2 : a;
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                h_r[c] = static_cast<Float>(2.0) * w[row][c][0];
                h_i[c] = static_cast<Float>(2.0) * w[row][c][1];
            }
        }
        Float t_r[Nc], t_i[Nc];
        if (adjoint) {
            su3_dagger_mult_vec(t_r, t_i, u_r, u_i, h_r, h_i);
        } else {
            su3_mult_vec(t_r, t_i, u_r, u_i, h_r, h_i);
        }
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            acc[row][c][0] += coef * t_r[c];
            acc[row][c][1] += coef * t_i[c];
            if (mu < 3) {
                acc[other][c][0] += coef * (fr * t_r[c] - fi * t_i[c]);
                acc[other][c][1] += coef * (fr * t_i[c] + fi * t_r[c]);
            }
        }
    }
}

// acc += H w at the site of coordinates xc and fifth coordinate `is`, with
// w = (b - c Y) psi of d_in; `sign` -1 gives H^dag. With `half`, d_in holds the sites of the
// parity opposite to that of xc. For a four-dimensional field: Ls = 1, b = 1, c = 0. The links
// are kept as load_link says.
template <typename Float>
__device__ __forceinline__ void hops(
    Float acc[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    const int xc[4], int is, int Ls,
    const int dims[4], const Float bc[4],
    bool half, Float sign, Float kappa,
    Float mass, Float b5, Float c5,
    int legs = 0xFF)
{
    const int nx = dims[0], ny = dims[1], nz = dims[2], nt = dims[3];
    const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
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
        // Where the neighbours are in d_in.
        const int in_f = half ? f[0] / 2 + (nx / 2) * (f[1] + ny * (f[2] + nz * f[3])) : site_f;
        const int in_b = half ? b[0] / 2 + (nx / 2) * (b[1] + ny * (b[2] + nz * b[3])) : site_b;
        const Float sign_f = wrap_f ? bc[mu] : static_cast<Float>(1.0);
        const Float sign_b = wrap_b ? bc[mu] : static_cast<Float>(1.0);

        Float w_f[Nspin][Nc][Ncomplex], w_b[Nspin][Nc][Ncomplex];
        fifth_term<Float>(w_f, d_in, is, Ls, in_f, mass, b5, -c5, false);
        fifth_term<Float>(w_b, d_in, is, Ls, in_b, mass, b5, -c5, false);
        Float u_f_r[Nc][Nc], u_f_i[Nc][Nc], u_b_r[Nc][Nc], u_b_i[Nc][Nc];
        load_link<Float>(u_f_r, u_f_i, d_gauge, mu, site, link_rows);
        load_link<Float>(u_b_r, u_b_i, d_gauge, mu, site_b, link_rows);
        // `legs`: bit mu for the forward hop in direction mu, bit 4 + mu for the backward one.
        if (legs & (1 << mu)) hop_leg<Float>(acc, w_f, u_f_r, u_f_i, false, mu, sign, -kappa * sign_f);
        if (legs & (16 << mu)) hop_leg<Float>(acc, w_b, u_b_r, u_b_i, true, mu, -sign, -kappa * sign_b);
    }
}

// -------------------------------------------------------------------------
// out = H (b - c Y) in, or with H^dag (`sign` -1): on the whole lattice (target_parity < 0),
// or out on the sites of parity target_parity and in on the others.
// With d_from not null, out = d_from - that (a field on the sites of out; it may be out).
// The fibres of a site: its spinors at every s of every vector.
// `Many`: whether the fields may hold more than one vector. These kernels have no register to
// spare, and where the vector of a thread begins costs a launch with one vector a sixth of
// its time: it is launched without.
// -------------------------------------------------------------------------
template <typename Float, bool Many>
__global__ void hop_kernel(
    Float* d_out,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    const Float* d_from,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls, int vectors,
    int target_parity,
    Float sign, Float kappa,
    Float mass, Float b5, Float c5)
{
    const bool half = (target_parity >= 0);
    const int sites = (half ? nx / 2 : nx) * ny * nz * nt;
    const int site  = blockIdx.x * blockDim.x + threadIdx.x;
    const int fibre = blockIdx.y * blockDim.y + threadIdx.y;
    if (site >= sites || fibre >= Ls * vectors) return;
    const int is = Many ? fibre % Ls : fibre;
    const size_t first = Many ? (fibre / Ls) * vector_reals(sites, Ls) : 0;
    d_out += first;
    d_in += first;
    if (d_from != nullptr) d_from += first;

    const int dims[4] = {nx, ny, nz, nt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    if (half) {
        half_site_to_coords(site, target_parity, xc, nx, ny, nz);
    } else {
        site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    }

    Float acc[Nspin][Nc][Ncomplex] = {0};
    hops<Float>(acc, d_in, d_gauge, link_rows, xc, is, Ls, dims, bc, half, sign, kappa, mass, b5, c5);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                const int k = aosoa_5d_spinor_idx(c, s, ri, is, Ls, site);
                d_out[k] = (d_from != nullptr) ? d_from[k] - acc[s][c][ri] : acc[s][c][ri];
            }
        }
    }
}

// -------------------------------------------------------------------------
// One leg of H (b - c Y) in on the whole lattice (leg mu: the forward hop in direction mu,
// 4 + mu: the backward one), sorted by whether it comes from another block of the lattice
// (blocks of bx by bz bt sites: the aggregates of a multigrid): cross = the leg where it
// does and zero elsewhere, and inside += the leg elsewhere. For the coarse operator.
// The fibres of a site as in hop_kernel.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void hop_split_kernel(
    Float* __restrict__ d_cross,
    Float* __restrict__ d_inside,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    int nx, int ny, int nz, int nt,
    Float bcx, Float bcy, Float bcz, Float bct,
    int Ls, int vectors,
    int leg,
    int bx, int by, int bz, int bt,
    Float kappa,
    Float mass, Float b5, Float c5)
{
    const int sites = nx * ny * nz * nt;
    const int site  = blockIdx.x * blockDim.x + threadIdx.x;
    const int fibre = blockIdx.y * blockDim.y + threadIdx.y;
    if (site >= sites || fibre >= Ls * vectors) return;
    const int is = fibre % Ls;
    const size_t first = (fibre / Ls) * vector_reals(sites, Ls);

    const int dims[4] = {nx, ny, nz, nt};
    const int block[4] = {bx, by, bz, bt};
    const Float bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    Float acc[Nspin][Nc][Ncomplex] = {0};
    hops<Float>(acc, d_in + first, d_gauge, link_rows, xc, is, Ls, dims, bc, false,
                static_cast<Float>(1.0), kappa, mass, b5, c5, 1 << leg);
    const int mu = leg % 4, at = xc[mu] % block[mu];
    const bool crosses = (leg < 4) ? (at == block[mu] - 1) : (at == 0);
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                const size_t k = first + aosoa_5d_spinor_idx(c, s, ri, is, Ls, site);
                d_cross[k] = crosses ? acc[s][c][ri] : static_cast<Float>(0.0);
                if (!crosses) d_inside[k] += acc[s][c][ri];
            }
        }
    }
}

// -------------------------------------------------------------------------
// A field on one parity out of the field on the whole lattice (to_half), or into it; the
// sites of the other parity are left as they are. The fibres of a site: its vectors.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void parity_copy_kernel(
    Float* __restrict__ d_half,
    Float* __restrict__ d_full,
    int nx, int ny, int nz, int nt,
    int Ls,
    int parity,
    int to_half,
    int vectors)
{
    const int sites = (nx / 2) * ny * nz * nt;
    const int half = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (half >= sites || v >= vectors) return;
    Float* __restrict__ h = d_half + v * vector_reals(sites, Ls);
    Float* __restrict__ f = d_full + v * vector_reals(2 * sites, Ls);
    int xc[4];
    half_site_to_coords(half, parity, xc, nx, ny, nz);
    const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    for (int is = 0; is < Ls; ++is) {
        #pragma unroll
        for (int j = 0; j < Nspin * Nc * Ncomplex; ++j) {
            const int sp = j / (Nc * Ncomplex), c = (j / Ncomplex) % Nc, ri = j % Ncomplex;
            const int at_half = aosoa_5d_spinor_idx(c, sp, ri, is, Ls, half);
            const int at_full = aosoa_5d_spinor_idx(c, sp, ri, is, Ls, site);
            if (to_half) {
                h[at_half] = f[at_full];
            } else {
                f[at_full] = h[at_half];
            }
        }
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_FERMION_HOPS_CUH
