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
            // conj(u0[a] u1[b] - u0[b] u1[a]), the fused products fixed (su3.cuh, mul_rn).
            u_r[2][c] = fma(u_r[0][a], u_r[1][b], -mul_rn(u_i[0][a], u_i[1][b]))
                      - fma(u_r[0][b], u_r[1][a], -mul_rn(u_i[0][b], u_i[1][a]));
            u_i[2][c] = fma(u_r[0][b], u_i[1][a], mul_rn(u_i[0][b], u_r[1][a]))
                      - fma(u_r[0][a], u_i[1][b], mul_rn(u_i[0][a], u_r[1][b]));
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
//
// In three steps, which the legs from other processes take apart (leg_half on the process that
// sends, add_leg on the one that receives): the two colour vectors h_a (leg_half), the link on
// each, and the four rows rebuilt from them (add_leg).

// The two colour vectors h_0, h_1 of (1 - s gamma_mu) w.
template <typename Float>
__device__ __forceinline__ void leg_half(
    Float h_r[2][Nc], Float h_i[2][Nc],
    const Float w[Nspin][Nc][Ncomplex],
    int mu, Float s)
{
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
        if (mu < 3) {
            const int other = dirac_col(mu, a);
            const Float vr = s * static_cast<Float>(dirac_re(mu, a));
            const Float vi = s * static_cast<Float>(dirac_im(mu, a));
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                h_r[a][c] = w[a][c][0] - (vr * w[other][c][0] - vi * w[other][c][1]);
                h_i[a][c] = w[a][c][1] - (vr * w[other][c][1] + vi * w[other][c][0]);
            }
        } else {
            const int row = (s > static_cast<Float>(0.0)) ? a + 2 : a;
            #pragma unroll
            for (int c = 0; c < Nc; ++c) {
                h_r[a][c] = static_cast<Float>(2.0) * w[row][c][0];
                h_i[a][c] = static_cast<Float>(2.0) * w[row][c][1];
            }
        }
    }
}

// acc += coef times the rows of (1 - s gamma_mu) w that t = U h_a (or U^dag h_a) gives.
template <typename Float>
__device__ __forceinline__ void add_leg(
    Float acc[Nspin][Nc][Ncomplex],
    const Float t_r[Nc], const Float t_i[Nc],
    int mu, int a, Float s, Float coef)
{
    int row, other = -1;
    Float fr = 0, fi = 0;
    if (mu < 3) {
        row = a;
        other = dirac_col(mu, a);
        fr = -s * static_cast<Float>(dirac_re(mu, other));
        fi = -s * static_cast<Float>(dirac_im(mu, other));
    } else {
        row = (s > static_cast<Float>(0.0)) ? a + 2 : a;
    }
    // Fused in a fixed order (su3.cuh, mul_rn); fr and fi are 0 or +-1, so their products are
    // exact.
    #pragma unroll
    for (int c = 0; c < Nc; ++c) {
        acc[row][c][0] = fma(coef, t_r[c], acc[row][c][0]);
        acc[row][c][1] = fma(coef, t_i[c], acc[row][c][1]);
        if (mu < 3) {
            acc[other][c][0] = fma(coef, fr * t_r[c] - fi * t_i[c], acc[other][c][0]);
            acc[other][c][1] = fma(coef, fr * t_i[c] + fi * t_r[c], acc[other][c][1]);
        }
    }
}

template <typename Float>
__device__ __forceinline__ void hop_leg(
    Float acc[Nspin][Nc][Ncomplex],
    const Float w[Nspin][Nc][Ncomplex],
    const Float u_r[Nc][Nc], const Float u_i[Nc][Nc],
    bool adjoint, int mu, Float s, Float coef)
{
    Float h_r[2][Nc], h_i[2][Nc];
    leg_half<Float>(h_r, h_i, w, mu, s);
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
        Float t_r[Nc], t_i[Nc];
        if (adjoint) {
            su3_dagger_mult_vec(t_r, t_i, u_r, u_i, h_r[a], h_i[a]);
        } else {
            su3_mult_vec(t_r, t_i, u_r, u_i, h_r[a], h_i[a]);
        }
        add_leg<Float>(acc, t_r, t_i, mu, a, s, coef);
    }
}

// -------------------------------------------------------------------------
// Several processes (the lattice cut into local lattices along the directions `cut`): the
// legs of a hop that come from another process arrive in the buffer of their face, one for
// each face (forward in direction mu: face mu, backward: face 4 + mu, as comm::FACES). What
// arrives is the half spinor of the leg, 2 colour vectors (12 reals) for every site of the face
// and every fibre (s and vector): h = (1 - s gamma_mu) w of the forward leg, to which the
// process that receives applies its link, and U^dag h of the backward leg, the link being the
// one of the sending site. The receiver multiplies by bc[face], the boundary of the whole
// lattice where the face crosses it and 1 elsewhere.
//
// A site of the face is numbered by its three other coordinates, lexically (d0 fastest); on
// the sites of one parity d0/2 (the local extents are even, so a line in d0 holds one site of
// each parity in every pair). Within a buffer: ((vector Ls + s) 12 + real) sites + site.
//
// The output of an operator splits into its interior, the sites none of whose legs comes from
// another process, and its surface. The interior is computed while the faces travel; the
// surface after, every leg of a site in the order of one process, so the result is that of one
// process bit for bit.
// -------------------------------------------------------------------------
template <typename Float>
struct Halo {
    const Float* recv[8];
    Float bc[8];
    int cut[4];
    // The sites of the surface, in the numbering of the output; their number.
    const int* surface;
    int surface_sites;
};

// Which part of the output a kernel computes.
enum Part { Whole = 0, Interior = 1, Surface = 2 };

// The number of site xc on the face of direction mu.
__host__ __device__ __forceinline__ int face_site(const int xc[4], int mu, const int dims[4], bool half)
{
    int d[3], k = 0;
    #pragma unroll
    for (int n = 0; n < 4; ++n) if (n != mu) d[k++] = n;
    const int rest = xc[d[1]] + dims[d[1]] * xc[d[2]];
    return half ? xc[d[0]] / 2 + (dims[d[0]] / 2) * rest : xc[d[0]] + dims[d[0]] * rest;
}

// The sites of a face.
__host__ __device__ __forceinline__ int face_sites(int mu, const int dims[4], bool half)
{
    const int all = dims[0] * dims[1] * dims[2] * dims[3] / dims[mu];
    return half ? all / 2 : all;
}

// Where real k of colour vector a of fibre (vec, is) of face site f is in a buffer.
__host__ __device__ __forceinline__ size_t in_face(int vec, int is, int Ls, int a, int k, int f, int sites)
{
    return (static_cast<size_t>(vec * Ls + is) * Nhalf + a * Nvc + k) * sites + f;
}

// Whether a leg of the site xc comes from another process.
template <typename Float>
__device__ __forceinline__ bool on_surface(const Halo<Float>& halo, const int xc[4], const int dims[4])
{
    bool on = false;
    #pragma unroll
    for (int mu = 0; mu < 4; ++mu) {
        on |= halo.cut[mu] && (xc[mu] == 0 || xc[mu] + 1 == dims[mu]);
    }
    return on;
}

// acc += H w at the site of coordinates xc and fifth coordinate `is`, with
// w = (b - c Y) psi of d_in; `sign` -1 gives H^dag. With `half`, d_in holds the sites of the
// parity opposite to that of xc. For a four-dimensional field: Ls = 1, b = 1, c = 0. The links
// are kept as load_link says.
//
// With `Split` (several processes), the legs across a cut direction come from the buffers of
// `halo` (fibre `vec`, `is`), and every leg is taken one after the other, in the order of one
// process: forward then backward, mu = 0 to 3, the same sums into acc.
template <typename Float, bool Split = false>
__device__ __forceinline__ void hops(
    Float acc[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    const int xc[4], int is, int Ls,
    const int dims[4], const Float bc[4],
    bool half, Float sign, Float kappa,
    Float mass, Float b5, Float c5,
    int legs = 0xFF,
    const Halo<Float>* halo = nullptr, int vec = 0)
{
    const int nx = dims[0], ny = dims[1], nz = dims[2], nt = dims[3];
    const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    if constexpr (Split) {
        #pragma unroll
        for (int mu = 0; mu < Ndim; ++mu) {
            int f[4] = {xc[0], xc[1], xc[2], xc[3]};
            int b[4] = {xc[0], xc[1], xc[2], xc[3]};
            const bool wrap_f = (xc[mu] + 1 == dims[mu]);
            const bool wrap_b = (xc[mu] == 0);
            const bool remote_f = halo->cut[mu] && wrap_f;
            const bool remote_b = halo->cut[mu] && wrap_b;
            f[mu] = wrap_f ? 0 : xc[mu] + 1;
            b[mu] = wrap_b ? dims[mu] - 1 : xc[mu] - 1;
            const int site_b = coords_to_site(b[0], b[1], b[2], b[3], nx, ny, nz, nt);
            const int in_f = half ? f[0] / 2 + (nx / 2) * (f[1] + ny * (f[2] + nz * f[3]))
                                  : coords_to_site(f[0], f[1], f[2], f[3], nx, ny, nz, nt);
            const int in_b = half ? b[0] / 2 + (nx / 2) * (b[1] + ny * (b[2] + nz * b[3])) : site_b;
            const int face = face_site(xc, mu, dims, half), sites = face_sites(mu, dims, half);
            if (legs & (1 << mu)) {
                Float u_r[Nc][Nc], u_i[Nc][Nc];
                load_link<Float>(u_r, u_i, d_gauge, mu, site, link_rows);
                if (remote_f) {
                    const Float* buf = halo->recv[mu];
                    const Float coef = -kappa * halo->bc[mu];
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        Float h_r[Nc], h_i[Nc], t_r[Nc], t_i[Nc];
                        #pragma unroll
                        for (int c = 0; c < Nc; ++c) {
                            h_r[c] = buf[in_face(vec, is, Ls, a, 2 * c, face, sites)];
                            h_i[c] = buf[in_face(vec, is, Ls, a, 2 * c + 1, face, sites)];
                        }
                        su3_mult_vec(t_r, t_i, u_r, u_i, h_r, h_i);
                        add_leg<Float>(acc, t_r, t_i, mu, a, sign, coef);
                    }
                } else {
                    Float w[Nspin][Nc][Ncomplex];
                    fifth_term<Float>(w, d_in, is, Ls, in_f, mass, b5, -c5, false);
                    const Float sign_f = wrap_f ? bc[mu] : static_cast<Float>(1.0);
                    hop_leg<Float>(acc, w, u_r, u_i, false, mu, sign, -kappa * sign_f);
                }
            }
            if (legs & (16 << mu)) {
                if (remote_b) {
                    const Float* buf = halo->recv[4 + mu];
                    const Float coef = -kappa * halo->bc[4 + mu];
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        Float t_r[Nc], t_i[Nc];
                        #pragma unroll
                        for (int c = 0; c < Nc; ++c) {
                            t_r[c] = buf[in_face(vec, is, Ls, a, 2 * c, face, sites)];
                            t_i[c] = buf[in_face(vec, is, Ls, a, 2 * c + 1, face, sites)];
                        }
                        add_leg<Float>(acc, t_r, t_i, mu, a, -sign, coef);
                    }
                } else {
                    Float w[Nspin][Nc][Ncomplex];
                    fifth_term<Float>(w, d_in, is, Ls, in_b, mass, b5, -c5, false);
                    Float u_r[Nc][Nc], u_i[Nc][Nc];
                    load_link<Float>(u_r, u_i, d_gauge, mu, site_b, link_rows);
                    const Float sign_b = wrap_b ? bc[mu] : static_cast<Float>(1.0);
                    hop_leg<Float>(acc, w, u_r, u_i, true, mu, -sign, -kappa * sign_b);
                }
            }
        }
        return;
    }
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

        if constexpr (sizeof(Float) == 8) {
            // In double precision the two legs one after the other: the spinor and the link of
            // one leg live at a time, not those of both (what lets hop_gemm_kernel<double> keep
            // to 168 registers, three blocks an SM). The sums into acc are in the same order.
            if (legs & (1 << mu)) {
                Float w[Nspin][Nc][Ncomplex];
                fifth_term<Float>(w, d_in, is, Ls, in_f, mass, b5, -c5, false);
                Float u_r[Nc][Nc], u_i[Nc][Nc];
                load_link<Float>(u_r, u_i, d_gauge, mu, site, link_rows);
                hop_leg<Float>(acc, w, u_r, u_i, false, mu, sign, -kappa * sign_f);
            }
            if (legs & (16 << mu)) {
                Float w[Nspin][Nc][Ncomplex];
                fifth_term<Float>(w, d_in, is, Ls, in_b, mass, b5, -c5, false);
                Float u_r[Nc][Nc], u_i[Nc][Nc];
                load_link<Float>(u_r, u_i, d_gauge, mu, site_b, link_rows);
                hop_leg<Float>(acc, w, u_r, u_i, true, mu, -sign, -kappa * sign_b);
            }
            continue;
        }
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
// The clover term T = 1 - kappa c_SW sigma F (clover.cuh): it commutes with gamma_5 and is kept
// as RutticeQCD keeps it, 72 reals per site, the block on gamma_5 = +1 and then the one on
// gamma_5 = -1, each a packed Hermitian 6 x 6 matrix (index colour + 3 spin): the six real
// diagonal entries, then the upper triangle row by row as (re, im). In the Dirac representation
// gamma_5 = [[0, 1], [1, 0]], with eigenvectors (e_a +- e_(a+2)) / sqrt(2). Here because the
// hops of the clover operator start from it (hop_kernel with Clover).
// -------------------------------------------------------------------------
// y = M x for the packed Hermitian 6 x 6 matrix `which` (0 or 1) of `site`.
template <typename Float>
__device__ __forceinline__ void hermitian6_mul(
    Float y[Nblock][Ncomplex],
    const Float* __restrict__ d_clover,
    int which, int site,
    const Float x[Nblock][Ncomplex])
{
    const int base = (Nclover / 2) * which;
    #pragma unroll
    for (int r = 0; r < Nblock; ++r) {
        const Float d = d_clover[aosoa_idx(base + r, site, Nclover)];
        y[r][0] = mul_rn(d, x[r][0]);
        y[r][1] = mul_rn(d, x[r][1]);
    }
    int k = Nblock;
    #pragma unroll
    for (int r = 0; r < Nblock; ++r) {
        #pragma unroll
        for (int c = r + 1; c < Nblock; ++c) {
            const Float zr = d_clover[aosoa_idx(base + k, site, Nclover)];
            const Float zi = d_clover[aosoa_idx(base + k + 1, site, Nclover)];
            // y_r += z x_c,  y_c += conj(z) x_r, fused in a fixed order (su3.cuh, mul_rn)
            c_macc(y[r][0], y[r][1], zr, zi, x[c][0], x[c][1]);
            c_conj_macc(y[c][0], y[c][1], zr, zi, x[r][0], x[r][1]);
            k += Ncomplex;
        }
    }
}

// out = T psi: psi at `at` of d_in, T that of the site `site` of the lattice.
template <typename Float>
__device__ __forceinline__ void clover_site(
    Float out[Nspin][Nc][Ncomplex],
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_clover,
    int at, int site)
{
    // The components along e_a + e_(a+2) and e_a - e_(a+2), without the 1/sqrt(2).
    Float plus[Nblock][Ncomplex], minus[Nblock][Ncomplex];
    #pragma unroll
    for (int s = 0; s < Nspin2; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                const Float upper = d_in[aosoa_spinor_idx(c, s, ri, at)];
                const Float lower = d_in[aosoa_spinor_idx(c, s + Nspin2, ri, at)];
                plus[c + Nc * s][ri]  = upper + lower;
                minus[c + Nc * s][ri] = upper - lower;
            }
        }
    }
    Float p[Nblock][Ncomplex], m[Nblock][Ncomplex];
    hermitian6_mul<Float>(p, d_clover, 0, site, plus);
    hermitian6_mul<Float>(m, d_clover, 1, site, minus);
    // Back to spin components; the two 1/sqrt(2) make 1/2.
    #pragma unroll
    for (int s = 0; s < Nspin2; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                const int k = c + Nc * s;
                out[s][c][ri]     = static_cast<Float>(0.5) * (p[k][ri] + m[k][ri]);
                out[s + Nspin2][c][ri] = static_cast<Float>(0.5) * (p[k][ri] - m[k][ri]);
            }
        }
    }
}

// The blocks of the Schwarz smoother (multigrid/sap.cuh): with extent[0] > 0 the hops keep only
// the legs inside a block of extent[mu] sites in direction mu, and the output is zero outside the
// blocks of colour `colour`. The colour of a block is the parity of its block coordinates on the
// whole lattice; `shift` that of the block coordinates of the process's origin.
struct BlockCut {
    int extent[Ndim];
    int colour;
    int shift;
};

// -------------------------------------------------------------------------
// out = H (b - c Y) in, or with H^dag (`sign` -1): on the whole lattice (target_parity < 0),
// or out on the sites of parity target_parity and in on the others.
// With d_from not null, out = d_from - that (a field on the sites of out; it may be out).
// The fibres of a site: its spinors at every s of every vector.
// `Many`: whether the fields may hold more than one vector. These kernels have no register to
// spare, and where the vector of a thread begins costs a launch with one vector a sixth of
// its time: it is launched without.
// -------------------------------------------------------------------------
// With several processes (`P`, see Halo): the interior, the sites of the output none of whose
// legs comes from another process (the others are left as they are), or the surface, the
// sites halo.surface lists.
// `Clover` (Ls = 1): out = T d_from + H in, T the blocks d_clover of the site: the sum starts
// from T d_from and the legs are added to it, in every part the same order, so the clover
// operator D = T + H is one pass on any number of processes.
// `FourD`: a four-dimensional operator, Ls = 1, b = 1, c = 0 known when compiling, so the
// fifth dimension leaves no trace in the code (with them only at run time the Wilson operator
// took half as long again).
// `Cut` (with `cut`, BlockCut): the operator restricted to the blocks of one colour, the block
// operator of the Schwarz smoother. A template parameter, so that the other instances keep all
// eight legs as a constant: with the legs known only at run time the Wilson operator in single
// precision took 40% longer.
template <typename Float, bool Many, int P = Whole, bool Clover = false, bool FourD = false,
          bool Cut = false>
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
    Float mass, Float b5, Float c5,
    Halo<Float> halo,
    const Float* __restrict__ d_clover,
    BlockCut cut)
{
    if constexpr (FourD) {
        Ls = 1;
        mass = 0;
        b5 = 1;
        c5 = 0;
    }
    const bool half = (target_parity >= 0);
    const int sites = (half ? nx / 2 : nx) * ny * nz * nt;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int fibre = blockIdx.y * blockDim.y + threadIdx.y;
    if (thread >= (P == Surface ? halo.surface_sites : sites) || fibre >= Ls * vectors) return;
    const int site = (P == Surface) ? halo.surface[thread] : thread;
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
    if (P == Interior && on_surface(halo, xc, dims)) return;

    int legs = 0xFF;
    if constexpr (Cut) {
        int colour = cut.shift;
        #pragma unroll
        for (int mu = 0; mu < Ndim; ++mu) {
            const int e = cut.extent[mu], within = xc[mu] % e;
            colour += xc[mu] / e;
            if (within == e - 1) legs &= ~(1 << mu);
            if (within == 0) legs &= ~(16 << mu);
        }
        if ((colour & 1) != cut.colour) {
            #pragma unroll
            for (int s = 0; s < Nspin; ++s)
                #pragma unroll
                for (int c = 0; c < Nc; ++c)
                    #pragma unroll
                    for (int ri = 0; ri < Ncomplex; ++ri)
                        d_out[aosoa_5d_spinor_idx(c, s, ri, is, Ls, site)] = 0;
            return;
        }
    }

    Float acc[Nspin][Nc][Ncomplex] = {0};
    if constexpr (Clover) {
        const int full = half ? coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt) : site;
        clover_site<Float>(acc, d_from, d_clover, site, full);
    }
    if constexpr (P == Surface) {
        hops<Float, true>(acc, d_in, d_gauge, link_rows, xc, is, Ls, dims, bc, half, sign, kappa,
                          mass, b5, c5, legs, &halo, Many ? fibre / Ls : 0);
    } else {
        hops<Float>(acc, d_in, d_gauge, link_rows, xc, is, Ls, dims, bc, half, sign, kappa, mass, b5, c5, legs);
    }
    // d_from may be d_out: all of it is read before anything is written, so the loads need not
    // wait for the stores one by one.
    if constexpr (!Clover) {
        if (d_from != nullptr) {
            #pragma unroll
            for (int s = 0; s < Nspin; ++s)
                #pragma unroll
                for (int c = 0; c < Nc; ++c)
                    #pragma unroll
                    for (int ri = 0; ri < Ncomplex; ++ri)
                        acc[s][c][ri] = d_from[aosoa_5d_spinor_idx(c, s, ri, is, Ls, site)] - acc[s][c][ri];
        }
    }
    #pragma unroll
    for (int s = 0; s < Nspin; ++s) {
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            #pragma unroll
            for (int ri = 0; ri < Ncomplex; ++ri) {
                d_out[aosoa_5d_spinor_idx(c, s, ri, is, Ls, site)] = acc[s][c][ri];
            }
        }
    }
}

// -------------------------------------------------------------------------
// The buffer of face `face` (see Halo) out of the field d_in on the whole lattice (in_parity
// < 0) or on the sites of parity in_parity: for every site of the face and every fibre, the
// half spinor the neighbour on the other side takes for its leg, w = (b - c Y) psi of d_in at
// the site: (1 - sign gamma_mu) w for a face backward (the neighbour's forward leg), and
// U_mu^dag (1 + sign gamma_mu) w for a face forward (its backward leg, on the link of this
// site). The same arithmetic as hops, so the neighbour sums what one process would.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void hop_pack_kernel(
    Float* __restrict__ d_send,
    const Float* __restrict__ d_in,
    const Float* __restrict__ d_gauge, int link_rows,
    int nx, int ny, int nz, int nt,
    int Ls, int vectors,
    int in_parity, int face,
    Float sign,
    Float mass, Float b5, Float c5)
{
    const bool half = (in_parity >= 0);
    const int dims[4] = {nx, ny, nz, nt};
    const int mu = face % 4;
    const bool forward = (face < 4);
    const int sites = face_sites(mu, dims, half);
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    const int fibre = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= sites || fibre >= Ls * vectors) return;
    const int is = fibre % Ls, vec = fibre / Ls;
    const int field_sites = (half ? nx / 2 : nx) * ny * nz * nt;
    d_in += vec * vector_reals(field_sites, Ls);

    // The coordinates of face site f: the inverse of face_site, x_mu on the face.
    int d[3], k = 0;
    #pragma unroll
    for (int n = 0; n < 4; ++n) if (n != mu) d[k++] = n;
    int xc[4];
    xc[mu] = forward ? dims[mu] - 1 : 0;
    const int line = half ? dims[d[0]] / 2 : dims[d[0]];
    int rest = f / line;
    xc[d[1]] = rest % dims[d[1]];
    xc[d[2]] = rest / dims[d[1]];
    xc[d[0]] = f % line;
    if (half) {
        const int others = xc[mu] + xc[d[1]] + xc[d[2]];
        xc[d[0]] = 2 * xc[d[0]] + ((in_parity + others) & 1);
    }
    const int site = coords_to_site(xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    const int in = half ? xc[0] / 2 + (nx / 2) * (xc[1] + ny * (xc[2] + nz * xc[3])) : site;

    Float w[Nspin][Nc][Ncomplex];
    fifth_term<Float>(w, d_in, is, Ls, in, mass, b5, -c5, false);
    Float h_r[2][Nc], h_i[2][Nc];
    leg_half<Float>(h_r, h_i, w, mu, forward ? -sign : sign);
    Float u_r[Nc][Nc], u_i[Nc][Nc];
    if (forward) load_link<Float>(u_r, u_i, d_gauge, mu, site, link_rows);
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
        Float t_r[Nc], t_i[Nc];
        if (forward) {
            su3_dagger_mult_vec(t_r, t_i, u_r, u_i, h_r[a], h_i[a]);
        } else {
            #pragma unroll
            for (int c = 0; c < Nc; ++c) { t_r[c] = h_r[a][c]; t_i[c] = h_i[a][c]; }
        }
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            d_send[in_face(vec, is, Ls, a, 2 * c, f, sites)] = t_r[c];
            d_send[in_face(vec, is, Ls, a, 2 * c + 1, f, sites)] = t_i[c];
        }
    }
}

// -------------------------------------------------------------------------
// One leg of H (b - c Y) in on the whole lattice (leg mu: the forward hop in direction mu,
// 4 + mu: the backward one), sorted by whether it comes from another block of the lattice
// (blocks of bx by bz bt sites: the aggregates of a multigrid): cross = the leg where it
// does and zero elsewhere, and inside += the leg elsewhere. For the coarse operator.
// The fibres of a site as in hop_kernel. In single precision two blocks of 256 an SM (128
// registers, no spill): with the 168 nvcc chose, one block, 8 warps, at 25% of the bandwidth on
// the A30. In double precision 128 registers spill 700 bytes, so it keeps its 168.
// With `Split` (several processes), a leg across a cut comes from the buffers of `halo`; a
// block never spans two processes, so such a leg always crosses.
// -------------------------------------------------------------------------
template <typename Float, bool Split = false>
__global__ void __launch_bounds__(256, sizeof(Float) == 4 ? 2 : 1) hop_split_kernel(
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
    Float mass, Float b5, Float c5,
    Halo<Float> halo = {})
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
    hops<Float, Split>(acc, d_in + first, d_gauge, link_rows, xc, is, Ls, dims, bc, false,
                       static_cast<Float>(1.0), kappa, mass, b5, c5, 1 << leg, &halo, fibre / Ls);
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
