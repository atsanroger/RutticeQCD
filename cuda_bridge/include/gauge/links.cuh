#ifndef BRIDGE_CUDA_GAUGE_LINKS_CUH
#define BRIDGE_CUDA_GAUGE_LINKS_CUH

#include "stout.cuh"

// What HMC does with the links and the momenta on the device, in double precision, on the
// fields of matrices of stout.cuh (one 3 x 3 complex matrix per link): the momentum update,
// U <- exp(eps iP) U with the reunitarisation of Bridge++, the traceless anti-Hermitian part,
// products link by link, the force and the energy density of the plaquette and rectangle
// gauge actions, and |P|^2 by site. As hmc/mod.rs and hmc/gauge_action.rs on the CPU.

namespace bridge_cuda {

// y += a x for n reals.
__global__ void links_axpy_kernel(double* __restrict__ y, double a, const double* __restrict__ x, size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// reunitarize of hmc/mod.rs: the first row normalised, the second made orthogonal to it and
// normalised, the third the complex conjugate of their cross product.
__device__ __forceinline__ M3 reunitarize(const M3& m)
{
    M3 r;
    double n0 = 0;
    #pragma unroll
    for (int k = 0; k < 3; ++k) n0 += m.re[0][k] * m.re[0][k] + m.im[0][k] * m.im[0][k];
    n0 = 1.0 / sqrt(n0);
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        r.re[0][k] = m.re[0][k] * n0;
        r.im[0][k] = m.im[0][k] * n0;
    }
    // dot = conj(r0) . r1
    double dr = 0, di = 0;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        dr += r.re[0][k] * m.re[1][k] + r.im[0][k] * m.im[1][k];
        di += r.re[0][k] * m.im[1][k] - r.im[0][k] * m.re[1][k];
    }
    double n1 = 0;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        r.re[1][k] = m.re[1][k] - (dr * r.re[0][k] - di * r.im[0][k]);
        r.im[1][k] = m.im[1][k] - (dr * r.im[0][k] + di * r.re[0][k]);
        n1 += r.re[1][k] * r.re[1][k] + r.im[1][k] * r.im[1][k];
    }
    n1 = 1.0 / sqrt(n1);
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        r.re[1][k] *= n1;
        r.im[1][k] *= n1;
    }
    // r2 = conj(r0 x r1)
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int a = (k + 1) % 3, b = (k + 2) % 3;
        const double xr = (r.re[0][a] * r.re[1][b] - r.im[0][a] * r.im[1][b])
                        - (r.re[0][b] * r.re[1][a] - r.im[0][b] * r.im[1][a]);
        const double xi = (r.re[0][a] * r.im[1][b] + r.im[0][a] * r.re[1][b])
                        - (r.re[0][b] * r.im[1][a] + r.im[0][b] * r.re[1][a]);
        r.re[2][k] = xr;
        r.im[2][k] = -xi;
    }
    return r;
}

// U <- exp(eps iP) U by the nested Taylor series to order n, then reunitarised.
__global__ void links_exp_update_kernel(double* __restrict__ u, const double* __restrict__ p, double eps, int n, int vol)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(vol) * 4) return;
    const int mu = static_cast<int>(t / vol);
    const size_t x = t % vol;
    const M3 u0 = load(u, mu, x), h = load(p, mu, x);
    M3 u1 = u0;
    for (int k = 0; k < n; ++k) u1 = add(scale(mul(h, u1), eps / (n - k)), u0);
    store(u, mu, x, reunitarize(u1));
}

// out = s TA(m) for every link.
__global__ void links_ta_kernel(double* __restrict__ out, const double* __restrict__ m, double s, int vol)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(vol) * 4) return;
    const int mu = static_cast<int>(t / vol);
    const size_t x = t % vol;
    store(out, mu, x, scale(traceless_antihermitian(load(m, mu, x)), s));
}

// out = a b, or a^dag b, link by link.
__global__ void links_mul_kernel(double* __restrict__ out, const double* __restrict__ a, const double* __restrict__ b, int adjoint_a, int vol)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(vol) * 4) return;
    const int mu = static_cast<int>(t / vol);
    const size_t x = t % vol;
    const M3 ma = load(a, mu, x);
    store(out, mu, x, mul(adjoint_a ? adj(ma) : ma, load(b, mu, x)));
}

// The product of the links along a path from x: steps (d, forward), returning where it ends.
__device__ __forceinline__ M3 walk(const double* u, size_t x, const int* dirs, const int* fwd, int len, const StoutGeometry& g, size_t* end = nullptr)
{
    M3 m = m3_identity();
    size_t y = x;
    for (int k = 0; k < len; ++k) {
        if (fwd[k]) {
            m = mul(m, load(u, dirs[k], y));
            y = shift(y, dirs[k], 1, g);
        } else {
            y = shift(y, dirs[k], -1, g);
            m = mul(m, adj(load(u, dirs[k], y)));
        }
    }
    if (end != nullptr) *end = y;
    return m;
}

// F_mu(x) = -(beta/3) TA[U_mu(x) Sigma_mu(x)^dag], Sigma the staples of the plaquettes (weight
// c_plaq) and of the 1 x 2 rectangles (c_rect), the paths of staple_paths of gauge_action.rs.
__global__ void gauge_force_kernel(double* __restrict__ f, const double* __restrict__ u, double beta, double c_plaq, double c_rect, StoutGeometry g)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(g.vol) * 4) return;
    const int mu = static_cast<int>(t / g.vol);
    const size_t here = t % g.vol, x = extended(here, g);
    M3 sigma = m3_zero();
    for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) continue;
        // (nu, f), (mu, f), (nu, b) | (nu, b), (mu, f), (nu, f)
        const int p1d[3] = {nu, mu, nu}, p1f[3] = {1, 1, 0};
        const int p2d[3] = {nu, mu, nu}, p2f[3] = {0, 1, 1};
        if (c_plaq != 0.0) {
            sigma = add(sigma, scale(walk(u, x, p1d, p1f, 3, g), c_plaq));
            sigma = add(sigma, scale(walk(u, x, p2d, p2f, 3, g), c_plaq));
        }
        if (c_rect != 0.0) {
            const int r[6][5] = {{nu, mu, mu, nu, mu}, {mu, nu, mu, mu, nu}, {nu, nu, mu, nu, nu},
                                 {nu, mu, mu, nu, mu}, {mu, nu, mu, mu, nu}, {nu, nu, mu, nu, nu}};
            const int rf[6][5] = {{1, 1, 1, 0, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 0, 0},
                                  {0, 1, 1, 1, 0}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}};
            #pragma unroll
            for (int k = 0; k < 6; ++k) sigma = add(sigma, scale(walk(u, x, r[k], rf[k], 5, g), c_rect));
        }
    }
    const M3 m = mul(load(u, mu, x), adj(sigma));
    store(f, mu, here, scale(traceless_antihermitian(m), -beta / 3.0));
}

// Per site: the sum of Re tr of the six plaquettes (mu < nu) and of the twelve rectangles
// (2 mu x nu for every ordered pair), as RectangleAction::action walks them.
__global__ void gauge_traces_kernel(double* __restrict__ plaq, double* __restrict__ rect, const double* __restrict__ u, int with_rect, StoutGeometry g)
{
    const size_t here = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (here >= static_cast<size_t>(g.vol)) return;
    const size_t x = extended(here, g);
    double p = 0, r = 0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            if (nu == mu) continue;
            if (mu < nu) {
                const int d[4] = {mu, nu, mu, nu}, f[4] = {1, 1, 0, 0};
                p += trace(walk(u, x, d, f, 4, g)).re;
            }
            if (with_rect) {
                const int d[6] = {mu, mu, nu, mu, mu, nu}, f[6] = {1, 1, 1, 0, 0, 0};
                r += trace(walk(u, x, d, f, 6, g)).re;
            }
        }
    }
    plaq[here] = p;
    rect[here] = r;
}

// The local field `local` into the interior of the extended one (see stout.cuh).
__global__ void links_extend_kernel(double* __restrict__ ext, const double* __restrict__ local, StoutGeometry g)
{
    const size_t here = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (here >= static_cast<size_t>(g.vol)) return;
    const size_t x = extended(here, g);
    #pragma unroll 4
    for (int c = 0; c < 72; ++c) ext[link_at(c, x)] = local[link_at(c, here)];
}

// The layers first .. first + depth[mu] - 1 of direction mu of the extended field, all of their
// extent in the other directions (the halos filled so far with them), to the buffer (`pack`) or
// from it. In the buffer site after site, numbered lexically over the layers, 72 reals each,
// real fastest across a warp of sites: (c / 72 ...) as link_at with the layer's sites.
__global__ void links_layers_kernel(double* __restrict__ buf, double* __restrict__ ext, int mu, int first, int pack, StoutGeometry g)
{
    int extent[4] = {g.dims[0], g.dims[1], g.dims[2], g.dims[3]};
    extent[mu] = g.depth[mu];
    const size_t sites = static_cast<size_t>(extent[0]) * extent[1] * extent[2] * extent[3];
    const size_t s = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= sites) return;
    int c[4];
    size_t r = s;
    #pragma unroll
    for (int nu = 0; nu < 4; ++nu) {
        c[nu] = static_cast<int>(r % extent[nu]);
        r /= extent[nu];
    }
    c[mu] += first;
    const size_t x = c[0] + static_cast<size_t>(g.dims[0]) * (c[1] + static_cast<size_t>(g.dims[1]) * (c[2] + static_cast<size_t>(g.dims[2]) * c[3]));
    #pragma unroll 4
    for (int k = 0; k < 72; ++k) {
        if (pack) {
            buf[link_at(k, s)] = ext[link_at(k, x)];
        } else {
            ext[link_at(k, x)] = buf[link_at(k, s)];
        }
    }
}

// Per site: the sum of the squares of the reals of its four matrices.
__global__ void links_norm2_kernel(double* __restrict__ out, const double* __restrict__ m, int vol)
{
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= static_cast<size_t>(vol)) return;
    double s = 0;
    for (int c = 0; c < 72; ++c) {
        const double v = m[link_at(c, x)];
        s += v * v;
    }
    out[x] = s;
}

} // namespace bridge_cuda

#endif
