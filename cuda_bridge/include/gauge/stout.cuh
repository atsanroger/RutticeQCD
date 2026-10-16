#ifndef BRIDGE_CUDA_GAUGE_STOUT_CUH
#define BRIDGE_CUDA_GAUGE_STOUT_CUH

#include "../base/constant.h"
#include <cuda_runtime.h>

// Stout smearing of the links and the chain rule of its force (Morningstar and Peardon,
// hep-lat/0311018), in double precision, as gauge/smearing.rs has them on the CPU.
//
// Links and the fields of matrices that go with them (staples, Sigma, i Theta) are whole 3 x 3
// complex matrices, row-major (re, im), 18 reals for each direction: real c of direction mu at
// site x is at link_at(18 mu + c, x), in blocks of W sites (fold.cuh's coarse fields are the
// same with 2 reals). Sites are numbered x + nx (y + ny (z + nz t)), the lattice periodic.

namespace bridge_cuda {

__host__ __device__ __forceinline__ size_t link_at(int c, size_t x) {
    return ((x / WarpSize) * 72 + c) * WarpSize + x % WarpSize;
}

struct StoutGeometry {
    int dims[4];
    int vol;
    double rho[4][4];
};

struct M3 {
    double re[3][3], im[3][3];
};

__device__ __forceinline__ M3 m3_zero() {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) m.re[i][j] = m.im[i][j] = 0;
    return m;
}

__device__ __forceinline__ M3 m3_identity() {
    M3 m = m3_zero();
    #pragma unroll
    for (int i = 0; i < 3; ++i) m.re[i][i] = 1;
    return m;
}

__device__ __forceinline__ M3 load(const double* f, int mu, size_t x) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            m.re[i][j] = f[link_at(18 * mu + 2 * (3 * i + j), x)];
            m.im[i][j] = f[link_at(18 * mu + 2 * (3 * i + j) + 1, x)];
        }
    return m;
}

__device__ __forceinline__ void store(double* f, int mu, size_t x, const M3& m) {
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            f[link_at(18 * mu + 2 * (3 * i + j), x)] = m.re[i][j];
            f[link_at(18 * mu + 2 * (3 * i + j) + 1, x)] = m.im[i][j];
        }
}

__device__ __forceinline__ M3 mul(const M3& a, const M3& b) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            double re = 0, im = 0;
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                re += a.re[i][k] * b.re[k][j] - a.im[i][k] * b.im[k][j];
                im += a.re[i][k] * b.im[k][j] + a.im[i][k] * b.re[k][j];
            }
            m.re[i][j] = re;
            m.im[i][j] = im;
        }
    return m;
}

__device__ __forceinline__ M3 adj(const M3& a) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            m.re[i][j] = a.re[j][i];
            m.im[i][j] = -a.im[j][i];
        }
    return m;
}

__device__ __forceinline__ M3 add(const M3& a, const M3& b) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            m.re[i][j] = a.re[i][j] + b.re[i][j];
            m.im[i][j] = a.im[i][j] + b.im[i][j];
        }
    return m;
}

__device__ __forceinline__ M3 sub(const M3& a, const M3& b) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            m.re[i][j] = a.re[i][j] - b.re[i][j];
            m.im[i][j] = a.im[i][j] - b.im[i][j];
        }
    return m;
}

__device__ __forceinline__ M3 scale(const M3& a, double s) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            m.re[i][j] = s * a.re[i][j];
            m.im[i][j] = s * a.im[i][j];
        }
    return m;
}

struct C {
    double re, im;
};

__device__ __forceinline__ C cmul(C a, C b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
__device__ __forceinline__ C cadd(C a, C b) { return {a.re + b.re, a.im + b.im}; }
__device__ __forceinline__ C csub(C a, C b) { return {a.re - b.re, a.im - b.im}; }
__device__ __forceinline__ C cscale(C a, double s) { return {s * a.re, s * a.im}; }

__device__ __forceinline__ C trace(const M3& a) {
    return {a.re[0][0] + a.re[1][1] + a.re[2][2], a.im[0][0] + a.im[1][1] + a.im[2][2]};
}

// (M - M^dag)/2 - tr(M - M^dag)/(2 N_c), as traceless_antihermitian of the CPU.
__device__ __forceinline__ M3 traceless_antihermitian(const M3& m) {
    M3 a = scale(sub(m, adj(m)), 0.5);
    const double shift = trace(a).im / 3.0;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        a.re[c][c] = 0.0;
        a.im[c][c] = a.im[c][c] - shift;
    }
    return a;
}

// a 1 + b Q + c Q^2, with Q = -i iQ and Q^2 = -(iQ)^2.
__device__ __forceinline__ M3 poly(C a, C b, C c, const M3& iq, const M3& iq2) {
    M3 m;
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            C one = (i == j) ? a : C{0.0, 0.0};
            const C q{iq.im[i][j], -iq.re[i][j]};
            const C q2{iq2.re[i][j], iq2.im[i][j]};
            C v = csub(cadd(one, cmul(b, q)), cmul(c, q2));
            m.re[i][j] = v.re;
            m.im[i][j] = v.im;
        }
    return m;
}

__device__ __forceinline__ double xi0(double w) { return (w == 0.0) ? 1.0 : sin(w) / w; }

__device__ __forceinline__ double xi1(double w) {
    if (w < 0.25) {
        const double w2 = w * w;
        const double c0 = -1.0 / 3.0, c1 = 1.0 / 30.0, c2 = -1.0 / 840.0, c3 = 1.0 / 45360.0,
                     c4 = -1.0 / 3991680.0;
        return c0 + w2 * (c1 + w2 * (c2 + w2 * (c3 + w2 * c4)));
    }
    return (w * cos(w) - sin(w)) / (w * w * w);
}

// exp(iQ) for the link u with weighted staple sum c (StoutExp of the CPU).
struct StoutExp {
    M3 iq, iq2;
    C f[3];
    double u, w;
    bool zero;

    __device__ StoutExp(const M3& c, const M3& link) {
        iq = traceless_antihermitian(mul(c, adj(link)));
        iq2 = mul(iq, iq);
        const M3 iq3 = mul(iq, iq2);
        double norm = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i)
            #pragma unroll
            for (int j = 0; j < 3; ++j) norm += iq.re[i][j] * iq.re[i][j] + iq.im[i][j] * iq.im[i][j];
        if (norm <= 1e-10) {
            f[0] = f[1] = f[2] = C{0.0, 0.0};
            u = w = 0.0;
            zero = true;
            return;
        }
        zero = false;
        const double c0 = -trace(iq3).im / 3.0;
        const double c1 = -0.5 * trace(iq2).re;
        const double c13r = sqrt(c1 / 3.0);
        const double c0max = 2.0 * c13r * c13r * c13r;
        const double theta = acos(c0 / c0max);
        u = c13r * cos(theta / 3.0);
        w = sqrt(c1) * sin(theta / 3.0);
        const double u2 = u * u, w2 = w * w;
        const double cos_w = cos(w), cos_u = cos(u), sin_u = sin(u);
        const double x0 = xi0(w);
        const C emiu{cos_u, -sin_u};
        const C e2iu{cos_u * cos_u - sin_u * sin_u, 2.0 * sin_u * cos_u};
        const C h0 = cadd(cmul(e2iu, C{u2 - w2, 0.0}),
                          cmul(emiu, C{8.0 * u2 * cos_w, 2.0 * u * (3.0 * u2 + w2) * x0}));
        const C h1 = csub(cmul(C{2.0 * u, 0.0}, e2iu),
                          cmul(emiu, C{2.0 * u * cos_w, -(3.0 * u2 - w2) * x0}));
        const C h2 = csub(e2iu, cmul(emiu, C{cos_w, 3.0 * u * x0}));
        const double fden = 1.0 / (9.0 * u2 - w2);
        f[0] = cscale(h0, fden);
        f[1] = cscale(h1, fden);
        f[2] = cscale(h2, fden);
    }

    __device__ M3 exp() const {
        if (zero) return m3_identity();
        return poly(f[0], f[1], f[2], iq, iq2);
    }

    // i Lambda of Morningstar-Peardon (73) for Sigma' at this link.
    __device__ M3 lambda(const M3& sigma, const M3& link) const {
        if (zero) return m3_zero();
        const double x0 = xi0(w), x1 = xi1(w);
        const double u2 = u * u, w2 = w * w;
        const double cos_w = cos(w);
        const C emiu{cos(u), -sin(u)};
        const C e2iu{cos(2.0 * u), sin(2.0 * u)};
        const C r01 = cadd(cmul(C{2.0 * u, 2.0 * (u2 - w2)}, e2iu),
                           cmul(emiu, C{16.0 * u * cos_w + 2.0 * u * (3.0 * u2 + w2) * x0,
                                        -8.0 * u2 * cos_w + 2.0 * (9.0 * u2 + w2) * x0}));
        const C r11 = cadd(cmul(C{2.0, 4.0 * u}, e2iu),
                           cmul(emiu, C{-2.0 * cos_w + (3.0 * u2 - w2) * x0,
                                        2.0 * u * cos_w + 6.0 * u * x0}));
        const C r21 = cadd(cmul(C{0.0, 2.0}, e2iu), cmul(emiu, C{-3.0 * u * x0, cos_w - 3.0 * x0}));
        const C r02 = cadd(cmul(C{-2.0, 0.0}, e2iu),
                           cmul(emiu, C{-8.0 * u2 * x0, 2.0 * u * (cos_w + x0 + 3.0 * u2 * x1)}));
        const C r12 = cmul(emiu, C{2.0 * u * x0, -cos_w - x0 + 3.0 * u2 * x1});
        const C r22 = cmul(emiu, C{x0, -3.0 * u * x1});
        const double fden = 1.0 / (2.0 * (9.0 * u2 - w2) * (9.0 * u2 - w2));
        const C k1{2.0 * u, 0.0}, k2{3.0 * u2 - w2, 0.0}, k3{30.0 * u2 + 2.0 * w2, 0.0};
        const C b10 = cscale(csub(cadd(cmul(k1, r01), cmul(k2, r02)), cmul(k3, f[0])), fden);
        const C b11 = cscale(csub(cadd(cmul(k1, r11), cmul(k2, r12)), cmul(k3, f[1])), fden);
        const C b12 = cscale(csub(cadd(cmul(k1, r21), cmul(k2, r22)), cmul(k3, f[2])), fden);
        const C k4{3.0 * u, 0.0}, k5{24.0 * u, 0.0};
        const C b20 = cscale(csub(csub(r01, cmul(k4, r02)), cmul(k5, f[0])), fden);
        const C b21 = cscale(csub(csub(r11, cmul(k4, r12)), cmul(k5, f[1])), fden);
        const C b22 = cscale(csub(csub(r21, cmul(k4, r22)), cmul(k5, f[2])), fden);
        const M3 b1 = poly(b10, b11, b12, iq, iq2);
        const M3 b2 = poly(b20, b21, b22, iq, iq2);
        const M3 us = mul(link, sigma);
        const C tr1 = trace(mul(us, b1));
        const C tr2 = trace(mul(us, b2));
        const M3 iqus = mul(iq, us);
        const M3 iusq = mul(us, iq);
        M3 gamma;
        #pragma unroll
        for (int i = 0; i < 3; ++i)
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                const C q{iq.im[i][j], -iq.re[i][j]};
                const C a{iqus.im[i][j], -iqus.re[i][j]};
                const C b{iusq.im[i][j], -iusq.re[i][j]};
                const C usij{us.re[i][j], us.im[i][j]};
                const C q2{iq2.re[i][j], iq2.im[i][j]};
                C qt = csub(cmul(tr1, q), cmul(tr2, q2));
                qt = cadd(qt, cmul(f[1], usij));
                qt = cadd(qt, cmul(f[2], a));
                qt = cadd(qt, cmul(f[2], b));
                // i Gamma
                gamma.re[i][j] = -qt.im;
                gamma.im[i][j] = qt.re;
            }
        return traceless_antihermitian(gamma);
    }
};

__device__ __forceinline__ size_t shift(size_t x, int mu, int step, const StoutGeometry& g) {
    int c[4];
    size_t r = x;
    #pragma unroll
    for (int nu = 0; nu < 4; ++nu) {
        c[nu] = static_cast<int>(r % g.dims[nu]);
        r /= g.dims[nu];
    }
    c[mu] = (c[mu] + step + g.dims[mu]) % g.dims[mu];
    return c[0] + static_cast<size_t>(g.dims[0]) * (c[1] + static_cast<size_t>(g.dims[1]) * (c[2] + static_cast<size_t>(g.dims[2]) * c[3]));
}

// C_mu(x) = sum_(nu != mu) rho_mu,nu [U_nu(x) U_mu(x+nu) U_nu(x+mu)^dag
//                                  + U_nu(x-nu)^dag U_mu(x-nu) U_nu(x-nu+mu)].
__device__ __forceinline__ M3 staples(const double* u, int mu, size_t x, const StoutGeometry& g) {
    M3 c = m3_zero();
    for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) continue;
        const double r = g.rho[mu][nu];
        if (r == 0.0) continue;
        const size_t xn = shift(x, nu, 1, g), xm = shift(x, mu, 1, g), y = shift(x, nu, -1, g);
        const size_t ym = shift(y, mu, 1, g);
        const M3 upper = mul(mul(load(u, nu, x), load(u, mu, xn)), adj(load(u, nu, xm)));
        const M3 lower = mul(mul(adj(load(u, nu, y)), load(u, mu, y)), load(u, nu, ym));
        c = add(c, scale(add(upper, lower), r));
    }
    return c;
}

// v = exp(iQ) U for every link: a thread for each (site, direction).
__global__ void stout_step_kernel(double* __restrict__ v, const double* __restrict__ u, StoutGeometry g)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(g.vol) * 4) return;
    const int mu = static_cast<int>(t / g.vol);
    const size_t x = t % g.vol;
    const M3 link = load(u, mu, x);
    const StoutExp e(staples(u, mu, x, g), link);
    store(v, mu, x, mul(e.exp(), link));
}

// The local part of a step of the chain rule: Sigma = Sigma' exp(iQ) + C^dag i Lambda and
// i Theta = i Lambda U.
__global__ void stout_chain_local_kernel(
    double* __restrict__ sigma, double* __restrict__ itheta, const double* __restrict__ sigma_prime,
    const double* __restrict__ u, StoutGeometry g)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(g.vol) * 4) return;
    const int mu = static_cast<int>(t / g.vol);
    const size_t x = t % g.vol;
    const M3 link = load(u, mu, x), c = staples(u, mu, x, g), s = load(sigma_prime, mu, x);
    const StoutExp e(c, link);
    const M3 lambda = e.lambda(s, link);
    store(itheta, mu, x, mul(lambda, link));
    store(sigma, mu, x, add(mul(s, e.exp()), mul(adj(c), lambda)));
}

// The terms of Sigma from the links whose staples contain U_mu(x), added to it.
__global__ void stout_chain_staples_kernel(
    double* __restrict__ sigma, const double* __restrict__ u, const double* __restrict__ itheta,
    StoutGeometry g)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(g.vol) * 4) return;
    const int mu = static_cast<int>(t / g.vol);
    const size_t x = t % g.vol;
    M3 acc = load(sigma, mu, x);
    for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) continue;
        const double r = g.rho[mu][nu];
        if (r == 0.0) continue;
        const size_t xm = shift(x, mu, 1, g), xn = shift(x, nu, 1, g), y = shift(x, nu, -1, g);
        const size_t ym = shift(y, mu, 1, g);
        const M3 vnu_xm = load(u, nu, xm);
        const M3 vmu_xn_d = adj(load(u, mu, xn));
        const M3 vnu_d = adj(load(u, nu, x));
        const M3 vnu_ym_d = adj(load(u, nu, ym));
        const M3 vmu_y_d = adj(load(u, mu, y));
        const M3 vnu_y = load(u, nu, y);
        M3 each = mul(mul(vnu_xm, vmu_xn_d), adj(load(itheta, nu, x)));
        each = add(each, mul(mul(vnu_ym_d, adj(load(itheta, mu, y))), vnu_y));
        each = add(each, mul(mul(vnu_ym_d, vmu_y_d), load(itheta, nu, y)));
        each = add(each, mul(mul(adj(load(itheta, nu, ym)), vmu_y_d), vnu_y));
        each = add(each, mul(mul(load(itheta, nu, xm), vmu_xn_d), vnu_d));
        each = add(each, mul(mul(vnu_xm, adj(load(itheta, mu, xn))), vnu_d));
        acc = add(acc, scale(each, r));
    }
    store(sigma, mu, x, acc);
}

} // namespace bridge_cuda

#endif
