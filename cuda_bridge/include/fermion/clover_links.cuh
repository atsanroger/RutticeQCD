#ifndef BRIDGE_CUDA_FERMION_CLOVER_LINKS_CUH
#define BRIDGE_CUDA_FERMION_CLOVER_LINKS_CUH

#include "../base/aosoa.cuh"
#include "../base/gamma.cuh"
#include "../gauge/links.cuh"

// What the clover term makes of the links: its blocks T and T^-1 (clover.cuh's layout), and the
// derivative of the clover term by the links (hmc/fermion_action.rs's CloverInsertion and
// clover_leaves). All in double precision.
//
// T = 1 - kappa c_SW sum_(mu<nu) (gamma_mu gamma_nu) (x) F_munu, F_munu = (Q - Q^dag)/8 with Q the
// four leaves at x (gauge/mod.rs's field_strength). T commutes with gamma_5; on the eigenvectors
// (e_a + s e_(a+2))/sqrt(2) of gamma_5 (s = +1, -1; a = 0, 1) it is two Hermitian 6 x 6 blocks,
// index colour + 3 a.
//
// The derivative goes in two steps. For every pair (zeta, eta) of the force,
//
//   B_munu(x) += kappa c_SW (A - A^dag)/4,   A_ij = sum_s (gamma_mu gamma_nu eta)_(s,i) conj(zeta_(s,j)),
//
// a matrix per site and plane; then once for all pairs, the leaves through every link with B
// inserted at their corner. B of planes 0-3 is kept as the four directions of a field of links,
// that of planes 4-5 as the first two of another, so the halo code of links extends them.
//
// The links and B are extended (stout.cuh) one site deep: a leaf leaves its corner by one step in
// each of two directions.

namespace bridge_cuda {

// The leaves of a plane at a site, the links of a leaf, and the blocks of T (gamma_5 = +1, -1).
constexpr int Leaves = 4;
constexpr int LeafLinks = 4;
constexpr int Chiralities = 2;

// The planes (mu, nu), mu < nu, in the order of gauge/mod.rs's PLANES.
__device__ __forceinline__ void plane_of(int plane, int& mu, int& nu) {
    const int m[Nplanes] = {0, 0, 0, 1, 1, 2}, n[Nplanes] = {1, 2, 3, 2, 3, 3};
    mu = m[plane];
    nu = n[plane];
}

// gamma_mu gamma_nu in the Dirac representation (gamma.cuh's table): row r has one entry, in
// column col[r], of value re[r] + i im[r]. (gamma_mu gamma_nu)_(r, col_nu[col_mu[r]]) =
// val_mu[r] val_nu[col_mu[r]].
__device__ __forceinline__ void gamma_pair(int mu, int nu, int col[Nspin], double re[Nspin], double im[Nspin]) {
    #pragma unroll
    for (int r = 0; r < Nspin; ++r) {
        const int k = dirac_col(mu, r);
        col[r] = dirac_col(nu, k);
        const double a = dirac_re(mu, r), b = dirac_im(mu, r), c = dirac_re(nu, k), d = dirac_im(nu, k);
        re[r] = a * c - b * d;
        im[r] = a * d + b * c;
    }
}

// The four leaves of the plane (mu, nu) at a site, as steps (direction, forward): x -> x+mu ->
// x+mu+nu -> x+nu -> x, then the three others turning the same way.
__device__ __forceinline__ void leaves_of(int mu, int nu, int d[Leaves][LeafLinks], int f[Leaves][LeafLinks]) {
    const int dd[Leaves][LeafLinks] = {{mu, nu, mu, nu}, {nu, mu, nu, mu}, {mu, nu, mu, nu}, {nu, mu, nu, mu}};
    const int ff[Leaves][LeafLinks] = {{1, 1, 0, 0}, {1, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 0}};
    #pragma unroll
    for (int l = 0; l < Leaves; ++l)
        #pragma unroll
        for (int k = 0; k < LeafLinks; ++k) {
            d[l][k] = dd[l][k];
            f[l][k] = ff[l][k];
        }
}

// F_munu at the extended site x.
__device__ __forceinline__ M3 field_strength(const double* u, size_t x, int mu, int nu, const StoutGeometry& g) {
    int d[Leaves][LeafLinks], f[Leaves][LeafLinks];
    leaves_of(mu, nu, d, f);
    M3 q = m3_zero();
    #pragma unroll
    for (int l = 0; l < Leaves; ++l) q = add(q, walk(u, x, d[l], f[l], LeafLinks, g));
    return scale(sub(q, adj(q)), 0.125);
}

// Real k of block w (0: gamma_5 = +1) of `site` in the blocks t.
__device__ __forceinline__ size_t block_at(int w, int k, size_t site) {
    return (site / W * Nclover + (Nclover / 2) * w + k) * W + site % W;
}

// Entry (r, c) of the packed Hermitian block w of `site`: the Nblock real diagonal entries, then
// the upper triangle row by row as (re, im); before row a come a (Nblock - 1) - a (a - 1) / 2 of
// them (row q has Nblock - 1 - q).
__device__ __forceinline__ C packed_entry(const double* t, int w, size_t site, int r, int c) {
    if (r == c) return {t[block_at(w, r, site)], 0.0};
    const bool upper = r < c;
    const int a = upper ? r : c, b = upper ? c : r;
    const int k = Nblock + Ncomplex * (a * (Nblock - 1) - a * (a - 1) / 2 + b - a - 1);
    const double im = t[block_at(w, k + 1, site)];
    return {t[block_at(w, k, site)], upper ? im : -im};
}

// Entry (spin a, colour i; spin b, colour j) of the 12 x 12 matrix of the two blocks of `site`:
// sum_s <a i|v_s> B_s <v_s|b j>, the components of v_s along e_(a+2) carrying s, those of the two
// 1/sqrt(2) making 1/2.
__device__ __forceinline__ C clover_entry(const double* t, size_t site, int a, int i, int b, int j) {
    C z = {0.0, 0.0};
    #pragma unroll
    for (int w = 0; w < Chiralities; ++w) {
        const double s = (w == 0) ? 1.0 : -1.0;
        const double f = 0.5 * ((a >= Nspin2) ? s : 1.0) * ((b >= Nspin2) ? s : 1.0);
        const int r = i + Nc * (a % Nspin2), c = j + Nc * (b % Nspin2);
        z = cadd(z, cscale(packed_entry(t, w, site, r, c), f));
    }
    return z;
}

// One thread a (site, block w): block w of T, of T^-1 (Gauss-Jordan with partial pivoting, as
// numerics::invert_complex), and ln |det| of the block, the sum of ln |pivot|. A pivot below the
// scale of the block sets `singular` to the site plus one (any such site).
__global__ void clover_blocks_kernel(
    double* __restrict__ t, double* __restrict__ t_inv, double* __restrict__ ln_det,
    double* __restrict__ singular, const double* __restrict__ u, double kappa_csw, StoutGeometry g)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= Chiralities * static_cast<size_t>(g.vol)) return;
    const int w = static_cast<int>(id / g.vol);
    const size_t here = id % g.vol, x = extended(here, g);
    const double s = (w == 0) ? 1.0 : -1.0;

    double ar[Nblock][Nblock], ai[Nblock][Nblock];
    #pragma unroll
    for (int r = 0; r < Nblock; ++r)
        #pragma unroll
        for (int c = 0; c < Nblock; ++c) {
            ar[r][c] = (r == c) ? 1.0 : 0.0;
            ai[r][c] = 0.0;
        }
    for (int plane = 0; plane < Nplanes; ++plane) {
        int mu, nu;
        plane_of(plane, mu, nu);
        const M3 fs = field_strength(u, x, mu, nu, g);
        int col[Nspin];
        double gre[Nspin], gim[Nspin];
        gamma_pair(mu, nu, col, gre, gim);
        // The 2 x 2 block of gamma_mu gamma_nu between the chiral vectors of sign s:
        // h_ab = (G_ab + s G_a,b+2 + s G_a+2,b + G_a+2,b+2) / 2.
        double hr[Nspin2][Nspin2] = {{0, 0}, {0, 0}}, hi[Nspin2][Nspin2] = {{0, 0}, {0, 0}};
        #pragma unroll
        for (int r = 0; r < Nspin; ++r) {
            const double f = 0.5 * ((r >= Nspin2) ? s : 1.0) * ((col[r] >= Nspin2) ? s : 1.0);
            hr[r % Nspin2][col[r] % Nspin2] += f * gre[r];
            hi[r % Nspin2][col[r] % Nspin2] += f * gim[r];
        }
        #pragma unroll
        for (int a = 0; a < Nspin2; ++a)
            #pragma unroll
            for (int b = 0; b < Nspin2; ++b)
                #pragma unroll
                for (int i = 0; i < Nc; ++i)
                    #pragma unroll
                    for (int j = 0; j < Nc; ++j) {
                        const double zr = hr[a][b] * fs.re[i][j] - hi[a][b] * fs.im[i][j];
                        const double zi = hr[a][b] * fs.im[i][j] + hi[a][b] * fs.re[i][j];
                        ar[i + Nc * a][j + Nc * b] -= kappa_csw * zr;
                        ai[i + Nc * a][j + Nc * b] -= kappa_csw * zi;
                    }
    }

    // Pack T before the elimination overwrites it.
    double scale_of = 0.0;
    int k = Nblock;
    #pragma unroll
    for (int r = 0; r < Nblock; ++r) {
        t[block_at(w, r, here)] = ar[r][r];
        #pragma unroll
        for (int c = r + 1; c < Nblock; ++c) {
            t[block_at(w, k, here)] = ar[r][c];
            t[block_at(w, k + 1, here)] = ai[r][c];
            k += Ncomplex;
        }
        #pragma unroll
        for (int c = 0; c < Nblock; ++c) scale_of = fmax(scale_of, fmax(fabs(ar[r][c]), fabs(ai[r][c])));
    }

    double br[Nblock][Nblock], bi[Nblock][Nblock];
    #pragma unroll
    for (int r = 0; r < Nblock; ++r)
        #pragma unroll
        for (int c = 0; c < Nblock; ++c) {
            br[r][c] = (r == c) ? 1.0 : 0.0;
            bi[r][c] = 0.0;
        }
    double log_det = 0.0;
    for (int col = 0; col < Nblock; ++col) {
        int pivot = col;
        double best = 0.0;
        for (int r = col; r < Nblock; ++r) {
            const double size = ar[r][col] * ar[r][col] + ai[r][col] * ai[r][col];
            if (size > best) {
                best = size;
                pivot = r;
            }
        }
        if (sqrt(best) <= scale_of * 2.220446049250313e-16 * Nblock) {
            *singular = static_cast<double>(here + 1);
            best = 1.0;
        }
        if (pivot != col) {
            for (int c = 0; c < Nblock; ++c) {
                double tmp;
                tmp = ar[pivot][c]; ar[pivot][c] = ar[col][c]; ar[col][c] = tmp;
                tmp = ai[pivot][c]; ai[pivot][c] = ai[col][c]; ai[col][c] = tmp;
                tmp = br[pivot][c]; br[pivot][c] = br[col][c]; br[col][c] = tmp;
                tmp = bi[pivot][c]; bi[pivot][c] = bi[col][c]; bi[col][c] = tmp;
            }
        }
        log_det += 0.5 * log(best);
        // The pivot row times 1/pivot = conj(pivot)/|pivot|^2.
        const double ir = ar[col][col] / best, ii = -ai[col][col] / best;
        for (int c = 0; c < Nblock; ++c) {
            double xr = ar[col][c], xi = ai[col][c];
            ar[col][c] = xr * ir - xi * ii;
            ai[col][c] = xr * ii + xi * ir;
            xr = br[col][c];
            xi = bi[col][c];
            br[col][c] = xr * ir - xi * ii;
            bi[col][c] = xr * ii + xi * ir;
        }
        for (int r = 0; r < Nblock; ++r) {
            if (r == col) continue;
            const double fr = ar[r][col], fi = ai[r][col];
            if (fr == 0.0 && fi == 0.0) continue;
            for (int c = 0; c < Nblock; ++c) {
                ar[r][c] -= fr * ar[col][c] - fi * ai[col][c];
                ai[r][c] -= fr * ai[col][c] + fi * ar[col][c];
                br[r][c] -= fr * br[col][c] - fi * bi[col][c];
                bi[r][c] -= fr * bi[col][c] + fi * br[col][c];
            }
        }
    }

    // The inverse of a Hermitian matrix is Hermitian up to rounding: pack its upper triangle.
    k = Nblock;
    #pragma unroll
    for (int r = 0; r < Nblock; ++r) {
        t_inv[block_at(w, r, here)] = br[r][r];
        #pragma unroll
        for (int c = r + 1; c < Nblock; ++c) {
            t_inv[block_at(w, k, here)] = br[r][c];
            t_inv[block_at(w, k + 1, here)] = bi[r][c];
            k += Ncomplex;
        }
    }
    ln_det[id] = log_det;
}

// B += coef (A - A^dag) for the six planes, A_ij = sum_s (G eta)_(s,i) conj(zeta_(s,j)), G =
// gamma_mu gamma_nu, at every site; zeta and eta spinor fields of one vector on the whole local
// lattice.
__global__ void clover_insertion_kernel(
    double* __restrict__ b_lo, double* __restrict__ b_hi,
    const double* __restrict__ zeta, const double* __restrict__ eta, double coef, int vol)
{
    const int here = blockIdx.x * blockDim.x + threadIdx.x;
    if (here >= vol) return;
    double er[Nspin][Nc], ei[Nspin][Nc], zr[Nspin][Nc], zi[Nspin][Nc];
    #pragma unroll
    for (int s = 0; s < Nspin; ++s)
        #pragma unroll
        for (int c = 0; c < Nc; ++c) {
            er[s][c] = eta[aosoa_spinor_idx(c, s, 0, here)];
            ei[s][c] = eta[aosoa_spinor_idx(c, s, 1, here)];
            zr[s][c] = zeta[aosoa_spinor_idx(c, s, 0, here)];
            zi[s][c] = zeta[aosoa_spinor_idx(c, s, 1, here)];
        }
    for (int plane = 0; plane < Nplanes; ++plane) {
        int mu, nu;
        plane_of(plane, mu, nu);
        int col[Nspin];
        double gre[Nspin], gim[Nspin];
        gamma_pair(mu, nu, col, gre, gim);
        M3 a = m3_zero();
        #pragma unroll
        for (int s = 0; s < Nspin; ++s) {
            const int t = col[s];
            #pragma unroll
            for (int i = 0; i < Nc; ++i) {
                const double gr = gre[s] * er[t][i] - gim[s] * ei[t][i];
                const double gi = gre[s] * ei[t][i] + gim[s] * er[t][i];
                #pragma unroll
                for (int j = 0; j < Nc; ++j) {
                    // (gr + i gi) conj(zr + i zi)
                    a.re[i][j] += gr * zr[s][j] + gi * zi[s][j];
                    a.im[i][j] += gi * zr[s][j] - gr * zi[s][j];
                }
            }
        }
        double* b = (plane < Ndim) ? b_lo : b_hi;
        const int p = plane % Ndim;
        store(b, p, here, add(load(b, p, here), scale(sub(a, adj(a)), coef)));
    }
}

// The insertion of the derivative of -2 ln |det T| on the sites of `parity` (x + y + z + t of the
// local coordinates; the caller turns the global parity into it): the pairs (T^-1 e_k, e_k) of
// the twelve unit vectors, A_ij = sum_s G_(s,col s) conj((T^-1)_(s j, col_s i)). t_inv are the
// blocks of T^-1 of the local lattice; g the local lattice (depths 0).
__global__ void clover_log_det_insertion_kernel(
    double* __restrict__ b_lo, double* __restrict__ b_hi,
    const double* __restrict__ t_inv, double coef, int parity, StoutGeometry g)
{
    const int here = blockIdx.x * blockDim.x + threadIdx.x;
    if (here >= g.vol) return;
    int c[Ndim], r = here, sum = 0;
    #pragma unroll
    for (int mu = 0; mu < Ndim; ++mu) {
        c[mu] = r % g.local[mu];
        r /= g.local[mu];
        sum += c[mu];
    }
    if (sum % 2 != parity) return;
    for (int plane = 0; plane < Nplanes; ++plane) {
        int mu, nu;
        plane_of(plane, mu, nu);
        int col[Nspin];
        double gre[Nspin], gim[Nspin];
        gamma_pair(mu, nu, col, gre, gim);
        M3 a = m3_zero();
        #pragma unroll
        for (int s = 0; s < Nspin; ++s)
            #pragma unroll
            for (int i = 0; i < Nc; ++i)
                #pragma unroll
                for (int j = 0; j < Nc; ++j) {
                    const C z = clover_entry(t_inv, here, s, j, col[s], i);
                    // (gre + i gim) conj(z)
                    a.re[i][j] += gre[s] * z.re + gim[s] * z.im;
                    a.im[i][j] += gim[s] * z.re - gre[s] * z.im;
                }
        double* b = (plane < Ndim) ? b_lo : b_hi;
        const int p = plane % Ndim;
        store(b, p, here, add(load(b, p, here), scale(sub(a, adj(a)), coef)));
    }
}

// M_rho(x) = sum of the leaves through the link (x, rho) with B inserted at their corner: for a
// leaf that takes the link forward, + W1 B W2; backward, - W1 B W2, W1 the rest of the leaf from
// the link to the corner and W2 from the corner back (hmc/fermion_action.rs's clover_leaves). u,
// b_lo and b_hi extended one site deep; one thread a link.
__global__ void clover_leaves_kernel(
    double* __restrict__ m, const double* __restrict__ u,
    const double* __restrict__ b_lo, const double* __restrict__ b_hi, StoutGeometry g)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= Ndim * static_cast<size_t>(g.vol)) return;
    const int rho = static_cast<int>(id / g.vol);
    const size_t here = id % g.vol, e = extended(here, g);
    M3 acc = m3_zero();
    for (int plane = 0; plane < Nplanes; ++plane) {
        int mu, nu;
        plane_of(plane, mu, nu);
        if (rho != mu && rho != nu) continue;
        int d[Leaves][LeafLinks], f[Leaves][LeafLinks];
        leaves_of(mu, nu, d, f);
        const double* b = (plane < Ndim) ? b_lo : b_hi;
        for (int l = 0; l < Leaves; ++l)
            for (int k = 0; k < LeafLinks; ++k) {
                if (d[l][k] != rho) continue;
                const int cut = f[l][k] ? k : k + 1;
                size_t corner;
                const M3 w1 = walk(u, e, d[l] + cut, f[l] + cut, LeafLinks - cut, g, &corner);
                const M3 w2 = walk(u, corner, d[l], f[l], cut, g);
                const M3 term = mul(mul(w1, load(b, plane % Ndim, corner)), w2);
                acc = f[l][k] ? add(acc, term) : sub(acc, term);
            }
    }
    store(m, rho, here, acc);
}

} // namespace bridge_cuda

#endif
