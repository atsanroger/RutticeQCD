#ifndef BRIDGE_CUDA_FERMION_FORCE_CUH
#define BRIDGE_CUDA_FERMION_FORCE_CUH

#include "hops.cuh"

// The derivative of a fermion action by the links, for the hops of the Wilson kernel (as
// hmc/fermion_action.rs's kernel_derivative): for every link,
//
//   M_mu(x) = -2 c sum_s sum_spin [U_mu(x) (1 - gamma_mu) eta(x + mu, s)] zeta(x, s)^dag
//            + 2 c sum_s sum_spin [gamma5 eta(x, s)] [U_mu(x) (1 - gamma_mu) gamma5 zeta(x + mu, s)]^dag,
//
// a colour matrix (dS = Re tr X M for dU = X U), with eta the field the hops act on and the
// boundary signs of the hops. In double precision; M in the layout of gauge/stout.cuh's
// fields of matrices (real c of direction mu at site x at ((x / W) 72 + 18 mu + c) W + x % W).

namespace bridge_cuda {

__global__ void hop_force_kernel(
    double* __restrict__ d_m,
    const double* __restrict__ d_zeta, const double* __restrict__ d_eta,
    const double* __restrict__ d_g5zeta, const double* __restrict__ d_g5eta,
    const double* __restrict__ d_gauge, int link_rows,
    int nx, int ny, int nz, int nt,
    double bcx, double bcy, double bcz, double bct,
    int Ls, double coef)
{
    const int sites = nx * ny * nz * nt;
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    const int dims[4] = {nx, ny, nz, nt};
    const double bc[4] = {bcx, bcy, bcz, bct};
    int xc[4];
    site_to_coords(site, xc[0], xc[1], xc[2], xc[3], nx, ny, nz, nt);
    auto load = [&](double v[Nspin][Nc][Ncomplex], const double* f, int is) {
        #pragma unroll
        for (int s = 0; s < Nspin; ++s)
            #pragma unroll
            for (int c = 0; c < Nc; ++c)
                #pragma unroll
                for (int ri = 0; ri < Ncomplex; ++ri) v[s][c][ri] = f[aosoa_5d_spinor_idx(c, s, ri, is, Ls, site)];
    };
    for (int mu = 0; mu < Ndim; ++mu) {
        double m_r[Nc][Nc] = {{0}}, m_i[Nc][Nc] = {{0}};
        for (int is = 0; is < Ls; ++is) {
            // The forward hop of eta and of gamma5 zeta: U (1 - gamma_mu) w(x + mu), with the
            // boundary sign (kappa = -1 makes the coefficient of hops +1).
            double h1[Nspin][Nc][Ncomplex] = {{{0}}}, h2[Nspin][Nc][Ncomplex] = {{{0}}};
            hops<double>(h1, d_eta, d_gauge, link_rows, xc, is, Ls, dims, bc, false, 1.0, -1.0, 0.0, 1.0, 0.0, 1 << mu);
            hops<double>(h2, d_g5zeta, d_gauge, link_rows, xc, is, Ls, dims, bc, false, 1.0, -1.0, 0.0, 1.0, 0.0, 1 << mu);
            double z[Nspin][Nc][Ncomplex], e[Nspin][Nc][Ncomplex];
            load(z, d_zeta, is);
            load(e, d_g5eta, is);
            #pragma unroll
            for (int s = 0; s < Nspin; ++s)
                #pragma unroll
                for (int i = 0; i < Nc; ++i)
                    #pragma unroll
                    for (int j = 0; j < Nc; ++j) {
                        // a b^dag: a_i conj(b_j)
                        m_r[i][j] += -2.0 * coef * (h1[s][i][0] * z[s][j][0] + h1[s][i][1] * z[s][j][1])
                                   + 2.0 * coef * (e[s][i][0] * h2[s][j][0] + e[s][i][1] * h2[s][j][1]);
                        m_i[i][j] += -2.0 * coef * (h1[s][i][1] * z[s][j][0] - h1[s][i][0] * z[s][j][1])
                                   + 2.0 * coef * (e[s][i][1] * h2[s][j][0] - e[s][i][0] * h2[s][j][1]);
                    }
        }
        #pragma unroll
        for (int i = 0; i < Nc; ++i)
            #pragma unroll
            for (int j = 0; j < Nc; ++j) {
                const size_t c = 18 * mu + 2 * (3 * i + j);
                const size_t at = ((static_cast<size_t>(site) / W) * 72 + c) * W + site % W;
                d_m[at] = m_r[i][j];
                d_m[at + W] = m_i[i][j];
            }
    }
}

} // namespace bridge_cuda

#endif
