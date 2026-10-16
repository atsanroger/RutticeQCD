#include "gauge/stout.cuh"
#include "bridge_cuda.h"

namespace {
    constexpr int BLOCK_SIZE = 128;

    bridge_cuda::StoutGeometry geometry(const int* dims, const double* rho)
    {
        bridge_cuda::StoutGeometry g;
        g.vol = 1;
        for (int mu = 0; mu < 4; ++mu) {
            g.dims[mu] = dims[mu];
            g.vol *= dims[mu];
            for (int nu = 0; nu < 4; ++nu) g.rho[mu][nu] = rho[4 * mu + nu];
        }
        return g;
    }

    inline int blocks_for(size_t n) { return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE); }
}

extern "C" {

void bridge_cuda_stout_step(double* d_v, const double* d_u, const int* dims, const double* rho, cudaStream_t stream)
{
    const auto g = geometry(dims, rho);
    bridge_cuda::stout_step_kernel<<<blocks_for(4 * static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(d_v, d_u, g);
}

void bridge_cuda_stout_chain(
    double* d_sigma, double* d_itheta, const double* d_sigma_prime, const double* d_u,
    const int* dims, const double* rho, cudaStream_t stream)
{
    const auto g = geometry(dims, rho);
    const int blocks = blocks_for(4 * static_cast<size_t>(g.vol));
    bridge_cuda::stout_chain_local_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_sigma, d_itheta, d_sigma_prime, d_u, g);
    bridge_cuda::stout_chain_staples_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_sigma, d_u, d_itheta, g);
}

} // extern "C"
