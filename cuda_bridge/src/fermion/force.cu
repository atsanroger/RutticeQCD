#include "fermion/force.cuh"
#include "bridge_cuda.h"

namespace {
    bridge_cuda::Halo<double> halo_of(const BridgeCudaHalo* h) {
        bridge_cuda::Halo<double> halo{};
        for (int f = 0; f < 8; ++f) {
            halo.recv[f] = static_cast<const double*>(h->recv[f]);
            halo.bc[f] = h->bc[f];
        }
        for (int mu = 0; mu < 4; ++mu) halo.cut[mu] = h->cut[mu];
        return halo;
    }
}

extern "C" {

void bridge_cuda_hop_force(
    double* d_m, const double* d_zeta, const double* d_eta, const double* d_g5zeta, const double* d_g5eta,
    const double* d_gauge, int link_rows, const int* dims, const int* bc, int Ls, double coef,
    const BridgeCudaHalo* h_eta, const BridgeCudaHalo* h_g5zeta, cudaStream_t stream)
{
    const int sites = dims[0] * dims[1] * dims[2] * dims[3];
    constexpr int threads = 64;
    const int blocks = (sites + threads - 1) / threads;
    if (h_eta == nullptr) {
        bridge_cuda::hop_force_kernel<false><<<blocks, threads, 0, stream>>>(
            d_m, d_zeta, d_eta, d_g5zeta, d_g5eta, d_gauge, link_rows,
            dims[0], dims[1], dims[2], dims[3], bc[0], bc[1], bc[2], bc[3], Ls, coef);
    } else {
        bridge_cuda::hop_force_kernel<true><<<blocks, threads, 0, stream>>>(
            d_m, d_zeta, d_eta, d_g5zeta, d_g5eta, d_gauge, link_rows,
            dims[0], dims[1], dims[2], dims[3], bc[0], bc[1], bc[2], bc[3], Ls, coef,
            halo_of(h_eta), halo_of(h_g5zeta));
    }
}

} // extern "C"
