#include "fermion/force.cuh"
#include "bridge_cuda.h"

extern "C" {

void bridge_cuda_hop_force(
    double* d_m, const double* d_zeta, const double* d_eta, const double* d_g5zeta, const double* d_g5eta,
    const double* d_gauge, int link_rows, const int* dims, const int* bc, int Ls, double coef,
    cudaStream_t stream)
{
    const int sites = dims[0] * dims[1] * dims[2] * dims[3];
    constexpr int threads = 64;
    bridge_cuda::hop_force_kernel<<<(sites + threads - 1) / threads, threads, 0, stream>>>(
        d_m, d_zeta, d_eta, d_g5zeta, d_g5eta, d_gauge, link_rows,
        dims[0], dims[1], dims[2], dims[3], bc[0], bc[1], bc[2], bc[3], Ls, coef);
}

} // extern "C"
