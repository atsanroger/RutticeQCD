#include "multigrid/fold.cuh"
#include "bridge_cuda.h"

namespace {
    constexpr int BLOCK_SIZE = 256;

    // dims: the extents of the fine lattice; gamma: col[4] and sign[4] of gamma_5.
    inline bridge_cuda::FoldGeometry geometry(const int* dims, int Ls, int parity, const int* gamma) {
        bridge_cuda::FoldGeometry g;
        g.nx = dims[0]; g.ny = dims[1]; g.nz = dims[2]; g.nt = dims[3];
        g.Ls = Ls;
        g.parity = parity;
        for (int a = 0; a < 4; ++a) {
            g.col[a] = gamma[a];
            g.sign[a] = gamma[4 + a];
        }
        return g;
    }
}

extern "C" {

void bridge_cuda_fold_restrict_f32(
    float* d_coarse, const float* d_fine, const float* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* gamma,
    int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * 2 * n + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::fold_restrict_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_coarse, d_fine, d_p, d_sites, n_coarse, block_volume, n, geometry(dims, Ls, parity, gamma));
}

void bridge_cuda_fold_prolong_f32(
    float* d_fine, const float* d_coarse, const float* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* gamma,
    int vectors, cudaStream_t stream)
{
    const size_t sites = static_cast<size_t>(parity >= 0 ? dims[0] / 2 : dims[0]) * dims[1] * dims[2] * dims[3];
    dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), vectors);
    bridge_cuda::fold_prolong_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_fine, d_coarse, d_p, d_coarse_of, n_coarse, n, geometry(dims, Ls, parity, gamma));
}

void bridge_cuda_fold_restrict_f64(
    double* d_coarse, const double* d_fine, const float* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* gamma,
    int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * 2 * n + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::fold_restrict_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_coarse, d_fine, d_p, d_sites, n_coarse, block_volume, n, geometry(dims, Ls, parity, gamma));
}

void bridge_cuda_fold_prolong_f64(
    double* d_fine, const double* d_coarse, const float* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* gamma,
    int vectors, cudaStream_t stream)
{
    const size_t sites = static_cast<size_t>(parity >= 0 ? dims[0] / 2 : dims[0]) * dims[1] * dims[2] * dims[3];
    dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), vectors);
    bridge_cuda::fold_prolong_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_fine, d_coarse, d_p, d_coarse_of, n_coarse, n, geometry(dims, Ls, parity, gamma));
}

} // extern "C"
