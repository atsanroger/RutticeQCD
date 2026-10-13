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

void bridge_cuda_coarse_pass_f32(
    float* d_out, const float* d_base, const float* d_diag_of, const float* d_hops_of,
    const float* d_diag, const float* d_links, const float* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors, cudaStream_t stream)
{
    dim3 grid(n_coarse, vectors);
    bridge_cuda::coarse_pass_kernel<float><<<grid, dim, 2 * dim * sizeof(float), stream>>>(
        d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse, d_neighbours, d_odd,
        n_coarse, dim, parity, minus ? -1 : 1, gamma5);
}

void bridge_cuda_coarse_gamma5_f32(float* d_v, int n_coarse, int dim, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_gamma5_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(d_v, n_coarse, dim);
}

void bridge_cuda_coarse_pass_f64(
    double* d_out, const double* d_base, const double* d_diag_of, const double* d_hops_of,
    const double* d_diag, const double* d_links, const double* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors, cudaStream_t stream)
{
    dim3 grid(n_coarse, vectors);
    bridge_cuda::coarse_pass_kernel<double><<<grid, dim, 2 * dim * sizeof(double), stream>>>(
        d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse, d_neighbours, d_odd,
        n_coarse, dim, parity, minus ? -1 : 1, gamma5);
}

void bridge_cuda_coarse_gamma5_f64(double* d_v, int n_coarse, int dim, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_gamma5_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_v, n_coarse, dim);
}

} // extern "C"
