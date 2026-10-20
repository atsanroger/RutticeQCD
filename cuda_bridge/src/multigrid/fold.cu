#include "multigrid/fold.cuh"
#include "bridge_cuda.h"

namespace {
    constexpr int BLOCK_SIZE = 256;

    // The groups of vectors the kernels of P take.
    inline int groups(int vectors) {
        return (vectors + bridge_cuda::FoldVectors - 1) / bridge_cuda::FoldVectors;
    }

    // dims: the extents of the fine lattice; fold: col[4] and sign[4] of gamma_5, then whether P
    // is in half precision.
    inline bridge_cuda::FoldGeometry geometry(
        const int* dims, int Ls, int parity, const int* gamma, int leg = -1, const int* block = nullptr)
    {
        bridge_cuda::FoldGeometry g;
        g.leg = leg;
        for (int mu = 0; mu < 4; ++mu) g.block[mu] = (block != nullptr) ? block[mu] : 1;
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
    float* d_coarse, const float* d_fine, const void* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* fold,
    int leg, const int* block, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * 2 * n + BLOCK_SIZE - 1) / BLOCK_SIZE, groups(vectors));
    const auto g = geometry(dims, Ls, parity, fold, leg, block);
    if (fold[8]) {
        bridge_cuda::fold_restrict_kernel<float, __half><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_coarse, d_fine, static_cast<const __half*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g);
    } else {
        bridge_cuda::fold_restrict_kernel<float, float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_coarse, d_fine, static_cast<const float*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g);
    }
}

void bridge_cuda_fold_prolong_f32(
    float* d_fine, const float* d_coarse, const void* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* fold,
    int vectors, cudaStream_t stream)
{
    const size_t sites = static_cast<size_t>(parity >= 0 ? dims[0] / 2 : dims[0]) * dims[1] * dims[2] * dims[3];
    dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), groups(vectors));
    const auto g = geometry(dims, Ls, parity, fold);
    if (fold[8]) {
        bridge_cuda::fold_prolong_kernel<float, __half><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_fine, d_coarse, static_cast<const __half*>(d_p), d_coarse_of, n_coarse, n, vectors, g);
    } else {
        bridge_cuda::fold_prolong_kernel<float, float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_fine, d_coarse, static_cast<const float*>(d_p), d_coarse_of, n_coarse, n, vectors, g);
    }
}

void bridge_cuda_fold_fill_f32(
    void* d_p, const float* d_fine, int n, int first, const int* dims, int Ls, const int* fold,
    int vectors, cudaStream_t stream)
{
    const size_t sites = static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3];
    dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), vectors);
    const auto g = geometry(dims, Ls, -1, fold);
    if (fold[8]) {
        bridge_cuda::fold_fill_kernel<float, __half><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<__half*>(d_p), d_fine, n, first, g);
    } else {
        bridge_cuda::fold_fill_kernel<float, float><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<float*>(d_p), d_fine, n, first, g);
    }
}

void bridge_cuda_fold_restrict_f64(
    double* d_coarse, const double* d_fine, const void* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* fold,
    int leg, const int* block, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * 2 * n + BLOCK_SIZE - 1) / BLOCK_SIZE, groups(vectors));
    const auto g = geometry(dims, Ls, parity, fold, leg, block);
    if (fold[8]) {
        bridge_cuda::fold_restrict_kernel<double, __half><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_coarse, d_fine, static_cast<const __half*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g);
    } else {
        bridge_cuda::fold_restrict_kernel<double, float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_coarse, d_fine, static_cast<const float*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g);
    }
}

void bridge_cuda_fold_prolong_f64(
    double* d_fine, const double* d_coarse, const void* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* fold,
    int vectors, cudaStream_t stream)
{
    const size_t sites = static_cast<size_t>(parity >= 0 ? dims[0] / 2 : dims[0]) * dims[1] * dims[2] * dims[3];
    dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), groups(vectors));
    const auto g = geometry(dims, Ls, parity, fold);
    if (fold[8]) {
        bridge_cuda::fold_prolong_kernel<double, __half><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_fine, d_coarse, static_cast<const __half*>(d_p), d_coarse_of, n_coarse, n, vectors, g);
    } else {
        bridge_cuda::fold_prolong_kernel<double, float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_fine, d_coarse, static_cast<const float*>(d_p), d_coarse_of, n_coarse, n, vectors, g);
    }
}

void bridge_cuda_fold_fill_f64(
    void* d_p, const double* d_fine, int n, int first, const int* dims, int Ls, const int* fold,
    int vectors, cudaStream_t stream)
{
    const size_t sites = static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3];
    dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), vectors);
    const auto g = geometry(dims, Ls, -1, fold);
    if (fold[8]) {
        bridge_cuda::fold_fill_kernel<double, __half><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<__half*>(d_p), d_fine, n, first, g);
    } else {
        bridge_cuda::fold_fill_kernel<double, float><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<float*>(d_p), d_fine, n, first, g);
    }
}

void bridge_cuda_fold_orthonormalise(
    void* d_p, int half, double* d_w, const unsigned* d_sites, unsigned* d_fail,
    int first_site, int count, int block_volume, int n, int kept, double scale,
    cudaStream_t stream)
{
    if (half) {
        bridge_cuda::fold_orthonormalise_kernel<__half><<<2 * count, bridge_cuda::FoldThreads, 0, stream>>>(
            static_cast<__half*>(d_p), d_w, d_sites, d_fail, first_site, block_volume, n, kept, scale);
    } else {
        bridge_cuda::fold_orthonormalise_kernel<float><<<2 * count, bridge_cuda::FoldThreads, 0, stream>>>(
            static_cast<float*>(d_p), d_w, d_sites, d_fail, first_site, block_volume, n, kept, scale);
    }
}

void bridge_cuda_coarse_pass_f32(
    float* d_out, const float* d_base, const float* d_diag_of, const float* d_hops_of,
    const float* d_diag, const float* d_links, const float* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors, cudaStream_t stream)
{
    dim3 grid(n_coarse, vectors);
    const int threads = bridge_cuda::WarpSize * bridge_cuda::coarse_pass_warps(dim);
    bridge_cuda::coarse_pass_kernel<float><<<grid, threads, 2 * dim * sizeof(float), stream>>>(
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
    const int threads = bridge_cuda::WarpSize * bridge_cuda::coarse_pass_warps(dim);
    bridge_cuda::coarse_pass_kernel<double><<<grid, threads, 2 * dim * sizeof(double), stream>>>(
        d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse, d_neighbours, d_odd,
        n_coarse, dim, parity, minus ? -1 : 1, gamma5);
}

void bridge_cuda_coarse_gamma5_f64(double* d_v, int n_coarse, int dim, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_gamma5_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_v, n_coarse, dim);
}

void bridge_cuda_blocks_inverse(
    double* d_out, const double* d_a, double* d_scratch, unsigned* d_fail,
    int n_sites, int dim, cudaStream_t stream)
{
    bridge_cuda::blocks_inverse_kernel<<<n_sites, dim, 0, stream>>>(d_out, d_a, d_scratch, d_fail, dim);
}

void bridge_cuda_blocks_abs_inverse_gamma5(
    double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim, cudaStream_t stream)
{
    bridge_cuda::blocks_abs_inverse_gamma5_kernel<<<n_sites, dim, 0, stream>>>(d_out, d_a, d_scratch, dim, 1, 0.0);
}

void bridge_cuda_blocks_abs_inverse(
    double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim, double cutoff,
    cudaStream_t stream)
{
    bridge_cuda::blocks_abs_inverse_gamma5_kernel<<<n_sites, dim, 0, stream>>>(d_out, d_a, d_scratch, dim, 0, cutoff);
}

} // extern "C"
