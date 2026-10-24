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

    // coarse = P^dag fine: fold_restrict_tiled_kernel, a block for each coarse site with about
    // 128 threads (dim of them for each of its sites at a time), or fold_restrict_kernel where
    // its shared memory would not fit.
    template <typename Float, typename Stored>
    void restrict_launch(Float* d_coarse, const Float* d_fine, const Stored* d_p, const unsigned* d_sites,
                         int n_coarse, int block_volume, int n, int vectors,
                         const bridge_cuda::FoldGeometry& g, cudaStream_t stream)
    {
        const int dim = 2 * n, kept = 12 * g.Ls;
        const int sites_at_once = dim < 128 ? 128 / dim : 1;
        const int threads = sites_at_once * dim;
        const int per_site = vectors < bridge_cuda::FoldVectors ? vectors : bridge_cuda::FoldVectors;
        const size_t shared =
            bridge_cuda::fold_restrict_shared<Float, Stored>(sites_at_once, dim, kept, per_site, block_volume);
        static int most = -1;
        static size_t allowed = 48 * 1024;
        if (most < 0) {
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        }
        if (threads > 1024 || shared > static_cast<size_t>(most)) {
            dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, groups(vectors));
            bridge_cuda::fold_restrict_kernel<Float, Stored><<<grid, BLOCK_SIZE, 0, stream>>>(
                d_coarse, d_fine, d_p, d_sites, n_coarse, block_volume, n, vectors, g);
            return;
        }
        if (shared > allowed) {
            cudaFuncSetAttribute(bridge_cuda::fold_restrict_tiled_kernel<Float, Stored>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
            allowed = shared;
        }
        dim3 grid(n_coarse, groups(vectors));
        bridge_cuda::fold_restrict_tiled_kernel<Float, Stored><<<grid, threads, shared, stream>>>(
            d_coarse, d_fine, d_p, d_sites, n_coarse, block_volume, n, vectors, per_site, g);
    }

    // One pass of the coarse operator (coarse_pass_kernel, with or without the hops), its shared
    // memory allowed above the default 48 kB where dim asks for it.
    template <typename Float, bool Hops>
    void coarse_pass_launch_as(Float* d_out, const Float* d_base, const Float* d_diag_of, const Float* d_hops_of,
                               const Float* d_diag, const Float* d_links, const Float* d_inverse,
                               const unsigned* d_neighbours, const unsigned* d_odd,
                               int n_coarse, int dim, int parity, int minus, int gamma5, int vectors,
                               const Float* d_ghost, int n_ghost, cudaStream_t stream)
    {
        const size_t shared = bridge_cuda::coarse_pass_shared<Float>(dim, Hops);
        static size_t allowed = 48 * 1024;
        if (shared > allowed) {
            cudaFuncSetAttribute(bridge_cuda::coarse_pass_kernel<Float, Hops>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
            allowed = shared;
        }
        dim3 grid(n_coarse, vectors);
        const int threads = bridge_cuda::WarpSize * bridge_cuda::coarse_pass_warps(dim);
        bridge_cuda::coarse_pass_kernel<Float, Hops><<<grid, threads, shared, stream>>>(
            d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse, d_neighbours, d_odd,
            n_coarse, dim, parity, minus ? -1 : 1, gamma5, d_ghost, n_ghost);
    }

    // |H|^-1 of the blocks: the kernel of the rounds of disjoint pairs with H and V in shared
    // memory, or H alone, where that fits; else the one of the pairs one at a time.
    template <bool VShared>
    void abs_inverse_rounds(double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim,
                            int gamma5, double cutoff, size_t shared, cudaStream_t stream)
    {
        static size_t allowed = 48 * 1024;
        if (shared > allowed) {
            cudaFuncSetAttribute(bridge_cuda::blocks_abs_inverse_gamma5_rounds_kernel<VShared>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
            allowed = shared;
        }
        bridge_cuda::blocks_abs_inverse_gamma5_rounds_kernel<VShared><<<n_sites, dim, shared, stream>>>(
            d_out, d_a, d_scratch, dim, gamma5, cutoff);
    }

    void abs_inverse_launch(double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim,
                            int gamma5, double cutoff, cudaStream_t stream)
    {
        static int most = -1;
        if (most < 0) {
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        }
        const size_t both = bridge_cuda::abs_inverse_shared(dim, true);
        const size_t h_only = bridge_cuda::abs_inverse_shared(dim, false);
        if (dim <= 1024 && both <= static_cast<size_t>(most)) {
            abs_inverse_rounds<true>(d_out, d_a, d_scratch, n_sites, dim, gamma5, cutoff, both, stream);
        } else if (dim <= 1024 && h_only <= static_cast<size_t>(most)) {
            abs_inverse_rounds<false>(d_out, d_a, d_scratch, n_sites, dim, gamma5, cutoff, h_only, stream);
        } else {
            bridge_cuda::blocks_abs_inverse_gamma5_kernel<<<n_sites, dim, 0, stream>>>(d_out, d_a, d_scratch, dim, gamma5, cutoff);
        }
    }

    template <typename Float>
    void coarse_pass_launch(Float* d_out, const Float* d_base, const Float* d_diag_of, const Float* d_hops_of,
                            const Float* d_diag, const Float* d_links, const Float* d_inverse,
                            const unsigned* d_neighbours, const unsigned* d_odd,
                            int n_coarse, int dim, int parity, int minus, int gamma5, int vectors,
                            cudaStream_t stream, const Float* d_ghost = nullptr, int n_ghost = 0)
    {
        if (d_hops_of != nullptr) {
            coarse_pass_launch_as<Float, true>(d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse,
                                               d_neighbours, d_odd, n_coarse, dim, parity, minus, gamma5, vectors,
                                               d_ghost, n_ghost, stream);
        } else {
            coarse_pass_launch_as<Float, false>(d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse,
                                                d_neighbours, d_odd, n_coarse, dim, parity, minus, gamma5, vectors,
                                                d_ghost, n_ghost, stream);
        }
    }

    // fine = P coarse: fold_prolong_tiled_kernel for one vector where its entries split evenly
    // and its tile fits the default shared memory, else fold_prolong_kernel.
    template <typename Float, typename Stored>
    void prolong_launch(Float* d_fine, const Float* d_coarse, const Stored* d_p, const unsigned* d_coarse_of,
                        int n_coarse, int n, int vectors, const bridge_cuda::FoldGeometry& g, cudaStream_t stream)
    {
        const size_t sites = static_cast<size_t>(g.parity >= 0 ? g.nx / 2 : g.nx) * g.ny * g.nz * g.nt;
        const int Ls = g.Ls;
        const size_t tile = static_cast<size_t>(24) * Ls * bridge_cuda::WarpSize * sizeof(Float);
        if (vectors == 1 && (6 * Ls) % bridge_cuda::FoldPerThread == 0 && tile <= 48 * 1024) {
            const int threads = bridge_cuda::WarpSize * (6 * Ls / bridge_cuda::FoldPerThread);
            const unsigned blocks = static_cast<unsigned>((sites + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize);
            bridge_cuda::fold_prolong_tiled_kernel<Float, Stored><<<blocks, threads, tile, stream>>>(
                d_fine, d_coarse, d_p, d_coarse_of, n_coarse, n, g);
            return;
        }
        dim3 grid(static_cast<unsigned>((sites * 6 * Ls + BLOCK_SIZE - 1) / BLOCK_SIZE), groups(vectors));
        bridge_cuda::fold_prolong_kernel<Float, Stored><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_fine, d_coarse, d_p, d_coarse_of, n_coarse, n, vectors, g);
    }
}

extern "C" {

void bridge_cuda_fold_restrict_f32(
    float* d_coarse, const float* d_fine, const void* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* fold,
    int leg, const int* block, int vectors, cudaStream_t stream)
{
    const auto g = geometry(dims, Ls, parity, fold, leg, block);
    if (fold[8]) {
        restrict_launch(d_coarse, d_fine, static_cast<const __half*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g, stream);
    } else {
        restrict_launch(d_coarse, d_fine, static_cast<const float*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g, stream);
    }
}

void bridge_cuda_fold_prolong_f32(
    float* d_fine, const float* d_coarse, const void* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* fold,
    int vectors, cudaStream_t stream)
{
    const auto g = geometry(dims, Ls, parity, fold);
    if (fold[8]) {
        prolong_launch(d_fine, d_coarse, static_cast<const __half*>(d_p), d_coarse_of, n_coarse, n, vectors, g, stream);
    } else {
        prolong_launch(d_fine, d_coarse, static_cast<const float*>(d_p), d_coarse_of, n_coarse, n, vectors, g, stream);
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
    const auto g = geometry(dims, Ls, parity, fold, leg, block);
    if (fold[8]) {
        restrict_launch(d_coarse, d_fine, static_cast<const __half*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g, stream);
    } else {
        restrict_launch(d_coarse, d_fine, static_cast<const float*>(d_p), d_sites, n_coarse, block_volume, n, vectors, g, stream);
    }
}

void bridge_cuda_fold_prolong_f64(
    double* d_fine, const double* d_coarse, const void* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* fold,
    int vectors, cudaStream_t stream)
{
    const auto g = geometry(dims, Ls, parity, fold);
    if (fold[8]) {
        prolong_launch(d_fine, d_coarse, static_cast<const __half*>(d_p), d_coarse_of, n_coarse, n, vectors, g, stream);
    } else {
        prolong_launch(d_fine, d_coarse, static_cast<const float*>(d_p), d_coarse_of, n_coarse, n, vectors, g, stream);
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

void bridge_cuda_fold_orthonormalise_cholesky(
    void* d_p, int half, double* d_w, const unsigned* d_sites, unsigned* d_fail,
    int first_site, int count, int block_volume, int n, int kept, double scale,
    cudaStream_t stream)
{
    const size_t shared = bridge_cuda::fold_cholesky_shared(n);
    if (half) {
        static size_t allowed = 48 * 1024;
        if (shared > allowed) {
            cudaFuncSetAttribute(bridge_cuda::fold_orthonormalise_cholesky_kernel<__half>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
            allowed = shared;
        }
        bridge_cuda::fold_orthonormalise_cholesky_kernel<__half><<<2 * count, bridge_cuda::FoldThreads, shared, stream>>>(
            static_cast<__half*>(d_p), d_w, d_sites, d_fail, first_site, block_volume, n, kept, scale);
    } else {
        static size_t allowed = 48 * 1024;
        if (shared > allowed) {
            cudaFuncSetAttribute(bridge_cuda::fold_orthonormalise_cholesky_kernel<float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
            allowed = shared;
        }
        bridge_cuda::fold_orthonormalise_cholesky_kernel<float><<<2 * count, bridge_cuda::FoldThreads, shared, stream>>>(
            static_cast<float*>(d_p), d_w, d_sites, d_fail, first_site, block_volume, n, kept, scale);
    }
}

void bridge_cuda_coarse_pass_f32(
    float* d_out, const float* d_base, const float* d_diag_of, const float* d_hops_of,
    const float* d_diag, const float* d_links, const float* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors, cudaStream_t stream)
{
    coarse_pass_launch<float>(d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse,
                              d_neighbours, d_odd, n_coarse, dim, parity, minus, gamma5, vectors, stream);
}

void bridge_cuda_coarse_pass_ghost_f32(
    float* d_out, const float* d_base, const float* d_diag_of, const float* d_hops_of,
    const float* d_diag, const float* d_links, const float* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors,
    const float* d_ghost, int n_ghost, cudaStream_t stream)
{
    coarse_pass_launch<float>(d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse,
                              d_neighbours, d_odd, n_coarse, dim, parity, minus, gamma5, vectors, stream,
                              d_ghost, n_ghost);
}

void bridge_cuda_coarse_gather_f32(float* d_buf, const float* d_field, const unsigned* d_sites, int n, int n_coarse,
                                  int dim, int vectors, cudaStream_t stream)
{
    if (n == 0) return;
    dim3 grid((n * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_gather_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(d_buf, d_field, d_sites, n, n_coarse, dim);
}

void bridge_cuda_coarse_place_f32(float* d_ghost, const float* d_buf, int first, int n, int n_ghost, int dim, int vectors,
                                 cudaStream_t stream)
{
    if (n == 0) return;
    dim3 grid((n * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_place_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(d_ghost, d_buf, first, n, n_ghost, dim);
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
    coarse_pass_launch<double>(d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse,
                               d_neighbours, d_odd, n_coarse, dim, parity, minus, gamma5, vectors, stream);
}

void bridge_cuda_coarse_pass_ghost_f64(
    double* d_out, const double* d_base, const double* d_diag_of, const double* d_hops_of,
    const double* d_diag, const double* d_links, const double* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors,
    const double* d_ghost, int n_ghost, cudaStream_t stream)
{
    coarse_pass_launch<double>(d_out, d_base, d_diag_of, d_hops_of, d_diag, d_links, d_inverse,
                              d_neighbours, d_odd, n_coarse, dim, parity, minus, gamma5, vectors, stream,
                              d_ghost, n_ghost);
}

void bridge_cuda_coarse_gather_f64(double* d_buf, const double* d_field, const unsigned* d_sites, int n, int n_coarse,
                                  int dim, int vectors, cudaStream_t stream)
{
    if (n == 0) return;
    dim3 grid((n * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_gather_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_buf, d_field, d_sites, n, n_coarse, dim);
}

void bridge_cuda_coarse_place_f64(double* d_ghost, const double* d_buf, int first, int n, int n_ghost, int dim, int vectors,
                                 cudaStream_t stream)
{
    if (n == 0) return;
    dim3 grid((n * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_place_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_ghost, d_buf, first, n, n_ghost, dim);
}

void bridge_cuda_coarse_gamma5_f64(double* d_v, int n_coarse, int dim, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_gamma5_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_v, n_coarse, dim);
}

void bridge_cuda_coarse_units_f32(float* d_v, int n_coarse, int dim, int first, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_units_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(d_v, n_coarse, dim, first);
}

void bridge_cuda_coarse_units_f64(double* d_v, int n_coarse, int dim, int first, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    bridge_cuda::coarse_units_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_v, n_coarse, dim, first);
}

void bridge_cuda_coarse_columns_f32(void* d_target, int target_single, const float* d_c, int n_coarse, int dim, int first, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    if (target_single)
        bridge_cuda::coarse_columns_kernel<float, float><<<grid, BLOCK_SIZE, 0, stream>>>(static_cast<float*>(d_target), d_c, n_coarse, dim, first);
    else
        bridge_cuda::coarse_columns_kernel<float, double><<<grid, BLOCK_SIZE, 0, stream>>>(static_cast<double*>(d_target), d_c, n_coarse, dim, first);
}

void bridge_cuda_coarse_columns_f64(void* d_target, int target_single, const double* d_c, int n_coarse, int dim, int first, int vectors, cudaStream_t stream)
{
    dim3 grid((n_coarse * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, vectors);
    if (target_single)
        bridge_cuda::coarse_columns_kernel<double, float><<<grid, BLOCK_SIZE, 0, stream>>>(static_cast<float*>(d_target), d_c, n_coarse, dim, first);
    else
        bridge_cuda::coarse_columns_kernel<double, double><<<grid, BLOCK_SIZE, 0, stream>>>(static_cast<double*>(d_target), d_c, n_coarse, dim, first);
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
    abs_inverse_launch(d_out, d_a, d_scratch, n_sites, dim, 1, 0.0, stream);
}

void bridge_cuda_blocks_abs_inverse(
    double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim, double cutoff,
    cudaStream_t stream)
{
    abs_inverse_launch(d_out, d_a, d_scratch, n_sites, dim, 0, cutoff, stream);
}

} // extern "C"
