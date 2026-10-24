#include "multigrid/sap.cuh"
#include "bridge_cuda.h"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = bridge_cuda::DefaultBlockSize;

    // A grid for a grid-stride loop over n elements, capped (base/aosoa.cuh, GridStride).
    inline int get_grid_size(size_t n) {
        return static_cast<int>(std::min<size_t>((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024));
    }

    template <typename Float>
    void keep_colour(Float* d_v, const unsigned* d_blocks, const unsigned* d_block_of, int sites,
                     int vectors, int c, cudaStream_t stream)
    {
        const size_t n = static_cast<size_t>(sites) * vectors;
        bridge_cuda::sap_keep_colour_kernel<Float><<<get_grid_size(n), BLOCK_SIZE, 0, stream>>>(
            d_v, d_blocks, d_block_of, sites, vectors, c);
    }

    template <typename Float>
    void alpha(double* d_alpha, const Float* d_p, const Float* d_r, const unsigned* d_blocks,
               const unsigned* d_sites, int n_blocks, int block_volume, int sites, int vectors,
               int c, cudaStream_t stream)
    {
        const dim3 grid(static_cast<unsigned>(n_blocks), static_cast<unsigned>(vectors));
        bridge_cuda::sap_alpha_kernel<Float><<<grid, bridge_cuda::SapThreads, 0, stream>>>(
            d_alpha, d_p, d_r, d_blocks, d_sites, block_volume, sites, vectors, c);
    }

    template <typename Float>
    void update(Float* d_e, Float* d_r, const Float* d_p, const double* d_alpha,
                const unsigned* d_block_of, int sites, int vectors, cudaStream_t stream)
    {
        const size_t n = static_cast<size_t>(sites) * vectors;
        bridge_cuda::sap_update_kernel<Float><<<get_grid_size(n), BLOCK_SIZE, 0, stream>>>(
            d_e, d_r, d_p, d_alpha, d_block_of, sites, vectors);
    }
}

extern "C" {

void bridge_cuda_sap_keep_colour_f32(float* d_v, const unsigned* d_blocks, const unsigned* d_block_of,
                                     int sites, int vectors, int c, cudaStream_t stream)
{
    keep_colour<float>(d_v, d_blocks, d_block_of, sites, vectors, c, stream);
}

void bridge_cuda_sap_keep_colour_f64(double* d_v, const unsigned* d_blocks, const unsigned* d_block_of,
                                     int sites, int vectors, int c, cudaStream_t stream)
{
    keep_colour<double>(d_v, d_blocks, d_block_of, sites, vectors, c, stream);
}

void bridge_cuda_sap_alpha_f32(double* d_alpha, const float* d_p, const float* d_r, const unsigned* d_blocks,
                               const unsigned* d_sites, int n_blocks, int block_volume, int sites,
                               int vectors, int c, cudaStream_t stream)
{
    alpha<float>(d_alpha, d_p, d_r, d_blocks, d_sites, n_blocks, block_volume, sites, vectors, c, stream);
}

void bridge_cuda_sap_alpha_f64(double* d_alpha, const double* d_p, const double* d_r, const unsigned* d_blocks,
                               const unsigned* d_sites, int n_blocks, int block_volume, int sites,
                               int vectors, int c, cudaStream_t stream)
{
    alpha<double>(d_alpha, d_p, d_r, d_blocks, d_sites, n_blocks, block_volume, sites, vectors, c, stream);
}

void bridge_cuda_sap_update_f32(float* d_e, float* d_r, const float* d_p, const double* d_alpha,
                                const unsigned* d_block_of, int sites, int vectors, cudaStream_t stream)
{
    update<float>(d_e, d_r, d_p, d_alpha, d_block_of, sites, vectors, stream);
}

void bridge_cuda_sap_update_f64(double* d_e, double* d_r, const double* d_p, const double* d_alpha,
                                const unsigned* d_block_of, int sites, int vectors, cudaStream_t stream)
{
    update<double>(d_e, d_r, d_p, d_alpha, d_block_of, sites, vectors, stream);
}

} // extern "C"
