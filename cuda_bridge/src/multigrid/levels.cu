#include "multigrid/levels.cuh"
#include "bridge_cuda.h"

namespace {
    constexpr int BLOCK_SIZE = 256;

    inline int blocks_for(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }

    template <typename Float>
    void restrict_any(Float* d_out, const Float* d_v, const Float* d_q, const unsigned* d_members,
                      int n_blocks, int nq, int bs, int n_parent, int pd, int vectors, cudaStream_t stream)
    {
        dim3 grid(blocks_for(n_blocks * nq), vectors);
        bridge_cuda::level_restrict_kernel<Float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_v, d_q, d_members, n_blocks, nq, bs, n_parent, pd);
    }

    template <typename Float>
    void prolong_any(Float* d_out, const Float* d_y, const Float* d_q, const unsigned* d_owner,
                     int n_blocks, int nq, int bs, int n_parent, int pd, int add, int vectors,
                     cudaStream_t stream)
    {
        dim3 grid(blocks_for(n_parent * pd), vectors);
        bridge_cuda::level_prolong_kernel<Float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_y, d_q, d_owner, n_blocks, nq, bs, n_parent, pd, add);
    }

    template <typename Float>
    void basis_any(double* d_q, const Float* d_tests, const unsigned* d_members,
                   int n_blocks, int n_t, int bs, int n_parent, int pd, cudaStream_t stream)
    {
        bridge_cuda::level_basis_kernel<Float><<<2 * n_blocks, bridge_cuda::LevelThreads, 0, stream>>>(
            d_q, d_tests, d_members, n_t, bs, n_parent, pd);
    }

    template <typename Float>
    void sources_any(Float* d_e, const unsigned* d_colour, unsigned colour, int n_sites, int nq,
                     cudaStream_t stream)
    {
        dim3 grid(blocks_for(n_sites * nq), nq);
        bridge_cuda::level_sources_kernel<Float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_e, d_colour, colour, n_sites, nq);
    }

    template <typename Float>
    void probe_any(double* d_diag, double* d_links, const Float* d_r, const unsigned* d_colour,
                   const unsigned* d_neighbours, unsigned colour, unsigned used, int n_sites, int nq,
                   cudaStream_t stream)
    {
        dim3 grid(blocks_for(n_sites * nq), nq);
        bridge_cuda::level_probe_kernel<Float><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_diag, d_links, d_r, d_colour, d_neighbours, colour, used, n_sites, nq);
    }
}

extern "C" {

#define LEVEL_FUNCTIONS(SUFFIX, Float)                                                          \
void bridge_cuda_level_restrict_##SUFFIX(                                                        \
    Float* d_out, const Float* d_v, const Float* d_q, const unsigned* d_members,                \
    int n_blocks, int nq, int bs, int n_parent, int pd, int vectors, cudaStream_t stream)       \
{                                                                                               \
    restrict_any<Float>(d_out, d_v, d_q, d_members, n_blocks, nq, bs, n_parent, pd, vectors, stream); \
}                                                                                               \
void bridge_cuda_level_prolong_##SUFFIX(                                                         \
    Float* d_out, const Float* d_y, const Float* d_q, const unsigned* d_owner,                  \
    int n_blocks, int nq, int bs, int n_parent, int pd, int add, int vectors, cudaStream_t stream) \
{                                                                                               \
    prolong_any<Float>(d_out, d_y, d_q, d_owner, n_blocks, nq, bs, n_parent, pd, add, vectors, stream); \
}                                                                                               \
void bridge_cuda_level_basis_##SUFFIX(                                                           \
    double* d_q, const Float* d_tests, const unsigned* d_members,                               \
    int n_blocks, int n_t, int bs, int n_parent, int pd, cudaStream_t stream)                   \
{                                                                                               \
    basis_any<Float>(d_q, d_tests, d_members, n_blocks, n_t, bs, n_parent, pd, stream);         \
}                                                                                               \
void bridge_cuda_level_sources_##SUFFIX(                                                         \
    Float* d_e, const unsigned* d_colour, unsigned colour, int n_sites, int nq, cudaStream_t stream) \
{                                                                                               \
    sources_any<Float>(d_e, d_colour, colour, n_sites, nq, stream);                             \
}                                                                                               \
void bridge_cuda_level_probe_##SUFFIX(                                                           \
    double* d_diag, double* d_links, const Float* d_r, const unsigned* d_colour,                \
    const unsigned* d_neighbours, unsigned colour, unsigned used, int n_sites, int nq,          \
    cudaStream_t stream)                                                                        \
{                                                                                               \
    probe_any<Float>(d_diag, d_links, d_r, d_colour, d_neighbours, colour, used, n_sites, nq, stream); \
}

LEVEL_FUNCTIONS(f32, float)
LEVEL_FUNCTIONS(f64, double)

void bridge_cuda_dense_apply_f32(float* d_out, const float* d_r, const float* d_m, int n, int vectors, cudaStream_t stream)
{
    dim3 grid(blocks_for(n), vectors);
    bridge_cuda::dense_apply_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(d_out, d_r, d_m, n);
}

void bridge_cuda_dense_apply_f64(double* d_out, const double* d_r, const double* d_m, int n, int vectors, cudaStream_t stream)
{
    dim3 grid(blocks_for(n), vectors);
    bridge_cuda::dense_apply_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(d_out, d_r, d_m, n);
}

void bridge_cuda_level_dense(
    double* d_dense, const double* d_diag, const double* d_links, const unsigned* d_neighbours,
    unsigned used, int n_sites, int nq, cudaStream_t stream)
{
    bridge_cuda::level_dense_kernel<<<blocks_for(n_sites * nq), BLOCK_SIZE, 0, stream>>>(
        d_dense, d_diag, d_links, d_neighbours, used, n_sites, nq);
}

} // extern "C"
