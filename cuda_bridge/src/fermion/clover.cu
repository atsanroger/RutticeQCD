#include "fermion/clover.cuh"
#include "bridge_cuda.h"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, 1024);
    }
}

extern "C" {

void bridge_cuda_clover_term_f32(
    float* d_out, const float* d_in, const float* d_clover,
    const BridgeCudaGeometry* geom, cudaStream_t stream)
{
    bridge_cuda::clover_term_kernel<float><<<get_grid_size(geom->vol), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_clover, geom->vol
    );
}

void bridge_cuda_clover_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    cudaStream_t stream)
{
    bridge_cuda::clover_dslash_kernel<float><<<get_grid_size(geom->vol), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        static_cast<float>(params->kappa)
    );
}

void bridge_cuda_clover_term_f64(
    double* d_out, const double* d_in, const double* d_clover,
    const BridgeCudaGeometry* geom, cudaStream_t stream)
{
    bridge_cuda::clover_term_kernel<double><<<get_grid_size(geom->vol), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_clover, geom->vol
    );
}

void bridge_cuda_clover_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    cudaStream_t stream)
{
    bridge_cuda::clover_dslash_kernel<double><<<get_grid_size(geom->vol), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        static_cast<double>(params->kappa)
    );
}

} // extern "C"
