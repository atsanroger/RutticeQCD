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
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::clover_term_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(params->csw)
    );
}

void bridge_cuda_clover_term_f64(
    double* d_out, const double* d_in, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::clover_term_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->csw
    );
}

void bridge_cuda_clover_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::clover_dslash_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(params->kappa),
        static_cast<float>(params->csw)
    );
}

void bridge_cuda_clover_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::clover_dslash_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->kappa,
        params->csw
    );
}

void bridge_cuda_clover_eo_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int target_parity, cudaStream_t stream)
{
    int vol_eo = geom->vol / 2;
    int grid = get_grid_size(vol_eo);
    bridge_cuda::clover_eo_dslash_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        target_parity,
        static_cast<float>(params->kappa),
        static_cast<float>(params->csw)
    );
}

void bridge_cuda_clover_eo_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int target_parity, cudaStream_t stream)
{
    int vol_eo = geom->vol / 2;
    int grid = get_grid_size(vol_eo);
    bridge_cuda::clover_eo_dslash_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        target_parity,
        params->kappa,
        params->csw
    );
}

} // extern "C"
