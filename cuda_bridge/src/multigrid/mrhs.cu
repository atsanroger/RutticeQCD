#include "../../include/bridge_cuda.h"
#include "../../include/multigrid/mrhs.cuh"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, 1024);
    }
}

extern "C" {

void bridge_cuda_mrhs_wilson_dslash_f32(
    float* const* d_out_ptrs, const float* const* d_in_ptrs, const float* d_gauge,
    int num_rhs, const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream
) {
    int vol = geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    int grid = get_grid_size(vol);

    bridge_cuda::mrhs_wilson_dslash_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out_ptrs, d_in_ptrs, d_gauge, num_rhs,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(params->kappa)
    );
}

void bridge_cuda_mrhs_wilson_dslash_f64(
    double* const* d_out_ptrs, const double* const* d_in_ptrs, const double* d_gauge,
    int num_rhs, const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream
) {
    int vol = geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    int grid = get_grid_size(vol);

    bridge_cuda::mrhs_wilson_dslash_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out_ptrs, d_in_ptrs, d_gauge, num_rhs,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->kappa
    );
}

} // extern "C"
