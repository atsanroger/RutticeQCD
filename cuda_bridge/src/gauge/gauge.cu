#include "../../include/bridge_cuda.h"
#include "../../include/gauge/gauge.cuh"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, 1024);
    }
}

extern "C" {

void bridge_cuda_gauge_plaquette_partial_f32(
    float* block_sums, const float* d_gauge, const BridgeCudaGeometry* geom,
    int grid_size, cudaStream_t stream
) {
    bridge_cuda::gauge_plaquette_partial_kernel<float, BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        block_sums, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3]
    );
}

void bridge_cuda_gauge_plaquette_partial_f64(
    double* block_sums, const double* d_gauge, const BridgeCudaGeometry* geom,
    int grid_size, cudaStream_t stream
) {
    bridge_cuda::gauge_plaquette_partial_kernel<double, BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        block_sums, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3]
    );
}

void bridge_cuda_gauge_ape_smear_f32(
    float* d_gauge_out, const float* d_gauge_in, const BridgeCudaGeometry* geom,
    const BridgeCudaSmearParams* params, cudaStream_t stream
) {
    int vol = geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    int grid = get_grid_size(vol);

    bridge_cuda::gauge_ape_smear_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_gauge_out, d_gauge_in,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(params->alpha_ape)
    );
}

void bridge_cuda_gauge_ape_smear_f64(
    double* d_gauge_out, const double* d_gauge_in, const BridgeCudaGeometry* geom,
    const BridgeCudaSmearParams* params, cudaStream_t stream
) {
    int vol = geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    int grid = get_grid_size(vol);

    bridge_cuda::gauge_ape_smear_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_gauge_out, d_gauge_in,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->alpha_ape
    );
}

} // extern "C"
