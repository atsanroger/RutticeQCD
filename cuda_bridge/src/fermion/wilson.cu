#include "fermion/wilson.cuh"
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

void bridge_cuda_wilson_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    int dagger, int vectors, cudaStream_t stream)
{
    bridge_cuda::wilson_interior_kernel<float><<<bridge_cuda::site_grid(geom->vol, 1, vectors), bridge_cuda::site_block(1, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        static_cast<float>(params->kappa),
        dagger ? -1 : 1, vectors
    );
}

void bridge_cuda_wilson_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    int dagger, int vectors, cudaStream_t stream)
{
    bridge_cuda::wilson_interior_kernel<double><<<bridge_cuda::site_grid(geom->vol, 1, vectors), bridge_cuda::site_block(1, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        params->kappa,
        dagger ? -1 : 1, vectors
    );
}

} // extern "C"
