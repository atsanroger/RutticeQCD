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
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::wilson_interior_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        static_cast<float>(params->kappa)
    );
}

void bridge_cuda_wilson_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::wilson_interior_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        params->kappa
    );
}

void bridge_cuda_wilson_pack_f32(
    float* d_send_bufs[8], const float* d_in,
    const BridgeCudaGeometry* geom, cudaStream_t stream)
{
    for (int dim = 0; dim < 4; ++dim) {
        for (int dir = 0; dir < 2; ++dir) {
            int buf_idx = dim * 2 + dir;
            int face_vol = geom->vol / geom->dims[dim];
            int grid = get_grid_size(face_vol);
            bridge_cuda::wilson_pack_face_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
                d_send_bufs[buf_idx], d_in, dim, dir,
                geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3]
            );
        }
    }
}

void bridge_cuda_wilson_pack_f64(
    double* d_send_bufs[8], const double* d_in,
    const BridgeCudaGeometry* geom, cudaStream_t stream)
{
    for (int dim = 0; dim < 4; ++dim) {
        for (int dir = 0; dir < 2; ++dir) {
            int buf_idx = dim * 2 + dir;
            int face_vol = geom->vol / geom->dims[dim];
            int grid = get_grid_size(face_vol);
            bridge_cuda::wilson_pack_face_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
                d_send_bufs[buf_idx], d_in, dim, dir,
                geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3]
            );
        }
    }
}

void bridge_cuda_wilson_exterior_f32(
    float* d_out, const float* const d_recv_bufs[8], const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream)
{
    (void)d_out; (void)d_recv_bufs; (void)d_gauge; (void)geom; (void)params; (void)stream;
}

void bridge_cuda_wilson_exterior_f64(
    double* d_out, const double* const d_recv_bufs[8], const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream)
{
    (void)d_out; (void)d_recv_bufs; (void)d_gauge; (void)geom; (void)params; (void)stream;
}

} // extern "C"
