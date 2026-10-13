#include "fermion/mdwf.cuh"
#include "fermion/mdwf_eo.cuh"
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

void bridge_cuda_mdwf_dslash_1pass_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    int vol4d = geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    constexpr int LS_PER_BLOCK = 4;
    dim3 block(bridge_cuda::WarpSize, LS_PER_BLOCK);
    dim3 grid((vol4d + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize,
              (geom->Ls + LS_PER_BLOCK - 1) / LS_PER_BLOCK);

    // The Dirac representation and alpha = 1 only (see the kernel).
    bridge_cuda::mdwf_dslash_1pass_kernel<float, LS_PER_BLOCK, BRIDGE_CUDA_GAMMA_DIRAC><<<grid, block, 0, stream>>>(
        d_out, d_in, d_gauge, vol4d, geom->Ls,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        static_cast<float>(params->M5),
        static_cast<float>(params->mass),
        static_cast<float>(params->b5),
        static_cast<float>(params->c5)
    );
}

void bridge_cuda_mdwf_dslash_1pass_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    int vol4d = geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    constexpr int LS_PER_BLOCK = 4;
    dim3 block(bridge_cuda::WarpSize, LS_PER_BLOCK);
    dim3 grid((vol4d + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize,
              (geom->Ls + LS_PER_BLOCK - 1) / LS_PER_BLOCK);

    // The Dirac representation and alpha = 1 only (see the kernel).
    bridge_cuda::mdwf_dslash_1pass_kernel<double, LS_PER_BLOCK, BRIDGE_CUDA_GAMMA_DIRAC><<<grid, block, 0, stream>>>(
        d_out, d_in, d_gauge, vol4d, geom->Ls,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        params->M5,
        params->mass,
        params->b5,
        params->c5
    );
}

void bridge_cuda_mdwf_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mdwf_dslash_1pass_f32(d_out, d_in, d_gauge, geom, params, stream);
}

void bridge_cuda_mdwf_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mdwf_dslash_1pass_f64(d_out, d_in, d_gauge, geom, params, stream);
}

void bridge_cuda_mdwf_eo_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int target_parity, cudaStream_t stream)
{
    int half = (geom->dims[0] / 2) * geom->dims[1] * geom->dims[2] * geom->dims[3];
    constexpr int LS_PER_BLOCK = 4;
    dim3 block(bridge_cuda::WarpSize, LS_PER_BLOCK);
    dim3 grid((half + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize,
              (geom->Ls + LS_PER_BLOCK - 1) / LS_PER_BLOCK);

    bridge_cuda::mdwf_eo_hop_kernel<float, LS_PER_BLOCK><<<grid, block, 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls, target_parity,
        static_cast<float>(params->mass),
        static_cast<float>(params->b5),
        static_cast<float>(params->c5)
    );
}

void bridge_cuda_mdwf_site_inv_f32(
    float* d_out, const float* d_in,
    int sites, int Ls, const BridgeCudaMdwfParams* params,
    int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_site_inv_kernel<float><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls,
        static_cast<float>(params->M5),
        static_cast<float>(params->mass),
        static_cast<float>(params->b5),
        static_cast<float>(params->c5),
        dagger
    );
}

void bridge_cuda_mdwf_eo_hop_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int target_parity, cudaStream_t stream)
{
    int half = (geom->dims[0] / 2) * geom->dims[1] * geom->dims[2] * geom->dims[3];
    constexpr int LS_PER_BLOCK = 4;
    dim3 block(bridge_cuda::WarpSize, LS_PER_BLOCK);
    dim3 grid((half + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize,
              (geom->Ls + LS_PER_BLOCK - 1) / LS_PER_BLOCK);

    bridge_cuda::mdwf_eo_hop_kernel<double, LS_PER_BLOCK><<<grid, block, 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls, target_parity,
        static_cast<double>(params->mass),
        static_cast<double>(params->b5),
        static_cast<double>(params->c5)
    );
}

void bridge_cuda_mdwf_site_inv_f64(
    double* d_out, const double* d_in,
    int sites, int Ls, const BridgeCudaMdwfParams* params,
    int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_site_inv_kernel<double><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls,
        static_cast<double>(params->M5),
        static_cast<double>(params->mass),
        static_cast<double>(params->b5),
        static_cast<double>(params->c5),
        dagger
    );
}

} // extern "C"
