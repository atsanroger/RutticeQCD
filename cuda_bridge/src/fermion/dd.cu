#include "fermion/dd.cuh"
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

void bridge_cuda_wilson_dd_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::wilson_dd_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        dd_params->block_size[0], dd_params->block_size[1], dd_params->block_size[2], dd_params->block_size[3],
        dd_params->ieo,
        static_cast<float>(params->kappa)
    );
}

void bridge_cuda_wilson_dd_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    bridge_cuda::wilson_dd_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        dd_params->block_size[0], dd_params->block_size[1], dd_params->block_size[2], dd_params->block_size[3],
        dd_params->ieo,
        params->kappa
    );
}

void bridge_cuda_mdwf_dd_5dir_f32(
    float* d_out, const float* d_in,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream)
{
    int vol5d = geom->vol * geom->Ls;
    int grid = get_grid_size(vol5d);

    if (geom->gamma_rep == BRIDGE_CUDA_GAMMA_DIRAC) {
        bridge_cuda::mdwf_dd_5dir_kernel<float, BRIDGE_CUDA_GAMMA_DIRAC><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_in,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            dd_params->block_size[0], dd_params->block_size[1], dd_params->block_size[2], dd_params->block_size[3],
            geom->Ls, dd_params->ieo,
            static_cast<float>(params->M5),
            static_cast<float>(params->mass),
            static_cast<float>(params->alpha),
            static_cast<float>(params->b5),
            static_cast<float>(params->c5)
        );
    } else {
        bridge_cuda::mdwf_dd_5dir_kernel<float, BRIDGE_CUDA_GAMMA_CHIRAL><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_in,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            dd_params->block_size[0], dd_params->block_size[1], dd_params->block_size[2], dd_params->block_size[3],
            geom->Ls, dd_params->ieo,
            static_cast<float>(params->M5),
            static_cast<float>(params->mass),
            static_cast<float>(params->alpha),
            static_cast<float>(params->b5),
            static_cast<float>(params->c5)
        );
    }
}

void bridge_cuda_mdwf_dd_5dir_f64(
    double* d_out, const double* d_in,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream)
{
    int vol5d = geom->vol * geom->Ls;
    int grid = get_grid_size(vol5d);

    if (geom->gamma_rep == BRIDGE_CUDA_GAMMA_DIRAC) {
        bridge_cuda::mdwf_dd_5dir_kernel<double, BRIDGE_CUDA_GAMMA_DIRAC><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_in,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            dd_params->block_size[0], dd_params->block_size[1], dd_params->block_size[2], dd_params->block_size[3],
            geom->Ls, dd_params->ieo,
            params->M5,
            params->mass,
            params->alpha,
            params->b5,
            params->c5
        );
    } else {
        bridge_cuda::mdwf_dd_5dir_kernel<double, BRIDGE_CUDA_GAMMA_CHIRAL><<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out, d_in,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            dd_params->block_size[0], dd_params->block_size[1], dd_params->block_size[2], dd_params->block_size[3],
            geom->Ls, dd_params->ieo,
            params->M5,
            params->mass,
            params->alpha,
            params->b5,
            params->c5
        );
    }
}

} // extern "C"
