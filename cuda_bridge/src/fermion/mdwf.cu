#include "fermion/mdwf.cuh"
#include "bridge_cuda.h"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;
    constexpr int LS_PER_BLOCK = 4;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, 1024);
    }

    inline int volume(const BridgeCudaGeometry* geom) {
        return geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    }

    // One thread for each (site, s) of `sites` sites.
    inline dim3 site_block() { return dim3(bridge_cuda::WarpSize, LS_PER_BLOCK); }
    inline dim3 site_grid(int sites, int Ls) {
        return dim3((sites + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize,
                    (Ls + LS_PER_BLOCK - 1) / LS_PER_BLOCK);
    }
}

extern "C" {

void bridge_cuda_mdwf_dslash_1pass_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda::mdwf_dslash_1pass_kernel<float, LS_PER_BLOCK><<<site_grid(volume(geom), geom->Ls), site_block(), 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls,
        static_cast<float>(params->M5), static_cast<float>(params->mass), static_cast<float>(params->b5), static_cast<float>(params->c5)
    );
}

void bridge_cuda_mdwf_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mdwf_dslash_1pass_f32(d_out, d_in, d_gauge, geom, params, stream);
}

void bridge_cuda_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, cudaStream_t stream)
{
    int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
    bridge_cuda::hop_kernel<float, LS_PER_BLOCK><<<site_grid(sites, geom->Ls), site_block(), 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls, target_parity,
        static_cast<float>(dagger ? -1 : 1), static_cast<float>(kappa),
        static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)
    );
}

void bridge_cuda_mdwf_fifth_f32(
    float* d_out, const float* d_x, double a1, double y1,
    const float* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_fifth_kernel<float><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_x, d_z, sites, Ls, static_cast<float>(mass),
        static_cast<float>(a1), static_cast<float>(y1), static_cast<float>(a2), static_cast<float>(y2), dagger
    );
}

void bridge_cuda_mdwf_site_inv_f32(
    float* d_out, const float* d_in,
    int sites, int Ls, const BridgeCudaMdwfParams* params,
    int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_site_inv_kernel<float><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls,
        static_cast<float>(params->M5), static_cast<float>(params->mass), static_cast<float>(params->b5), static_cast<float>(params->c5),
        dagger
    );
}

void bridge_cuda_parity_copy_f32(
    float* d_half, float* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    cudaStream_t stream)
{
    bridge_cuda::parity_copy_kernel<float><<<get_grid_size(volume(geom) / 2), BLOCK_SIZE, 0, stream>>>(
        d_half, d_full,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        geom->Ls, parity, to_half
    );
}

void bridge_cuda_gamma5_f32(
    float* d_out, const float* d_in, int sites, int Ls, int reflect, cudaStream_t stream)
{
    bridge_cuda::gamma5_kernel<float><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls, reflect
    );
}

void bridge_cuda_mdwf_dslash_1pass_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda::mdwf_dslash_1pass_kernel<double, LS_PER_BLOCK><<<site_grid(volume(geom), geom->Ls), site_block(), 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls,
        static_cast<double>(params->M5), static_cast<double>(params->mass), static_cast<double>(params->b5), static_cast<double>(params->c5)
    );
}

void bridge_cuda_mdwf_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mdwf_dslash_1pass_f64(d_out, d_in, d_gauge, geom, params, stream);
}

void bridge_cuda_hop_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, cudaStream_t stream)
{
    int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
    bridge_cuda::hop_kernel<double, LS_PER_BLOCK><<<site_grid(sites, geom->Ls), site_block(), 0, stream>>>(
        d_out, d_in, d_gauge,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls, target_parity,
        static_cast<double>(dagger ? -1 : 1), static_cast<double>(kappa),
        static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)
    );
}

void bridge_cuda_mdwf_fifth_f64(
    double* d_out, const double* d_x, double a1, double y1,
    const double* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_fifth_kernel<double><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_x, d_z, sites, Ls, static_cast<double>(mass),
        static_cast<double>(a1), static_cast<double>(y1), static_cast<double>(a2), static_cast<double>(y2), dagger
    );
}

void bridge_cuda_mdwf_site_inv_f64(
    double* d_out, const double* d_in,
    int sites, int Ls, const BridgeCudaMdwfParams* params,
    int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_site_inv_kernel<double><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls,
        static_cast<double>(params->M5), static_cast<double>(params->mass), static_cast<double>(params->b5), static_cast<double>(params->c5),
        dagger
    );
}

void bridge_cuda_parity_copy_f64(
    double* d_half, double* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    cudaStream_t stream)
{
    bridge_cuda::parity_copy_kernel<double><<<get_grid_size(volume(geom) / 2), BLOCK_SIZE, 0, stream>>>(
        d_half, d_full,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        geom->Ls, parity, to_half
    );
}

void bridge_cuda_gamma5_f64(
    double* d_out, const double* d_in, int sites, int Ls, int reflect, cudaStream_t stream)
{
    bridge_cuda::gamma5_kernel<double><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls, reflect
    );
}

} // extern "C"
