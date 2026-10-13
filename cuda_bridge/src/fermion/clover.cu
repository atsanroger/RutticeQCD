#include "fermion/clover.cuh"
#include "bridge_cuda.h"

extern "C" {

void bridge_cuda_clover_term_f32(
    float* d_out, const float* d_in, const float* d_clover,
    const BridgeCudaGeometry* geom, int parity, int vectors, cudaStream_t stream)
{
    int sites = (parity >= 0) ? geom->vol / 2 : geom->vol;
    bridge_cuda::clover_term_kernel<float><<<bridge_cuda::site_grid(sites, 1, vectors), bridge_cuda::site_block(1, vectors), 0, stream>>>(
        d_out, d_in, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], parity, vectors
    );
}

void bridge_cuda_clover_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int dagger, int vectors, cudaStream_t stream)
{
    bridge_cuda::clover_dslash_kernel<float><<<bridge_cuda::site_grid(geom->vol, 1, vectors), bridge_cuda::site_block(1, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        static_cast<float>(params->kappa),
        dagger ? -1 : 1, vectors
    );
}

void bridge_cuda_clover_term_f64(
    double* d_out, const double* d_in, const double* d_clover,
    const BridgeCudaGeometry* geom, int parity, int vectors, cudaStream_t stream)
{
    int sites = (parity >= 0) ? geom->vol / 2 : geom->vol;
    bridge_cuda::clover_term_kernel<double><<<bridge_cuda::site_grid(sites, 1, vectors), bridge_cuda::site_block(1, vectors), 0, stream>>>(
        d_out, d_in, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], parity, vectors
    );
}

void bridge_cuda_clover_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int dagger, int vectors, cudaStream_t stream)
{
    bridge_cuda::clover_dslash_kernel<double><<<bridge_cuda::site_grid(geom->vol, 1, vectors), bridge_cuda::site_block(1, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows, d_clover,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        static_cast<double>(params->kappa),
        dagger ? -1 : 1, vectors
    );
}

} // extern "C"
