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

} // extern "C"
