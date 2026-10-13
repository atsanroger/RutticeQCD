#include "multigrid/coarse.cuh"
#include "multigrid/coarse_eo.cuh"
#include "multigrid/transfer.cuh"
#include "multigrid/mdwf_transfer.cuh"
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

void bridge_cuda_mg_coarse_dslash_f32(
    float* d_out, const float* d_in, const float* d_clover_inv, const float* d_coarse_links,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    int nx_c = geom->dims[0] / params->block_size[0];
    int ny_c = geom->dims[1] / params->block_size[1];
    int nz_c = geom->dims[2] / params->block_size[2];
    int nt_c = geom->dims[3] / params->block_size[3];
    int vol_c = nx_c * ny_c * nz_c * nt_c;
    int grid = get_grid_size(vol_c);

    bridge_cuda::mg_coarse_dslash_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_clover_inv, d_coarse_links,
        nx_c, ny_c, nz_c, nt_c, params->num_colors
    );
}

void bridge_cuda_mg_coarse_dslash_f64(
    double* d_out, const double* d_in, const double* d_clover_inv, const double* d_coarse_links,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    int nx_c = geom->dims[0] / params->block_size[0];
    int ny_c = geom->dims[1] / params->block_size[1];
    int nz_c = geom->dims[2] / params->block_size[2];
    int nt_c = geom->dims[3] / params->block_size[3];
    int vol_c = nx_c * ny_c * nz_c * nt_c;
    int grid = get_grid_size(vol_c);

    bridge_cuda::mg_coarse_dslash_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, d_clover_inv, d_coarse_links,
        nx_c, ny_c, nz_c, nt_c, params->num_colors
    );
}

void bridge_cuda_mg_restrict_f32(
    float* d_coarse, const float* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mg_restrict_unified_f32(
        d_coarse, d_fine, d_basis, num_vectors, geom, params, -1, stream);
}

void bridge_cuda_mg_restrict_f64(
    double* d_coarse, const double* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mg_restrict_unified_f64(
        d_coarse, d_fine, d_basis, num_vectors, geom, params, -1, stream);
}

void bridge_cuda_mg_prolong_f32(
    float* d_fine, const float* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mg_prolong_unified_f32(
        d_fine, d_coarse, d_basis, num_vectors, geom, params, -1, stream);
}

void bridge_cuda_mg_prolong_f64(
    double* d_fine, const double* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mg_prolong_unified_f64(
        d_fine, d_coarse, d_basis, num_vectors, geom, params, -1, stream);
}

void bridge_cuda_mg_block_normalize_f32(
    float* d_basis, int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mg_block_normalize_unified_f32(
        d_basis, num_vectors, geom, params, stream);
}


void bridge_cuda_mg_restrict_unified_f32(
    float* d_coarse, const float* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    int total_blocks = (geom->dims[0] / params->block_size[0]) *
                       (geom->dims[1] / params->block_size[1]) *
                       (geom->dims[2] / params->block_size[2]) *
                       (geom->dims[3] / params->block_size[3]);
    int grid = get_grid_size(total_blocks);
    int ls = (geom->Ls > 0) ? geom->Ls : 1;

    bridge_cuda::mg_restrict_unified_kernel<float, float, float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_coarse, d_fine, d_basis, num_vectors,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->block_size[0], params->block_size[1], params->block_size[2], params->block_size[3],
        ls, target_parity
    );
}

void bridge_cuda_mg_restrict_unified_f64(
    double* d_coarse, const double* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    int total_blocks = (geom->dims[0] / params->block_size[0]) *
                       (geom->dims[1] / params->block_size[1]) *
                       (geom->dims[2] / params->block_size[2]) *
                       (geom->dims[3] / params->block_size[3]);
    int grid = get_grid_size(total_blocks);
    int ls = (geom->Ls > 0) ? geom->Ls : 1;

    bridge_cuda::mg_restrict_unified_kernel<double, float, double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_coarse, d_fine, d_basis, num_vectors,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->block_size[0], params->block_size[1], params->block_size[2], params->block_size[3],
        ls, target_parity
    );
}

void bridge_cuda_mg_prolong_unified_f32(
    float* d_fine, const float* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    int ls = (geom->Ls > 0) ? geom->Ls : 1;

    bridge_cuda::mg_prolong_unified_kernel<float, float, float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_fine, d_coarse, d_basis, num_vectors,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->block_size[0], params->block_size[1], params->block_size[2], params->block_size[3],
        ls, target_parity
    );
}

void bridge_cuda_mg_prolong_unified_f64(
    double* d_fine, const double* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    int grid = get_grid_size(geom->vol);
    int ls = (geom->Ls > 0) ? geom->Ls : 1;

    bridge_cuda::mg_prolong_unified_kernel<double, float, double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_fine, d_coarse, d_basis, num_vectors,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->block_size[0], params->block_size[1], params->block_size[2], params->block_size[3],
        ls, target_parity
    );
}

void bridge_cuda_mg_block_normalize_unified_f32(
    float* d_basis, int num_vectors,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    int total_blocks = (geom->dims[0] / params->block_size[0]) *
                       (geom->dims[1] / params->block_size[1]) *
                       (geom->dims[2] / params->block_size[2]) *
                       (geom->dims[3] / params->block_size[3]);
    int grid = get_grid_size(total_blocks);
    int ls = (geom->Ls > 0) ? geom->Ls : 1;

    bridge_cuda::mg_block_normalize_unified_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_basis, num_vectors,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        params->block_size[0], params->block_size[1], params->block_size[2], params->block_size[3],
        ls
    );
}

void bridge_cuda_mdwf_mg_restrict_5d_f32(
    float* d_coarse, const float* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    bridge_cuda_mg_restrict_unified_f32(
        d_coarse, d_fine, d_basis, num_vectors, geom, params, target_parity, stream);
}

void bridge_cuda_mdwf_mg_restrict_5d_f64(
    double* d_coarse, const double* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    bridge_cuda_mg_restrict_unified_f64(
        d_coarse, d_fine, d_basis, num_vectors, geom, params, target_parity, stream);
}

void bridge_cuda_mdwf_mg_prolong_5d_f32(
    float* d_fine, const float* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    bridge_cuda_mg_prolong_unified_f32(
        d_fine, d_coarse, d_basis, num_vectors, geom, params, target_parity, stream);
}

void bridge_cuda_mdwf_mg_prolong_5d_f64(
    double* d_fine, const double* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream)
{
    bridge_cuda_mg_prolong_unified_f64(
        d_fine, d_coarse, d_basis, num_vectors, geom, params, target_parity, stream);
}

void bridge_cuda_mdwf_mg_block_normalize_5d_f32(
    float* d_basis, int num_vectors,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mg_block_normalize_unified_f32(
        d_basis, num_vectors, geom, params, stream);
}


void bridge_cuda_mg_coarse_pass_f32(
    float* d_out, const float* d_base, const float* d_diag_in, const float* d_hop_in,
    const float* d_diag, const float* d_diag_inv, const float* d_links,
    int nx_c, int ny_c, int nz_c, int nt_c, int dim, int target_parity,
    bool apply_diag, bool apply_hops, bool minus_hops, bool apply_inv,
    cudaStream_t stream)
{
    int vol_c = nx_c * ny_c * nz_c * nt_c;
    int grid = get_grid_size(vol_c);

    bridge_cuda::mg_coarse_pass_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_base, d_diag_in, d_hop_in,
        d_diag, d_diag_inv, d_links,
        nx_c, ny_c, nz_c, nt_c, dim, target_parity,
        apply_diag, apply_hops, minus_hops, apply_inv
    );
}

void bridge_cuda_mg_coarse_pass_f64(
    double* d_out, const double* d_base, const double* d_diag_in, const double* d_hop_in,
    const double* d_diag, const double* d_diag_inv, const double* d_links,
    int nx_c, int ny_c, int nz_c, int nt_c, int dim, int target_parity,
    bool apply_diag, bool apply_hops, bool minus_hops, bool apply_inv,
    cudaStream_t stream)
{
    int vol_c = nx_c * ny_c * nz_c * nt_c;
    int grid = get_grid_size(vol_c);

    bridge_cuda::mg_coarse_pass_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out, d_base, d_diag_in, d_hop_in,
        d_diag, d_diag_inv, d_links,
        nx_c, ny_c, nz_c, nt_c, dim, target_parity,
        apply_diag, apply_hops, minus_hops, apply_inv
    );
}

void bridge_cuda_mg_coarse_gamma5_f32(
    float* d_vec, int total_sites, int dim, cudaStream_t stream)
{
    int grid = get_grid_size(total_sites);
    bridge_cuda::mg_coarse_gamma5_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_vec, total_sites, dim
    );
}

void bridge_cuda_mg_coarse_gamma5_f64(
    double* d_vec, int total_sites, int dim, cudaStream_t stream)
{
    int grid = get_grid_size(total_sites);
    bridge_cuda::mg_coarse_gamma5_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_vec, total_sites, dim
    );
}

void bridge_cuda_mg_coarse_invert_diagonal_blocks_f32(
    float* d_diag_inv, const float* d_diag, int vol_c, int dim, cudaStream_t stream)
{
    int grid = get_grid_size(vol_c);
    bridge_cuda::mg_coarse_invert_diagonal_blocks_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_diag_inv, d_diag, vol_c, dim
    );
}

void bridge_cuda_mg_coarse_invert_diagonal_blocks_f64(
    double* d_diag_inv, const double* d_diag, int vol_c, int dim, cudaStream_t stream)
{
    int grid = get_grid_size(vol_c);
    bridge_cuda::mg_coarse_invert_diagonal_blocks_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
        d_diag_inv, d_diag, vol_c, dim
    );
}

} // extern "C"
