#ifndef BRIDGE_CUDA_H
#define BRIDGE_CUDA_H

#include "base/types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Wilson Operator
// =========================================================================

void bridge_cuda_wilson_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    int dagger, cudaStream_t stream
);

void bridge_cuda_wilson_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    int dagger, cudaStream_t stream
);

void bridge_cuda_wilson_pack_f32(
    float* d_send_bufs[8], const float* d_in,
    const BridgeCudaGeometry* geom, cudaStream_t stream
);

void bridge_cuda_wilson_pack_f64(
    double* d_send_bufs[8], const double* d_in,
    const BridgeCudaGeometry* geom, cudaStream_t stream
);

void bridge_cuda_wilson_exterior_f32(
    float* d_out, const float* const d_recv_bufs[8], const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream
);

void bridge_cuda_wilson_exterior_f64(
    double* d_out, const double* const d_recv_bufs[8], const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream
);

// =========================================================================
// Clover Operator
// =========================================================================

// out = T in: d_clover holds the blocks of T, or of its inverse (fermion/clover.cuh); on the
// whole lattice (parity < 0) or on the sites of one parity.
void bridge_cuda_clover_term_f32(
    float* d_out, const float* d_in, const float* d_clover,
    const BridgeCudaGeometry* geom, int parity, cudaStream_t stream
);

// out = D in = T in - kappa hops; csw of the parameters is in T already.
void bridge_cuda_clover_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int dagger, cudaStream_t stream
);

// out = T in: d_clover holds the blocks of T, or of its inverse (fermion/clover.cuh); on the
// whole lattice (parity < 0) or on the sites of one parity.
void bridge_cuda_clover_term_f64(
    double* d_out, const double* d_in, const double* d_clover,
    const BridgeCudaGeometry* geom, int parity, cudaStream_t stream
);

// out = D in = T in - kappa hops; csw of the parameters is in T already.
void bridge_cuda_clover_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int dagger, cudaStream_t stream
);

// =========================================================================
// Mobius Domain Wall Fermion (MDWF)
// =========================================================================

void bridge_cuda_mdwf_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream
);

void bridge_cuda_mdwf_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream
);

void bridge_cuda_mdwf_dslash_1pass_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream
);

void bridge_cuda_mdwf_dslash_1pass_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream
);

// =========================================================================
// Domain Decomposition (SAP / Block DD)
// =========================================================================

void bridge_cuda_wilson_dd_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream
);

void bridge_cuda_wilson_dd_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream
);

void bridge_cuda_mdwf_dd_5dir_f32(
    float* d_out, const float* d_in,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream
);

void bridge_cuda_mdwf_dd_5dir_f64(
    double* d_out, const double* d_in,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    const BridgeCudaDdParams* dd_params,
    cudaStream_t stream
);

// =========================================================================
// Multigrid (MG) Coarse Operator & Transfers
// =========================================================================

void bridge_cuda_mg_coarse_dslash_f32(
    float* d_out, const float* d_in, const float* d_clover_inv, const float* d_coarse_links,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

void bridge_cuda_mg_coarse_dslash_f64(
    double* d_out, const double* d_in, const double* d_clover_inv, const double* d_coarse_links,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

void bridge_cuda_mg_restrict_f32(
    float* d_coarse, const float* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

void bridge_cuda_mg_restrict_f64(
    double* d_coarse, const double* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

void bridge_cuda_mg_prolong_f32(
    float* d_fine, const float* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

void bridge_cuda_mg_prolong_f64(
    double* d_fine, const double* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

void bridge_cuda_mg_block_normalize_f32(
    float* d_basis, int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

// -------------------------------------------------------------------------
// Unified Multigrid Transfer Operators (Unifies Clover Ls=1 and MDWF Ls>1)
// -------------------------------------------------------------------------
void bridge_cuda_mg_restrict_unified_f32(
    float* d_coarse, const float* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mg_restrict_unified_f64(
    double* d_coarse, const double* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mg_prolong_unified_f32(
    float* d_fine, const float* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mg_prolong_unified_f64(
    double* d_fine, const double* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mg_block_normalize_unified_f32(
    float* d_basis, int num_vectors,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

// -------------------------------------------------------------------------
// MDWF 5D Multigrid Transfer Operators
// -------------------------------------------------------------------------
void bridge_cuda_mdwf_mg_restrict_5d_f32(
    float* d_coarse, const float* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mdwf_mg_restrict_5d_f64(
    double* d_coarse, const double* d_fine, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mdwf_mg_prolong_5d_f32(
    float* d_fine, const float* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mdwf_mg_prolong_5d_f64(
    double* d_fine, const double* d_coarse, const float* d_basis,
    int num_vectors, const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    int target_parity, cudaStream_t stream
);

void bridge_cuda_mdwf_mg_block_normalize_5d_f32(
    float* d_basis, int num_vectors,
    const BridgeCudaGeometry* geom, const BridgeCudaMgParams* params,
    cudaStream_t stream
);

// -------------------------------------------------------------------------
// Coarse Grid Even-Odd Operators & Schur Complement Pass
// -------------------------------------------------------------------------
void bridge_cuda_mg_coarse_pass_f32(
    float* d_out, const float* d_base, const float* d_diag_in, const float* d_hop_in,
    const float* d_diag, const float* d_diag_inv, const float* d_links,
    int nx_c, int ny_c, int nz_c, int nt_c, int dim, int target_parity,
    bool apply_diag, bool apply_hops, bool minus_hops, bool apply_inv,
    cudaStream_t stream
);

void bridge_cuda_mg_coarse_pass_f64(
    double* d_out, const double* d_base, const double* d_diag_in, const double* d_hop_in,
    const double* d_diag, const double* d_diag_inv, const double* d_links,
    int nx_c, int ny_c, int nz_c, int nt_c, int dim, int target_parity,
    bool apply_diag, bool apply_hops, bool minus_hops, bool apply_inv,
    cudaStream_t stream
);

void bridge_cuda_mg_coarse_gamma5_f32(
    float* d_vec, int total_sites, int dim, cudaStream_t stream
);

void bridge_cuda_mg_coarse_gamma5_f64(
    double* d_vec, int total_sites, int dim, cudaStream_t stream
);

void bridge_cuda_mg_coarse_invert_diagonal_blocks_f32(
    float* d_diag_inv, const float* d_diag, int vol_c, int dim, cudaStream_t stream
);

void bridge_cuda_mg_coarse_invert_diagonal_blocks_f64(
    double* d_diag_inv, const double* d_diag, int vol_c, int dim, cudaStream_t stream
);

// The parts of the MDWF operator (fermion/mdwf.cuh): the plain operator, Dirac representation.

// out = H (b - c Y) in, or with H^dag (fermion/hops.cuh): on the whole lattice
// (target_parity < 0), or out on the sites of parity target_parity and in on the others. The
// hop between the parities of the Wilson and clover operators (Ls = 1, b = 1, c = 0) and of
// the domain-wall operator (kappa = 1/2, its b and c).
void bridge_cuda_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, cudaStream_t stream
);

// The field on one parity out of the one on the whole lattice (to_half), or into it.
void bridge_cuda_parity_copy_f32(
    float* d_half, float* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    cudaStream_t stream
);

// out = (a1 + y1 Y) x + (a2 + y2 Y) z, or with Y^dag; d_z may be null.
void bridge_cuda_mdwf_fifth_f32(
    float* d_out, const float* d_x, double a1, double y1,
    const float* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream
);

// out = C^-1 in, or C^dag^-1 in, on `sites` sites with Ls spinors each.
void bridge_cuda_mdwf_site_inv_f32(
    float* d_out, const float* d_in,
    int sites, int Ls, const BridgeCudaMdwfParams* params,
    int dagger, cudaStream_t stream
);

// out = gamma_5 in, with `reflect` out_s = gamma_5 in_(Ls-1-s); Ls = 1 for a 4D field.
void bridge_cuda_gamma5_f32(
    float* d_out, const float* d_in, int sites, int Ls, int reflect, cudaStream_t stream
);

// out = H (b - c Y) in, or with H^dag (fermion/hops.cuh): on the whole lattice
// (target_parity < 0), or out on the sites of parity target_parity and in on the others. The
// hop between the parities of the Wilson and clover operators (Ls = 1, b = 1, c = 0) and of
// the domain-wall operator (kappa = 1/2, its b and c).
void bridge_cuda_hop_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, cudaStream_t stream
);

// The field on one parity out of the one on the whole lattice (to_half), or into it.
void bridge_cuda_parity_copy_f64(
    double* d_half, double* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    cudaStream_t stream
);

// out = (a1 + y1 Y) x + (a2 + y2 Y) z, or with Y^dag; d_z may be null.
void bridge_cuda_mdwf_fifth_f64(
    double* d_out, const double* d_x, double a1, double y1,
    const double* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream
);

// out = C^-1 in, or C^dag^-1 in, on `sites` sites with Ls spinors each.
void bridge_cuda_mdwf_site_inv_f64(
    double* d_out, const double* d_in,
    int sites, int Ls, const BridgeCudaMdwfParams* params,
    int dagger, cudaStream_t stream
);

// out = gamma_5 in, with `reflect` out_s = gamma_5 in_(Ls-1-s); Ls = 1 for a 4D field.
void bridge_cuda_gamma5_f64(
    double* d_out, const double* d_in, int sites, int Ls, int reflect, cudaStream_t stream
);

// =========================================================================
// Multi-RHS (MRHS)
// =========================================================================

void bridge_cuda_mrhs_wilson_dslash_f32(
    float* const* d_out_ptrs, const float* const* d_in_ptrs, const float* d_gauge,
    int num_rhs, const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream
);

void bridge_cuda_mrhs_wilson_dslash_f64(
    double* const* d_out_ptrs, const double* const* d_in_ptrs, const double* d_gauge,
    int num_rhs, const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    cudaStream_t stream
);

// =========================================================================
// Gauge Actions & Smearing
// =========================================================================

void bridge_cuda_gauge_plaquette_partial_f32(
    float* block_sums, const float* d_gauge, const BridgeCudaGeometry* geom,
    int grid_size, cudaStream_t stream
);

void bridge_cuda_gauge_plaquette_partial_f64(
    double* block_sums, const double* d_gauge, const BridgeCudaGeometry* geom,
    int grid_size, cudaStream_t stream
);

void bridge_cuda_gauge_ape_smear_f32(
    float* d_gauge_out, const float* d_gauge_in, const BridgeCudaGeometry* geom,
    const BridgeCudaSmearParams* params, cudaStream_t stream
);

void bridge_cuda_gauge_ape_smear_f64(
    double* d_gauge_out, const double* d_gauge_in, const BridgeCudaGeometry* geom,
    const BridgeCudaSmearParams* params, cudaStream_t stream
);

// =========================================================================
// Linear Algebra (BLAS, Cast, Basis Conversion)
// =========================================================================

void bridge_cuda_blas_axpy_f32(float* y, const float* x, float a, int n, cudaStream_t stream);
void bridge_cuda_blas_axpy_f64(double* y, const double* x, double a, int n, cudaStream_t stream);

// Complex y += a x and y = a y + x on fields of n reals (blocks of 32 real parts, then 32
// imaginary parts), |x|^2 and <x|y> = sum conj(x) y. The reductions sum the CUDA blocks on the
// host in a fixed order, so they are deterministic; they wait for the device.
void bridge_cuda_blas_caxpy_f32(float* y, const float* x, float ar, float ai, int n, cudaStream_t stream);
void bridge_cuda_blas_caypx_f32(float* y, const float* x, float ar, float ai, int n, cudaStream_t stream);
double bridge_cuda_blas_norm2_f32(const float* x, int n, cudaStream_t stream);
void bridge_cuda_blas_dotc_f32(double* re, double* im, const float* x, const float* y, int n, cudaStream_t stream);
void bridge_cuda_blas_caxpy_f64(double* y, const double* x, double ar, double ai, int n, cudaStream_t stream);
void bridge_cuda_blas_caypx_f64(double* y, const double* x, double ar, double ai, int n, cudaStream_t stream);
double bridge_cuda_blas_norm2_f64(const double* x, int n, cudaStream_t stream);
void bridge_cuda_blas_dotc_f64(double* re, double* im, const double* x, const double* y, int n, cudaStream_t stream);

void bridge_cuda_blas_scal_f32(float* x, float a, int n, cudaStream_t stream);
void bridge_cuda_blas_scal_f64(double* x, double a, int n, cudaStream_t stream);

void bridge_cuda_blas_norm2_partial_f32(float* block_sums, const float* x, int n, int grid_size, cudaStream_t stream);
void bridge_cuda_blas_norm2_partial_f64(double* block_sums, const double* x, int n, int grid_size, cudaStream_t stream);

void bridge_cuda_cast_f64_to_f32(float* dst, const double* src, int total_elements, cudaStream_t stream);
void bridge_cuda_cast_f32_to_f64(double* dst, const float* src, int total_elements, cudaStream_t stream);

void bridge_cuda_convert_dirac_chiral_f32(float* d_out, const float* d_in, int total_sites, cudaStream_t stream);
void bridge_cuda_convert_dirac_chiral_f64(double* d_out, const double* d_in, int total_sites, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_CUDA_H
