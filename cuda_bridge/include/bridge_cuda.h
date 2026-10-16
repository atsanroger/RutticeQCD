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
    int dagger, int vectors, cudaStream_t stream
);

void bridge_cuda_wilson_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaWilsonParams* params,
    int dagger, int vectors, cudaStream_t stream
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
    const BridgeCudaGeometry* geom, int parity, int vectors, cudaStream_t stream
);

// out = D in = T in - kappa hops; csw of the parameters is in T already.
void bridge_cuda_clover_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int dagger, int vectors, cudaStream_t stream
);

// out = T in: d_clover holds the blocks of T, or of its inverse (fermion/clover.cuh); on the
// whole lattice (parity < 0) or on the sites of one parity.
void bridge_cuda_clover_term_f64(
    double* d_out, const double* d_in, const double* d_clover,
    const BridgeCudaGeometry* geom, int parity, int vectors, cudaStream_t stream
);

// out = D in = T in - kappa hops; csw of the parameters is in T already.
void bridge_cuda_clover_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_clover,
    const BridgeCudaGeometry* geom, const BridgeCudaCloverParams* params,
    int dagger, int vectors, cudaStream_t stream
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
    int vectors, cudaStream_t stream
);

void bridge_cuda_mdwf_dslash_1pass_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, cudaStream_t stream
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
// the domain-wall operator (kappa = 1/2, its b and c). With d_from not null, d_from - that.
void bridge_cuda_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream
);

// out = M in, or d_from - M in if d_from is not null: the operators within a site of the
// domain-wall operator as a matrix product (fermion/fifth_gemm.cuh), M the padded
// (2 Ls) x (2 Ls) matrix on the device. In double precision on the tensor cores.
void bridge_cuda_fifth_gemm_f32(
    float* d_out, const float* d_in, const float* d_from, const float* d_m,
    int sites, int Ls, cudaStream_t stream
);

// The field on one parity out of the one on the whole lattice (to_half), or into it.
void bridge_cuda_parity_copy_f32(
    float* d_half, float* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    int vectors, cudaStream_t stream
);

// out = (a1 + y1 Y) x + (a2 + y2 Y) z, or with Y^dag; d_z may be null.
void bridge_cuda_mdwf_fifth_f32(
    float* d_out, const float* d_x, double a1, double y1,
    const float* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream
);

// out = gamma_5 in, with `reflect` out_s = gamma_5 in_(Ls-1-s); Ls = 1 for a 4D field.
void bridge_cuda_gamma5_f32(
    float* d_out, const float* d_in, int sites, int Ls, int reflect, cudaStream_t stream
);

// out = H (b - c Y) in, or with H^dag (fermion/hops.cuh): on the whole lattice
// (target_parity < 0), or out on the sites of parity target_parity and in on the others. The
// hop between the parities of the Wilson and clover operators (Ls = 1, b = 1, c = 0) and of
// the domain-wall operator (kappa = 1/2, its b and c). With d_from not null, d_from - that.
void bridge_cuda_hop_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream
);

// out = M in, or d_from - M in if d_from is not null: the operators within a site of the
// domain-wall operator as a matrix product (fermion/fifth_gemm.cuh), M the padded
// (2 Ls) x (2 Ls) matrix on the device. In double precision on the tensor cores.
void bridge_cuda_fifth_gemm_f64(
    double* d_out, const double* d_in, const double* d_from, const double* d_m,
    int sites, int Ls, cudaStream_t stream
);

// The field on one parity out of the one on the whole lattice (to_half), or into it.
void bridge_cuda_parity_copy_f64(
    double* d_half, double* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    int vectors, cudaStream_t stream
);

// out = (a1 + y1 Y) x + (a2 + y2 Y) z, or with Y^dag; d_z may be null.
void bridge_cuda_mdwf_fifth_f64(
    double* d_out, const double* d_x, double a1, double y1,
    const double* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream
);

// out = gamma_5 in, with `reflect` out_s = gamma_5 in_(Ls-1-s); Ls = 1 for a 4D field.
void bridge_cuda_gamma5_f64(
    double* d_out, const double* d_in, int sites, int Ls, int reflect, cudaStream_t stream
);

// The prolongation of the multigrid of domain-wall fermions, folded on the fifth dimension
// (multigrid/fold.cuh): coarse = P^dag fine and fine = P coarse, for fields on the whole
// lattice (parity < 0) or on the sites of one parity, every vector. dims: the extents of the
// fine lattice; gamma: col[4] and sign[4] of gamma_5. The restriction with leg >= 0: of a
// field that is zero but where that leg of the hops comes from another aggregate of `block`
// sites (the `cross` of bridge_cuda_hop_split).
// Making that P (multigrid/fold.cuh): the chirality parts of the vectors of a field on the
// whole lattice as test vectors first, first + 1, ...; then, for `count` coarse sites from
// first_site, the parts made orthonormal over every aggregate and scaled. d_w: scratch of
// 2 count n block_volume kept doubles; d_fail: 2 count numbers, 0 or 1 + the first vector
// that depends on the earlier ones in that aggregate and chirality.
// Of dim x dim complex matrices on n_sites coarse sites, row-major (re, im), in double
// precision (multigrid/fold.cuh): the inverses (d_fail[site] = 1 where there is none), and
// |H|^-1 of the Hermitian part H of A Gamma_5. d_scratch: 4 n_sites dim^2 doubles. dim at
// most 1024.
void bridge_cuda_blocks_inverse(
    double* d_out, const double* d_a, double* d_scratch, unsigned* d_fail,
    int n_sites, int dim, cudaStream_t stream
);
void bridge_cuda_blocks_abs_inverse_gamma5(
    double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim, cudaStream_t stream
);
// The same for the Hermitian part of A itself, the eigenvalues not above `cutoff` times the
// largest in magnitude left out.
void bridge_cuda_blocks_abs_inverse(
    double* d_out, const double* d_a, double* d_scratch, int n_sites, int dim, double cutoff,
    cudaStream_t stream
);

// P of the multigrid of domain-wall fermions (multigrid/fold.cuh), in single or in half
// precision: coarse = P^dag fine, fine = P coarse, the chirality parts of test vectors into P,
// and those made orthonormal over the aggregates. fold: col[4] and sign[4] of gamma_5, then
// whether P is in half precision.
void bridge_cuda_fold_restrict_f32(
    float* d_coarse, const float* d_fine, const void* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* fold,
    int leg, const int* block, int vectors, cudaStream_t stream
);
void bridge_cuda_fold_prolong_f32(
    float* d_fine, const float* d_coarse, const void* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* fold,
    int vectors, cudaStream_t stream
);
void bridge_cuda_fold_fill_f32(
    void* d_p, const float* d_fine, int n, int first, const int* dims, int Ls, const int* fold,
    int vectors, cudaStream_t stream
);
void bridge_cuda_fold_restrict_f64(
    double* d_coarse, const double* d_fine, const void* d_p, const unsigned* d_sites,
    int n_coarse, int block_volume, int n, const int* dims, int Ls, int parity, const int* fold,
    int leg, const int* block, int vectors, cudaStream_t stream
);
void bridge_cuda_fold_prolong_f64(
    double* d_fine, const double* d_coarse, const void* d_p, const unsigned* d_coarse_of,
    int n_coarse, int n, const int* dims, int Ls, int parity, const int* fold,
    int vectors, cudaStream_t stream
);
void bridge_cuda_fold_fill_f64(
    void* d_p, const double* d_fine, int n, int first, const int* dims, int Ls, const int* fold,
    int vectors, cudaStream_t stream
);
void bridge_cuda_fold_orthonormalise(
    void* d_p, int half, double* d_w, const unsigned* d_sites, unsigned* d_fail,
    int first_site, int count, int block_volume, int n, int kept, double scale,
    cudaStream_t stream
);

// One pass of the coarse operator of a multigrid (multigrid/fold.cuh, coarse_pass_kernel):
// out(X) = [M(X)] (base(X) + A(X) d(X) -+ sum_dir B_dir(X) h(X + dir)) on the coarse sites of
// one parity, the others set to zero (parity < 0: all); every part may be null. dim at most
// 1024. And Gamma_5 of a coarse field, in place.
void bridge_cuda_coarse_pass_f32(
    float* d_out, const float* d_base, const float* d_diag_of, const float* d_hops_of,
    const float* d_diag, const float* d_links, const float* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors, cudaStream_t stream
);
void bridge_cuda_coarse_gamma5_f32(float* d_v, int n_coarse, int dim, int vectors, cudaStream_t stream);
void bridge_cuda_coarse_pass_f64(
    double* d_out, const double* d_base, const double* d_diag_of, const double* d_hops_of,
    const double* d_diag, const double* d_links, const double* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors, cudaStream_t stream
);
void bridge_cuda_coarse_gamma5_f64(double* d_v, int n_coarse, int dim, int vectors, cudaStream_t stream);

// Stout smearing (gauge/stout.cuh), in double precision, links whole: v = exp(iQ) u for one
// step, and one step of the chain rule of its force (Sigma of u from Sigma' of v, with i Theta
// as scratch). rho: the 4 x 4 weights, row-major.
void bridge_cuda_stout_step(double* d_v, const double* d_u, const int* dims, const double* rho, cudaStream_t stream);
void bridge_cuda_stout_chain(
    double* d_sigma, double* d_itheta, const double* d_sigma_prime, const double* d_u,
    const int* dims, const double* rho, cudaStream_t stream);

// The levels below the coarse lattice (multigrid/levels.cuh): out = Q^dag v and out (+)= Q y
// between a level and the one above, Q from test vectors on the level above (in double
// precision), the sources of the probing of a level's operator and its blocks from the
// response, and the operator of the last level as a dense matrix.
#define BRIDGE_CUDA_LEVEL_FUNCTIONS(SUFFIX, Float)                                               \
void bridge_cuda_level_restrict_##SUFFIX(                                                        \
    Float* d_out, const Float* d_v, const Float* d_q, const unsigned* d_members,                \
    int n_blocks, int nq, int bs, int n_parent, int pd, int vectors, cudaStream_t stream);      \
void bridge_cuda_level_prolong_##SUFFIX(                                                         \
    Float* d_out, const Float* d_y, const Float* d_q, const unsigned* d_owner,                  \
    int n_blocks, int nq, int bs, int n_parent, int pd, int add, int vectors, cudaStream_t stream); \
void bridge_cuda_level_basis_##SUFFIX(                                                           \
    double* d_q, const Float* d_tests, const unsigned* d_members,                               \
    int n_blocks, int n_t, int bs, int n_parent, int pd, cudaStream_t stream);                  \
void bridge_cuda_level_sources_##SUFFIX(                                                         \
    Float* d_e, const unsigned* d_colour, unsigned colour, int n_sites, int nq, cudaStream_t stream); \
void bridge_cuda_level_probe_##SUFFIX(                                                           \
    double* d_diag, double* d_links, const Float* d_r, const unsigned* d_colour,                \
    const unsigned* d_neighbours, unsigned colour, unsigned used, int n_sites, int nq,          \
    cudaStream_t stream);
BRIDGE_CUDA_LEVEL_FUNCTIONS(f32, float)
BRIDGE_CUDA_LEVEL_FUNCTIONS(f64, double)
#undef BRIDGE_CUDA_LEVEL_FUNCTIONS
void bridge_cuda_level_dense(
    double* d_dense, const double* d_diag, const double* d_links, const unsigned* d_neighbours,
    unsigned used, int n_sites, int nq, cudaStream_t stream);
// The blocks of the exact adjoint of a coarse operator: out[d][X] = m[d][Y]^dag, Y =
// d_from[d n_coarse + X] (or X if d_from is null), for `dirs` directions.
void bridge_cuda_coarse_adjoint_f32(float* d_out, const float* d_m, const unsigned* d_from, int dirs, int n_coarse, int dim, cudaStream_t stream);
void bridge_cuda_coarse_adjoint_f64(double* d_out, const double* d_m, const unsigned* d_from, int dirs, int n_coarse, int dim, cudaStream_t stream);
// A dense n x n matrix applied to fields of n complex numbers (one site of all of them).
void bridge_cuda_dense_apply_f32(float* d_out, const float* d_r, const float* d_m, int n, int vectors, cudaStream_t stream);
void bridge_cuda_dense_apply_f64(double* d_out, const double* d_r, const double* d_m, int n, int vectors, cudaStream_t stream);
// Dense complex n x n matrices, row-major (re, im), in double precision, of any n, by
// cuSOLVER (multigrid/dense.cu): the inverse (LU with partial pivoting), and V |Lambda|^+ V^dag
// for (A + A^dag)/2 = V Lambda V^dag, the eigenvalues not above `cutoff` times the largest in
// magnitude left out. They return 0, a positive number for a singular matrix or eigensolver
// that did not converge, or a negative cuSOLVER or cuBLAS status.
int bridge_cuda_dense_inverse(double* d_inverse, const double* d_a, int n);
int bridge_cuda_dense_abs_inverse(double* d_out, const double* d_a, int n, double cutoff);

// One leg of the hops H (b - c Y) in on the whole lattice, sorted by whether it comes from
// another block of `block` sites (fermion/hops.cuh, hop_split_kernel): cross = the leg where
// it does, inside += the leg elsewhere. leg mu: forward in direction mu, 4 + mu: backward.
void bridge_cuda_hop_split_f32(
    float* d_cross, float* d_inside, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, cudaStream_t stream
);
void bridge_cuda_hop_split_f64(
    double* d_cross, double* d_inside, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, cudaStream_t stream
);

// Between a field of one spinor per site and the walls of one of Ls (fermion/mdwf.cuh,
// walls_kernel), every vector: to_4d, out = P- in_(s_minus) + P+ in_(s_plus); otherwise
// out_(s_minus) = P- in and out_(s_plus) = P+ in, the other s of out left as they are.
void bridge_cuda_walls_f32(
    float* d_out, const float* d_in, int sites, int Ls, int s_minus, int s_plus, int to_4d,
    int vectors, cudaStream_t stream
);
void bridge_cuda_walls_f64(
    double* d_out, const double* d_in, int sites, int Ls, int s_minus, int s_plus, int to_4d,
    int vectors, cudaStream_t stream
);

// bridge_cuda_fifth_gemm_f32 on the tensor cores, in TF32: M and the input rounded to ten bits
// of mantissa. Ls at most 32.
void bridge_cuda_fifth_gemm_tf32_f32(
    float* d_out, const float* d_in, const float* d_from, const float* d_m,
    int sites, int Ls, cudaStream_t stream
);

// The hops and the product in one pass (fermion/fifth_gemm.cuh, hop_gemm_kernel):
// out = M H (b - c Y) in, or d_from - that. Returns 0, having done nothing, if the spinors of
// a block do not fit in the shared memory for this Ls: then bridge_cuda_hop and
// bridge_cuda_fifth_gemm one after the other do the same.
int bridge_cuda_hop_gemm_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_m,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream
);

// The hops and the product in one pass (fermion/fifth_gemm.cuh, hop_gemm_kernel):
// out = M H (b - c Y) in, or d_from - that. Returns 0, having done nothing, if the spinors of
// a block do not fit in the shared memory for this Ls: then bridge_cuda_hop and
// bridge_cuda_fifth_gemm one after the other do the same.
int bridge_cuda_hop_gemm_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_m,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream
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
// imaginary parts).
// x_v *= a_v (a[v], real), y_v += a_v x_v and y_v = a_v y_v + x_v (a[2 v], a[2 v + 1] the real
// and the imaginary part) for `vectors` vectors of n reals one after the other.
void bridge_cuda_blas_scal_f32(float* x, const double* a, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_caxpy_f32(float* y, const float* x, const double* a, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_caypx_f32(float* y, const float* x, const double* a, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_scal_f64(double* x, const double* a, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_caxpy_f64(double* y, const double* x, const double* a, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_caypx_f64(double* y, const double* x, const double* a, int n, int vectors, cudaStream_t stream);

// x_j <- sum_i x_i c[i vectors + j] for the vectors of a set of n reals each, in place; c on the
// device, (re, im), row-major. At most 32 vectors.
void bridge_cuda_blas_rotate_f32(float* x, const double* c, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_rotate_f64(double* x, const double* c, int n, int vectors, cudaStream_t stream);

// |x_v|^2 and <x_v|y_v> = sum conj(x_v) y_v for `vectors` vectors of n reals one after the other.
// The sums of the CUDA blocks of all of them are fetched together and summed on the host in a
// fixed order: deterministic, and one wait for the device.
void bridge_cuda_blas_norm2_f32(double* out, const float* x, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_dotc_f32(double* re, double* im, const float* x, const float* y, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_norm2_f64(double* out, const double* x, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_dotc_f64(double* re, double* im, const double* x, const double* y, int n, int vectors, cudaStream_t stream);

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
