#ifndef BRIDGE_CUDA_H
#define BRIDGE_CUDA_H

#include "base/types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Clover Operator
// =========================================================================

// out = T in: d_clover holds the blocks of T, or of its inverse (fermion/clover.cuh); on the
// whole lattice (parity < 0) or on the sites of one parity.
void bridge_cuda_clover_term_f32(
    float* d_out, const float* d_in, const float* d_clover,
    const BridgeCudaGeometry* geom, int parity, int vectors, cudaStream_t stream
);

// out = T in: d_clover holds the blocks of T, or of its inverse (fermion/clover.cuh); on the
// whole lattice (parity < 0) or on the sites of one parity.
void bridge_cuda_clover_term_f64(
    double* d_out, const double* d_in, const double* d_clover,
    const BridgeCudaGeometry* geom, int parity, int vectors, cudaStream_t stream
);

// =========================================================================
// Mobius Domain Wall Fermion (MDWF)
// =========================================================================

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

// The parts of the MDWF operator (fermion/mdwf.cuh): the plain operator, Dirac representation.

// out = H (b - c Y) in, or with H^dag (fermion/hops.cuh): on the whole lattice
// (target_parity < 0), or out on the sites of parity target_parity and in on the others. The
// hop between the parities of the Wilson and clover operators (Ls = 1, b = 1, c = 0) and of
// the domain-wall operator (kappa = 1/2, its b and c). With d_from not null, d_from - that;
// with d_clover not null as well (Ls = 1), T d_from + H in, T the clover blocks of d_clover:
// the clover operator D = T + H (kappa), and the Wilson operator is d_from = in, kappa -> -kappa.
void bridge_cuda_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream
);

// The four-dimensional operator on the blocks of one colour (fermion/hops.cuh, BlockCut; cut:
// extent[4], colour, shift): out = d_from - H in at kappa, or with the clover blocks d_clover
// T d_from + H in, on those blocks with the legs that stay inside them; zero elsewhere. The
// block operator of the Schwarz smoother (multigrid/sap.cuh).
void bridge_cuda_hop_block_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_clover,
    const BridgeCudaGeometry* geom, double kappa, int dagger, int vectors, const int* cut,
    cudaStream_t stream);
void bridge_cuda_hop_block_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_clover,
    const BridgeCudaGeometry* geom, double kappa, int dagger, int vectors, const int* cut,
    cudaStream_t stream);

// The Schwarz smoother (multigrid/sap.cuh), four-dimensional fields of `vectors` vectors on
// `sites` sites: v = 0 outside the blocks of colour c; alpha = <p, r> / <p, p> of every block
// of colour c and vector (0 for the others), in double precision; e += alpha r, r -= alpha p.
// blocks: the colour of every block; block_sites: the block_volume sites of every block;
// block_of: the block of every site.
void bridge_cuda_sap_keep_colour_f32(float* d_v, const unsigned* d_blocks, const unsigned* d_block_of,
                                     int sites, int vectors, int c, cudaStream_t stream);
void bridge_cuda_sap_keep_colour_f64(double* d_v, const unsigned* d_blocks, const unsigned* d_block_of,
                                     int sites, int vectors, int c, cudaStream_t stream);
void bridge_cuda_sap_alpha_f32(double* d_alpha, const float* d_p, const float* d_r, const unsigned* d_blocks,
                               const unsigned* d_block_sites, int n_blocks, int block_volume, int sites,
                               int vectors, int c, cudaStream_t stream);
void bridge_cuda_sap_alpha_f64(double* d_alpha, const double* d_p, const double* d_r, const unsigned* d_blocks,
                               const unsigned* d_block_sites, int n_blocks, int block_volume, int sites,
                               int vectors, int c, cudaStream_t stream);
void bridge_cuda_sap_update_f32(float* d_e, float* d_r, const float* d_p, const double* d_alpha,
                                const unsigned* d_block_of, int sites, int vectors, cudaStream_t stream);
void bridge_cuda_sap_update_f64(double* d_e, double* d_r, const double* d_p, const double* d_alpha,
                                const unsigned* d_block_of, int sites, int vectors, cudaStream_t stream);

// Several processes (fermion/hops.cuh, Halo). The hops of bridge_cuda_hop and D of
// bridge_cuda_mdwf_dslash_1pass on the interior of the output (part 1) or on its surface
// (part 2, the legs from other processes from the buffers of `halo`); and the buffer of one
// face out of the input field (on the whole lattice for in_parity < 0).
void bridge_cuda_hop_part_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, const BridgeCudaHalo* halo, int part,
    cudaStream_t stream);
void bridge_cuda_hop_part_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, const BridgeCudaHalo* halo, int part,
    cudaStream_t stream);
void bridge_cuda_mdwf_dslash_part_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, const BridgeCudaHalo* halo, int part, cudaStream_t stream);
void bridge_cuda_mdwf_dslash_part_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, const BridgeCudaHalo* halo, int part, cudaStream_t stream);
// bridge_cuda_hop_split with the legs across a cut from the buffers of `halo`.
void bridge_cuda_hop_split_halo_f32(
    float* d_cross, float* d_inside, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, const BridgeCudaHalo* halo, cudaStream_t stream);
void bridge_cuda_hop_split_halo_f64(
    double* d_cross, double* d_inside, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, const BridgeCudaHalo* halo, cudaStream_t stream);
void bridge_cuda_hop_pack_f32(
    float* d_send, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int in_parity, int face, int vectors, cudaStream_t stream);
void bridge_cuda_hop_pack_f64(
    double* d_send, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int in_parity, int face, int vectors, cudaStream_t stream);

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
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_clover,
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
// The same by CholeskyQR2 (fold_orthonormalise_cholesky_kernel), n <= 48: where d_fail is
// 0xffffffff afterwards, P is left alone and bridge_cuda_fold_orthonormalise is to be run.
void bridge_cuda_fold_orthonormalise_cholesky(
    void* d_p, int half, double* d_w, const unsigned* d_sites, unsigned* d_fail,
    int first_site, int count, int block_volume, int n, int kept, double scale,
    cudaStream_t stream
);

// Several processes (multigrid/fold.cuh): the coarse pass with the neighbours Y >= n_coarse read
// from d_ghost, a coarse field of n_ghost sites; the sites d_sites of a coarse field gathered
// into a coarse field of n sites (a face to send); such a field placed at sites first .. of the
// ghosts.
void bridge_cuda_coarse_pass_ghost_f32(
    float* d_out, const float* d_base, const float* d_diag_of, const float* d_hops_of,
    const float* d_diag, const float* d_links, const float* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors,
    const float* d_ghost, int n_ghost, cudaStream_t stream);
void bridge_cuda_coarse_gather_f32(float* d_buf, const float* d_field, const unsigned* d_sites, int n, int n_coarse,
                                  int dim, int vectors, cudaStream_t stream);
void bridge_cuda_coarse_place_f32(float* d_ghost, const float* d_buf, int first, int n, int n_ghost, int dim, int vectors,
                                 cudaStream_t stream);
void bridge_cuda_coarse_pass_ghost_f64(
    double* d_out, const double* d_base, const double* d_diag_of, const double* d_hops_of,
    const double* d_diag, const double* d_links, const double* d_inverse,
    const unsigned* d_neighbours, const unsigned* d_odd,
    int n_coarse, int dim, int parity, int minus, int gamma5, int vectors,
    const double* d_ghost, int n_ghost, cudaStream_t stream);
void bridge_cuda_coarse_gather_f64(double* d_buf, const double* d_field, const unsigned* d_sites, int n, int n_coarse,
                                  int dim, int vectors, cudaStream_t stream);
void bridge_cuda_coarse_place_f64(double* d_ghost, const double* d_buf, int first, int n, int n_ghost, int dim, int vectors,
                                 cudaStream_t stream);

// One pass of the coarse operator of a multigrid (multigrid/fold.cuh, coarse_pass_kernel):
// out(X) = [M(X)] (base(X) + A(X) d(X) -+ sum_dir B_dir(X) h(X + dir)) on the coarse sites of
// one parity, the others set to zero (parity < 0: all); every part may be null. dim even and at
// most 1024. And Gamma_5 of a coarse field, in place.
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
// The coarse unit vectors of the components first .. first + vectors, in a field of that many
// vectors that is zero; and column first + v of the matrices of every coarse site (the layout
// of bridge_cuda_coarse_pass_*, float with target_single, else double) from vector v of a
// coarse field: the coarse operator put together on the device (multigrid/fold.cuh).
void bridge_cuda_coarse_units_f32(float* d_v, int n_coarse, int dim, int first, int vectors, cudaStream_t stream);
void bridge_cuda_coarse_units_f64(double* d_v, int n_coarse, int dim, int first, int vectors, cudaStream_t stream);
void bridge_cuda_coarse_columns_f32(void* d_target, int target_single, const float* d_c, int n_coarse, int dim, int first, int vectors, cudaStream_t stream);
void bridge_cuda_coarse_columns_f64(void* d_target, int target_single, const double* d_c, int n_coarse, int dim, int first, int vectors, cudaStream_t stream);

// The derivative of a fermion action by the links for the hops of the Wilson kernel
// (fermion/force.cuh), in double precision: M of -2 Re zeta^dag dD eta with eta the field the
// hops act on, gamma5 zeta and gamma5 eta given; coef the coefficient of the hops in D.
void bridge_cuda_hop_force(
    double* d_m, const double* d_zeta, const double* d_eta, const double* d_g5zeta, const double* d_g5eta,
    const double* d_gauge, int link_rows, const int* dims, const int* bc, int Ls, double coef,
    const BridgeCudaHalo* h_eta, const BridgeCudaHalo* h_g5zeta, cudaStream_t stream);

// The clover term made of the links (fermion/clover_links.cuh), in double precision. dims: the
// local extents, then the depths of the halo of the extended fields (stout.cuh).
// The blocks of T = 1 - kappa c_SW sigma F and of T^-1 (clover.cuh's layout) from the links u
// extended one site deep, ln |det| of every block (block w of local site x at w vol + x), and
// singular set to a site plus one if a block there cannot be inverted.
void bridge_cuda_clover_blocks(
    double* d_t, double* d_t_inv, double* d_ln_det, double* d_singular, const double* d_u,
    double kappa_csw, const int* dims, cudaStream_t stream);
// B += coef (A - A^dag), A_munu = sum_spin gamma_mu gamma_nu eta zeta^dag at every site: planes
// 0-3 in b_lo, 4-5 in the first two directions of b_hi (fields of links).
void bridge_cuda_clover_insertion(
    double* d_b_lo, double* d_b_hi, const double* d_zeta, const double* d_eta, double coef, int vol,
    cudaStream_t stream);
// The same for the pairs (T^-1 e_k, e_k) on the local sites whose coordinates add up to parity
// (mod 2): the derivative of ln |det T| there.
void bridge_cuda_clover_log_det_insertion(
    double* d_b_lo, double* d_b_hi, const double* d_t_inv, double coef, int parity, const int* dims,
    cudaStream_t stream);
// M of the leaves through every link with B inserted at their corners; u, b_lo, b_hi extended.
void bridge_cuda_clover_leaves(
    double* d_m, const double* d_u, const double* d_b_lo, const double* d_b_hi, const int* dims,
    cudaStream_t stream);

// Stout smearing (gauge/stout.cuh), in double precision, links whole: v = exp(iQ) u for one
// step, and one step of the chain rule of its force (Sigma of u from Sigma' of v) in two
// parts: the local one, which makes i Theta, and the terms of the staples, which read i Theta
// at the neighbours. rho: the 4 x 4 weights, row-major. dims, here and below: the local
// extents and the depths of the halo of the fields read at the neighbours (u, and i Theta in
// the second part), which are extended (stout.cuh); 0 on one process.
void bridge_cuda_stout_step(double* d_v, const double* d_u, const int* dims, const double* rho, cudaStream_t stream);
void bridge_cuda_stout_chain_local(
    double* d_sigma, double* d_itheta, const double* d_sigma_prime, const double* d_u,
    const int* dims, const double* rho, cudaStream_t stream);
void bridge_cuda_stout_chain_staples(
    double* d_sigma, const double* d_u, const double* d_itheta,
    const int* dims, const double* rho, cudaStream_t stream);

// The extended field of links (gauge/links.cuh): the local field into its interior, and the
// layers first .. first + depth - 1 of direction mu to a buffer (pack) or from it.
void bridge_cuda_links_extend(double* d_ext, const double* d_local, const int* dims, cudaStream_t stream);
void bridge_cuda_links_layers(double* d_buf, double* d_ext, int mu, int first, int pack, const int* dims, cudaStream_t stream);

// HMC on the device (gauge/links.cuh), fields of matrices as for stout smearing: y += a x
// (n reals), U <- exp(eps iP) U (Taylor series to `order`, reunitarised), out = s TA(m),
// out = a b or a^dag b link by link, the force of the gauge action of plaquettes and
// rectangles, the sums of Re tr of the plaquettes and of the rectangles of every site, and
// the sum of the squares of the reals of every site.
void bridge_cuda_links_axpy(double* d_y, double a, const double* d_x, size_t n, cudaStream_t stream);
void bridge_cuda_links_exp_update(double* d_u, const double* d_p, double eps, int order, int vol, cudaStream_t stream);
void bridge_cuda_links_ta(double* d_out, const double* d_m, double s, int vol, cudaStream_t stream);
void bridge_cuda_links_mul(double* d_out, const double* d_a, const double* d_b, int adjoint_a, int vol, cudaStream_t stream);
void bridge_cuda_gauge_force(double* d_f, const double* d_u, double beta, double c_plaq, double c_rect, const int* dims, cudaStream_t stream);
void bridge_cuda_gauge_traces(double* d_plaq, double* d_rect, const double* d_u, int with_rect, const int* dims, cudaStream_t stream);
void bridge_cuda_links_norm2(double* d_out, const double* d_m, int vol, cudaStream_t stream);

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
// Linear Algebra (BLAS, Cast)
// =========================================================================

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
// out_j = sum_i x_i c[i n_out + j] for the n_in vectors of x and the n_out of out (another set),
// any number of them; c on the device as in rotate.
void bridge_cuda_blas_combine_f32(float* out, const float* x, const double* c, int n, int n_in, int n_out, cudaStream_t stream);
void bridge_cuda_blas_combine_f64(double* out, const double* x, const double* c, int n, int n_in, int n_out, cudaStream_t stream);

// Reductions over `vectors` vectors of n reals one after the other, in one pass over the
// fields: |x_v|^2; <x_v|y_v> = sum conj(x_v) y_v; y_v += a_v x_v then |y_v|^2; s_v = x_v + a_v y_v
// then |s_v|^2 and <z_v|s_v>; <s_v|x_v>, |s_v|^2 and |x_v|^2. Summed on the device in two
// stages in a fixed order (deterministic); only the results come to the host (out, re, im and
// the norms: host memory, `vectors` doubles each), one wait for the stream. Return a
// cudaError_t (0: success).
int bridge_cuda_blas_norm2_f32(double* out, const float* x, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_dotc_f32(double* re, double* im, const float* x, const float* y, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_axpy_norm2_f32(float* y, const float* x, const double* a, double* norm, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_xpay_norm2_dot_f32(float* s, const float* x, const double* a, const float* y, const float* z, double* norm, double* re, double* im, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_dot_norms_f32(const float* s, const float* x, double* re, double* im, double* norm_s, double* norm_x, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_norm2_f64(double* out, const double* x, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_dotc_f64(double* re, double* im, const double* x, const double* y, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_axpy_norm2_f64(double* y, const double* x, const double* a, double* norm, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_xpay_norm2_dot_f64(double* s, const double* x, const double* a, const double* y, const double* z, double* norm, double* re, double* im, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_dot_norms_f64(const double* s, const double* x, double* re, double* im, double* norm_s, double* norm_x, int n, int vectors, cudaStream_t stream);
// <x_i|y_j> for the nx vectors of x and the ny of y (sets of their own, vectors of n reals):
// re[i ny + j], im[i ny + j], host memory. Every vector read once for every few of the other
// set (linalg/blas.cuh, blas_gram_kernel). As the reductions above.
int bridge_cuda_blas_gram_f32(double* re, double* im, const float* x, const float* y, int n, int nx, int ny, cudaStream_t stream);
int bridge_cuda_blas_gram_f64(double* re, double* im, const double* x, const double* y, int n, int nx, int ny, cudaStream_t stream);
// s_v = c_v s_v + a_v x_v + b_v y_v, then |s_v|^2 (a step of a Chebyshev recurrence); c real
// (one double for each vector), a and b complex (re, im). As the reductions above.
int bridge_cuda_blas_scale_axpy2_norm2_f32(float* s, const double* c, const double* a, const float* x, const double* b, const float* y, double* norm, int n, int vectors, cudaStream_t stream);
int bridge_cuda_blas_scale_axpy2_norm2_f64(double* s, const double* c, const double* a, const double* x, const double* b, const double* y, double* norm, int n, int vectors, cudaStream_t stream);

// In one pass: s_v = x_v + a_v y_v; s_v += a_v x_v + b_v y_v; s_v = y_v + b_v (s_v + a_v x_v).
// Coefficients complex, (re, im) for each vector, on the host.
void bridge_cuda_blas_xpay_f32(float* s, const float* x, const double* a, const float* y, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_axpy2_f32(float* s, const double* a, const float* x, const double* b, const float* y, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_axpy_aypx_f32(float* s, const double* a, const float* x, const double* b, const float* y, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_xpay_f64(double* s, const double* x, const double* a, const double* y, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_axpy2_f64(double* s, const double* a, const double* x, const double* b, const double* y, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_axpy_aypx_f64(double* s, const double* a, const double* x, const double* b, const double* y, int n, int vectors, cudaStream_t stream);

// A step of Chebyshev iteration in one pass: out_v += d_v, r_v -= ad_v, d_v = c_v d_v + a_v r_v;
// c real (one for each vector), a complex (re, im), on the host.
void bridge_cuda_blas_chebyshev_step_f32(float* out, float* d, float* r, const float* ad, const double* c, const double* a, int n, int vectors, cudaStream_t stream);
void bridge_cuda_blas_chebyshev_step_f64(double* out, double* d, double* r, const double* ad, const double* c, const double* a, int n, int vectors, cudaStream_t stream);

void bridge_cuda_cast_f64_to_f32(float* dst, const double* src, int total_elements, cudaStream_t stream);
void bridge_cuda_cast_f32_to_f64(double* dst, const float* src, int total_elements, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_CUDA_H
