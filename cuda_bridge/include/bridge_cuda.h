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
// bridge_cuda_hop_faces with the legs across a cut from the buffers of `halo`.
void bridge_cuda_hop_faces_halo_f32(
    float* d_faces, float* d_inside, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    const int* block, int vectors, const BridgeCudaHalo* halo, cudaStream_t stream);
void bridge_cuda_hop_faces_halo_f64(
    double* d_faces, double* d_inside, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    const int* block, int vectors, const BridgeCudaHalo* halo, cudaStream_t stream);
int bridge_cuda_dense_inverse(double* d_inverse, const double* d_a, int n);
int bridge_cuda_dense_abs_inverse(double* d_out, const double* d_a, int n, double cutoff);

// The hops H (b - c Y) in on the whole lattice, sorted by whether a leg comes from another
// block of `block` sites (fermion/hops.cuh, hop_faces_kernel): inside += the legs where it does
// not, in the order of the legs; each leg where it does into its face in d_faces (the faces of
// the 8 legs one after the other, aggregate_faces_offset). Leg mu: forward in direction mu,
// 4 + mu: backward.
void bridge_cuda_hop_faces_f32(
    float* d_faces, float* d_inside, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    const int* block, int vectors, cudaStream_t stream
);
void bridge_cuda_hop_faces_f64(
    double* d_faces, double* d_inside, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    const int* block, int vectors, cudaStream_t stream
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
