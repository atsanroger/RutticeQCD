#ifndef BRIDGE_CUDA_MULTIGRID_COARSE_EO_CUH
#define BRIDGE_CUDA_MULTIGRID_COARSE_EO_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include <cuda_runtime.h>

namespace bridge_cuda {

// =========================================================================
// Coarse Grid Even-Odd Operators and Schur Complement Pass
//
// Aligned with RutticeQCD CoarseOp::pass:
// out(X) = [A(X)^{-1}] * ( base(X) + A(X) * d(X) +/- sum_{dir=0..7} B_dir(X) * h(X + dir) )
//
// Every term is optional:
// - d_base: base vector (optional)
// - d_diag_in: vector multiplied by diagonal block A(X) (optional)
// - d_hop_in: vector hopped from neighbors h(X + dir) (optional)
// - apply_inv: whether to multiply the accumulator by A(X)^{-1}
// =========================================================================

template <typename Float, int MAX_DIM = 32>
__global__ void mg_coarse_pass_kernel(
    Float* __restrict__ d_out,
    const Float* __restrict__ d_base,
    const Float* __restrict__ d_diag_in,
    const Float* __restrict__ d_hop_in,
    const Float* __restrict__ d_diag,
    const Float* __restrict__ d_diag_inv,
    const Float* __restrict__ d_links,
    int nx_c, int ny_c, int nz_c, int nt_c,
    int dim,
    int target_parity,
    bool apply_diag,
    bool apply_hops,
    bool minus_hops,
    bool apply_inv)
{
    int vol_c = nx_c * ny_c * nz_c * nt_c;
    int matrix_size = 2 * dim * dim;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol_c; site += blockDim.x * gridDim.x) {
        int x = site % nx_c; int rem = site / nx_c;
        int y = rem % ny_c; rem /= ny_c;
        int z = rem % nz_c;
        int t = rem / nz_c;

        int site_parity = (x + y + z + t) & 1;

        // Check parity filter (target_parity == 0 for Even, 1 for Odd, -1 for all)
        if (target_parity >= 0 && site_parity != target_parity) {
            // Sites of the other parity are set to zero (matching RutticeQCD CoarseOp::pass)
            for (int r = 0; r < dim; ++r) {
                d_out[(site * dim + r) * 2 + 0] = static_cast<Float>(0);
                d_out[(site * dim + r) * 2 + 1] = static_cast<Float>(0);
            }
            continue;
        }

        Float acc_r[MAX_DIM] = {0};
        Float acc_i[MAX_DIM] = {0};

        // 1. Optional Base vector: acc += base(X)
        if (d_base != nullptr) {
            for (int r = 0; r < dim; ++r) {
                int b_idx = (site * dim + r) * 2;
                acc_r[r] += d_base[b_idx + 0];
                acc_i[r] += d_base[b_idx + 1];
            }
        }

        // 2. Optional Diagonal block term: acc += A(X) * d(X)
        if (apply_diag && d_diag_in != nullptr && d_diag != nullptr) {
            const Float* mat = d_diag + site * matrix_size;
            for (int r = 0; r < dim; ++r) {
                Float sum_r = 0, sum_i = 0;
                for (int c = 0; c < dim; ++c) {
                    int m_idx = (r * dim + c) * 2;
                    Float a = mat[m_idx + 0];
                    Float b = mat[m_idx + 1];

                    int v_idx = (site * dim + c) * 2;
                    Float vr = d_diag_in[v_idx + 0];
                    Float vi = d_diag_in[v_idx + 1];

                    sum_r += a * vr - b * vi;
                    sum_i += a * vi + b * vr;
                }
                acc_r[r] += sum_r;
                acc_i[r] += sum_i;
            }
        }

        // 3. Optional 8-direction Hopping terms: acc +=/- sum_dir B_dir(X) * h(X + dir)
        if (apply_hops && d_hop_in != nullptr && d_links != nullptr) {
            Float hop_sign = minus_hops ? static_cast<Float>(-1.0) : static_cast<Float>(1.0);

            for (int dir = 0; dir < 8; ++dir) {
                int mu = dir % 4;
                bool is_forward = (dir < 4);

                int x_n = x, y_n = y, z_n = z, t_n = t;
                if (is_forward) {
                    if (mu == 0) x_n = (x + 1 < nx_c) ? x + 1 : 0;
                    if (mu == 1) y_n = (y + 1 < ny_c) ? y + 1 : 0;
                    if (mu == 2) z_n = (z + 1 < nz_c) ? z + 1 : 0;
                    if (mu == 3) t_n = (t + 1 < nt_c) ? t + 1 : 0;
                } else {
                    if (mu == 0) x_n = (x - 1 >= 0) ? x - 1 : nx_c - 1;
                    if (mu == 1) y_n = (y - 1 >= 0) ? y - 1 : ny_c - 1;
                    if (mu == 2) z_n = (z - 1 >= 0) ? z - 1 : nz_c - 1;
                    if (mu == 3) t_n = (t - 1 >= 0) ? t - 1 : nt_c - 1;
                }
                int neighbor_site = x_n + nx_c * (y_n + ny_c * (z_n + nz_c * t_n));

                const Float* link_mat = d_links + (dir * vol_c + site) * matrix_size;

                for (int r = 0; r < dim; ++r) {
                    Float sum_r = 0, sum_i = 0;
                    for (int c = 0; c < dim; ++c) {
                        int m_idx = (r * dim + c) * 2;
                        Float a = link_mat[m_idx + 0];
                        Float b = link_mat[m_idx + 1];

                        int v_idx = (neighbor_site * dim + c) * 2;
                        Float vr = d_hop_in[v_idx + 0];
                        Float vi = d_hop_in[v_idx + 1];

                        sum_r += a * vr - b * vi;
                        sum_i += a * vi + b * vr;
                    }
                    acc_r[r] += hop_sign * sum_r;
                    acc_i[r] += hop_sign * sum_i;
                }
            }
        }

        // 4. Optional Inverse factor: out = A(X)^{-1} * acc
        if (apply_inv && d_diag_inv != nullptr) {
            const Float* inv_mat = d_diag_inv + site * matrix_size;
            for (int r = 0; r < dim; ++r) {
                Float res_r = 0, res_i = 0;
                for (int c = 0; c < dim; ++c) {
                    int m_idx = (r * dim + c) * 2;
                    Float a = inv_mat[m_idx + 0];
                    Float b = inv_mat[m_idx + 1];

                    res_r += a * acc_r[c] - b * acc_i[c];
                    res_i += a * acc_i[c] + b * acc_r[c];
                }
                int out_idx = (site * dim + r) * 2;
                d_out[out_idx + 0] = res_r;
                d_out[out_idx + 1] = res_i;
            }
        } else {
            // Write directly to output
            for (int r = 0; r < dim; ++r) {
                int out_idx = (site * dim + r) * 2;
                d_out[out_idx + 0] = acc_r[r];
                d_out[out_idx + 1] = acc_i[r];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Coarse Gamma_5 Kernel: in-place v = Gamma_5 * v
// Multiplies the second half of coarse color components [dim/2 .. dim-1] by -1.
// Aligned with RutticeQCD coarse_gamma5.
// -------------------------------------------------------------------------
template <typename Float>
__global__ void mg_coarse_gamma5_kernel(
    Float* __restrict__ d_vec,
    int total_sites,
    int dim)
{
    int half_dim = dim / 2;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < total_sites; site += blockDim.x * gridDim.x) {
        for (int r = half_dim; r < dim; ++r) {
            int idx = (site * dim + r) * 2;
            d_vec[idx + 0] = -d_vec[idx + 0];
            d_vec[idx + 1] = -d_vec[idx + 1];
        }
    }
}

// -------------------------------------------------------------------------
// Coarse Block Matrix Inversion Helper on Host/Device:
// Inverts each dim x dim complex matrix along the coarse grid.
// Uses Gauss-Jordan elimination with partial pivoting.
// -------------------------------------------------------------------------
template <typename Float, int MAX_DIM = 32>
__device__ bool invert_complex_block(
    int n,
    const Float* __restrict__ a_in,
    Float* __restrict__ a_inv)
{
    Float a_re[MAX_DIM][MAX_DIM];
    Float a_im[MAX_DIM][MAX_DIM];
    Float inv_re[MAX_DIM][MAX_DIM];
    Float inv_im[MAX_DIM][MAX_DIM];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = (i * n + j) * 2;
            a_re[i][j] = a_in[idx + 0];
            a_im[i][j] = a_in[idx + 1];
            inv_re[i][j] = (i == j) ? static_cast<Float>(1.0) : static_cast<Float>(0.0);
            inv_im[i][j] = static_cast<Float>(0.0);
        }
    }

    // Gauss-Jordan elimination
    for (int col = 0; col < n; ++col) {
        // Pivot search
        int pivot_row = col;
        Float max_mag = a_re[col][col] * a_re[col][col] + a_im[col][col] * a_im[col][col];
        for (int r = col + 1; r < n; ++r) {
            Float mag = a_re[r][col] * a_re[r][col] + a_im[r][col] * a_im[r][col];
            if (mag > max_mag) {
                max_mag = mag;
                pivot_row = r;
            }
        }

        if (max_mag < static_cast<Float>(1e-28)) {
            return false; // Singular matrix
        }

        // Swap rows
        if (pivot_row != col) {
            for (int j = 0; j < n; ++j) {
                Float tmp_r = a_re[col][j]; Float tmp_i = a_im[col][j];
                a_re[col][j] = a_re[pivot_row][j]; a_im[col][j] = a_im[pivot_row][j];
                a_re[pivot_row][j] = tmp_r; a_im[pivot_row][j] = tmp_i;

                Float tmp_ir = inv_re[col][j]; Float tmp_ii = inv_im[col][j];
                inv_re[col][j] = inv_re[pivot_row][j]; inv_im[col][j] = inv_im[pivot_row][j];
                inv_re[pivot_row][j] = tmp_ir; inv_im[pivot_row][j] = tmp_ii;
            }
        }

        // Scale pivot row
        Float p_r = a_re[col][col];
        Float p_i = a_im[col][col];
        Float denom = p_r * p_r + p_i * p_i;
        Float inv_p_r = p_r / denom;
        Float inv_p_i = -p_i / denom;

        for (int j = 0; j < n; ++j) {
            Float ar = a_re[col][j]; Float ai = a_im[col][j];
            a_re[col][j] = ar * inv_p_r - ai * inv_p_i;
            a_im[col][j] = ar * inv_p_i + ai * inv_p_r;

            Float ir = inv_re[col][j]; Float ii = inv_im[col][j];
            inv_re[col][j] = ir * inv_p_r - ii * inv_p_i;
            inv_im[col][j] = ir * inv_p_i + ii * inv_p_r;
        }

        // Eliminate column entries in other rows
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            Float factor_r = a_re[r][col];
            Float factor_i = a_im[r][col];

            for (int j = 0; j < n; ++j) {
                Float ar = a_re[col][j]; Float ai = a_im[col][j];
                a_re[r][j] -= (factor_r * ar - factor_i * ai);
                a_im[r][j] -= (factor_r * ai + factor_i * ar);

                Float ir = inv_re[col][j]; Float ii = inv_im[col][j];
                inv_re[r][j] -= (factor_r * ir - factor_i * ii);
                inv_im[r][j] -= (factor_r * ii + factor_i * ir);
            }
        }
    }

    // Write back inverse
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = (i * n + j) * 2;
            a_inv[idx + 0] = inv_re[i][j];
            a_inv[idx + 1] = inv_im[i][j];
        }
    }

    return true;
}

// Kernel to invert all coarse diagonal blocks in parallel on GPU
template <typename Float, int MAX_DIM = 32>
__global__ void mg_coarse_invert_diagonal_blocks_kernel(
    Float* __restrict__ d_diag_inv,
    const Float* __restrict__ d_diag,
    int vol_c,
    int dim)
{
    int matrix_size = 2 * dim * dim;

    for (int site = blockIdx.x * blockDim.x + threadIdx.x; site < vol_c; site += blockDim.x * gridDim.x) {
        const Float* a_in = d_diag + site * matrix_size;
        Float* a_inv = d_diag_inv + site * matrix_size;

        invert_complex_block<Float, MAX_DIM>(dim, a_in, a_inv);
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_COARSE_EO_CUH
