#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include "bridge_cuda.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << " Test: Coarse Grid Even-Odd Pass, Diag Inv & Schur   " << std::endl;
    std::cout << "======================================================" << std::endl;

    const int nx_c = 4, ny_c = 4, nz_c = 4, nt_c = 4;
    const int vol_c = nx_c * ny_c * nz_c * nt_c; // 256 coarse sites
    const int dim = 8; // 8 coarse colors (e.g. 4 test vectors * 2 chiralities)
    const int matrix_size = 2 * dim * dim; // 128 doubles per block

    // 1. Generate diagonally dominant diagonal blocks A(X) so they are invertible
    std::vector<double> h_diag(vol_c * matrix_size, 0.0);
    for (int site = 0; site < vol_c; ++site) {
        for (int r = 0; r < dim; ++r) {
            for (int c = 0; c < dim; ++c) {
                int idx = site * matrix_size + (r * dim + c) * 2;
                if (r == c) {
                    h_diag[idx + 0] = 5.0 + sin(site * 0.1 + r);
                    h_diag[idx + 1] = 0.5 * cos(site * 0.1 + r);
                } else {
                    h_diag[idx + 0] = 0.1 * sin(site + r * dim + c);
                    h_diag[idx + 1] = 0.1 * cos(site + r * dim + c);
                }
            }
        }
    }

    // Generate random 8-direction links B_dir(X)
    std::vector<double> h_links(8 * vol_c * matrix_size, 0.0);
    for (size_t i = 0; i < h_links.size(); ++i) {
        h_links[i] = 0.2 * sin(i * 0.03 + 0.7);
    }

    double* d_diag = nullptr;
    double* d_diag_inv = nullptr;
    double* d_links = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diag, vol_c * matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diag_inv, vol_c * matrix_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_links, 8 * vol_c * matrix_size * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_diag, h_diag.data(), vol_c * matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_links, h_links.data(), 8 * vol_c * matrix_size * sizeof(double), cudaMemcpyHostToDevice));

    // [Step 1] Invert all diagonal blocks on GPU
    std::cout << "[Step 1] Inverting all diagonal blocks in parallel on GPU..." << std::endl;
    bridge_cuda_mg_coarse_invert_diagonal_blocks_f64(d_diag_inv, d_diag, vol_c, dim, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_diag_inv(vol_c * matrix_size);
    CUDA_CHECK(cudaMemcpy(h_diag_inv.data(), d_diag_inv, vol_c * matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify A * A^{-1} == I on Host
    double max_inv_err = 0.0;
    for (int site = 0; site < vol_c; ++site) {
        for (int r = 0; r < dim; ++r) {
            for (int c = 0; c < dim; ++c) {
                std::complex<double> prod(0, 0);
                for (int k = 0; k < dim; ++k) {
                    int idx_a = site * matrix_size + (r * dim + k) * 2;
                    int idx_inv = site * matrix_size + (k * dim + c) * 2;
                    std::complex<double> a(h_diag[idx_a + 0], h_diag[idx_a + 1]);
                    std::complex<double> inv(h_diag_inv[idx_inv + 0], h_diag_inv[idx_inv + 1]);
                    prod += a * inv;
                }
                std::complex<double> expected = (r == c) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
                double err = std::abs(prod - expected);
                if (err > max_inv_err) max_inv_err = err;
            }
        }
    }
    std::cout << "  -> Max ||A * A^{-1} - I|| deviation: " << max_inv_err << std::endl;
    if (max_inv_err > 1e-12) {
        std::cerr << "FAILED: Coarse diagonal block inversion error too large!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: All 256 diagonal blocks inverted accurately to machine precision!" << std::endl;

    // [Step 2] Test Coarse Gamma_5 reflection
    std::cout << "[Step 2] Testing Coarse Gamma_5 operator (chiral sign flip)..." << std::endl;
    std::vector<double> h_vec(vol_c * dim * 2, 1.0);
    double* d_vec = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec, vol_c * dim * 2 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), vol_c * dim * 2 * sizeof(double), cudaMemcpyHostToDevice));

    bridge_cuda_mg_coarse_gamma5_f64(d_vec, vol_c, dim, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_vec.data(), d_vec, vol_c * dim * 2 * sizeof(double), cudaMemcpyDeviceToHost));

    bool gamma5_ok = true;
    for (int site = 0; site < vol_c; ++site) {
        for (int r = 0; r < dim; ++r) {
            double expected = (r < dim / 2) ? 1.0 : -1.0;
            if (std::abs(h_vec[(site * dim + r) * 2 + 0] - expected) > 1e-15 ||
                std::abs(h_vec[(site * dim + r) * 2 + 1] - expected) > 1e-15) {
                gamma5_ok = false;
            }
        }
    }
    if (!gamma5_ok) {
        std::cerr << "FAILED: Coarse Gamma_5 sign flip incorrect!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Coarse Gamma_5 acts as exact chiral reflection on [dim/2 .. dim-1]!" << std::endl;

    // [Step 3] Test Coarse Even-Odd Schur Complement: S v_e = A_ee v_e - B_eo A_oo^{-1} B_oe v_e
    std::cout << "[Step 3] Testing Coarse Schur Complement Pass S = A_ee - B_eo A_oo^{-1} B_oe..." << std::endl;
    std::vector<double> h_v_even(vol_c * dim * 2, 0.0);
    for (int site = 0; site < vol_c; ++site) {
        int x = site % nx_c; int rem = site / nx_c;
        int y = rem % ny_c; rem /= ny_c;
        int z = rem % nz_c; int t = rem / nz_c;
        if ((x + y + z + t) % 2 == 0) { // Even site
            for (int r = 0; r < dim; ++r) {
                h_v_even[(site * dim + r) * 2 + 0] = sin(site * 0.2 + r);
                h_v_even[(site * dim + r) * 2 + 1] = cos(site * 0.2 + r);
            }
        }
    }

    double* d_v_even = nullptr;
    double* d_t_odd = nullptr;
    double* d_s_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v_even, vol_c * dim * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t_odd, vol_c * dim * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_s_out, vol_c * dim * 2 * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_v_even, h_v_even.data(), vol_c * dim * 2 * sizeof(double), cudaMemcpyHostToDevice));

    // Pass 1 on Odd sites: t_o = A_oo^{-1} (B_oe v_e)
    // target_parity = 1 (Odd), apply_diag = false, apply_hops = true, minus_hops = false, apply_inv = true
    bridge_cuda_mg_coarse_pass_f64(
        d_t_odd, nullptr, nullptr, d_v_even,
        d_diag, d_diag_inv, d_links,
        nx_c, ny_c, nz_c, nt_c, dim, 1,
        false, true, false, true, 0
    );

    // Pass 2 on Even sites: s_e = A_ee v_e - B_eo t_o
    // target_parity = 0 (Even), apply_diag = true (from v_even), apply_hops = true (from t_odd), minus_hops = true, apply_inv = false
    bridge_cuda_mg_coarse_pass_f64(
        d_s_out, nullptr, d_v_even, d_t_odd,
        d_diag, d_diag_inv, d_links,
        nx_c, ny_c, nz_c, nt_c, dim, 0,
        true, true, true, false, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_s_out(vol_c * dim * 2);
    CUDA_CHECK(cudaMemcpy(h_s_out.data(), d_s_out, vol_c * dim * 2 * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute CPU Ground Truth for Schur Complement
    std::vector<double> h_t_odd_cpu(vol_c * dim * 2, 0.0);
    // CPU Hop from Even to Odd: B_oe v_e
    for (int site = 0; site < vol_c; ++site) {
        int x = site % nx_c; int rem = site / nx_c;
        int y = rem % ny_c; rem /= ny_c;
        int z = rem % nz_c; int t = rem / nz_c;
        if ((x + y + z + t) % 2 != 1) continue; // Only Odd sites

        std::vector<std::complex<double>> hop_sum(dim, 0.0);
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
            int n_site = x_n + nx_c * (y_n + ny_c * (z_n + nz_c * t_n));

            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < dim; ++c) {
                    int m_idx = (dir * vol_c + site) * matrix_size + (r * dim + c) * 2;
                    int v_idx = (n_site * dim + c) * 2;
                    std::complex<double> b_mat(h_links[m_idx + 0], h_links[m_idx + 1]);
                    std::complex<double> vn(h_v_even[v_idx + 0], h_v_even[v_idx + 1]);
                    hop_sum[r] += b_mat * vn;
                }
            }
        }

        // Multiply by A_oo^{-1}
        for (int r = 0; r < dim; ++r) {
            std::complex<double> res(0, 0);
            for (int c = 0; c < dim; ++c) {
                int inv_idx = site * matrix_size + (r * dim + c) * 2;
                std::complex<double> inv(h_diag_inv[inv_idx + 0], h_diag_inv[inv_idx + 1]);
                res += inv * hop_sum[c];
            }
            h_t_odd_cpu[(site * dim + r) * 2 + 0] = res.real();
            h_t_odd_cpu[(site * dim + r) * 2 + 1] = res.imag();
        }
    }

    // CPU Pass 2 on Even sites: A_ee v_e - B_eo t_o
    double max_schur_diff = 0.0;
    for (int site = 0; site < vol_c; ++site) {
        int x = site % nx_c; int rem = site / nx_c;
        int y = rem % ny_c; rem /= ny_c;
        int z = rem % nz_c; int t = rem / nz_c;
        if ((x + y + z + t) % 2 != 0) {
            // Odd site should be 0
            for (int r = 0; r < dim; ++r) {
                double val = std::abs(h_s_out[(site * dim + r) * 2 + 0]) + std::abs(h_s_out[(site * dim + r) * 2 + 1]);
                if (val > max_schur_diff) max_schur_diff = val;
            }
            continue;
        }

        // Diagonal part A_ee v_e
        std::vector<std::complex<double>> s_cpu(dim, 0.0);
        for (int r = 0; r < dim; ++r) {
            for (int c = 0; c < dim; ++c) {
                int d_idx = site * matrix_size + (r * dim + c) * 2;
                int v_idx = (site * dim + c) * 2;
                std::complex<double> d(h_diag[d_idx + 0], h_diag[d_idx + 1]);
                std::complex<double> v(h_v_even[v_idx + 0], h_v_even[v_idx + 1]);
                s_cpu[r] += d * v;
            }
        }

        // Subtract B_eo t_o
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
            int n_site = x_n + nx_c * (y_n + ny_c * (z_n + nz_c * t_n));

            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < dim; ++c) {
                    int m_idx = (dir * vol_c + site) * matrix_size + (r * dim + c) * 2;
                    int v_idx = (n_site * dim + c) * 2;
                    std::complex<double> b_mat(h_links[m_idx + 0], h_links[m_idx + 1]);
                    std::complex<double> to(h_t_odd_cpu[v_idx + 0], h_t_odd_cpu[v_idx + 1]);
                    s_cpu[r] -= b_mat * to;
                }
            }
        }

        for (int r = 0; r < dim; ++r) {
            int out_idx = (site * dim + r) * 2;
            double diff_r = std::abs(h_s_out[out_idx + 0] - s_cpu[r].real());
            double diff_i = std::abs(h_s_out[out_idx + 1] - s_cpu[r].imag());
            if (diff_r > max_schur_diff) max_schur_diff = diff_r;
            if (diff_i > max_schur_diff) max_schur_diff = diff_i;
        }
    }

    std::cout << "  -> Max Schur complement difference GPU vs CPU ground truth: " << max_schur_diff << std::endl;
    if (max_schur_diff > 1e-13) {
        std::cerr << "FAILED: Coarse Schur complement pass difference too large!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Coarse Schur complement Pass matches CPU ground truth to < 1e-13!" << std::endl;

    // Cleanup
    cudaFree(d_diag);
    cudaFree(d_diag_inv);
    cudaFree(d_links);
    cudaFree(d_vec);
    cudaFree(d_v_even);
    cudaFree(d_t_odd);
    cudaFree(d_s_out);

    std::cout << "\nALL COARSE EVEN-ODD & SCHUR COMPLEMENT TESTS PASSED PERFECTLY!" << std::endl;
    return 0;
}
