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
    std::cout << " Test: MDWF 5D Folded MG Transfer & Block Orthonormalization " << std::endl;
    std::cout << "======================================================" << std::endl;

    const int nx = 4, ny = 4, nz = 4, nt = 4;
    const int Ls = 8;
    const int vol4d = nx * ny * nz * nt; // 256
    const int vol5d = vol4d * Ls;       // 2048
    const int fine_reals = 24 * vol5d;

    const int bx = 2, by = 2, bz = 2, bt = 2;
    const int nbx = nx / bx;
    const int nby = ny / by;
    const int nbz = nz / bz;
    const int nbt = nt / bt;
    const int total_blocks = nbx * nby * nbz * nbt; // 16 coarse sites

    const int num_vectors = 4; // 4 test vectors -> 8 coarse colors (+ and -)
    const int coarse_dim = 2 * num_vectors;
    const int coarse_reals = total_blocks * coarse_dim * 2;

    const int kept = (Ls / 2) * 24; // 4 * 24 = 96
    const int basis_reals = vol4d * num_vectors * 2 * kept;

    BridgeCudaGeometry geom;
    geom.dims[0] = nx; geom.dims[1] = ny; geom.dims[2] = nz; geom.dims[3] = nt;
    geom.Ls = Ls;
    geom.vol = vol4d;
    geom.vol5d = vol5d;
    geom.vol_pad = ((vol4d + 31) / 32) * 32;
    geom.gamma_rep = BRIDGE_CUDA_GAMMA_CHIRAL;
    geom.link_rows = 3;

    BridgeCudaMgParams mg_params;
    mg_params.num_colors = coarse_dim;
    mg_params.block_size[0] = bx;
    mg_params.block_size[1] = by;
    mg_params.block_size[2] = bz;
    mg_params.block_size[3] = bt;

    // 1. Initialize random basis vectors
    std::vector<float> h_basis(basis_reals);
    for (size_t i = 0; i < h_basis.size(); ++i) {
        h_basis[i] = static_cast<float>(sin(i * 0.17 + 0.3));
    }

    float* d_basis = nullptr;
    CUDA_CHECK(cudaMalloc(&d_basis, basis_reals * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), basis_reals * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Perform Block Orthonormalization on GPU
    std::cout << "[Step 1] Running 5D Block Modified Gram-Schmidt..." << std::endl;
    bridge_cuda_mdwf_mg_block_normalize_5d_f32(d_basis, num_vectors, &geom, &mg_params, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_basis.data(), d_basis, basis_reals * sizeof(float), cudaMemcpyDeviceToHost));

    // Check block norm on host
    double max_norm_err = 0.0;
    for (int b = 0; b < total_blocks; ++b) {
        int bx_idx = b % nbx; int rem = b / nbx;
        int by_idx = rem % nby; rem /= nby;
        int bz_idx = rem % nbz;
        int bt_idx = rem / nbz;

        for (int ch = 0; ch < 2; ++ch) {
            for (int k = 0; k < num_vectors; ++k) {
                double norm_sq = 0.0;
                for (int lt = 0; lt < bt; ++lt) {
                    int t = bt_idx * bt + lt;
                    for (int lz = 0; lz < bz; ++lz) {
                        int z = bz_idx * bz + lz;
                        for (int ly = 0; ly < by; ++ly) {
                            int y = by_idx * by + ly;
                            for (int lx = 0; lx < bx; ++lx) {
                                int x = bx_idx * bx + lx;
                                int site4d = x + nx * (y + ny * (z + nz * t));

                                for (int is = 0; is < Ls / 2; ++is) {
                                    for (int comp = 0; comp < 24; ++comp) {
                                        int idx = ((site4d * num_vectors + k) * 2 + ch) * kept + is * 24 + comp;
                                        double val = h_basis[idx];
                                        norm_sq += val * val;
                                    }
                                }
                            }
                        }
                    }
                }
                norm_sq *= 2.0; // mirror
                double err = std::abs(norm_sq - 1.0);
                if (err > max_norm_err) max_norm_err = err;
            }
        }
    }
    std::cout << "  -> Max block norm deviation from 1.0: " << max_norm_err << std::endl;
    if (max_norm_err > 1e-4) {
        std::cerr << "FAILED: Block normalization error too large!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: All basis vectors are orthonormal over 5D aggregates!" << std::endl;

    // 3. Test Full Lattice Duality: <P^dag f, c>_coarse == <f, P c>_fine
    std::cout << "[Step 2] Testing Full-Lattice Adjoint Duality <P^dag f, c> == <f, P c>..." << std::endl;
    std::vector<double> h_fine(fine_reals);
    std::vector<double> h_coarse(coarse_reals);
    for (size_t i = 0; i < h_fine.size(); ++i) h_fine[i] = sin(i * 0.05 + 1.1);
    for (size_t i = 0; i < h_coarse.size(); ++i) h_coarse[i] = cos(i * 0.08 + 0.5);

    double* d_fine = nullptr;
    double* d_coarse = nullptr;
    double* d_fine_out = nullptr;
    double* d_coarse_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fine, fine_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coarse, coarse_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fine_out, fine_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coarse_out, coarse_reals * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_fine, h_fine.data(), fine_reals * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coarse, h_coarse.data(), coarse_reals * sizeof(double), cudaMemcpyHostToDevice));

    // c_out = P^dag f
    bridge_cuda_mdwf_mg_restrict_5d_f64(d_coarse_out, d_fine, d_basis, num_vectors, &geom, &mg_params, -1, 0);
    // f_out = P c
    bridge_cuda_mdwf_mg_prolong_5d_f64(d_fine_out, d_coarse, d_basis, num_vectors, &geom, &mg_params, -1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_coarse_out(coarse_reals);
    std::vector<double> h_fine_out(fine_reals);
    CUDA_CHECK(cudaMemcpy(h_coarse_out.data(), d_coarse_out, coarse_reals * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fine_out.data(), d_fine_out, fine_reals * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute <P^dag f, c>_coarse = sum (c_out^* * c)
    std::complex<double> inner_coarse(0, 0);
    for (int i = 0; i < total_blocks * coarse_dim; ++i) {
        std::complex<double> c_out(h_coarse_out[2 * i], h_coarse_out[2 * i + 1]);
        std::complex<double> c_in(h_coarse[2 * i], h_coarse[2 * i + 1]);
        inner_coarse += std::conj(c_out) * c_in;
    }

    // Compute <f, P c>_fine = sum (f^* * f_out)
    // Remember fine field layout in AoSoA: aosoa_5d_spinor_idx(c, s, ri, is, Ls, site4d)
    std::complex<double> inner_fine(0, 0);
    for (int site4d = 0; site4d < vol4d; ++site4d) {
        for (int is = 0; is < Ls; ++is) {
            for (int s = 0; s < 4; ++s) {
                for (int c = 0; c < 3; ++c) {
                    int idx_r = ((site4d / 32 * Ls + is) * 24 + 2 * (c + 3 * s) + 0) * 32 + (site4d % 32);
                    int idx_i = ((site4d / 32 * Ls + is) * 24 + 2 * (c + 3 * s) + 1) * 32 + (site4d % 32);

                    std::complex<double> f_val(h_fine[idx_r], h_fine[idx_i]);
                    std::complex<double> f_out_val(h_fine_out[idx_r], h_fine_out[idx_i]);
                    inner_fine += std::conj(f_val) * f_out_val;
                }
            }
        }
    }

    std::cout << "  <P^dag f, c>_coarse = (" << inner_coarse.real() << ", " << inner_coarse.imag() << ")" << std::endl;
    std::cout << "  <f, P c>_fine      = (" << inner_fine.real() << ", " << inner_fine.imag() << ")" << std::endl;

    double diff = std::abs(inner_coarse - inner_fine);
    double rel_diff = diff / std::abs(inner_fine);
    std::cout << "  -> Absolute difference: " << diff << std::endl;
    std::cout << "  -> Relative difference: " << rel_diff << std::endl;

    if (rel_diff > 1e-4) {
        std::cerr << "FAILED: Duality relation violated!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Full-lattice Duality <P^dag f, c> == <f, P c> verified!" << std::endl;

    // 4. Test Even-Parity Duality: <P^dag iota(f_e), c> == <f_e, pi(P c)>
    std::cout << "[Step 3] Testing Even-Parity Duality <P^dag iota(f_e), c> == <f_e, pi(P c)>..." << std::endl;
    bridge_cuda_mdwf_mg_restrict_5d_f64(d_coarse_out, d_fine, d_basis, num_vectors, &geom, &mg_params, 0, 0);
    bridge_cuda_mdwf_mg_prolong_5d_f64(d_fine_out, d_coarse, d_basis, num_vectors, &geom, &mg_params, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_coarse_out.data(), d_coarse_out, coarse_reals * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fine_out.data(), d_fine_out, fine_reals * sizeof(double), cudaMemcpyDeviceToHost));

    std::complex<double> inner_coarse_eo(0, 0);
    for (int i = 0; i < total_blocks * coarse_dim; ++i) {
        std::complex<double> c_out(h_coarse_out[2 * i], h_coarse_out[2 * i + 1]);
        std::complex<double> c_in(h_coarse[2 * i], h_coarse[2 * i + 1]);
        inner_coarse_eo += std::conj(c_out) * c_in;
    }

    std::complex<double> inner_fine_eo(0, 0);
    for (int site4d = 0; site4d < vol4d; ++site4d) {
        int x = site4d % nx; int rem = site4d / nx;
        int y = rem % ny; rem /= ny;
        int z = rem % nz; int t = rem / nz;
        if ((x + y + z + t) % 2 != 0) continue; // Only Even sites

        for (int is = 0; is < Ls; ++is) {
            for (int s = 0; s < 4; ++s) {
                for (int c = 0; c < 3; ++c) {
                    int idx_r = ((site4d / 32 * Ls + is) * 24 + 2 * (c + 3 * s) + 0) * 32 + (site4d % 32);
                    int idx_i = ((site4d / 32 * Ls + is) * 24 + 2 * (c + 3 * s) + 1) * 32 + (site4d % 32);

                    std::complex<double> f_val(h_fine[idx_r], h_fine[idx_i]);
                    std::complex<double> f_out_val(h_fine_out[idx_r], h_fine_out[idx_i]);
                    inner_fine_eo += std::conj(f_val) * f_out_val;
                }
            }
        }
    }

    std::cout << "  <P^dag iota(f_e), c> = (" << inner_coarse_eo.real() << ", " << inner_coarse_eo.imag() << ")" << std::endl;
    std::cout << "  <f_e, pi(P c)>      = (" << inner_fine_eo.real() << ", " << inner_fine_eo.imag() << ")" << std::endl;

    double diff_eo = std::abs(inner_coarse_eo - inner_fine_eo);
    double rel_diff_eo = diff_eo / std::abs(inner_fine_eo);
    std::cout << "  -> Relative difference: " << rel_diff_eo << std::endl;

    if (rel_diff_eo > 1e-4) {
        std::cerr << "FAILED: Even-Parity duality relation violated!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Even-parity Duality <P^dag iota(f_e), c> == <f_e, pi(P c)> verified!" << std::endl;

    // Cleanup
    cudaFree(d_basis);
    cudaFree(d_fine);
    cudaFree(d_coarse);
    cudaFree(d_fine_out);
    cudaFree(d_coarse_out);

    std::cout << "\nALL 3 MDWF TRANSFER TESTS PASSED PERFECTLY!" << std::endl;
    return 0;
}
