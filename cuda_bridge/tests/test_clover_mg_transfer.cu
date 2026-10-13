#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include "bridge_cuda.h"
#include "base/aosoa.cuh"

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
    std::cout << " Test: Clover/Wilson 4D Unified MG Transfer (Ls = 1)   " << std::endl;
    std::cout << "======================================================" << std::endl;

    const int nx = 8, ny = 8, nz = 8, nt = 8;
    const int Ls = 1;
    const int vol4d = nx * ny * nz * nt; // 4096
    const int fine_reals = 24 * vol4d;

    const int bx = 2, by = 2, bz = 2, bt = 2;
    const int nbx = nx / bx;
    const int nby = ny / by;
    const int nbz = nz / bz;
    const int nbt = nt / bt;
    const int total_blocks = nbx * nby * nbz * nbt; // 256 coarse sites

    const int num_vectors = 6; // 6 test vectors -> 12 coarse colors (+ and -)
    const int coarse_dim = 2 * num_vectors;
    const int coarse_reals = total_blocks * coarse_dim * 2;

    const int kept = 24; // 1 spinor for 4D
    const int basis_reals = vol4d * num_vectors * 2 * kept;

    BridgeCudaGeometry geom;
    geom.dims[0] = nx; geom.dims[1] = ny; geom.dims[2] = nz; geom.dims[3] = nt;
    geom.Ls = Ls;
    geom.vol = vol4d;
    geom.vol5d = vol4d;
    geom.vol_pad = ((vol4d + 31) / 32) * 32;
    geom.gamma_rep = BRIDGE_CUDA_GAMMA_CHIRAL;

    BridgeCudaMgParams mg_params;
    mg_params.num_colors = coarse_dim;
    mg_params.block_size[0] = bx;
    mg_params.block_size[1] = by;
    mg_params.block_size[2] = bz;
    mg_params.block_size[3] = bt;

    // 1. Initialize independent random basis vectors using standard PRNG (LCG)
    std::vector<float> h_basis(basis_reals);
    uint32_t seed = 12345;
    for (int site4d = 0; site4d < vol4d; ++site4d) {
        for (int v = 0; v < num_vectors; ++v) {
            for (int ch = 0; ch < 2; ++ch) {
                for (int comp = 0; comp < kept; ++comp) {
                    seed = seed * 1664525u + 1013904223u;
                    int idx = ((site4d * num_vectors + v) * 2 + ch) * kept + comp;
                    h_basis[idx] = (static_cast<float>(seed) / 4294967296.0f) * 2.0f - 1.0f;
                }
            }
        }
    }

    float* d_basis = nullptr;
    CUDA_CHECK(cudaMalloc(&d_basis, basis_reals * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_basis, h_basis.data(), basis_reals * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Perform Block Orthonormalization on GPU for Ls = 1
    std::cout << "[Step 1] Running 4D Block Modified Gram-Schmidt..." << std::endl;
    bridge_cuda_mg_block_normalize_f32(d_basis, num_vectors, &geom, &mg_params, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_basis.data(), d_basis, basis_reals * sizeof(float), cudaMemcpyHostToDevice));

    // Check block norm on host
    double max_norm_err = 0.0;
    for (int b = 0; b < total_blocks; ++b) {
        int bx_idx = b % nbx; int rem = b / nbx;
        int by_idx = rem % nby; rem /= nby;
        int bz_idx = rem % nbz;
        int bt_idx = rem / nbz;

        for (int ch = 0; ch < 2; ++ch) {
            for (int v1 = 0; v1 < num_vectors; ++v1) {
                for (int v2 = 0; v2 < num_vectors; ++v2) {
                    double dot_r = 0.0, dot_i = 0.0;
                    for (int lt = 0; lt < bt; ++lt) {
                        int t = bt_idx * bt + lt;
                        for (int lz = 0; lz < bz; ++lz) {
                            int z = bz_idx * bz + lz;
                            for (int ly = 0; ly < by; ++ly) {
                                int y = by_idx * by + ly;
                                for (int lx = 0; lx < bx; ++lx) {
                                    int x = bx_idx * bx + lx;
                                    int site4d = x + nx * (y + ny * (z + nz * t));

                                    for (int sp = 0; sp < 12; ++sp) {
                                        int idx1_r = ((site4d * num_vectors + v1) * 2 + ch) * kept + (2 * sp + 0);
                                        int idx1_i = ((site4d * num_vectors + v1) * 2 + ch) * kept + (2 * sp + 1);
                                        int idx2_r = ((site4d * num_vectors + v2) * 2 + ch) * kept + (2 * sp + 0);
                                        int idx2_i = ((site4d * num_vectors + v2) * 2 + ch) * kept + (2 * sp + 1);

                                        double v1r = h_basis[idx1_r], v1i = h_basis[idx1_i];
                                        double v2r = h_basis[idx2_r], v2i = h_basis[idx2_i];

                                        // dot += v1^* * v2
                                        dot_r += (v1r * v2r + v1i * v2i);
                                        dot_i += (v1r * v2i - v1i * v2r);
                                    }
                                }
                            }
                        }
                    }

                    double target_r = (v1 == v2) ? 1.0 : 0.0;
                    double target_i = 0.0;
                    double err = std::sqrt((dot_r - target_r) * (dot_r - target_r) + (dot_i - target_i) * (dot_i - target_i));
                    if (err > max_norm_err) max_norm_err = err;
                }
            }
        }
    }

    std::cout << "  -> Max 4D block orthonormal deviation: " << max_norm_err << std::endl;
    if (max_norm_err > 1e-4) {
        std::cerr << "FAIL: Block orthonormalization error too large!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: All 4D basis vectors are orthonormal over aggregates!" << std::endl;

    // 3. Test Full-Lattice Duality: <P^dag f, c>_coarse == <f, P c>_fine in FP64
    std::cout << "[Step 2] Testing 4D Full-Lattice Adjoint Duality <P^dag f, c> == <f, P c>..." << std::endl;
    std::vector<double> h_fine(fine_reals), h_coarse(coarse_reals);
    for (size_t i = 0; i < h_fine.size(); ++i) h_fine[i] = sin(i * 0.07 + 0.1);
    for (size_t i = 0; i < h_coarse.size(); ++i) h_coarse[i] = cos(i * 0.11 + 0.2);

    double* d_fine = nullptr;
    double* d_coarse = nullptr;
    double* d_res_coarse = nullptr;
    double* d_res_fine = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fine, fine_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coarse, coarse_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_res_coarse, coarse_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_res_fine, fine_reals * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_fine, h_fine.data(), fine_reals * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coarse, h_coarse.data(), coarse_reals * sizeof(double), cudaMemcpyHostToDevice));

    // res_coarse = P^dag * fine
    bridge_cuda_mg_restrict_f64(d_res_coarse, d_fine, d_basis, num_vectors, &geom, &mg_params, 0);

    // res_fine = P * coarse
    bridge_cuda_mg_prolong_f64(d_res_fine, d_coarse, d_basis, num_vectors, &geom, &mg_params, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_res_coarse(coarse_reals), h_res_fine(fine_reals);
    CUDA_CHECK(cudaMemcpy(h_res_coarse.data(), d_res_coarse, coarse_reals * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_fine.data(), d_res_fine, fine_reals * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute coarse dot: <res_coarse, c> = sum (res_coarse)^* * c
    std::complex<double> coarse_dot(0, 0);
    for (int i = 0; i < coarse_reals / 2; ++i) {
        std::complex<double> r(h_res_coarse[2 * i + 0], h_res_coarse[2 * i + 1]);
        std::complex<double> c(h_coarse[2 * i + 0], h_coarse[2 * i + 1]);
        coarse_dot += std::conj(r) * c;
    }

    // Compute fine dot: <f, res_fine> = sum f^* * res_fine
    std::complex<double> fine_dot(0, 0);
    for (int site4d = 0; site4d < vol4d; ++site4d) {
        for (int s = 0; s < 4; ++s) {
            for (int c = 0; c < 3; ++c) {
                int idx_r = bridge_cuda::aosoa_spinor_idx(c, s, 0, site4d);
                int idx_i = bridge_cuda::aosoa_spinor_idx(c, s, 1, site4d);

                std::complex<double> f(h_fine[idx_r], h_fine[idx_i]);
                std::complex<double> p(h_res_fine[idx_r], h_res_fine[idx_i]);
                fine_dot += std::conj(f) * p;
            }
        }
    }

    double diff = std::abs(coarse_dot - fine_dot);
    double rel_diff = diff / std::abs(fine_dot);
    std::cout << "  <P^dag f, c>_coarse = (" << coarse_dot.real() << ", " << coarse_dot.imag() << ")" << std::endl;
    std::cout << "  <f, P c>_fine      = (" << fine_dot.real() << ", " << fine_dot.imag() << ")" << std::endl;
    std::cout << "  -> Absolute difference: " << diff << std::endl;
    std::cout << "  -> Relative difference: " << rel_diff << std::endl;

    if (rel_diff > 1e-13) {
        std::cerr << "FAIL: 4D Duality relation failed!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: 4D Full-lattice Duality <P^dag f, c> == <f, P c> verified!" << std::endl;

    // 4. Test Even-Parity Duality: <P^dag iota(f_e), c> == <f_e, pi(P c)>
    std::cout << "[Step 3] Testing 4D Even-Parity Duality <P^dag iota(f_e), c> == <f_e, pi(P c)>..." << std::endl;
    bridge_cuda_mg_restrict_unified_f64(d_res_coarse, d_fine, d_basis, num_vectors, &geom, &mg_params, 0, 0);
    bridge_cuda_mg_prolong_unified_f64(d_res_fine, d_coarse, d_basis, num_vectors, &geom, &mg_params, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_res_coarse.data(), d_res_coarse, coarse_reals * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_fine.data(), d_res_fine, fine_reals * sizeof(double), cudaMemcpyDeviceToHost));

    std::complex<double> eo_coarse_dot(0, 0);
    for (int i = 0; i < coarse_reals / 2; ++i) {
        std::complex<double> r(h_res_coarse[2 * i + 0], h_res_coarse[2 * i + 1]);
        std::complex<double> c(h_coarse[2 * i + 0], h_coarse[2 * i + 1]);
        eo_coarse_dot += std::conj(r) * c;
    }

    std::complex<double> eo_fine_dot(0, 0);
    for (int site4d = 0; site4d < vol4d; ++site4d) {
        int x = site4d % nx; int rem = site4d / nx;
        int y = rem % ny; rem /= ny;
        int z = rem % nz;
        int t = rem / nz;
        if (((x + y + z + t) & 1) != 0) continue; // Only even sites

        for (int s = 0; s < 4; ++s) {
            for (int c = 0; c < 3; ++c) {
                int idx_r = bridge_cuda::aosoa_spinor_idx(c, s, 0, site4d);
                int idx_i = bridge_cuda::aosoa_spinor_idx(c, s, 1, site4d);

                std::complex<double> f(h_fine[idx_r], h_fine[idx_i]);
                std::complex<double> p(h_res_fine[idx_r], h_res_fine[idx_i]);
                eo_fine_dot += std::conj(f) * p;
            }
        }
    }

    double eo_diff = std::abs(eo_coarse_dot - eo_fine_dot);
    double eo_rel = eo_diff / std::abs(eo_fine_dot);
    std::cout << "  <P^dag iota(f_e), c> = (" << eo_coarse_dot.real() << ", " << eo_coarse_dot.imag() << ")" << std::endl;
    std::cout << "  <f_e, pi(P c)>      = (" << eo_fine_dot.real() << ", " << eo_fine_dot.imag() << ")" << std::endl;
    std::cout << "  -> Relative difference: " << eo_rel << std::endl;

    if (eo_rel > 1e-13) {
        std::cerr << "FAIL: 4D Even-parity duality failed!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: 4D Even-parity Duality <P^dag iota(f_e), c> == <f_e, pi(P c)> verified!" << std::endl;

    std::cout << "\nALL 4D CLOVER/WILSON MG TRANSFER TESTS PASSED PERFECTLY!" << std::endl;

    CUDA_CHECK(cudaFree(d_basis));
    CUDA_CHECK(cudaFree(d_fine));
    CUDA_CHECK(cudaFree(d_coarse));
    CUDA_CHECK(cudaFree(d_res_coarse));
    CUDA_CHECK(cudaFree(d_res_fine));
    return 0;
}
