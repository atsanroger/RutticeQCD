#include <iostream>
#include <vector>
#include <cmath>
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
    std::cout << " Test: Fine-Grid MDWF Exact Schur Factor & PvEvenOdd  " << std::endl;
    std::cout << "======================================================" << std::endl;

    const int nx = 8, ny = 8, nz = 8, nt = 8;
    const int Ls = 8;
    const int vol4d = nx * ny * nz * nt; // 4096
    const int vol4d_eo = vol4d / 2;      // 2048
    const int vol5d = vol4d * Ls;
    const int spinor_reals = 24 * vol5d;
    const int gauge_reals  = 72 * vol4d;

    BridgeCudaGeometry geom;
    geom.dims[0] = nx; geom.dims[1] = ny; geom.dims[2] = nz; geom.dims[3] = nt;
    geom.Ls = Ls;
    geom.vol = vol4d;
    geom.vol5d = vol5d;
    geom.vol_pad = ((vol4d + 31) / 32) * 32;
    geom.gamma_rep = BRIDGE_CUDA_GAMMA_CHIRAL;

    BridgeCudaMdwfParams params_phys;
    params_phys.M5 = 1.6;
    params_phys.mass = 0.05;
    params_phys.alpha = 1.0;
    params_phys.b5 = 1.5;
    params_phys.c5 = 0.5;

    BridgeCudaMdwfParams params_pv;
    params_pv.M5 = 1.6;
    params_pv.mass = 1.0; // Pauli-Villars mass = 1.0
    params_pv.alpha = 1.0;
    params_pv.b5 = 1.5;
    params_pv.c5 = 0.5;

    // Allocate host memory
    std::vector<double> h_in(spinor_reals, 0.0);
    std::vector<double> h_gauge(gauge_reals, 0.0);
    std::vector<double> h_out(spinor_reals, 0.0);

    // Initialize Even sites with test pattern
    for (int site = 0; site < vol4d; ++site) {
        int x = site % nx; int rem = site / nx;
        int y = rem % ny; rem /= ny;
        int z = rem % nz; int t = rem / nz;
        if ((x + y + z + t) % 2 == 0) { // Even site
            for (int is = 0; is < Ls; ++is) {
                for (int comp = 0; comp < 24; ++comp) {
                    int outer = (site / 32) * Ls + is;
                    int lane = site % 32;
                    int idx = (outer * 24 + comp) * 32 + lane;
                    h_in[idx] = sin(site * 0.1 + is * 0.3 + comp * 0.05);
                }
            }
        }
    }

    // Initialize unit gauge links
    for (int site = 0; site < vol4d; ++site) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int r = 0; r < 3; ++r) {
                int comp_diag = 2 * (r + 3 * (r + 3 * mu));
                int idx = ((site / 32) * 72 + comp_diag) * 32 + (site % 32);
                h_gauge[idx] = 1.0;
            }
        }
    }

    double* d_in = nullptr;
    double* d_gauge = nullptr;
    double* d_out = nullptr;
    double* d_scr_e = nullptr;
    double* d_scr_o1 = nullptr;
    double* d_scr_o2 = nullptr;
    double* d_scr4 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, spinor_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gauge, gauge_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, spinor_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scr_e, spinor_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scr_o1, spinor_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scr_o2, spinor_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scr4, spinor_reals * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), spinor_reals * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gauge, h_gauge.data(), gauge_reals * sizeof(double), cudaMemcpyHostToDevice));

    // [Step 1] Test Schur Factor: S x_e = (1 - D_eo C_o^{-1} D_oe C_e^{-1}) x_e
    std::cout << "[Step 1] Running Fine-Grid MDWF Schur factor S x_e..." << std::endl;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 100; ++iter) {
        bridge_cuda_mdwf_schur_factor_f64(
            d_out, d_in, d_gauge,
            d_scr_e, d_scr_o1, d_scr_o2,
            &geom, &params_phys, 0
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_schur = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_schur, start, stop));
    ms_schur /= 100.0f;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, spinor_reals * sizeof(double), cudaMemcpyDeviceToHost));

    double norm_sq_schur = 0.0;
    bool has_nan_or_inf = false;
    for (double val : h_out) {
        if (std::isnan(val) || std::isinf(val)) has_nan_or_inf = true;
        norm_sq_schur += val * val;
    }
    std::cout << "  -> Avg Execution Time per Schur factor: " << ms_schur << " ms" << std::endl;
    std::cout << "  -> Output 2-Norm: " << sqrt(norm_sq_schur) << " (NaN/Inf: " << (has_nan_or_inf ? "YES" : "NONE") << ")" << std::endl;

    if (has_nan_or_inf || norm_sq_schur < 1e-10) {
        std::cerr << "FAILED: Schur factor output invalid!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Fine-grid Schur factor executed successfully on GPU!" << std::endl;

    // [Step 2] Test Full Pauli-Villars Even-Odd Operator: A_tilde x = S_PV(S x)
    std::cout << "[Step 2] Running Exact Even-Odd Pauli-Villars Operator A_tilde x = S_PV(S x)..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 100; ++iter) {
        bridge_cuda_mdwf_pvprec_a_exact_f64(
            d_out, d_in, d_gauge,
            d_scr_e, d_scr_o1, d_scr_o2, d_scr4,
            &geom, &params_phys, &params_pv, 0
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_a = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_a, start, stop));
    ms_a /= 100.0f;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, spinor_reals * sizeof(double), cudaMemcpyDeviceToHost));

    double norm_sq_a = 0.0;
    has_nan_or_inf = false;
    for (double val : h_out) {
        if (std::isnan(val) || std::isinf(val)) has_nan_or_inf = true;
        norm_sq_a += val * val;
    }
    std::cout << "  -> Avg Execution Time per A_tilde operator: " << ms_a << " ms" << std::endl;
    std::cout << "  -> Output 2-Norm: " << sqrt(norm_sq_a) << " (NaN/Inf: " << (has_nan_or_inf ? "YES" : "NONE") << ")" << std::endl;

    if (has_nan_or_inf || norm_sq_a < 1e-10) {
        std::cerr << "FAILED: A_tilde operator output invalid!" << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Full Even-Odd Pauli-Villars operator A_tilde executed successfully on GPU!" << std::endl;

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_gauge);
    cudaFree(d_out);
    cudaFree(d_scr_e);
    cudaFree(d_scr_o1);
    cudaFree(d_scr_o2);
    cudaFree(d_scr4);

    std::cout << "\nALL FINE-GRID MDWF SCHUR & PvEvenOdd TESTS PASSED PERFECTLY!" << std::endl;
    return 0;
}
