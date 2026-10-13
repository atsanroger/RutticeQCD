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
    std::cout << " Test: Fused 1-Pass Mobius Domain Wall Dslash Kernel  " << std::endl;
    std::cout << "======================================================" << std::endl;

    // Lattice geometry: 8^4 with Ls = 8
    const int nx = 8, ny = 8, nz = 8, nt = 8;
    const int Ls = 8;
    const int vol4d = nx * ny * nz * nt; // 4096
    const int vol5d = vol4d * Ls;       // 32768
    const int spinor_reals = 24 * vol5d;
    const int gauge_reals  = 72 * vol4d;

    BridgeCudaGeometry geom;
    geom.dims[0] = nx;
    geom.dims[1] = ny;
    geom.dims[2] = nz;
    geom.dims[3] = nt;
    geom.Ls = Ls;
    geom.vol = vol4d;
    geom.vol5d = vol5d;
    geom.vol_pad = ((vol4d + 31) / 32) * 32;
    geom.gamma_rep = BRIDGE_CUDA_GAMMA_CHIRAL;

    BridgeCudaMdwfParams params;
    params.M5 = 1.6;
    params.mass = 0.05;
    params.alpha = 1.0;
    params.b5 = 1.5;
    params.c5 = 0.5;

    // Allocate host memory
    std::vector<double> h_in(spinor_reals, 1.0);
    std::vector<double> h_gauge(gauge_reals, 0.0);
    std::vector<double> h_out(spinor_reals, 0.0);

    // Set gauge field to unit matrix (U = identity)
    for (int site = 0; site < vol4d; ++site) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int r = 0; r < 3; ++r) {
                int comp_diag = 2 * (r + 3 * (r + 3 * mu)); // diagonal real component
                int idx = ( (site / 32) * 72 + comp_diag ) * 32 + (site % 32);
                h_gauge[idx] = 1.0;
            }
        }
    }

    // Allocate device memory
    double *d_in = nullptr;
    double *d_gauge = nullptr;
    double *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, spinor_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gauge, gauge_reals * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, spinor_reals * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), spinor_reals * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gauge, h_gauge.data(), gauge_reals * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, spinor_reals * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up run
    bridge_cuda_mdwf_dslash_1pass_f64(d_out, d_in, d_gauge, &geom, &params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int n_iter = 50;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < n_iter; ++i) {
        bridge_cuda_mdwf_dslash_1pass_f64(d_out, d_in, d_gauge, &geom, &params, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / n_iter;

    // Check output data
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, spinor_reals * sizeof(double), cudaMemcpyDeviceToHost));

    double norm_out = 0.0;
    bool has_nan = false;
    for (int i = 0; i < spinor_reals; ++i) {
        norm_out += h_out[i] * h_out[i];
        if (std::isnan(h_out[i]) || std::isinf(h_out[i])) {
            has_nan = true;
        }
    }
    norm_out = std::sqrt(norm_out);

    std::cout << "Lattice: " << nx << "^4 x " << Ls << " (Ls=" << Ls << ")" << std::endl;
    std::cout << "Params: M5=" << params.M5 << ", mass=" << params.mass 
              << ", b5=" << params.b5 << ", c5=" << params.c5 << std::endl;
    std::cout << "Avg execution time: " << avg_ms << " ms" << std::endl;

    // FLOPs and Bandwidth calculations
    // Nominal FLOPs: (464 * Nc + 240) * Ls * vol4d = 1632 * 8 * 4096 = 53,477,376 FLOPs per sweep
    double gflops_per_sweep = 1632.0 * Ls * vol4d * 1e-9;
    double tflops_achieved = gflops_per_sweep / (avg_ms * 1e-3) * 1e-3;
    
    // 1-Pass compulsory memory traffic: (48 * Ls + 72) * vol4d * 8 bytes
    double bytes_per_sweep = (48.0 * Ls + 72.0) * vol4d * sizeof(double);
    double bw_achieved = (bytes_per_sweep * 1e-9) / (avg_ms * 1e-3);

    std::cout << "Output 2-norm: " << norm_out << " (NaN/Inf: " << (has_nan ? "FAIL" : "NONE") << ")" << std::endl;
    std::cout << "Compute throughput: " << tflops_achieved * 1e3 << " GFLOPS (" << tflops_achieved << " TFLOPS)" << std::endl;
    std::cout << "Effective Memory Traffic: " << bw_achieved << " GB/s" << std::endl;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_gauge));
    CUDA_CHECK(cudaFree(d_out));

    if (has_nan || norm_out == 0.0) {
        std::cerr << "Verification FAILED!" << std::endl;
        return 1;
    }

    std::cout << "Verification PASSED: 1-Pass MDWF kernel completed successfully!" << std::endl;
    std::cout << "======================================================" << std::endl;
    return 0;
}
