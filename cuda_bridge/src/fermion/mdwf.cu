#include "fermion/mdwf.cuh"
#include "fermion/fifth_gemm.cuh"
#include "bridge_cuda.h"
#include <algorithm>

namespace {
    constexpr int BLOCK_SIZE = 256;

    inline int get_grid_size(int total_elems) {
        int grid = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return std::min(grid, 1024);
    }

    inline int volume(const BridgeCudaGeometry* geom) {
        return geom->dims[0] * geom->dims[1] * geom->dims[2] * geom->dims[3];
    }

    // The limits of a block of threads: its threads (hops.cuh) and its shared memory.
    constexpr int MAX_THREADS = bridge_cuda::MaxThreads;
    constexpr size_t MAX_SHARED = 48 * 1024;

    using bridge_cuda::site_block;
    using bridge_cuda::site_grid;
}

extern "C" {

void bridge_cuda_mdwf_dslash_1pass_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, cudaStream_t stream)
{
    auto kernel = (vectors == 1) ? bridge_cuda::mdwf_dslash_1pass_kernel<float, false> : bridge_cuda::mdwf_dslash_1pass_kernel<float, true>;
    kernel<<<site_grid(volume(geom), geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls, vectors,
        static_cast<float>(params->M5), static_cast<float>(params->mass), static_cast<float>(params->b5), static_cast<float>(params->c5)
    );
}

void bridge_cuda_mdwf_dslash_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mdwf_dslash_1pass_f32(d_out, d_in, d_gauge, geom, params, 1, stream);
}

void bridge_cuda_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream)
{
    int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
    auto kernel = (vectors == 1) ? bridge_cuda::hop_kernel<float, false> : bridge_cuda::hop_kernel<float, true>;
    kernel<<<site_grid(sites, geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows, d_from,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls, vectors, target_parity,
        static_cast<float>(dagger ? -1 : 1), static_cast<float>(kappa),
        static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)
    );
}

void bridge_cuda_mdwf_fifth_f32(
    float* d_out, const float* d_x, double a1, double y1,
    const float* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_fifth_kernel<float><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_x, d_z, sites, Ls, static_cast<float>(mass),
        static_cast<float>(a1), static_cast<float>(y1), static_cast<float>(a2), static_cast<float>(y2), dagger
    );
}

void bridge_cuda_parity_copy_f32(
    float* d_half, float* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    int vectors, cudaStream_t stream)
{
    bridge_cuda::parity_copy_kernel<float><<<site_grid(volume(geom) / 2, 1, vectors), site_block(1, vectors), 0, stream>>>(
        d_half, d_full,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        geom->Ls, parity, to_half, vectors
    );
}

void bridge_cuda_gamma5_f32(
    float* d_out, const float* d_in, int sites, int Ls, int reflect, cudaStream_t stream)
{
    bridge_cuda::gamma5_kernel<float><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls, reflect
    );
}

void bridge_cuda_mdwf_dslash_1pass_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, cudaStream_t stream)
{
    auto kernel = (vectors == 1) ? bridge_cuda::mdwf_dslash_1pass_kernel<double, false> : bridge_cuda::mdwf_dslash_1pass_kernel<double, true>;
    kernel<<<site_grid(volume(geom), geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls, vectors,
        static_cast<double>(params->M5), static_cast<double>(params->mass), static_cast<double>(params->b5), static_cast<double>(params->c5)
    );
}

void bridge_cuda_mdwf_dslash_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    cudaStream_t stream)
{
    bridge_cuda_mdwf_dslash_1pass_f64(d_out, d_in, d_gauge, geom, params, 1, stream);
}

void bridge_cuda_hop_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream)
{
    int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
    auto kernel = (vectors == 1) ? bridge_cuda::hop_kernel<double, false> : bridge_cuda::hop_kernel<double, true>;
    kernel<<<site_grid(sites, geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_out, d_in, d_gauge, geom->link_rows, d_from,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls, vectors, target_parity,
        static_cast<double>(dagger ? -1 : 1), static_cast<double>(kappa),
        static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)
    );
}

void bridge_cuda_mdwf_fifth_f64(
    double* d_out, const double* d_x, double a1, double y1,
    const double* d_z, double a2, double y2,
    int sites, int Ls, double mass, int dagger, cudaStream_t stream)
{
    bridge_cuda::mdwf_fifth_kernel<double><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_x, d_z, sites, Ls, static_cast<double>(mass),
        static_cast<double>(a1), static_cast<double>(y1), static_cast<double>(a2), static_cast<double>(y2), dagger
    );
}

void bridge_cuda_parity_copy_f64(
    double* d_half, double* d_full, const BridgeCudaGeometry* geom, int parity, int to_half,
    int vectors, cudaStream_t stream)
{
    bridge_cuda::parity_copy_kernel<double><<<site_grid(volume(geom) / 2, 1, vectors), site_block(1, vectors), 0, stream>>>(
        d_half, d_full,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        geom->Ls, parity, to_half, vectors
    );
}

void bridge_cuda_gamma5_f64(
    double* d_out, const double* d_in, int sites, int Ls, int reflect, cudaStream_t stream)
{
    bridge_cuda::gamma5_kernel<double><<<get_grid_size(sites), BLOCK_SIZE, 0, stream>>>(
        d_out, d_in, sites, Ls, reflect
    );
}

void bridge_cuda_fifth_gemm_f32(
    float* d_out, const float* d_in, const float* d_from, const float* d_m,
    int sites, int Ls, cudaStream_t stream)
{
    int blocks = (sites + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize;
    int grid = get_grid_size(blocks * bridge_cuda::FifthColumns);
    // The padded matrix: 8 NT rows of K4 reals.
    int nt = (2 * Ls + 7) / 8, k4 = (2 * Ls + 3) / 4 * 4;
    size_t shared = static_cast<size_t>(8 * nt * k4) * sizeof(float);
    switch (nt) {
        case 1: bridge_cuda::fifth_gemm_columns_kernel<float, 1><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 2: bridge_cuda::fifth_gemm_columns_kernel<float, 2><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 3: bridge_cuda::fifth_gemm_columns_kernel<float, 3><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 4: bridge_cuda::fifth_gemm_columns_kernel<float, 4><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 5: bridge_cuda::fifth_gemm_columns_kernel<float, 5><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 6: bridge_cuda::fifth_gemm_columns_kernel<float, 6><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 7: bridge_cuda::fifth_gemm_columns_kernel<float, 7><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 8: bridge_cuda::fifth_gemm_columns_kernel<float, 8><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 9: bridge_cuda::fifth_gemm_columns_kernel<float, 9><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 10: bridge_cuda::fifth_gemm_columns_kernel<float, 10><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 11: bridge_cuda::fifth_gemm_columns_kernel<float, 11><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 12: bridge_cuda::fifth_gemm_columns_kernel<float, 12><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 13: bridge_cuda::fifth_gemm_columns_kernel<float, 13><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 14: bridge_cuda::fifth_gemm_columns_kernel<float, 14><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 15: bridge_cuda::fifth_gemm_columns_kernel<float, 15><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 16: bridge_cuda::fifth_gemm_columns_kernel<float, 16><<<grid, BLOCK_SIZE, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        default: break;  // Ls is at most 64
    }
}

void bridge_cuda_fifth_gemm_f64(
    double* d_out, const double* d_in, const double* d_from, const double* d_m,
    int sites, int Ls, cudaStream_t stream)
{
    int blocks = (sites + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize;
    // A warp for every strip of 8 columns of every block, 8 warps to a block of threads.
    int grid = (blocks * bridge_cuda::FifthStrips + 7) / 8;
    // The copy of the matrix in the shared memory: 8 NT rows of K16 reals.
    int nt = (2 * Ls + 7) / 8, k16 = (2 * Ls + 15) / 16 * 16;
    size_t shared = static_cast<size_t>(8 * nt * k16) * sizeof(double);
    switch (nt) {
        case 1: bridge_cuda::fifth_gemm_mma_kernel<1><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 2: bridge_cuda::fifth_gemm_mma_kernel<2><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 3: bridge_cuda::fifth_gemm_mma_kernel<3><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 4: bridge_cuda::fifth_gemm_mma_kernel<4><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 5: bridge_cuda::fifth_gemm_mma_kernel<5><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 6: bridge_cuda::fifth_gemm_mma_kernel<6><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 7: bridge_cuda::fifth_gemm_mma_kernel<7><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 8: bridge_cuda::fifth_gemm_mma_kernel<8><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 9: bridge_cuda::fifth_gemm_mma_kernel<9><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 10: bridge_cuda::fifth_gemm_mma_kernel<10><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 11: bridge_cuda::fifth_gemm_mma_kernel<11><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 12: bridge_cuda::fifth_gemm_mma_kernel<12><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 13: bridge_cuda::fifth_gemm_mma_kernel<13><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 14: bridge_cuda::fifth_gemm_mma_kernel<14><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 15: bridge_cuda::fifth_gemm_mma_kernel<15><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 16: bridge_cuda::fifth_gemm_mma_kernel<16><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        default: break;  // Ls is at most 64
    }
}

int bridge_cuda_hop_gemm_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_m,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream)
{
    int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
    int Ls = geom->Ls, nt = (2 * Ls + 7) / 8;
    // The sites of a block of threads: as many as its threads and its shared memory hold,
    // with whole warps where the tensor cores are used.
    const size_t matrix = static_cast<size_t>(8 * nt * ((2 * Ls + 3) / 4 * 4)) * sizeof(float);
    int lanes = bridge_cuda::WarpSize;
    for (; lanes >= 4; lanes /= 2) {
        const size_t tile = static_cast<size_t>(2 * Ls) * 12 * lanes * sizeof(float);
        if (lanes * Ls <= MAX_THREADS && tile + matrix <= MAX_SHARED && (!false || (lanes * Ls) % 32 == 0)) break;
    }
    if (lanes < 4 || nt > 8) return 0;
    const size_t shared = static_cast<size_t>(2 * Ls) * 12 * lanes * sizeof(float) + matrix;
    dim3 block(lanes, Ls);
    dim3 grid((sites + lanes - 1) / lanes, vectors);
    const float bc[4] = {static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
                        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3])};
    const float sign = dagger ? -1 : 1;
    switch (nt) {
        case 1: bridge_cuda::hop_gemm_kernel<float, 1><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 2: bridge_cuda::hop_gemm_kernel<float, 2><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 3: bridge_cuda::hop_gemm_kernel<float, 3><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 4: bridge_cuda::hop_gemm_kernel<float, 4><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 5: bridge_cuda::hop_gemm_kernel<float, 5><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 6: bridge_cuda::hop_gemm_kernel<float, 6><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 7: bridge_cuda::hop_gemm_kernel<float, 7><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        case 8: bridge_cuda::hop_gemm_kernel<float, 8><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)); break;
        default: return 0;
    }
    return 1;
}

int bridge_cuda_hop_gemm_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_m,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream)
{
    int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
    int Ls = geom->Ls, nt = (2 * Ls + 7) / 8;
    // The sites of a block of threads: as many as its threads and its shared memory hold,
    // with whole warps where the tensor cores are used.
    const size_t matrix = static_cast<size_t>(8 * nt * ((2 * Ls + 15) / 16 * 16)) * sizeof(double);
    int lanes = bridge_cuda::WarpSize;
    for (; lanes >= 4; lanes /= 2) {
        const size_t tile = static_cast<size_t>(2 * Ls) * 12 * lanes * sizeof(double);
        if (lanes * Ls <= MAX_THREADS && tile + matrix <= MAX_SHARED && (!true || (lanes * Ls) % 32 == 0)) break;
    }
    if (lanes < 4 || nt > 8) return 0;
    const size_t shared = static_cast<size_t>(2 * Ls) * 12 * lanes * sizeof(double) + matrix;
    dim3 block(lanes, Ls);
    dim3 grid((sites + lanes - 1) / lanes, vectors);
    const double bc[4] = {static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
                        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3])};
    const double sign = dagger ? -1 : 1;
    switch (nt) {
        case 1: bridge_cuda::hop_gemm_kernel<double, 1><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 2: bridge_cuda::hop_gemm_kernel<double, 2><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 3: bridge_cuda::hop_gemm_kernel<double, 3><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 4: bridge_cuda::hop_gemm_kernel<double, 4><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 5: bridge_cuda::hop_gemm_kernel<double, 5><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 6: bridge_cuda::hop_gemm_kernel<double, 6><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 7: bridge_cuda::hop_gemm_kernel<double, 7><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        case 8: bridge_cuda::hop_gemm_kernel<double, 8><<<grid, block, shared, stream>>>(d_out, d_in, d_gauge, geom->link_rows, d_from, d_m, geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3], bc[0], bc[1], bc[2], bc[3], Ls, target_parity, sign, static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)); break;
        default: return 0;
    }
    return 1;
}

} // extern "C"
