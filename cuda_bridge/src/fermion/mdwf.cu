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

    template <typename Float>
    bridge_cuda::Halo<Float> halo_of(const BridgeCudaHalo* h) {
        bridge_cuda::Halo<Float> halo{};
        for (int f = 0; f < 8; ++f) {
            halo.recv[f] = static_cast<const Float*>(h->recv[f]);
            halo.bc[f] = static_cast<Float>(h->bc[f]);
        }
        for (int mu = 0; mu < 4; ++mu) halo.cut[mu] = h->cut[mu];
        halo.surface = h->surface;
        halo.surface_sites = h->surface_sites;
        return halo;
    }

    // The hops (hop_kernel) on the interior (part 1) or the surface (part 2) of the output.
    // The instance of hop_kernel for part P of the output: one vector or several; the clover
    // operator, another four-dimensional one (Ls 1, b 1, c 0), or a five-dimensional one.
    template <typename Float, int P>
    auto hop_instance(int vectors, bool clover, bool four_d)
    {
        using namespace bridge_cuda;
        if (clover) return (vectors == 1) ? hop_kernel<Float, false, P, true, true> : hop_kernel<Float, true, P, true, true>;
        if (four_d) return (vectors == 1) ? hop_kernel<Float, false, P, false, true> : hop_kernel<Float, true, P, false, true>;
        return (vectors == 1) ? hop_kernel<Float, false, P> : hop_kernel<Float, true, P>;
    }

    // The hops (hop_kernel) on the whole output (part 0, h null), its interior (part 1) or its
    // surface (part 2); with the blocks d_clover, the clover operator T d_from + H d_in.
    template <typename Float>
    void hop_part(
        Float* d_out, const Float* d_in, const Float* d_gauge, const Float* d_from,
        const Float* d_clover, const BridgeCudaGeometry* geom, double kappa, double mass,
        double b5, double c5, int dagger, int target_parity, int vectors,
        const BridgeCudaHalo* h, int part, cudaStream_t stream)
    {
        using namespace bridge_cuda;
        const int sites = (target_parity >= 0) ? volume(geom) / 2 : volume(geom);
        const int threads = (part == Surface) ? h->surface_sites : sites;
        if (threads == 0) return;
        const bool clover = d_clover != nullptr;
        const bool four_d = geom->Ls == 1 && b5 == 1.0 && c5 == 0.0;
        auto kernel = (part == Surface) ? hop_instance<Float, Surface>(vectors, clover, four_d)
            : (part == Interior) ? hop_instance<Float, Interior>(vectors, clover, four_d)
            : hop_instance<Float, Whole>(vectors, clover, four_d);
        kernel<<<site_grid(threads, geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
            d_out, d_in, d_gauge, geom->link_rows, d_from,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            static_cast<Float>(geom->bc[0]), static_cast<Float>(geom->bc[1]),
            static_cast<Float>(geom->bc[2]), static_cast<Float>(geom->bc[3]),
            geom->Ls, vectors, target_parity,
            static_cast<Float>(dagger ? -1 : 1), static_cast<Float>(kappa),
            static_cast<Float>(mass), static_cast<Float>(b5), static_cast<Float>(c5),
            (h != nullptr) ? halo_of<Float>(h) : Halo<Float>{}, d_clover);
    }

    // D in one pass (mdwf_dslash_1pass_kernel) on the interior or the surface.
    template <typename Float>
    void dslash_part(
        Float* d_out, const Float* d_in, const Float* d_gauge,
        const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
        int vectors, const BridgeCudaHalo* h, int part, cudaStream_t stream)
    {
        using namespace bridge_cuda;
        const int threads = (part == Surface) ? h->surface_sites : volume(geom);
        if (threads == 0) return;
        auto kernel = (part == Surface)
            ? ((vectors == 1) ? mdwf_dslash_1pass_kernel<Float, false, Surface> : mdwf_dslash_1pass_kernel<Float, true, Surface>)
            : ((vectors == 1) ? mdwf_dslash_1pass_kernel<Float, false, Interior> : mdwf_dslash_1pass_kernel<Float, true, Interior>);
        kernel<<<site_grid(threads, geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
            d_out, d_in, d_gauge, geom->link_rows,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            static_cast<Float>(geom->bc[0]), static_cast<Float>(geom->bc[1]),
            static_cast<Float>(geom->bc[2]), static_cast<Float>(geom->bc[3]),
            geom->Ls, vectors,
            static_cast<Float>(params->M5), static_cast<Float>(params->mass),
            static_cast<Float>(params->b5), static_cast<Float>(params->c5),
            halo_of<Float>(h));
    }

    // The buffer of one face (hop_pack_kernel).
    template <typename Float>
    void hop_pack(
        Float* d_send, const Float* d_in, const Float* d_gauge,
        const BridgeCudaGeometry* geom, double mass, double b5, double c5,
        int dagger, int in_parity, int face, int vectors, cudaStream_t stream)
    {
        using namespace bridge_cuda;
        const int sites = face_sites(face % 4, geom->dims, in_parity >= 0);
        const int fibres = geom->Ls * vectors;
        const dim3 block(32, 8);
        const dim3 grid((sites + block.x - 1) / block.x, (fibres + block.y - 1) / block.y);
        hop_pack_kernel<Float><<<grid, block, 0, stream>>>(
            d_send, d_in, d_gauge, geom->link_rows,
            geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
            geom->Ls, vectors, in_parity, face,
            static_cast<Float>(dagger ? -1 : 1),
            static_cast<Float>(mass), static_cast<Float>(b5), static_cast<Float>(c5));
    }
}

extern "C" {

void bridge_cuda_hop_part_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, const BridgeCudaHalo* halo, int part,
    cudaStream_t stream)
{
    hop_part<float>(d_out, d_in, d_gauge, d_from, d_clover, geom, kappa, mass, b5, c5, dagger,
                    target_parity, vectors, halo, part, stream);
}

void bridge_cuda_hop_part_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, const BridgeCudaHalo* halo, int part,
    cudaStream_t stream)
{
    hop_part<double>(d_out, d_in, d_gauge, d_from, d_clover, geom, kappa, mass, b5, c5, dagger,
                     target_parity, vectors, halo, part, stream);
}

void bridge_cuda_mdwf_dslash_part_f32(
    float* d_out, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, const BridgeCudaHalo* halo, int part, cudaStream_t stream)
{
    dslash_part<float>(d_out, d_in, d_gauge, geom, params, vectors, halo, part, stream);
}

void bridge_cuda_mdwf_dslash_part_f64(
    double* d_out, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, const BridgeCudaMdwfParams* params,
    int vectors, const BridgeCudaHalo* halo, int part, cudaStream_t stream)
{
    dslash_part<double>(d_out, d_in, d_gauge, geom, params, vectors, halo, part, stream);
}

void bridge_cuda_hop_pack_f32(
    float* d_send, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int in_parity, int face, int vectors, cudaStream_t stream)
{
    hop_pack<float>(d_send, d_in, d_gauge, geom, mass, b5, c5, dagger, in_parity, face, vectors, stream);
}

void bridge_cuda_hop_pack_f64(
    double* d_send, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double mass, double b5, double c5,
    int dagger, int in_parity, int face, int vectors, cudaStream_t stream)
{
    hop_pack<double>(d_send, d_in, d_gauge, geom, mass, b5, c5, dagger, in_parity, face, vectors, stream);
}

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
        static_cast<float>(params->M5), static_cast<float>(params->mass), static_cast<float>(params->b5), static_cast<float>(params->c5),
        bridge_cuda::Halo<float>{}
    );
}

void bridge_cuda_hop_f32(
    float* d_out, const float* d_in, const float* d_gauge, const float* d_from, const float* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream)
{
    hop_part<float>(d_out, d_in, d_gauge, d_from, d_clover, geom, kappa, mass, b5, c5, dagger,
                  target_parity, vectors, nullptr, bridge_cuda::Whole, stream);
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
        static_cast<double>(params->M5), static_cast<double>(params->mass), static_cast<double>(params->b5), static_cast<double>(params->c5),
        bridge_cuda::Halo<double>{}
    );
}

void bridge_cuda_hop_f64(
    double* d_out, const double* d_in, const double* d_gauge, const double* d_from, const double* d_clover,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int dagger, int target_parity, int vectors, cudaStream_t stream)
{
    hop_part<double>(d_out, d_in, d_gauge, d_from, d_clover, geom, kappa, mass, b5, c5, dagger,
                  target_parity, vectors, nullptr, bridge_cuda::Whole, stream);
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

void bridge_cuda_hop_split_halo_f32(
    float* d_cross, float* d_inside, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, const BridgeCudaHalo* halo, cudaStream_t stream)
{
    bridge_cuda::hop_split_kernel<float, true><<<site_grid(volume(geom), geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_cross, d_inside, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls, vectors, leg, block[0], block[1], block[2], block[3],
        static_cast<float>(kappa), static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5),
        halo_of<float>(halo)
    );
}

void bridge_cuda_hop_split_halo_f64(
    double* d_cross, double* d_inside, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, const BridgeCudaHalo* halo, cudaStream_t stream)
{
    bridge_cuda::hop_split_kernel<double, true><<<site_grid(volume(geom), geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_cross, d_inside, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls, vectors, leg, block[0], block[1], block[2], block[3],
        kappa, mass, b5, c5,
        halo_of<double>(halo)
    );
}

void bridge_cuda_hop_split_f32(
    float* d_cross, float* d_inside, const float* d_in, const float* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, cudaStream_t stream)
{
    bridge_cuda::hop_split_kernel<float><<<site_grid(volume(geom), geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_cross, d_inside, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<float>(geom->bc[0]), static_cast<float>(geom->bc[1]),
        static_cast<float>(geom->bc[2]), static_cast<float>(geom->bc[3]),
        geom->Ls, vectors, leg, block[0], block[1], block[2], block[3],
        static_cast<float>(kappa), static_cast<float>(mass), static_cast<float>(b5), static_cast<float>(c5)
    );
}

void bridge_cuda_hop_split_f64(
    double* d_cross, double* d_inside, const double* d_in, const double* d_gauge,
    const BridgeCudaGeometry* geom, double kappa, double mass, double b5, double c5,
    int leg, const int* block, int vectors, cudaStream_t stream)
{
    bridge_cuda::hop_split_kernel<double><<<site_grid(volume(geom), geom->Ls, vectors), site_block(geom->Ls, vectors), 0, stream>>>(
        d_cross, d_inside, d_in, d_gauge, geom->link_rows,
        geom->dims[0], geom->dims[1], geom->dims[2], geom->dims[3],
        static_cast<double>(geom->bc[0]), static_cast<double>(geom->bc[1]),
        static_cast<double>(geom->bc[2]), static_cast<double>(geom->bc[3]),
        geom->Ls, vectors, leg, block[0], block[1], block[2], block[3],
        static_cast<double>(kappa), static_cast<double>(mass), static_cast<double>(b5), static_cast<double>(c5)
    );
}

void bridge_cuda_walls_f32(
    float* d_out, const float* d_in, int sites, int Ls, int s_minus, int s_plus, int to_4d,
    int vectors, cudaStream_t stream)
{
    bridge_cuda::walls_kernel<float><<<site_grid(sites, 1, vectors), site_block(1, vectors), 0, stream>>>(
        d_out, d_in, sites, Ls, s_minus, s_plus, to_4d, vectors
    );
}

void bridge_cuda_walls_f64(
    double* d_out, const double* d_in, int sites, int Ls, int s_minus, int s_plus, int to_4d,
    int vectors, cudaStream_t stream)
{
    bridge_cuda::walls_kernel<double><<<site_grid(sites, 1, vectors), site_block(1, vectors), 0, stream>>>(
        d_out, d_in, sites, Ls, s_minus, s_plus, to_4d, vectors
    );
}

void bridge_cuda_fifth_gemm_tf32_f32(
    float* d_out, const float* d_in, const float* d_from, const float* d_m,
    int sites, int Ls, cudaStream_t stream)
{
    int blocks = (sites + bridge_cuda::WarpSize - 1) / bridge_cuda::WarpSize;
    // A warp for every strip of 8 columns of every block, 8 warps to a block of threads.
    int grid = (blocks * bridge_cuda::FifthStrips + 7) / 8;
    int nt = (2 * Ls + 15) / 16, k32 = (2 * Ls + 31) / 32 * 32;
    size_t shared = static_cast<size_t>(16 * nt * k32) * sizeof(float);
    switch (nt) {
        case 1: bridge_cuda::fifth_gemm_tf32_kernel<1><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 2: bridge_cuda::fifth_gemm_tf32_kernel<2><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 3: bridge_cuda::fifth_gemm_tf32_kernel<3><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        case 4: bridge_cuda::fifth_gemm_tf32_kernel<4><<<grid, 256, shared, stream>>>(d_out, d_in, d_from, d_m, blocks, Ls); break;
        default: break;  // Ls is at most 32
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
        if (lanes * Ls <= bridge_cuda::HopGemmThreadsDouble && tile + matrix <= MAX_SHARED && (!true || (lanes * Ls) % 32 == 0)) break;
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
