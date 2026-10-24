#include "gauge/links.cuh"
// The kernels that walk paths of links share this unit: their headers define them outright.
#include "fermion/clover_links.cuh"
#include "bridge_cuda.h"

namespace {
    constexpr int BLOCK_SIZE = 128;

    // dims: the local extents, then the depths of the halo (stout.cuh).
    bridge_cuda::StoutGeometry geometry(const int* dims, const double* rho)
    {
        bridge_cuda::StoutGeometry g;
        g.vol = 1;
        for (int mu = 0; mu < 4; ++mu) {
            g.local[mu] = dims[mu];
            g.depth[mu] = dims[4 + mu];
            g.dims[mu] = dims[mu] + 2 * dims[4 + mu];
            g.vol *= dims[mu];
            for (int nu = 0; nu < 4; ++nu) g.rho[mu][nu] = rho[4 * mu + nu];
        }
        return g;
    }

    const double ONE[16] = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};

    inline int blocks_for(size_t n) { return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE); }
}

extern "C" {

void bridge_cuda_stout_step(double* d_v, const double* d_u, const int* dims, const double* rho, cudaStream_t stream)
{
    const auto g = geometry(dims, rho);
    bridge_cuda::stout_step_kernel<<<blocks_for(4 * static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(d_v, d_u, g);
}

void bridge_cuda_stout_chain_local(
    double* d_sigma, double* d_itheta, const double* d_sigma_prime, const double* d_u,
    const int* dims, const double* rho, cudaStream_t stream)
{
    const auto g = geometry(dims, rho);
    const int blocks = blocks_for(4 * static_cast<size_t>(g.vol));
    bridge_cuda::stout_chain_local_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_sigma, d_itheta, d_sigma_prime, d_u, g);
}

void bridge_cuda_stout_chain_staples(
    double* d_sigma, const double* d_u, const double* d_itheta,
    const int* dims, const double* rho, cudaStream_t stream)
{
    const auto g = geometry(dims, rho);
    const int blocks = blocks_for(4 * static_cast<size_t>(g.vol));
    bridge_cuda::stout_chain_staples_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_sigma, d_u, d_itheta, g);
}

void bridge_cuda_links_extend(double* d_ext, const double* d_local, const int* dims, cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    bridge_cuda::links_extend_kernel<<<blocks_for(static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(d_ext, d_local, g);
}

void bridge_cuda_links_layers(double* d_buf, double* d_ext, int mu, int first, int pack, const int* dims, cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    size_t sites = static_cast<size_t>(g.depth[mu]);
    for (int nu = 0; nu < 4; ++nu) if (nu != mu) sites *= g.dims[nu];
    bridge_cuda::links_layers_kernel<<<blocks_for(sites), BLOCK_SIZE, 0, stream>>>(d_buf, d_ext, mu, first, pack, g);
}

void bridge_cuda_links_axpy(double* d_y, double a, const double* d_x, size_t n, cudaStream_t stream)
{
    bridge_cuda::links_axpy_kernel<<<blocks_for(n), BLOCK_SIZE, 0, stream>>>(d_y, a, d_x, n);
}

void bridge_cuda_links_exp_update(double* d_u, const double* d_p, double eps, int order, int vol, cudaStream_t stream)
{
    bridge_cuda::links_exp_update_kernel<<<blocks_for(4 * static_cast<size_t>(vol)), BLOCK_SIZE, 0, stream>>>(d_u, d_p, eps, order, vol);
}

void bridge_cuda_links_ta(double* d_out, const double* d_m, double s, int vol, cudaStream_t stream)
{
    bridge_cuda::links_ta_kernel<<<blocks_for(4 * static_cast<size_t>(vol)), BLOCK_SIZE, 0, stream>>>(d_out, d_m, s, vol);
}

void bridge_cuda_links_mul(double* d_out, const double* d_a, const double* d_b, int adjoint_a, int vol, cudaStream_t stream)
{
    bridge_cuda::links_mul_kernel<<<blocks_for(4 * static_cast<size_t>(vol)), BLOCK_SIZE, 0, stream>>>(d_out, d_a, d_b, adjoint_a, vol);
}

void bridge_cuda_gauge_force(double* d_f, const double* d_u, double beta, double c_plaq, double c_rect, const int* dims, cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    bridge_cuda::gauge_force_kernel<<<blocks_for(4 * static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(d_f, d_u, beta, c_plaq, c_rect, g);
}

void bridge_cuda_gauge_traces(double* d_plaq, double* d_rect, const double* d_u, int with_rect, const int* dims, cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    bridge_cuda::gauge_traces_kernel<<<blocks_for(static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(d_plaq, d_rect, d_u, with_rect, g);
}

void bridge_cuda_links_norm2(double* d_out, const double* d_m, int vol, cudaStream_t stream)
{
    bridge_cuda::links_norm2_kernel<<<blocks_for(static_cast<size_t>(vol)), BLOCK_SIZE, 0, stream>>>(d_out, d_m, vol);
}

// The clover term made of the links (fermion/clover_links.cuh).
void bridge_cuda_clover_blocks(
    double* d_t, double* d_t_inv, double* d_ln_det, double* d_singular, const double* d_u,
    double kappa_csw, const int* dims, cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    bridge_cuda::clover_blocks_kernel<<<blocks_for(bridge_cuda::Chiralities * static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(
        d_t, d_t_inv, d_ln_det, d_singular, d_u, kappa_csw, g);
}

void bridge_cuda_clover_insertion(
    double* d_b_lo, double* d_b_hi, const double* d_zeta, const double* d_eta, double coef, int vol,
    cudaStream_t stream)
{
    bridge_cuda::clover_insertion_kernel<<<blocks_for(static_cast<size_t>(vol)), BLOCK_SIZE, 0, stream>>>(
        d_b_lo, d_b_hi, d_zeta, d_eta, coef, vol);
}

void bridge_cuda_clover_log_det_insertion(
    double* d_b_lo, double* d_b_hi, const double* d_t_inv, double coef, int parity, const int* dims,
    cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    bridge_cuda::clover_log_det_insertion_kernel<<<blocks_for(static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(
        d_b_lo, d_b_hi, d_t_inv, coef, parity, g);
}

void bridge_cuda_clover_leaves(
    double* d_m, const double* d_u, const double* d_b_lo, const double* d_b_hi, const int* dims,
    cudaStream_t stream)
{
    const auto g = geometry(dims, ONE);
    bridge_cuda::clover_leaves_kernel<<<blocks_for(4 * static_cast<size_t>(g.vol)), BLOCK_SIZE, 0, stream>>>(
        d_m, d_u, d_b_lo, d_b_hi, g);
}

} // extern "C"
