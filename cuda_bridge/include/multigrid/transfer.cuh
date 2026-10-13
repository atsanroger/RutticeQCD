#ifndef BRIDGE_CUDA_MULTIGRID_TRANSFER_CUH
#define BRIDGE_CUDA_MULTIGRID_TRANSFER_CUH

#include "../base/types.h"
#include "../base/constant.h"
#include "../base/aosoa.cuh"
#include "../base/gamma.cuh"
#include "../base/su3.cuh"
#include <cuda_runtime.h>

namespace bridge_cuda {

// =========================================================================
// Unified Multigrid Transfer Indexing
//
// Unifies 4D Clover/Wilson MG (Ls = 1) and 5D Domain Wall MG (Ls > 1).
// Basis memory layout: [fine_site4d][test_vec][chirality][kept]
// - For Clover / Wilson (Ls = 1): kept = 24 reals (1 spinor).
// - For MDWF (Ls > 1, folded):    kept = (Ls / 2) * 24 reals.
// =========================================================================

__device__ __forceinline__ int mg_basis_idx(
    int site4d, int vec_idx, int chirality, int is_half, int comp, int num_vectors, int Ls)
{
    int kept = (Ls == 1) ? BRIDGE_CUDA_NSP : (Ls / 2) * BRIDGE_CUDA_NSP;
    int offset = is_half * BRIDGE_CUDA_NSP + comp;
    return ((site4d * num_vectors + vec_idx) * 2 + chirality) * kept + offset;
}

// -------------------------------------------------------------------------
// Unified Restriction Kernel: coarse = P^dag * iota(fine)
//
// Supports:
// - Ls = 1: 4D Wilson / Clover Multigrid with chiral decomposition
// - Ls > 1: 5D Möbius Domain Wall Multigrid with chiral folding
//
// Supports target_parity:
//   -1: All sites in block B contribute (Full lattice)
//    0: Only Even sites contribute (Even-parity injection iota)
//    1: Only Odd sites contribute
// -------------------------------------------------------------------------
template <
    typename FloatFine,
    typename FloatBasis = float,
    typename FloatCoarse = FloatFine,
    int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL,
    int MAX_BASIS = 32>
__global__ void mg_restrict_unified_kernel(
    FloatCoarse* __restrict__ d_coarse,
    const FloatFine* __restrict__ d_fine,
    const FloatBasis* __restrict__ d_basis,
    int num_vectors,
    int nx, int ny, int nz, int nt,
    int bx, int by, int bz, int bt,
    int Ls,
    int target_parity)
{
    int nbx = nx / bx;
    int nby = ny / by;
    int nbz = nz / bz;
    int nbt = nt / bt;
    int total_blocks = nbx * nby * nbz * nbt;
    int half_ls = (Ls == 1) ? 1 : (Ls / 2);

    for (int block_id = blockIdx.x * blockDim.x + threadIdx.x; block_id < total_blocks; block_id += blockDim.x * gridDim.x) {
        int b_x = block_id % nbx; int rem = block_id / nbx;
        int b_y = rem % nby; rem /= nby;
        int b_z = rem % nbz;
        int b_t = rem / nbz;

        // Accumulators for coarse components: 2 * num_vectors (chiral +/- for each basis vector)
        FloatCoarse sum_r[2 * MAX_BASIS] = {0};
        FloatCoarse sum_i[2 * MAX_BASIS] = {0};

        for (int lt = 0; lt < bt; ++lt) {
            int t = b_t * bt + lt;
            for (int lz = 0; lz < bz; ++lz) {
                int z = b_z * bz + lz;
                for (int ly = 0; ly < by; ++ly) {
                    int y = b_y * by + ly;
                    for (int lx = 0; lx < bx; ++lx) {
                        int x = b_x * bx + lx;

                        // Parity check
                        int site_parity = (x + y + z + t) & 1;
                        if (target_parity >= 0 && site_parity != target_parity) {
                            continue;
                        }

                        int site4d = x + nx * (y + ny * (z + nz * t));

                        for (int is = 0; is < half_ls; ++is) {
                            FloatFine f_folded[2][Nspin][Nc][Ncomplex];

                            if (Ls == 1) {
                                // 4D Wilson / Clover: direct 4D spinor (no folding across Ls)
                                #pragma unroll
                                for (int s = 0; s < Nspin; ++s) {
                                    #pragma unroll
                                    for (int c = 0; c < Nc; ++c) {
                                        FloatFine r = d_fine[aosoa_spinor_idx(c, s, 0, site4d)];
                                        FloatFine i = d_fine[aosoa_spinor_idx(c, s, 1, site4d)];
                                        f_folded[0][s][c][0] = r; f_folded[0][s][c][1] = i;
                                        f_folded[1][s][c][0] = r; f_folded[1][s][c][1] = i;
                                    }
                                }
                            } else {
                                // 5D MDWF: Fold near and far slices via gamma_5
                                int is_mirror = Ls - 1 - is;
                                FloatFine sp_near[Nspin][Nc][Ncomplex], sp_far[Nspin][Nc][Ncomplex];
                                #pragma unroll
                                for (int s = 0; s < Nspin; ++s) {
                                    #pragma unroll
                                    for (int c = 0; c < Nc; ++c) {
                                        sp_near[s][c][0] = d_fine[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site4d)];
                                        sp_near[s][c][1] = d_fine[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site4d)];
                                        sp_far[s][c][0]  = d_fine[aosoa_5d_spinor_idx(c, s, 0, is_mirror, Ls, site4d)];
                                        sp_far[s][c][1]  = d_fine[aosoa_5d_spinor_idx(c, s, 1, is_mirror, Ls, site4d)];
                                    }
                                }

                                FloatFine g_far[Nspin][Nc][Ncomplex];
                                mult_gamma5<FloatFine, GammaRep>(g_far, sp_far);

                                #pragma unroll
                                for (int s = 0; s < Nspin; ++s) {
                                    #pragma unroll
                                    for (int c = 0; c < Nc; ++c) {
                                        f_folded[0][s][c][0] = sp_near[s][c][0] + g_far[s][c][0];
                                        f_folded[0][s][c][1] = sp_near[s][c][1] + g_far[s][c][1];
                                        f_folded[1][s][c][0] = sp_near[s][c][0] - g_far[s][c][0];
                                        f_folded[1][s][c][1] = sp_near[s][c][1] - g_far[s][c][1];
                                    }
                                }
                            }

                            // Contract with basis vectors for both chiralities
                            for (int ch = 0; ch < 2; ++ch) {
                                for (int k = 0; k < num_vectors; ++k) {
                                    int coarse_idx = ch * num_vectors + k;

                                    #pragma unroll
                                    for (int s = 0; s < Nspin; ++s) {
                                        #pragma unroll
                                        for (int c = 0; c < Nc; ++c) {
                                            int comp_r = 2 * (c + Nc * s) + 0;
                                            int comp_i = 2 * (c + Nc * s) + 1;
                                            int b_idx_r = mg_basis_idx(site4d, k, ch, is, comp_r, num_vectors, Ls);
                                            int b_idx_i = mg_basis_idx(site4d, k, ch, is, comp_i, num_vectors, Ls);

                                            FloatBasis b_r = d_basis[b_idx_r];
                                            FloatBasis b_i = d_basis[b_idx_i];

                                            FloatFine f_r = f_folded[ch][s][c][0];
                                            FloatFine f_i = f_folded[ch][s][c][1];

                                            // sum += b^dag * f = (b_r - i b_i) * (f_r + i f_i)
                                            sum_r[coarse_idx] += static_cast<FloatCoarse>(b_r * f_r + b_i * f_i);
                                            sum_i[coarse_idx] += static_cast<FloatCoarse>(b_r * f_i - b_i * f_r);
                                        }
                                    }
                                }
                            }
                        } // is
                    } // lx
                } // ly
            } // lz
        } // lt

        // Store out to coarse field at block_id
        int total_coarse_dim = 2 * num_vectors;
        for (int k = 0; k < total_coarse_dim; ++k) {
            int out_idx = (block_id * total_coarse_dim + k) * 2;
            d_coarse[out_idx + 0] = sum_r[k];
            d_coarse[out_idx + 1] = sum_i[k];
        }
    }
}

// -------------------------------------------------------------------------
// Unified Prolongation Kernel: fine = pi(P * coarse)
//
// Supports:
// - Ls = 1: 4D Wilson / Clover Multigrid
// - Ls > 1: 5D Möbius Domain Wall Multigrid
// -------------------------------------------------------------------------
template <
    typename FloatFine,
    typename FloatBasis = float,
    typename FloatCoarse = FloatFine,
    int GammaRep = BRIDGE_CUDA_GAMMA_CHIRAL,
    int MAX_BASIS = 32>
__global__ void mg_prolong_unified_kernel(
    FloatFine* __restrict__ d_fine,
    const FloatCoarse* __restrict__ d_coarse,
    const FloatBasis* __restrict__ d_basis,
    int num_vectors,
    int nx, int ny, int nz, int nt,
    int bx, int by, int bz, int bt,
    int Ls,
    int target_parity)
{
    int vol4d = nx * ny * nz * nt;
    int nbx = nx / bx;
    int nby = ny / by;
    int nbz = nz / bz;
    int half_ls = (Ls == 1) ? 1 : (Ls / 2);
    int total_coarse_dim = 2 * num_vectors;

    for (int site4d = blockIdx.x * blockDim.x + threadIdx.x; site4d < vol4d; site4d += blockDim.x * gridDim.x) {
        int x = site4d % nx; int rem = site4d / nx;
        int y = rem % ny; rem /= ny;
        int z = rem % nz;
        int t = rem / nz;

        int site_parity = (x + y + z + t) & 1;
        bool skip_site = (target_parity >= 0 && site_parity != target_parity);

        int b_x = x / bx;
        int b_y = y / by;
        int b_z = z / bz;
        int b_t = t / bt;
        int block_id = b_x + nbx * (b_y + nby * (b_z + nbz * b_t));

        for (int is = 0; is < half_ls; ++is) {
            int is_mirror = (Ls == 1) ? 0 : (Ls - 1 - is);

            if (skip_site) {
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        if (Ls == 1) {
                            d_fine[aosoa_spinor_idx(c, s, 0, site4d)] = static_cast<FloatFine>(0);
                            d_fine[aosoa_spinor_idx(c, s, 1, site4d)] = static_cast<FloatFine>(0);
                        } else {
                            d_fine[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site4d)] = static_cast<FloatFine>(0);
                            d_fine[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site4d)] = static_cast<FloatFine>(0);
                            d_fine[aosoa_5d_spinor_idx(c, s, 0, is_mirror, Ls, site4d)] = static_cast<FloatFine>(0);
                            d_fine[aosoa_5d_spinor_idx(c, s, 1, is_mirror, Ls, site4d)] = static_cast<FloatFine>(0);
                        }
                    }
                }
                continue;
            }

            // Accumulate u_plus and u_minus for this site and slice is
            FloatFine u_plus[Nspin][Nc][Ncomplex] = {0};
            FloatFine u_minus[Nspin][Nc][Ncomplex] = {0};

            for (int k = 0; k < num_vectors; ++k) {
                int c_idx_p = (block_id * total_coarse_dim + k) * 2;
                int c_idx_m = (block_id * total_coarse_dim + num_vectors + k) * 2;

                FloatCoarse c_p_r = d_coarse[c_idx_p + 0];
                FloatCoarse c_p_i = d_coarse[c_idx_p + 1];
                FloatCoarse c_m_r = d_coarse[c_idx_m + 0];
                FloatCoarse c_m_i = d_coarse[c_idx_m + 1];

                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        int comp_r = 2 * (c + Nc * s) + 0;
                        int comp_i = 2 * (c + Nc * s) + 1;

                        // Chiral plus
                        int bp_r_idx = mg_basis_idx(site4d, k, 0, is, comp_r, num_vectors, Ls);
                        int bp_i_idx = mg_basis_idx(site4d, k, 0, is, comp_i, num_vectors, Ls);
                        FloatBasis bp_r = d_basis[bp_r_idx];
                        FloatBasis bp_i = d_basis[bp_i_idx];

                        u_plus[s][c][0] += static_cast<FloatFine>(bp_r * c_p_r - bp_i * c_p_i);
                        u_plus[s][c][1] += static_cast<FloatFine>(bp_r * c_p_i + bp_i * c_p_r);

                        // Chiral minus
                        int bm_r_idx = mg_basis_idx(site4d, k, 1, is, comp_r, num_vectors, Ls);
                        int bm_i_idx = mg_basis_idx(site4d, k, 1, is, comp_i, num_vectors, Ls);
                        FloatBasis bm_r = d_basis[bm_r_idx];
                        FloatBasis bm_i = d_basis[bm_i_idx];

                        u_minus[s][c][0] += static_cast<FloatFine>(bm_r * c_m_r - bm_i * c_m_i);
                        u_minus[s][c][1] += static_cast<FloatFine>(bm_r * c_m_i + bm_i * c_m_r);
                    }
                }
            }

            if (Ls == 1) {
                // 4D Wilson / Clover: direct sum of chiral parts
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        d_fine[aosoa_spinor_idx(c, s, 0, site4d)] = u_plus[s][c][0] + u_minus[s][c][0];
                        d_fine[aosoa_spinor_idx(c, s, 1, site4d)] = u_plus[s][c][1] + u_minus[s][c][1];
                    }
                }
            } else {
                // 5D MDWF: Unfold near and far slices via gamma_5
                FloatFine diff[Nspin][Nc][Ncomplex];
                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        FloatFine p_r = u_plus[s][c][0];
                        FloatFine p_i = u_plus[s][c][1];
                        FloatFine m_r = u_minus[s][c][0];
                        FloatFine m_i = u_minus[s][c][1];

                        d_fine[aosoa_5d_spinor_idx(c, s, 0, is, Ls, site4d)] = p_r + m_r;
                        d_fine[aosoa_5d_spinor_idx(c, s, 1, is, Ls, site4d)] = p_i + m_i;

                        diff[s][c][0] = p_r - m_r;
                        diff[s][c][1] = p_i - m_i;
                    }
                }

                FloatFine mirrored[Nspin][Nc][Ncomplex];
                mult_gamma5<FloatFine, GammaRep>(mirrored, diff);

                #pragma unroll
                for (int s = 0; s < Nspin; ++s) {
                    #pragma unroll
                    for (int c = 0; c < Nc; ++c) {
                        d_fine[aosoa_5d_spinor_idx(c, s, 0, is_mirror, Ls, site4d)] = mirrored[s][c][0];
                        d_fine[aosoa_5d_spinor_idx(c, s, 1, is_mirror, Ls, site4d)] = mirrored[s][c][1];
                    }
                }
            }
        } // is
    }
}

// -------------------------------------------------------------------------
// Unified Block Modified Gram-Schmidt Orthonormalization Kernel
//
// Supports:
// - Ls = 1 (4D Clover / Wilson)
// - Ls > 1 (5D MDWF)
// -------------------------------------------------------------------------
template <typename FloatBasis = float, int MAX_BASIS = 32>
__global__ void mg_block_normalize_unified_kernel(
    FloatBasis* __restrict__ d_basis,
    int num_vectors,
    int nx, int ny, int nz, int nt,
    int bx, int by, int bz, int bt,
    int Ls)
{
    int nbx = nx / bx;
    int nby = ny / by;
    int nbz = nz / bz;
    int nbt = nt / bt;
    int total_blocks = nbx * nby * nbz * nbt;
    int half_ls = (Ls == 1) ? 1 : (Ls / 2);
    double mirror_factor = (Ls == 1) ? 1.0 : 2.0;

    for (int block_id = blockIdx.x * blockDim.x + threadIdx.x; block_id < total_blocks; block_id += blockDim.x * gridDim.x) {
        int b_x = block_id % nbx; int rem = block_id / nbx;
        int b_y = rem % nby; rem /= nby;
        int b_z = rem % nbz;
        int b_t = rem / nbz;

        for (int ch = 0; ch < 2; ++ch) {
            for (int j = 0; j < num_vectors; ++j) {
                // 1. Compute block norm of vector j
                double norm_sq = 0.0;
                for (int lt = 0; lt < bt; ++lt) {
                    int t = b_t * bt + lt;
                    for (int lz = 0; lz < bz; ++lz) {
                        int z = b_z * bz + lz;
                        for (int ly = 0; ly < by; ++ly) {
                            int y = b_y * by + ly;
                            for (int lx = 0; lx < bx; ++lx) {
                                int x = b_x * bx + lx;
                                int site4d = x + nx * (y + ny * (z + nz * t));

                                for (int is = 0; is < half_ls; ++is) {
                                    for (int comp = 0; comp < BRIDGE_CUDA_NSP; ++comp) {
                                        int idx = mg_basis_idx(site4d, j, ch, is, comp, num_vectors, Ls);
                                        double val = static_cast<double>(d_basis[idx]);
                                        norm_sq += val * val;
                                    }
                                }
                            }
                        }
                    }
                }

                norm_sq *= mirror_factor;
                double inv_norm = (norm_sq > 1e-24) ? (1.0 / sqrt(norm_sq)) : 0.0;

                // 2. Normalize vector j
                for (int lt = 0; lt < bt; ++lt) {
                    int t = b_t * bt + lt;
                    for (int lz = 0; lz < bz; ++lz) {
                        int z = b_z * bz + lz;
                        for (int ly = 0; ly < by; ++ly) {
                            int y = b_y * by + ly;
                            for (int lx = 0; lx < bx; ++lx) {
                                int x = b_x * bx + lx;
                                int site4d = x + nx * (y + ny * (z + nz * t));

                                for (int is = 0; is < half_ls; ++is) {
                                    for (int comp = 0; comp < BRIDGE_CUDA_NSP; ++comp) {
                                        int idx = mg_basis_idx(site4d, j, ch, is, comp, num_vectors, Ls);
                                        d_basis[idx] = static_cast<FloatBasis>(d_basis[idx] * inv_norm);
                                    }
                                }
                            }
                        }
                    }
                }

                // 3. Orthogonalize remaining vectors k > j against vector j
                for (int k = j + 1; k < num_vectors; ++k) {
                    double proj_r = 0.0, proj_i = 0.0;
                    for (int lt = 0; lt < bt; ++lt) {
                        int t = b_t * bt + lt;
                        for (int lz = 0; lz < bz; ++lz) {
                            int z = b_z * bz + lz;
                            for (int ly = 0; ly < by; ++ly) {
                                int y = b_y * by + ly;
                                for (int lx = 0; lx < bx; ++lx) {
                                    int x = b_x * bx + lx;
                                    int site4d = x + nx * (y + ny * (z + nz * t));

                                    for (int is = 0; is < half_ls; ++is) {
                                        for (int sp = 0; sp < 12; ++sp) {
                                            int idx_j_r = mg_basis_idx(site4d, j, ch, is, 2 * sp + 0, num_vectors, Ls);
                                            int idx_j_i = mg_basis_idx(site4d, j, ch, is, 2 * sp + 1, num_vectors, Ls);
                                            int idx_k_r = mg_basis_idx(site4d, k, ch, is, 2 * sp + 0, num_vectors, Ls);
                                            int idx_k_i = mg_basis_idx(site4d, k, ch, is, 2 * sp + 1, num_vectors, Ls);

                                            double jr = static_cast<double>(d_basis[idx_j_r]);
                                            double ji = static_cast<double>(d_basis[idx_j_i]);
                                            double kr = static_cast<double>(d_basis[idx_k_r]);
                                            double ki = static_cast<double>(d_basis[idx_k_i]);

                                            proj_r += (jr * kr + ji * ki);
                                            proj_i += (jr * ki - ji * kr);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    proj_r *= mirror_factor;
                    proj_i *= mirror_factor;

                    for (int lt = 0; lt < bt; ++lt) {
                        int t = b_t * bt + lt;
                        for (int lz = 0; lz < bz; ++lz) {
                            int z = b_z * bz + lz;
                            for (int ly = 0; ly < by; ++ly) {
                                int y = b_y * by + ly;
                                for (int lx = 0; lx < bx; ++lx) {
                                    int x = b_x * bx + lx;
                                    int site4d = x + nx * (y + ny * (z + nz * t));

                                    for (int is = 0; is < half_ls; ++is) {
                                        for (int sp = 0; sp < 12; ++sp) {
                                            int idx_j_r = mg_basis_idx(site4d, j, ch, is, 2 * sp + 0, num_vectors, Ls);
                                            int idx_j_i = mg_basis_idx(site4d, j, ch, is, 2 * sp + 1, num_vectors, Ls);
                                            int idx_k_r = mg_basis_idx(site4d, k, ch, is, 2 * sp + 0, num_vectors, Ls);
                                            int idx_k_i = mg_basis_idx(site4d, k, ch, is, 2 * sp + 1, num_vectors, Ls);

                                            double jr = static_cast<double>(d_basis[idx_j_r]);
                                            double ji = static_cast<double>(d_basis[idx_j_i]);
                                            double kr = static_cast<double>(d_basis[idx_k_r]);
                                            double ki = static_cast<double>(d_basis[idx_k_i]);

                                            kr -= (proj_r * jr - proj_i * ji);
                                            ki -= (proj_r * ji + proj_i * jr);

                                            d_basis[idx_k_r] = static_cast<FloatBasis>(kr);
                                            d_basis[idx_k_i] = static_cast<FloatBasis>(ki);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } // k
            } // j
        } // ch
    }
}

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_MULTIGRID_TRANSFER_CUH
