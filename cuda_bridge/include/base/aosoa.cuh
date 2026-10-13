#ifndef BRIDGE_CUDA_BASE_AOSOA_CUH
#define BRIDGE_CUDA_BASE_AOSOA_CUH

#include "types.h"
#include <cuda_runtime.h>

namespace bridge_cuda {

constexpr int W = BRIDGE_CUDA_WARP_SIZE; // AoSoA inner SIMD/SIMT lane width W=32

// -------------------------------------------------------------------------
// AoSoA Indexing functions (aligned with RutticeQCD Aosoa<32>)
// -------------------------------------------------------------------------

// Generic AoSoA array index: field has N components per site
__host__ __device__ __forceinline__ int aosoa_idx(int comp, int site, int N) {
    int outer = site / W;
    int lane  = site % W;
    return (outer * N + comp) * W + lane;
}

// 4D Spinor field index: color c in [0,2], spin s in [0,3], ri in [0,1]
__host__ __device__ __forceinline__ int aosoa_spinor_idx(int c, int s, int ri, int site) {
    int comp = ri + 2 * (c + BRIDGE_CUDA_NC * s); // 0 .. 23
    return aosoa_idx(comp, site, BRIDGE_CUDA_NSP);
}

// 5D Spinor field index: is in [0, Ls-1], site4d in [0, vol4d-1]
// Outer layout: warps of W sites, with Ls slices bundled per warp: outer = (site4d / W) * Ls + is
__host__ __device__ __forceinline__ int aosoa_5d_spinor_idx(int c, int s, int ri, int is, int Ls, int site4d) {
    int outer = (site4d / W) * Ls + is;
    int lane  = site4d % W;
    int comp  = ri + 2 * (c + BRIDGE_CUDA_NC * s);
    return (outer * BRIDGE_CUDA_NSP + comp) * W + lane;
}

// SU(3) Gauge field index: direction mu in [0,3], row r in [0,2], col c in [0,2], ri in [0,1]
__host__ __device__ __forceinline__ int aosoa_gauge_idx(int mu, int r, int c, int ri, int site) {
    int comp = ri + 2 * (c + BRIDGE_CUDA_NC * (r + BRIDGE_CUDA_NC * mu)); // 0 .. 71
    return aosoa_idx(comp, site, 4 * BRIDGE_CUDA_NDF);
}


// -------------------------------------------------------------------------
// 4D Site Index and Coordinates Mapping
// -------------------------------------------------------------------------
__host__ __device__ __forceinline__ void site_to_coords(
    int site, int &x, int &y, int &z, int &t,
    int nx, int ny, int nz, int nt)
{
    x = site % nx;
    int rem = site / nx;
    y = rem % ny;
    rem = rem / ny;
    z = rem % nz;
    t = rem / nz;
}

__host__ __device__ __forceinline__ int coords_to_site(
    int x, int y, int z, int t,
    int nx, int ny, int nz, int nt)
{
    return x + nx * (y + ny * (z + nz * t));
}

// -------------------------------------------------------------------------
// Grid-Stride Loop Helpers
// -------------------------------------------------------------------------
struct GridStride {
    int start;
    int stride;
    int total;

    __host__ __device__ __forceinline__ GridStride(int n)
        : start(blockIdx.x * blockDim.x + threadIdx.x),
          stride(blockDim.x * gridDim.x),
          total(n) {}

    struct Iterator {
        int i;
        int stride;
        __host__ __device__ __forceinline__ int operator*() const { return i; }
        __host__ __device__ __forceinline__ Iterator& operator++() { i += stride; return *this; }
        __host__ __device__ __forceinline__ bool operator!=(const Iterator& o) const { return i < o.i; }
    };

    __host__ __device__ __forceinline__ Iterator begin() const { return {start, stride}; }
    __host__ __device__ __forceinline__ Iterator end() const { return {total, stride}; }
};

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_AOSOA_CUH
