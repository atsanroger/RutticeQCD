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
// The grid-stride loop: `for (size_t i : GridStride(n))` gives thread t of the grid the indices
// t, t + T, t + 2T, ... below n, T the threads of the grid. A kernel over n independent
// elements is launched with any grid (capped, e.g. at 1024 blocks) and still covers them all;
// consecutive threads take consecutive indices, so the reads of a warp are coalesced. Indices
// are size_t: n may be vectors times the reals of a field.
// -------------------------------------------------------------------------
struct GridStride {
    size_t start;
    size_t stride;
    size_t total;

    __device__ __forceinline__ explicit GridStride(size_t n)
        : start(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x),
          stride(static_cast<size_t>(blockDim.x) * gridDim.x),
          total(n) {}

    struct Iterator {
        size_t i;
        size_t stride;
        __device__ __forceinline__ size_t operator*() const { return i; }
        __device__ __forceinline__ Iterator& operator++() { i += stride; return *this; }
        __device__ __forceinline__ bool operator!=(const Iterator& end) const { return i < end.i; }
    };

    __device__ __forceinline__ Iterator begin() const { return {start, stride}; }
    __device__ __forceinline__ Iterator end() const { return {total, stride}; }
};

} // namespace bridge_cuda

#endif // BRIDGE_CUDA_BASE_AOSOA_CUH
