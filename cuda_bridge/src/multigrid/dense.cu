// Dense complex matrices of the last level below the coarse lattice, of any size, by cuSOLVER
// and cuBLAS on the default stream. The matrices are row-major (re, im) in double precision;
// read as column-major they are the transposes, which is accounted for below.

#include "bridge_cuda.h"
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cusolverDn.h>

namespace {

cusolverDnHandle_t solver()
{
    static cusolverDnHandle_t handle = nullptr;
    if (handle == nullptr) cusolverDnCreate(&handle);
    return handle;
}

cublasHandle_t blas()
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) cublasCreate(&handle);
    return handle;
}

constexpr int THREADS = 256;

inline int blocks_for(size_t n) { return static_cast<int>((n + THREADS - 1) / THREADS); }

__global__ void identity_kernel(cuDoubleComplex* b, int n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n) * n) return;
    b[i] = make_cuDoubleComplex((i / n == i % n) ? 1.0 : 0.0, 0.0);
}

// h = (a + a^dag) / 2.
__global__ void hermitian_part_kernel(cuDoubleComplex* h, const cuDoubleComplex* a, int n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n) * n) return;
    const size_t r = i / n, c = i % n;
    const cuDoubleComplex x = a[r * n + c], y = cuConj(a[c * n + r]);
    h[i] = make_cuDoubleComplex(0.5 * (x.x + y.x), 0.5 * (x.y + y.y));
}

// w_j = 1 / |lambda_j| where |lambda_j| > cutoff max |lambda|, else 0. One thread.
__global__ void weights_kernel(double* w, const double* lambda, int n, double cutoff)
{
    double top = 0;
    for (int j = 0; j < n; ++j) top = fmax(top, fabs(lambda[j]));
    for (int j = 0; j < n; ++j) {
        const double l = fabs(lambda[j]);
        w[j] = (l > cutoff * top) ? 1.0 / l : 0.0;
    }
}

// Column j of the column-major v times w_j.
__global__ void scale_columns_kernel(cuDoubleComplex* out, const cuDoubleComplex* v, const double* w, int n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n) * n) return;
    const double s = w[i / n];
    out[i] = make_cuDoubleComplex(s * v[i].x, s * v[i].y);
}

int info_of(const int* d_info)
{
    int info = 0;
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    return info;
}

} // namespace

extern "C" {

int bridge_cuda_dense_inverse(double* d_inverse, const double* d_a, int n)
{
    // Column-major, the buffer of a is a^T: LU of a^T, then a^T X = 1 gives X = (a^T)^-1,
    // whose buffer read row-major is a^-1.
    const size_t count = static_cast<size_t>(n) * n;
    cuDoubleComplex* lu = nullptr;
    int* pivots = nullptr;
    int* d_info = nullptr;
    cuDoubleComplex* work = nullptr;
    int result = 0;
    int lwork = 0;
    cusolverStatus_t status;
    auto* inverse = reinterpret_cast<cuDoubleComplex*>(d_inverse);
    if (cudaMalloc(&lu, count * sizeof(cuDoubleComplex)) != cudaSuccess ||
        cudaMalloc(&pivots, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_info, sizeof(int)) != cudaSuccess) {
        result = -1;
        goto done;
    }
    cudaMemcpy(lu, d_a, count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    identity_kernel<<<blocks_for(count), THREADS>>>(inverse, n);
    status = cusolverDnZgetrf_bufferSize(solver(), n, n, lu, n, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) { result = -100 - static_cast<int>(status); goto done; }
    if (cudaMalloc(&work, static_cast<size_t>(lwork) * sizeof(cuDoubleComplex)) != cudaSuccess) {
        result = -1;
        goto done;
    }
    status = cusolverDnZgetrf(solver(), n, n, lu, n, work, pivots, d_info);
    if (status != CUSOLVER_STATUS_SUCCESS) { result = -100 - static_cast<int>(status); goto done; }
    result = info_of(d_info);
    if (result != 0) goto done;
    status = cusolverDnZgetrs(solver(), CUBLAS_OP_N, n, n, lu, n, pivots, inverse, n, d_info);
    if (status != CUSOLVER_STATUS_SUCCESS) { result = -100 - static_cast<int>(status); goto done; }
    result = info_of(d_info);
done:
    cudaDeviceSynchronize();
    cudaFree(work);
    cudaFree(d_info);
    cudaFree(pivots);
    cudaFree(lu);
    return result;
}

int bridge_cuda_dense_abs_inverse(double* d_out, const double* d_a, int n, double cutoff)
{
    // Column-major, the Hermitian part h is read as h^T = conj(h); its |.|^+ is conj(m) for
    // m = |h|^+, whose buffer read row-major is conj(m)^T = m.
    const size_t count = static_cast<size_t>(n) * n;
    cuDoubleComplex* h = nullptr;
    cuDoubleComplex* scaled = nullptr;
    double* lambda = nullptr;
    double* weights = nullptr;
    int* d_info = nullptr;
    cuDoubleComplex* work = nullptr;
    int result = 0;
    int lwork = 0;
    cusolverStatus_t status;
    cublasStatus_t blas_status;
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    if (cudaMalloc(&h, count * sizeof(cuDoubleComplex)) != cudaSuccess ||
        cudaMalloc(&scaled, count * sizeof(cuDoubleComplex)) != cudaSuccess ||
        cudaMalloc(&lambda, n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&weights, n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_info, sizeof(int)) != cudaSuccess) {
        result = -1;
        goto done;
    }
    hermitian_part_kernel<<<blocks_for(count), THREADS>>>(h, reinterpret_cast<const cuDoubleComplex*>(d_a), n);
    status = cusolverDnZheevd_bufferSize(solver(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                         n, h, n, lambda, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) { result = -100 - static_cast<int>(status); goto done; }
    if (cudaMalloc(&work, static_cast<size_t>(lwork) * sizeof(cuDoubleComplex)) != cudaSuccess) {
        result = -1;
        goto done;
    }
    status = cusolverDnZheevd(solver(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, h, n,
                              lambda, work, lwork, d_info);
    if (status != CUSOLVER_STATUS_SUCCESS) { result = -100 - static_cast<int>(status); goto done; }
    result = info_of(d_info);
    if (result != 0) goto done;
    // h now holds the eigenvectors V (columns): out = (V w) V^dag.
    weights_kernel<<<1, 1>>>(weights, lambda, n, cutoff);
    scale_columns_kernel<<<blocks_for(count), THREADS>>>(scaled, h, weights, n);
    blas_status = cublasZgemm(blas(), CUBLAS_OP_N, CUBLAS_OP_C, n, n, n, &one, scaled, n, h, n, &zero,
                              reinterpret_cast<cuDoubleComplex*>(d_out), n);
    if (blas_status != CUBLAS_STATUS_SUCCESS) result = -200 - static_cast<int>(blas_status);
done:
    cudaDeviceSynchronize();
    cudaFree(work);
    cudaFree(d_info);
    cudaFree(weights);
    cudaFree(lambda);
    cudaFree(scaled);
    cudaFree(h);
    return result;
}

} // extern "C"
