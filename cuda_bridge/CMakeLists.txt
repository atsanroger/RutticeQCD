cmake_minimum_required(VERSION 3.18)
project(bridge_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

include_directories(include)

set(BRIDGE_CUDA_SOURCES
    src/fermion/wilson.cu
    src/fermion/clover.cu
    src/fermion/mdwf.cu
    src/fermion/force.cu
    src/multigrid/fold.cu
    src/multigrid/levels.cu
    src/multigrid/dense.cu
    src/gauge/stout.cu
    src/linalg/blas.cu
)

add_library(bridge_cuda_static STATIC ${BRIDGE_CUDA_SOURCES})
add_library(bridge_cuda SHARED ${BRIDGE_CUDA_SOURCES})

# The dense matrices of the last level below the coarse lattice (multigrid/dense.cu).
find_package(CUDAToolkit REQUIRED)
target_link_libraries(bridge_cuda_static PUBLIC CUDA::cusolver CUDA::cublas)
target_link_libraries(bridge_cuda PUBLIC CUDA::cusolver CUDA::cublas)

set_target_properties(bridge_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# The device link is done in the archive, so that a linker other than nvcc (cargo's) can use it.
set_target_properties(bridge_cuda_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(test_mdwf_1pass tests/test_mdwf_1pass.cu)
target_link_libraries(test_mdwf_1pass PRIVATE bridge_cuda_static)
