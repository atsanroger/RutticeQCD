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
    src/fermion/dd.cu
    src/multigrid/multigrid.cu
    src/multigrid/mrhs.cu
    src/multigrid/fold.cu
    src/gauge/gauge.cu
    src/linalg/blas.cu
)

add_library(bridge_cuda_static STATIC ${BRIDGE_CUDA_SOURCES})
add_library(bridge_cuda SHARED ${BRIDGE_CUDA_SOURCES})

set_target_properties(bridge_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# The device link is done in the archive, so that a linker other than nvcc (cargo's) can use it.
set_target_properties(bridge_cuda_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(test_mdwf_1pass tests/test_mdwf_1pass.cu)
target_link_libraries(test_mdwf_1pass PRIVATE bridge_cuda_static)

add_executable(test_mdwf_transfer tests/test_mdwf_transfer.cu)
target_link_libraries(test_mdwf_transfer PRIVATE bridge_cuda_static)

add_executable(test_coarse_schur tests/test_coarse_schur.cu)
target_link_libraries(test_coarse_schur PRIVATE bridge_cuda_static)

add_executable(test_clover_mg_transfer tests/test_clover_mg_transfer.cu)
target_link_libraries(test_clover_mg_transfer PRIVATE bridge_cuda_static)
