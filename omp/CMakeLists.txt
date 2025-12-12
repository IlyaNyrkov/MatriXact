cmake_minimum_required(VERSION 3.28)
project(matrixact_omp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(_LIBOMP_HINTS
            /opt/homebrew/opt/libomp
            /usr/local/opt/libomp)
    find_path(LIBOMP_INCLUDE_DIR omp.h HINTS ${_LIBOMP_HINTS} PATH_SUFFIXES include)
    find_library(LIBOMP_LIBRARY omp HINTS ${_LIBOMP_HINTS} PATH_SUFFIXES lib)

    if (NOT LIBOMP_INCLUDE_DIR OR NOT LIBOMP_LIBRARY)
        message(FATAL_ERROR
                "libomp not found. Install with:  brew install libomp  \n"
                "Expected at one of: ${_LIBOMP_HINTS}")
    endif()

    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
            INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${LIBOMP_LIBRARY}"
    )
else()
    # Non-AppleClang toolchains can use the standard find_package
    find_package(OpenMP REQUIRED)
endif()

add_library(matrixact_omp INTERFACE
        benchmarks/bench_gemm.cpp
        examples/main.cpp
        omp_playground/omp_playground.cpp
)

target_include_directories(matrixact_omp INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(matrixact_omp INTERFACE OpenMP::OpenMP_CXX)
target_compile_definitions(matrixact_omp INTERFACE OZK_HAS_OMP=1)

option(MX_OMP_BUILD_TESTS "Build tests" ON)
option(MX_OMP_BUILD_EXAMPLES "Build examples" ON)
option(MX_OMP_BUILD_PLAYGROUND "Build playground" ON)

if (MX_OMP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if (MX_OMP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(MX_OMP_BUILD_BENCH "Build benchmarks" ON)
if (MX_OMP_BUILD_BENCH)
    add_subdirectory(benchmarks)
endif()

if (MX_OMP_BUILD_PLAYGROUND)
    add_subdirectory(omp_playground)
endif()
