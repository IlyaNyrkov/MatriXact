cmake_minimum_required(VERSION 3.28)

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        gbm
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.4.zip
)
FetchContent_MakeAvailable(gbm)

add_executable(matrixact_omp_bench
        bench_gemm.cpp
)

target_link_libraries(matrixact_omp_bench PRIVATE
        matrixact_omp
        benchmark::benchmark
)

target_compile_options(matrixact_omp_bench PRIVATE -O3 -DNDEBUG)
if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(matrixact_omp_bench PRIVATE -Wno-unused-parameter -Wno-padded)
endif()
