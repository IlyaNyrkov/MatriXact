cmake_minimum_required(VERSION 3.28)

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(matrixact_omp_tests
        test_gemm_int.cpp
)

target_link_libraries(matrixact_omp_tests PRIVATE
        matrixact_omp
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matrixact_omp_tests
        DISCOVERY_MODE PRE_TEST
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=4"
)