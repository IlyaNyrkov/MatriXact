cmake_minimum_required(VERSION 3.10)
project(MatrixActPlayground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/../include)

add_executable(omp_playground omp_playground.cpp)
target_link_libraries(omp_playground PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(omp_playground PRIVATE matrixact_omp)