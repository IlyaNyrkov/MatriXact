#pragma once
#include <cstddef>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace matrixact::omp {

template <class T>
void gemm(const T* A, const T* B, T* C, std::size_t M, std::size_t N, std::size_t K) {
#pragma omp parallel for collapse(2)
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      T acc = 0;
      for (std::size_t k = 0; k < K; ++k) acc += A[i*K + k] * B[k*N + j];
      C[i*N + j] = acc;
    }
}

} // namespace matrixact::omp