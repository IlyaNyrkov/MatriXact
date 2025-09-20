//
// Created by Ilya Nyrkov on 20.09.25.
//

#include <benchmark/benchmark.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#ifdef _OPENMP
  #include <omp.h>
#endif

// Your API
#include "matrixact_omp.hpp"

// ----- reference single-thread naive GEMM (row-major) -----
template <class T>
void gemm_naive(const T* A, const T* B, T* C,
                std::size_t M, std::size_t N, std::size_t K) {
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      long long acc = 0; // wider to avoid overflow for int
      for (std::size_t k = 0; k < K; ++k)
        acc += static_cast<long long>(A[i*K + k]) * B[k*N + j];
      C[i*N + j] = static_cast<T>(acc);
    }
}

// ----- helpers -----
template <class T>
static void fill_pattern(std::vector<T>& v, T start, T step, T minv, T maxv) {
  T x = start;
  for (auto& e : v) {
    e = x;
    long long nx = static_cast<long long>(x) + static_cast<long long>(step);
    if (nx > maxv) nx = minv;
    x = static_cast<T>(nx);
  }
}

static inline double gflops_per_call(std::size_t M, std::size_t N, std::size_t K) {
  // GEMM does 2*M*N*K FLOPs
  return (2.0 * M * N * K) / 1e9;
}

// ----- BENCH: Ozaki/OpenMP GEMM -----
template <class T>
static void BM_GEMM_Omp(benchmark::State& state) {
  const std::size_t M = state.range(0);
  const std::size_t N = state.range(1);
  const std::size_t K = state.range(2);
  const int threads    = static_cast<int>(state.range(3));

#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif

  std::vector<T> A(M*K), B(K*N), C(M*N);
  fill_pattern(A, T(-3), T(2), T(-9), T(9));
  fill_pattern(B, T( 4), T(3), T(-9), T(9));

  // Warm cache for fairness
  std::fill(C.begin(), C.end(), T{0});
  gemm<T>(A.data(), B.data(), C.data(), M, N, K);

  for (auto _ : state) {
    std::fill(C.begin(), C.end(), T{0});
    benchmark::DoNotOptimize(C.data());
    gemm<T>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::ClobberMemory();
  }

  state.counters["GFLOP/s"] = benchmark::Counter(
      gflops_per_call(M,N,K) * state.iterations(),
      benchmark::Counter::kIsRate);
  state.counters["Threads"] = threads;
  state.SetLabel(std::is_same_v<T,int32_t> ? "int32" : "T");
}

// ----- BENCH: Naive single-thread baseline -----
template <class T>
static void BM_GEMM_Naive(benchmark::State& state) {
  const std::size_t M = state.range(0);
  const std::size_t N = state.range(1);
  const std::size_t K = state.range(2);

#ifdef _OPENMP
  omp_set_num_threads(1); // ensure single-thread
#endif

  std::vector<T> A(M*K), B(K*N), C(M*N);
  fill_pattern(A, T(-3), T(2), T(-9), T(9));
  fill_pattern(B, T( 4), T(3), T(-9), T(9));

  // warm-up
  std::fill(C.begin(), C.end(), T{0});
  gemm_naive<T>(A.data(), B.data(), C.data(), M, N, K);

  for (auto _ : state) {
    std::fill(C.begin(), C.end(), T{0});
    benchmark::DoNotOptimize(C.data());
    gemm_naive<T>(A.data(), B.data(), C.data(), M, N, K);
    benchmark::ClobberMemory();
  }

  state.counters["GFLOP/s"] = benchmark::Counter(
      gflops_per_call(M,N,K) * state.iterations(),
      benchmark::Counter::kIsRate);
  state.counters["Threads"] = 1;
  state.SetLabel(std::is_same_v<T,int32_t> ? "int32" : "T");
}

// ----- Register benchmarks -----
// Shapes & thread counts; add more Args as needed.
#define SQUARE_CASES(T) \
  BENCHMARK_TEMPLATE(BM_GEMM_Omp,   T)->Args({256,256,256,1})->UseRealTime(); \
  BENCHMARK_TEMPLATE(BM_GEMM_Omp,   T)->Args({256,256,256,4})->UseRealTime(); \
  BENCHMARK_TEMPLATE(BM_GEMM_Omp,   T)->Args({512,512,512,8})->UseRealTime(); \
  BENCHMARK_TEMPLATE(BM_GEMM_Naive, T)->Args({256,256,256,0})->UseRealTime(); \
  BENCHMARK_TEMPLATE(BM_GEMM_Naive, T)->Args({512,512,512,0})->UseRealTime();

SQUARE_CASES(int32_t)

BENCHMARK_MAIN();
