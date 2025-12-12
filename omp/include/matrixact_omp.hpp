#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <cfloat>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace matrixact::omp {

std::vector<double> generate_random_matrix(std::size_t N, std::size_t M, double min_val, double max_val, int precision) {
  std::vector<double> matrix(N * M);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_val, max_val);

  double factor = std::pow(10, precision);

  for (std::size_t i = 0; i < N * M; ++i) {
    double val = dis(gen);
    matrix[i] = std::round(val * factor) / factor;
  }
  return matrix;
}

template <class T>
void print_matrix(const std::vector<T>& matrix, std::size_t N, std::size_t M) {
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      std::cout << matrix[i * M + j] << " ";
    }
    std::cout << std::endl;
  }
}

const int B_UPPER_BOUND = 7;
// 2^53
const int MAX_DOUBLE_MANTISSA_LEN = 53;

const double THREE_FOURTHS = 0.75;

// M x K times K x N = M x N
template <class T1, class T2>
void gemm(const T1* A, const T1* B, T2* C, std::size_t M, std::size_t N, std::size_t K) {
#pragma omp parallel for collapse(2)
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      T2 acc = 0;
      for (std::size_t k = 0; k < K; ++k) acc += A[i*K + k] * B[k*N + j];
      C[i*N + j] += acc;
    }
}

double max_abs_in_row(const int row_id, const double* matrix,
  const std::size_t M, const std::size_t N) {
  if (row_id >= M) throw std::exception();
  double max_elem = 0;

  for (int i = 0; i < N; i++) {
    max_elem = std::max(max_elem, std::fabs(matrix[row_id * N + i]));
  }

  return max_elem;
}

static inline int8_t sat_int8(long long x) {
  if (x > 127)  return 127;
  if (x < -127) return -127;
  return static_cast<int8_t>(x);
}

int calculate_bit_per_slice(const std::size_t N) {
  int bits_per_slice_from_n = std::floor((31 - log2(N))/ 2);
  int bits_per_slice = std::min(B_UPPER_BOUND,bits_per_slice_from_n);
  if (bits_per_slice < 1) bits_per_slice = 1;
  return bits_per_slice;
}

void split_fp64_matrix_int8_algo8(double* matrix, const std::size_t M,
 const std::size_t N, int k, int8_t* slices, double* scale_factor) {
  // step 1: calculate bits per slice
  const int bits_per_slice = calculate_bit_per_slice(N);
  // step 2: calculate u
  for (int i = 0; i < M; i++) {
    double max_abs = max_abs_in_row(i, matrix, M, N);
    if (max_abs == 0.0 || !std::isfinite(max_abs)) {
      scale_factor[i] = std::ldexp(1.0, 1 - bits_per_slice);
    } else {
      int e = std::ceil(std::log2(max_abs));          // fast log2
      scale_factor[i] = std::ldexp(1.0, e + 1 - bits_per_slice);
    }
  }

  // step 3: caclulate slices
  std::vector<double> sigma(M);
  for (int s = 0; s < k; s++) {

    // step 4: calculate σ_i = 0.75 * 2^{53 - β*s} * μ_i
    int shift = MAX_DOUBLE_MANTISSA_LEN - bits_per_slice * s;
    for (int i = 0; i < M; i++) {
      sigma[i] = 0.75 * std::ldexp(scale_factor[i], shift);
    }

    // for each element in original matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
      double inversed_scale_factor = (scale_factor[i] != 0.0) ? 1.0 / scale_factor[i] : 0.0;
      int negative_shift = -bits_per_slice * s;
      for (int j = 0; j < N; j++) {
        // step 5: extract top bits from original matrix element
        double aij = matrix[i * N + j];
        double top = (aij + sigma[i]) - sigma[i];

        // step 6: convert to int8 and store in slice
        double normalized = top * inversed_scale_factor;
        double scaled = std::ldexp(normalized, negative_shift);
        slices[s * M * N + i * N + j] = static_cast<int8_t>(scaled);

        // step7: remove top bits
        matrix[i * N + j] -= top;
      }
    }
  }
}

void reconstruct_fp64_matrix_from_int8_slices(const std::size_t M, const std::size_t N,
  int k, const int8_t* slices, double* scale_factor, double* result_matrix) {
  std::fill(result_matrix, result_matrix + M * N, 0.0);
  const int bits_per_slice = calculate_bit_per_slice(N);
  for (int s = 0; s < k; ++s) {
    int shift = bits_per_slice * s; // 2^{β s}
    for (std::size_t i = 0; i < M; ++i) {
      double scale_row = std::ldexp(scale_factor[i], shift);
      for (std::size_t j = 0; j < N; ++j) {
        std::size_t idx = s * (M*N) + i * N + j;
        result_matrix[i*N + j] += scale_row * static_cast<double>(slices[idx]);
      }
    }
  }
}

void multiply_fp64_slices_naive(const int8_t* slices_A, const int8_t* slices_B,
  const std::size_t k, const std::size_t M, const std::size_t N, const std::size_t P,
  double* u, double* v, double* result_matrix) {
  // step 1: calculate bits per slice
  int bits_per_slice = calculate_bit_per_slice(N);
  // step 2: initialize empty result matrix
  std::fill(result_matrix, result_matrix + M * P, 0.0);
  for (int s = 0; s < k; s++) {
    for (int t = 0; t < k; t++) {
        int slice_A_id = s * M * N;
        int slice_B_id = t * N * P;

        std::vector<int32_t> slice_C(M * P, 0);
        // step 3: compute using gemm
        gemm<int8_t, int32_t>(slices_A + slice_A_id, // A
          slices_B + slice_B_id, // B
          slice_C.data(), // C
          M, P, N);
        // step 4: scale and accumulate to result matrix
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
          double scale_row = std::ldexp(u[i], bits_per_slice * s);
          for (int j = 0; j < P; j++) {
            double scale_col =  std::ldexp(v[j], bits_per_slice * t);
            result_matrix[i * P + j] += scale_row  *
              static_cast<double>(slice_C[i * P + j]) * scale_col;
          }
        }

    }
  }
}

void multiply_fp64_slices_triangular(const int8_t* slices_A, const int8_t* slices_B,
  const std::size_t k, const std::size_t M, const std::size_t N, const std::size_t P,
  double* u, double* v, double* result_matrix) {
  // step 1: calculate bits per slice
  int bits_per_slice = calculate_bit_per_slice(N);
  // step 2: initialize empty result matrix
  std::fill(result_matrix, result_matrix + M * P, 0.0);
  for (int s = 0; s < k; s++) {
    for (int t = 0; t < k - s; t++) {
      int slice_A_id = s * M * N;
      int slice_B_id = t * N * P;

      std::vector<int32_t> slice_C(M * P, 0);
      // step 3: compute using gemm
      gemm<int8_t, int32_t>(slices_A + slice_A_id, // A
        slices_B + slice_B_id, // B
        slice_C.data(), // C
        M, P, N);
      // step 4: scale and accumulate to result matrix
      #pragma omp parallel for collapse(2)
      for (int i = 0; i < M; i++) {
        double scale_row = std::ldexp(u[i], bits_per_slice * s);
        for (int j = 0; j < P; j++) {
          double scale_col =  std::ldexp(v[j], bits_per_slice * t);
          result_matrix[i * P + j] += scale_row  *
            static_cast<double>(slice_C[i * P + j]) * scale_col;
        }
      }
    }
  }
}

void multiply_fp64_slices_algo4(const int8_t* slices_A, const int8_t* slices_B,
  const std::size_t k, const std::size_t M, const std::size_t N, const std::size_t P,
  double* u, double* v, double* result_matrix) {
  // step 1: calculate bits per slice
  int bits_per_slice = calculate_bit_per_slice(N);
  // step 2: initialize empty result matrix
  std::fill(result_matrix, result_matrix + M * P, 0.0);
  // step 3: diagonally loop over slices
  for (int g = 0; g < k; g++) {
    // step 4: for each slice pair (s, g-s)
    for (int s = 0; s <= g; s++) {
      std::vector<int32_t> accumulation(M * P, 0);
      int slice_A_id = s * M * N;
      int slice_B_id = (g - s) * N * P;
      // step 5: compute using gemm
      gemm<int8_t, int32_t>(slices_A + slice_A_id,
          slices_B + slice_B_id,
          accumulation.data(),
          M, P, N);
      // step 6: scale and accumulate to result matrix
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
          result_matrix[i * P + j] +=  u[i] * pow(2, -bits_per_slice * s + 1) *
            pow(2, -bits_per_slice*(g - s) + 1) * static_cast<double>(accumulation[i * P + j]) * v[j];
        }
      }

    }
  }
}

void multiply_fp64_slices_algo8(const int8_t* slices_A, const int8_t* slices_B,
  const std::size_t k, const std::size_t M, const std::size_t N, const std::size_t P,
  double* u, double* v, double* result_matrix) {
  // step 1: calculate bits per slice
  int bits_per_slice = calculate_bit_per_slice(N);
  // step 2: initialize empty result matrix
  std::fill(result_matrix, result_matrix + M * P, 0.0);
  // step 3: diagonally loop over slices
  for (int g = 1; g <= k; g++) {
    // step 4: initialize accumulation matrix
    std::vector<int32_t> accumulation(M * P, 0);
    // step 5: for each slice pair (s, g-s)
    for (int s = 0; s < g - 1; s++) {
      int slice_A_id = s;
      int slice_B_id = g - s;

      // step 6: compute using gemm
      gemm<int8_t, int32_t>(slices_A +( N * M * slice_A_id), // A
        slices_B + (N * P * slice_B_id), // B
        accumulation.data(), // C
        M, P, N);
    } // step 7: end for each slice pair

    // step 8: scale and accumulate to result matrix
    double scale = std::ldexp(1.0, -bits_per_slice * g + 2);
    for (int i = 0; i < M; i++) {
      double scale_row = scale * u[i];
      for (int j = 0; j < P; j++) {
        result_matrix[i * P + j] += scale_row  *
          static_cast<double>(accumulation[i * P + j]) * v[j];
      }
    }
  }
}

std::pair<double, std::pair<int, int>> max_abs_diff(const double* A, const double* B, std::size_t M, std::size_t N) {
  double e = 0.0;
  auto id = std::make_pair(-1, -1);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (std::fabs(A[i * N + j] - B[i * N + j]) > e) {
        e = std::fabs(A[i * N + j] - B[i * N + j]);
        id = std::make_pair(i, j);
      }
    }
  }

  return std::make_pair(e, id);
}


} // namespace matrixact::omp