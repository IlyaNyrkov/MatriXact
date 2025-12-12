//
// Created by Ilya Nyrkov on 25.09.25.
//
#include <iostream>
#include <matrixact_omp.hpp>
using namespace std;

template <class T>
double riemann_sum_parallel(T func, double a, double b, int iterations) {
  auto delta = (b - a) / iterations;
  double total = 0.0;
  #pragma omp parallel for reduction(+:total)
  for (int i = 0; i < iterations; ++i) {
    total += func(a + i * delta) * delta;
  }

  return total;
}

template <class T>
double riemann_sum_serial(T func, double a, double b, int iterations) {
  auto delta = (b - a) / iterations;
  double total = 0.0;
  for (int i = 0; i < iterations; ++i) {
    total += func(a + i * delta) * delta;
  }
  return total;
}

int sum_parallel(int* arr, size_t size) {
  int total = 0;
  #pragma omp parallel reduction(+:total)
  for (int i = 0; i < size; ++i) {
      total += arr[i];
  }
  return total;
}


std::vector<double> generate_ones_matrix(std::size_t N, std::size_t M, int factor) {
  std::vector<double> matrix(N * M);

  for (std::size_t i = 0; i < N * M; ++i) {
    matrix[i] = 1.75251 * factor;
  }
  return matrix;
}

int main() {

  // === INTIALIZATION ===
  int N = 1000;
  int M = 1000;
  double min_val = 0.0;
  double max_val = 10.0;
  int precision = 8;
  //auto matrix_A = matrixact::omp::generate_random_matrix(N, M, min_val, max_val, precision);
  auto matrix_A = generate_ones_matrix(N, M, 1);

  cout << "Matrix A" << endl;
  //matrixact::omp::print_matrix(matrix_A, N, M);


  vector<double> u(N);
  int k = 7;
  vector<int8_t> slices_A(N * M * k);

  auto risidual_matrix_A = matrix_A;
  matrixact::omp::split_fp64_matrix_int8_algo8(risidual_matrix_A.data(), N, M, k, slices_A.data(), u.data());
  cout << "Remainder of matrix A" << endl;
  //matrixact::omp::print_matrix(risidual_matrix_A, N, M);

  vector<double> reconstructed_A(N * M, 0.0);
  matrixact::omp::reconstruct_fp64_matrix_from_int8_slices(N, M, k, slices_A.data(), u.data(), reconstructed_A.data());
  cout << "Reconstructed matrix A" << endl;
  //matrixact::omp::print_matrix(reconstructed_A, N, M);

  cout << "Max abs diff between original and reconstructed A: ";
  auto [e, id] = matrixact::omp::max_abs_diff(matrix_A.data(), reconstructed_A.data(), M, N);
  cout << e << " at " << id.first << ", " << id.second << endl;

  auto matrix_b = matrixact::omp::generate_random_matrix(M, N, min_val, max_val, precision);
  //auto matrix_b = generate_ones_matrix(M, N, 3);
  cout << "Matrix B" << endl;
  //matrixact::omp::print_matrix(matrix_b, M, N);
  vector<double> v(N);
  vector<int8_t> slices_B(N * M * k);
  auto risidual_matrix_B = matrix_b;
  matrixact::omp::split_fp64_matrix_int8_algo8(risidual_matrix_B.data(), M, N, k, slices_B.data(), v.data());
  cout << "Remainder of matrix B" << endl;
  //matrixact::omp::print_matrix(risidual_matrix_B, M, N);
  vector<double> reconstructed_B(N * M, 0.0);
  matrixact::omp::reconstruct_fp64_matrix_from_int8_slices(M, N, k, slices_B.data(), v.data(), reconstructed_B.data());
  cout << "Reconstructed matrix B" << endl;
  //matrixact::omp::print_matrix(reconstructed_B, M, N);
  cout << "Max abs diff between original and reconstructed B: ";
  auto [e2, id2] = matrixact::omp::max_abs_diff(matrix_b.data(), reconstructed_B.data(), M, N);
  cout << e2 << " at " << id2.first << ", " << id2.second << endl;

  // === MULTIPLICATION ===
  vector<double> reference_C(N * N, 0.0);
  auto start = omp_get_wtime();
  matrixact::omp::gemm<double, double>(matrix_A.data(), matrix_b.data(), reference_C.data(), N, N, M);
  auto end = omp_get_wtime();
  cout << "Time for reference multiplication: " << (end - start) * 1000 << " ms" << endl;
  cout << "Reference C = A * B" << endl;
  //matrixact::omp::print_matrix(reference_C, N, N);

  // FP64 slices multiplication

  vector<double> matrix_C(N * N, 0.0);
  start = omp_get_wtime();
  matrixact::omp::multiply_fp64_slices_triangular(slices_A.data(), slices_B.data(),
    k, N, N, N, u.data(), v.data(), matrix_C.data());
  end = omp_get_wtime();
  cout << "Time for slices multiplication: " << (end - start) * 1000 << " ms" << endl;
  cout << "C = A * B using FP64 slices" << endl;
  //matrixact::omp::print_matrix(matrix_C, N, N);
  auto [e3, id3] = matrixact::omp::max_abs_diff(reference_C.data(), matrix_C.data(), N, N);
  cout << "Max abs diff between reference and slices multiplication C: ";
  cout << e3 << " at " << id3.first << ", " << id3.second << endl;
  return 0;

}