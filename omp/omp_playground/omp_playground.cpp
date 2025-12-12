//
// Created by Ilya Nyrkov on 29.10.25.
//

#include <iostream>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include <vector>
using namespace std;

vector<vector<double>> generate_matrix(int N) {
  vector<vector<double>> matrix(N, vector<double>(N));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrix[i][j] = (double)(i * N + j  + 1);
    }
  }

  return matrix;
}

void print_matrix(const vector<vector<double>>& matrix) {
  int N = matrix.size();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      cout << matrix[i][j] << " ";
    }
    cout << endl;
  }
}

vector<vector<double>> transpose_matrix(const vector<vector<double>>& matrix) {
  int N = matrix.size();
  vector<vector<double>> transposed(N, vector<double>(N));
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      transposed[j][i] = matrix[i][j];
    }
  }
  return transposed;
}

void transpose_matrix_inplace(vector<vector<double>>& matrix) {
  int N = matrix.size();
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      std::swap(matrix[i][j], matrix[j][i]);
    }
  }
}

static long num_steps = 100000;
double step;
int main () {

  double pi = 0.0;

  int num_threads = 8;

  step = 1.0/(double) num_steps;

  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    int ID = omp_get_thread_num();
    double x;
    double sum = 0.0;
    for (int i=ID;i< num_steps; i=i+num_threads) {
      x = (i+0.5)*step;
      sum += 4.0/(1.0+x*x);
    }

    #pragma omp critical
    {
      pi += sum * step;
    }
  }

  std::cout << pi << std::endl;
}