//
// Created by Ilya Nyrkov on 20.09.25.
//

// omp/tests/test_gemm_int.cpp
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>

// Include your header with the gemm<T> declaration/definition
#include "matrixact_omp.hpp"   // adjust if your header path differs

namespace {

// Reference matmul (row-major), int-safe (no overflow checks)
template <class T>
void ref_gemm(const T* A, const T* B, T* C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            long long acc = 0; // wider accumulator to reduce overflow risk in tests
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<long long>(A[i*K + k]) *
                       static_cast<long long>(B[k*N + j]);
            }
            C[i*N + j] = static_cast<T>(acc);
        }
    }
}

template <class T>
void expect_eq_matrix(const std::vector<T>& C, const std::vector<T>& R) {
    ASSERT_EQ(C.size(), R.size());
    for (std::size_t i = 0; i < C.size(); ++i) {
        ASSERT_EQ(C[i], R[i]) << "Mismatch at index " << i;
    }
}

// Helper to fill small deterministic patterns without overflow
template <class T>
void fill_seq(std::vector<T>& v, T start, T step, T minv, T maxv) {
    T x = start;
    for (auto& e : v) {
        e = x;
        long long nx = static_cast<long long>(x) + static_cast<long long>(step);
        if (nx > maxv) nx = minv;
        x = static_cast<T>(nx);
    }
}

} // namespace

// --- TESTS ---

TEST(GemmInt, Trivial_1x1) {
    using T = int32_t;
    const std::size_t M=1,N=1,K=1;
    std::vector<T> A = {3};
    std::vector<T> B = {7};
    std::vector<T> C(M * N, 0), R(M * N, 0);

    matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ref_gemm<T>(A.data(), B.data(), R.data(), M,N,K);
    expect_eq_matrix(C, R);
}

TEST(GemmInt, DotProduct_1xN_Nx1) {
    using T = int32_t;
    const std::size_t M=1,N=1,K=5;
    std::vector<T> A = {1,2,3,4,5};      // 1x5
    std::vector<T> B = {6,7,8,9,10};     // 5x1 (stored row-major as contiguous length-5, we still use KxN = 5x1)
    std::vector<T> C(M*N, 0), R(M*N, 0);

   matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ref_gemm<T>(A.data(), B.data(), R.data(), M,N,K);
    expect_eq_matrix(C, R);
}

TEST(GemmInt, Rectangular_2x3_3x2) {
    using T = int32_t;
    const std::size_t M=2,N=2,K=3;
    std::vector<T> A = {
        1, 2, 3,
        -1, 0, 2
    }; // 2x3
    std::vector<T> B = {
        4, -2,
        5,  3,
        6,  1
    }; // 3x2
    std::vector<T> C(M*N, 0), R(M*N, 0);

   matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ref_gemm<T>(A.data(), B.data(), R.data(), M,N,K);
    expect_eq_matrix(C, R);
}

TEST(GemmInt, KEqualsZero_ZeroResult) {
    using T = int32_t;
    const std::size_t M=3,N=4,K=0;
    std::vector<T> A(M*K, 5); // size 0
    std::vector<T> B(K*N, 7); // size 0
    std::vector<T> C(M*N, 123); // prefilled garbage; gemm should overwrite with zeros if implemented as C=A*B
    std::vector<T> R(M*N, 0);

    // Reference: with K=0, result is zero matrix
    // Our ref_gemm handles K=0
    ref_gemm<T>(A.data(), B.data(), R.data(), M,N,K);

   matrixact::omp::gemm<T>(nullptr, nullptr, C.data(), M,N,K); // A,B ignored as K=0; okay if your gemm doesn’t deref them
    expect_eq_matrix(C, R);
}

TEST(GemmInt, Empty_MZero) {
    using T = int32_t;
    const std::size_t M=0,N=5,K=7;
    std::vector<T> A(M*K), B(K*N, 1), C(M*N); // sizes with M=0 => A,C empty
    // Should not crash
   matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ASSERT_EQ(C.size(), 0u);
}

TEST(GemmInt, Empty_NZero) {
    using T = int32_t;
    const std::size_t M=5,N=0,K=7;
    std::vector<T> A(M*K, 1), B(K*N), C(M*N); // N=0 => B,C empty
    // Should not crash
   matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ASSERT_EQ(C.size(), 0u);
}

TEST(GemmInt, NegativesAndCancellation) {
    using T = int32_t;
    const std::size_t M=3,N=3,K=3;
    std::vector<T> A = {
        2, -3,  4,
       -1,  5, -2,
        0, -6,  1
    };
    std::vector<T> B = {
        -7,  8,  0,
         9, -1,  3,
         2,  4, -5
    };
    std::vector<T> C(M * N, 0), R(M * N, 0);

    matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ref_gemm<T>(A.data(), B.data(), R.data(), M,N,K);
    expect_eq_matrix(C, R);
}

TEST(GemmInt, PatternFill_NoOverflow_MediumSizes) {
    using T = int32_t;
    const std::size_t M=8,N=7,K=6;
    std::vector<T> A(M*K), B(K*N), C(M*N, 0), R(M*N, 0);

    // Fill with small bounded ints to avoid overflow: values in [-9..9]
    fill_seq<T>(A, /*start=*/-3, /*step=*/2, /*minv=*/-9, /*maxv=*/9);
    fill_seq<T>(B, /*start=*/4, /*step=*/3, /*minv=*/-9, /*maxv=*/9);

    matrixact::omp::gemm<T>(A.data(), B.data(), C.data(), M,N,K);
    ref_gemm<T>(A.data(), B.data(), R.data(), M,N,K);
    expect_eq_matrix(C, R);
}

// (Optional) If your contract forbids aliasing, you can ASSERT it by copying first.
// If aliasing is allowed and handled, add tests ensuring correct behavior when C==A or C==B.

