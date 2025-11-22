#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <cmath>


// AMD headers
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

// Matrix sizes
const std::size_t P = 2<<10;
const std::size_t Q = 2<<10;
const std::size_t R = 2<<10;

// Matrix print counts
const int PRINT_ROWS = 5;
const int PRINT_COLS = 5;

// Matrix value parameters
const double MATRIX_MIN_VAL = 0.0;
const double MATRIX_MAX_VAL = 10.0;

// Ozaki parameters
const size_t slices = 3;

// Matrix helpers

void print_matrix(const std::string& name, const std::vector<double>& matrix,
                  int rows, int cols, int n_rows, int n_cols) {
    std::cout << name << " (first " << n_rows << "x" << n_cols << "):" << std::endl;
    for (int i = 0; i < std::min(n_rows, rows); ++i) {
        for (int j = 0; j < std::min(n_cols, cols); ++j) {
            std::cout << std::setw(8) << std::setprecision(4)
                      << static_cast<double>(matrix[i * cols + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void generate_random_matrix(std::vector<double>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(MATRIX_MIN_VAL, MATRIX_MAX_VAL);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

double compute_max_abs_error(const std::vector<double>& A,
                             const std::vector<double>& B) {
    double max_error = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double error = fabs(A[i] - B[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

// Benchmark function template
// Will be used to compare different Ozaki implementations techniques
template<typename Func>
float benchmark_gemm(Func&& gemm_func, const std::string& name, 
                     rocblas_handle handle, double* d_A, double* d_B, double* d_C) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    // Warm-up
    gemm_func();
    hipDeviceSynchronize();
    
    // Benchmark
    hipEventRecord(start);
    gemm_func();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    double flops = 2.0 * P * Q * R;
    double gflops = (flops * 1e-9) / (milliseconds * 1e-3);
    
    std::cout << name << " - Time: " << milliseconds << " ms, GFLOPS: " << gflops << std::endl;
    return milliseconds;
}

// Ozaki constants

const uint8_t moduli_all[] = {255, 253, 251, 247, 239, 233, 229, 227};

const uint64_t moduli_products[] = {
    // s=1
    255,
    // s=2
    64515,
    // s=3
    16193265,
    // s=4
    3999736455,
    // s=5
    955937012745,
    // s=6
    222733323969585,
    // s=7
    51005931189034965,
    // s=8
    11578346379910937055
    };

const uint64_t partial_moduli[8][8] = {
    // s=1
    {1},
    // s=2
    {253ULL, 255ULL},
    // s=3  
    {253ULL*251ULL, 255ULL*251ULL, 255ULL*253ULL},
    // s=4
    {253ULL*251ULL*247ULL, 255ULL*251ULL*247ULL, 255ULL*253ULL*247ULL, 255ULL*253ULL*251ULL},
    // s=5
    {253ULL*251ULL*247ULL*239ULL, 255ULL*251ULL*247ULL*239ULL, 255ULL*253ULL*247ULL*239ULL,
         255ULL*253ULL*251ULL*239ULL, 255ULL*253ULL*251ULL*247ULL},
    // s=6
    {253ULL*251ULL*247ULL*239ULL*233ULL, 255ULL*251ULL*247ULL*239ULL*233ULL, 255ULL*253ULL*247ULL*239ULL*233ULL, 
        255ULL*253ULL*251ULL*239ULL*233ULL, 255ULL*253ULL*251ULL*247ULL*233ULL, 255ULL*253ULL*251ULL*247ULL*239ULL},
    // s=7
    {253ULL*251ULL*247ULL*239ULL*233ULL*229ULL, 255ULL*251ULL*247ULL*239ULL*233ULL*229ULL, 255ULL*253ULL*247ULL*239ULL*233ULL*229ULL,
        255ULL*253ULL*251ULL*239ULL*233ULL*229ULL, 255ULL*253ULL*251ULL*247ULL*233ULL*229ULL, 255ULL*253ULL*251ULL*247ULL*239ULL*229ULL,
        255ULL*253ULL*251ULL*247ULL*239ULL*233ULL},
    // s=8
    {253ULL*251ULL*247ULL*239ULL*233ULL*229ULL*227ULL, 255ULL*251ULL*247ULL*239ULL*233ULL*229ULL*227ULL,
        255ULL*253ULL*247ULL*239ULL*233ULL*229ULL*227ULL, 255ULL*253ULL*251ULL*239ULL*233ULL*229ULL*227ULL,
        255ULL*253ULL*251ULL*247ULL*233ULL*229ULL*227ULL, 255ULL*253ULL*251ULL*247ULL*239ULL*229ULL*227ULL,
        255ULL*253ULL*251ULL*247ULL*239ULL*233ULL*227ULL, 255ULL*253ULL*251ULL*247ULL*239ULL*233ULL*229ULL}
};

const uint64_t mod_inv[8][8] = {
    // s=1
    {1},
    // s=2
    {127, 127},
    // s=3
    {32, 63, 157}, 
    // s=4
    {251, 116, 149, 238},
    // s=5  
    {64, 64, 134, 32, 199},
    // s=6
    {113, 98, 132, 33, 166, 190},
    // s=7
    {182, 17, 245, 204, 79, 69, 181},
    // s=8
    {121, 48, 63, 138, 93, 105, 24, 132}
};


// Ozaki methods

// Step 2.1 Calculate shift matrices
__global__ void compute_row_max_exponent_A(const double* __restrict__ A,
                                           int rows, int cols,
                                           int32_t* __restrict__ row_exp)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    double v = fabs(A[row * cols + col]);

    // protect against log2(0) -> -inf
    int e = 0;
    if (v > 0.0) {
        double lg = log2(v);
        e = (int)ceil(lg);
        if (e < 0) e = 0;  // clamp if you want
    }

    // we want max exponent per row
    atomicMax(&row_exp[row], e);
}

// For B: size rowsB x colsB, row-major, we want per-column exponents
__global__ void compute_col_max_exponent_B(const double* __restrict__ B,
                                           int rows, int cols,
                                           int32_t* __restrict__ col_exp)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    double v = fabs(B[row * cols + col]);

    int e = 0;
    if (v > 0.0) {
        double lg = log2(v);
        e = (int)ceil(lg);
        if (e < 0) e = 0;
    }

    // we want max exponent per column
    atomicMax(&col_exp[col], e);
}
__global__ void compute_shifts_from_exponents(const int32_t* __restrict__ exp_vec,
                                              int n, int K,
                                              int32_t* __restrict__ shifts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int e = exp_vec[i];
    shifts[i] = K - e;   // Ozaki scaling
}

//  Step 2.1 Calculate shift matrices
__global__ void truncate_and_multiply_A(double* A, int32_t* shifts, int rows, int cols, int64_t* result)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        result[idx] = static_cast<int64_t>(llrint(ldexp(A[idx], shifts[row])));
    }
}

__global__ void truncate_and_multiply_B(double* B, int32_t* shifts, int rows, int cols, int64_t* result)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        result[idx] = static_cast<int64_t>(llrint(ldexp(B[idx], shifts[col])));
    }
}

// Step 2.3 Compute s modulo matrices
// each slice iteration can be parallelized aswell
__global__ void compute_modulo_matrices(int64_t* matrix, uint8_t* moduli, int rows, int cols,
     int slices, uint8_t* mod_matrices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        for (int s = 0; s < slices; ++s) {
            int idx = row * cols + col;
            mod_matrices[s * rows * cols + idx] = static_cast<uint8_t>(matrix[idx] % moduli[s]);
        }
    }
}

// Step 4 Convert matrix products to uint8
__global__ void convert_int32_uint8_modulo(int32_t* matrices, int rows, int cols, int slices,
     uint8_t* moduli, uint8_t* uint8_matrices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        for (int s = 0; s < slices; ++s) {
            int idx = s * rows * cols + row * cols + col;
            uint8_matrices[idx] = static_cast<uint8_t>(matrices[idx] % moduli[s]);
        }
    }
}

// Step 5 Accumulate matrix products
__global__ void accumulate_matrix_products(uint8_t* mod_C_matrices, int rows, int cols,
     int slices, int64_t* accumulated_C, int64_t* partial_moduli, int64_t* mod_inv, uint64_t matrix_product) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        for (int s = 0; s < slices; s++) {
            int64_t partial_mod = partial_moduli[s];
            int64_t inv = mod_inv[s];
            int idx = row * cols + col;
            int64_t mod_C = static_cast<int64_t>(mod_C_matrices[s * rows * cols + idx]);
            accumulated_C[idx] += mod_C * partial_mod * inv;
            accumulated_C[idx] %= matrix_product;
        }
    }
}

// Stage 6: Inversely scale matrix product

__global__ void inversely_scale_matrix(int64_t* C, int rows, int cols, int32_t* shift_vector_A, int32_t* shift_vector_B,
     uint64_t moduli_product, double* results) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        uint32_t e = shift_vector_A[row] + shift_vector_B[col];
        results[idx] = ldexp(static_cast<double>(C[idx] % moduli_product), -e);
    }
}

void ozaki2_gemm(const std::size_t P, const std::size_t Q, const std::size_t R,
                 double* d_A, double* d_B, double* d_C_fp64_out,
                 int slices,
                 const uint8_t* h_moduli,
                 const uint64_t M,
                 const uint64_t* h_partial_mod,
                 const uint64_t* h_mod_inv)
{
    // -------------------------------------------------------------------------
    // 0. rocBLAS handle
    // -------------------------------------------------------------------------
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // -------------------------------------------------------------------------
    // 1. Sizes
    // -------------------------------------------------------------------------
    const std::size_t sizeA = P * Q;
    const std::size_t sizeB = Q * R;
    const std::size_t sizeC = P * R;
    // -------------------------------------------------------------------------
    // 2. Device temporaries
    // -------------------------------------------------------------------------

    // Shift vectors (D for rows of A, E for cols of B)
    int32_t* d_shift_A = nullptr;
    int32_t* d_shift_B = nullptr;
    hipMalloc(&d_shift_A, P * sizeof(int32_t));
    hipMalloc(&d_shift_B, R * sizeof(int32_t));
    hipMemset(d_shift_A, 0, P * sizeof(int32_t));
    hipMemset(d_shift_B, 0, R * sizeof(int32_t));

    // Scaled & truncated integer matrices A', B'
    int64_t* d_A_int = nullptr;
    int64_t* d_B_int = nullptr;
    hipMalloc(&d_A_int, sizeA * sizeof(int64_t));
    hipMalloc(&d_B_int, sizeB * sizeof(int64_t));

    // INT8 slices A'_t, B'_t (step 2.3)
    uint8_t* d_A_slices = nullptr;
    uint8_t* d_B_slices = nullptr;
    hipMalloc(&d_A_slices, slices * sizeA * sizeof(uint8_t));
    hipMalloc(&d_B_slices, slices * sizeB * sizeof(uint8_t));

    // INT32 tensor-core outputs C'_t (step 3)
    int32_t* d_C_tc = nullptr;
    hipMalloc(&d_C_tc, slices * sizeC * sizeof(int32_t));

    // UINT8 residues C''_t (step 4)
    uint8_t* d_C_resid = nullptr;
    hipMalloc(&d_C_resid, slices * sizeC * sizeof(uint8_t));

    // Accumulated CRT integer matrix C''' (step 5)
    int64_t* d_C_accum = nullptr;
    hipMalloc(&d_C_accum, sizeC * sizeof(int64_t));
    hipMemset(d_C_accum, 0, sizeC * sizeof(int64_t));

    // Device copies of moduli, partial_moduli, mod_inv
    uint8_t* d_moduli = nullptr;
    hipMalloc(&d_moduli, slices * sizeof(uint8_t));
    hipMemcpy(d_moduli, h_moduli, slices * sizeof(uint8_t),
              hipMemcpyHostToDevice);

    int64_t* d_partial_moduli = nullptr;
    int64_t* d_mod_inv = nullptr;
    hipMalloc(&d_partial_moduli, slices * sizeof(int64_t));
    hipMalloc(&d_mod_inv, slices * sizeof(int64_t));
    hipMemcpy(d_partial_moduli, h_partial_mod, slices * sizeof(int64_t),
              hipMemcpyHostToDevice);
    hipMemcpy(d_mod_inv, h_mod_inv, slices * sizeof(int64_t),
              hipMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // 3. Stage 2.1 – determine shift values D, E
    // -------------------------------------------------------------------------
    dim3 block(16, 16);
    dim3 gridA((P + block.x - 1) / block.x,
               (Q + block.y - 1) / block.y);
    dim3 gridB((Q + block.x - 1) / block.x,
               (R + block.y - 1) / block.y);

    const int32_t K = std::floor(0.5 * std::log2((M/2.0 - 1.0)/Q));;

    int32_t* d_row_exp_A = nullptr;
    int32_t* d_col_exp_B = nullptr;
    hipMalloc(&d_row_exp_A, P * sizeof(int32_t));
    hipMalloc(&d_col_exp_B, R * sizeof(int32_t));
    hipMemset(d_row_exp_A, 0, P * sizeof(int32_t));
    hipMemset(d_col_exp_B, 0, R * sizeof(int32_t));


    // 1) find max exponents
    compute_row_max_exponent_A<<<gridA, block>>>(
        d_A, (int)P, (int)Q, d_row_exp_A);

    compute_col_max_exponent_B<<<gridB, block>>>(
        d_B, (int)Q, (int)R, d_col_exp_B);

    hipDeviceSynchronize();

    // 2) turn exponents into shifts
    {
        int threads = 256;
        int blocks_A = (P + threads - 1) / threads;
        int blocks_B = (R + threads - 1) / threads;

        compute_shifts_from_exponents<<<blocks_A, threads>>>(d_row_exp_A, (int)P, K, d_shift_A);
        compute_shifts_from_exponents<<<blocks_B, threads>>>(d_col_exp_B, (int)R, K, d_shift_B);
    }

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 4. Stage 2.2 – trunc( D*A ), trunc( B*E )
    // -------------------------------------------------------------------------


    truncate_and_multiply_A<<<gridA, block>>>(
        d_A, d_shift_A, P, Q, d_A_int);
    truncate_and_multiply_B<<<gridB, block>>>(
        d_B, d_shift_B, Q, R, d_B_int);

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 5. Stage 2.3 – build modulo slices A'_t, B'_t
    // -------------------------------------------------------------------------
    compute_modulo_matrices<<<gridA, block>>>(
        d_A_int, d_moduli, P, Q, slices, d_A_slices);

    compute_modulo_matrices<<<gridB, block>>>(
        d_B_int, d_moduli, Q, R, slices, d_B_slices);

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 6. Stage 3 – INT8 GEMM on tensor cores
    // -------------------------------------------------------------------------
    const int m = static_cast<int>(P);
    const int n = static_cast<int>(R);
    const int k = static_cast<int>(Q);

    const int lda = static_cast<int>(P);
    const int ldb = static_cast<int>(Q);
    const int ldc = static_cast<int>(P);

    const int64_t strideA = static_cast<int64_t>(sizeA);
    const int64_t strideB = static_cast<int64_t>(sizeB);
    const int64_t strideC = static_cast<int64_t>(sizeC);

    const int32_t alpha_i32 = 1;
    const int32_t beta_i32  = 0;

    // simplest: loop over slices, one GEMM per slice
    for (int t = 0; t < slices; ++t) {
        uint8_t*  A_t = d_A_slices + t * strideA;
        uint8_t*  B_t = d_B_slices + t * strideB;
        int32_t* C_t = d_C_tc      + t * strideC;

        rocblas_gemm_ex(
            handle,
            rocblas_operation_transpose, rocblas_operation_transpose,
            m, n, k,
            &alpha_i32,
            B_t, rocblas_datatype_i8_r, lda,
            A_t, rocblas_datatype_i8_r, ldb,
            &beta_i32,
            C_t, rocblas_datatype_i32_r, ldc,
            C_t, rocblas_datatype_i32_r, ldc,  // C as output
            rocblas_datatype_i32_r,
            rocblas_gemm_algo_standard, 0, 0);
    }

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 7. Stage 4 – convert INT32 C'_t to UINT8 residues C''_t
    // -------------------------------------------------------------------------
    dim3 gridC((P + block.x - 1) / block.x,
               (R + block.y - 1) / block.y);

    convert_int32_uint8_modulo<<<gridC, block>>>(
        d_C_tc, P, R, slices, d_moduli, d_C_resid);

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 8. Stage 5 – CRT accumulation: C''' = Σ C''_t * partial_moduli[t] * inv[t]
    // -------------------------------------------------------------------------
    accumulate_matrix_products<<<gridC, block>>>(
        d_C_resid, P, R, slices,
        d_C_accum, d_partial_moduli, d_mod_inv, moduli_products[slices - 1]);

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 9. Stage 6 – inversely scale and convert back to FP64
    // -------------------------------------------------------------------------

    inversely_scale_matrix<<<gridC, block>>>(
        d_C_accum, // or use a separate int64 kernel arg
        P, R,
        d_shift_A, d_shift_B,
        M,
        d_C_fp64_out);

    hipDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 10. Cleanup
    // -------------------------------------------------------------------------
    hipFree(d_shift_A);
    hipFree(d_shift_B);
    hipFree(d_A_int);
    hipFree(d_B_int);
    hipFree(d_A_slices);
    hipFree(d_B_slices);
    hipFree(d_C_tc);
    hipFree(d_C_resid);
    hipFree(d_C_accum);
    hipFree(d_moduli);
    hipFree(d_partial_moduli);
    hipFree(d_mod_inv);

    rocblas_destroy_handle(handle);
}

int main() {
    // Problem sizes (use the constants you defined)
    const std::size_t m = P; // rows of A, C
    const std::size_t n = Q; // cols of A / rows of B
    const std::size_t p = R; // cols of B, C

    std::cout << "Matrix sizes: A = " << m << " x " << n
              << ", B = " << n << " x " << p << "\n";
    std::cout << "Slices (Ozaki): " << slices << "\n\n";

    // Host matrices
    std::vector<double> h_A(m * n);
    std::vector<double> h_B(n * p);
    std::vector<double> h_C_ref(m * p, 0.0);
    std::vector<double> h_C_ozaki(m * p, 0.0);

    generate_random_matrix(h_A, (int)m, (int)n);
    generate_random_matrix(h_B, (int)n, (int)p);

    // Device buffers
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C_ref = nullptr;
    double* d_C_ozaki = nullptr;

    hipSetDevice(0);
    hipMalloc(&d_A, m * n * sizeof(double));
    hipMalloc(&d_B, n * p * sizeof(double));
    hipMalloc(&d_C_ref, m * p * sizeof(double));
    hipMalloc(&d_C_ozaki, m * p * sizeof(double));

    hipMemcpy(d_A, h_A.data(), m * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), n * p * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(d_C_ref, 0, m * p * sizeof(double));
    hipMemset(d_C_ozaki, 0, m * p * sizeof(double));

    // ---------------------- rocBLAS reference GEMM -------------------------
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    hipEvent_t start_ref, stop_ref;
    hipEventCreate(&start_ref);
    hipEventCreate(&stop_ref);

    const double alpha = 1.0;
    const double beta  = 0.0;

    // We store matrices in row-major, rocBLAS assumes column-major.
    // Trick: interpret row-major A,B,C as the transposes of column-major matrices
    // and call GEMM with transposed ops:
    //
    //   C_row = A_row * B_row
    //   => C_col^T = B_col^T * A_col^T
    //
    // so we compute: C^T = B^T * A^T
    //
    int mm = (int)m;
    int nn = (int)n;
    int pp = (int)p;

    hipEventRecord(start_ref);
    rocblas_dgemm(
        handle,
        rocblas_operation_transpose,   // B^T: (p x n)
        rocblas_operation_transpose,   // A^T: (n x m)
        pp, mm, nn,                    // m = p, n = m, k = n
        &alpha,
        d_B, nn,                       // B is n x p in row-major ⇒ p x n in column-major
        d_A, nn,                       // A is m x n in row-major ⇒ n x m in column-major
        &beta,
        d_C_ref, pp                    // C is m x p row-major ⇒ p x m column-major
    );
    hipEventRecord(stop_ref);
    hipEventSynchronize(stop_ref);

    float ms_ref = 0.0f;
    hipEventElapsedTime(&ms_ref, start_ref, stop_ref);

    double flops = 2.0 * (double)m * (double)n * (double)p;
    double gflops_ref = (flops * 1e-9) / (ms_ref * 1e-3);

    std::cout << "rocBLAS dgemm:\n"
              << "  Time:   " << ms_ref << " ms\n"
              << "  GFLOPS: " << gflops_ref << "\n\n";

    // ---------------------- Ozaki2 GEMM ------------------------------------

    hipEvent_t start_oz, stop_oz;
    hipEventCreate(&start_oz);
    hipEventCreate(&stop_oz);

    hipEventRecord(start_oz);

    ozaki2_gemm(
        m, n, p,
        d_A, d_B, d_C_ozaki,
        (int)slices,
        moduli_all,
        moduli_products[slices - 1],
        partial_moduli[slices - 1],
        mod_inv[slices - 1]
    );

    hipEventRecord(stop_oz);
    hipEventSynchronize(stop_oz);

    float ms_oz = 0.0f;
    hipEventElapsedTime(&ms_oz, start_oz, stop_oz);

    double gflops_oz = (flops * 1e-9) / (ms_oz * 1e-3);

    std::cout << "Ozaki2 GEMM:\n"
              << "  Time:   " << ms_oz << " ms\n"
              << "  GFLOPS: " << gflops_oz << "\n\n";

    // ---------------------- Copy results back & compare --------------------

    hipMemcpy(h_C_ref.data(),   d_C_ref,   m * p * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_C_ozaki.data(), d_C_ozaki, m * p * sizeof(double), hipMemcpyDeviceToHost);

    // Print first block of both matrices
    print_matrix("C_ref (rocBLAS)",  h_C_ref,   (int)m, (int)p, PRINT_ROWS, PRINT_COLS);
    print_matrix("C_ozaki",          h_C_ozaki, (int)m, (int)p, PRINT_ROWS, PRINT_COLS);

    // Compute max absolute error
    double max_err = compute_max_abs_error(h_C_ref, h_C_ozaki);
    std::cout << "Max absolute error between rocBLAS and Ozaki2: "
              << std::setprecision(8) << max_err << "\n";

    // ---------------------- Cleanup ---------------------------------------

    hipEventDestroy(start_ref);
    hipEventDestroy(stop_ref);
    hipEventDestroy(start_oz);
    hipEventDestroy(stop_oz);

    rocblas_destroy_handle(handle);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C_ref);
    hipFree(d_C_ozaki);

    return 0;
}
