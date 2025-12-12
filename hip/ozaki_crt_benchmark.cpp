#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>

// AMD headers
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

// TODO
// ADD Speedup numbers rocblas to ozaki

// ==========================================
// CONSTANTS & HELPERS
// ==========================================

const double MATRIX_MIN_VAL = 1.0; // Reduced range to prevent overflow in validation
const double MATRIX_MAX_VAL = 10.0;

#define CHECK_HIP(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_ROCBLAS(error) \
    if (error != rocblas_status_success) { \
        fprintf(stderr, "ROCBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

void generate_random_matrix(std::vector<double>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(MATRIX_MIN_VAL, MATRIX_MAX_VAL);
    for (int i = 0; i < rows * cols; ++i) matrix[i] = dist(gen);
}

double compute_max_abs_error(const std::vector<double>& A, const std::vector<double>& B) {
    double max_error = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double error = std::abs(A[i] - B[i]);
        if (error > max_error) max_error = error;
    }
    return max_error;
}

// ==========================================
// OZAKI KERNELS (Unchanged logic, just organization)
// ==========================================

const uint8_t moduli_all[] = {255, 253, 251, 247, 239, 233, 229, 227};

const uint64_t moduli_products[] = {
    255ULL, 64515ULL, 16193265ULL, 3999736455ULL, 
    955937012745ULL, 222733323969585ULL, 
    51005931189034965ULL, 11578346379910937055ULL
};

constexpr uint64_t partial_moduli[8][8] = {
    {1},
    {253ULL, 255ULL},
    {253ULL*251ULL, 255ULL*251ULL, 255ULL*253ULL},
    {253ULL*251ULL*247ULL, 255ULL*251ULL*247ULL, 255ULL*253ULL*247ULL, 255ULL*253ULL*251ULL},
    {253ULL*251ULL*247ULL*239ULL, 255ULL*251ULL*247ULL*239ULL, 255ULL*253ULL*247ULL*239ULL, 255ULL*253ULL*251ULL*239ULL, 255ULL*253ULL*251ULL*247ULL},
    {253ULL*251ULL*247ULL*239ULL*233ULL, 255ULL*251ULL*247ULL*239ULL*233ULL, 255ULL*253ULL*247ULL*239ULL*233ULL, 255ULL*253ULL*251ULL*239ULL*233ULL, 255ULL*253ULL*251ULL*247ULL*233ULL, 255ULL*253ULL*251ULL*247ULL*239ULL},
    {253ULL*251ULL*247ULL*239ULL*233ULL*229ULL, 255ULL*251ULL*247ULL*239ULL*233ULL*229ULL, 255ULL*253ULL*247ULL*239ULL*233ULL*229ULL, 255ULL*253ULL*251ULL*239ULL*233ULL*229ULL, 255ULL*253ULL*251ULL*247ULL*233ULL*229ULL, 255ULL*253ULL*251ULL*247ULL*239ULL*229ULL, 255ULL*253ULL*251ULL*247ULL*239ULL*233ULL},
    {0,0,0,0,0,0,0,0} // s=8 placeholder
};

const uint64_t mod_inv[8][8] = {
    {1},
    {127, 127},
    {32, 63, 157}, 
    {251, 116, 149, 238},
    {64, 64, 134, 32, 199},
    {113, 98, 132, 33, 166, 190},
    {182, 17, 245, 204, 79, 69, 181},
    {0,0,0,0,0,0,0,0}
};

__device__ int64_t symmetric_mod(int64_t a, uint64_t m) {
    int64_t q = std::floor((long double)a / (long double)m + 0.5L);
    return a - q * m;
}

__device__ int8_t symmetric_mod(int64_t a, uint8_t m) {
    int8_t q = std::floor((long double)a / (long double)m + 0.5L);
    return a - q * m;
}

__global__ void compute_row_max_exponent_A(const double* __restrict__ A, int rows, int cols, int32_t* __restrict__ row_exp) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;
    double v = fabs(A[row * cols + col]);
    int e = (int)ceil(log2(v));
    atomicMax(&row_exp[row], e);
}

__global__ void compute_col_max_exponent_B(const double* __restrict__ B, int rows, int cols, int32_t* __restrict__ col_exp) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;
    double v = fabs(B[row * cols + col]);
    int e = (int)ceil(log2(v));
    atomicMax(&col_exp[col], e);
}

__global__ void compute_shifts_from_exponents(const int32_t* __restrict__ exp_vec, int n, int K, int32_t* __restrict__ shifts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    shifts[i] = K - exp_vec[i];
}

__global__ void truncate_and_multiply_A(double* A, int32_t* shifts, int rows, int cols, int64_t* result) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    if (row < rows && col < cols) {
        result[row * cols + col] = static_cast<int64_t>(llrint(ldexp(A[row * cols + col], shifts[row])));
    }
}

__global__ void truncate_and_multiply_B(double* B, int32_t* shifts, int rows, int cols, int64_t* result) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        result[row * cols + col] = static_cast<int64_t>(llrint(ldexp(B[row * cols + col], shifts[col])));
    }
}

__global__ void compute_modulo_matrices(int64_t* matrix, uint8_t* moduli, int rows, int cols, int slices, int8_t* mod_matrices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        for (int s = 0; s < slices; ++s) {
            mod_matrices[s * rows * cols + idx] = symmetric_mod(matrix[idx], moduli[s]);
        }
    }
}

__global__ void convert_int32_uint8_modulo(int32_t* matrices, int rows, int cols, int slices, uint8_t* moduli, uint8_t* uint8_matrices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        for (int s = 0; s < slices; ++s) {
            int idx = s * rows * cols + row * cols + col;
            int32_t val = matrices[idx];
            int32_t mod_val = moduli[s];
            int32_t rem = val % mod_val;
            if (rem < 0) rem += mod_val;
            uint8_matrices[idx] = static_cast<uint8_t>(rem);
        }
    }
}

__global__ void accumulate_matrix_products(uint8_t* mod_C_matrices, int rows, int cols, int slices, int64_t* accumulated_C, int64_t* partial_moduli, int64_t* mod_inv, uint64_t matrix_product) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        for (int s = 0; s < slices; s++) {
            int64_t mod_C = static_cast<int64_t>(mod_C_matrices[s * rows * cols + idx]);
            accumulated_C[idx] += mod_C * partial_moduli[s] * mod_inv[s];
            accumulated_C[idx] %= matrix_product;
        }
    }
}

__global__ void inversely_scale_matrix(int64_t* C, int rows, int cols, int32_t* shift_vector_A, int32_t* shift_vector_B, uint64_t moduli_product, double* results) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        uint32_t e = shift_vector_A[row] + shift_vector_B[col];
        results[idx] = ldexp(static_cast<double>(C[idx]), -e);
    }
}

// ==========================================
// OZAKI WORKSPACE & EXECUTION
// ==========================================

// Struct to hold all pre-allocated memory to avoid malloc in the benchmark loop
struct OzakiWorkspace {
    int32_t *d_shift_A, *d_shift_B;
    int64_t *d_A_int, *d_B_int;
    int8_t *d_A_slices, *d_B_slices;
    int32_t *d_C_tc;
    uint8_t *d_C_resid;
    int64_t *d_C_accum;
    uint8_t *d_moduli;
    int64_t *d_partial_moduli, *d_mod_inv;
    int32_t *d_row_exp_A, *d_col_exp_B;
    
    // Config
    int slices;
    size_t sizeA, sizeB, sizeC;

    void allocate(size_t P, size_t Q, size_t R, int s) {
        slices = s;
        sizeA = P * Q;
        sizeB = Q * R;
        sizeC = P * R;

        CHECK_HIP(hipMalloc(&d_shift_A, P * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_shift_B, R * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_A_int, sizeA * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_B_int, sizeB * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_A_slices, slices * sizeA * sizeof(int8_t)));
        CHECK_HIP(hipMalloc(&d_B_slices, slices * sizeB * sizeof(int8_t)));
        CHECK_HIP(hipMalloc(&d_C_tc, slices * sizeC * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_C_resid, slices * sizeC * sizeof(uint8_t)));
        CHECK_HIP(hipMalloc(&d_C_accum, sizeC * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_moduli, slices * sizeof(uint8_t)));
        CHECK_HIP(hipMalloc(&d_partial_moduli, slices * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_mod_inv, slices * sizeof(int64_t)));
        CHECK_HIP(hipMalloc(&d_row_exp_A, P * sizeof(int32_t)));
        CHECK_HIP(hipMalloc(&d_col_exp_B, R * sizeof(int32_t)));
    }

    void free() {
        hipFree(d_shift_A); hipFree(d_shift_B);
        hipFree(d_A_int); hipFree(d_B_int);
        hipFree(d_A_slices); hipFree(d_B_slices);
        hipFree(d_C_tc); hipFree(d_C_resid);
        hipFree(d_C_accum);
        hipFree(d_moduli); hipFree(d_partial_moduli); hipFree(d_mod_inv);
        hipFree(d_row_exp_A); hipFree(d_col_exp_B);
    }
};

void run_ozaki(rocblas_handle handle, OzakiWorkspace& ws, 
               double* d_A, double* d_B, double* d_C_out,
               size_t P, size_t Q, size_t R, 
               const uint64_t M_prod) 
{
    // Setup constants on device
    CHECK_HIP(hipMemcpy(ws.d_moduli, moduli_all, ws.slices * sizeof(uint8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(ws.d_partial_moduli, partial_moduli[ws.slices-1], ws.slices * sizeof(int64_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(ws.d_mod_inv, mod_inv[ws.slices-1], ws.slices * sizeof(int64_t), hipMemcpyHostToDevice));
    
    CHECK_HIP(hipMemset(ws.d_shift_A, 0, P * sizeof(int32_t)));
    CHECK_HIP(hipMemset(ws.d_shift_B, 0, R * sizeof(int32_t)));
    CHECK_HIP(hipMemset(ws.d_row_exp_A, 0, P * sizeof(int32_t)));
    CHECK_HIP(hipMemset(ws.d_col_exp_B, 0, R * sizeof(int32_t)));
    CHECK_HIP(hipMemset(ws.d_C_accum, 0, ws.sizeC * sizeof(int64_t)));

    dim3 block(16, 16);
    dim3 gridA((P + block.x - 1) / block.x, (Q + block.y - 1) / block.y);
    dim3 gridB((Q + block.x - 1) / block.x, (R + block.y - 1) / block.y);
    dim3 gridC((P + block.x - 1) / block.x, (R + block.y - 1) / block.y);

    // Calculate K (Internal Logic)
    double Md = static_cast<double>(M_prod);
    double Qd = static_cast<double>(Q);
    double frac = (Md / 2.0 - 1.0) / Qd;
    int32_t K = static_cast<int32_t>(std::floor(0.5 * std::log2(frac)));

    // 1. Exponents
    compute_row_max_exponent_A<<<gridA, block>>>(d_A, P, Q, ws.d_row_exp_A);
    compute_col_max_exponent_B<<<gridB, block>>>(d_B, Q, R, ws.d_col_exp_B);

    // 2. Shifts
    {
        int threads = 256;
        compute_shifts_from_exponents<<<(P + threads - 1)/threads, threads>>>(ws.d_row_exp_A, P, K, ws.d_shift_A);
        compute_shifts_from_exponents<<<(R + threads - 1)/threads, threads>>>(ws.d_col_exp_B, R, K, ws.d_shift_B);
    }

    // 3. Truncate
    truncate_and_multiply_A<<<gridA, block>>>(d_A, ws.d_shift_A, P, Q, ws.d_A_int);
    truncate_and_multiply_B<<<gridB, block>>>(d_B, ws.d_shift_B, Q, R, ws.d_B_int);

    // 4. Slices
    compute_modulo_matrices<<<gridA, block>>>(ws.d_A_int, ws.d_moduli, P, Q, ws.slices, ws.d_A_slices);
    compute_modulo_matrices<<<gridB, block>>>(ws.d_B_int, ws.d_moduli, Q, R, ws.slices, ws.d_B_slices);

    // 5. INT8 GEMM (Batching could optimize this, but loop for now)
    const int32_t alpha = 1, beta = 0;
    for (int t = 0; t < ws.slices; ++t) {
        rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_transpose,
            P, R, Q, &alpha,
            ws.d_B_slices + t * ws.sizeB, rocblas_datatype_i8_r, Q, // B^T
            ws.d_A_slices + t * ws.sizeA, rocblas_datatype_i8_r, P, // A^T
            &beta,
            ws.d_C_tc + t * ws.sizeC, rocblas_datatype_i32_r, P,
            ws.d_C_tc + t * ws.sizeC, rocblas_datatype_i32_r, P,
            rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0);
    }

    // 6. Residues
    convert_int32_uint8_modulo<<<gridC, block>>>(ws.d_C_tc, P, R, ws.slices, ws.d_moduli, ws.d_C_resid);

    // 7. Accumulate
    accumulate_matrix_products<<<gridC, block>>>(ws.d_C_resid, P, R, ws.slices, ws.d_C_accum, ws.d_partial_moduli, ws.d_mod_inv, M_prod);

    // 8. Scale Back
    inversely_scale_matrix<<<gridC, block>>>(ws.d_C_accum, P, R, ws.d_shift_A, ws.d_shift_B, M_prod, d_C_out);
}

// ==========================================
// MAIN BENCHMARK LOOP
// ==========================================

int main() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    std::vector<int> matrix_sizes = {1024, 2048, 3072, 4096, 5120,
                                     6144, 7168, 8192, 9216, 10240,
                                     11264, 12288, 13312, 14336, 15360,
                                     16396, 17408, 18432};
    std::vector<int> split_configs = {3, 4, 5, 6, 7};
    
    // Benchmark Config
    const int warmup_iters = 3;
    const int bench_iters = 5; // Reduced for 8192 size, increase for accuracy if needed
    
    // CSV Header
    std::cout << "Algorithm,Size,Split,TimeMS,GFLOPS,MaxAbsError" << std::endl;

    for (int N : matrix_sizes) {
        int M = N, K = N; // Square for now
        size_t size_bytes = (size_t)N * N * sizeof(double);

        // Host Alloc
        std::vector<double> h_A(N * N);
        std::vector<double> h_B(N * N);
        std::vector<double> h_C_ref(N * N);
        std::vector<double> h_C_ozaki(N * N);

        generate_random_matrix(h_A, N, N);
        generate_random_matrix(h_B, N, N);

        // Device Alloc (Inputs)
        double *d_A, *d_B, *d_C_ref, *d_C_ozaki;
        CHECK_HIP(hipMalloc(&d_A, size_bytes));
        CHECK_HIP(hipMalloc(&d_B, size_bytes));
        CHECK_HIP(hipMalloc(&d_C_ref, size_bytes));
        CHECK_HIP(hipMalloc(&d_C_ozaki, size_bytes));

        CHECK_HIP(hipMemcpy(d_A, h_A.data(), size_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_B, h_B.data(), size_bytes, hipMemcpyHostToDevice));

        double ops = 2.0 * (double)N * (double)N * (double)N;

        // ------------------------------------------
        // 1. BASELINE: rocBLAS Double GEMM
        // ------------------------------------------
        {
            double alpha = 1.0, beta = 0.0;
            // Warmup
            for(int i=0; i<warmup_iters; i++) {
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, 
                              N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_ref, N);
            }
            hipDeviceSynchronize();

            // Timing
            hipEvent_t start, stop;
            hipEventCreate(&start); hipEventCreate(&stop);
            hipEventRecord(start);
            
            for(int i=0; i<bench_iters; i++) {
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, 
                              N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_ref, N);
            }
            
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            float ms = 0; 
            hipEventElapsedTime(&ms, start, stop);
            float avg_ms = ms / bench_iters;
            double gflops = (ops * 1e-9) / (avg_ms * 1e-3);
            
            std::cout << "rocBLAS," << N << ",0," << avg_ms << "," << gflops << ",0.0" << std::endl;
            hipEventDestroy(start); hipEventDestroy(stop);
            
            // Bring reference back for error checking later
            if(N <= 4096) { // Only check accuracy for fit-in-RAM sizes
                CHECK_HIP(hipMemcpy(h_C_ref.data(), d_C_ref, size_bytes, hipMemcpyDeviceToHost));
            }
        }

        // ------------------------------------------
        // 2. OZAKI BENCHMARKS
        // ------------------------------------------
        for(int s : split_configs) {
            OzakiWorkspace ws;
            ws.allocate(N, N, N, s);

            // Warmup
            for(int i=0; i<warmup_iters; i++) {
                run_ozaki(handle, ws, d_A, d_B, d_C_ozaki, N, N, N, moduli_products[s-1]);
            }
            hipDeviceSynchronize();

            // Alternating Loop (Best Practice: Heat consistency)
            float total_ozaki_time = 0;
            
            hipEvent_t start, stop;
            hipEventCreate(&start); hipEventCreate(&stop);

            for(int i=0; i<bench_iters; i++) {
                // A. Run rocBLAS (Untimed - just to keep GPU busy/hot)
                double alpha = 1.0, beta = 0.0;
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, 
                              N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_ref, N);
                
                // B. Run Ozaki (Timed)
                hipEventRecord(start);
                run_ozaki(handle, ws, d_A, d_B, d_C_ozaki, N, N, N, moduli_products[s-1]);
                hipEventRecord(stop);
                hipEventSynchronize(stop);
                
                float ms;
                hipEventElapsedTime(&ms, start, stop);
                total_ozaki_time += ms;
            }

            hipEventDestroy(start); hipEventDestroy(stop);

            float avg_ms = total_ozaki_time / bench_iters;
            double gflops = (ops * 1e-9) / (avg_ms * 1e-3);

            // Error Checking
            double max_err = 0.0;
            if(N <= 20000) { // Check error for reasonable sizes
                CHECK_HIP(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, size_bytes, hipMemcpyDeviceToHost));
                max_err = compute_max_abs_error(h_C_ref, h_C_ozaki);
            }

            std::cout << "Ozaki," << N << "," << s << "," << avg_ms << "," << gflops << "," << max_err << std::endl;

            ws.free();
        }

        // Cleanup Size
        hipFree(d_A); hipFree(d_B); hipFree(d_C_ref); hipFree(d_C_ozaki);
    }

    rocblas_destroy_handle(handle);
    return 0;
}