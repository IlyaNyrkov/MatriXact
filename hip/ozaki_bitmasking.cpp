#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

// AMD headers
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

// NOTES:
// 0) Benchmark memory allocation and each kernel separately to find bottlenecks. Plot as histograms.
//
// 1) Matrix Slicing Kernel
// Matrix slice kernel using bitmasking is hard to parallelize over slices due to residual update
// CRT splitting can be more parallelizable and faster in general
// (CRT gives linear amount of GEMMs, Bitmasking gives squared
// Each thread computes all slices for one element sequentially
// Use warp shuffles to share scale factors within a warp
//
// 2) Many GEMM operations
// GEMM calls could be batched or parallelized via streams
// instead of launching each sequentially
// Depends on gpu architecture and tensor core count
// It might be faster to launch fewer larger GEMMs than many small ones
// It might be useless to launch multiple GEMMs in parallel because Matrix is too large
//
// 3) Final summation kernel can either be implemented via warp shuffles 
// or tensor cores C = C + αA*βB, α = 1, β = 0
//
// 4) Use shared memory instead of accessing global memory for slices and scale factors
//
// 5) Remove as much hipDeviceSynchronize() calls as possible
//    Potentially use events to synchronize between stream
//    Test different memory allocation types
//    Problem: It might not be possible to use shared memory everywhere instead of global memory 
//    due to large matrix sizes
// 6) Compare performance and accuracy with cuBLAS Ozaki scheme implementation
//
// FIND test case where LU factorization fails
// Triangular matrix TRMM

// Kernel launch parameters
const int BLOCK_SIZE = 128;

// Matrix sizes
const std::size_t M = 2<<11;
const std::size_t N = 2<<11;
const std::size_t P = 2<<11;

// Matrix print counts
const int PRINT_ROWS = 5;
const int PRINT_COLS = 5;

// Matrix value parameters
const double MATRIX_MIN_VAL = 0.0;
const double MATRIX_MAX_VAL = 10.0;

// Ozaki parameters
const int MAX_DOUBLE_MANTISSA_LEN = 53;
const int BITS_PER_SLICE = 8;
const int NUM_STREAMS = 4;

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
    
    double flops = 2.0 * M * N * P;
    double gflops = (flops * 1e-9) / (milliseconds * 1e-3);
    
    std::cout << name << " - Time: " << milliseconds << " ms, GFLOPS: " << gflops << std::endl;
    return milliseconds;
}

template<typename Func>
float benchmark_stage(Func&& stage_func, const std::string& name) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    stage_func();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    std::cout << "  " << name << ": " << milliseconds << " ms" << std::endl;
    return milliseconds;
}

// Ozaki methods
int calculate_num_slices(int N) {
    return std::min(8, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 1-8 slices
}

// Method for splitting FP64 matrix into int8 slices
__global__ void split_matrix_kernel(double* matrix, int M, int N, int k, 
                                   int8_t* slices, double* scale_factors) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    int idx = row * N + col;
    double scale_factor = scale_factors[row];
    double inversed_scale_factor = (scale_factor != 0.0) ? 1.0 / scale_factor : 0.0;
    
    // Calculate each slice element in one thread
    double aij = matrix[idx];

    for (int s = 0; s < k; s++) {
        int shift = MAX_DOUBLE_MANTISSA_LEN - BITS_PER_SLICE * s;
        double sigma = 0.75 * ldexp(scale_factor, shift);
        
        
        double top = (aij + sigma) - sigma;
        
        // Convert to int8 and store
        double normalized = top * inversed_scale_factor;
        double scaled = ldexp(normalized, -BITS_PER_SLICE * s);
        slices[s * M * N + idx] = static_cast<int8_t>(scaled);
        
        // Update residual on last slice 
        // This instruction doesn't allow parallelism over slices 
        // i.e. thread per element per slice
        if (s == k - 1) {
            aij -= top;
        }
    }
}


// Method for computing scale factors for each row
// Potential optimization: use warp shuffles to find max
__global__ void compute_scale_factors_kernel(const double* matrix, int M, int N, 
                                            double* scale_factors) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    // Find max absolute value in this row (!potential warp shuffle optimization!)
    double max_abs = 0.0;
    for (int col = 0; col < N; col++) {
        double abs_val = fabs(matrix[row * N + col]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    // Compute scale factor
    if (max_abs == 0.0 || !isfinite(max_abs)) {
        scale_factors[row] = ldexp(1.0, 1 - BITS_PER_SLICE);
    } else {
        int e = ceil(log2(max_abs));
        scale_factors[row] = ldexp(1.0, e + 1 - BITS_PER_SLICE);
    }
}

// Method for checking precision of original matrix after slicing
void reconstruct_fp64_matrix_from_int8_slices(const std::size_t M, const std::size_t N,
  int k, const int8_t* slices, double* scale_factor, double* result_matrix);

// Accumulate results from each AxB slices GEMM calls 
__global__ void accumulate_results_kernel(const int32_t* gemm_results, 
                                         const double* u, const double* v,
                                         double* result, int k, int M, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= M || j >= P) return;
    
    int idx = i * P + j;
    double accum = 0.0;
    int gemm_idx = 0;
    
    // Accumulate all triangular slice combinations
    // each thread computes one element of the result matrix
    // potential optimization: warp shuffle for u[i] and v[j]
    for (int s = 0; s < k; s++) {
        double scale_row = ldexp(u[i], BITS_PER_SLICE * s);
        for (int t = 0; t < k - s; t++) {
            double scale_col = ldexp(v[j], BITS_PER_SLICE * t);
            int32_t gemm_val = gemm_results[gemm_idx * M * P + idx];
            accum += scale_row * static_cast<double>(gemm_val) * scale_col;
            gemm_idx++;
        }
    }
    
    result[idx] = accum;
}


// Main Ozaki GEMM function
void ozaki_gemm(const std::size_t M, const std::size_t N, const std::size_t P,
               double* d_A, double* d_B, double* d_C) {
    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // Calculate number of slices
    int k = calculate_num_slices(N);
    
    // Allocate device memory
    double *d_u, *d_v;
    int8_t *d_slices_A, *d_slices_B;
    int32_t *d_gemm_results;
    
    hipMalloc(&d_u, M * sizeof(double));
    hipMalloc(&d_v, P * sizeof(double));
    hipMalloc(&d_slices_A, k * M * N * sizeof(int8_t));
    hipMalloc(&d_slices_B, k * N * P * sizeof(int8_t));
    
    int total_gemms = k * (k + 1) / 2;
    hipMalloc(&d_gemm_results, total_gemms * M * P * sizeof(int32_t));
    
    // Step 1: Split matrix A
    dim3 blockDim(16, 16);
    dim3 gridDimA((M + blockDim.x - 1) / blockDim.x, 
                  (N + blockDim.y - 1) / blockDim.y);
    
    // Compute scale factors for A
    compute_scale_factors_kernel<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, M, N, d_u);
    
    // Split A into slices
    split_matrix_kernel<<<gridDimA, blockDim>>>(d_A, M, N, k, d_slices_A, d_u);
    
    // Step 2: Split matrix B  
    dim3 gridDimB((N + blockDim.x - 1) / blockDim.x,
                  (P + blockDim.y - 1) / blockDim.y);
    
    // Compute scale factors for B
    compute_scale_factors_kernel<<<(P + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_B, P, N, d_v);
    
    // Split B into slices (transposed view: P x N)
    split_matrix_kernel<<<gridDimB, blockDim>>>(d_B, P, N, k, d_slices_B, d_v);
    
    // Step 3: Perform all GEMM operations
    int gemm_idx = 0;
    int32_t alpha = 1, beta = 0;
    
    for (int s = 0; s < k; s++) {
        for (int t = 0; t < k - s; t++) {
            const int8_t* slice_A = d_slices_A + s * M * N;
            const int8_t* slice_B = d_slices_B + t * N * P;
            int32_t* result_slice = d_gemm_results + gemm_idx * M * P;
            
            // Use rocBLAS for int8 GEMM (will use matrix cores)
            rocblas_gemm_ex(handle, 
                           rocblas_operation_none, rocblas_operation_none,
                           M, P, N, &alpha,
                           slice_A, rocblas_datatype_i8_r, M,
                           slice_B, rocblas_datatype_i8_r, N,
                           &beta,
                           result_slice, rocblas_datatype_i32_r, M,
                           result_slice, rocblas_datatype_i32_r, M,
                           rocblas_datatype_i32_r,
                           rocblas_gemm_algo_standard, 0, 0);
            
            gemm_idx++;
        }
    }
    
    // Wait for all GEMMs to complete
    
    // Step 4: Accumulate results
    dim3 gridDimC((M + blockDim.x - 1) / blockDim.x,
                  (P + blockDim.y - 1) / blockDim.y);
    
    accumulate_results_kernel<<<gridDimC, blockDim>>>(d_gemm_results, d_u, d_v,
                                                     d_C, k, M, P);    
    // Cleanup
    hipFree(d_u);
    hipFree(d_v);
    hipFree(d_slices_A);
    hipFree(d_slices_B);
    hipFree(d_gemm_results);
    rocblas_destroy_handle(handle);
}

// Ozaki scheme with stage benchmarking

void ozaki_gemm_benchmarked(const std::size_t M, const std::size_t N, const std::size_t P,
               double* d_A, double* d_B, double* d_C) {
    std::cout << "=== Ozaki GEMM Stage Breakdown ===" << std::endl;
    
    float total_time = 0.0;
    float stage_time = 0.0;
    
    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // Calculate number of slices
    int k = calculate_num_slices(N);
    std::cout << "Slices: " << k << " (total GEMMs: " << (k * (k + 1) / 2) << ")" << std::endl;
    
    // Stage 1: Memory Allocation
    stage_time = benchmark_stage([&]() {
        // Allocate device memory
        double *d_u, *d_v;
        int8_t *d_slices_A, *d_slices_B;
        int32_t *d_gemm_results;
        
        hipMalloc(&d_u, M * sizeof(double));
        hipMalloc(&d_v, P * sizeof(double));
        hipMalloc(&d_slices_A, k * M * N * sizeof(int8_t));
        hipMalloc(&d_slices_B, k * N * P * sizeof(int8_t));
        
        int total_gemms = k * (k + 1) / 2;
        hipMalloc(&d_gemm_results, total_gemms * M * P * sizeof(int32_t));
        
        // Store pointers for later use
        // (In real code, you'd need to store these in a struct)
    }, "Memory Allocation");
    total_time += stage_time;
    
    // Re-allocate for actual computation (since we can't capture in lambda easily)
    double *d_u, *d_v;
    int8_t *d_slices_A, *d_slices_B;
    int32_t *d_gemm_results;
    
    hipMalloc(&d_u, M * sizeof(double));
    hipMalloc(&d_v, P * sizeof(double));
    hipMalloc(&d_slices_A, k * M * N * sizeof(int8_t));
    hipMalloc(&d_slices_B, k * N * P * sizeof(int8_t));
    int total_gemms = k * (k + 1) / 2;
    hipMalloc(&d_gemm_results, total_gemms * M * P * sizeof(int32_t));
    
    // Step 1: Split matrix A
    stage_time = benchmark_stage([&]() {
        dim3 blockDim(16, 16);
        dim3 gridDimA((M + blockDim.x - 1) / blockDim.x, 
                      (N + blockDim.y - 1) / blockDim.y);
        
        // Compute scale factors for A
        compute_scale_factors_kernel<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, M, N, d_u);
        hipDeviceSynchronize();
        
        // Split A into slices
        split_matrix_kernel<<<gridDimA, blockDim>>>(d_A, M, N, k, d_slices_A, d_u);
        hipDeviceSynchronize();
    }, "Split Matrix A");
    total_time += stage_time;
    
    // Step 2: Split matrix B
    stage_time = benchmark_stage([&]() {
        dim3 blockDim(16, 16);
        dim3 gridDimB((P + blockDim.x - 1) / blockDim.x,  // Fixed: P rows for B
                      (N + blockDim.y - 1) / blockDim.y); // N columns for B
        
        // Compute scale factors for B
        compute_scale_factors_kernel<<<(P + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_B, P, N, d_v);
        hipDeviceSynchronize();
        
        // Split B into slices
        split_matrix_kernel<<<gridDimB, blockDim>>>(d_B, P, N, k, d_slices_B, d_v);
        hipDeviceSynchronize();
    }, "Split Matrix B");
    total_time += stage_time;
    
    // Step 3: Perform all GEMM operations
    stage_time = benchmark_stage([&]() {
    std::vector<hipStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        hipStreamCreate(&streams[i]);
    }
    
    int gemm_idx = 0;
    int32_t alpha = 1, beta = 0;
    // try batched aswell rocblas_gemm_batched_ex
    for (int s = 0; s < k; s++) {
        for (int t = 0; t < k - s; t++) {
            hipStream_t stream = streams[gemm_idx % NUM_STREAMS];
            rocblas_set_stream(handle, stream);
            
            const int8_t* slice_A = d_slices_A + s * M * N;
            const int8_t* slice_B = d_slices_B + t * N * P;
            int32_t* result_slice = d_gemm_results + gemm_idx * M * P;
            
            rocblas_gemm_ex(handle, 
                           rocblas_operation_none, rocblas_operation_none,
                           M, P, N, &alpha,
                           slice_A, rocblas_datatype_i8_r, M,
                           slice_B, rocblas_datatype_i8_r, N,
                           &beta,
                           result_slice, rocblas_datatype_i32_r, M,
                           result_slice, rocblas_datatype_i32_r, M,
                           rocblas_datatype_i32_r,
                           rocblas_gemm_algo_standard, 0, 0, rocblas_gemm_flags_pack_int8x4);
            
            gemm_idx++;
        }
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        hipStreamSynchronize(streams[i]);
        hipStreamDestroy(streams[i]);
    }
    }, "All GEMM Operations (Parallel)");

    total_time += stage_time;
    
    // Step 4: Accumulate results
    stage_time = benchmark_stage([&]() {
        dim3 blockDim(16, 16);
        dim3 gridDimC((M + blockDim.x - 1) / blockDim.x,
                      (P + blockDim.y - 1) / blockDim.y);
        
        accumulate_results_kernel<<<gridDimC, blockDim>>>(d_gemm_results, d_u, d_v,
                                                         d_C, k, M, P);
        hipDeviceSynchronize();
    }, "Accumulate Results");
    total_time += stage_time;
    
    // Memory Deallocation time
    stage_time = benchmark_stage([&]() {
        hipFree(d_u);
        hipFree(d_v);
        hipFree(d_slices_A);
        hipFree(d_slices_B);
        hipFree(d_gemm_results);
    }, "Memory Deallocation");
    total_time += stage_time;
    
    rocblas_destroy_handle(handle);
    
    std::cout << "Total Ozaki GEMM time: " << total_time << " ms" << std::endl;
    std::cout << "Theoretical peak GFLOPS: " << (2.0 * M * N * P * 1e-9) / (total_time * 1e-3) << std::endl;
    std::cout << "=================================" << std::endl;
}


int main() {
    // Initialize
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // Matrix sizes
    size_t size_A = M * P, size_B = P * N, size_C = M * N;
    
    // Host matrices
    std::vector<double> h_A(size_A), h_B(size_B), h_C_ozaki(size_C), h_C_rocblas(size_C);
    generate_random_matrix(h_A, M, P);
    generate_random_matrix(h_B, P, N);
    
    // Print input
    print_matrix("Matrix A", h_A, M, P, PRINT_ROWS, PRINT_COLS);
    print_matrix("Matrix B", h_B, P, N, PRINT_ROWS, PRINT_COLS);
    
    // Device memory
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size_A * sizeof(double));
    hipMalloc(&d_B, size_B * sizeof(double));
    hipMalloc(&d_C, size_C * sizeof(double));
    
    // Copy to device
    hipMemcpy(d_A, h_A.data(), size_A * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B * sizeof(double), hipMemcpyHostToDevice);
    
    std::cout << "=== Performance Comparison ===" << std::endl;
    std::cout << "Matrix size: " << M << " x " << N << " x " << P << std::endl << std::endl;

    // Benchmark rocBLAS double GEMM
    float rocblas_time = benchmark_gemm([&]() {
        double alpha = 1.0, beta = 0.0;
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                     M, N, P, &alpha, d_A, M, d_B, P, &beta, d_C, M);
    }, "rocBLAS FP64", handle, d_A, d_B, d_C);
    
    // Copy rocBLAS result  
    hipMemcpy(h_C_rocblas.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost);
    
    // Benchmark Ozaki GEMM
    ozaki_gemm_benchmarked(M, N, P, d_A, d_B, d_C);
    // Copy Ozaki result
    hipMemcpy(h_C_ozaki.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost);

    
    // Compare results
    std::cout << std::endl << "=== Accuracy Comparison ===" << std::endl;
    
    double max_error = 0.0, avg_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        double error = fabs(h_C_ozaki[i] - h_C_rocblas[i]);
        max_error = fmax(max_error, error);
        avg_error += error;
    }
    avg_error /= (M * N);
    
    std::cout << "Max absolute error: " << max_error << std::endl;
    std::cout << "Avg absolute error: " << avg_error << std::endl;
    //std::cout << "Speedup: " << rocblas_time / ozaki_time << "x" << std::endl;
    
    // Print sample results
    print_matrix("Ozaki Result (sample)", h_C_ozaki, M, N, PRINT_ROWS, PRINT_COLS);
    print_matrix("rocBLAS Result (sample)", h_C_rocblas, M, N, PRINT_ROWS, PRINT_COLS);
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    rocblas_destroy_handle(handle);
    
    std::cout << "Comparison completed!" << std::endl;
    return 0;
}