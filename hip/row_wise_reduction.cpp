#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

// Matrix value range
constexpr double MATRIX_MIN_VAL = 0.1;
constexpr double MATRIX_MAX_VAL = 1000.0;

// Kernel to calculate row-wise maximums and store power-of-2 shifts
__global__ void calculate_shift_matrix_A(double* A, int P, int Q, int32_t* shifts, int slices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < P && idy < Q) {
        double row_max = fabs(A[idy * P + idx]);
        
        // Use warp shuffle to find maximum within warp
        for (int offset = 32; offset > 0; offset /= 2) {
            row_max = fmax(row_max, __shfl_down(row_max, offset));
        }
        
        // Let the first thread in each warp update the shift value
        if (threadIdx.x % 64 == 0) {
            // Calculate the power of 2 shift (ceiling of log2(row_max))
            int32_t shift_val = static_cast<int32_t>(ceil(log2(row_max)));
            atomicMax(&shifts[idy], shift_val);
        }
    }
}

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

void print_shifts(const std::string& name, const std::vector<int32_t>& shifts, int n_to_print) {
    std::cout << name << " (first " << n_to_print << " elements):" << std::endl;
    for (int i = 0; i < std::min(n_to_print, static_cast<int>(shifts.size())); ++i) {
        std::cout << "Row " << i << ": shift=" << shifts[i] 
                  << " (2^" << shifts[i] << " = " << (1 << shifts[i]) << ")" << std::endl;
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

// Helper function to check HIP errors
void checkHIPError(hipError_t err, const std::string& msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix dimensions
    const int P = 256;  // columns
    const int Q = 128;  // rows
    const int slices = 1;
    
    std::cout << "ROCm Matrix Shifts Example" << std::endl;
    std::cout << "Matrix dimensions: " << Q << " x " << P << std::endl;
    std::cout << "Finding row-wise maximums and calculating power-of-2 shifts" << std::endl << std::endl;

    // Host memory allocation
    std::vector<double> h_A(Q * P);
    std::vector<int32_t> h_shifts(Q, 0);  // Initialize with zeros

    // Generate random matrix
    generate_random_matrix(h_A, Q, P);
    print_matrix("Input Matrix A", h_A, Q, P, 5, 5);

    // Device memory allocation
    double* d_A = nullptr;
    int32_t* d_shifts = nullptr;
    
    checkHIPError(hipMalloc(&d_A, Q * P * sizeof(double)), "Failed to allocate device memory for A");
    checkHIPError(hipMalloc(&d_shifts, Q * sizeof(int32_t)), "Failed to allocate device memory for shifts");

    // Copy data to device
    checkHIPError(hipMemcpy(d_A, h_A.data(), Q * P * sizeof(double), hipMemcpyHostToDevice), 
                  "Failed to copy matrix A to device");
    checkHIPError(hipMemset(d_shifts, 0, Q * sizeof(int32_t)), 
                  "Failed to initialize shifts on device");

    // Kernel configuration
    dim3 blockDim(64, 4);  // 256 threads per block
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, 
                 (Q + blockDim.y - 1) / blockDim.y);

    std::cout << "Kernel configuration:" << std::endl;
    std::cout << "  Grid dimensions: " << gridDim.x << " x " << gridDim.y << std::endl;
    std::cout << "  Block dimensions: " << blockDim.x << " x " << blockDim.y << std::endl;
    std::cout << "  Total threads: " << gridDim.x * blockDim.x * gridDim.y * blockDim.y << std::endl << std::endl;

    // Launch kernel
    hipLaunchKernelGGL(calculate_shift_matrix_A, 
                       gridDim, blockDim, 0, 0, 
                       d_A, P, Q, d_shifts, slices);
    
    checkHIPError(hipGetLastError(), "Kernel launch failed");
    checkHIPError(hipDeviceSynchronize(), "Kernel execution failed");

    // Copy results back to host
    checkHIPError(hipMemcpy(h_shifts.data(), d_shifts, Q * sizeof(int32_t), hipMemcpyDeviceToHost),
                  "Failed to copy shifts from device");

    // Print results
    print_shifts("Power-of-2 Shifts", h_shifts, 10);

    // Calculate and print the actual scaling factors
    std::cout << "Scaling factors (2^shift) for first 10 rows:" << std::endl;
    for (int i = 0; i < std::min(10, Q); ++i) {
        double scale_factor = pow(2.0, h_shifts[i]);
        std::cout << "Row " << i << ": " << scale_factor << "x scaling" << std::endl;
    }
    std::cout << std::endl;

    // Verify results by checking manual calculation for first few rows
    std::cout << "Verification for first 3 rows:" << std::endl;
    for (int row = 0; row < std::min(3, Q); ++row) {
        double manual_max = 0.0;
        for (int col = 0; col < P; ++col) {
            manual_max = fmax(manual_max, fabs(h_A[row * P + col]));
        }
        int32_t manual_shift = static_cast<int32_t>(ceil(log2(manual_max)));
        std::cout << "Row " << row << ": Manual max=" << manual_max 
                  << ", Manual shift=" << manual_shift 
                  << ", GPU shift=" << h_shifts[row] 
                  << " (" << (manual_shift == h_shifts[row] ? "PASS" : "FAIL") << ")" << std::endl;
    }

    // Cleanup
    hipFree(d_A);
    hipFree(d_shifts);

    std::cout << std::endl << "Example completed successfully!" << std::endl;

    return 0;
}