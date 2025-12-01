// Online C++ compiler to run C++ program online
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <iomanip>
#include <cmath>
using namespace std;

const uint8_t mod_inv[8][8] = {
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

// Matrix multiplication for 2x2 matrices
template <typename T>
vector<T> matrix_multiply(const vector<T>& A, const vector<T>& B) {
    vector<T> C(4, 0.0);
    
    C[0] = A[0] * B[0] + A[1] * B[2];
    C[1] = A[0] * B[1] + A[1] * B[3];
    C[2] = A[2] * B[0] + A[3] * B[2];
    C[3] = A[2] * B[1] + A[3] * B[3];
    
    return C;
}

int symm_mod(int a, int m) {
    int q = std::floor((long double)a / (long double)m + 0.5L);
    return a - q * m;
}

template <typename T>
void print_vector(const vector<T>& a, const string& name) {
    cout << name << ": { ";
    for (int i = 0; i < a.size() - 1; i++) {
        cout << static_cast<int>(a[i]) << ", ";
    }
    
    if (a.size() != 0) {
        cout << static_cast<int>(a[a.size() - 1]);
    }
    
    cout << "}" << endl;
}

void print_matrix(const string& name, const vector<double>& matrix,
                  int rows, int cols, int n_rows, int n_cols) {
    cout << name << " (first " << n_rows << "x" << n_cols << "):" << endl;
    for (int i = 0; i < std::min(n_rows, rows); ++i) {
        for (int j = 0; j < std::min(n_cols, cols); ++j) {
            cout << setw(8) << setprecision(11)
                      << static_cast<double>(matrix[i * cols + j]) << " ";
        }
        cout << endl;
    }
    cout << endl;
}

template <typename T>
void print_2x2_matrix(const string& name, const vector<T>& matrix) {
    cout << name << ":" << endl;
    cout << "[ " << matrix[0] << "  " << matrix[1] << " ]" << endl;
    cout << "[ " << matrix[2] << "  " << matrix[3] << " ]" << endl;
    cout << endl;
}


template <typename T>
void print_2x2_matrix_int8(const string& name, const vector<T>& matrix) {
    cout << name << ":" << endl;
    cout << "[ " << (int)(matrix[0]) << "  " <<
    (int)(matrix[1]) << " ]" << endl;
    cout << "[ " << (int)(matrix[2]) << "  " <<
    (int)(matrix[3]) << " ]" << endl;
    cout << endl;
}

int main() {
    // Define 2x2 matrices instead of single values
    vector<double> A_dbl = {12.125, 3.5, 2.25, 8.75};  // 2x2 matrix A
    vector<double> B_dbl = {13.4195, 1.25, 4.75, 6.125}; // 2x2 matrix B
    
    cout << "Original matrices:" << endl;
    print_2x2_matrix("Matrix A", A_dbl);
    print_2x2_matrix("Matrix B", B_dbl);
    
    // Compute reference result using floating-point
    vector<double> C_ref = matrix_multiply(A_dbl, B_dbl);
    print_2x2_matrix("Reference result (A * B)", C_ref);
    
    int mods[] = {255, 253, 251, 247, 239, 233, 229, 227};
    int N = 6;
    long long M = 1;
    for (int i = 0; i < N; i++) {
        M *= mods[i];
    }
    
    double fraction = (M / 2.0 - 1);
    int K = 0.5 * std::ilogb(fraction / 2.0);
    cout << "K = " << K << endl;
    
    vector<int64_t> A_trunc(N);
    vector<int64_t> B_trunc(N);
    
    for (int elem = 0; elem < 4; elem++) {
        int a_pow = K;
        A_trunc[elem] = static_cast<int>(A_dbl[elem] * pow(2, a_pow));
        
        int b_pow = K;
        B_trunc[elem] = static_cast<int>(B_dbl[elem] * pow(2, b_pow));
    }
    
    print_2x2_matrix("Matrix A trunc", A_trunc);
    print_2x2_matrix("Matrix B trunc", B_trunc);
    
    cout << "M = " << M << endl;
    
    // Compute modular representations for each matrix element
    vector<vector<int8_t>> A_mods(N, vector<int8_t>(4));
    vector<vector<int8_t>> B_mods(N, vector<int8_t>(4));
    
    for (int i = 0; i < N; i++) {
        for (int elem = 0; elem < 4; elem++) {
            A_mods[i][elem] = symm_mod(A_trunc[elem], mods[i]);
            B_mods[i][elem] = symm_mod(B_trunc[elem], mods[i]);
        }
        
        cout << "slice num: " << i << endl;
        
        print_2x2_matrix_int8("A mod matrix", A_mods[i]);
        print_2x2_matrix_int8("B mod matrix", B_mods[i]);
    }
    
    
    
    // Perform matrix multiplication in each modulus
    vector<vector<int>> C_prods(N, vector<int>(4, 0));
    
    for (int mod_idx = 0; mod_idx < N; mod_idx++) {
        // Extract matrices for current modulus
        vector<int> A_mod(4), B_mod(4);
        for (int i = 0; i < 4; i++) {
            A_mod[i] = A_mods[mod_idx][i];
            B_mod[i] = B_mods[mod_idx][i];
        }
        
        // Perform matrix multiplication for current modulus
        vector<int> C_mod = matrix_multiply(A_mod, B_mod);
        
        // Store results
        for (int i = 0; i < 4; i++) {
            C_prods[mod_idx][i] = static_cast<int>(C_mod[i]);
        }
        
        cout << "slice num: "  << mod_idx << endl;
        
        print_2x2_matrix("C mod matrix", C_mod);
    }
    
    // Apply modulus to each result
    vector<vector<uint8_t>> C_uint8(N, vector<uint8_t>(4));
    for (int i = 0; i < N; i++) {
        for (int elem = 0; elem < 4; elem++) {
            C_uint8[i][elem] = C_prods[i][elem] % mods[i];
        }
        
        cout << "slice num: "  << i << endl;
        print_2x2_matrix_int8("Uint matrix", C_uint8[i]);
    }
    
    // Reconstruct results using Chinese Remainder Theorem
    vector<double> C_result(4, 0.0);
    
    for (int i = 0; i < N; i++) {
        uint64_t c_accum = 0;
        for (int elem = 0; elem < N; elem++) {
            c_accum += C_uint8[i][elem] * M / mods[i] * mod_inv[N - 1][i];
            c_accum %= M;
            C_result[elem] = c_accum * pow(2, -(2 * K));
        }
    }
    
    cout << "Final result using Ozaki method:" << endl;
    print_2x2_matrix("Matrix C (A * B)", C_result);
    
    // Compare with reference
    cout << "Comparison with reference:" << endl;
    print_2x2_matrix("Reference", C_ref);
    print_2x2_matrix("Ozaki Result", C_result);
    
    cout << "Differences:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "Element " << i << ": " << abs(C_result[i] - C_ref[i]) << endl;
    }
    
    return 0;
}
