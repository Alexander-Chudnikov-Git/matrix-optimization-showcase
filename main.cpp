#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

#ifdef USE_SIMD
#include <immintrin.h>
#endif

// Define the matrix size
const int MATRIX_SIZE = 10240;

// Function to initialize matrices with random values
void initialize_matrices(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
    {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Unoptimized matrix addition and subtraction
volatile void matrix_add_sub_unoptimized(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, std::vector<float>& D)
{
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
    {
        D[i] = A[i] + B[i] - C[i];
    }
}

// SIMD-optimized matrix addition and subtraction
#ifdef USE_SIMD
void matrix_add_sub_simd(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, std::vector<float>& D)
{
    int i = 0;
    for (; i <= MATRIX_SIZE * MATRIX_SIZE - 8; i += 8)
    {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 c = _mm256_loadu_ps(&C[i]);

        __m256 result = _mm256_add_ps(a, b);
        result = _mm256_sub_ps(result, c);

        _mm256_storeu_ps(&D[i], result);
    }

    for (; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
    {
        D[i] = A[i] + B[i] - C[i];
    }
}
#endif

template <typename Func>
void measure_time(const std::string& label, Func func, const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, std::vector<float>& D)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(A, B, C, D);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << label << " time: " << elapsed.count() << " seconds.\n";
}

int main()
{
    // Initialize matrices
    std::vector<float> A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> B(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> C(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> D(MATRIX_SIZE * MATRIX_SIZE);

    initialize_matrices(A, B, C);

    #ifdef NO_OPTIMIZATION
    // Measure time for unoptimized version
    measure_time("Unoptimized", matrix_add_sub_unoptimized, A, B, C, D);
    #endif

    #ifdef MAX_OPTIMIZATION
    // Measure time for compiler-optimized version
    measure_time("Compiler Optimized", matrix_add_sub_unoptimized, A, B, C, D);
    #endif

    #ifdef USE_SIMD
    // Measure time for SIMD-optimized version
    measure_time("SIMD Optimized", matrix_add_sub_simd, A, B, C, D);
    #endif

    return 0;
}

