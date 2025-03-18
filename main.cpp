#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric> // for std::iota
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_SIMD
#include <immintrin.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

uint64_t MATRIX_SIZE = 10240;

void initialize_matrices(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
	float rand_max_f = static_cast<float>(RAND_MAX);
	for (uint64_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
	{
		A[i] = static_cast<float>(rand()) / rand_max_f;
		B[i] = static_cast<float>(rand()) / rand_max_f;
		C[i] = static_cast<float>(rand()) / rand_max_f;
	}
}

// 1. Baseline (Unoptimized)
volatile void matrix_add_sub_baseline(const std::vector<float>& A,
									  const std::vector<float>& B,
									  const std::vector<float>& C,
									  std::vector<float>&		D)
{
	for (uint64_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
	{
		D[i] = A[i] + B[i] - C[i];
	}
}

// 2. Compiler Optimized (same code as baseline, optimized by compiler flags)
volatile void matrix_add_sub_compiler_optimized(const std::vector<float>& A,
												const std::vector<float>& B,
												const std::vector<float>& C,
												std::vector<float>&		  D)
{
	for (uint64_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
	{
		D[i] = A[i] + B[i] - C[i];
	}
}

// 3. Vectorized (SIMD Intrinsics)
#ifdef USE_SIMD
volatile void matrix_add_sub_simd(const std::vector<float>& A,
								  const std::vector<float>& B,
								  const std::vector<float>& C,
								  std::vector<float>&		D)
{
	uint64_t i			= 0;
	uint64_t total_size = MATRIX_SIZE * MATRIX_SIZE;
	for (; i <= total_size - 8; i += 8)
	{
		__m256 a = _mm256_loadu_ps(&A[i]);
		__m256 b = _mm256_loadu_ps(&B[i]);
		__m256 c = _mm256_loadu_ps(&C[i]);

		__m256 result = _mm256_add_ps(a, b);
		result		  = _mm256_sub_ps(result, c);

		_mm256_storeu_ps(&D[i], result);
	}
	for (; i < total_size; ++i)
	{ // остаток
		D[i] = A[i] + B[i] - C[i];
	}
}
#endif

// 4. Parallelized (OpenMP)
#ifdef USE_OPENMP
volatile void matrix_add_sub_parallel_openmp(const std::vector<float>& A,
											 const std::vector<float>& B,
											 const std::vector<float>& C,
											 std::vector<float>&	   D)
{
#pragma omp parallel for
	for (uint64_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
	{
		D[i] = A[i] + B[i] - C[i];
	}
}
#endif

// 5. Parallelized and Vectorized (OpenMP + SIMD Intrinsics)
#ifdef USE_OPENMP
#ifdef USE_SIMD
volatile void matrix_add_sub_parallel_simd_openmp(const std::vector<float>& A,
												  const std::vector<float>& B,
												  const std::vector<float>& C,
												  std::vector<float>&		D)
{
	uint64_t total_size = MATRIX_SIZE * MATRIX_SIZE;
#pragma omp parallel for
	for (uint64_t block_start = 0; block_start < total_size; block_start += 8)
	{
		uint64_t i = block_start;
		if (i <= total_size - 8)
		{
			__m256 a = _mm256_loadu_ps(&A[i]);
			__m256 b = _mm256_loadu_ps(&B[i]);
			__m256 c = _mm256_loadu_ps(&C[i]);

			__m256 result = _mm256_add_ps(a, b);
			result		  = _mm256_sub_ps(result, c);

			_mm256_storeu_ps(&D[i], result);
		}
		else
		{
			for (uint64_t j = i; j < total_size; ++j)
			{ 
				D[j] = A[j] + B[j] - C[j];
			}
		}
	}
}
#endif
#endif

// 6. Distributed Parallel (MPI + SIMD Intrinsics)
#ifdef USE_MPI
#ifdef USE_SIMD
void matrix_add_sub_mpi_simd(const std::vector<float>& A,
							 const std::vector<float>& B,
							 const std::vector<float>& C,
							 std::vector<float>&	   D,
							 uint64_t				   rank,
							 uint64_t				   size,
							 uint64_t				   local_size) 
{
	uint64_t start_index =
		rank * (MATRIX_SIZE * MATRIX_SIZE / size); 
	uint64_t end_index = start_index + local_size;
	uint64_t i		   = 0;						  

	for (; i <= local_size - 8; i += 8) 
	{
		__m256 a = _mm256_loadu_ps(&A[i]);
		__m256 b = _mm256_loadu_ps(&B[i]);
		__m256 c = _mm256_loadu_ps(&C[i]);

		__m256 result = _mm256_add_ps(a, b);
		result		  = _mm256_sub_ps(result, c);

		_mm256_storeu_ps(&D[i], result);
	}

	for (; i < local_size; ++i) 
	{
		D[i] = A[i] + B[i] - C[i];
	}
}
#endif
#endif

template<typename Func>
void measure_time(const std::string&		label,
				  Func						func,
				  const std::vector<float>& A,
				  const std::vector<float>& B,
				  const std::vector<float>& C,
				  std::vector<float>&		D)
{
	auto start = std::chrono::high_resolution_clock::now();
	func(A, B, C, D);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = end - start;

	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();

	auto sec = ms / 1000;
	ms %= 1000;
	us %= 1000;
	ns %= 1000;

	std::cout << label << " time: " << sec << "s " << ms << "ms " << us << "μs " << ns << "ns\n";
}

int main(int argc, char* argv[])
{
	std::vector<float> A_full(MATRIX_SIZE * MATRIX_SIZE);
	std::vector<float> B_full(MATRIX_SIZE * MATRIX_SIZE);
	std::vector<float> C_full(MATRIX_SIZE * MATRIX_SIZE);
	std::vector<float> D_full(MATRIX_SIZE * MATRIX_SIZE);

	initialize_matrices(A_full, B_full, C_full);

#ifdef USE_MPI
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::cout << "Process: " << rank << ", size: " << size << "\n";

	uint64_t total_size = MATRIX_SIZE * MATRIX_SIZE;
	uint64_t chunk_size = total_size / size;
	uint64_t local_size = chunk_size;

	if (rank == size - 1)
	{
		local_size = total_size - rank * chunk_size;
	}

	std::vector<float> local_A(local_size);
	std::vector<float> local_B(local_size);
	std::vector<float> local_C(local_size);
	std::vector<float> local_D(local_size);

	MPI_Scatter(&A_full[0], chunk_size, MPI_FLOAT, &local_A[0], local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&B_full[0], chunk_size, MPI_FLOAT, &local_B[0], local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&C_full[0], chunk_size, MPI_FLOAT, &local_C[0], local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	std::cout << "Rank " << rank << " of " << size << ", local_size: " << local_size << std::endl;

#ifdef USE_SIMD
	measure_time(
		"6. MPI + SIMD Intrinsics",
		[&](const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, std::vector<float>& D) {
			matrix_add_sub_mpi_simd(A, B, C, D, rank, size, local_size); // pass local_size here
		},
		local_A, local_B, local_C, local_D);
#endif

	MPI_Finalize();

#else

	std::vector<float> D(MATRIX_SIZE * MATRIX_SIZE);
	if (argc > 1)
	{
		MATRIX_SIZE = std::stoul(argv[1]);
		A_full.resize(MATRIX_SIZE * MATRIX_SIZE);
		B_full.resize(MATRIX_SIZE * MATRIX_SIZE);
		C_full.resize(MATRIX_SIZE * MATRIX_SIZE);
		D.resize(MATRIX_SIZE * MATRIX_SIZE);
		initialize_matrices(A_full, B_full, C_full);
	}

#ifdef BASELINE
	measure_time("1. Baseline (Unoptimized)", matrix_add_sub_baseline, A_full, B_full, C_full, D);
#endif

#ifdef COMPILER_OPTIMIZED
	measure_time("2. Compiler Optimized", matrix_add_sub_compiler_optimized, A_full, B_full, C_full, D);
#endif

#ifdef USE_SIMD
	measure_time("3. Vectorized (SIMD Intrinsics)", matrix_add_sub_simd, A_full, B_full, C_full, D);
#endif

#ifdef USE_OPENMP
	measure_time("4. Parallelized (OpenMP)", matrix_add_sub_parallel_openmp, A_full, B_full, C_full, D);
#endif

#ifdef USE_OPENMP
#ifdef USE_SIMD
	measure_time("5. Parallelized + Vectorized (OpenMP + SIMD)", matrix_add_sub_parallel_simd_openmp, A_full, B_full, C_full,
				 D);
#endif
#endif

#endif // USE_MPI

	return 0;
}
