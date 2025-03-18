# Matrix Optimization Showcase

## Project Overview
This project demonstrates optimization techniques for a program that performs addition and subtraction of large dense matrices. The goal is to achieve minimal runtime by applying different levels of optimization:

1. **Baseline (Unoptimized)** - Compiled without any optimization flags (`-O0`).
2. **Compiler Optimized** - Compiled with standard compiler optimization flags (`-O3 -march=native`).
3. **Vectorized (SIMD Optimized)** - Enhanced with manual vectorization using SIMD (AVX2) intrinsics.
4. **Parallelized (OpenMP)** - Parallelized using OpenMP for multi-core execution.
5. **Parallelized + Vectorized (OpenMP + SIMD)** - Combines OpenMP parallelization with SIMD vectorization.
6. **Distributed Parallel (MPI + SIMD Intrinsics)** - Distributed across multiple processes using MPI, with SIMD vectorization within each process.

## Tested Configuration
This project was tested and proven to work with the following setup:
- **Compiler**: GCC (GCC) 14.2.1 20240910
- **CMake**: Version 3.30.5
- **Operating System Kernel**: Compiled and tested under 6.11.3-arch1-1
- **MPI Implementation**: Open MPI or similar MPI library (required for MPI build).

## Build and Run Instructions

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/Alexander-Chudnikov-Git/matrix-optimization-showcase.git
cd ./matrix-optimization-showcase
```

### Step 2: Configure and Compile the Project with CMake

You can configure and compile different versions of the project using CMake definitions.

**Basic Builds:**

* **Baseline (Unoptimized):**
  ```bash
  cmake -DOPTIMIZATION_LEVEL=BASELINE . -B ./build/baseline
  cmake --build ./build/baseline --config Release
  ```
  Run: `./build/baseline/MatrixOptimization_baseline`

* **Compiler Optimized:**
  ```bash
  cmake -DOPTIMIZATION_LEVEL=COMPILER_OPTIMIZED . -B ./build/compiler_optimized
  cmake --build ./build/compiler_optimized --config Release
  ```
  Run: `./build/compiler_optimized/MatrixOptimization_compiler_optimized`

* **Vectorized (SIMD Intrinsics):**
  ```bash
  cmake -DUSE_SIMD=ON -DOPTIMIZATION_LEVEL=SIMD . -B ./build/simd
  cmake --build ./build/simd --config Release
  ```
  Run: `./build/simd/MatrixOptimization_simd`


**Parallelized Builds:**

* **Parallelized (OpenMP):**
  ```bash
  cmake -DUSE_OPENMP=ON -DOPTIMIZATION_LEVEL=OPENMP . -B ./build/openmp
  cmake --build ./build/openmp --config Release
  ```
  Run: `./build/openmp/MatrixOptimization_openmp`

* **Parallelized + Vectorized (OpenMP + SIMD):**
  ```bash
  cmake -DUSE_OPENMP=ON -DUSE_SIMD=ON -DOPTIMIZATION_LEVEL=OPENMP_SIMD . -B ./build/openmp_simd
  cmake --build ./build/openmp_simd --config Release
  ```
  Run: `./build/openmp_simd/MatrixOptimization_openmp_simd`

* **Distributed Parallel (MPI + SIMD Intrinsics):**
  ```bash
  cmake -DUSE_MPI=ON -DUSE_SIMD=ON -DOPTIMIZATION_LEVEL=MPI_SIMD . -B ./build/mpi_simd
  cmake --build ./build/mpi_simd --config Release
  ```
  Run (Example for 4 MPI processes):
  ```bash
  mpirun -n 4 ./build/mpi_simd/MatrixOptimization_mpi_simd
  ```

**Note:**
*  Ensure you have an MPI implementation installed (e.g., Open MPI, MPICH) to build the MPI version.
*  The `OPTIMIZATION_LEVEL` CMake definition controls which optimization function is used in the main program for time measurement.
*  The `USE_OPENMP`, `USE_SIMD`, and `USE_MPI` CMake definitions enable the corresponding features in the code during compilation.

### Step 3: Run the Programs

Run the executables from the respective build directories to measure the runtime. For MPI, use `mpirun` or your MPI launcher to execute the distributed version.

Each target will display the runtime of the matrix operations, allowing for a performance comparison between different optimization levels.

## Results
Results for average execution time in seconds for each build target are as follows (these results may vary based on your hardware and system configuration):

| Optimization Level                       | CMake Definition                                    | Execution Time (s) | Speedup (vs Baseline) |
|------------------------------------------|-----------------------------------------------------|--------------------|-----------------------|
| 1. Baseline (Unoptimized)              | `-DOPTIMIZATION_LEVEL=BASELINE`                     | 0.251              | 1.00                  |
| 2. Compiler Optimized                   | `-DOPTIMIZATION_LEVEL=COMPILER_OPTIMIZED`           | 0.203              | 1.24                  |
| 3. Vectorized (SIMD Intrinsics)           | `-DUSE_SIMD=ON -DOPTIMIZATION_LEVEL=SIMD`           | 0.168              | 1.49                  |
| 4. Parallelized (OpenMP)                  | `-DUSE_OPENMP=ON -DOPTIMIZATION_LEVEL=OPENMP`         | 0.092              | 2.73                  |
| 5. Parallelized + Vectorized (OpenMP + SIMD) | `-DUSE_OPENMP=ON -DUSE_SIMD=ON -DOPTIMIZATION_LEVEL=OPENMP_SIMD` | 0.096              | 2.61                  |
| 6. Distributed Parallel (MPI + SIMD Intrinsics) | `-DUSE_MPI=ON -DUSE_SIMD=ON -DOPTIMIZATION_LEVEL=MPI_SIMD`     | 0.066              | 3.80                  |

**Note:** The "Execution Time" for MPI + SIMD Intrinsics represents the average time across all MPI processes.

## Conclusion
This project demonstrates significant performance gains achievable through a combination of compiler optimizations, manual SIMD vectorization using AVX2 instructions, and parallelization techniques (OpenMP and MPI).

- Compiler optimizations and SIMD vectorization provide noticeable speed improvements on a single core.
- OpenMP parallelization effectively utilizes multi-core processors for further performance enhancement.
- Distributed parallelization with MPI, especially when combined with SIMD, offers the highest performance by leveraging distributed computing resources.

The optimal optimization strategy depends on the specific hardware, problem size, and available resources. This showcase provides a practical example of how different optimization levels can impact performance for matrix operations.

## License
This project is licensed under the MIT License.
