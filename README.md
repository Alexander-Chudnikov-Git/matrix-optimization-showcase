# Matrix Optimization Showcase

## Project Overview
This project demonstrates optimization techniques for a program that performs addition and subtraction of large dense matrices. The goal is to achieve minimal runtime by applying different levels of optimization:
1. **Unoptimized** - Compiled without any optimization.
2. **Compiler Optimized** - Compiled with standard compiler optimization flags.
3. **SIMD Optimized** - Enhanced with manual vectorization using SIMD (AVX2) intrinsics.

## Tested Configuration
This project was tested and proven to work with the following setup:
- **Compiler**: GCC (GCC) 14.2.1 20240910
- **CMake**: Version 3.30.5
- **Operating System Kernel**: Compiled and tested under 6.11.3-arch1-1

## Build and Run Instructions

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone [repository_url]
cd [repository_directory]
```

### Step 2: Compile the Project
To compile the project, run the following commands:
```bash
cmake . -B ./build/
cmake --build ./build
```

### Step 3: Run the Program
Run each executable to measure the runtime:
```bash
./build/MatrixOptimization_no_optim
./build/MatrixOptimization_max_optim
./build/MatrixOptimization_simd
```

Each target will display the runtime of the matrix operations, allowing for a performance comparison between the three levels of optimization.

## Results
Results for execution time in seconds for each build target are as follows:

| Optimization Level       | Execution Time (s) |
|--------------------------|--------------------|
| Unoptimized              | 0.487367           |
| Compiler Optimized       | 0.0675439          |
| SIMD Optimized           | 0.0564953          |

## Conclusion
This project illustrates the significant performance gains achievable through compiler optimizations and manual SIMD vectorization using AVX2 instructions. The SIMD-optimized version achieved an 8.63x speedup compared to the unoptimized version.

## License
This project is licensed under the MIT License.

