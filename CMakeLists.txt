cmake_minimum_required(VERSION 3.10)
project(MatrixOptimization)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Base source file
set(SOURCE_FILE main.cpp)

# Target: No optimization
add_executable(MatrixOptimization_no_optim ${SOURCE_FILE})
target_compile_options(MatrixOptimization_no_optim PRIVATE -O0)
target_compile_definitions(MatrixOptimization_no_optim PRIVATE NO_OPTIMIZATION)

# Target: Max optimization
add_executable(MatrixOptimization_max_optim ${SOURCE_FILE})
target_compile_options(MatrixOptimization_max_optim PRIVATE -O3 -march=native)
target_compile_definitions(MatrixOptimization_max_optim PRIVATE MAX_OPTIMIZATION)

# Target: SIMD optimization (with AVX2)
add_executable(MatrixOptimization_simd ${SOURCE_FILE})
target_compile_options(MatrixOptimization_simd PRIVATE -O3 -march=native -mavx2)
target_compile_definitions(MatrixOptimization_simd PRIVATE USE_SIMD)

