cmake_minimum_required(VERSION 3.10)
project(matrix_optimization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(SOURCE_FILES main.cpp)

function(add_optimized_executable target_name optimization_flags compile_definitions link_libraries)
    add_executable(${target_name} ${SOURCE_FILES})
    if(compile_definitions)
        target_compile_definitions(${target_name} PRIVATE ${compile_definitions})
    endif()
    if(optimization_flags)
        target_compile_options(${target_name} PRIVATE ${optimization_flags})
    endif()
    if(link_libraries)
        target_link_libraries(${target_name} PRIVATE ${link_libraries})
    endif()
endfunction()

# 1. Baseline (No Optimization)
add_optimized_executable(baseline
    ""
    "BASELINE"
    ""
)

# 2. Compiler Optimized
add_optimized_executable(compiler_optimized
    "-O3"
    "COMPILER_OPTIMIZED"
    ""
)

# 3. Vectorized (SIMD Intrinsics)
add_optimized_executable(simd_intrinsics
    "-O3;-march=native"
    "USE_SIMD"
    ""
)

# 4. Parallelized (OpenMP)
find_package(OpenMP)
if(OpenMP_FOUND)
    add_optimized_executable(openmp_parallel
        "-O3;${OpenMP_CXX_FLAGS}"
        "USE_OPENMP"
        "${OpenMP_CXX_LIBRARIES}"
    )
else()
    message(WARNING "OpenMP not found. OpenMP parallel version will not be built.")
endif()

# 5. Parallelized and Vectorized (OpenMP + SIMD Intrinsics)
if(OpenMP_FOUND)
    add_optimized_executable(openmp_simd
        "-O3;-march=native;${OpenMP_CXX_FLAGS}"
        "USE_OPENMP;USE_SIMD"
        "${OpenMP_CXX_LIBRARIES}"
    )
else()
    message(WARNING "OpenMP not found. OpenMP+SIMD version will not be built.")
endif()


# 6. Distributed Parallel (MPI + SIMD Intrinsics)
find_package(MPI REQUIRED)
if(MPI_FOUND)
    add_optimized_executable(mpi_simd
        "-O3;-march=native"
        "USE_MPI;USE_SIMD"
        "MPI::MPI_CXX"
    )
else()
    message(WARNING "MPI not found. MPI+SIMD version will not be built.")
endif()
