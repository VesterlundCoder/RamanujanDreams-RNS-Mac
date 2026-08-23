cmake_minimum_required(VERSION 3.20)
project(DreamsCPUEngine C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(dreams_cmf_cpu STATIC src/cmf_cpu.c)
target_include_directories(dreams_cmf_cpu PUBLIC include)
target_compile_options(dreams_cmf_cpu PRIVATE -O3 -fno-fast-math -ffp-contract=off)
target_link_libraries(dreams_cmf_cpu PUBLIC Threads::Threads m)

add_executable(dreams_fixed_start_cpu src/dreams_fixed_start_cpu.c)
target_include_directories(dreams_fixed_start_cpu PRIVATE include)
target_compile_options(dreams_fixed_start_cpu PRIVATE -O3 -fno-fast-math -ffp-contract=off)
target_link_libraries(dreams_fixed_start_cpu PRIVATE dreams_cmf_cpu Threads::Threads m)

add_executable(matmul6_bench src/matmul6_bench.c)
target_compile_options(matmul6_bench PRIVATE -O3)
target_link_libraries(matmul6_bench PRIVATE Threads::Threads m)

add_executable(rns_matmul6_bench src/rns_matmul6_bench.c)
target_compile_options(rns_matmul6_bench PRIVATE -O3)
target_link_libraries(rns_matmul6_bench PRIVATE Threads::Threads m)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_compile_options(dreams_cmf_cpu PRIVATE -mcpu=native)
  target_compile_options(dreams_fixed_start_cpu PRIVATE -mcpu=native)
  target_compile_options(matmul6_bench PRIVATE -mcpu=native)
  target_compile_options(rns_matmul6_bench PRIVATE -mcpu=native)
endif()
