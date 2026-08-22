cmake_minimum_required(VERSION 3.20)
project(DreamsRNSMetal LANGUAGES CXX OBJCXX)
set(CMAKE_CXX_STANDARD 20)
option(BUILD_EXACT_RNS "Build the boost-dependent exact RNS binary" OFF)
if(BUILD_EXACT_RNS)
  add_executable(dreams_rns_metal src/main.mm)
  target_include_directories(dreams_rns_metal PRIVATE include src)
  target_compile_options(dreams_rns_metal PRIVATE -fobjc-arc)
  target_link_libraries(dreams_rns_metal PRIVATE "-framework Foundation" "-framework Metal")
endif()

add_executable(dreams_rns_df64 src/main_df64.mm)
target_include_directories(dreams_rns_df64 PRIVATE include src)
target_compile_options(dreams_rns_df64 PRIVATE -fobjc-arc -O2)
target_link_libraries(dreams_rns_df64 PRIVATE "-framework Foundation" "-framework Metal")

add_executable(dreams_rns_cmf_df64 src/main_cmf_df64.mm)
target_include_directories(dreams_rns_cmf_df64 PRIVATE include src)
target_compile_options(dreams_rns_cmf_df64 PRIVATE -fobjc-arc -O2)
target_link_libraries(dreams_rns_cmf_df64 PRIVATE "-framework Foundation" "-framework Metal")

# Fixed absolute-start optimizer. Propagates only two selected rows and
# emits two checkpoints, reducing both compute and I/O for trajectory
# optimization after a start point's limit has already been identified.
add_executable(dreams_fixed_start_df64 src/main_fixed_start_df64.mm)
target_include_directories(dreams_fixed_start_df64 PRIVATE include src)
target_compile_options(dreams_fixed_start_df64 PRIVATE -fobjc-arc -O2)
target_link_libraries(dreams_fixed_start_df64 PRIVATE "-framework Foundation" "-framework Metal")

add_executable(dreams_meijer_df64 src/main_meijer_df64.mm)
target_include_directories(dreams_meijer_df64 PRIVATE include src)
target_compile_options(dreams_meijer_df64 PRIVATE -fobjc-arc -O2)
target_link_libraries(dreams_meijer_df64 PRIVATE "-framework Foundation" "-framework Metal")
