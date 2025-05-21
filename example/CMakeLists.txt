cmake_minimum_required(VERSION 3.30)
project(bigint_example1)

add_executable(bigint_example1 main.cc
    ../bigint.h)

set_target_properties(bigint_example1 PROPERTIES CXX_STANDARD 17)
set_target_properties(bigint_example1 PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(bigint_example1 PRIVATE cxx_std_17)

target_include_directories(bigint_example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
