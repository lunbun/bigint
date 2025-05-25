cmake_minimum_required(VERSION 3.30)
project(bigint_example1)

add_executable(bigint_example1 main.cc
    ../bigint.h)

set_target_properties(bigint_example1 PROPERTIES CXX_STANDARD 17)
set_target_properties(bigint_example1 PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(bigint_example1 PRIVATE cxx_std_17)

target_compile_options(bigint_example1 PRIVATE
        "-fno-rtti"
        "$<$<CONFIG:DEBUG>:-O0;-g;-Wall;-Wextra;-Wpedantic>"
        "$<$<CONFIG:RELEASE>:-Ofast;-march=native>")
target_compile_definitions(bigint_example1 PRIVATE
        "$<$<CONFIG:RELEASE>:NDEBUG>")

target_include_directories(bigint_example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
