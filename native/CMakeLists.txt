cmake_minimum_required(VERSION 3.16)
project(revia_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimisation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(revia_native SHARED src/revia_native.cpp)

# Strip symbols in release for smaller binary
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(revia_native PRIVATE -s)
endif()

# Install to project root for easy ctypes loading
install(TARGETS revia_native DESTINATION ${CMAKE_SOURCE_DIR}/..)
