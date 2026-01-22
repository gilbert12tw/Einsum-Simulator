cmake_minimum_required(VERSION 3.18)
project(einsum_nvqir LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for CLion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA-Q root path (default for Docker container)
if(NOT DEFINED CUDAQ_ROOT)
    set(CUDAQ_ROOT "/opt/nvidia/cudaq")
endif()

message(STATUS "CUDAQ_ROOT: ${CUDAQ_ROOT}")

# Include directories
# Check for local extracted headers first (for CLion code completion)
if(EXISTS "${CMAKE_SOURCE_DIR}/cudaq-include")
    message(STATUS "Using local CUDA-Q headers for code completion")
    include_directories(${CMAKE_SOURCE_DIR}/cudaq-include)
else()
    include_directories(${CUDAQ_ROOT}/include)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)

# Link directories
link_directories(${CUDAQ_ROOT}/lib)

# Source files
set(SOURCES
    src/EinsumSimulator.cpp
)

# Build the shared library
add_library(nvqir-einsum SHARED ${SOURCES})

# Link against NVQIR
target_link_libraries(nvqir-einsum
    nvqir
    cudaq-common
)

# Set output name (will produce libnvqir-einsum.so)
set_target_properties(nvqir-einsum PROPERTIES
    OUTPUT_NAME "nvqir-einsum"
    PREFIX "lib"
)

# Installation rules
install(TARGETS nvqir-einsum
    LIBRARY DESTINATION lib
)