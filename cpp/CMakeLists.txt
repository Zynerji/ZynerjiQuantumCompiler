cmake_minimum_required(VERSION 3.18)
project(zynerji_qc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Spectra (header-only, vendored)
add_library(spectra INTERFACE)
target_include_directories(spectra INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library
add_library(zynerji_qc_core STATIC
    src/spectral_engine.cpp
    src/lap_solver.cpp
)
target_link_libraries(zynerji_qc_core PUBLIC Eigen3::Eigen spectra)
target_include_directories(zynerji_qc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(zynerji_qc_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(zynerji_qc_core PUBLIC ZQC_HAS_OPENMP)
endif()

# Python bindings
pybind11_add_module(_zynerji_qc_cpp src/bindings.cpp)
target_link_libraries(_zynerji_qc_cpp PRIVATE zynerji_qc_core)
