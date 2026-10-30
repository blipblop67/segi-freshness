cmake_minimum_required(VERSION 3.22)

# Jetson Orin Nano / Orin NX / AGX Orin = SM 8.7. Override with
# -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs (e.g. 86 for a dev RTX card).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

project(segi_cuda_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cuda_ops bindings.cpp preprocess.cu color_stats.cu)
target_link_libraries(_cuda_ops PRIVATE CUDA::cudart)
target_compile_options(_cuda_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)

install(TARGETS _cuda_ops LIBRARY DESTINATION .)
