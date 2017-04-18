cmake_minimum_required(VERSION 3.6)
project(fft_1d)

include_directories(../include)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
    common/inc/AOCLUtils/aocl_utils.h
    common/inc/AOCLUtils/opencl.h
    common/inc/AOCLUtils/options.h
    common/inc/AOCLUtils/scoped_ptrs.h
    common/src/AOCLUtils/opencl.cpp
    common/src/AOCLUtils/options.cpp
    fft_1d/cmake-build-debug/CMakeFiles/3.6.2/CompilerIdC/CMakeCCompilerId.c
    fft_1d/cmake-build-debug/CMakeFiles/3.6.2/CompilerIdCXX/CMakeCXXCompilerId.cpp
    fft_1d/cmake-build-debug/CMakeFiles/feature_tests.c
    fft_1d/cmake-build-debug/CMakeFiles/feature_tests.cxx
    fft_1d/host/src/main.cpp
        )

add_executable(fft_1d ${SOURCE_FILES})