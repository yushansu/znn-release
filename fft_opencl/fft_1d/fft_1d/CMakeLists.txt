cmake_minimum_required(VERSION 3.6)
project(hello_world)

include_directories(../include)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(hello_world ${SOURCE_FILES})


