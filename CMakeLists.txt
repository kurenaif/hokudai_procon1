cmake_minimum_required(VERSION 3.6)
project(hokudai_procon1)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        Source.cpp)

set(CMAKE_CXX_FLAGS "-Wall -fno-rtti")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

add_executable(hokudai_procon1 ${SOURCE_FILES})