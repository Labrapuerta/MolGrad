cmake_minimum_required(VERSION 3.16)

project(MolGrad VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(core
    src/core/tensor.cpp
    src/core/device.cpp
    src/core/storage.cpp
)

target_include_directories(core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

add_executable(test_tensor
    tests/main.cpp
)

target_link_libraries(test_tensor
    PRIVATE core
)