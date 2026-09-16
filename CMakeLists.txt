cmake_minimum_required(VERSION 3.20)
project(PolicyForge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(policyforge_core
    src/gridworld.cpp
    src/algorithms.cpp
    src/report.cpp
)
target_include_directories(policyforge_core PUBLIC include)
target_link_libraries(policyforge_core PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(policyforge_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(policyforge_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(policyforge src/main.cpp)
target_link_libraries(policyforge PRIVATE policyforge_core)

enable_testing()
add_executable(policyforge_tests tests/test_main.cpp)
target_link_libraries(policyforge_tests PRIVATE policyforge_core)
add_test(NAME policyforge_tests COMMAND policyforge_tests)
set_tests_properties(policyforge_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
