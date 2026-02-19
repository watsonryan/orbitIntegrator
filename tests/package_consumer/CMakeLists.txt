cmake_minimum_required(VERSION 3.23)
project(ode_package_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ode CONFIG REQUIRED)

add_executable(ode_package_consumer main.cpp)
target_link_libraries(ode_package_consumer PRIVATE ode::ode)
