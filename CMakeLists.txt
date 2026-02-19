cmake_minimum_required(VERSION 3.23)
project(orbitIntegrator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ODE_BUILD_TESTS "Build tests" ON)
option(ODE_WERROR "Treat warnings as errors" OFF)

add_library(ode INTERFACE)
add_library(ode::ode ALIAS ode)
target_include_directories(ode INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (MSVC)
  target_compile_options(ode INTERFACE /W4 $<$<BOOL:${ODE_WERROR}>:/WX>)
else()
  target_compile_options(ode INTERFACE -Wall -Wextra -Wpedantic $<$<BOOL:${ODE_WERROR}>:-Werror>)
endif()

if(ODE_BUILD_TESTS)
  enable_testing()

  add_executable(ode_test_order tests/test_order.cpp)
  target_link_libraries(ode_test_order PRIVATE ode)
  add_test(NAME ode_test_order COMMAND ode_test_order)

  add_executable(ode_test_adaptive tests/test_adaptive.cpp)
  target_link_libraries(ode_test_adaptive PRIVATE ode)
  add_test(NAME ode_test_adaptive COMMAND ode_test_adaptive)

  add_executable(ode_test_backward tests/test_backward.cpp)
  target_link_libraries(ode_test_backward PRIVATE ode)
  add_test(NAME ode_test_backward COMMAND ode_test_backward)

  add_executable(ode_test_robustness tests/test_robustness.cpp)
  target_link_libraries(ode_test_robustness PRIVATE ode)
  add_test(NAME ode_test_robustness COMMAND ode_test_robustness)
endif()
