cmake_minimum_required(VERSION 3.23)
project(orbitIntegrator VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ODE_BUILD_TESTS "Build tests" ON)
option(ODE_BUILD_EXAMPLES "Build examples" ON)
option(ODE_BUILD_DOCS "Build Doxygen docs target" OFF)
option(ODE_WERROR "Treat warnings as errors" OFF)
option(ODE_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ODE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

include(cmake/dependencies.cmake)

add_library(ode INTERFACE)
add_library(ode::ode ALIAS ode)
target_compile_features(ode INTERFACE cxx_std_20)
target_include_directories(ode INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (MSVC)
  target_compile_options(ode INTERFACE /W4 $<$<BOOL:${ODE_WERROR}>:/WX>)
else()
  target_compile_options(ode INTERFACE -Wall -Wextra -Wpedantic $<$<BOOL:${ODE_WERROR}>:-Werror>)
  if(ODE_ENABLE_ASAN)
    target_compile_options(ode INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(ode INTERFACE -fsanitize=address)
  endif()
  if(ODE_ENABLE_UBSAN)
    target_compile_options(ode INTERFACE -fsanitize=undefined)
    target_link_options(ode INTERFACE -fsanitize=undefined)
  endif()
endif()

if(ODE_BUILD_EXAMPLES)
  add_executable(ode_two_body_example examples/two_body_example.cpp)
  target_link_libraries(ode_two_body_example PRIVATE ode $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>)
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

  add_executable(ode_test_state_array tests/test_state_array.cpp)
  target_link_libraries(ode_test_state_array PRIVATE ode)
  add_test(NAME ode_test_state_array COMMAND ode_test_state_array)

  add_executable(ode_test_algebra_adapters tests/test_algebra_adapters.cpp)
  target_link_libraries(ode_test_algebra_adapters PRIVATE ode)
  add_test(NAME ode_test_algebra_adapters COMMAND ode_test_algebra_adapters)

  add_executable(ode_test_dense_events tests/test_dense_events.cpp)
  target_link_libraries(ode_test_dense_events PRIVATE ode)
  add_test(NAME ode_test_dense_events COMMAND ode_test_dense_events)

  add_executable(ode_test_stiff tests/test_stiff.cpp)
  target_link_libraries(ode_test_stiff PRIVATE ode)
  add_test(NAME ode_test_stiff COMMAND ode_test_stiff)

  add_executable(ode_test_sundman tests/test_sundman.cpp)
  target_link_libraries(ode_test_sundman PRIVATE ode)
  add_test(NAME ode_test_sundman COMMAND ode_test_sundman)

  add_executable(ode_test_multistep tests/test_multistep.cpp)
  target_link_libraries(ode_test_multistep PRIVATE ode)
  add_test(NAME ode_test_multistep COMMAND ode_test_multistep)

  add_executable(ode_test_nordsieck tests/test_nordsieck.cpp)
  target_link_libraries(ode_test_nordsieck PRIVATE ode)
  add_test(NAME ode_test_nordsieck COMMAND ode_test_nordsieck)

  add_executable(ode_test_uncertainty tests/test_uncertainty.cpp)
  target_link_libraries(ode_test_uncertainty PRIVATE ode)
  add_test(NAME ode_test_uncertainty COMMAND ode_test_uncertainty)

  add_executable(ode_perf_benchmark tests/ode_perf_benchmark.cpp)
  target_link_libraries(ode_perf_benchmark PRIVATE ode $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>)

  add_executable(ode_method_compare_benchmark tests/ode_method_compare_benchmark.cpp)
  target_link_libraries(ode_method_compare_benchmark PRIVATE ode $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>)

  add_test(
    NAME ode_package_install_smoke
    COMMAND ${CMAKE_COMMAND}
      -DODE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DODE_BINARY_DIR=${CMAKE_BINARY_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/package_install_smoke.cmake
  )
endif()

if(ODE_BUILD_DOCS)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(ode_docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  else()
    message(STATUS "Doxygen not found; ode_docs target will not be generated.")
  endif()
endif()

install(TARGETS ode EXPORT odeTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/odeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/odeConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ode
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/odeConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT odeTargets
  FILE odeTargets.cmake
  NAMESPACE ode::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ode
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/odeConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/odeConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ode
)
