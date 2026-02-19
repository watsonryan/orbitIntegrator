# Dependency policy for orbitIntegrator.

option(ODE_FETCH_DEPS "Allow fetching third-party dependencies from network" OFF)
option(ODE_ENABLE_EIGEN "Enable Eigen backend and Eigen convenience APIs" ON)

if(ODE_FETCH_DEPS)
  include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

  CPMAddPackage(
    NAME fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0
    OPTIONS "FMT_TEST OFF" "FMT_DOC OFF" "FMT_INSTALL ON"
  )

  CPMAddPackage(
    NAME spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
    OPTIONS "SPDLOG_BUILD_TESTS OFF" "SPDLOG_BUILD_EXAMPLE OFF" "SPDLOG_BUILD_BENCH OFF" "SPDLOG_FMT_EXTERNAL ON"
  )

  if(ODE_ENABLE_EIGEN)
    CPMAddPackage(
      NAME eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 5.0.1
      OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF" "BUILD_TESTING OFF"
    )
  endif()
else()
  find_package(fmt CONFIG QUIET)
  find_package(spdlog CONFIG QUIET)
endif()

if(NOT TARGET fmt::fmt)
  message(WARNING "fmt not found. Set ODE_FETCH_DEPS=ON or install fmt to enable standardized fmt-based logging output.")
endif()

if(ODE_ENABLE_EIGEN)
  find_package(Eigen3 5.0 QUIET CONFIG)

  if(NOT TARGET Eigen3::Eigen AND TARGET eigen)
    # CPM package source layout is header-only under source dir root.
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
  endif()

  if(NOT TARGET Eigen3::Eigen)
    message(WARNING "Eigen >= 5.0 not found. Set ODE_FETCH_DEPS=ON or install Eigen 5.x to enable Eigen APIs.")
  endif()
endif()

if(NOT TARGET spdlog::spdlog AND NOT TARGET spdlog::spdlog_header_only)
  message(FATAL_ERROR "spdlog is required. Set ODE_FETCH_DEPS=ON or install spdlog.")
endif()
