# Dependency policy for orbitIntegrator.

option(ODE_FETCH_DEPS "Allow fetching third-party dependencies from network" OFF)

if(ODE_FETCH_DEPS)
  include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

  CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    VERSION 11.0.2
    OPTIONS "FMT_TEST OFF" "FMT_DOC OFF" "FMT_INSTALL ON"
  )
else()
  find_package(fmt CONFIG QUIET)
endif()

if(NOT TARGET fmt::fmt)
  message(WARNING "fmt not found. Set ODE_FETCH_DEPS=ON or install fmt to enable standardized fmt-based logging output.")
endif()
