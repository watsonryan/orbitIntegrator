# CPM.cmake - package manager helper script
# Source: https://github.com/cpm-cmake/CPM.cmake
# Kept as a standalone module so top-level CMakeLists stays clean.

set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_HASH_SUM "")

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
    ${CPM_DOWNLOAD_LOCATION}
    EXPECTED_HASH SHA256=${CPM_HASH_SUM}
    TLS_VERIFY ON
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})
