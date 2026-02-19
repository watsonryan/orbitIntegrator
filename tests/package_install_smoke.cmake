if(NOT DEFINED ODE_SOURCE_DIR OR NOT DEFINED ODE_BINARY_DIR)
  message(FATAL_ERROR "ODE_SOURCE_DIR and ODE_BINARY_DIR are required")
endif()

set(INSTALL_PREFIX "${ODE_BINARY_DIR}/package-install")
set(CONSUMER_BUILD "${ODE_BINARY_DIR}/package-consumer-build")

file(REMOVE_RECURSE "${INSTALL_PREFIX}" "${CONSUMER_BUILD}")

execute_process(
  COMMAND ${CMAKE_COMMAND} --install "${ODE_BINARY_DIR}" --prefix "${INSTALL_PREFIX}"
  RESULT_VARIABLE install_rc
)
if(NOT install_rc EQUAL 0)
  message(FATAL_ERROR "Install step failed")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND}
    -S "${ODE_SOURCE_DIR}/tests/package_consumer"
    -B "${CONSUMER_BUILD}"
    -G Ninja
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}
  RESULT_VARIABLE cfg_rc
)
if(NOT cfg_rc EQUAL 0)
  message(FATAL_ERROR "Consumer configure failed")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${CONSUMER_BUILD}"
  RESULT_VARIABLE build_rc
)
if(NOT build_rc EQUAL 0)
  message(FATAL_ERROR "Consumer build failed")
endif()

execute_process(
  COMMAND "${CONSUMER_BUILD}/ode_package_consumer"
  RESULT_VARIABLE run_rc
)
if(NOT run_rc EQUAL 0)
  message(FATAL_ERROR "Consumer run failed")
endif()
