##
#  CMake script for poisson test solver.
##

# Set the name of the project and target:
SET(TARGET dipole_source)

# Declare all source files the target consists of:
SET(TARGET_SRC
  src/${TARGET}.cc
  # You can specify additional files here!
  )


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#find_package(GSL REQUIRED)
#include_directories(${GSL_INCLUDE_DIRS})
#set(LIBS ${LIBS} ${GSL_LIBRARIES})
#target_link_libraries(${TARGET} ${LIBS})


# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()


