cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW) # avoid annoing warnings
project(imscript-lite C)

message(STATUS "Configuring project: " ${PROJECT_NAME})

###############################################################################

# NOTE: -fPIC is added so that the static library can be included into
#       shared libraries.
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -w -fPIC -O3 ")
set(CMAKE_BUILD_TYPE Release)

###############################################################################

# Enable C99
if (CMAKE_VERSION VERSION_LESS "3.1")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else ()
	set (CMAKE_C_STANDARD 99)
endif ()

###############################################################################

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###############################################################################

include_directories(${IIO_INCLUDE_DIR} src)

# iion executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(iion src/iion.c)
target_link_libraries(iion iio)

add_executable(plambda src/plambda.c)
target_link_libraries(plambda iio m)

add_executable(awgn src/awgn.c)
target_link_libraries(awgn iio m)

###############################################################################
