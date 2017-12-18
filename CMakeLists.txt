cmake_policy(SET CMP0048 NEW) # avoid annoing warnings

project(nlkalman-bwd VERSION 1.0 LANGUAGES C)
message(STATUS "Configuring main project: " ${CMAKE_PROJECT_NAME})

# flags #######################################################################

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -ggdb -fsanitize=address")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -ggdb")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -ggdb -O6 -w -s -ffast-math -fno-finite-math-only -fomit-frame-pointer -fstrength-reduce")

set(CMAKE_BUILD_TYPE Release)

# cmake global ################################################################

cmake_minimum_required(VERSION 2.8.2)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# dependencies ################################################################

find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_path (FFTW_INCLUDES fftw3.h)
find_library (FFTWF_LIBRARIES NAMES fftw3f)

# add project directories #####################################################

add_subdirectory(lib/iio)      # image i/o
add_subdirectory(lib/tvl1flow) # tvl1 optical flow
add_subdirectory(lib/argparse) # command line parsing
add_subdirectory(src)
add_subdirectory(scripts)

# configure cpack #############################################################

set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})

set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_IGNORE_FILES
"\\\\.*\\\\.swp"
"\\\\*\\\\.o"
"^${PROJECT_SOURCE_DIR}/build/"
"git")

set(CPACK_SOURCE_PACKAGE_FILE_NAME
	"${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}-src")

include(CPack)
# make package <-- TODO : configure properly
# make package_source
