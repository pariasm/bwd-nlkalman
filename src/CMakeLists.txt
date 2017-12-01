
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Enable C99
if (CMAKE_VERSION VERSION_LESS "3.1")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else ()
	set (CMAKE_C_STANDARD 99)
endif ()

# internal includes
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${ARGPARSE_INCLUDE_DIR})
include_directories(${IIO_INCLUDE_DIR})

add_executable(nlkalman-bwd main.c)
add_executable(test-dct test-dct.c)

link_directories(${CMAKE_BINARY_DIR}/lib)
target_link_libraries(nlkalman-bwd iio argparse)
target_link_libraries(test-dct iio argparse ${FFTWF_LIBRARIES})

