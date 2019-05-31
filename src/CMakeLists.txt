
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

#add_executable(nlkalman-seq main-seq.c nlkalman.c)
add_executable(nlkalman-flt main-flt.c nlkalman.c)
add_executable(nlkalman-smo main-smo.c nlkalman.c)

link_directories(${CMAKE_BINARY_DIR}/lib)
#target_link_libraries(nlkalman-seq iio argparse ${FFTWF_LIBRARIES})
target_link_libraries(nlkalman-flt iio argparse ${FFTWF_LIBRARIES})
target_link_libraries(nlkalman-smo iio argparse ${FFTWF_LIBRARIES})

