project(cuda_raytracer)

option( CUDA "use cuda as default target" OFF )

if (${CUDA})
	add_compile_definitions(CUDA)
endif ()

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(cr ${SOURCES})