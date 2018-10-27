project(cuda_raytracer)

option( CUDA "use cuda as default target" OFF )

if (${CUDA})
	add_compile_definitions(CUDA)
endif ()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_compile_options(-Wno-narrowing)

add_executable(cr ${SOURCES})