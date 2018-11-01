cmake_minimum_required(VERSION 3.11)
project(cuda_raytracer)

option( CUDA "use cuda as default target" OFF )
option( GL "use opengl as default visualizer" ON )

if (${CUDA})
	add_definitions(-DKOISHI_USE_CUDA)
endif ()

if (${GL})
	add_definitions(-DKOISHI_USE_GL)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.c)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses -Wno-comment")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pg -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

link_libraries(assimpd)
if (${GL})
	link_libraries(glfw3 ${OPENGL_LIBRARIES})
endif ()

add_compile_options(-Wno-narrowing)

add_executable(cr ${SOURCES})