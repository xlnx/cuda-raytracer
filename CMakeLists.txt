cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 11)
project(cuda_raytracer)

option( CUDA "use cuda as default target" OFF )
option( GL "use opengl as default visualizer" ON )

if (${CUDA} STREQUAL "ON")
	find_package(CUDA REQUIRED)
	add_definitions(-DKOISHI_USE_CUDA)
endif ()

if (${GL} STREQUAL "ON")
	add_definitions(-DKOISHI_USE_GL)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.c)

set(CMAKE_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -std=c++11 -x cu")

link_libraries(assimp)
if (${GL} STREQUAL "ON")
	#link_libraries(glfw3 ${OPENGL_LIBRARIES})
endif ()

if (${CUDA} STREQUAL "ON")
	set(CMAKE_CXX_COMPILER nvcc)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -x cu")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -DDEBUG -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
	cuda_add_executable(cr ${SOURCES})
else ()
	add_compile_options(-Wno-narrowing)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-parentheses -Wno-comment")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pg -DDEBUG -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
	add_executable(cr ${SOURCES})
endif ()
