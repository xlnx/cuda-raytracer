cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 14)
project(cuda_raytracer)

option( CUDA "use cuda as default target" OFF )
option( GL "use opengl as default visualizer" ON )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-parentheses -Wno-comment -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pg -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE SOURCES 
	${CMAKE_SOURCE_DIR}/src/*.cpp 
	${CMAKE_SOURCE_DIR}/src/*.cu
)

find_package(assimp REQUIRED)
if (assimp_FOUND)
    link_directories(${ASSIMP_LIBRARY_DIRS})
    include_directories(${ASSIMP_INCLUDE_DIRS})
# link_libraries (assimpd)
    link_libraries (${ASSIMP_LIBRARIES})
endif(assimp_FOUND)

if (${GL} STREQUAL "ON")
	add_definitions(-DKOISHI_USE_GL)
	file(GLOB_RECURSE GLAD_SOURCE
		${CMAKE_SOURCE_DIR}/src/*.c 
	)
	set(SOURCES ${SOURCES} ${GLAD_SOURCE})
	link_libraries(glfw3 ${OPENGL_LIBRARIES})
endif ()

if (${CUDA} STREQUAL "ON")
	find_package(CUDA REQUIRED)
	add_definitions(-DKOISHI_USE_CUDA)
	set(CUDA_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -std=c++11 -O3 --verbose "
		" -Xcompiler -fopenmp -Xcudafe \"--diag_suppress=\\\"set_but_not_used,code_is_unreachable\\\"\" ")
	#link_libraries(assimp)
	cuda_add_executable(cr ${SOURCES})
else ()
	#link_libraries(assimpd)
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
	add_compile_options(-Wno-narrowing -x c++)
	add_executable(cr ${SOURCES})
endif ()
