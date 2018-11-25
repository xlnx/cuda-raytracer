cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 14)
project(cuda_raytracer)

option( CUDA "use cuda as default target" OFF )
option( GL "use opengl as default visualizer" ON )
option( DEBUG_LEVEL "debug level" 0 )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-parentheses -Wno-comment -Wvirtual-move-assign -Wno-stringop-overflow")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pg -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories(
	${CMAKE_SOURCE_DIR}/include 
	${CMAKE_SOURCE_DIR}/src
)

file(GLOB_RECURSE SOURCES 
	${CMAKE_SOURCE_DIR}/src/**/*.cpp 
	${CMAKE_SOURCE_DIR}/src/**/*.cu
)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

file(GLOB TEST_SOURCES 
	${CMAKE_SOURCE_DIR}/test/*.cpp
	${CMAKE_SOURCE_DIR}/test/*.cu
)
	
# find_package(assimp REQUIRED)
# if (assimp_FOUND)
#     link_directories(${ASSIMP_LIBRARY_DIRS})
#     include_directories(${ASSIMP_INCLUDE_DIRS})
# link_libraries (assimpd)
    link_libraries (assimp)
# endif(assimp_FOUND)

add_definitions(-DKOISHI_DEBUG=${DEBUG_LEVEL})

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
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
	set(CUDA_SEPARABLE_COMPILATION ON)
	add_definitions(-DKOISHI_USE_CUDA)
	set(CUDA_NVCC_FLAGS ${CMAKE_NVCC_FLAGS} 
		-std=c++14 
		-g
		# -O3 
		-maxrregcount
		32
		-w  
		-lineinfo
		-Xcudafe 
		--diag_suppress="set_but_not_used,code_is_unreachable"
	)
	# cuda_add_executable(cr ${SOURCES})
else ()
	set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
	add_compile_options(-Wno-narrowing -x c++)
	# add_executable(cr ${SOURCES})
endif ()

function (koishi_add_executable EXE_NAME )
	if (${CUDA} STREQUAL "ON")
		cuda_add_executable(${EXE_NAME} ${ARGN})
	else ()
		set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CXX)
		add_executable(${EXE_NAME} ${ARGN})
	endif ()
endfunction ()

foreach(TEST_FILE ${TEST_SOURCES})
	string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${TEST_FILE})
	string(REGEX MATCH "^.*/([^\\\\/]+)$" dummy ${CMAKE_MATCH_1})
	set(TEST_EXECUTABLE_NAME "t_${CMAKE_MATCH_1}")
	koishi_add_executable(${TEST_EXECUTABLE_NAME} ${SOURCES} ${TEST_FILE})
	target_link_libraries(
		${TEST_EXECUTABLE_NAME}
		${GTEST_LIBRARY} 
		${GTEST_MAIN_LIBRARY}
	)
	add_test(
		${TEST_EXECUTABLE_NAME}
		${TEST_EXECUTABLE_NAME}
	)
endforeach(TEST_FILE)

koishi_add_executable(cr ${SOURCES} ${CMAKE_SOURCE_DIR}/src/cr.cu)
