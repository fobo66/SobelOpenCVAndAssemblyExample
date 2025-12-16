cmake_minimum_required(VERSION 3.15)

project(lab_asm_mmx_lena)

find_package( OpenCV REQUIRED )

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

message( STATUS "Architecture: ${ARCHITECTURE}" )

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE_EXE main.cpp)

add_executable(main ${SOURCE_EXE})

if(ARCHITECTURE EQUAL "arm64")
  target_include_directories(lab_asm_mmx_lena PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

target_link_libraries(main ${OpenCV_LIBS})
