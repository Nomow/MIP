cmake_minimum_required(VERSION 2.8)
project(morphological-image-processing)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -DNDEBUG -Wall -Wall -g -Wextra -Wno-sign-compare -frounding-math")
include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory(src)
add_subdirectory(include)
