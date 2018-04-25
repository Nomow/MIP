set(MORPHOLOGY
  morphology.cpp
  morphology.h

)

set(MAIN_INCLUDE
  main.cpp
)

set(CUDA_MORPHOLOGY
  cuda/morphology.cu
  cuda/morphology.cuh
)

# CUDA
find_package(CUDA 8.0 REQUIRED)
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  message("CUDA LIBRARY FOUND")
  enable_language("CUDA")
else(CUDA_FOUND)
  message("CUDA LIBRARY DOES NOT EXIST")
endif(CUDA_FOUND)

# EIGEN
find_package(Eigen3 3.2.0 QUIET)
if(EIGEN3_FOUND)
  message("EIGEN LIBRARY FOUND")
  include(${EIGEN3_USE_FILE})
  INCLUDE_DIRECTORIES (${EIGEN3_INCLUDE_DIR})
else(EIGEN3_FOUND)
  message("EIGEN LIBRARY DOES NOT EXIST")
endif(EIGEN3_FOUND)

# ITK
find_package(ITK REQUIRED)
if(ITK_FOUND)
include_directories(${ITK_INCLUDE_DIRS})
message(STATUS "ITK status:")
message(STATUS "    libraries: ${ITK_LIBRARIES}")
message(STATUS "    include path: ${ITK_INCLUDE_DIRS}")
endif(ITK_FOUND)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60, -lineinfo)
cuda_add_executable(main ${MAIN_INCLUDE} ${MORPHOLOGY} ${CUDA_MORPHOLOGY} OPTIONS -g -G)
target_link_libraries(main ${ITK_LIBRARIES} ${CUDA_LIBRARIES})
