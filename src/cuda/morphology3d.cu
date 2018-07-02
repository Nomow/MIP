#include "morphology3d.cuh"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <cmath>
#define kTrheadsPerDim 128
#define kBlocksPerDim 65535
/**
  cuda error checking helper methods for data copy between host and device and device and host and kernel error check.
*/
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

/**
  allocates memory in device and copies data from host memory to padded device memory.
*/
void cudamorph3d::CopyFromDeviceToHostMemory(cudaPitchedPtr inDeviceSrc, uint32_t inWidth, uint32_t inHeight, uint32_t inDepth, uint8_t *&outHostDst) {
    outHostDst = (uint8_t*)malloc(inWidth * inHeight * inDepth * sizeof(uint8_t));
  cudaExtent extent = make_cudaExtent(inWidth * sizeof(uint8_t), inHeight, inDepth);
  cudaMemcpy3DParms cpyParam = {0};
  cpyParam.srcPtr = inDeviceSrc;
  cpyParam.dstPtr = make_cudaPitchedPtr((void*)outHostDst, inWidth * sizeof(uint8_t), inHeight, inDepth);
	cpyParam.extent = extent;
	cpyParam.kind 	= cudaMemcpyDeviceToHost;
	CudaSafeCall(cudaMemcpy3D(&cpyParam));




}

/**
  allocates memory in device and copies data from host memory to padded device memory.
*/
void cudamorph3d::CopyFromHostToDeviceMemory(uint8_t *inHostSrc, uint32_t inWidth, uint32_t inHeight, uint32_t inDepth, cudaPitchedPtr &outDeviceDst) {
	cudaExtent extent = make_cudaExtent(inWidth * sizeof(uint8_t), inHeight, inDepth);
  cudaMalloc3D(&outDeviceDst, extent);
  cudaMemcpy3DParms copyParam = {0};
  copyParam.srcPtr = make_cudaPitchedPtr((void*)inHostSrc, inWidth * sizeof(uint8_t), inHeight, inDepth);
  copyParam.srcPtr = make_cudaPitchedPtr((void*)inHostSrc, inWidth * sizeof(uint8_t), inHeight, inDepth);
	copyParam.dstPtr = outDeviceDst;
	copyParam.extent = extent;
	copyParam.kind 	= cudaMemcpyHostToDevice;
	CudaSafeCall(cudaMemcpy3D(&copyParam));
}

void cudamorph3d::CopyFromHostToDeviceMemory(int32_t *inHostSrc, uint32_t inSrcSize, int32_t *&outDeviceDst) {
  CudaSafeCall(cudaMalloc(&outDeviceDst,  sizeof(int32_t) * inSrcSize));
  CudaSafeCall(cudaMemcpy(outDeviceDst, inHostSrc,  inSrcSize * sizeof(int32_t), cudaMemcpyHostToDevice));
}

/**
  gets pixel value of x y z of pitched ptr
*/
__device__ unsigned char getPixel(cudaPitchedPtr inSrc, uint32_t inX, uint32_t inY, uint32_t inZ) {
  char* srcPtr = (char*)inSrc.ptr;
  size_t pitch = inSrc.pitch;
  size_t slicePitch = pitch * inSrc.ysize;
  char* slice = srcPtr + inZ * slicePitch;
  unsigned char* row = (unsigned char*)(slice + inY * pitch);
  return row[inX];
}

/**
  calculates global kernel thread
*/
__device__ uint32_t GetGlobal3DThread() {
  uint32_t blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  uint32_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

/**
  paints image with difference set in gpu
*/
__global__ void CudaPaintDifferenceSet(cudaPitchedPtr inDeviceSrc, uint32_t inX, uint32_t inY, uint32_t inZ, int32_t *inDifferenceSet, uint32_t inSize) {
  uint32_t threadId = GetGlobal3DThread();
  if (GetGlobal3DThread() < inSize) {
    uint32_t paintX = inX + *(inDifferenceSet + 3 * threadId);
    uint32_t paintY = inY + *(inDifferenceSet + 3 * threadId + 1);
    uint32_t paintZ = inZ + *(inDifferenceSet + 3 * threadId + 2);

    // paints position
    char* srcPtr = (char*)inDeviceSrc.ptr;
    size_t step = inDeviceSrc.pitch;
    size_t sliceStep = step * inDeviceSrc.ysize;
    char* slice = srcPtr + paintZ * sliceStep;
    unsigned char* row = (unsigned char*)(slice + paintY * step);
    row[paintX] = 0;
  }
}

/**
  calculates 3d block size from 1d size
**/
__host__ void ThreadsPerBlock(uint32_t inSize, dim3 &outDim) {

  uint32_t threadsInBlock = kTrheadsPerDim;
  uint32_t blockCount = std::ceil((double)inSize / threadsInBlock);
  if (blockCount <= kBlocksPerDim) {
    outDim = {blockCount, 1, 1};
  } else if (blockCount <= kBlocksPerDim * kBlocksPerDim) {
    uint32_t xBlocks = std::ceil(std::sqrt(blockCount));
    uint32_t yBlocks = std::ceil(blockCount / xBlocks);
    outDim = {xBlocks, yBlocks, 1};
  } else {
    uint32_t xBlocks = std::ceil(std::pow(blockCount, 1.0/3));
    uint32_t yBlocks = std::ceil(blockCount / xBlocks);
    uint32_t zBlocks = std::ceil(blockCount / (xBlocks * yBlocks));
    outDim = {xBlocks, yBlocks, zBlocks};
  }
}

/**
  paints difference set
*/
void cudamorph3d::PaintDifferenceSet(cudaPitchedPtr &inDeviceSrc, uint32_t inX, uint32_t inY, uint32_t inZ, int32_t *inDeviceDifferenceSet, uint32_t inSize, cudaStream_t inStream) {

  dim3 blocksPerDim;
  inSize = inSize / 3;
  ThreadsPerBlock(inSize, blocksPerDim);
  dim3 threadsPerBlock(kTrheadsPerDim, 1, 1);
  CudaPaintDifferenceSet<<<blocksPerDim, threadsPerBlock, 0, inStream>>>(inDeviceSrc, inX, inY, inZ, inDeviceDifferenceSet, inSize);
}
