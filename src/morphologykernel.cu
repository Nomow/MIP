#include "morphologykernel.h"
#include <stdio.h>
#include <vector>
#define THREADSPERDIM = 8
#define BLOCKSPERDIM = 65535

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

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
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





__host__ void CopyFromHostToDeviceMemory(int *inData, int size, int *&outData) {
  cudaMalloc(&outData, size * sizeof(int));
  cudaMemcpy(outData, inData, size * sizeof(int), cudaMemcpyHostToDevice);
}

__host__ void GetForegroundData(unsigned char* inData, int inDimensions[3], std::vector<int> &x, std::vector<int> &y, std::vector<int> &z) {
    int totalSize = inDimensions[0] * inDimensions[1] * inDimensions[2];
    x.reserve(totalSize);
    y.reserve(totalSize);
    z.reserve(totalSize);

    for (int iz = 0; iz  < inDimensions[2]; ++iz) {
      for (int iy = 0; iy < inDimensions[1]; ++iy) {
        for (int ix = 0; ix < inDimensions[0]; ++ix) {
           unsigned char pixel = *(inData +  (iz *( inDimensions[0] * inDimensions[1] ) + (iy * inDimensions[0]) + ix));
           if (pixel == 1) {
             x.push_back(ix);
             y.push_back(iy);
             z.push_back(iz);
           }
        }
      }
    }
}

__device__ int GetGlobal3dThreadIdx() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

__device__ unsigned char getPixel(cudaPitchedPtr inImg, int x, int y, int z) {
  char* devPtr = (char*)inImg.ptr;
  size_t pitch = inImg.pitch;
  size_t slicePitch = pitch * inImg.ysize;
  char* slice = devPtr + z * slicePitch;
  unsigned char* row = (unsigned char*)(slice + y * pitch);
  return row[x];
}

__device__ bool Erode(int inOrigx, int inOrigy, int inOrigz,
                      cudaPitchedPtr inImg, int inPaddingx, int inPaddingy, int inPaddingz,
                      cudaPitchedPtr inGpuStructElem, int inStructElemRadiusx, int inStructElemRadiusy, int inStructElemRadiusz,
                      int *inGpuStructElemForegroundDataPosx, int *inGpuStructElemForegroundDataPosy, int *inGpuStructElemForegroundDataPosz, int kernelDimSize) {

    int x = inOrigx + inPaddingx - inStructElemRadiusx;
    int y = inOrigy + inPaddingy - inStructElemRadiusy;
    int z = inOrigz + inPaddingz - inStructElemRadiusz;

    for (int i = 0; i < kernelDimSize; ++i) {

      // gets kernel pixel
      int kx = *(inGpuStructElemForegroundDataPosx + i);
      int ky = *(inGpuStructElemForegroundDataPosy + i);
      int kz = *(inGpuStructElemForegroundDataPosz + i);
      unsigned char kernelPixel = getPixel(inGpuStructElem, kx, ky, kz);

      // gets image pixel
      int imx = x + kx;
      int imy = y + ky;
      int imz = z + kz;
      unsigned char imagePixel = getPixel(inImg, imx, imy, imz);

      if (!(kernelPixel && imagePixel)) {
        return true;
      }
    }
    return false;
}

__global__ void ImgErosion(cudaPitchedPtr inGpuOrigImg, int *inGpuOrigImgForegoundDataPosx, int *inGpuOrigImgForegoundDataPosy,  int *inGpuOrigImgForegoundDataPosz, int inImgDimSize,
                           cudaPitchedPtr inPaddedImg, int inPaddingx, int inPaddingy, int inPaddingz,
                           cudaPitchedPtr inGpuStructElem, int inStructElemRadiusx, int inStructElemRadiusy, int inStructElemRadiusz,
                           int *inGpuStructElemForegroundDataPosx, int *inGpuStructElemForegroundDataPosy, int *inGpuStructElemForegroundDataPosz, int kernelDimSize) {

  int threadId = GetGlobal3dThreadIdx();
  if(threadId < inImgDimSize) {
    int x = *(inGpuOrigImgForegoundDataPosx + threadId);
    int y = *(inGpuOrigImgForegoundDataPosy + threadId);
    int z = *(inGpuOrigImgForegoundDataPosz + threadId);
    int isEroded = Erode(x, y, z, inPaddedImg, inPaddingx, inPaddingy, inPaddingz,
                         inGpuStructElem, inStructElemRadiusx, inStructElemRadiusy, inStructElemRadiusz,
                         inGpuStructElemForegroundDataPosx, inGpuStructElemForegroundDataPosy, inGpuStructElemForegroundDataPosz, kernelDimSize);
    if (isEroded) {
      char* devPtr = (char*)inGpuOrigImg.ptr;
      size_t pitch = inGpuOrigImg.pitch;
      size_t slicePitch = pitch * inGpuOrigImg.ysize;
      char* slice = devPtr + z * slicePitch;
      unsigned char* row = (unsigned char*)(slice + y * pitch);
      row[x] = 0;
   }
  }
}

__host__ void CopyFromDevice3dToHostMemory(cudaPitchedPtr inData, int x, int y, int z, unsigned char *&outData) {
  cudaExtent extent = make_cudaExtent(x * sizeof(unsigned char), y, z);
  cudaMemcpy3DParms cpyParam = {0};
  cpyParam.srcPtr = inData;
  cpyParam.dstPtr = make_cudaPitchedPtr( (void*)outData, x * sizeof(unsigned char), y, z);
	cpyParam.extent = extent;
	cpyParam.kind 	= cudaMemcpyDeviceToHost;
	cudaMemcpy3D(&cpyParam);
}




__host__ void CopyFromHostToDevice3dMemory(unsigned char *inData, int x, int y, int z, cudaPitchedPtr &outData) {
	cudaExtent extent = make_cudaExtent(x * sizeof(unsigned char), y, z);
  cudaMalloc3D(&outData, extent);
  cudaMemcpy3DParms copyParam = {0};
  copyParam.srcPtr = make_cudaPitchedPtr( (void*)inData, x * sizeof(unsigned char), y, z );
  copyParam.srcPtr = make_cudaPitchedPtr( (void*)inData, x * sizeof(unsigned char), y, z);
	copyParam.dstPtr = outData;
	copyParam.extent = extent;
	copyParam.kind 	= cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParam);
}


void Erode(unsigned char *inOrigImg, int inOrigImgDims[3], unsigned char *inStructElem, int inStructElemDims[3], unsigned char *&outErodedImg) {



  //
  // std::vector<int> origImgForegoundDataPosx;
  // std::vector<int> origImgForegoundDataPosy;
  // std::vector<int> origImgForegoundDataPosz;
  // GetForegroundData(inOrigImg, inOrigImgDims, origImgForegoundDataPosx, origImgForegoundDataPosy, origImgForegoundDataPosz);
  // int imgDimSize = origImgForegoundDataPosx.size();
  //
  // // moves original image pos data to gpu
  // int* gpuOrigImgForegoundDataPosx;
  // CopyFromHostToDeviceMemory(&origImgForegoundDataPosx[0], origImgForegoundDataPosx.size(), gpuOrigImgForegoundDataPosx);
  // int* gpuOrigImgForegoundDataPosy;
  // CopyFromHostToDeviceMemory(&origImgForegoundDataPosy[0], origImgForegoundDataPosy.size(), gpuOrigImgForegoundDataPosy);
  // int* gpuOrigImgForegoundDataPosz;
  // CopyFromHostToDeviceMemory(&origImgForegoundDataPosz[0], origImgForegoundDataPosz.size(), gpuOrigImgForegoundDataPosz);
  //
  // // gets foreground pixel pos from kernel
  // std::vector<int> structElemForegroundDataPosx;
  // std::vector<int> structElemForegroundDataPosy;
  // std::vector<int> structElemForegroundDataPosz;
  // GetForegroundData(inStructElem, inStructElemDims, structElemForegroundDataPosx, structElemForegroundDataPosy, structElemForegroundDataPosz);
  // int kernelDimSize = structElemForegroundDataPosx.size();
  //
  // // moves structuring element pos data to gpu
  // int* gpuStructElemForegroundDataPosx;
  // CopyFromHostToDeviceMemory(&structElemForegroundDataPosx[0], structElemForegroundDataPosx.size(), gpuStructElemForegroundDataPosx);
  // int* gpuStructElemForegroundDataPosy;
  // CopyFromHostToDeviceMemory(&structElemForegroundDataPosy[0], structElemForegroundDataPosy.size(), gpuStructElemForegroundDataPosy);
  // int* gpuStructElemForegroundDataPosz;
  // CopyFromHostToDeviceMemory(&structElemForegroundDataPosz[0], structElemForegroundDataPosz.size(), gpuStructElemForegroundDataPosz);
  //
  // // moves data to gpu
  // cudaPitchedPtr gpuOrigImg;
  // CopyFromHostToDevice3dMemory(inOrigImg, inOrigImgDims[0], inOrigImgDims[1], inOrigImgDims[2], gpuOrigImg);
  // cudaPitchedPtr gpuPaddedImg;
  // CopyFromHostToDevice3dMemory(inPaddedImg, inPaddedImgDims[0], inPaddedImgDims[1], inPaddedImgDims[2], gpuPaddedImg);
  // cudaPitchedPtr gpuStructElem;
  // CopyFromHostToDevice3dMemory(inStructElem, inStructElemDims[0], inStructElemDims[1], inStructElemDims[2], gpuStructElem);
  //
  // // calculates difference of padded img and original image
  // int padding[3];
  // for (auto i = 0; i < 3; ++i) {
  //     padding[i] = (inPaddedImgDims[i] - inOrigImgDims[i]) / 2;
  // }
  //
  // // calculates kernel radius
  // int kernelRadius[3];
  // for (auto i = 0; i < 3; ++i) {
  //     kernelRadius[i] = (inStructElemDims[i] - 1) / 2;
  // }
  // printf("%d", origImgForegoundDataPosz.size());
  // int s = std::ceil((double)imgDimSize / 512);
  // ImgErosion<<<dim3(s, 1, 1), dim3(8, 8, 8)>>>(gpuOrigImg, gpuOrigImgForegoundDataPosx, gpuOrigImgForegoundDataPosy, gpuOrigImgForegoundDataPosz, imgDimSize,
  //                                                  gpuPaddedImg, padding[0], padding[1], padding[2],
  //                                                  gpuStructElem, kernelRadius[0], kernelRadius[1], kernelRadius[2],
  //                                                  gpuStructElemForegroundDataPosx, gpuStructElemForegroundDataPosy, gpuStructElemForegroundDataPosz,
  //                                                  kernelDimSize);
  // CudaCheckError();
  // outErodedImg = (unsigned char*)malloc(inOrigImgDims[0] * inOrigImgDims[1] * inOrigImgDims[2] * sizeof(unsigned char));
  // CopyFromDevice3dToHostMemory(gpuOrigImg, inOrigImgDims[0], inOrigImgDims[1], inOrigImgDims[2], outErodedImg);
  //
  // cudaFree(gpuOrigImg.ptr);
  // cudaFree(gpuPaddedImg.ptr);
  // cudaFree(gpuStructElem.ptr);
  // cudaFree(gpuStructElem.ptr);
  // cudaFree(gpuOrigImgForegoundDataPosx);
  // cudaFree(gpuOrigImgForegoundDataPosy);
  // cudaFree(gpuOrigImgForegoundDataPosz);
  // cudaFree(gpuStructElemForegroundDataPosx);
  // cudaFree(gpuStructElemForegroundDataPosy);
  // cudaFree(gpuStructElemForegroundDataPosz);

}
