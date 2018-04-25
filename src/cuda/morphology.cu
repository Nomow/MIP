#include "morphology.cuh"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <vector>
#include <nppdefs.h>
#include <nppi.h>
#define THREADSPERDIM = 8
#define BLOCKSPERDIM = 65535


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
__host__ void ConvertHostToDevice(uint8_t *inHostSrc, uint32_t srcX, uint32_t srcY, uint8_t *&outDeviceDst, size_t &outDstStep) {
  CudaSafeCall(cudaMallocPitch(&outDeviceDst, &outDstStep, srcX * sizeof(uint8_t), srcY));
  CudaSafeCall(cudaMemcpy2D(outDeviceDst, outDstStep, inHostSrc,  srcX * sizeof(uint8_t),
                            srcX * sizeof(uint8_t), srcY, cudaMemcpyHostToDevice));
}


/**
  allocates memory in device and copies data from host memory to padded device memory.
*/
__host__ void ConvertHostToDevice(uint8_t *inHostSrc, uint32_t srcSize, uint8_t *&outDeviceDst) {
  CudaSafeCall(cudaMalloc(&outDeviceDst, srcSize * sizeof(uint8_t)));
  CudaSafeCall(cudaMemcpy(outDeviceDst, inHostSrc,  srcSize * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

/**
  allocates memory in host and copies data from device memory to host memory.
*/
__host__ void ConvertDeviceToHost(uint8_t *inDeviceSrc, size_t &inSrcStep, uint32_t srcX, uint32_t srcY, uint8_t *&outHostDst) {
    outHostDst = (uint8_t*)malloc(srcX * srcY);
    CudaSafeCall(cudaMemcpy2D(outHostDst, srcX * sizeof(uint8_t), inDeviceSrc,
                              inSrcStep, srcX * sizeof(uint8_t), srcY, cudaMemcpyDeviceToHost));
}

/**
  erosion of 2d image
*/
void Erode(uint8_t* inSrc, uint32_t srcX, uint32_t srcY, uint8_t* inMask, uint8_t maskX, uint8_t maskY, uint32_t *&outDst) {

  size_t deviceSrcPitch;
  uint8_t* deviceSrc;
  ConvertHostToDevice(inSrc, srcX, srcY, deviceSrc, deviceSrcPitch);

  NppiSize srcSize;
  srcSize.width = srcX;
  srcSize.height = srcY;

  uint8_t* deviceMask;
  uint32_t maskSize = maskX * maskY;
  ConvertHostToDevice(inMask, maskSize, deviceMask);

  NppiSize maskSize;
  maskSize.width = maskX;
  maskSize.height = maskY;

  // allocates memory for output
  size_t deviceDstPitch;
  uint8_t *deviceDst;
  CudaSafeCall(cudaMallocPitch(&deviceDst, &deviceDstPitch, srcX * sizeof(uint8_t), srcY));

  NppiPoint maskCenter;
  maskCenter.x = maskX / 2;
  maskCenter.y = maskY / 2;

  nppiErode_8u_C1R(inSrc, deviceSrcPitch, deviceDst, deviceDstPitch, srcSize, deviceMask, maskSize, maskCenter);
 
}











//
// __device__ int GetGlobal3dThreadIdx() {
//   int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//   int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//   return threadId;
// }
//
//
// __device__ bool Erode(int inOrigx, int inOrigy, int inOrigz,
//                       cudaPitchedPtr inImg, int inPaddingx, int inPaddingy, int inPaddingz,
//                       cudaPitchedPtr inGpuStructElem, int inStructElemRadiusx, int inStructElemRadiusy, int inStructElemRadiusz,
//                       int *inGpuStructElemForegroundDataPosx, int *inGpuStructElemForegroundDataPosy, int *inGpuStructElemForegroundDataPosz, int kernelDimSize) {
//
//     int x = inOrigx + inPaddingx - inStructElemRadiusx;
//     int y = inOrigy + inPaddingy - inStructElemRadiusy;
//     int z = inOrigz + inPaddingz - inStructElemRadiusz;
//
//     for (int i = 0; i < kernelDimSize; ++i) {
//
//       // gets kernel pixel
//       int kx = *(inGpuStructElemForegroundDataPosx + i);
//       int ky = *(inGpuStructElemForegroundDataPosy + i);
//       int kz = *(inGpuStructElemForegroundDataPosz + i);
//       unsigned char kernelPixel = getPixel(inGpuStructElem, kx, ky, kz);
//
//       // gets image pixel
//       int imx = x + kx;
//       int imy = y + ky;
//       int imz = z + kz;
//       unsigned char imagePixel = getPixel(inImg, imx, imy, imz);
//
//       if (!(kernelPixel && imagePixel)) {
//         return true;
//       }
//     }
//     return false;and
// }
//
//
// __global__ void ImgErosion(cudaPitchedPtr inGpuOrigImg, int *inGpuOrigImgForegoundDataPosx, int *inGpuOrigImgForegoundDataPosy,  int *inGpuOrigImgForegoundDataPosz, int inImgDimSize,
//                            cudaPitchedPtr inPaddedImg, int inPaddingx, int inPaddingy, int inPaddingz,
//                            cudaPitchedPtr inGpuStructElem, int inStructElemRadiusx, int inStructElemRadiusy, int inStructElemRadiusz,
//                            int *inGpuStructElemForegroundDataPosx, int *inGpuStructElemForegroundDataPosy, int *inGpuStructElemForegroundDataPosz, int kernelDimSize) {
//
//   int threadId = GetGlobal3dThreadIdx();
//   if(threadId < inImgDimSize) {
//     int x = *(inGpuOrigImgForegoundDataPosx + threadId);
//     int y = *(inGpuOrigImgForegoundDataPosy + threadId);
//     int z = *(inGpuOrigImgForegoundDataPosz + threadId);
//     int isEroded = Erode(x, y, z, inPaddedImg, inPaddingx, inPaddingy, inPaddingz,
//                          inGpuStructElem, inStructElemRadiusx, inStructElemRadiusy, inStructElemRadiusz,
//                          inGpuStructElemForegroundDataPosx, inGpuStructElemForegroundDataPosy, inGpuStructElemForegroundDataPosz, kernelDimSize);
//     if (isEroded) {
//       char* devPtr = (char*)inGpuOrigImg.ptr;
//       size_t pitch = inGpuOrigImg.pitch;
//       size_t slicePitch = pitch * inGpuOrigImg.ysize;
//       char* slice = devPtr + z * slicePitch;
//       unsigned char* row = (unsigned char*)(slice + y * pitch);
//       row[x] = 0;
//    }
//   }
// }
//



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

__device__ unsigned char getPixel(cudaPitchedPtr inImg, int x, int y, int z) {
  char* devPtr = (char*)inImg.ptr;
  size_t pitch = inImg.pitch;
  size_t slicePitch = pitch * inImg.ysize;
  char* slice = devPtr + z * slicePitch;
  unsigned char* row = (unsigned char*)(slice + y * pitch);
  return row[x];
}

__global__ void FindBorders(cudaPitchedPtr inImgData, int dimX, int dimY, int dimZ,
                            cudaPitchedPtr inKernelData,
                            int radiusX, int radiusY, int radiusZ) {

  int threadX = blockIdx.x * blockDim.x + threadIdx.x;
  int threadY = blockIdx.y * blockDim.y + threadIdx.y;
  int threadZ = blockIdx.z * blockDim.z + threadIdx.z;

  if (0 < threadX && dimX > threadX && 0 < threadY && dimY > threadY && 0 < threadZ && dimZ > threadZ) {
    unsigned char imgPixel = getPixel(inImgData, threadX, threadY, threadZ);
    if (imgPixel == 0) {
      int startX = threadX - radiusX;
      int startY = threadY - radiusY;
      int startZ = threadZ - radiusZ;

      for (int i = startX; i < threadX + 2 * radiusX; ++i) {
        for (int j = startY; j < threadY + 2 * radiusY; ++j) {
          for (int k = startZ; k < threadZ + 2 * radiusZ; ++k) {
              unsigned char compPixel = getPixel(inImgData, k, j, i);
              // finds border pixels by comparing if adjacent pixel is foreground
              if(compPixel == 1) {
                char* devPtr = (char*)inImgData.ptr;
                size_t pitch = inImgData.pitch;
                size_t slicePitch = pitch * inImgData.ysize;
                char* slice = devPtr + threadZ * slicePitch;
                unsigned char* row = (unsigned char*)(slice + threadY * pitch);
                row[threadX] = 2;
                break;
              }
          }
        }
      }
    }
  }
}



void PaintObjectAndVolImgBorderKernel(unsigned char *inImgData, int inImgDataDims[3], unsigned char *inKernelData, int inKernelRadius[3], unsigned char *&outImgData) {
  cudaPitchedPtr gpuImgData;
  CopyFromHostToDevice3dMemory(inImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], gpuImgData);

  cudaPitchedPtr gpuKernelData;
  CopyFromHostToDevice3dMemory(inKernelData, inKernelRadius[0], inKernelRadius[1], inKernelRadius[2], gpuKernelData);
  // FindBorders(cudaPitchedPtr inImgData, int dimX, int dimY, int dimZ,
  //                             cudaPitchedPtr inKernelData,
  //                             int radiusX, int radiusY, int radiusZ)
  int tpb0 = (inImgDataDims[0] + 8) / 8 + 1;
  int tpb1 = (inImgDataDims[1] + 8) / 8 + 1;
  int tpb2 = (inImgDataDims[2] + 8) / 8 + 1;

  dim3 blocks_per_grid(tpb0, tpb1, tpb2);
  dim3 threads_per_block(8, 8, 8);
  FindBorders<<<blocks_per_grid, threads_per_block>>>(gpuImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], gpuKernelData, inKernelRadius[0], inKernelRadius[1], inKernelRadius[2]);
  CudaCheckError();
  outImgData = (unsigned char*)malloc(inImgDataDims[0] * inImgDataDims[1] * inImgDataDims[2] * sizeof(unsigned char));
  CopyFromDevice3dToHostMemory(gpuImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], outImgData);
  CudaCheckError();
  cudaFree(gpuImgData.ptr);
  cudaFree(gpuKernelData.ptr);
}


__host__ void GetKernelForegroundIndex(unsigned char* inData, int inDimensions[3], std::vector<int> &x, std::vector<int> &y, std::vector<int> &z) {
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




__host__ void CopyFromHostToDeviceMemory(int *inData, int size, int *&outData) {
  cudaMalloc(&outData, size * sizeof(int));
  cudaMemcpy(outData, inData, size * sizeof(int), cudaMemcpyHostToDevice);
}


__global__ void PaintObject(cudaPitchedPtr inImgData, int dimX, int dimY, int dimZ,
                            int kernelDimX, int kernelDimY, int kernelDimZ,
                            int *gpuKernelForegroundPosX, int *gpuKernelForegroundPosY, int *gpuKernelForegroundPosZ, int foreGroundCount,
                            cudaPitchedPtr outImgData) {

  int threadX = blockIdx.x * blockDim.x + threadIdx.x;
  int threadY = blockIdx.y * blockDim.y + threadIdx.y;
  int threadZ = blockIdx.z * blockDim.z + threadIdx.z;

  if (dimX > threadX && dimY > threadY && dimZ > threadZ) {
    unsigned char imgPixel = getPixel(inImgData, threadX, threadY, threadZ);
    if(imgPixel == 2) {
      int startX = threadX - kernelDimX;
      int startY = threadY - kernelDimY;
      int startZ = threadZ - kernelDimZ;
      for (int i = 0; i < foreGroundCount; ++i) {
        // kernel foreground position
        int kernelForegroundX = *(gpuKernelForegroundPosX + i);
        int kernelForegroundY = *(gpuKernelForegroundPosY + i);
        int kernelForegroundZ = *(gpuKernelForegroundPosZ + i);

        // position of element in image
        int posX = startX + kernelForegroundX;
        int posY = startY + kernelForegroundY;
        int posZ = startZ + kernelForegroundZ;

        // asigns value to image
        char* devPtr = (char*)outImgData.ptr;
        size_t pitch = outImgData.pitch;
        size_t slicePitch = pitch * outImgData.ysize;
        char* slice = devPtr + posZ * slicePitch;
        unsigned char* row = (unsigned char*)(slice + posY * pitch);
        row[posX] = 0;
      }
    }
  }
}

void PaintObjectBorderOfVolImgWithKernelObjectKernel(unsigned char *inImgData, int inImgDataDims[3], unsigned char *inKernelData, int inKernelRadius[3], unsigned char *&outImgData) {
cudaProfilerStart();
  cudaPitchedPtr gpuImgData;
  CopyFromHostToDevice3dMemory(inImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], gpuImgData);
  cudaPitchedPtr gpuOutImgData;
  CopyFromHostToDevice3dMemory(inImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], gpuOutImgData);

  cudaPitchedPtr gpuKernelData;
  CopyFromHostToDevice3dMemory(inKernelData, inKernelRadius[0], inKernelRadius[1], inKernelRadius[2], gpuKernelData);

  int kernelDims[3];
  for (int i = 0; i < 3; ++i) {
    kernelDims[i] = inKernelRadius[i] * 2 + 1;
  }

  std::vector<int> kernelPosX, kernelPosY, kernelPosZ;
  GetKernelForegroundIndex(inKernelData, kernelDims, kernelPosX, kernelPosY, kernelPosZ);
  int dimSize = kernelPosX.size();
  printf("%d \n", dimSize);
  int *gpuKernelPosX;
  CopyFromHostToDeviceMemory(&kernelPosX[0], dimSize, gpuKernelPosX);
  int *gpuKernelPosY;
  CopyFromHostToDeviceMemory(&kernelPosY[0], dimSize, gpuKernelPosY);
  int *gpuKernelPosZ;
  CopyFromHostToDeviceMemory(&kernelPosZ[0], dimSize, gpuKernelPosZ);

  int tpb0 = (inImgDataDims[0] + 8) / 8 + 1;
  int tpb1 = (inImgDataDims[1] + 8) / 8 + 1;
  int tpb2 = (inImgDataDims[2] + 8) / 8 + 1;

  dim3 blocks_per_grid(tpb0, tpb1, tpb2);
  dim3 threads_per_block(8, 8, 8);
  PaintObject<<<blocks_per_grid, threads_per_block>>>(gpuImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], kernelDims[0],
      kernelDims[1], kernelDims[2], gpuKernelPosX, gpuKernelPosY, gpuKernelPosZ, dimSize, gpuOutImgData);
  CudaCheckError();
  outImgData = (unsigned char*)malloc(inImgDataDims[0] * inImgDataDims[1] * inImgDataDims[2] * sizeof(unsigned char));
  CopyFromDevice3dToHostMemory(gpuOutImgData, inImgDataDims[0], inImgDataDims[1], inImgDataDims[2], outImgData);
  CudaCheckError();

  cudaFree(gpuImgData.ptr);
  cudaFree(gpuOutImgData.ptr);
  cudaFree(gpuKernelPosX);
  cudaFree(gpuKernelPosY);
  cudaFree(gpuKernelPosZ);
  cudaProfilerStop();
}


//
//
// void Erode(unsigned char *inOrigImg, int inOrigImgDims[3], unsigned char *inStructElem, int inStructElemDims[3], unsigned char *&outErodedImg) {
//
//
//
//   //
//   // std::vector<int> origImgForegoundDataPosx;
//   // std::vector<int> origImgForegoundDataPosy;
//   // std::vector<int> origImgForegoundDataPosz;
//   // GetForegroundData(inOrigImg, inOrigImgDims, origImgForegoundDataPosx, origImgForegoundDataPosy, origImgForegoundDataPosz);
//   // int imgDimSize = origImgForegoundDataPosx.size();
//   //
//   // // moves original image pos data to gpu
//   // int* gpuOrigImgForegoundDataPosx;
//   // CopyFromHostToDeviceMemory(&origImgForegoundDataPosx[0], origImgForegoundDataPosx.size(), gpuOrigImgForegoundDataPosx);
//   // int* gpuOrigImgForegoundDataPosy;
//   // CopyFromHostToDeviceMemory(&origImgForegoundDataPosy[0], origImgForegoundDataPosy.size(), gpuOrigImgForegoundDataPosy);
//   // int* gpuOrigImgForegoundDataPosz;
//   // CopyFromHostToDeviceMemory(&origImgForegoundDataPosz[0], origImgForegoundDataPosz.size(), gpuOrigImgForegoundDataPosz);
//   //
//   // // gets foreground pixel pos from kernel
//   // std::vector<int> structElemForegroundDataPosx;
//   // std::vector<int> structElemForegroundDataPosy;
//   // std::vector<int> structElemForegroundDataPosz;
//   // GetForegroundData(inStructElem, inStructElemDims, structElemForegroundDataPosx, structElemForegroundDataPosy, structElemForegroundDataPosz);
//   // int kernelDimSize = structElemForegroundDataPosx.size();
//   //
//   // // moves structuring element pos data to gpu
//   // int* gpuStructElemForegroundDataPosx;
//   // CopyFromHostToDeviceMemory(&structElemForegroundDataPosx[0], structElemForegroundDataPosx.size(), gpuStructElemForegroundDataPosx);
//   // int* gpuStructElemForegroundDataPosy;
//   // CopyFromHostToDeviceMemory(&structElemForegroundDataPosy[0], structElemForegroundDataPosy.size(), gpuStructElemForegroundDataPosy);
//   // int* gpuStructElemForegroundDataPosz;
//   // CopyFromHostToDeviceMemory(&structElemForegroundDataPosz[0], structElemForegroundDataPosz.size(), gpuStructElemForegroundDataPosz);
//   //
//   // // moves data to gpu
//   // cudaPitchedPtr gpuOrigImg;
//   // CopyFromHostToDevice3dMemory(inOrigImg, inOrigImgDims[0], inOrigImgDims[1], inOrigImgDims[2], gpuOrigImg);
//   // cudaPitchedPtr gpuPaddedImg;
//   // CopyFromHostToDevice3dMemory(inPaddedImg, inPaddedImgDims[0], inPaddedImgDims[1], inPaddedImgDims[2], gpuPaddedImg);
//   // cudaPitchedPtr gpuStructElem;
//   // CopyFromHostToDevice3dMemory(inStructElem, inStructElemDims[0], inStructElemDims[1], inStructElemDims[2], gpuStructElem);
//   //
//   // // calculates difference of padded img and original image
//   // int padding[3];
//   // for (auto i = 0; i < 3; ++i) {
//   //     padding[i] = (inPaddedImgDims[i] - inOrigImgDims[i]) / 2;
//   // }
//   //
//   // // calculates kernel radius
//   // int kernelRadius[3];
//   // for (auto i = 0; i < 3; ++i) {
//   //     kernelRadius[i] = (inStructElemDims[i] - 1) / 2;
//   // }
//   // printf("%d", origImgForegoundDataPosz.size());
//   // int s = std::ceil((double)imgDimSize / 512);
//   // ImgErosion<<<dim3(s, 1, 1), dim3(8, 8, 8)>>>(gpuOrigImg, gpuOrigImgForegoundDataPosx, gpuOrigImgForegoundDataPosy, gpuOrigImgForegoundDataPosz, imgDimSize,
//   //                                                  gpuPaddedImg, padding[0], padding[1], padding[2],
//   //                                                  gpuStructElem, kernelRadius[0], kernelRadius[1], kernelRadius[2],
//   //                                                  gpuStructElemForegroundDataPosx, gpuStructElemForegroundDataPosy, gpuStructElemForegroundDataPosz,
//   //                                                  kernelDimSize);
//   // CudaCheckError();
//   // outErodedImg = (unsigned char*)malloc(inOrigImgDims[0] * inOrigImgDims[1] * inOrigImgDims[2] * sizeof(unsigned char));
//   // CopyFromDevice3dToHostMemory(gpuOrigImg, inOrigImgDims[0], inOrigImgDims[1], inOrigImgDims[2], outErodedImg);
//   //
//   // cudaFree(gpuOrigImg.ptr);
//   // cudaFree(gpuPaddedImg.ptr);
//   // cudaFree(gpuStructElem.ptr);
//   // cudaFree(gpuStructElem.ptr);
//   // cudaFree(gpuOrigImgForegoundDataPosx);
//   // cudaFree(gpuOrigImgForegoundDataPosy);
//   // cudaFree(gpuOrigImgForegoundDataPosz);
//   // cudaFree(gpuStructElemForegroundDataPosx);
//   // cudaFree(gpuStructElemForegroundDataPosy);
//   // cudaFree(gpuStructElemForegroundDataPosz);
//
// }
