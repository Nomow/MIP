#ifndef MORPHOLOGY3D_CUH
#define MORPHOLOGY3D_CUH
#include <cstdint>
#include <cuda_runtime.h>

namespace cudamorph3d{
  void CopyFromDeviceToHostMemory(cudaPitchedPtr inDeviceSrc, uint32_t inWidth, uint32_t inHeight, uint32_t inDepth, uint8_t *&outHostDst);
  void CopyFromHostToDeviceMemory(uint8_t *inHostSrc, uint32_t inWidth, uint32_t inHeight, uint32_t inDepth, cudaPitchedPtr &outDeviceDst);
  void CopyFromHostToDeviceMemory(int32_t *inHostSrc, uint32_t inSrcSize, int32_t *&outDeviceDst);
  void PaintDifferenceSet(cudaPitchedPtr &inDeviceSrc, uint32_t inX, uint32_t inY, uint32_t inZ, int32_t *inDeviceDifferenceSet, uint32_t inSize, cudaStream_t inStream);

}
#endif
