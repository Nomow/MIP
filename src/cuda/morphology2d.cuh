#ifndef MORPHOLOGY_CUH
#define MORPHOLOGY_CUH
#include <cstdint>
  void Erode(uint8_t *inSrc, uint32_t inSrcWidth, uint32_t inSrcHeight, uint8_t* inKernel, uint8_t inKernelWidth, uint8_t inKernelHeight, uint8_t *&outDst);
  void ErodeWithBorderControl(uint8_t *inSrc, uint32_t inSrcWidth, uint32_t inSrcHeight, uint8_t* inKernel, uint8_t inKernelWidth, uint8_t inKernelHeight, uint8_t *&outDst);
  void Erode(unsigned char *inOrigImg, int inOrigImgDims[3], unsigned char *inStructElem, int inStructElemDims[3], unsigned char *&outErodedImg);
  void PaintObjectAndVolImgBorderKernel(unsigned char *inImgData, int inImgDataDims[3], unsigned char *inKernelData, int inKernelRadius[3], unsigned char *&outImgData);
  void PaintObjectBorderOfVolImgWithKernelObjectKernel(unsigned char *inImgData, int inImgDataDims[3], unsigned char *inKernelData, int inKernelRadius[3], unsigned char *&outImgData);


#endif
