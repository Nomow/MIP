#ifndef MORPHOLOGY_CUH
#define MORPHOLOGY_CUH
#include <cstdint>

void Erode(uint8_t* inSrc, uint32_t srcX, uint32_t srcY, uint8_t* inMask, uint8_t maskX, uint8_t maskY, uint32_t *&outDst);

void Erode(unsigned char *inOrigImg, int inOrigImgDims[3], unsigned char *inStructElem, int inStructElemDims[3], unsigned char *&outErodedImg);
void PaintObjectAndVolImgBorderKernel(unsigned char *inImgData, int inImgDataDims[3], unsigned char *inKernelData, int inKernelRadius[3], unsigned char *&outImgData);
void PaintObjectBorderOfVolImgWithKernelObjectKernel(unsigned char *inImgData, int inImgDataDims[3], unsigned char *inKernelData, int inKernelRadius[3], unsigned char *&outImgData);

#endif
