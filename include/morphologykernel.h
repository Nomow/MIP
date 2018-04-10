#ifndef MORPHOLOGY_KERNEL_H
#define MORPHOLOGY_KERNEL_H

void Erode(unsigned char *inOrigImg, int inOrigImgDims[3], unsigned char *inStructElem, int inStructElemDims[3], unsigned char *&outErodedImg);
#endif
