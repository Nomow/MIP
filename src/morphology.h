#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <itkImage.h>
#include <itkSize.h>
#include <itkFlatStructuringElement.h>


class Morphology {
 public:
   Morphology();
   void Erode3d(const itk::Image<uint8_t, 3>::Pointer &inSrc, itk::FlatStructuringElement<3> inKernel, itk::Image<uint8_t, 3>::Pointer &outDst);

   void GetCenter(itk::Size<3> inDims, itk::Size<3> &outCenter);

   void GetOffsetFromCenter(itk::Size<3> inCenter, itk::Size<3> inPos, int outOffset[3]);

   void AddPaddingToImgRegion(const itk::Image<uint8_t, 3>::Pointer &inSrc,
                        itk::Size<3> inLowerBound,
                        itk::Size<3> inUpperBound,
                        uint8_t inPaddingVal,
                        itk::Image<uint8_t, 3>::Pointer &outDst);

  void AddPaddingToImg(const itk::Image<uint8_t, 3>::Pointer &inSrc,
                                   itk::Size<3> inLowerBound,
                                   itk::Size<3> inUpperBound,
                                   uint8_t inPaddingVal,
                                   itk::Image<uint8_t, 3>::Pointer &outDst);

   void CopyDataFromBufferToImg(uint8_t *inSrc,
                                itk::Index<3> inStartIndex,
                                itk::Size<3> inDims,
                                itk::Image<uint8_t, 3>::Pointer &outDst);

   void SubstractSameSizedImgs(const itk::Image<uint8_t, 3>::Pointer &inSrc1,
                               const itk::Image<uint8_t, 3>::Pointer &inSrc2,
                               itk::Image<uint8_t, 3>::Pointer &outDst);

   void Save(const itk::Image<uint8_t, 3>::Pointer &inSrc, std::string inName);

   void TranslateImg(const itk::Image<uint8_t, 3>::Pointer &inSrc, int32_t inX, int32_t inY,
                     int32_t inZ, itk::Image<uint8_t, 3>::Pointer &outDst);

   void CropVolImg(const itk::Image<uint8_t, 3>::Pointer &inSrc, itk::Size<3> inUpperBound,
                   itk::Size<3> inLowerBound, itk::Image<uint8_t, 3>::Pointer &outDst);

   void KernelToImg(itk::FlatStructuringElement<3> inKernel, itk::Image<uint8_t, 3>::Pointer &outDst);

   void GetDifferenceInEachDirection(itk::FlatStructuringElement<3> inKernel, std::vector<std::vector<int32_t> > &outDifferenceSet);

   void GetComponentOffsetFromCenter(const itk::Image<uint8_t, 3>::Pointer &inVolImg, std::vector<int32_t> &outConnectedComponents);
   void itkErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                               itk::FlatStructuringElement<3> inStructElem,
                               itk::Image<unsigned char, 3>::Pointer &outVolImg);

 private:
   void PasteImgToImg(const itk::Image<uint8_t, 3>::Pointer &inSrc1,
                      const itk::Image<uint8_t, 3>::Pointer &inSrc2,
                      itk::Index<3> inStartIndex,
                      itk::Image<uint8_t, 3>::Pointer &outDst);

};

#endif
