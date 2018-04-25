#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <itkImage.h>
#include <itkSize.h>
#include <itkFlatStructuringElement.h>


class Morphology {
 public:
   Morphology();

   void PaintObjectBorderOfVolImgWithKernelObject(const itk::Image<unsigned char, 3>::Pointer &inVolImg, itk::FlatStructuringElement<3> inKernel, itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void PaintObjectAndVolImgBorder(const itk::Image<unsigned char, 3>::Pointer &inVolImg, itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void CopyDataToImageFromBuffer(unsigned char *inVolImgData,
                                              itk::Index<3> inStartIndex,
                                              itk::Size<3> inVolImgDims,
                                              itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void TranslateImage(const itk::Image<unsigned char, 3>::Pointer &inVolImg, int x, int y, int z, itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void gpuErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                             itk::FlatStructuringElement<3> inKernel,
                             itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void SubstractSameSizedImages(const itk::Image<unsigned char, 3>::Pointer &img1,
                                                  const itk::Image<unsigned char, 3>::Pointer &img2,
                                                  itk::Image<unsigned char, 3>::Pointer &out_vol_img);

   void GetDifferenceInEachDirection(itk::FlatStructuringElement<3> inKernel);

   void GetOffsetFromCenter(itk::Size<3> inCenter, itk::Size<3> inPos, itk::Size<3> &outOffset );

   void GetKernelimageConnectedComponents(const itk::Image<unsigned char, 3>::Pointer &inVolImg, std::vector<int> &outConnectedComponents);

   void convertKernelToVolumetricImage(itk::FlatStructuringElement<3> inKernel, itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void CropVolImg(const itk::Image<unsigned char, 3>::Pointer &inVolImg, itk::Size<3> inUpperBound, itk::Size<3> inLowerBound, itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void GetSize(const itk::Image<unsigned char, 3>::Pointer &inVolImg, int outSize[3]);

   void GetSize(const itk::FlatStructuringElement<3> &inKernel, int outSize[3]);

   void GetRadius(const itk::FlatStructuringElement<3> &inKernel, int outSize[3]);

   void AddPaddingToImage(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                                      itk::Size<3> inLowerBound,
                                      itk::Size<3> inUpperBound,
                                      unsigned char inPaddingValue,
                                      itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void itkErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                               itk::FlatStructuringElement<3> inStructElem,
                               itk::Image<unsigned char, 3>::Pointer &outVolImg);

   void GetCenter(itk::Size<3> inDims, itk::Size<3> &outCenter);






 private:

};

#endif
