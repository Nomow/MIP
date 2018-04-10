#include <itkSize.h>
#include <itkConstantPadImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkBinaryErodeImageFilter.h>

#include "morphologykernel.h"
#include "morphology.h"

Morphology::Morphology() {}

void Morphology::gpuErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                          itk::FlatStructuringElement<3> inStructElem,
                          itk::Image<unsigned char, 3>::Pointer &outVolImg) {

  typedef itk::Image<unsigned char, 3> ImageType;

  // pads image to fit structuring element to check for border pixels
  itk::Size<3> radius;
  radius.Fill(1);
  ImageType::Pointer paddedImg = ImageType::New();
  AddPaddingToImage(inVolImg, radius, radius, 1, paddedImg);



  // image dimensi
  itk::Size<3> imgDims = inVolImg->GetLargestPossibleRegion().GetSize();
  itk::Size<3> structElemDims = inStructElem.GetSize();

  // // converts data to buffer pointers to pass to gpu
  // unsigned char *imgData = inVolImg->GetBufferPointer();
  // unsigned char *structElemData = reinterpret_cast<unsigned char *>(inStructElem.Begin());
  // unsigned char *erodedData;
  // int convertedImgDims[3] = {imgDims[0], imgDims[1], imgDims[2]};
  // int convertedStructElemDims[3] = {structElemDims[0], structElemDims[1], structElemDims[2]};

}

void Morphology::AddPaddingToImage(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                                   itk::Size<3> inLowerBound,
                                   itk::Size<3> inUpperBound,
                                   unsigned char inPaddingValue,
                                   itk::Image<unsigned char, 3>::Pointer &outVolImg) {

  typedef itk::Image<unsigned char, 3> ImageType;
  typedef itk::ConstantPadImageFilter<ImageType, ImageType> ConstantPadImageFilterType;

  ConstantPadImageFilterType::Pointer padFilter = ConstantPadImageFilterType::New();
  padFilter->SetInput(inVolImg);
  padFilter->SetPadLowerBound(inLowerBound);
  padFilter->SetPadUpperBound(inUpperBound);
  padFilter->SetConstant(inPaddingValue);
  padFilter->Update();
  outVolImg = padFilter->GetOutput();
}

//
//
// void Morphology::Cv2dErosion(cv::cuda::GpuMat src, int elemShape, int radius,  cv::cuda::GpuMat &dst) {
//   cv::Mat element = cv::getStructuringElement(elemShape, cv::Size(radius * 2 + 1, radius * 2 + 1), cv::Point(radius, radius));
//   cv::Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, src.type(), element);
//   erodeFilter->apply(src, dst);
// }
//
// void Morphology::CopyDataToImageFromBuffer2d(unsigned char *inVolImgData, itk::Index<2> inStartIndex,
//                                            itk::Size<2> inVolImgDims, itk::Image<unsigned char, 2>::Pointer &outVolImg) {
//
//   typedef itk::Image<unsigned char, 2> ImageType;
//   typedef ImageType::RegionType RegionType;
//
//   outVolImg = ImageType::New();
//   ImageType::RegionType region(inStartIndex, inVolImgDims);
//   outVolImg->SetRegions(region);
//   outVolImg->Allocate();
//   outVolImg->FillBuffer(0);
//
//   for (auto i = 0; i < inVolImgDims[1]; ++i) {
//     for (auto j = 0; j < inVolImgDims[0]; ++j) {
//       itk::Index<2> pixelIndex = {i, j};
//        outVolImg->SetPixel(pixelIndex, *(inVolImgData + (i * inVolImgDims[0] + j)));
//     }
//   }
// }
//
// void Morphology::CopyDataToImageFromBuffer(unsigned char *inVolImgData, itk::Index<3> inStartIndex,
//                                            itk::Size<3> inVolImgDims, itk::Image<unsigned char, 3>::Pointer &outVolImg) {
//
//   typedef itk::Image<unsigned char, 3> ImageType;
//   typedef ImageType::RegionType RegionType;
//
//   outVolImg = ImageType::New();
//   ImageType::RegionType region(inS  m.CopyDataToImageFromBuffer(structElemData, index, structElemDims, structimg);
// tartIndex, inVolImgDims);
//   outVolImg->SetRegions(region);
//   outVolImg->Allocate();
//   outVolImg->FillBuffer(0);
//
//   for (auto i = 0; i < inVolImgDims[2]; ++i) {
//     for (auto j = 0; j < inVolImgDims[1]; ++j) {
//       for (auto k = 0; k < inVolImgDims[0]; ++k) {
//         itk::Index<3> pixelIndex = {k, j, i};
//         outVolImg->SetPixel(pixelIndex, *(inVolImgData +  (i *( inVolImgDims[0] * inVolImgDims[1] ) + (j * inVolImgDims[0]) + k)));
//       }
//     }
//   }
// }
//
// void Morphology::itk2dErode(const itk::Image<unsigned char, 2>::Pointer &inVolImg,
//                             itk::FlatStructuringElement<2> inStructElem,
//                             itk::Image<unsigned char, 2>::Pointer &outVolImg) {
//
//   typedef itk::Image<unsigned char, 2> ImageType;
//   typedef itk::FlatStructuringElement<2> FlatStructuringElementType;
//   typedef itk::BinaryErodeImageFilter <ImageType, ImageType, FlatStructuringElementType> BinaryErodeImageFilterType;
//
//   BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
//   erodeFilter->SetInput(inVolImg);
//   erodeFilter->SetKernel(inStructElem);
//   erodeFilter->SetErodeValue(100);
//   erodeFilter->SetForegroundValue(100);
//   erodeFilter->Update();
//   outVolImg = erodeFilter->GetOutput();
//
// }
//
// void Morphology::itkErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
//                             itk::FlatStructuringElement<3> inStructElem,
//                             itk::Image<unsigned char, 3>::Pointer &outVolImg) {
//
//   typedef itk::Image<unsigned char, 3> ImageType;
//   typedef itk::FlatStructuringElement<3> FlatStructuringElementType;
//   typedef itk::BinaryErodeImageFilter <ImageType, ImageType, FlatStructuringElementType> BinaryErodeImageFilterType;
//
//   BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
//   erodeFilter->SetInput(inVolImg);
//   erodeFilter->SetKernel(inStructElem);
//   erodeFilter->SetErodeValue(1);
//   erodeFilter->SetForegroundValue(1);
//   erodeFilter->Update();
//   outVolImg = erodeFilter->GetOutput();
// }
//
// void Morphology::AddPaddingToImage(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
//                                    int inLowerBound[3],
//                                    int inUpperBound[3],
//                                    unsigned char inPaddingValue,
//                                    itk::Image<unsigned char, 3>::Pointer &outVolImg) {
//
//   typedef itk::Image<unsigned char, 3> ImageType;
//   typedef itk::ConstantPadImageFilter<ImageType, ImageType> ConstantPadImageFilterType;
//
//   itk::Size<3> convertedLowerBound = {inLowerBound[0], inLowerBound[1], inLowerBound[2]};
//   itk::Size<3> convertedUpperBound = {inUpperBound[0], inUpperBound[1], inUpperBound[2]};
//
//   ConstantPadImageFilterType::Pointer padFilter = ConstantPadImageFilterType::New();
//   padFilter->SetInput(inVolImg);
//   padFilter->SetPadLowerBound(convertedLowerBound);
//   padFilter->SetPadUpperBound(convertedUpperBound);
//   padFilter->SetConstant(inPaddingValue);
//   padFilter->Update();
//   outVolImg = padFilter->GetOutput();
// }
//

//
//
// void Morphology::AddPaddingToImage(const itk::Image<unsigned char, 2>::Pointer &inVolImg,
//                                    itk::Size<2> inLowerBound,
//                                    itk::Size<2> inUpperBound,
//                                    unsigned char inPaddingValue,
//                                    itk::Image<unsigned char, 2>::Pointer &outVolImg) {
//
//   typedef itk::Image<unsigned char, 2> ImageType;
//   typedef itk::ConstantPadImageFilter<ImageType, ImageType> ConstantPadImageFilterType;
//
//   ConstantPadImageFilterType::Pointer padFilter = ConstantPadImageFilterType::New();
//   padFilter->SetInput(inVolImg);
//   padFilter->SetPadLowerBound(inLowerBound);
//   padFilter->SetPadUpperBound(inUpperBound);
//   padFilter->SetConstant(inPaddingValue);
//   padFilter->Update();
//   outVolImg = padFilter->GetOutput();
// }
