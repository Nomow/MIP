#include <itkSize.h>
#include <itkConstantPadImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkCropImageFilter.h>
#include <itkImageRegionIterator.h>
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"

#include "cuda/morphology.cuh"
#include "morphology.h"



Morphology::Morphology() {}

void Morphology::PaintObjectBorderOfVolImgWithKernelObject(const itk::Image<unsigned char, 3>::Pointer &inVolImg, itk::FlatStructuringElement<3> inKernel, itk::Image<unsigned char, 3>::Pointer &outVolImg) {
  typedef itk::Image<unsigned char, 3> ImageType;

  itk::Size<3> padBy = inKernel.GetRadius();

  // pads image to fit structuring element
  ImageType::Pointer paddedImg;
  AddPaddingToImage(inVolImg, padBy, padBy, 1, paddedImg);
  unsigned char *imgData = paddedImg->GetBufferPointer();
  unsigned char *kernelData = reinterpret_cast<unsigned char *>(inKernel.Begin());
  int imgDims[3];
  GetSize(paddedImg, imgDims);
  int kernelRadius[3];
  GetRadius(inKernel, kernelRadius);

  unsigned char *outData;
  PaintObjectBorderOfVolImgWithKernelObjectKernel(imgData, imgDims, kernelData, kernelRadius, outData);
  itk::Index<3> tempIndex = {0, 0, 0};
  itk::Size<3> tempImgDims = paddedImg->GetLargestPossibleRegion().GetSize();
  ImageType::Pointer paintedVolImg;
  CopyDataToImageFromBuffer(outData, tempIndex, tempImgDims, paintedVolImg);
  CropVolImg(paintedVolImg, padBy, padBy, outVolImg);
}

void Morphology::PaintObjectAndVolImgBorder(const itk::Image<unsigned char, 3>::Pointer &inVolImg, itk::Image<unsigned char, 3>::Pointer &outVolImg) {

  typedef itk::FlatStructuringElement<3> FlatStructuringElementType;
  typedef itk::Image<unsigned char, 3> ImageType;

  const int radius = 1;
  itk::Size<3> padBy;
  padBy.Fill(radius);

  // pads image to fit structuring element
  ImageType::Pointer paddedImg;
  AddPaddingToImage(inVolImg, padBy, padBy, 0, paddedImg);

  FlatStructuringElementType::RadiusType kernelRegionRadius;
  kernelRegionRadius.Fill(radius);
  FlatStructuringElementType kernel = FlatStructuringElementType::Box(kernelRegionRadius);

  // converts data for gpu kernel
  unsigned char *imgData = paddedImg->GetBufferPointer();
  unsigned char *kernelData = reinterpret_cast<unsigned char *>(kernel.Begin());
  int imgDims[3];
  GetSize(paddedImg, imgDims);
  int kernelRadius[3];
  GetRadius(kernel, kernelRadius);

  unsigned char *borderData;
  PaintObjectAndVolImgBorderKernel(imgData, imgDims, kernelData, kernelRadius, borderData);
  itk::Index<3> tempIndex = {0, 0, 0};
  itk::Size<3> tempImgDims = paddedImg->GetLargestPossibleRegion().GetSize();
  ImageType::Pointer borderImg;
  CopyDataToImageFromBuffer(borderData, tempIndex, tempImgDims, borderImg);
  CropVolImg(borderImg, padBy, padBy, outVolImg);
}

void Morphology::CopyDataToImageFromBuffer(unsigned char *inVolImgData,
                                           itk::Index<3> inStartIndex,
                                           itk::Size<3> inVolImgDims,
                                           itk::Image<unsigned char, 3>::Pointer &outVolImg) {

  typedef itk::Image<unsigned char, 3> ImageType;
  typedef ImageType::RegionType RegionType;

  outVolImg = ImageType::New();
  ImageType::RegionType region(inStartIndex, inVolImgDims);
  outVolImg->SetRegions(region);
  outVolImg->Allocate();
  outVolImg->FillBuffer(0);

  for (auto i = 0; i < inVolImgDims[2]; ++i) {
    for (auto j = 0; j < inVolImgDims[1]; ++j) {
      for (auto k = 0; k < inVolImgDims[0]; ++k) {
        itk::Index<3> pixelIndex = {k, j, i};
        outVolImg->SetPixel(pixelIndex, *(inVolImgData +  (i *( inVolImgDims[0] * inVolImgDims[1] ) + (j * inVolImgDims[0]) + k)));
      }
    }
  }
}

void Morphology::TranslateImage(const itk::Image<unsigned char, 3>::Pointer &inVolImg, int x, int y, int z, itk::Image<unsigned char, 3>::Pointer &outVolImg) {
  typedef itk::TranslationTransform<double, 3> TranslationTransformType;
  typedef itk::Image<unsigned char, 3> ImageType;

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  TranslationTransformType::Pointer transform = TranslationTransformType::New();
  TranslationTransformType::OutputVectorType translation;
  translation[0] = x;
  translation[1] = y;
  translation[2] = z;

  transform->Translate(translation);

  ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
  resampleFilter->SetTransform(transform.GetPointer());
  resampleFilter->SetInput(inVolImg);
  resampleFilter->SetSize(inVolImg->GetLargestPossibleRegion().GetSize());
  resampleFilter->Update();
  outVolImg = resampleFilter->GetOutput();
}



void Morphology::gpuErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                          itk::FlatStructuringElement<3> inKernel,
                          itk::Image<unsigned char, 3>::Pointer &outVolImg) {

  // typedef itk::Image<unsigned char, 3> ImageType;
  // ImageType::Pointer borderVolImg;
  // PaintObjectAndVolImgBorder(inVolImg, borderVolImg);
  // PaintObjectBorderOfVolImgWithKernelObject(borderVolImg, inKernel, outVolImg);
  outVolImg = inVolImg;
  GetDifferenceInEachDirection(inKernel);
}

void Morphology::SubstractSameSizedImages(const itk::Image<unsigned char, 3>::Pointer &img1,
                                               const itk::Image<unsigned char, 3>::Pointer &img2,
                                               itk::Image<unsigned char, 3>::Pointer &out_vol_img) {

  typedef itk::Image<unsigned char, 3> ImageType;

  itk::Size<3> dimensions1 = img1->GetLargestPossibleRegion().GetSize();
  out_vol_img  = ImageType::New();

    // fills image with min values
    out_vol_img->SetRegions(dimensions1);
    out_vol_img->Allocate();
    out_vol_img->FillBuffer(0);

    for (auto i = 0; i < dimensions1[2]; ++i) {
      for (auto j = 0; j < dimensions1[1]; ++j) {
        for (auto k = 0; k < dimensions1[0]; ++k) {
          itk::Index<3> pixel_index = {k, j, i};

          // assigns max value
          if(img1->GetPixel(pixel_index) == 1 && img2->GetPixel(pixel_index) == 0) {
            out_vol_img->SetPixel(pixel_index, 1);
          }
        }
      }
    }
}

void Morphology::GetDifferenceInEachDirection(itk::FlatStructuringElement<3> inKernel) {
  typedef itk::FlatStructuringElement<3> FlatStructuringElementType;
  typedef itk::Image<unsigned char, 3> ImageType;
  itk::Size<3> size = {3, 3, 3};

  ImageType::Pointer kernelVolImg;
  convertKernelToVolumetricImage(inKernel, kernelVolImg);
  ImageType::Pointer tempVolImg;
  itk::Size<3> center;
  int counter = 0;
  std::vector<std::vector<int> > difference(27);
  for (int i = 0; i < size[2]; ++i) {
    for (int j = 0; j < size[1]; ++j) {
      for (int k = 0; k < size[0]; ++k) {
        ImageType::Pointer substractedImg;
        if(counter != 13) {
          TranslateImage(kernelVolImg, k -1, j - 1, i - 1, tempVolImg);
          SubstractSameSizedImages(kernelVolImg, tempVolImg, substractedImg);
        } else {
          substractedImg = kernelVolImg;
        }
        std::vector<int> ConnectedComponent;
        GetKernelimageConnectedComponents(substractedImg, ConnectedComponent);
        difference[counter] = ConnectedComponent;
        ++counter;
      }
    }
  }
}


void Morphology::GetCenter(itk::Size<3> inDims, itk::Size<3> &outCenter) {
  for (int i = 0; i < 3; ++i) {
    outCenter[i] = inDims[i] / 2;
  }
}


void Morphology::GetOffsetFromCenter(itk::Size<3> inCenter, itk::Size<3> inPos, itk::Size<3> &outOffset ) {
  for(int i = 0; i < 3; ++i) {
      outOffset[i] = inPos[i] - inCenter[i];
  }
}

void Morphology::GetKernelimageConnectedComponents(const itk::Image<unsigned char, 3>::Pointer &inVolImg, std::vector<int> &outConnectedComponents) {
   itk::Size<3> size = inVolImg->GetLargestPossibleRegion().GetSize();
   itk::Size<3> center;
   GetCenter(size, center);
   outConnectedComponents.reserve(size[0] * size[1] * size[2] * 3);
   for (int z = 0; z  < size[2]; ++z) {
     for (int y = 0; y < size[1]; ++y) {
       for (int x = 0; x < size[0]; ++x) {
          itk::Index<3> pixelIndex = {z, y, x};
          unsigned char pixel = inVolImg->GetPixel(pixelIndex);
          if (pixel == 1) {
            itk::Size<3> pos = {x, y, z};
            itk::Size<3> offset;
            GetOffsetFromCenter(center , pos, offset);
            outConnectedComponents.push_back(offset[0]);
            outConnectedComponents.push_back(offset[1]);
            outConnectedComponents.push_back(offset[2]);
          }
       }
     }
   }
}

void Morphology::convertKernelToVolumetricImage(itk::FlatStructuringElement<3> inKernel, itk::Image<unsigned char, 3>::Pointer &outVolImg) {
  typedef itk::Image<unsigned char, 3> ImageType;
  typedef itk::FlatStructuringElement<3> FlatStructuringElementType;

  outVolImg = ImageType::New();
  outVolImg->SetRegions(inKernel.GetSize());
  outVolImg->Allocate();

  itk::ImageRegionIterator<ImageType> kernelImageIt;
  kernelImageIt = itk::ImageRegionIterator<ImageType>(outVolImg, outVolImg->GetRequestedRegion() );
  FlatStructuringElementType::ConstIterator kernelIt = inKernel.Begin();

   while (!kernelImageIt.IsAtEnd())  {
     kernelImageIt.Set(*kernelIt ? true : false);
     ++kernelImageIt;
     ++kernelIt;
   }
}


void Morphology::CropVolImg(const itk::Image<unsigned char, 3>::Pointer &inVolImg, itk::Size<3> inUpperBound, itk::Size<3> inLowerBound, itk::Image<unsigned char, 3>::Pointer &outVolImg) {
  typedef itk::Image<unsigned char, 3> ImageType;
  typedef itk::CropImageFilter <ImageType, ImageType> CropImageFilterType;
  CropImageFilterType::Pointer cropFilter = CropImageFilterType::New();
  cropFilter->SetInput(inVolImg);
  cropFilter->SetUpperBoundaryCropSize(inUpperBound);
  cropFilter->SetLowerBoundaryCropSize(inLowerBound);
  cropFilter->Update();
  outVolImg = cropFilter->GetOutput();
}

void Morphology::GetSize(const itk::Image<unsigned char, 3>::Pointer &inVolImg, int outSize[3]) {
  itk::Size<3> dims = inVolImg->GetLargestPossibleRegion().GetSize();
  outSize[0] = dims[0];
  outSize[1] = dims[1];
  outSize[2] = dims[2];
}

void Morphology::GetSize(const itk::FlatStructuringElement<3> &inKernel, int outSize[3]) {
  itk::Size<3> dims = inKernel.GetSize();
  outSize[0] = dims[0];
  outSize[1] = dims[1];
  outSize[2] = dims[2];
}


void Morphology::GetRadius(const itk::FlatStructuringElement<3> &inKernel, int outSize[3]) {
  itk::Size<3> dims = inKernel.GetRadius();
  outSize[0] = dims[0];
  outSize[1] = dims[1];
  outSize[2] = dims[2];
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


void Morphology::itkErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                            itk::FlatStructuringElement<3> inStructElem,
                            itk::Image<unsigned char, 3>::Pointer &outVolImg) {

  typedef itk::Image<unsigned char, 3> ImageType;
  typedef itk::FlatStructuringElement<3> FlatStructuringElementType;
  typedef itk::BinaryErodeImageFilter <ImageType, ImageType, FlatStructuringElementType> BinaryErodeImageFilterType;

  BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
  erodeFilter->SetInput(inVolImg);
  erodeFilter->SetKernel(inStructElem);
  erodeFilter->SetErodeValue(1);
  erodeFilter->SetForegroundValue(1);
  erodeFilter->Update();
  outVolImg = erodeFilter->GetOutput();
}
