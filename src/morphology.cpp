#include <itkSize.h>
#include <itkConstantPadImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkCropImageFilter.h>
#include <itkImageRegionIterator.h>
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "cuda/morphology2d.cuh"
#include "cuda/morphology3d.cuh"
#include <itkPasteImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkMetaImageIO.h>
#include <queue>
#include "morphology.h"

#include <chrono>  // for high_resolution_clock

Morphology::Morphology() {}



void Morphology::Erode3d(const itk::Image<uint8_t, 3>::Pointer &inSrc, itk::FlatStructuringElement<3> inKernel, itk::Image<uint8_t, 3>::Pointer &outDst) {
  typedef itk::Image<uint8_t, 3> ImageType;
  typedef ImageType::RegionType RegionType;

  itk::Size<3> srcDims = inSrc->GetLargestPossibleRegion().GetSize();
  // applies padding to src image for border detection
  ImageType::Pointer paddedBorderSrc;
  itk::Size<3> borderPad = {1, 1, 1};
  uint8_t pval = 0;
  AddPaddingToImg(inSrc, borderPad, borderPad, 1, paddedBorderSrc);
  Save(paddedBorderSrc, "ppp.mhd");
  itk::Size<3> paddedBorderSrcDims = paddedBorderSrc->GetLargestPossibleRegion().GetSize();

  // pads image by kernel radius
  itk::Size<3> kernelRadius = inKernel.GetRadius();
  ImageType::Pointer paddedByRadiusSrc;

  AddPaddingToImg(inSrc, kernelRadius, kernelRadius, 1, paddedByRadiusSrc);
  itk::Size<3> paddedSrcDims = paddedByRadiusSrc->GetLargestPossibleRegion().GetSize();
  Save(paddedByRadiusSrc, "paddedByRadiusSrc.mhd");
  // copies padded by radius image to device memory
  uint8_t *paddedSrcPtr = paddedByRadiusSrc->GetBufferPointer();
  cudaPitchedPtr paddedDeviceSrc;
  cudamorph3d::CopyFromHostToDeviceMemory(paddedSrcPtr, paddedSrcDims[0], paddedSrcDims[1], paddedSrcDims[2], paddedDeviceSrc);

  // gets difference set of kernel
  std::vector<std::vector<int> > hostDifferenceSet;
  GetDifferenceInEachDirection(inKernel, hostDifferenceSet);

  // copies differenceset to device memory
  std::vector<int32_t*> deviceDifferenceSet(27);
  for (int i = 0; i < 27; ++i) {
    std::cout << hostDifferenceSet[i].size() << std::endl;
    cudamorph3d::CopyFromHostToDeviceMemory(&hostDifferenceSet[i][0], hostDifferenceSet[i].size(), deviceDifferenceSet[i]);
  }
  const int numStreams = 8;
  cudaStream_t streams[numStreams];
  for (int i = 0; i < numStreams; i++) {
     cudaStreamCreate(&streams[i]);
  }
  int kernelCalls = 0;
  // 0 - not  processed pixels
  // 1 - foreground pixels
  // 2 - processed pixel
  auto start = std::chrono::high_resolution_clock::now();
  std::queue<itk::Index<3>> indexQueue;
    // iterates over each pixel of an image in specified region
  for (auto z = borderPad[2]; z < paddedBorderSrcDims[2] - borderPad[2]; ++z) {
    for (auto y = borderPad[1]; y < paddedBorderSrcDims[1] - borderPad[1]; ++y) {
      for (auto x = borderPad[0]; x < paddedBorderSrcDims[0] - borderPad[0]; ++x) {

        // 0 pixel value means unprocessed pixel and check neighbors for foreground value
        itk::Index<3> pixelIndex = {x, y, z};
        //std::cout << pixelIndex << std::endl;

        uint8_t pixelVal = paddedBorderSrc->GetPixel(pixelIndex);
        if (pixelVal == 0) {
          bool isBorderPixel = false;
          // iterates over neighbours to find a border pixel
          for (auto nz = -1; nz < 2; ++nz) {
            for (auto ny = -1; ny < 2; ++ny) {
              for (auto nx = -1; nx < 2; ++nx) {
                itk::Index<3> neighbPixelIndex = {x + nx, y + ny, z + nz};
                uint8_t neighbPixelVal = paddedBorderSrc->GetPixel(neighbPixelIndex);
                // if neighbour is foreground value then uses full kernel to erode image
                if (neighbPixelVal == 1) {
                  isBorderPixel = true;
                  break;
                }
              }
            }
          }

          // finds all border pixels
          if (isBorderPixel) {
            uint32_t cx = pixelIndex[0] + kernelRadius[0] - 1;
            uint32_t cy = pixelIndex[1] + kernelRadius[1] - 1;
            uint32_t cz = pixelIndex[2] + kernelRadius[2] - 1;
              cudamorph3d::PaintDifferenceSet(paddedDeviceSrc, cx, cy, cz, deviceDifferenceSet[13], hostDifferenceSet[13].size(), 0);
            //
            //  uint8_t * ptr;
            //  cudamorph3d::CopyFromDeviceToHostMemory(paddedDeviceSrc, paddedSrcDims[0], paddedSrcDims[1], paddedSrcDims[2], ptr);
            //  itk::Index<3> index = {0, 0, 0};
            //  cudaDeviceSynchronize();
            //  CopyDataFromBufferToImg(ptr, index, paddedSrcDims,paddedBorderSrc);
            //
            //  Save(paddedBorderSrc, "ffff" + std::to_string(x* y * z)+".mhd");

             std::cout << hostDifferenceSet[13].size() << std::endl;
             ++kernelCalls;
             paddedBorderSrc->SetPixel(pixelIndex, 2); // full kernel
             indexQueue.push(pixelIndex);
             std::cout << "queue started" << std::endl;
             std::cout << indexQueue.size() << std::endl;
             while(!indexQueue.empty()) {
               itk::Index<3> currIndex = indexQueue.front();
               indexQueue.pop();
               for (auto nz = -1, iz = 2; nz < 2; ++nz, --iz) {
                 for (auto ny = -1, iy = 2; ny < 2; ++ny, --iy) {
                   for (auto nx = -1, ix = 2; nx < 2; ++nx, --ix) {

                     itk::Index<3> neighbPixelIndex = {currIndex[0] + nx, currIndex[1] + ny, currIndex[2] + nz};
                     uint8_t neighbPixelVal = paddedBorderSrc->GetPixel(neighbPixelIndex);
                     if (neighbPixelVal == 0) {
                       bool isBorderPixel1 = false;
                       for (auto nnz = -1; nnz < 2; ++nnz) {
                         for (auto nny = -1; nny < 2; ++nny) {
                           for (auto nnx = -1; nnx < 2; ++nnx) {
                             itk::Index<3> nneighbPixelIndex = {neighbPixelIndex[0] + nnx, neighbPixelIndex[1] + nny, neighbPixelIndex[2] + nnz};
                             uint8_t nneighbPixelVal = paddedBorderSrc->GetPixel(nneighbPixelIndex);
                             if(nneighbPixelVal == 1) {
                               isBorderPixel1 = true;
                               break;
                             }
                           }
                         }
                       }


                       if(isBorderPixel1) {
                          int ind1 = iz * (3 * 3 ) + (iy * 3) + ix;
                          uint32_t ccx = neighbPixelIndex[0] + kernelRadius[0] -1;
                          uint32_t ccy = neighbPixelIndex[1] + kernelRadius[1] -1;
                          uint32_t ccz = neighbPixelIndex[2] + kernelRadius[2] - 1;

                          cudamorph3d::PaintDifferenceSet(paddedDeviceSrc, ccx, ccy, ccz, deviceDifferenceSet[ind1], hostDifferenceSet[ind1].size(), streams[kernelCalls % numStreams]);
                          ++kernelCalls;
                          paddedBorderSrc->SetPixel(neighbPixelIndex, 2); // full kernel
                          indexQueue.push(neighbPixelIndex);
                       } else {
                         paddedBorderSrc->SetPixel(neighbPixelIndex, 5); // full kernel
                       }


                     }


                   }
                 }
               }
             }

          } else {
            paddedBorderSrc->SetPixel(pixelIndex, 5); // full kernel
          }




        }
      }
    }
  }
  Save(paddedBorderSrc, "runned.mhd");

  uint8_t * ptr;
  cudamorph3d::CopyFromDeviceToHostMemory(paddedDeviceSrc, paddedSrcDims[0], paddedSrcDims[1], paddedSrcDims[2], ptr);
  itk::Index<3> index = {0, 0, 0};
  cudaDeviceSynchronize();

  CopyDataFromBufferToImg(ptr, index, paddedSrcDims,paddedBorderSrc);
  Save(paddedBorderSrc, "paddedBorderSrcbeforeCrop.mhd");

    CropVolImg(paddedBorderSrc, kernelRadius, kernelRadius, paddedBorderSrc);



auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = finish - start;
std::cout << "Elapsed time: " << elapsed.count() << " s\n";
Save(paddedBorderSrc, "paddedBorderSrc555.mhd");

}

void Morphology::PasteImgToImg(const itk::Image<uint8_t, 3>::Pointer &inSrc1,
                               const itk::Image<uint8_t, 3>::Pointer &inSrc2,
                               itk::Index<3> inStartIndex,
                               itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;
  typedef itk::PasteImageFilter <ImageType, ImageType> PasteImageFilterType;

  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();
  pasteFilter->SetSourceImage(inSrc1);
  pasteFilter->SetDestinationImage(inSrc2);
  pasteFilter->SetSourceRegion(inSrc1->GetLargestPossibleRegion());
  pasteFilter->SetDestinationIndex(inStartIndex);
  pasteFilter->Update();
  outDst = pasteFilter->GetOutput();
}

void Morphology::GetCenter(itk::Size<3> inDims, itk::Size<3> &outCenter) {
  for (int i = 0; i < 3; ++i) {
    outCenter[i] = inDims[i] / 2;
  }
}

void Morphology::GetOffsetFromCenter(itk::Size<3> inCenter, itk::Size<3> inPos, int32_t outOffset[3] ) {
  for(int i = 0; i < 3; ++i) {
      outOffset[i] = (int32_t) inCenter[i] - inPos[i];

  }
}

void Morphology::AddPaddingToImg(const itk::Image<uint8_t, 3>::Pointer &inSrc,
                                 itk::Size<3> inLowerBound,
                                 itk::Size<3> inUpperBound,
                                 uint8_t inPaddingVal,
                                 itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;
  typedef ImageType::RegionType RegionType;

  itk::Size<3> srcDims = inSrc->GetLargestPossibleRegion().GetSize();
  ImageType::RegionType region;
  ImageType::IndexType start;
  start.Fill( 0 );
  region.SetIndex(start);
  ImageType::SizeType dims;
  dims[0] = srcDims[0] + inLowerBound[0] + inUpperBound[0];
  dims[1] = srcDims[1] + inLowerBound[1] + inUpperBound[1];
  dims[2] = srcDims[2] + inLowerBound[2] + inUpperBound[2];
  ImageType::Pointer padImg = ImageType::New();
  padImg->SetRegions(dims);
  padImg->Allocate();
  padImg->FillBuffer(inPaddingVal);
  itk::Index<3> startIndex = {inLowerBound[0], inLowerBound[1], inLowerBound[2]};
  PasteImgToImg(inSrc, padImg, startIndex, outDst);

}

void Morphology::AddPaddingToImgRegion(const itk::Image<uint8_t, 3>::Pointer &inSrc,
                                 itk::Size<3> inLowerBound,
                                 itk::Size<3> inUpperBound,
                                 uint8_t inPaddingVal,
                                 itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;
  typedef itk::ConstantPadImageFilter<ImageType, ImageType> ConstantPadImageFilterType;

  ConstantPadImageFilterType::Pointer padFilter = ConstantPadImageFilterType::New();
  padFilter->SetInput(inSrc);
  padFilter->SetPadLowerBound(inLowerBound);
  padFilter->SetPadUpperBound(inUpperBound);
  padFilter->SetConstant(inPaddingVal);
  padFilter->Update();
  outDst = padFilter->GetOutput();
}

void Morphology::CopyDataFromBufferToImg(uint8_t *inSrc,
                                         itk::Index<3> inStartIndex,
                                         itk::Size<3> inDims,
                                         itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;
  typedef ImageType::RegionType RegionType;

  // inits img
  outDst = ImageType::New();
  ImageType::RegionType region(inStartIndex, inDims);
  outDst->SetRegions(region);
  outDst->Allocate();
  outDst->FillBuffer(0);

  // copies data from buffer to image
  for (auto i = 0; i < inDims[2]; ++i) {
    for (auto j = 0; j < inDims[1]; ++j) {
      for (auto k = 0; k < inDims[0]; ++k) {
        itk::Index<3> pixelIndex = {k, j, i};
        outDst->SetPixel(pixelIndex, *(inSrc +  (i *( inDims[0] * inDims[1] ) + (j * inDims[0]) + k)));
      }
    }
  }
}

void Morphology::SubstractSameSizedImgs(const itk::Image<uint8_t, 3>::Pointer &inSrc1,
                                        const itk::Image<uint8_t, 3>::Pointer &inSrc2,
                                        itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;

  itk::Size<3> dimensions1 = inSrc1->GetLargestPossibleRegion().GetSize();

  // inits image
  outDst  = ImageType::New();
  outDst->SetRegions(dimensions1);
  outDst->Allocate();
  outDst->FillBuffer(0);

  // substracts pixels
  for (auto i = 0; i < dimensions1[2]; ++i) {
    for (auto j = 0; j < dimensions1[1]; ++j) {
      for (auto k = 0; k < dimensions1[0]; ++k) {
        itk::Index<3> pixel_index = {k, j, i};
        if(inSrc1->GetPixel(pixel_index) == 1 && inSrc2->GetPixel(pixel_index) == 0) {
          outDst->SetPixel(pixel_index, 1);
        }
      }
    }
  }
}

void Morphology::Save(const itk::Image<uint8_t, 3>::Pointer &inSrc, std::string inName) {

  typedef  itk::ImageFileWriter<itk::Image<uint8_t, 3>> VolumeWriterType;
  typename VolumeWriterType::Pointer writer = VolumeWriterType::New();

  itk::MetaImageIO::Pointer metaWriter = itk::MetaImageIO::New();
  writer->SetImageIO(metaWriter);
  metaWriter->SetDataFileName("LOCAL");
  writer->SetFileName(inName);
  writer->SetInput(inSrc);
  writer->Write();
}

void Morphology::TranslateImg(const itk::Image<uint8_t, 3>::Pointer &inSrc, int32_t inX, int32_t inY,
                              int32_t inZ, itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::TranslationTransform<double, 3> TranslationTransformType;
  typedef itk::Image<uint8_t, 3> ImageType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  // inits translation vector
  TranslationTransformType::Pointer transform = TranslationTransformType::New();
  TranslationTransformType::OutputVectorType translation;
  translation[0] = inX;
  translation[1] = inY;
  translation[2] = inZ;
  transform->Translate(translation);

  // resmaples image
  ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
  resampleFilter->SetTransform(transform.GetPointer());
  resampleFilter->SetInput(inSrc);
  resampleFilter->SetSize(inSrc->GetLargestPossibleRegion().GetSize());
  resampleFilter->Update();
  outDst = resampleFilter->GetOutput();
}

void Morphology::CropVolImg(const itk::Image<uint8_t, 3>::Pointer &inSrc, itk::Size<3> inUpperBound,
                            itk::Size<3> inLowerBound, itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;
  typedef itk::CropImageFilter <ImageType, ImageType> CropImageFilterType;

  CropImageFilterType::Pointer cropFilter = CropImageFilterType::New();
  cropFilter->SetInput(inSrc);
  cropFilter->SetUpperBoundaryCropSize(inUpperBound);
  cropFilter->SetLowerBoundaryCropSize(inLowerBound);
  cropFilter->Update();
  outDst = cropFilter->GetOutput();
}

void Morphology::KernelToImg(itk::FlatStructuringElement<3> inKernel, itk::Image<uint8_t, 3>::Pointer &outDst) {

  typedef itk::Image<uint8_t, 3> ImageType;
  typedef itk::FlatStructuringElement<3> FlatStructuringElementType;

  // inits img
  outDst = ImageType::New();
  outDst->SetRegions(inKernel.GetSize());
  outDst->Allocate();

  // iterates over img and copies kernel data
  itk::ImageRegionIterator<ImageType> kernelImageIt;
  kernelImageIt = itk::ImageRegionIterator<ImageType>(outDst, outDst->GetRequestedRegion() );
  FlatStructuringElementType::ConstIterator kernelIt = inKernel.Begin();

  while (!kernelImageIt.IsAtEnd())  {
     kernelImageIt.Set(*kernelIt ? 1 : 0);
     ++kernelImageIt;
     ++kernelIt;
  }
}

void Morphology::GetDifferenceInEachDirection(itk::FlatStructuringElement<3> inKernel, std::vector<std::vector<int32_t> > &outDifferenceSet) {

  typedef itk::FlatStructuringElement<3> FlatStructuringElementType;
  typedef itk::Image<uint8_t, 3> ImageType;

  itk::Size<3> size = {3, 3, 3};
  ImageType::Pointer kernelImg;
  KernelToImg(inKernel, kernelImg);
  ImageType::Pointer tempVolImg;
  itk::Size<3> center;
  int32_t counter = 0;
  outDifferenceSet.resize(27);
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto k = 0; k < 3; ++k) {
        ImageType::Pointer substractedImg;
        if(counter != 13) {

          TranslateImg(kernelImg, k - 1, j - 1, i - 1, tempVolImg);
          SubstractSameSizedImgs(kernelImg, tempVolImg, substractedImg);
          if(counter == 2) {
            Save(kernelImg, "kernelImg.mhd");
            Save(tempVolImg, "tempVolImg.mhd");
            Save(substractedImg, "substractedImg.mhd");

          }
        } else {
          substractedImg = kernelImg;
        }
        //Save(substractedImg, "kerneldiff" + std::to_string(counter) + ".mhd");
        std::vector<int32_t> ConnectedComponent;
        GetComponentOffsetFromCenter(substractedImg, ConnectedComponent);
        outDifferenceSet[counter] = ConnectedComponent;
        ++counter;
      }
    }
  }
}

void Morphology::GetComponentOffsetFromCenter(const itk::Image<uint8_t, 3>::Pointer &inVolImg, std::vector<int32_t> &outConnectedComponents) {
   itk::Size<3> size = inVolImg->GetLargestPossibleRegion().GetSize();
   itk::Size<3> center;
   GetCenter(size, center);
   outConnectedComponents.reserve(size[0] * size[1] * size[2] * 3);
   std::cout << "========" << std::endl;
   std::cout << "size: " << size[0] << " " << size[1] << " " << size[2] << " s: " <<  outConnectedComponents.size() << std::endl;

   for (auto z = 0; z  < size[2]; ++z) {
     for (auto y = 0; y < size[1]; ++y) {
       for (auto x = 0; x < size[0]; ++x) {
          itk::Index<3> pixelIndex = {x, y, z};
          uint8_t pixel = inVolImg->GetPixel(pixelIndex);
          if (pixel == 1) {
            itk::Size<3> pos = {x, y, z};
            int32_t offset[3];
            GetOffsetFromCenter(center , pos, offset);
            outConnectedComponents.push_back(offset[0]);
            outConnectedComponents.push_back(offset[1]);
            outConnectedComponents.push_back(offset[2]);
          }
       }
     }
   }
   std::cout << "size1: " << size[0] << " " << size[1] << " " << size[2] << " s: " <<  outConnectedComponents.size() << std::endl;

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
