#include <time.h>

#include <itkImage.h>
#include <itkFlatStructuringElement.h>
#include <itkMetaImageIO.h>
#include <itkImageFileWriter.h>
#include <itkSize.h>
#include <itkIndex.h>
#include <itkImageFileReader.h>
#include <itkMetaImageIO.h>
#include "morphology.h"


template< class T>
void Save(typename T::Pointer inVolImg, std::string inFileName) {

  typedef  itk::ImageFileWriter<T> VolumeWriterType;

  typename VolumeWriterType::Pointer writer = VolumeWriterType::New();
  itk::MetaImageIO::Pointer metaWriter = itk::MetaImageIO::New();
  writer->SetImageIO(metaWriter);
  metaWriter->SetDataFileName("LOCAL");
  writer->SetFileName(inFileName);
  writer->SetInput(inVolImg);
  writer->Write();
}

int main() {
  typedef itk::Image<unsigned char, 3> ImageType;
  typedef itk::FlatStructuringElement<3> StructuringElementType;
  typedef itk::ImageRegion< 3 > RegionType;

  ImageType::Pointer volImg = ImageType::New();
  itk::Size<3> size = {500, 500, 500};
  itk::Index<3> index = {0, 0, 0};
  ImageType::RegionType region(index, size);
  volImg->SetRegions(region);
  volImg->Allocate();
  volImg->FillBuffer(0);
  for (auto i = 0; i < size[2]; ++i) {
    for (auto j = 0; j < size[1]; ++j) {
      for (auto k = 0; k < size[0]; ++k) {
        if (i > 50 && i < 140 && j > 50 && j < 140 && k > 50 && k < 140) {
          itk::Index<3> pixel_index = {i, j, k};
          volImg->SetPixel(pixel_index, 1);
        }
      }
    }
  }

  StructuringElementType::RadiusType elementRadius;
  elementRadius.Fill(20);


  StructuringElementType structuringElement = StructuringElementType::Ball(elementRadius);

  Morphology morphology;

  // ImageType::Pointer itkImg;

  //m.itkErode(volImg, structuringElement, itkImg);
  //
  // unsigned char *structElemData = reinterpret_cast<unsigned char *>(structuringElement.Begin());
  // ImageType::Pointer structimg;
  // itk::Size<3> structElemDims = structuringElement.GetSize();
  // m.CopyDataToImageFromBuffer(structElemData, index, structElemDims, structimg);
  // Save<ImageType>(structimg, "structelem.mhd");

  return 0;
}
