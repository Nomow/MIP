#include <time.h>

#include <itkImage.h>
#include <itkFlatStructuringElement.h>
#include <itkSize.h>
#include <itkIndex.h>
#include "cuda/morphology2d.cuh"
#include <chrono>
#include "morphology.h"



int main() {


  typedef itk::FlatStructuringElement<3> StructuringElementType;
  typedef itk::Image<uint8_t, 3> ImageType;
  typedef itk::ImageRegion< 3 > RegionType;

  for (auto nnz = -1; nnz < 2; ++nnz) {
    for (auto nny = -1; nny < 2; ++nny) {
      for (auto nnx = -1; nnx < 2; ++nnx) {
        std::cout << "(" << nnx << " "  << nny << " "  << nnz << ")" << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }

  ImageType::Pointer volImg = ImageType::New();
  itk::Size<3> size = { 250, 250, 250};
  itk::Index<3> index = {0, 0, 0};
  ImageType::RegionType region(index, size);
  volImg->SetRegions(region);
  volImg->Allocate();
  volImg->FillBuffer(0);
  for (auto i = 0; i < size[2]; ++i) {
    for (auto j = 0; j < size[1]; ++j) {
      for (auto k = 0; k < size[0]; ++k) {
        if (i > 20 && i < 230 && j > 20 && j < 230 && k > 20 && k < 230) {
          itk::Index<3> pixel_index = {i, j,k};
          volImg->SetPixel(pixel_index, 1);
        }
      }
    }
  }

  StructuringElementType::RadiusType elementRadius;
  elementRadius.Fill(15);
  StructuringElementType structuringElement = StructuringElementType::Ball(elementRadius);
  Morphology m;
  m.Save(volImg, "volimg.mhd");

  ImageType::Pointer img1;
  m.Erode3d(volImg, structuringElement, img1);
  std::cout << "ITK ERODE:" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  m.itkErode(volImg, structuringElement, img1);

  auto finish = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = finish - start;

  std::cout << "Elapsed time: " << elapsed.count() << " s\n";

  m.Save(img1, "itkerode.mhd");

  //
  //
  // StructuringElementType::RadiusType elementRadius;
  // elementRadius.Fill(25);
  //
  //
  // StructuringElementType structuringElement = StructuringElementType::Ball(elementRadius);
  //
  // Morphology morphology;
  // ImageType::Pointer erodedImg;
  //
  // unsigned char *structElemData = reinterpret_cast<unsigned char *>(structuringElement.Begin());
  // ImageType::Pointer structimg;
  // itk::Size<3> structElemDims = structuringElement.GetSize();
  // morphology.CopyDataToImageFromBuffer(structElemData, index, structElemDims, structimg);
  // Save1<ImageType>(structimg, "structelem.mhd");
  //
  //
  // auto t1 = std::chrono::high_resolution_clock::now();
  // morphology.gpuErode(volImg, structuringElement, erodedImg);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
  // std::cout << duration << std::endl;
  // Save1<ImageType>(erodedImg, "eroded-vol-img.mhd");
  // Save1<ImageType>(volImg, "volImg.mhd");
  // t1 = std::chrono::high_resolution_clock::now();
  // //morphology.itkErode(volImg, structuringElement, erodedImg);
  // t2 = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
  // std::cout << duration << std::endl;

 //Save1<ImageType>(erodedImg, "itk eroded-vol-img.mhd");

  // ImageType::Pointer itkImg;

  //m.itkErode(volImg, structuringElement, itkImg);
  //


  return 0;
}
