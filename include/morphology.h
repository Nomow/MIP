#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <itkImage.h>
#include <itkSize.h>
#include <itkFlatStructuringElement.h>


class Morphology {
 public:
  Morphology();
  void gpuErode(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                itk::FlatStructuringElement<3> inStructElem,
                itk::Image<unsigned char, 3>::Pointer &outVolImg);

  void AddPaddingToImage(const itk::Image<unsigned char, 3>::Pointer &inVolImg,
                                     itk::Size<3> inLowerBound,
                                     itk::Size<3> inUpperBound,
                                     unsigned char inPaddingValue,
                                     itk::Image<unsigned char, 3>::Pointer &outVolImg);


 private:

};

#endif
