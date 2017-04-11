#pragma once

#include <iostream>
#include "../types.hpp"
#include <cmath>

/* Placeholder for forward FFT */
void forward_r2c_3d_fpga(const int64_t n0,
                         const int64_t n1,
                         const int64_t n2,
                         float * datain,
                         std::complex<float>* dataout)
{
  double prod_sz = (double)(n0*n1*n2);
  int64_t output_size = n0*n1*(n2/2+1);
  for(int64_t i = 0 ; i < n0*n1*n2 ; i++)
  {
    int64_t i2 = i/2;
    if(i % 2)
    {
      dataout[i2].real(datain[i]*prod_sz);
    }
    else
    {
      dataout[i2].imag(datain[i]*prod_sz);
    }
  }
}

/* Placeholder for backward FFT */
void backward_c2r_3d_fpga(const int64_t n0,
                         const int64_t n1,
                         const int64_t n2,
                         std::complex<float> * datain,
                         float * dataout)
{
  int64_t output_size = n0*n1*(n2/2+1);
  for( int64_t i = 0 ; i < n0*n1*n2 ; i++)
  {
        int64_t i2 = i/2;
        if(i % 2)
        {
          dataout[i] = datain[i2].real();
        }
        else
        {
          dataout[i] = datain[i2].imag();
        }
  }
}
