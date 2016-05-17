#include <stdio.h>
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);
void computeGold( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  for( unsigned int i = 1; i < len; ++i) 
  {
      total_sum += idata[i-1];
      reference[i] = idata[i-1] + reference[i-1];
  }
  if (total_sum != reference[len-1])
      printf("Warning: exceeding single-precision accuracy！\n");
}

