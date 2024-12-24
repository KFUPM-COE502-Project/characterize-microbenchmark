/* vec.h
 *
 * Author: Khalid Al-Hawaj
 * Date  : 13 Nov. 2023
 *
 * Header for the vectorized function.
 */

#ifndef __IMPL_VEC_H_
#define __IMPL_VEC_H_

/* Function declaration */
void* impl_vector(void* args);
void blackScholesAVX(const float* sptprice, const float* strike, const float* rate, const float* volatility,
                     const float* otime, const char* otype, float* result);

#endif //__IMPL_VEC_H_
