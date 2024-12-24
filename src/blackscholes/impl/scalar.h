/* scalar.h
 *
 * Author: Khalid Al-Hawaj
 * Date  : 13 Nov. 2023
 *
 * Header for the scalar function.
 */

#ifndef __IMPL_SCALAR_H_
#define __IMPL_SCALAR_H_

/* Function declaration */
void* impl_scalar(void* args);
void* impl_scalar_manual(void* args);
void blackScholes(float sptprice, float strike, float rate, float volatility, float otime, char otype, float timet, float* result);

#endif //__IMPL_SCALAR_H_
