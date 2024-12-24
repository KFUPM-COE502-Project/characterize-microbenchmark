/* types.h
 *
 * Author: Khalid Al-Hawaj
 * Date  : 13 Nov. 2023
 * 
 * This file contains all required types decalartions.
*/

#ifndef __INCLUDE_TYPES_H_
#define __INCLUDE_TYPES_H_

typedef struct {
  byte*   input0;
  byte*   input1;
  byte*   output;

  size_t input_A_m;
  size_t input_A_n;
  size_t input_B_p;
  size_t size;

  size_t size_B;
  size_t impType;
  size_t i_start;
  size_t i_end;
  size_t j_start;
  size_t j_end;
  size_t block_size;
  int     cpu;
  int     nthreads;
} args_t;

#endif //__INCLUDE_TYPES_H_
