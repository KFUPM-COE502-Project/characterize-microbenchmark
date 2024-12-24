/* naive.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"

/* Naive Implementation */
#pragma GCC push_options
#pragma GCC optimize ("O1")
void* impl_scalar_naive(void* args)
{
  /* Get the argument struct */
  args_t * parsed_args = (args_t * ) args;

  /* Get all the arguments */
  register float * dest = (float * )(parsed_args -> output);
  register
  const float * src0 = (const float * )(parsed_args -> input0);
  register
  const float * src1 = (const float * )(parsed_args -> input1);
  register size_t Matrix_A_m = parsed_args -> input_A_m;
  register size_t Matrix_A_n = parsed_args -> input_A_n;
  register size_t Matrix_B_p = parsed_args -> input_B_p;

  /*Matrix Multiplication*/
  for (register int i = 0; i < Matrix_A_m; i++) {
    for (register int j = 0; j < Matrix_B_p; j++) {
      dest[i * Matrix_B_p + j] = 0;
      for (register int k = 0; k < Matrix_A_n; k++) {
        dest[i * Matrix_B_p + j] = dest[i * Matrix_B_p + j] + src0[i * Matrix_A_n + k] * src1[k * Matrix_B_p + j];
      }
    }
  }
  return NULL;
}
#pragma GCC pop_options
