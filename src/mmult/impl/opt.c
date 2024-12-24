/* opt.c
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

/* Alternative Implementation */
#define MIN(a, b)(((a) < (b)) ? (a) : (b))
void* impl_scalar_opt(void* args)
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
  register size_t b = parsed_args -> size_B;
  register size_t min_ii_b_M = 0;
  register size_t min_jj_b_P = 0;
  register size_t min_kk_b_N = 0;
  register float val = 0;
  /*Matrix Multiplication*/

  for (register int i = 0; i < Matrix_A_m; i++) {
    for (register int j = 0; j < Matrix_B_p; j++) {
      dest[i * Matrix_B_p + j] = 0;
    }
  }

  for (register int ii = 0; ii < Matrix_A_m; ii += b) {
    for (register int jj = 0; jj < Matrix_B_p; jj += b) {
      for (register int kk = 0; kk < Matrix_A_n; kk += b) {
        min_ii_b_M = MIN(ii + b, Matrix_A_m);
        for (register int i = ii; i < min_ii_b_M; i++) {
          min_jj_b_P = MIN(jj + b, Matrix_B_p);
          for (register int j = jj; j < min_jj_b_P; j++) {
            val = dest[i * Matrix_B_p + j];
            min_kk_b_N = MIN(kk + b, Matrix_A_n);
            for (register int k = kk; k < min_kk_b_N; k++) {
              val += src0[i * Matrix_A_n + k] * src1[k * Matrix_B_p + j];
            }
            dest[i * Matrix_B_p + j] = val;
          }
        }
      }
    }
  }
  /* Done */
  return NULL;
}
