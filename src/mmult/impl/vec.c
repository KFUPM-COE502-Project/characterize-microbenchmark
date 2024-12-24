/* vec.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/*  -> SIMD header file  */
#if defined(__amd64__) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch__) || defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif
#define MIN(a, b)(((a) < (b)) ? (a) : (b))

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

#include "common/vmath.h"
/* Include application-specific headers */
#include "include/types.h"

/* Alternative Implementation */
void* impl_vector(void* args)
{
  #if defined(__amd64__) || defined(__x86_64__)
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
  int implemintationType = parsed_args -> impType; // 0 for naive, 1 for optimal
  const int max_vlen = 32 / sizeof(float);
  register size_t b = parsed_args -> size_B;

  /* Prepare for matrix multiplication */
  register size_t min_ii_b_M = 0;
  register size_t min_jj_b_P = 0;
  register size_t min_kk_b_N = 0;
  __m256i vm = _mm256_set1_epi32(0xFFFFFFFF);

  /* Transpose src1 matrix */
  float * tranSrc1 = (float * ) malloc(Matrix_B_p * Matrix_A_n * sizeof(float));
  for (register size_t i = 0; i < Matrix_A_n; ++i) {
    for (register size_t j = 0; j < Matrix_B_p; ++j) {
      tranSrc1[j * Matrix_A_n + i] = src1[i * Matrix_B_p + j];
    }
  }

  /* Initialize destination matrix */
  for (register int i = 0; i < Matrix_A_m * Matrix_B_p; i++) {
    dest[i] = 0;
  }

  /* Matrix multiplication (Naive or Blocked) */
  if (implemintationType == 0) { // Naive Implementation
    for (register int i = 0; i < Matrix_A_m; i++) {
      for (register int j = 0; j < Matrix_B_p; j++) {
        dest[i * Matrix_B_p + j] = 0; // Initialize destination element to 0
        vm = _mm256_set1_epi32(0xFFFFFFFF); // Initialize mask to load all elements
        __m256 val_v = _mm256_setzero_ps(); // Zero out the vector accumulator

        // Process k-blocks for matrix multiplication
        for (register int k = 0; k < Matrix_A_n; k += max_vlen) {
          register int rem = Matrix_A_n - k;
          int hw_vlen = rem < max_vlen ? rem : max_vlen; // Determine valid number of elements

          // Update the mask to handle partial loading
          if (hw_vlen < max_vlen) {
            unsigned int m[max_vlen];
            for (size_t jj = 0; jj < max_vlen; jj++) {
              m[jj] = (jj < hw_vlen) ? 0xFFFFFFFF : 0x00000000; // Set valid elements to 1
            }
            vm = _mm256_setr_epi32(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]); // Set the mask
          }

          // Load data for matrix A and matrix B
          __m256 vecA = _mm256_maskload_ps( & src0[i * Matrix_A_n + k], vm);
          __m256 vecB = _mm256_maskload_ps( & tranSrc1[j * Matrix_A_n + k], vm);

          // Perform element-wise multiplication
          __m256 vecMult = _mm256_mul_ps(vecA, vecB);
          //val_v = _mm256_add_ps(val_v, vecMult);
          // Accumulate the results into val_v
          float temp[max_vlen];
          _mm256_storeu_ps(temp, vecMult);

          // Sum the values in temp and add to the destination matrix
          for (size_t l = 0; l < hw_vlen; l++) {
            dest[i * Matrix_B_p + j] += temp[l]; // Accumulate the elements
          }

        }

      }
    }
  } else { // Blocked Implementation
    for (register int ii = 0; ii < Matrix_A_m; ii += b) {
      for (register int jj = 0; jj < Matrix_B_p; jj += b) {
        for (register int kk = 0; kk < Matrix_A_n; kk += b) {
          min_ii_b_M = MIN(ii + b, Matrix_A_m);
          for (register int i = ii; i < min_ii_b_M; i++) {
            min_jj_b_P = MIN(jj + b, Matrix_B_p);
            for (register int j = jj; j < min_jj_b_P; j++) {
              min_kk_b_N = MIN(kk + b, Matrix_A_n);
              __m256 val_v = _mm256_setzero_ps();
              vm = _mm256_set1_epi32(0x80000000);

              // Iterate over the k-blocks
              for (register int k = kk; k < min_kk_b_N; k += max_vlen) {
                // Calculate number of valid elements in this block
                register int rem = min_kk_b_N - k;
                int hw_vlen = rem < max_vlen ? rem : max_vlen;

                // Create mask for loading the valid elements
                unsigned int m[max_vlen];
                for (size_t jj = 0; jj < max_vlen; jj++) {
                  m[jj] = (jj < hw_vlen) ? 0x80000000 : 0x00000000;
                }
                vm = _mm256_setr_epi32(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);

                // Load vectors from src0 and transposed src1
                __m256 vecA = _mm256_maskload_ps( & src0[i * Matrix_A_n + k], vm);
                __m256 vecB = _mm256_maskload_ps( & tranSrc1[j * Matrix_A_n + k], vm);

                // Perform the element-wise multiplication
                __m256 vecMult = _mm256_mul_ps(vecA, vecB);
                //val_v = _mm256_add_ps(val_v, vecMult);

                float temp[max_vlen];
                _mm256_storeu_ps(temp, vecMult);

                // Sum the values in temp and add to the destination matrix
                for (size_t l = 0; l < hw_vlen; l++) {
                  dest[i * Matrix_B_p + j] += temp[l]; // Accumulate the elements
                }
              }
            }
          }
        }
      }
    }
  }
  free(tranSrc1);
  return NULL;
  #elif defined(__aarch__) || defined(__aarch64__) || defined(__arm64__)
  #endif
}
