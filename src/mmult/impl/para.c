/* para.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

#define _GNU_SOURCE
/* Standard C includes */
#include <stdlib.h>

#include <pthread.h>
#include <sched.h>
#include <assert.h>
/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* If we are on Darwin, include the compatibility header */
#if defined(__APPLE__)
#include "common/mach_pthread_compatibility.h"
#endif

/* Include application-specific headers */
#include "include/types.h"

#define MIN(a, b)(((a) < (b)) ? (a) : (b))

void * worker(void * args) {
  args_t * p_args = (args_t * ) args;

  register float * C = (float * ) p_args -> output;
  register
  const float * A = (const float * ) p_args -> input0;
  register
  const float * B = (const float * ) p_args -> input1;

  register size_t i_start = p_args -> i_start;
  register size_t i_end = p_args -> i_end;
  register size_t j_start = p_args -> j_start;
  register size_t j_end = p_args -> j_end;

  register size_t size_A_n = p_args -> input_A_n;
  register size_t size_B_p = p_args -> input_B_p;
  register size_t b = p_args -> block_size; /* Tiling block size */

  for (register size_t i = i_start; i < i_end; i++) {
    for (register size_t j = j_start; j < j_end; j++) {
      C[i * size_B_p + j] = 0.0;
    }
  }

  for (register size_t ii = i_start; ii < i_end; ii += b) {
    register size_t ii_end = MIN(ii + b, i_end);

    for (register size_t jj = j_start; jj < j_end; jj += b) {
      register size_t jj_end = MIN(jj + b, j_end);

      for (register size_t kk = 0; kk < size_A_n; kk += b) {
        register size_t kk_end = MIN(kk + b, size_A_n);

        /* Now multiply the small tile [ii..ii_end) x [jj..jj_end) */
        for (register size_t i = ii; i < ii_end; i++) {
          for (register size_t j = jj; j < jj_end; j++) {
            register float sum = C[i * size_B_p + j];
            for (register size_t k = kk; k < kk_end; k++) {
              sum += A[i * size_A_n + k] * B[k * size_B_p + j];
            }
            C[i * size_B_p + j] = sum;
          }
        }
      }
    }
  }

  return NULL;
}
/* Alternative Implementation */
void* impl_parallel(void* args)
{
  args_t * p_args = (args_t * ) args;

  register float * dest = (float * ) p_args -> output;
  register
  const float * A = (const float * ) p_args -> input0;
  register
  const float * B = (const float * ) p_args -> input1;

  register size_t size_A_m = p_args -> input_A_m;
  //register size_t size_A_n          = p_args->input_A_n;
  register size_t size_B_p = p_args -> input_B_p;

  register size_t nthreads = p_args -> nthreads;
  register size_t cpu = p_args -> cpu;

  /* Decide how to split M,P across threads. For instance: */
  register size_t rowBlocks = 1;
  register size_t colBlocks = nthreads;

  if (nthreads == 4) {
    rowBlocks = 2;
    colBlocks = 2;
  }
  if (rowBlocks * colBlocks != nthreads) {
    rowBlocks = 1;
    colBlocks = nthreads;
  }

  pthread_t tid[nthreads];
  args_t targs[nthreads];
  cpu_set_t cpuset[nthreads];

  register size_t rowBlockSize = size_A_m / rowBlocks;
  register size_t rowRemainder = size_A_m % rowBlocks;
  register size_t colBlockSize = size_B_p / colBlocks;
  register size_t colRemainder = size_B_p % colBlocks;

  size_t threadIndex = 0;
  size_t rowStart = 0;
  for (size_t rb = 0; rb < rowBlocks; rb++) {
    size_t thisBlockHeight = rowBlockSize + (rb == rowBlocks - 1 ? rowRemainder : 0);
    size_t rowEnd = rowStart + thisBlockHeight;

    size_t colStart = 0;
    for (size_t cb = 0; cb < colBlocks; cb++) {
      size_t thisBlockWidth = colBlockSize + (cb == colBlocks - 1 ? colRemainder : 0);
      size_t colEnd = colStart + thisBlockWidth;

      /* Copy the global args into each thread's args, then overwrite subrange fields */
      targs[threadIndex] = * p_args;
      targs[threadIndex].i_start = rowStart;
      targs[threadIndex].i_end = rowEnd;
      targs[threadIndex].j_start = colStart;
      targs[threadIndex].j_end = colEnd;

      /* Affinity */
      CPU_ZERO( & cpuset[threadIndex]);
      CPU_SET((cpu + threadIndex) % nthreads, & cpuset[threadIndex]);

      /* Create the thread using the unified worker */
      pthread_create( & tid[threadIndex], NULL, worker, & targs[threadIndex]);
      pthread_setaffinity_np(tid[threadIndex], sizeof(cpuset[threadIndex]), & cpuset[threadIndex]);

      colStart = colEnd;
      threadIndex++;
    }
    rowStart = rowEnd;
  }

  /* Join all threads */
  for (register size_t i = 0; i < threadIndex; i++) {
    pthread_join(tid[i], NULL);
  }
  return NULL;
}
