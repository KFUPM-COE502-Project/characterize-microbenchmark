/* para.c
 *
 * Author: Khaleel Alhaboub
 * Date  : 21/12/2024
 *
 *  Implmentation of MIMD blackScholes
 */

/* Standard C includes */
#include <stdlib.h>
#include <pthread.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"
#include "vec.h"
#include "scalar.h"

#define VEC_SIZE 8
/* Gaurd the argument with start and end IDs */
typedef struct {
    args_t* shared_args;
    int start_idx;
    int end_idx;
} thread_arg_t;

static void* thread_func(void* arg)
{
    thread_arg_t* targ = (thread_arg_t*)arg;
    args_t* myargs = targ->shared_args;

    float* sptPrice  = myargs->sptPrice;
    float* strike    = myargs->strike;
    float* rate      = myargs->rate;
    float* volatility= myargs->volatility;
    float* otime     = myargs->otime;
    char*  otype     = myargs->otype;
    float* output    = myargs->output;

    for (int i = targ->start_idx; i < targ->end_idx; i += VEC_SIZE) {
        if (i + VEC_SIZE <= targ->end_idx) {
            // Full SIMD chunk
            blackScholesAVX(&sptPrice[i], &strike[i], &rate[i], &volatility[i], &otime[i], &otype[i], &output[i]);
        } else {
            // Scalar fallback for remaining elements
            for (int j = i; j < targ->end_idx; j++) {
                blackScholes(sptPrice[j], strike[j], rate[j], volatility[j], otime[j], otype[j], 0, &output[j]);
            }
        }
    }
    return NULL;
}


/* Alternative Implementation */
void* impl_parallel(void* args)
{
    args_t* a = (args_t*)args;

    int n         = a->num_stocks;
    int nthreads  = a->nthreads;
    int chunkSize = (n + nthreads - 1) / nthreads; 

    pthread_t*    threads = (pthread_t*)malloc(sizeof(pthread_t)    * nthreads);
    thread_arg_t* targs   = (thread_arg_t*)malloc(sizeof(thread_arg_t)* nthreads);

    /* Launch each thread */
    for (int t = 0; t < nthreads; t++) {
        targs[t].shared_args = a;
        targs[t].start_idx   = t * chunkSize;
        targs[t].end_idx     = (t + 1) * chunkSize;

        if (targs[t].end_idx > n) {
            targs[t].end_idx = n; // to ensure we don't go out of bound
        }
        pthread_create(&threads[t], NULL, thread_func, &targs[t]);
    }

    /* Wait for all threads */
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(targs);

  return NULL;
}
