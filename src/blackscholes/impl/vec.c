/* vec.c
 *
 * Author: Khaleel Alhaboub
 * Date  : 21/12/2024
 *
 *  Implmentation of vectorized blackScholes
 */

/* Standard C includes */
#include <stdlib.h>
#include <immintrin.h> // AVX/AVX2 intrinsics (on x86)
#include <math.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"
#include "include/avx_mathfun.h" // Alternative implmentation of log256_ps and exp256_ps
#include "scalar.h" // To switch to scaller execution for the remaining elements

#define VEC_SIZE 8
#define inv_sqrt_2xPI 0.39894228040143270286

__m256 CNDF_AVX(__m256 input)
{
    // Check for negative value of InputX
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 signMask = _mm256_cmp_ps(input, zero, _CMP_LT_OS);
    __m256 absX = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), input);

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    // expValues = exp(-0.5 * x^2)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 sq = _mm256_mul_ps(absX, absX);
    __m256 halfSq = _mm256_mul_ps(half, sq);
    __m256 negHalf = _mm256_sub_ps(zero, halfSq);
    __m256 expValues = exp256_ps(negHalf);

    // xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;
    __m256 xNPrimeofX = _mm256_mul_ps(expValues, _mm256_set1_ps(inv_sqrt_2xPI));

    // xK2 = 0.2316419 * xInput;
    // xK2 = 1.0 + xK2;
	// xK2 = 1.0 / xK2;
    __m256 c2316419 = _mm256_set1_ps(0.2316419f);
    __m256 t1 = _mm256_add_ps(_mm256_mul_ps(c2316419, absX), _mm256_set1_ps(1.0f));
    __m256 xK2 = _mm256_div_ps(_mm256_set1_ps(1.0f), t1);

    __m256 xK2_2 = _mm256_mul_ps(xK2, xK2);
    __m256 xK2_3 = _mm256_mul_ps(xK2_2, xK2);
    __m256 xK2_4 = _mm256_mul_ps(xK2_3, xK2);
    __m256 xK2_5 = _mm256_mul_ps(xK2_4, xK2);

    __m256 fl0 = _mm256_set1_ps(0.319381530f);
    __m256 fl1 = _mm256_set1_ps(-0.356563782f);
    __m256 fl2 = _mm256_set1_ps(1.781477937f);
    __m256 fl3 = _mm256_set1_ps(-1.821255978f);
    __m256 fl4 = _mm256_set1_ps(1.330274429f);

    __m256 xLocal_1 = _mm256_mul_ps(xK2, fl0);
    __m256 xLocal_2 = _mm256_mul_ps(xK2_2, fl1);
    __m256 xLocal_3 = _mm256_mul_ps(xK2_3, fl2);
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);
    xLocal_3 = _mm256_mul_ps(xK2_4, fl3);
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);
    xLocal_3 = _mm256_mul_ps(xK2_5, fl4);
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);
    xLocal_1 = _mm256_add_ps(xLocal_2, xLocal_1);

    // xLocal = xLocal_1 * xNPrimeofX;
	// xLocal = 1.0 - xLocal;
    __m256 xLocal = _mm256_mul_ps(xLocal_1, xNPrimeofX);
    xLocal = _mm256_sub_ps(_mm256_set1_ps(1.0f), xLocal);
    
    // check the sign
    __m256 OutputX = _mm256_sub_ps(_mm256_set1_ps(1.0f), xLocal);
    __m256 result = _mm256_blendv_ps(xLocal, OutputX, signMask);
    return result;
}

void blackScholesAVX(const float* sptprice, const float* strike, const float* rate, const float* volatility,
                     const float* otime, const char* otype, float* result) {
    // Constants
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);

    // Load data into AVX registers
    __m256 xStockPrice = _mm256_loadu_ps(sptprice);
    __m256 xStrikePrice = _mm256_loadu_ps(strike);
    __m256 xRiskFreeRate = _mm256_loadu_ps(rate);
    __m256 xVolatility = _mm256_loadu_ps(volatility);
    __m256 xTime = _mm256_loadu_ps(otime);
    __m256 xSqrtTime = _mm256_sqrt_ps(xTime);
    __m256 logValues = log256_ps(_mm256_div_ps(xStockPrice, xStrikePrice));

    // xPowerTerm = xPowerTerm * 0.5;
    __m256 xPowerTerm = _mm256_mul_ps(xVolatility, xVolatility);
    xPowerTerm = _mm256_mul_ps(xPowerTerm, half);

    // xD1 = xRiskFreeRate + xPowerTerm;
    __m256 xD1 = _mm256_add_ps(xRiskFreeRate, xPowerTerm);
    xD1 = _mm256_mul_ps(xD1, xTime);
    xD1 = _mm256_add_ps(xD1, logValues);
    __m256 xDen = _mm256_mul_ps(xVolatility, xSqrtTime);
    xD1 = _mm256_div_ps(xD1, xDen);
    __m256 xD2 = _mm256_sub_ps(xD1, xDen);

    __m256 NofXd1 = CNDF_AVX(xD1);
    __m256 NofXd2 = CNDF_AVX(xD2);

    // FutureValueX = strike * (exp(-(rate) * (otime)));
    __m256 negRiskFreeRate = _mm256_sub_ps(_mm256_setzero_ps(), xRiskFreeRate);
    __m256 FutureValueX = _mm256_mul_ps(xStrikePrice, exp256_ps(_mm256_mul_ps(negRiskFreeRate, xTime)));

    // otype == 'c' :: (sptprice * NofXd1) - (FutureValueX * NofXd2);
    __m256 callPrice = _mm256_sub_ps(_mm256_mul_ps(xStockPrice, NofXd1), _mm256_mul_ps(FutureValueX, NofXd2));

    // otype != 'c' :: 
    // NegNofXd1 = (1.0 - NofXd1);
	// NegNofXd2 = (1.0 - NofXd2);
	// OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    __m256 putPrice = _mm256_sub_ps(_mm256_mul_ps(FutureValueX, _mm256_sub_ps(one, NofXd2)),
                                    _mm256_mul_ps(xStockPrice, _mm256_sub_ps(one, NofXd1)));

    for (int j = 0; j < 8; ++j) {
        if (otype[j] == 'c' || otype[j] == 'C') {
            result[j] = ((float*)&callPrice)[j];
        } else {
            result[j] = ((float*)&putPrice)[j];
        }
    }
}

/* Alternative Implementation */
void* impl_vector(void* args)
{
    args_t* myargs = (args_t*)args;

    int    n         = myargs->num_stocks;
    float* sptPrice  = myargs->sptPrice;
    float* strike    = myargs->strike;
    float* rate      = myargs->rate;
    float* volatility= myargs->volatility;
    float* otime     = myargs->otime;
    char*  otype     = myargs->otype;
    float* output    = myargs->output;

    /* Process in chunks of 8 */
    int i = 0;
    for (; i + VEC_SIZE - 1 < n; i += VEC_SIZE) {
        blackScholesAVX(&sptPrice[i], &strike[i], &rate[i], &volatility[i], &otime[i], &otype[i], &output[i]);
    }

    /* For remaining elements not multiple of 8, perform regualr scaler */
    for (; i < n; i++) {
		blackScholes(sptPrice[i], strike[i], rate[i], volatility[i], otime[i], otype[i], 0, &output[i]);
    }

  return NULL;
}
