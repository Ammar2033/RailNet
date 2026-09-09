#define RAILNET_EXPORTS
#include "rail_kernel.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
  #include <cpuid.h>
  #include <immintrin.h>
#endif

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(v128);
    __m128 sums = _mm_add_ps(v128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline double hsum256_pd(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(vlow, vhigh);
    __m128d shuf = _mm_unpackhi_pd(sum128, sum128);
    return _mm_cvtsd_f64(_mm_add_sd(sum128, shuf));
}
#endif

RAILNET_API int railnet_has_avx2(void) {
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5: AVX2
    }
    return 0;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 5)) != 0;
    }
    return 0;
#else
    return 0;
#endif
}

RAILNET_API int railnet_has_avx512(void) {
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 16)) != 0; // EBX bit 16: AVX512F
    }
    return 0;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 16)) != 0;
    }
    return 0;
#else
    return 0;
#endif
}

RAILNET_API int railnet_get_num_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

RAILNET_API void railnet_set_num_threads(int n) {
#if defined(_OPENMP)
    if (n > 0) {
        omp_set_num_threads(n);
    }
#else
    (void)n;
#endif
}

RAILNET_API int railnet_linear_fp32(
    const float* x,
    const int32_t* route_ids,
    const int32_t* term_rail,
    const int8_t* term_sign,
    const float* rails,
    float* y,
    int64_t out_features,
    int64_t in_features,
    int32_t rail_count,
    int32_t max_terms,
    int32_t num_threads
) {
    if (!x || !route_ids || !term_rail || !term_sign || !rails || !y) {
        return -1;
    }
    if (out_features <= 0 || in_features <= 0 || rail_count <= 0 || max_terms <= 0) {
        return -2;
    }

#if defined(_OPENMP)
    int orig_threads = omp_get_max_threads();
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    const int STACK_LIMIT = 512;

    #pragma omp parallel
    {
        float stack_acc[STACK_LIMIT];
        float* acc = (rail_count <= STACK_LIMIT) ? stack_acc : (float*)malloc(rail_count * sizeof(float));

        #pragma omp for schedule(static)
        for (int64_t j = 0; j < out_features; ++j) {
            // Zero-out local rail accumulator
            memset(acc, 0, rail_count * sizeof(float));

            const int32_t* row_routes = route_ids + j * in_features;

            // Stage-A: Gather input activations into shared rails (ZERO Multiplications!)
            for (int64_t i = 0; i < in_features; ++i) {
                int32_t g = row_routes[i];
                const int32_t* r_ptr = term_rail + g * max_terms;
                const int8_t* s_ptr = term_sign + g * max_terms;
                float xi = x[i];

                for (int32_t t = 0; t < max_terms; ++t) {
                    int8_t s = s_ptr[t];
                    if (s != 0) {
                        int32_t r = r_ptr[t];
                        if (r >= 0 && r < rail_count) {
                            acc[r] += (s > 0) ? xi : -xi;
                        }
                    }
                }
            }

            // Stage-B: Rail accumulation (SIMD dot-product)
            float sum = 0.0f;
#if defined(__AVX2__)
            __m256 y_vec = _mm256_setzero_ps();
            int32_t r = 0;
            for (; r + 8 <= rail_count; r += 8) {
                __m256 a = _mm256_loadu_ps(acc + r);
                __m256 b = _mm256_loadu_ps(rails + r);
#if defined(__FMA__)
                y_vec = _mm256_fmadd_ps(a, b, y_vec);
#else
                y_vec = _mm256_add_ps(y_vec, _mm256_mul_ps(a, b));
#endif
            }
            sum = hsum256_ps(y_vec);
            for (; r < rail_count; ++r) {
                sum += acc[r] * rails[r];
            }
#else
            for (int32_t r = 0; r < rail_count; ++r) {
                sum += acc[r] * rails[r];
            }
#endif
            y[j] = sum;
        }

        if (rail_count > STACK_LIMIT && acc) {
            free(acc);
        }
    }

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(orig_threads);
    }
#endif

    return 0;
}

RAILNET_API int railnet_linear_fp64(
    const double* x,
    const int32_t* route_ids,
    const int32_t* term_rail,
    const int8_t* term_sign,
    const double* rails,
    double* y,
    int64_t out_features,
    int64_t in_features,
    int32_t rail_count,
    int32_t max_terms,
    int32_t num_threads
) {
    if (!x || !route_ids || !term_rail || !term_sign || !rails || !y) {
        return -1;
    }
    if (out_features <= 0 || in_features <= 0 || rail_count <= 0 || max_terms <= 0) {
        return -2;
    }

#if defined(_OPENMP)
    int orig_threads = omp_get_max_threads();
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    const int STACK_LIMIT = 512;

    #pragma omp parallel
    {
        double stack_acc[STACK_LIMIT];
        double* acc = (rail_count <= STACK_LIMIT) ? stack_acc : (double*)malloc(rail_count * sizeof(double));

        #pragma omp for schedule(static)
        for (int64_t j = 0; j < out_features; ++j) {
            memset(acc, 0, rail_count * sizeof(double));

            const int32_t* row_routes = route_ids + j * in_features;

            // Stage-A: Gather input activations into shared rails
            for (int64_t i = 0; i < in_features; ++i) {
                int32_t g = row_routes[i];
                const int32_t* r_ptr = term_rail + g * max_terms;
                const int8_t* s_ptr = term_sign + g * max_terms;
                double xi = x[i];

                for (int32_t t = 0; t < max_terms; ++t) {
                    int8_t s = s_ptr[t];
                    if (s != 0) {
                        int32_t r = r_ptr[t];
                        if (r >= 0 && r < rail_count) {
                            acc[r] += (s > 0) ? xi : -xi;
                        }
                    }
                }
            }

            // Stage-B: Rail accumulation
            double sum = 0.0;
#if defined(__AVX2__)
            __m256d y_vec = _mm256_setzero_pd();
            int32_t r = 0;
            for (; r + 4 <= rail_count; r += 4) {
                __m256d a = _mm256_loadu_pd(acc + r);
                __m256d b = _mm256_loadu_pd(rails + r);
#if defined(__FMA__)
                y_vec = _mm256_fmadd_pd(a, b, y_vec);
#else
                y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(a, b));
#endif
            }
            sum = hsum256_pd(y_vec);
            for (; r < rail_count; ++r) {
                sum += acc[r] * rails[r];
            }
#else
            for (int32_t r = 0; r < rail_count; ++r) {
                sum += acc[r] * rails[r];
            }
#endif
            y[j] = sum;
        }

        if (rail_count > STACK_LIMIT && acc) {
            free(acc);
        }
    }

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(orig_threads);
    }
#endif

    return 0;
}

RAILNET_API int railnet_linear_int8_w8a_float(
    const float* x,
    const int32_t* route_ids,
    const int32_t* term_rail,
    const int8_t* term_sign,
    const int32_t* rails,
    float* y,
    float scale,
    int64_t out_features,
    int64_t in_features,
    int32_t rail_count,
    int32_t max_terms,
    int32_t num_threads
) {
    if (!x || !route_ids || !term_rail || !term_sign || !rails || !y) {
        return -1;
    }
    if (out_features <= 0 || in_features <= 0 || rail_count <= 0 || max_terms <= 0) {
        return -2;
    }

#if defined(_OPENMP)
    int orig_threads = omp_get_max_threads();
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    const int STACK_LIMIT = 512;

    #pragma omp parallel
    {
        float stack_acc[STACK_LIMIT];
        float* acc = (rail_count <= STACK_LIMIT) ? stack_acc : (float*)malloc(rail_count * sizeof(float));

        #pragma omp for schedule(static)
        for (int64_t j = 0; j < out_features; ++j) {
            memset(acc, 0, rail_count * sizeof(float));
            const int32_t* row_routes = route_ids + j * in_features;

            for (int64_t i = 0; i < in_features; ++i) {
                int32_t g = row_routes[i];
                const int32_t* r_ptr = term_rail + g * max_terms;
                const int8_t* s_ptr = term_sign + g * max_terms;
                float xi = x[i];

                for (int32_t t = 0; t < max_terms; ++t) {
                    int8_t s = s_ptr[t];
                    if (s != 0) {
                        int32_t r = r_ptr[t];
                        if (r >= 0 && r < rail_count) {
                            acc[r] += (s > 0) ? xi : -xi;
                        }
                    }
                }
            }

            float sum = 0.0f;
#if defined(__AVX2__)
            __m256 y_vec = _mm256_setzero_ps();
            int32_t r = 0;
            for (; r + 8 <= rail_count; r += 8) {
                __m256 a = _mm256_loadu_ps(acc + r);
                __m256i r_i = _mm256_loadu_si256((const __m256i*)(rails + r));
                __m256 b = _mm256_cvtepi32_ps(r_i);
#if defined(__FMA__)
                y_vec = _mm256_fmadd_ps(a, b, y_vec);
#else
                y_vec = _mm256_add_ps(y_vec, _mm256_mul_ps(a, b));
#endif
            }
            sum = hsum256_ps(y_vec);
            for (; r < rail_count; ++r) {
                sum += acc[r] * (float)rails[r];
            }
#else
            for (int32_t r = 0; r < rail_count; ++r) {
                sum += acc[r] * (float)rails[r];
            }
#endif
            y[j] = sum * scale;
        }

        if (rail_count > STACK_LIMIT && acc) {
            free(acc);
        }
    }

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(orig_threads);
    }
#endif

    return 0;
}

RAILNET_API int railnet_linear_int8_w8a16(
    const int16_t* x,
    const int32_t* route_ids,
    const int32_t* term_rail,
    const int8_t* term_sign,
    const int32_t* rails,
    int32_t* y,
    int64_t out_features,
    int64_t in_features,
    int32_t rail_count,
    int32_t max_terms,
    int32_t num_threads
) {
    if (!x || !route_ids || !term_rail || !term_sign || !rails || !y) {
        return -1;
    }
    if (out_features <= 0 || in_features <= 0 || rail_count <= 0 || max_terms <= 0) {
        return -2;
    }

#if defined(_OPENMP)
    int orig_threads = omp_get_max_threads();
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    const int STACK_LIMIT = 512;

    #pragma omp parallel
    {
        int32_t stack_acc[STACK_LIMIT];
        int32_t* acc = (rail_count <= STACK_LIMIT) ? stack_acc : (int32_t*)malloc(rail_count * sizeof(int32_t));

        #pragma omp for schedule(static)
        for (int64_t j = 0; j < out_features; ++j) {
            memset(acc, 0, rail_count * sizeof(int32_t));
            const int32_t* row_routes = route_ids + j * in_features;

            for (int64_t i = 0; i < in_features; ++i) {
                int32_t g = row_routes[i];
                const int32_t* r_ptr = term_rail + g * max_terms;
                const int8_t* s_ptr = term_sign + g * max_terms;
                int32_t xi = (int32_t)x[i];

                for (int32_t t = 0; t < max_terms; ++t) {
                    int8_t s = s_ptr[t];
                    if (s != 0) {
                        int32_t r = r_ptr[t];
                        if (r >= 0 && r < rail_count) {
                            acc[r] += (s > 0) ? xi : -xi;
                        }
                    }
                }
            }

            int32_t sum = 0;
            for (int32_t r = 0; r < rail_count; ++r) {
                sum += acc[r] * rails[r];
            }
            y[j] = sum;
        }

        if (rail_count > STACK_LIMIT && acc) {
            free(acc);
        }
    }

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(orig_threads);
    }
#endif

    return 0;
}

