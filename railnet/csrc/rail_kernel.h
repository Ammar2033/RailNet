#ifndef RAILNET_KERNEL_H
#define RAILNET_KERNEL_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef RAILNET_EXPORTS
    #define RAILNET_API __declspec(dllexport)
  #else
    #define RAILNET_API __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define RAILNET_API __attribute__((visibility("default")))
  #else
    #define RAILNET_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check CPU instruction set capabilities at runtime.
 */
RAILNET_API int railnet_has_avx2(void);
RAILNET_API int railnet_has_avx512(void);

/**
 * Thread management.
 */
RAILNET_API int railnet_get_num_threads(void);
RAILNET_API void railnet_set_num_threads(int n);

/**
 * RailNet high-speed linear layer inference (FP32).
 *
 * Computes:
 *   For each neuron j in [0, out_features):
 *     acc[r] = sum_{i=0}^{in_features-1} sum_{t=0}^{max_terms-1} sign[g, t] * x[i]  where g = route_ids[j * in_features + i], r = term_rail[g, t]
 *     y[j] = sum_{r=0}^{rail_count-1} acc[r] * rails[r]
 *
 * @param x Input activation vector [in_features]
 * @param route_ids Flattened route IDs matrix [out_features * in_features]
 * @param term_rail Rail lookup table [65536 * max_terms]
 * @param term_sign Sign lookup table [65536 * max_terms] (+1, -1, or 0)
 * @param rails Shared rail values [rail_count]
 * @param y Output vector [out_features]
 * @param out_features Number of output features (neurons)
 * @param in_features Number of input features
 * @param rail_count Number of shared rails (e.g. 96, 128, 192)
 * @param max_terms Max active terms per route (e.g. 2, 3, 4)
 * @param num_threads Number of OpenMP threads (<=0 for default/all cores)
 */
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
);

/**
 * RailNet high-speed linear layer inference (FP64 bit-exact mode).
 */
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
);

/**
 * RailNet INT8 inference with float activations (W8A_Float).
 * Dequantizes on-the-fly via scale factor: y = scale * sum(G[r] * rails[r]).
 */
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
);

/**
 * RailNet INT8 exact integer inference (W8A16 mode matching hardware RTL).
 * Input: signed 16-bit integer activations. Output: signed 32-bit integer.
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* RAILNET_KERNEL_H */
