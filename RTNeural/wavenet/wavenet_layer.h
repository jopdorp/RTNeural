#ifndef WAVENET_LAYER_H_INCLUDED
#define WAVENET_LAYER_H_INCLUDED

/**
 * WaveNet optimized layer for NAM models.
 *
 * This layer fuses the following operations into a single optimized loop:
 * - Dilated causal convolution
 * - Bias addition
 * - Activation (tanh/sigmoid/ReLU)
 * - Residual addition
 * - Skip accumulation (optional)
 *
 * Supports block-based inference for improved performance.
 */

#include "../config.h"

#if RTNEURAL_USE_EIGEN
#include "wavenet_layer_eigen.h"
#elif RTNEURAL_USE_XSIMD
#include "wavenet_layer_xsimd.h"
#else
#include "wavenet_layer_stl.h"
#endif

#endif // WAVENET_LAYER_H_INCLUDED
