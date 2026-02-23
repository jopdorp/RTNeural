#ifndef WAVENET_LAYER_XSIMD_H_INCLUDED
#define WAVENET_LAYER_XSIMD_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include "../config.h"
#include <array>
#include <cmath>
#include <numeric>

namespace RTNEURAL_NAMESPACE
{

/**
 * Block size for block-based inference.
 */
static constexpr int WAVENET_BLOCK_SIZE = 32;

/**
 * Activation types supported by the WaveNet layer.
 */
enum class WaveNetActivation
{
    Tanh,
    Sigmoid,
    ReLU,
    GatedTanh
};

/**
 * Static implementation of a fused WaveNet layer for NAM models (XSIMD backend).
 *
 * Uses per-output-channel weight storage with inner_product + reduce_add
 * for correct SIMD matrix-vector multiplication (matching Conv1DT pattern).
 */
template <typename T,
    int in_sizet,
    int kernel_size,
    int dilation_rate,
    WaveNetActivation activation = WaveNetActivation::Tanh,
    bool has_residual = true,
    bool has_skip = false>
class WaveNetLayerT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_io_size = ceil_div(in_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = in_sizet;
    static constexpr auto block_size = WAVENET_BLOCK_SIZE;
    static constexpr auto state_size = (kernel_size - 1) * dilation_rate + 1;

    WaveNetLayerT()
    {
        for (int i = 0; i < out_size; ++i)
            for (int k = 0; k < kernel_size; ++k)
                for (int j = 0; j < v_io_size; ++j)
                    weights[i][k][j] = v_type((T)0);

        for (int i = 0; i < v_io_size; ++i)
            bias[i] = v_type((T)0);

        reset();
    }

    std::string getName() const noexcept { return "wavenet"; }
    constexpr bool isActivation() const noexcept { return false; }

    RTNEURAL_REALTIME void reset()
    {
        for (int col = 0; col < state_size; ++col)
            for (int i = 0; i < v_io_size; ++i)
                state[col][i] = v_type((T)0);

        state_ptr = 0;

        for (int i = 0; i < v_io_size; ++i)
        {
            outs[i] = v_type((T)0);
            skip_outs[i] = v_type((T)0);
        }
    }

    RTNEURAL_REALTIME inline void forward(const v_type (&ins)[v_io_size]) noexcept
    {
        // Store input in circular buffer
        for (int i = 0; i < v_io_size; ++i)
            state[state_ptr][i] = ins[i];

        // Fused convolution + bias + activation + residual
        // Per-output-channel matmul using inner_product + reduce_add
        for (int i = 0; i < v_io_size; ++i)
        {
            alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
            for (int m = 0; m < v_size && (i * v_size + m) < out_size; ++m)
            {
                const int out_ch = i * v_size + m;
                T ch_sum = (T)0;
                for (int k = 0; k < kernel_size; ++k)
                {
                    // Branchless state index computation (no modulo)
                    int idx = state_ptr - k * dilation_rate;
                    idx += (idx < 0) * state_size;

                    for (int j = 0; j < v_io_size; ++j)
                        ch_sum += xsimd::reduce_add(weights[out_ch][k][j] * state[idx][j]);
                }
                out_sum[m] = ch_sum;
            }

            v_type conv_out = xsimd::load_aligned(out_sum) + bias[i];
            v_type activated = applyActivation(conv_out);

            if (has_residual)
                outs[i] = ins[i] + activated;
            else
                outs[i] = activated;

            if (has_skip)
                skip_outs[i] += activated;
        }

        // Advance state pointer (branchless wrap)
        ++state_ptr;
        state_ptr -= (state_ptr >= state_size) * state_size;
    }

    /**
     * True block-based forward propagation.
     * Processes block_size samples in a fused kernel.
     */
    RTNEURAL_REALTIME inline void forwardBlock(const T* ins, T* output) noexcept
    {
        // Process all samples
        int cur_ptr = state_ptr;
        for (int sample = 0; sample < block_size; ++sample)
        {
            const T* in_ptr = ins + sample * in_size;
            T* out_ptr = output + sample * out_size;

            // Load input using unaligned load (input buffer not guaranteed aligned)
            v_type sample_in[v_io_size];
            for (int i = 0; i < v_io_size; ++i)
                sample_in[i] = xsimd::load_unaligned(in_ptr + i * v_size);

            // Store input in circular buffer
            for (int i = 0; i < v_io_size; ++i)
                state[cur_ptr][i] = sample_in[i];

            // Fused convolution + bias + activation + residual
            for (int i = 0; i < v_io_size; ++i)
            {
                alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
                for (int m = 0; m < v_size && (i * v_size + m) < out_size; ++m)
                {
                    const int out_ch = i * v_size + m;
                    T ch_sum = (T)0;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        // Branchless state index computation (no modulo)
                        int idx = cur_ptr - k * dilation_rate;
                        idx += (idx < 0) * state_size;

                        for (int j = 0; j < v_io_size; ++j)
                            ch_sum += xsimd::reduce_add(weights[out_ch][k][j] * state[idx][j]);
                    }
                    out_sum[m] = ch_sum;
                }

                v_type conv_out = xsimd::load_aligned(out_sum) + bias[i];
                v_type activated = applyActivation(conv_out);

                v_type out_val;
                if (has_residual)
                    out_val = sample_in[i] + activated;
                else
                    out_val = activated;

                // Store output using unaligned store
                xsimd::store_unaligned(out_ptr + i * v_size, out_val);
                outs[i] = out_val;

                if (has_skip)
                    skip_outs[i] += activated;
            }

            // Advance state pointer (branchless wrap)
            ++cur_ptr;
            cur_ptr -= (cur_ptr >= state_size) * state_size;
        }

        // Update state pointer
        state_ptr = cur_ptr;
    }

    RTNEURAL_REALTIME void resetSkip() noexcept
    {
        for (int i = 0; i < v_io_size; ++i)
            skip_outs[i] = v_type((T)0);
    }

    /**
     * Sets the layer weights.
     * w[out_ch][in_ch][kernel_pos] — per-output-channel storage for correct matmul.
     */
    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<std::vector<T>>>& w)
    {
        for (int i = 0; i < out_size; ++i)
        {
            for (int j = 0; j < in_size; ++j)
            {
                for (int k = 0; k < kernel_size; ++k)
                {
                    auto& wv = weights[i][k][j / v_size];
                    wv = set_value(wv, j % v_size, w[i][j][k]);
                }
            }
        }
    }

    RTNEURAL_REALTIME void setBias(const std::vector<T>& biasVals)
    {
        for (int i = 0; i < out_size; ++i)
        {
            const int vi = i / v_size;
            const int vi_off = i % v_size;
            bias[vi] = set_value(bias[vi], vi_off, biasVals[i]);
        }
    }

    RTNEURAL_REALTIME int getKernelSize() const noexcept { return kernel_size; }
    RTNEURAL_REALTIME int getDilationRate() const noexcept { return dilation_rate; }

    v_type outs[v_io_size];
    v_type skip_outs[v_io_size];

private:
    v_type state[state_size][v_io_size];
    int state_ptr = 0;

    // Per-output-channel weight storage: weights[out_ch][kernel_pos][v_io_size]
    // Each SIMD vector holds v_size input channel weights for one output channel.
    v_type weights[out_size][kernel_size][v_io_size];
    v_type bias[v_io_size];

    RTNEURAL_REALTIME inline v_type applyActivation(v_type x) const noexcept
    {
        switch (activation)
        {
        case WaveNetActivation::Tanh:
            return xsimd::tanh(x);
        case WaveNetActivation::Sigmoid:
            return v_type((T)1) / (v_type((T)1) + xsimd::exp(-x));
        case WaveNetActivation::ReLU:
            return xsimd::max(x, v_type((T)0));
        case WaveNetActivation::GatedTanh:
            return xsimd::tanh(x) * (v_type((T)1) / (v_type((T)1) + xsimd::exp(-x)));
        default:
            return xsimd::tanh(x);
        }
    }
};

/**
 * Fused WaveNet block for NAM models (XSIMD backend).
 *
 * Uses per-output-channel weight storage with inner_product + reduce_add
 * for correct SIMD matrix-vector multiplication (matching Conv1DT pattern).
 */
template <typename T,
    int channel_size,
    int kernel_size,
    int num_layers,
    WaveNetActivation activation = WaveNetActivation::Tanh>
class WaveNetBlockT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_channel_size = ceil_div(channel_size, v_size);

    static constexpr int getDilationRate(int layer)
    {
        return 1 << layer;
    }

    static constexpr int getStateSize(int dilation)
    {
        return (kernel_size - 1) * dilation + 1;
    }

    static constexpr int max_dilation = getDilationRate(num_layers - 1);
    static constexpr int max_state_size = getStateSize(max_dilation);

public:
    static constexpr auto in_size = channel_size;
    static constexpr auto out_size = channel_size;
    static constexpr auto block_size = WAVENET_BLOCK_SIZE;

    WaveNetBlockT()
    {
        for (int layer = 0; layer < num_layers; ++layer)
            for (int i = 0; i < channel_size; ++i)
                for (int k = 0; k < kernel_size; ++k)
                    for (int j = 0; j < v_channel_size; ++j)
                        weights[layer][i][k][j] = v_type((T)0);

        for (int layer = 0; layer < num_layers; ++layer)
            for (int i = 0; i < v_channel_size; ++i)
                biases[layer][i] = v_type((T)0);

        reset();
    }

    std::string getName() const noexcept { return "wavenet_block"; }
    constexpr bool isActivation() const noexcept { return false; }

    RTNEURAL_REALTIME void reset()
    {
        for (int layer = 0; layer < num_layers; ++layer)
        {
            for (int col = 0; col < max_state_size; ++col)
                for (int i = 0; i < v_channel_size; ++i)
                    states[layer][col][i] = v_type((T)0);
            state_ptrs[layer] = 0;
        }

        for (int i = 0; i < v_channel_size; ++i)
        {
            skip_sum[i] = v_type((T)0);
            outs[i] = v_type((T)0);
        }
    }

    RTNEURAL_REALTIME inline void forward(const v_type (&ins)[v_channel_size]) noexcept
    {
        for (int i = 0; i < v_channel_size; ++i)
            skip_sum[i] = v_type((T)0);

        v_type x[v_channel_size];
        for (int i = 0; i < v_channel_size; ++i)
            x[i] = ins[i];

        for (int layer = 0; layer < num_layers; ++layer)
        {
            const int dilation = getDilationRate(layer);
            const int layer_state_size = getStateSize(dilation);
            const int cur_ptr = state_ptrs[layer];

            // Store input in layer's circular buffer
            for (int i = 0; i < v_channel_size; ++i)
                states[layer][cur_ptr][i] = x[i];

            // Compute convolution with per-output-channel matmul
            for (int i = 0; i < v_channel_size; ++i)
            {
                alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
                for (int m = 0; m < v_size && (i * v_size + m) < channel_size; ++m)
                {
                    const int out_ch = i * v_size + m;
                    T ch_sum = (T)0;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        // Branchless state index (no modulo)
                        int idx = cur_ptr - k * dilation;
                        idx += (idx < 0) * layer_state_size;

                        for (int j = 0; j < v_channel_size; ++j)
                            ch_sum += xsimd::reduce_add(weights[layer][out_ch][k][j] * states[layer][idx][j]);
                    }
                    out_sum[m] = ch_sum;
                }

                v_type conv_out = xsimd::load_aligned(out_sum) + biases[layer][i];

                // Apply activation and residual
                v_type activated = applyActivation(conv_out);
                skip_sum[i] += activated;
                x[i] += activated;
            }

            // Advance state pointer (branchless wrap)
            int new_ptr = cur_ptr + 1;
            new_ptr -= (new_ptr >= layer_state_size) * layer_state_size;
            state_ptrs[layer] = new_ptr;
        }

        for (int i = 0; i < v_channel_size; ++i)
            outs[i] = skip_sum[i];
    }

    /**
     * True block-based forward propagation.
     * Processes block_size samples in a fused kernel without calling forward().
     */
    RTNEURAL_REALTIME inline void forwardBlock(const T* ins, T* output) noexcept
    {
        // Process each sample through all layers
        for (int sample = 0; sample < block_size; ++sample)
        {
            const T* in_ptr = ins + sample * in_size;
            T* out_ptr = output + sample * out_size;

            // Load input using unaligned load
            v_type sample_in[v_channel_size];
            for (int i = 0; i < v_channel_size; ++i)
                sample_in[i] = xsimd::load_unaligned(in_ptr + i * v_size);

            // Reset skip sum for this sample
            for (int i = 0; i < v_channel_size; ++i)
                skip_sum[i] = v_type((T)0);

            v_type x[v_channel_size];
            for (int i = 0; i < v_channel_size; ++i)
                x[i] = sample_in[i];

            // Process all layers
            for (int layer = 0; layer < num_layers; ++layer)
            {
                const int dilation = getDilationRate(layer);
                const int layer_state_size = getStateSize(dilation);
                const int cur_ptr = state_ptrs[layer];

                // Store x in layer's circular buffer
                for (int i = 0; i < v_channel_size; ++i)
                    states[layer][cur_ptr][i] = x[i];

                // Compute convolution with per-output-channel matmul
                for (int i = 0; i < v_channel_size; ++i)
                {
                    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T out_sum[v_size] {};
                    for (int m = 0; m < v_size && (i * v_size + m) < channel_size; ++m)
                    {
                        const int out_ch = i * v_size + m;
                        T ch_sum = (T)0;
                        for (int k = 0; k < kernel_size; ++k)
                        {
                            int idx = cur_ptr - k * dilation;
                            idx += (idx < 0) * layer_state_size;

                            for (int j = 0; j < v_channel_size; ++j)
                                ch_sum += xsimd::reduce_add(weights[layer][out_ch][k][j] * states[layer][idx][j]);
                        }
                        out_sum[m] = ch_sum;
                    }

                    v_type conv_out = xsimd::load_aligned(out_sum) + biases[layer][i];

                    // Fused activation + residual + skip
                    v_type activated = applyActivation(conv_out);
                    skip_sum[i] += activated;
                    x[i] += activated;
                }

                // Advance state pointer (branchless wrap)
                int new_ptr = cur_ptr + 1;
                new_ptr -= (new_ptr >= layer_state_size) * layer_state_size;
                state_ptrs[layer] = new_ptr;
            }

            // Write output (skip sum) using unaligned store
            for (int i = 0; i < v_channel_size; ++i)
            {
                xsimd::store_unaligned(out_ptr + i * v_size, skip_sum[i]);
                outs[i] = skip_sum[i];
            }
        }
    }

    RTNEURAL_REALTIME void setLayerWeights(int layer, int kernel_pos, const std::vector<std::vector<T>>& w)
    {
        if (layer >= num_layers || kernel_pos >= kernel_size)
            return;

        for (int i = 0; i < channel_size; ++i)
        {
            for (int j = 0; j < channel_size; ++j)
            {
                auto& wv = weights[layer][i][kernel_pos][j / v_size];
                wv = set_value(wv, j % v_size, w[i][j]);
            }
        }
    }

    RTNEURAL_REALTIME void setLayerBias(int layer, const std::vector<T>& biasVals)
    {
        if (layer >= num_layers)
            return;

        for (int i = 0; i < channel_size; ++i)
        {
            const int vi = i / v_size;
            const int vi_off = i % v_size;
            biases[layer][vi] = set_value(biases[layer][vi], vi_off, biasVals[i]);
        }
    }

    v_type outs[v_channel_size];

private:
    v_type states[num_layers][max_state_size][v_channel_size];
    int state_ptrs[num_layers];

    // Per-output-channel weight storage: weights[layer][out_ch][kernel_pos][v_channel_size]
    v_type weights[num_layers][channel_size][kernel_size][v_channel_size];
    v_type biases[num_layers][v_channel_size];

    v_type skip_sum[v_channel_size];

    RTNEURAL_REALTIME inline v_type applyActivation(v_type x) const noexcept
    {
        switch (activation)
        {
        case WaveNetActivation::Tanh:
            return xsimd::tanh(x);
        case WaveNetActivation::Sigmoid:
            return v_type((T)1) / (v_type((T)1) + xsimd::exp(-x));
        case WaveNetActivation::ReLU:
            return xsimd::max(x, v_type((T)0));
        case WaveNetActivation::GatedTanh:
            return xsimd::tanh(x) * (v_type((T)1) / (v_type((T)1) + xsimd::exp(-x)));
        default:
            return xsimd::tanh(x);
        }
    }
};

} // namespace RTNEURAL_NAMESPACE

#endif // WAVENET_LAYER_XSIMD_H_INCLUDED
